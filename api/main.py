"""
Nairobi Flood Guard API - model serving decoupled from the Streamlit UI.

Endpoints:
    GET  /health              liveness + model version
    GET  /registry            model registry (features, threshold, metrics)
    GET  /wards/risk          scored wards (optionally filtered by county)
    GET  /wards/{ward}/risk   one ward's risk + features
    GET  /reroutes            rerouting options (precomputed cache if fresh,
                              otherwise computed on demand - slow first call)
    GET  /reroutes/gtfs-rt    the same options as a GTFS-Realtime protobuf feed
    POST /reports             flood report intake (JSON)
    POST /reports/sms         Africa's Talking inbound-SMS webhook (form)
    GET  /reports             list stored flood reports
    POST /subscribers         opt a phone number into SMS alerts for a ward/county
    GET  /alerts              alert audit log (phones masked)

Run locally:   uvicorn api.main:app --reload
Production:    uvicorn api.main:app --host 0.0.0.0 --port $PORT

Flood reports are stored in SQLite at ``REPORTS_DB_PATH`` (default
``cache/flood_reports.db``). On platforms with ephemeral filesystems (Render,
Streamlit Cloud) point this at a mounted persistent disk, or replace the
storage layer with a managed database - the schema is intentionally a single
flat table to make that swap trivial.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import joblib
import pandas as pd
from fastapi import FastAPI, Form, HTTPException, Query, Response
from pydantic import BaseModel, Field

from Utils import alert_store
from Utils.feature_engineering import engineer_features

BASE = Path(__file__).resolve().parent.parent
FLOODS_GPKG = BASE / "Data" / "floods.gpkg"
REGISTRY_PATH = BASE / "Models" / "model_registry.json"
ROAD_GRAPH = BASE / "Data" / "nairobi_road_network.graphml"
GTFS_DIR = BASE / "Data" / "GTFS_FEED_2019"
PRECOMPUTED_REROUTES = BASE / "cache" / "precomputed_reroutes.json"
PRECOMPUTED_MAX_AGE_HOURS = 6.0
REPORTS_DB_PATH = Path(os.environ.get("REPORTS_DB_PATH", BASE / "cache" / "flood_reports.db"))

app = FastAPI(
    title="Nairobi Flood Guard API",
    description="Calibrated ward-level flood risk and matatu rerouting.",
    version="3.0",
)

_lock = threading.Lock()
_cache: dict[str, Any] = {}


def _registry() -> dict:
    if "registry" not in _cache:
        with open(REGISTRY_PATH, encoding="utf-8") as f:
            _cache["registry"] = json.load(f)
    return _cache["registry"]


def _model():
    if "model" not in _cache:
        _cache["model"] = joblib.load(BASE / _registry()["model_path"])
    return _cache["model"]


def _scored_wards() -> gpd.GeoDataFrame:
    """Wards scored with historical (April 2024) rainfall features."""
    if "scored" not in _cache:
        wards = gpd.read_file(FLOODS_GPKG)
        wards = engineer_features(wards)
        feature_cols = _registry()["feature_cols"]
        X = wards[feature_cols].fillna(wards[feature_cols].median())
        wards["flood_prob"] = _model().predict_proba(X)[:, 1]
        _cache["scored"] = wards
    return _cache["scored"]


def _reports_db() -> sqlite3.Connection:
    # Schema (flood_reports + subscribers + alerts_sent) lives in
    # Utils.alert_store so the API, the app and the alerting script share
    # exactly one definition.
    return alert_store.get_conn(REPORTS_DB_PATH)


class FloodReport(BaseModel):
    """Ground-truth flood observation from the field. These accumulate as
    candidate labels for retraining - the scarcest asset in this system."""

    text: str = Field(min_length=1, max_length=2000)
    phone: str | None = None
    ward: str | None = None
    county: str | None = None
    lat: float | None = Field(default=None, ge=-90, le=90)
    lon: float | None = Field(default=None, ge=-180, le=180)


def _insert_report(source: str, report: FloodReport) -> int:
    with _lock, _reports_db() as conn:
        cur = conn.execute(
            "INSERT INTO flood_reports "
            "(created_at, source, phone, ward, county, lat, lon, text) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                datetime.now(UTC).isoformat(),
                source,
                report.phone,
                report.ward,
                report.county,
                report.lat,
                report.lon,
                report.text,
            ),
        )
        return int(cur.lastrowid)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_version": _registry()["version"]}


@app.get("/registry")
def registry() -> dict:
    return _registry()


@app.get("/wards/risk")
def wards_risk(
    county: str | None = Query(default=None),
    threshold: float | None = Query(default=None, ge=0.0, le=1.0),
) -> dict:
    reg = _registry()
    thr = threshold if threshold is not None else reg["threshold"]
    wards = _scored_wards()
    if county:
        wards = wards[wards["county"].str.lower() == county.lower()]
        if wards.empty:
            raise HTTPException(status_code=404, detail=f"Unknown county: {county}")
    records = (
        wards[["ward", "subcounty", "county", "flood_prob"]]
        .assign(high_risk=lambda d: d["flood_prob"] >= thr)
        .sort_values("flood_prob", ascending=False)
        .to_dict(orient="records")
    )
    return {
        "threshold": thr,
        "rainfall": "historical (CHIRPS Feb-Apr 2024)",
        "n_wards": len(records),
        "n_high_risk": int(sum(r["high_risk"] for r in records)),
        "wards": records,
    }


@app.get("/wards/{ward}/risk")
def ward_risk(ward: str) -> dict:
    wards = _scored_wards()
    match = wards[wards["ward"].str.lower() == ward.lower()]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Unknown ward: {ward}")
    row = match.iloc[0]
    return {
        "ward": row["ward"],
        "subcounty": row["subcounty"],
        "county": row["county"],
        "flood_prob": float(row["flood_prob"]),
        "high_risk": bool(row["flood_prob"] >= _registry()["threshold"]),
        "features": {
            col: float(row[col]) for col in _registry()["feature_cols"]
        },
    }


@app.get("/reroutes")
def reroutes(
    preference: str = Query(default="balanced"),
    threshold: float | None = Query(default=None, ge=0.0, le=1.0),
) -> dict:
    """Rerouting options for affected matatu routes.

    Serves the precomputed cache written by ``scripts/refresh_cache.py`` when
    it is fresh and matches the requested threshold; otherwise computes on
    demand (loads the ~87k-node road graph on first call - expect ~30s).
    """
    from Utils.live_routing import ALPHA_OPTIONS, select_option

    labels = [label for label, _ in ALPHA_OPTIONS]
    if preference not in labels:
        raise HTTPException(
            status_code=422, detail=f"preference must be one of {labels}"
        )
    thr = threshold if threshold is not None else _registry()["threshold"]
    payload = _reroutes_payload(thr)

    options_df = pd.DataFrame(payload["options"])
    picked = (
        select_option(options_df, preference).to_dict(orient="records")
        if not options_df.empty
        else []
    )
    return {
        "threshold": thr,
        "preference": preference,
        "generated_at": payload["generated_at"],
        "served_from": payload["served_from"],
        "meta": payload["meta"],
        "routes": picked,
        "all_options": payload["options"],
    }


def _reroutes_payload(threshold: float) -> dict:
    """The rerouting payload shared by /reroutes and /reroutes/gtfs-rt:
    precomputed cache when fresh and threshold-matched, else on demand."""
    if PRECOMPUTED_REROUTES.exists():
        with open(PRECOMPUTED_REROUTES, encoding="utf-8") as f:
            candidate = json.load(f)
        age_h = (time.time() - candidate["generated_at_unix"]) / 3600
        if age_h <= PRECOMPUTED_MAX_AGE_HOURS and abs(
            candidate["threshold"] - threshold
        ) < 1e-9:
            candidate["served_from"] = "precomputed"
            return candidate

    payload = _compute_reroutes(threshold)
    payload["served_from"] = "on_demand"
    return payload


def _gtfs_tables() -> dict:
    with _lock:
        if "gtfs" not in _cache:
            _cache["gtfs"] = {
                "stops": pd.read_csv(GTFS_DIR / "stops.txt"),
                "stop_times": pd.read_csv(GTFS_DIR / "stop_times.txt"),
                "trips": pd.read_csv(GTFS_DIR / "trips.txt"),
            }
    return _cache["gtfs"]


def _affected_stop_ids(nairobi: gpd.GeoDataFrame, threshold: float) -> list[str]:
    """Stop IDs inside currently high-risk wards. Computed from the scored
    wards + GTFS tables alone (no road graph needed), so it stays cheap."""
    from Utils.live_routing import compute_affected_routes

    gtfs = _gtfs_tables()
    stops_gdf = gpd.GeoDataFrame(
        gtfs["stops"],
        geometry=gpd.points_from_xy(gtfs["stops"]["stop_lon"], gtfs["stops"]["stop_lat"]),
        crs="EPSG:4326",
    )
    _routes, affected = compute_affected_routes(
        nairobi, stops_gdf, gtfs["stop_times"], gtfs["trips"], threshold
    )
    return sorted(affected)


def _compute_reroutes(threshold: float) -> dict:
    from Utils.live_routing import load_road_graph, run_live_rerouting

    with _lock:
        if "graph" not in _cache:
            _cache["graph"] = load_road_graph(ROAD_GRAPH)
    gtfs = _gtfs_tables()
    wards = _scored_wards()
    nairobi = wards[wards["county"].str.lower() == "nairobi"].copy()
    options_df, _geoms, meta = run_live_rerouting(
        _cache["graph"],
        nairobi,
        gtfs["stops"],
        gtfs["stop_times"],
        gtfs["trips"],
        threshold=threshold,
    )
    now = datetime.now(UTC)
    return {
        "generated_at": now.isoformat(),
        "generated_at_unix": now.timestamp(),
        "threshold": threshold,
        "meta": meta,
        "options": options_df.to_dict(orient="records"),
        "affected_stop_ids": _affected_stop_ids(nairobi, threshold),
    }


@app.get("/reroutes/gtfs-rt")
def reroutes_gtfs_rt(
    threshold: float | None = Query(default=None, ge=0.0, le=1.0),
) -> Response:
    """The current rerouting option set as a GTFS-Realtime v2.0 feed
    (protobuf), immediately consumable by existing transit infrastructure:
    one ``ADDED`` TripUpdate per trip of every affected route, with stops in
    high-risk wards flagged ``SKIPPED``."""
    from Utils.gtfs_rt import build_gtfs_rt_feed

    thr = threshold if threshold is not None else _registry()["threshold"]
    payload = _reroutes_payload(thr)
    gtfs = _gtfs_tables()

    affected = payload.get("affected_stop_ids")
    if affected is None:
        # Cache file predates the affected-stop-ids field; recompute cheaply.
        wards = _scored_wards()
        nairobi = wards[wards["county"].str.lower() == "nairobi"].copy()
        affected = _affected_stop_ids(nairobi, thr)

    blob = build_gtfs_rt_feed(
        pd.DataFrame(payload["options"]),
        gtfs["trips"],
        gtfs["stop_times"],
        affected,
    )
    return Response(content=blob, media_type="application/x-protobuf")


@app.post("/reports", status_code=201)
def create_report(report: FloodReport) -> dict:
    report_id = _insert_report("api", report)
    return {"id": report_id, "status": "stored"}


@app.post("/reports/sms", status_code=201)
def create_report_sms(
    text: str = Form(...),
    sender: str = Form(default="", alias="from"),
) -> dict:
    """Africa's Talking inbound-SMS webhook. AT posts form-encoded fields
    including 'from' (sender MSISDN) and 'text' (message body); extra fields
    such as 'to', 'date' and 'id' are ignored."""
    report = FloodReport(text=text, phone=sender or None)
    report_id = _insert_report("sms", report)
    return {"id": report_id, "status": "stored"}


@app.get("/reports")
def list_reports(limit: int = Query(default=100, ge=1, le=1000)) -> dict:
    rows = alert_store.list_reports(limit=limit, db_path=REPORTS_DB_PATH)
    return {"n": len(rows), "reports": rows}


class Subscriber(BaseModel):
    """SMS/WhatsApp alert subscription for one ward or whole county."""

    phone: str = Field(pattern=r"^\+[1-9]\d{6,14}$")
    ward_or_county: str = Field(min_length=1, max_length=100)
    language: str = Field(default="en", pattern=r"^(en|sw)$")


@app.post("/subscribers", status_code=201)
def create_subscriber(sub: Subscriber) -> dict:
    # Idempotent on (phone, ward_or_county): re-subscribing reactivates.
    sub_id, _created = alert_store.add_subscriber(
        sub.phone, sub.ward_or_county, db_path=REPORTS_DB_PATH, language=sub.language
    )
    return {"id": sub_id, "status": "active"}


@app.delete("/subscribers")
def delete_subscriber(phone: str, ward_or_county: str) -> dict:
    n = alert_store.unsubscribe(phone, ward_or_county, db_path=REPORTS_DB_PATH)
    if n == 0:
        raise HTTPException(status_code=404, detail="No such subscription")
    return {"status": "deactivated"}


@app.get("/alerts")
def list_alert_history(limit: int = Query(default=200, ge=1, le=1000)) -> dict:
    """The alert audit log. Phone numbers are masked - they only ever leave
    the system through the SMS send path itself."""
    rows = alert_store.list_alerts(limit=limit, db_path=REPORTS_DB_PATH)
    for r in rows:
        r["phone"] = alert_store.mask_phone(r["phone"])
    return {"n": len(rows), "alerts": rows}


@app.get("/alerts/feed")
def alerts_feed(limit: int = Query(default=50, ge=1, le=200)) -> dict:
    """Public JSON feed of recent alerts (no auth). Picker-uppable by partners."""
    rows = alert_store.list_alerts(limit=limit, db_path=REPORTS_DB_PATH)
    items = []
    for r in rows:
        items.append(
            {
                "id": r.get("alert_id") or r["id"],
                "timestamp": r["timestamp"],
                "ward": r.get("ward"),
                "severity": r.get("severity"),
                "channel": r.get("channel", "sms"),
                "status": r["status"],
                "message": r["message"],
            }
        )
    return {
        "title": "Nairobi Flood Guard Live Alerts",
        "sender": "Nairobi Flood Guard (complements KMD & Kenya Red Cross)",
        "updated": items[0]["timestamp"] if items else None,
        "alerts": items,
    }


@app.get("/alerts/feed/rss")
def alerts_feed_rss(limit: int = Query(default=50, ge=1, le=200)) -> Response:
    """RSS 2.0 wrapper around the JSON alert feed."""
    payload = alerts_feed(limit=limit)
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        "<rss version=\"2.0\">",
        "<channel>",
        f"<title>{payload['title']}</title>",
        f"<description>{payload['sender']}</description>",
    ]
    for item in payload["alerts"]:
        lines.extend(
            [
                "<item>",
                f"<title>{item.get('ward') or 'Flood alert'} [{item.get('severity', '')}]</title>",
                f"<pubDate>{item['timestamp']}</pubDate>",
                f"<description>{item['message']}</description>",
                "</item>",
            ]
        )
    lines.extend(["</channel>", "</rss>"])
    return Response(
        content="\n".join(lines), media_type="application/rss+xml; charset=utf-8"
    )


@app.get("/alerts/cap/feed")
def alerts_cap_feed(limit: int = Query(default=20, ge=1, le=100)) -> Response:
    """CAP 1.2 XML feed of recent sent alerts."""
    from Utils.alerts import Alert, Certainty, Severity, Urgency

    rows = [
        r
        for r in alert_store.list_alerts(limit=limit, db_path=REPORTS_DB_PATH)
        if r["status"] == "sent"
    ]
    if not rows:
        empty = Alert(
            description_en="No active alerts.",
            instruction_en="Monitor KMD official advisories.",
            severity=Severity.MINOR,
            urgency=Urgency.PAST,
            certainty=Certainty.UNLIKELY,
        )
        return Response(content=empty.to_cap_xml(), media_type="application/xml")

    # Bundle as CAP feed (concatenated alerts - common for simple feeds)
    alerts_xml = []
    for r in rows:
        alert = Alert(
            ward=r.get("ward") or "",
            description_en=r["message"],
            instruction_en="Follow local authority guidance.",
            severity=Severity(r["severity"]) if r.get("severity") else Severity.MODERATE,
            identifier=str(r.get("alert_id") or r["id"]),
        )
        alerts_xml.append(alert.to_cap_xml())
    body = "\n".join(alerts_xml)
    return Response(content=body, media_type="application/xml")
