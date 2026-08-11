"""Autonomous threshold-crossing alerts: the half of "early warning" that
reaches out instead of waiting to be looked at.

Flow (driven by ``scripts/refresh_cache.py`` on its schedule):

    score wards -> diff against previous run's snapshot (cache/last_scored.json)
    -> wards that NEWLY crossed from below-threshold to at-or-above-threshold
       (or escalated into the critical band while already above threshold)
    -> active subscribers for those wards / their counties
    -> build CAP-shaped Alert objects (Utils.alerts)
    -> send via SMS/WhatsApp through Utils.sms_sender
    -> log every decision into the alerts_sent table

Idempotency: only a genuine *new* crossing fires (never a ward that stays
above threshold across runs), and the snapshot is updated even when sending
is impossible (no credentials), so a crossing is never re-alerted later.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from Utils import alert_queue, alert_store
from Utils.alerts import Alert, Language, build_alert
from Utils.sms_sender import SmsConfigError, send_sms, send_whatsapp

BASE = Path(__file__).resolve().parent.parent
SNAPSHOT_PATH = BASE / "cache" / "last_scored.json"
CRITICAL_PROB = 0.70  # mirrors app_lib.theme.risk_label's critical band


@dataclass
class Crossing:
    ward: str
    county: str
    prev_prob: float | None
    curr_prob: float
    kind: str  # "new" | "escalation"


def ward_key(ward: str, county: str) -> str:
    """Ward names are not nationally unique; key snapshots by (ward, county)."""
    return f"{ward.strip().title()}|{county.strip().title()}"


def load_snapshot(path: Path | str = SNAPSHOT_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_snapshot(scored: pd.DataFrame, threshold: float, path=SNAPSHOT_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "threshold": threshold,
        "wards": {
            ward_key(row.ward, row.county): {"prob": float(row.flood_prob)}
            for row in scored.itertuples()
        },
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp, path)


def find_threshold_crossings(
    prev: dict,
    scored: pd.DataFrame,
    threshold: float,
    critical: float = CRITICAL_PROB,
) -> list[Crossing]:
    prev_wards = prev.get("wards", {})
    crossings = []
    for row in scored.itertuples():
        key = ward_key(row.ward, row.county)
        curr = float(row.flood_prob)
        prev_entry = prev_wards.get(key)
        prev_prob = float(prev_entry["prob"]) if prev_entry else None
        if prev_prob is None:
            continue
        if prev_prob < threshold <= curr:
            crossings.append(
                Crossing(row.ward, row.county, prev_prob, curr, "new")
            )
        elif threshold <= prev_prob < critical <= curr:
            crossings.append(
                Crossing(row.ward, row.county, prev_prob, curr, "escalation")
            )
    return crossings


def match_subscribers(
    crossings: list[Crossing], subscribers: list[dict]
) -> list[tuple[dict, Crossing]]:
    pairs = []
    for sub in subscribers:
        target = sub["ward_or_county"].strip().lower()
        for c in crossings:
            if target in (c.ward.strip().lower(), c.county.strip().lower()):
                pairs.append((sub, c))
    return pairs


def _baseline_crossings(scored: pd.DataFrame, threshold: float) -> list[Crossing]:
    return [
        Crossing(row.ward, row.county, None, float(row.flood_prob), "new")
        for row in scored.itertuples()
        if float(row.flood_prob) >= threshold
    ]


def crossing_to_alert(
    crossing: Crossing,
    threshold: float,
    language: Language = "en",
    horizon_hours: int = 0,
) -> Alert:
    return build_alert(
        crossing.ward,
        crossing.county,
        crossing.curr_prob,
        threshold,
        prev_prob=crossing.prev_prob,
        kind=crossing.kind,
        horizon_hours=horizon_hours,
        language=language,
    )


def build_alert_message(crossing: Crossing, threshold: float) -> str:
    """Backward-compatible SMS text (English)."""
    return crossing_to_alert(crossing, threshold).to_sms("en")


def dispatch_alert(
    alert: Alert,
    phone: str,
    *,
    channel: str = "sms",
    username: str | None = None,
    api_key: str | None = None,
    queue_on_failure: bool = True,
    db_path=None,
) -> bool:
    """Send one alert on the requested channel. Returns True on success."""
    lang: Language = alert.language
    message = alert.to_sms(lang)
    try:
        if channel == "whatsapp":
            result = send_whatsapp(
                message, [phone], username=username, api_key=api_key
            )
        else:
            result = send_sms(message, [phone], username=username, api_key=api_key)
        ok = phone in result["sent"] or not result["failed"]
        if not ok and queue_on_failure:
            alert_queue.enqueue(phone, channel, alert, db_path=db_path)
        return ok
    except (SmsConfigError, Exception):
        if queue_on_failure:
            alert_queue.enqueue(phone, channel, alert, db_path=db_path)
        raise


def process_alerts(
    scored: pd.DataFrame,
    threshold: float,
    snapshot_path=SNAPSHOT_PATH,
    db_path=None,
    alert_on_baseline: bool = False,
    username: str | None = None,
    api_key: str | None = None,
    channels: tuple[str, ...] = ("sms",),
    horizon_hours: int = 0,
    dispatch=None,
) -> dict:
    """Diff -> notify -> log -> snapshot. Returns a run summary."""
    send_fn = dispatch or dispatch_alert
    prev = load_snapshot(snapshot_path)
    is_baseline = not prev.get("wards")

    if is_baseline:
        crossings = _baseline_crossings(scored, threshold) if alert_on_baseline else []
    else:
        crossings = find_threshold_crossings(prev, scored, threshold)

    subscribers = alert_store.list_subscribers(active_only=True, db_path=db_path)
    pairs = match_subscribers(crossings, subscribers)

    summary = {"crossings": crossings, "sent": 0, "failed": 0, "skipped": 0, "queued": 0}

    with alert_store.get_conn(db_path) as conn:
        for crossing in crossings:
            targets = [sub for sub, c in pairs if c is crossing]
            if not targets:
                alert = crossing_to_alert(crossing, threshold)
                alert_store.log_alert(
                    crossing.ward,
                    None,
                    alert.to_sms("en"),
                    "no_subscribers",
                    conn=conn,
                    severity=alert.severity.value,
                    alert_id=alert.identifier,
                )
                summary["skipped"] += 1
                continue
            for sub in targets:
                lang: Language = sub.get("language", "en") or "en"
                alert = crossing_to_alert(
                    crossing, threshold, language=lang, horizon_hours=horizon_hours
                )
                phone = sub["phone"]
                for channel in channels:
                    message = alert.to_sms(lang)
                    try:
                        ok = send_fn(
                            alert,
                            phone,
                            channel=channel,
                            username=username,
                            api_key=api_key,
                            queue_on_failure=dispatch is None,
                            db_path=db_path,
                        )
                        status = "sent" if ok else "failed"
                    except SmsConfigError:
                        status = "no_credentials"
                        alert_queue.enqueue(phone, channel, alert, db_path=db_path, conn=conn)
                        summary["queued"] += 1
                    except Exception:
                        status = "failed"
                        summary["queued"] += 1
                    alert_store.log_alert(
                        crossing.ward,
                        phone,
                        message,
                        status,
                        conn=conn,
                        channel=channel,
                        severity=alert.severity.value,
                        alert_id=alert.identifier,
                    )
                    if status == "sent":
                        summary["sent"] += 1
                    elif status == "no_credentials":
                        summary["skipped"] += 1
                    else:
                        summary["failed"] += 1

    save_snapshot(scored, threshold, snapshot_path)

    if api_key:
        retry = alert_queue.process_pending(
            dispatch_alert,
            db_path=db_path,
            username=username,
            api_key=api_key,
        )
        summary["retry"] = retry

    return summary
