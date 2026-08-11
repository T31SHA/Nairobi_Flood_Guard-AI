"""FastAPI service tests (graph-dependent /reroutes is exercised separately -
it needs the ~100MB road network, which is too heavy for unit CI)."""

import pytest
from fastapi.testclient import TestClient

import api.main as api_main


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api_main, "REPORTS_DB_PATH", tmp_path / "reports.db")
    return TestClient(api_main.app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_version"]


def test_registry_endpoint(client):
    body = client.get("/registry").json()
    assert "feature_cols" in body
    assert "threshold" in body


def test_wards_risk_county_filter(client):
    body = client.get("/wards/risk", params={"county": "Nairobi"}).json()
    assert body["n_wards"] > 0
    assert all(w["county"].lower() == "nairobi" for w in body["wards"])
    probs = [w["flood_prob"] for w in body["wards"]]
    assert probs == sorted(probs, reverse=True)


def test_wards_risk_unknown_county_404(client):
    assert client.get("/wards/risk", params={"county": "Atlantis"}).status_code == 404


def test_single_ward_lookup(client):
    listing = client.get("/wards/risk", params={"county": "Nairobi"}).json()
    ward = listing["wards"][0]["ward"]
    body = client.get(f"/wards/{ward}/risk").json()
    assert body["ward"] == ward
    assert 0.0 <= body["flood_prob"] <= 1.0
    assert set(body["features"]) == set(client.get("/registry").json()["feature_cols"])


def test_report_intake_json(client):
    resp = client.post(
        "/reports",
        json={"text": "Road under water near Ruai", "ward": "Ruai", "lat": -1.28, "lon": 36.9},
    )
    assert resp.status_code == 201
    listing = client.get("/reports").json()
    assert listing["n"] == 1
    assert listing["reports"][0]["ward"] == "Ruai"
    assert listing["reports"][0]["source"] == "api"


def test_report_intake_sms_webhook(client):
    resp = client.post(
        "/reports/sms",
        data={"from": "+254700000001", "text": "mafuriko mtaani", "to": "20880"},
    )
    assert resp.status_code == 201
    listing = client.get("/reports").json()
    assert listing["reports"][0]["phone"] == "+254700000001"
    assert listing["reports"][0]["source"] == "sms"


def test_report_validation(client):
    assert client.post("/reports", json={"text": ""}).status_code == 422
    assert client.post("/reports", json={"text": "x", "lat": 999}).status_code == 422


def test_reroutes_invalid_preference(client):
    assert client.get("/reroutes", params={"preference": "yolo"}).status_code == 422


def _stub_reroutes_payload(monkeypatch):
    """Synthetic rerouting payload + GTFS tables so /reroutes/gtfs-rt can be
    exercised without the ~100MB road network."""
    import pandas as pd

    payload = {
        "generated_at": "2026-01-01T00:00:00+00:00",
        "generated_at_unix": 1_767_225_600.0,
        "threshold": 0.3,
        "meta": {},
        "options": [{"route_id": "R1", "option": "balanced"}],
        "affected_stop_ids": ["s2"],
    }
    tables = {
        "stops": pd.DataFrame({"stop_id": ["s1", "s2"]}),
        "trips": pd.DataFrame({"route_id": ["R1"], "trip_id": ["T1"]}),
        "stop_times": pd.DataFrame(
            {"trip_id": ["T1", "T1"], "stop_id": ["s1", "s2"], "stop_sequence": [1, 2]}
        ),
    }
    monkeypatch.setattr(api_main, "_reroutes_payload", lambda thr: payload)
    monkeypatch.setattr(api_main, "_gtfs_tables", lambda: tables)


def test_reroutes_gtfs_rt_returns_parseable_protobuf(client, monkeypatch):
    from google.transit import gtfs_realtime_pb2

    _stub_reroutes_payload(monkeypatch)
    resp = client.get("/reroutes/gtfs-rt")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/x-protobuf")

    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(resp.content)
    assert len(feed.entity) == 1
    trip_update = feed.entity[0].trip_update
    assert trip_update.trip.trip_id == "T1"
    updates = list(trip_update.stop_time_update)
    rel = gtfs_realtime_pb2.TripUpdate.StopTimeUpdate
    assert updates[0].schedule_relationship == rel.SCHEDULED
    assert updates[1].schedule_relationship == rel.SKIPPED  # s2 is affected


def test_subscribe_and_unsubscribe(client):
    resp = client.post(
        "/subscribers", json={"phone": "+254712345678", "ward_or_county": "Kibera"}
    )
    assert resp.status_code == 201
    assert resp.json()["status"] == "active"

    # Invalid phone numbers are rejected like FloodReport field validation.
    assert (
        client.post(
            "/subscribers", json={"phone": "0712", "ward_or_county": "Kibera"}
        ).status_code
        == 422
    )
    assert (
        client.delete(
            "/subscribers",
            params={"phone": "+254712345678", "ward_or_county": "Kibera"},
        ).json()["status"]
        == "deactivated"
    )
    assert (
        client.delete(
            "/subscribers",
            params={"phone": "+254700000000", "ward_or_county": "Nowhere"},
        ).status_code
        == 404
    )


def test_alerts_audit_log_masks_phones(client):
    from Utils import alert_store

    alert_store.log_alert(
        "Kibera", "+254712345678", "FLOOD ALERT: ...", "sent",
        db_path=api_main.REPORTS_DB_PATH,
        severity="Severe",
        alert_id="test-alert-1",
    )
    body = client.get("/alerts").json()
    assert body["n"] == 1
    assert body["alerts"][0]["phone"] == "+254****5678"
    assert body["alerts"][0]["ward"] == "Kibera"


def test_alerts_public_feed(client):
    from Utils import alert_store

    alert_store.log_alert(
        "Ruai Ward", None, "FLOOD ALERT: Ruai", "sent",
        db_path=api_main.REPORTS_DB_PATH, severity="Severe",
    )
    body = client.get("/alerts/feed").json()
    assert "alerts" in body
    assert body["alerts"][0]["ward"] == "Ruai Ward"

    rss = client.get("/alerts/feed/rss")
    assert rss.status_code == 200
    assert "rss" in rss.text.lower()

    cap = client.get("/alerts/cap/feed")
    assert cap.status_code == 200
    assert "<alert" in cap.text


def test_subscriber_language(client):
    resp = client.post(
        "/subscribers",
        json={"phone": "+254799999999", "ward_or_county": "Ruai", "language": "sw"},
    )
    assert resp.status_code == 201
