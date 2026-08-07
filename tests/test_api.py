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
