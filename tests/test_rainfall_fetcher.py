"""Rainfall ingestion: feature windows, horizon filtering, retry handling."""

import json
from datetime import datetime, timedelta

import pytest

from Utils import rainfall_fetcher as rf


def test_compute_features_windows():
    daily = [1.0] * 100  # 100 days of 1mm
    feats = rf._compute_features(daily)
    assert feats["rain_cumulative_mm"] == pytest.approx(90.0)  # capped at 90d
    assert feats["rain_max_daily_mm"] == pytest.approx(1.0)
    assert feats["rain_preflood_7d_mm"] == pytest.approx(7.0)


def test_compute_features_handles_empty_and_nan():
    assert rf._compute_features([]) == {c: 0.0 for c in rf.RAIN_COLS}
    feats = rf._compute_features([float("nan"), 5.0])
    assert feats["rain_cumulative_mm"] == pytest.approx(5.0)


def test_horizon_features_filter_future_days():
    today = datetime.now(rf.NAIROBI_TZ).date()
    dates = [(today + timedelta(days=d)).isoformat() for d in range(-3, 3)]
    daily = [1.0, 1.0, 1.0, 1.0, 10.0, 20.0]  # 10mm tomorrow, 20mm day after
    by_horizon = rf._compute_horizon_features(daily, dates, horizons_hours=(0, 24, 48))
    assert by_horizon["0"]["rain_cumulative_mm"] == pytest.approx(4.0)
    assert by_horizon["24"]["rain_cumulative_mm"] == pytest.approx(14.0)
    assert by_horizon["48"]["rain_cumulative_mm"] == pytest.approx(34.0)


class FakeResponse:
    def __init__(self, status_code=200, payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else []
        self.headers = headers or {}
        self.text = json.dumps(self._payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise rf.requests.HTTPError(f"HTTP {self.status_code}")


def test_fetch_batch_retries_on_429_then_succeeds(monkeypatch):
    calls = {"n": 0}
    payload = [{"daily": {"time": [], "precipitation_sum": []}}]

    def fake_get(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return FakeResponse(429, headers={"Retry-After": "0"})
        return FakeResponse(200, payload)

    monkeypatch.setattr(rf.requests, "get", fake_get)
    monkeypatch.setattr(rf.time, "sleep", lambda _s: None)
    out = rf._fetch_batch([0.0], [36.8])
    assert out == payload
    assert calls["n"] == 2


def test_fetch_batch_client_error_raises(monkeypatch):
    monkeypatch.setattr(
        rf.requests, "get", lambda *a, **k: FakeResponse(400, {"reason": "bad"})
    )
    with pytest.raises(RuntimeError, match="rejected"):
        rf._fetch_batch([0.0], [36.8])


def test_fetch_grid_rainfall_rejects_mismatched_batch(monkeypatch):
    # 2 points requested, 1 result returned -> must refuse to zip silently
    monkeypatch.setattr(
        rf,
        "_fetch_batch",
        lambda *a, **k: [{"daily": {"time": [], "precipitation_sum": []}}],
    )
    with pytest.raises(RuntimeError, match="refusing"):
        rf.fetch_grid_rainfall([(0.0, 36.8), (0.1, 36.9)])


def test_file_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(rf, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(rf, "CACHE_FILE", tmp_path / "rain.json")
    grid = {(0.0, 36.8): {"0": {c: 1.0 for c in rf.RAIN_COLS}}}
    meta = {
        "fetched_at": datetime.now(rf.timezone.utc).isoformat(),
        "source": "test",
        "grid_resolution_deg": 0.25,
        "n_grid_points": 1,
        "past_days": 90,
        "forecast_days": 3,
        "forecast_horizons_hours": [0],
    }
    rf._save_file_cache(grid, meta)
    cached = rf._load_file_cache(max_age_hours=1.0)
    assert cached is not None
    parsed = rf._parse_cached_grid(cached)
    assert parsed == grid
    assert rf._cache_has_horizon(cached, 0)
    assert not rf._cache_has_horizon(cached, 24)
