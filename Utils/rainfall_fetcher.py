"""
Near-real-time rainfall ingestion for Nairobi Flood Guard.

Fetches daily precipitation from the Open-Meteo Forecast API (no API key required),
aggregates to ward level via a spatial grid, and computes time-indexed rainfall
features used by the XGBoost model:

  - rain_cumulative_mm  : total over the last 90 days
  - rain_max_daily_mm   : maximum single-day total in that window
  - rain_preflood_7d_mm : total over the last 7 days

Open-Meteo rate-limits by weighted call count (locations x days/14 x variables/10).
Large batches with 90-day windows are heavily weighted, so requests are kept small
with pauses between batches and results are cached aggressively.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import requests

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
VISUALCROSSING_URL = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
NAIROBI_TZ = timezone(timedelta(hours=3))  # EAT, fixed offset, no DST
DEFAULT_GRID_RES = 0.25  # degrees (~28 km; fewer API calls than 0.1 deg)
# Open-Meteo accepts up to 1000 locations per request (free tier: 600 req/min,
# 5000/hr, 10000/day). A 90-day/1-variable request weighs ~0.64 per location,
# so even 300 locations in one call is ~190 weight - nowhere near the limit.
# There is normally no need for more than one batch; this cap just protects
# against an unexpectedly huge grid (e.g. a future nationwide-live toggle).
BATCH_SIZE = 500
BATCH_DELAY_SEC = 1.0  # only matters if grid_points > BATCH_SIZE
MAX_RETRIES = 3
REQUEST_TIMEOUT_SEC = 20  # Open-Meteo normally responds in <1s; fail fast, don't hang
PAST_DAYS = 90
FORECAST_DAYS = 3
FORECAST_HORIZONS_HOURS = (0, 24, 48)
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
CACHE_FILE = CACHE_DIR / "rainfall_live.json"
RAIN_COLS = ["rain_cumulative_mm", "rain_max_daily_mm", "rain_preflood_7d_mm"]


def _asof_date(horizon_hours: int) -> datetime.date:
    return (datetime.now(NAIROBI_TZ) + timedelta(hours=horizon_hours)).date()


def _compute_features(
    daily_mm: list[float],
    daily_dates: list[Any] | None = None,
    horizon_hours: int = 0,
) -> dict[str, float]:
    """`daily_dates`, if given, must already be `date` objects (see
    _compute_horizon_features, which parses the ISO strings once up front
    instead of re-parsing them for every horizon)."""
    if daily_dates:
        asof = _asof_date(horizon_hours)
        daily_mm = [mm for day, mm in zip(daily_dates, daily_mm) if day <= asof]

    arr = np.array(daily_mm, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0)
    if len(arr) == 0:
        return {col: 0.0 for col in RAIN_COLS}

    last_90 = arr[-90:] if len(arr) >= 90 else arr
    last_7 = arr[-7:] if len(arr) >= 7 else arr
    return {
        "rain_cumulative_mm": float(last_90.sum()),
        "rain_max_daily_mm": float(last_90.max()),
        "rain_preflood_7d_mm": float(last_7.sum()),
    }


def _compute_horizon_features(
    daily_mm: list[float],
    daily_dates: list[str] | None = None,
    horizons_hours: tuple[int, ...] = FORECAST_HORIZONS_HOURS,
) -> dict[str, dict[str, float]]:
    parsed_dates = (
        [datetime.fromisoformat(d).date() for d in daily_dates] if daily_dates else None
    )
    return {
        str(horizon): _compute_features(daily_mm, parsed_dates, horizon)
        for horizon in horizons_hours
    }


def _fetch_batch(
    lats: list[float],
    lons: list[float],
    past_days: int = PAST_DAYS,
    forecast_days: int = FORECAST_DAYS,
) -> list[dict[str, Any]]:
    """Fetch one batch with retries and backoff on HTTP 429.

    Streamlit Cloud occasionally behaves differently from local environments
    around outbound POST requests, so try the documented GET request first and
    keep POST as a fallback for long coordinate lists.
    """
    params = {
        "latitude": ",".join(f"{lat:.4f}" for lat in lats),
        "longitude": ",".join(f"{lon:.4f}" for lon in lons),
        "daily": "precipitation_sum",
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": "Africa/Nairobi",
    }
    headers = {
        "User-Agent": "Nairobi-Flood-Guard/1.0 (+https://streamlit.io)",
        "Accept": "application/json",
    }

    last_error: Exception | None = None
    last_detail = "no response"
    methods = ("GET", "POST")

    for attempt in range(MAX_RETRIES):
        already_waited = False
        for method in methods:
            try:
                if method == "GET":
                    resp = requests.get(
                        OPEN_METEO_URL,
                        params=params,
                        headers=headers,
                        timeout=REQUEST_TIMEOUT_SEC,
                    )
                else:
                    resp = requests.post(
                        OPEN_METEO_URL,
                        data=params,
                        headers=headers,
                        timeout=REQUEST_TIMEOUT_SEC,
                    )

                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    wait = (
                        float(retry_after) if retry_after else min(20, 5 * (2**attempt))
                    )
                    last_detail = f"{method} HTTP 429 rate limited"
                    time.sleep(wait)
                    already_waited = True
                    break

                if resp.status_code == 414 and method == "GET":
                    last_detail = "GET HTTP 414 URI too long; retrying with POST"
                    continue

                if 400 <= resp.status_code < 500:
                    raise RuntimeError(
                        f"Open-Meteo rejected {method} request "
                        f"(HTTP {resp.status_code}): {resp.text[:300]}"
                    )

                resp.raise_for_status()
                data = resp.json()
                return data if isinstance(data, list) else [data]

            except RuntimeError:
                raise
            except requests.RequestException as exc:
                last_error = exc
                last_detail = f"{method} {exc.__class__.__name__}: {exc}"
                continue

        if not already_waited and attempt < MAX_RETRIES - 1:
            time.sleep(min(10, 2 * (2**attempt)))

    raise RuntimeError(
        f"Open-Meteo request failed after {MAX_RETRIES} attempts; "
        f"last error: {last_detail}"
    ) from last_error


def _fetch_visualcrossing_point(
    lat: float,
    lon: float,
    api_key: str,
    past_days: int,
    forecast_days: int,
) -> dict[str, Any]:
    """One grid point per request - Visual Crossing's Timeline API doesn't
    support Open-Meteo-style comma-separated multi-location batching. Shapes
    the response identically to Open-Meteo's so downstream feature code
    (_compute_horizon_features etc.) needs no changes."""
    start = (datetime.now(NAIROBI_TZ) - timedelta(days=past_days)).date().isoformat()
    end = (datetime.now(NAIROBI_TZ) + timedelta(days=forecast_days)).date().isoformat()
    url = f"{VISUALCROSSING_URL}/{lat:.4f},{lon:.4f}/{start}/{end}"

    resp = requests.get(
        url,
        params={
            "key": api_key,
            "unitGroup": "metric",
            "include": "days",
            "elements": "datetime,precip",
            "contentType": "json",
        },
        timeout=REQUEST_TIMEOUT_SEC,
    )
    resp.raise_for_status()
    data = resp.json()
    days = data.get("days", [])
    return {
        "daily": {
            "time": [d["datetime"] for d in days],
            "precipitation_sum": [d.get("precip") or 0.0 for d in days],
        }
    }


def fetch_grid_rainfall_visualcrossing(
    grid_points: list[tuple[float, float]],
    api_key: str,
    past_days: int = PAST_DAYS,
    forecast_days: int = FORECAST_DAYS,
    horizons_hours: tuple[int, ...] = FORECAST_HORIZONS_HOURS,
) -> dict[tuple[float, float], dict[str, dict[str, float]]]:
    """1,000 free records/day, rate-limited per API key (not per shared IP),
    so this works fine on platforms like Streamlit Cloud where Open-Meteo's
    keyless per-IP limit gets exhausted by unrelated apps sharing the same
    egress IP. Each (past_days + forecast_days) window costs ~1 record per
    day per point, so keep grid_points small and cache aggressively."""
    features: dict[tuple[float, float], dict[str, dict[str, float]]] = {}
    for lat, lon in grid_points:
        payload = _fetch_visualcrossing_point(
            lat, lon, api_key, past_days, forecast_days
        )
        precip = payload["daily"]["precipitation_sum"]
        dates = payload["daily"]["time"]
        features[(lat, lon)] = _compute_horizon_features(
            precip, dates, horizons_hours=horizons_hours
        )
    return features


def fetch_grid_rainfall(
    grid_points: list[tuple[float, float]],
    past_days: int = PAST_DAYS,
    forecast_days: int = FORECAST_DAYS,
    horizons_hours: tuple[int, ...] = FORECAST_HORIZONS_HOURS,
    on_progress: Any | None = None,
) -> dict[tuple[float, float], dict[str, dict[str, float]]]:
    """Fetch precipitation features for a list of (lat, lon) grid cells."""
    features: dict[tuple[float, float], dict[str, dict[str, float]]] = {}
    n_batches = max(1, (len(grid_points) + BATCH_SIZE - 1) // BATCH_SIZE)

    for batch_idx, i in enumerate(range(0, len(grid_points), BATCH_SIZE)):
        batch = grid_points[i : i + BATCH_SIZE]
        lats = [p[0] for p in batch]
        lons = [p[1] for p in batch]
        responses = _fetch_batch(
            lats, lons, past_days=past_days, forecast_days=forecast_days
        )
        if len(responses) != len(batch):
            raise RuntimeError(
                f"Open-Meteo returned {len(responses)} results for a "
                f"{len(batch)}-point batch; refusing to silently zip "
                "mismatched results onto the wrong grid points."
            )

        for point, payload in zip(batch, responses):
            daily = payload.get("daily", {})
            precip = daily.get("precipitation_sum", [])
            dates = daily.get("time", [])
            features[point] = _compute_horizon_features(
                precip, dates, horizons_hours=horizons_hours
            )

        if on_progress is not None:
            on_progress(batch_idx + 1, n_batches)

        if i + BATCH_SIZE < len(grid_points):
            time.sleep(BATCH_DELAY_SEC)

    return features


def _load_file_cache(max_age_hours: float | None) -> dict[str, Any] | None:
    if not CACHE_FILE.exists():
        return None

    try:
        with open(CACHE_FILE, encoding="utf-8") as f:
            cached = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    if max_age_hours is None:
        return cached

    fetched_at = datetime.fromisoformat(cached["fetched_at"])
    age_hours = (datetime.now(UTC) - fetched_at).total_seconds() / 3600
    if age_hours > max_age_hours:
        return None

    return cached


def _save_file_cache(
    grid_features: dict[tuple[float, float], dict[str, dict[str, float]]],
    meta: dict[str, Any],
) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "fetched_at": meta["fetched_at"],
            "source": meta["source"],
            "grid_resolution_deg": meta["grid_resolution_deg"],
            "n_grid_points": meta["n_grid_points"],
            "past_days": meta["past_days"],
            "forecast_days": meta["forecast_days"],
            "forecast_horizons_hours": meta["forecast_horizons_hours"],
            "grid_features": {
                f"{lat},{lon}": feats for (lat, lon), feats in grid_features.items()
            },
        }
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except OSError:
        # Caching is an optimization, not a requirement.
        pass


def _parse_cached_grid(
    cached: dict[str, Any],
) -> dict[tuple[float, float], dict[str, dict[str, float]]]:
    grid_features: dict[tuple[float, float], dict[str, dict[str, float]]] = {}
    for key, feats in cached["grid_features"].items():
        lat_s, lon_s = key.split(",")
        if any(col in feats for col in RAIN_COLS):
            grid_features[(float(lat_s), float(lon_s))] = {"0": feats}
        else:
            grid_features[(float(lat_s), float(lon_s))] = feats
    return grid_features


def _cache_has_horizon(cached: dict[str, Any] | None, horizon_hours: int) -> bool:
    if cached is None:
        return False
    if not cached.get("grid_features"):
        return False
    if horizon_hours == 0:
        return True
    first_features = next(iter(cached["grid_features"].values()), {})
    return str(horizon_hours) in first_features


def _assign_rainfall(
    df: gpd.GeoDataFrame,
    grid_features: dict[tuple[float, float], dict[str, dict[str, float]]],
    horizon_hours: int = 0,
) -> gpd.GeoDataFrame:
    keys = list(zip(df["_grid_lat"], df["_grid_lon"]))
    horizon_key = str(horizon_hours)
    for col in RAIN_COLS:
        df[col] = [
            grid_features.get(k, {}).get(horizon_key, {}).get(col, 0.0) for k in keys
        ]
    return df.drop(columns=["_grid_lat", "_grid_lon"])


def apply_live_rainfall(
    gdf: gpd.GeoDataFrame,
    resolution: float = DEFAULT_GRID_RES,
    use_cache: bool = True,
    max_cache_age_hours: float = 6.0,
    horizon_hours: int = 0,
    on_progress: Any | None = None,
    visualcrossing_api_key: str | None = None,
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """
    Replace static rainfall columns with Open-Meteo values for a forecast horizon.

    Returns a copy of the GeoDataFrame and metadata about the fetch.
    Tries Open-Meteo first; if that fails (e.g. shared-IP rate limiting on
    platforms like Streamlit Cloud) and `visualcrossing_api_key` is provided,
    falls back to Visual Crossing (rate-limited per API key, not per IP).
    Falls back further to stale on-disk cache (up to 24 h old), and finally
    lets the exception propagate if nothing is available.
    """
    df = gdf.copy()
    # Compute centroids in a projected CRS (geographic-degree centroids are
    # slightly biased and geopandas warns about them), then bucket in WGS84.
    centroids = df.geometry.to_crs("EPSG:32737").centroid.to_crs("EPSG:4326")
    df["_grid_lat"] = np.round(centroids.y / resolution) * resolution
    df["_grid_lon"] = np.round(centroids.x / resolution) * resolution

    grid_points = sorted(set(zip(df["_grid_lat"], df["_grid_lon"])))
    meta: dict[str, Any] = {
        "source": "Open-Meteo Forecast API",
        "grid_resolution_deg": resolution,
        "n_grid_points": len(grid_points),
        "n_wards": len(df),
        "past_days": PAST_DAYS,
        "forecast_days": FORECAST_DAYS,
        "forecast_horizons_hours": list(FORECAST_HORIZONS_HOURS),
        "horizon_hours": horizon_hours,
        "from_cache": False,
        "stale_cache": False,
    }

    grid_features: dict[tuple[float, float], dict[str, dict[str, float]]] = {}

    if use_cache:
        cached = _load_file_cache(max_cache_age_hours)
        if (
            cached is not None
            and cached.get("grid_resolution_deg") == resolution
            and _cache_has_horizon(cached, horizon_hours)
        ):
            grid_features = _parse_cached_grid(cached)
            meta["fetched_at"] = cached["fetched_at"]
            meta["source"] = cached.get("source", meta["source"])
            meta["from_cache"] = True

    if not grid_features:
        try:
            grid_features = fetch_grid_rainfall(grid_points, on_progress=on_progress)
            meta["source"] = "Open-Meteo Forecast API"
            meta["fetched_at"] = datetime.now(UTC).isoformat()
            _save_file_cache(grid_features, meta)
        except Exception as open_meteo_exc:
            if visualcrossing_api_key:
                try:
                    grid_features = fetch_grid_rainfall_visualcrossing(
                        grid_points, visualcrossing_api_key
                    )
                    meta["source"] = "Visual Crossing (fallback)"
                    meta["fetched_at"] = datetime.now(UTC).isoformat()
                    meta["fallback_reason"] = str(open_meteo_exc)
                    _save_file_cache(grid_features, meta)
                except Exception as vc_exc:
                    stale = _load_file_cache(max_age_hours=24.0)
                    if stale is not None and _cache_has_horizon(stale, horizon_hours):
                        grid_features = _parse_cached_grid(stale)
                        meta["fetched_at"] = stale["fetched_at"]
                        meta["source"] = stale.get("source", meta["source"])
                        meta["from_cache"] = True
                        meta["stale_cache"] = True
                    else:
                        raise RuntimeError(
                            f"Open-Meteo failed ({open_meteo_exc}); "
                            f"Visual Crossing fallback also failed ({vc_exc})"
                        ) from vc_exc
            else:
                stale = _load_file_cache(max_age_hours=24.0)
                if stale is not None and _cache_has_horizon(stale, horizon_hours):
                    grid_features = _parse_cached_grid(stale)
                    meta["fetched_at"] = stale["fetched_at"]
                    meta["source"] = stale.get("source", meta["source"])
                    meta["from_cache"] = True
                    meta["stale_cache"] = True
                else:
                    raise

    df = _assign_rainfall(df, grid_features, horizon_hours=horizon_hours)
    return df, meta


def apply_forecast_rainfall(
    gdf: gpd.GeoDataFrame,
    horizon_hours: int,
    resolution: float = DEFAULT_GRID_RES,
    use_cache: bool = True,
    max_cache_age_hours: float = 6.0,
    on_progress: Any | None = None,
    visualcrossing_api_key: str | None = None,
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Replace rainfall columns with Open-Meteo features as-of a forecast horizon."""
    if horizon_hours not in FORECAST_HORIZONS_HOURS:
        raise ValueError(
            f"horizon_hours must be one of {FORECAST_HORIZONS_HOURS}; "
            f"got {horizon_hours}"
        )
    return apply_live_rainfall(
        gdf,
        resolution=resolution,
        use_cache=use_cache,
        max_cache_age_hours=max_cache_age_hours,
        horizon_hours=horizon_hours,
        on_progress=on_progress,
        visualcrossing_api_key=visualcrossing_api_key,
    )


def rainfall_summary(meta: dict[str, Any]) -> str:
    """Human-readable label for the active rainfall data source."""
    if meta.get("source") == "historical":
        return "Historical · CHIRPS Feb-Apr 2024"

    fetched = meta.get("fetched_at", "")
    if fetched:
        try:
            ts = datetime.fromisoformat(fetched).astimezone(NAIROBI_TZ)
            ts_label = ts.strftime("%Y-%m-%d %H:%M EAT")
        except ValueError:
            ts_label = fetched
    else:
        ts_label = "unknown"

    scope_note = f" · {meta['scope']}" if meta.get("scope") else ""
    horizon = int(meta.get("horizon_hours", 0) or 0)
    mode = "Live" if horizon == 0 else f"+{horizon}h forecast"
    provider = meta.get("source", "Open-Meteo Forecast API").replace(
        " Forecast API", ""
    )

    if meta.get("stale_cache"):
        return f"{mode} · {provider} · stale cache from {ts_label}{scope_note}"
    cache_note = " (cached)" if meta.get("from_cache") else ""
    return f"{mode} · {provider} · updated {ts_label}{cache_note}{scope_note}"
