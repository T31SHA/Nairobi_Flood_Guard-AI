"""Feature engineering must be leakage-free, complete, and deterministic."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

from Utils.feature_engineering import (
    FEATS,
    RAW_INPUT_COLS,
    engineer_features,
    ward_transport_exposure,
)


def make_wards(n=6, include_label=False) -> gpd.GeoDataFrame:
    rng = np.random.default_rng(7)
    polys = [
        Polygon(
            [
                (36.7 + 0.1 * i, -1.4),
                (36.8 + 0.1 * i, -1.4),
                (36.8 + 0.1 * i, -1.3),
                (36.7 + 0.1 * i, -1.3),
            ]
        )
        for i in range(n)
    ]
    df = gpd.GeoDataFrame(
        {
            "ward": [f"W{i}" for i in range(n)],
            "county": ["Nairobi"] * (n // 2) + ["Kiambu"] * (n - n // 2),
            "pop2009": rng.integers(5_000, 60_000, n),
            "rain_cumulative_mm": rng.uniform(0, 700, n),
            "rain_max_daily_mm": rng.uniform(0, 80, n),
            "rain_preflood_7d_mm": rng.uniform(0, 150, n),
            "elevation_mean_m": rng.uniform(100, 2500, n),
            "elevation_min_m": rng.uniform(50, 1500, n),
            "elevation_max_m": rng.uniform(1500, 3000, n),
            "slope_mean_deg": rng.uniform(0.5, 15, n),
        },
        geometry=polys,
        crs="EPSG:4326",
    )
    if include_label:
        df["flooded"] = rng.integers(0, 2, n)
    return df


def test_works_without_label_column():
    """Leakage regression test: the pipeline must never require (or read)
    the 'flooded' label, so it can run on wards whose outcome is unknown."""
    df = make_wards(include_label=False)
    out = engineer_features(df)
    for feat in FEATS:
        assert feat in out.columns, f"missing feature {feat}"


def test_label_does_not_influence_features():
    df = make_wards(include_label=True)
    flipped = df.copy()
    flipped["flooded"] = 1 - flipped["flooded"]
    a = engineer_features(df)[FEATS]
    b = engineer_features(flipped)[FEATS]
    pd.testing.assert_frame_equal(a, b)


def test_all_features_finite():
    out = engineer_features(make_wards())
    vals = out[FEATS].to_numpy(dtype=float)
    assert np.isfinite(vals).all()


def test_deterministic():
    df = make_wards()
    a = engineer_features(df)[FEATS]
    b = engineer_features(df)[FEATS]
    pd.testing.assert_frame_equal(a, b)


def test_missing_raw_column_raises():
    df = make_wards().drop(columns=["slope_mean_deg"])
    with pytest.raises(ValueError, match="slope_mean_deg"):
        engineer_features(df)


def test_no_fabricated_constants():
    """Every feature must vary across wards - constant columns are dead
    weight (the old pipeline shipped is_terminal=0, is_long_rain=1, etc.)."""
    out = engineer_features(make_wards(n=10))
    nunique = out[FEATS].nunique()
    assert (nunique > 1).all(), f"constant features: {list(nunique[nunique <= 1].index)}"


def test_ward_transport_exposure_counts():
    wards = make_wards(n=2)
    # two stops in ward 0, one in ward 1
    stops = pd.DataFrame(
        {
            "stop_id": ["s1", "s2", "s3"],
            "stop_lon": [36.75, 36.76, 36.85],
            "stop_lat": [-1.35, -1.35, -1.35],
        }
    )
    stop_times = pd.DataFrame(
        {
            "trip_id": ["t1", "t1", "t2"],
            "stop_id": ["s1", "s2", "s3"],
        }
    )
    trips = pd.DataFrame(
        {
            "trip_id": ["t1", "t2"],
            "route_id": ["r1", "r2"],
        }
    )
    out = ward_transport_exposure(wards, stops, stop_times, trips)
    assert out.loc[0, "n_stops"] == 2
    assert out.loc[0, "n_routes"] == 1
    assert out.loc[1, "n_stops"] == 1
    assert out.loc[1, "n_routes"] == 1
    assert (out["stop_density"] >= 0).all()


def test_raw_input_cols_documented():
    assert set(RAW_INPUT_COLS).isdisjoint({"flooded"})
