"""Cached data/model loaders and prediction helpers.

Everything model-related is driven by the registry (Models/model_registry.json):
the model artifact, the exact feature list and the operating threshold all come
from there, so the app can never drift from what training produced.
"""

from __future__ import annotations

import hashlib
import json

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

from app_lib.config import (
    FLOODS_GPKG,
    GTFS_DIR,
    MODEL_COMPARISON_CSV,
    PRECOMPUTED_REROUTES,
    REROUTING_CSV,
    ROAD_GRAPH,
    ROUTE_GEOMETRIES,
    get_secret,
    registry_booster_path,
    registry_feature_cols,
    registry_model_path,
)
from app_lib.theme import risk_label
from Utils.feature_engineering import engineer_features
from Utils.live_routing import run_live_rerouting
from Utils.rainfall_fetcher import RAIN_COLS, apply_live_rainfall

try:
    import osmnx as ox  # noqa: F401

    OSMNX_AVAILABLE = True
except ImportError:
    OSMNX_AVAILABLE = False


@st.cache_data
def load_data():
    df = gpd.read_file(FLOODS_GPKG)
    df["elevation_range_m"] = df["elevation_max_m"] - df["elevation_min_m"]
    df["ward"] = df["ward"].str.title()
    df["county"] = df["county"].str.title()
    df["subcounty"] = df["subcounty"].str.title()
    return df


@st.cache_resource
def load_model():
    """The calibrated production model referenced by the registry."""
    return joblib.load(registry_model_path())


@st.cache_resource
def load_explainer_model():
    """Uncalibrated native-XGBoost booster for SHAP explanations.

    The production model is a CalibratedClassifierCV ensemble, which SHAP's
    TreeExplainer can't unwrap. Isotonic calibration is monotonic, so the
    uncalibrated booster's feature attributions explain the same ranking.
    """
    clf = xgb.XGBClassifier()
    clf.load_model(registry_booster_path())
    return clf


@st.cache_resource(show_spinner=False)
def load_road_graph():
    """~87k nodes / ~213k edges - load once per process, never per rerun."""
    from Utils.live_routing import load_road_graph as _load

    return _load(ROAD_GRAPH)


@st.cache_data
def load_gtfs():
    routes = pd.read_csv(GTFS_DIR / "routes.txt")
    trips = pd.read_csv(GTFS_DIR / "trips.txt")
    shapes = pd.read_csv(GTFS_DIR / "shapes.txt")
    stops = pd.read_csv(GTFS_DIR / "stops.txt")
    stop_times = pd.read_csv(GTFS_DIR / "stop_times.txt")
    return routes, trips, shapes, stops, stop_times


@st.cache_data
def load_rerouting():
    """Legacy single-option rerouting CSV from the April 2024 notebook run,
    normalised to the current options-table format."""
    df = pd.read_csv(REROUTING_CSV)
    if "option" not in df.columns:
        df["option"] = "safest"
        df["alpha"] = 1_000_000.0
        df["same_as_original"] = False
        for col in (
            "original_max_flood_prob",
            "original_risk_time_frac",
            "alternative_max_flood_prob",
            "alternative_risk_time_frac",
            "stops_total",
            "stops_served",
            "stops_dropped",
        ):
            df[col] = pd.NA
    return df


@st.cache_data
def load_model_comparison(path):
    return pd.read_csv(path)


def model_comparison_path():
    return MODEL_COMPARISON_CSV


@st.cache_data
def load_route_geometries():
    if not ROUTE_GEOMETRIES.exists():
        return {}
    with open(ROUTE_GEOMETRIES) as f:
        return json.load(f)


@st.cache_data(ttl=60, show_spinner=False)
def load_precomputed_reroutes():
    """The latest scheduled-refresh payload (cache/precomputed_reroutes.json)
    written by scripts/refresh_cache.py, or None when absent/stale. Lets the
    dashboard quote rerouting impact numbers without paying the graph-load
    cost on every rerun."""
    if not PRECOMPUTED_REROUTES.exists():
        return None
    try:
        with open(PRECOMPUTED_REROUTES, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def generate_predictions(model, df):
    engineered = engineer_features(df)
    feature_cols = registry_feature_cols()
    X = engineered[feature_cols].fillna(engineered[feature_cols].median())
    return model.predict_proba(X)[:, 1]


def add_risk_columns(model, df):
    scored = df.copy()
    scored["flood_prob"] = generate_predictions(model, scored)
    scored["risk_label"], _ = zip(*scored["flood_prob"].map(risk_label))
    return scored


def flood_prob_fingerprint(wards_gdf, threshold, horizon_hours=0) -> str:
    """Hash of ward flood_prob values + threshold/horizon - changes whenever
    the underlying risk data (or routing params) actually change, so the
    live-routing cache invalidates correctly without hashing the full gdf."""
    vals = tuple(
        np.round(wards_gdf.sort_values("ward")["flood_prob"].values, 4).tolist()
    )
    return hashlib.md5(f"{vals}-{threshold}-{horizon_hours}".encode()).hexdigest()


@st.cache_resource(max_entries=1, show_spinner=False)
def get_weighted_graph(_G, _wards_gdf, fingerprint):
    """Flood-weighted graph + edge probability map for on-demand routing
    (e.g. the stop-preserving detour). cache_resource stores by reference,
    so the graph copy is never pickled; the fingerprint of the underlying
    flood probabilities invalidates it when risk actually changes."""
    from Utils.live_routing import build_flood_weighted_graph, compute_edge_flood_map

    fmap = compute_edge_flood_map(_G, _wards_gdf)
    return build_flood_weighted_graph(_G, fmap), fmap


@st.cache_data(ttl=3600, max_entries=3, show_spinner=False)
def get_live_routes(_G, _wards_gdf, _stops_df, _stop_times, _trips, threshold, fingerprint):
    """
    Recompute flood-weighted rerouting against current ward flood_prob.
    Leading-underscore args are excluded from Streamlit's hash (the graph and
    GTFS tables are large/static); `fingerprint` - a hash of the actual
    flood_prob values plus threshold - is what invalidates the cache when
    risk data genuinely changes. Returns the full Pareto option set.
    """
    return run_live_rerouting(
        _G, _wards_gdf, _stops_df, _stop_times, _trips, threshold=threshold
    )


def apply_horizon_rainfall(gdf, horizon_hours: int, use_cache: bool):
    vc_key = get_secret("VISUALCROSSING_API_KEY")
    return apply_live_rainfall(
        gdf,
        use_cache=use_cache,
        horizon_hours=horizon_hours,
        visualcrossing_api_key=vc_key,
    )


@st.cache_data(ttl=21600, max_entries=6, show_spinner=False)
def get_open_meteo_ward_dataframe(
    cache_bust: int, skip_file_cache: bool, horizon_hours: int
):
    """
    Fetch Open-Meteo rainfall for Nairobi wards only (~91 wards) and merge it
    into the full nationwide dataframe; other counties keep their historical
    CHIRPS values. Horizon 0 is the live/as-of-now dataset, while 24 and 48 use
    forecast precipitation rolled into the model rainfall windows.
    """
    base = load_data()
    nairobi_mask = base["county"].str.lower() == "nairobi"

    forecast_nairobi, meta = apply_horizon_rainfall(
        base[nairobi_mask],
        horizon_hours=horizon_hours,
        use_cache=not skip_file_cache,
    )

    combined = base.copy()
    combined.loc[nairobi_mask, RAIN_COLS] = forecast_nairobi[RAIN_COLS].values
    meta["scope"] = (
        f"Nairobi ({int(nairobi_mask.sum())} wards) · other counties remain historical"
    )
    return combined, meta
