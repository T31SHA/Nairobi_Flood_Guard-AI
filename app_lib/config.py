"""Paths, constants and the model registry (single source of truth)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import streamlit as st

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "Data"
MODELS = BASE / "Models"
GTFS_DIR = DATA / "GTFS_FEED_2019"
REPORTS = BASE / "Route_Optimization" / "Reports"

FLOODS_GPKG = DATA / "floods.gpkg"
REGISTRY_PATH = MODELS / "model_registry.json"
REROUTING_CSV = REPORTS / "rerouting_summary.csv"
TRADEOFF_PNG = REPORTS / "rerouting_tradeoff.png"
ROUTE_GEOMETRIES = REPORTS / "route_geometries.json"
ROAD_GRAPH = DATA / "nairobi_road_network.graphml"
MODEL_COMPARISON_CSV = DATA / "model_comparison.csv"
PRECOMPUTED_REROUTES = BASE / "cache" / "precomputed_reroutes.json"

NAIROBI_LAT, NAIROBI_LON = -1.286389, 36.817223
GROQ_MODEL = "llama-3.3-70b-versatile"


@lru_cache(maxsize=1)
def load_registry() -> dict:
    """The model registry written by Models/train.py. The app reads the
    model path, feature list and operating threshold from here so they can
    never drift from what training produced."""
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        return json.load(f)


def registry_feature_cols() -> list[str]:
    return list(load_registry()["feature_cols"])


def registry_threshold() -> float:
    return float(load_registry()["threshold"])


def registry_model_path() -> Path:
    return BASE / load_registry()["model_path"]


def registry_booster_path() -> Path:
    return BASE / load_registry()["booster_json_path"]


def get_secret(key: str, default=None):
    """st.secrets raises StreamlitSecretNotFoundError when no secrets file
    exists at all (fresh local checkout), rather than behaving like an empty
    mapping - so every optional secret goes through this guard."""
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default
