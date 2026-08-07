"""The registry is the training/serving contract - verify it end to end."""

import json
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pytest

from Utils.feature_engineering import FEATS, engineer_features

BASE = Path(__file__).resolve().parent.parent
REGISTRY_PATH = BASE / "Models" / "model_registry.json"
FLOODS_GPKG = BASE / "Data" / "floods.gpkg"


@pytest.fixture(scope="module")
def registry():
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_registry_matches_feature_pipeline(registry):
    assert registry["feature_cols"] == FEATS
    assert registry["n_features"] == len(FEATS)


def test_registry_has_honest_metrics(registry):
    assert "metrics_spatial_oof" in registry
    for metric in ("roc_auc", "pr_auc", "recall", "precision", "f1", "brier"):
        assert metric in registry["metrics_spatial_oof"]
    assert 0.0 < registry["threshold"] < 1.0
    assert "ward_hist_rate" not in registry["feature_cols"]


def test_model_artifacts_exist(registry):
    assert (BASE / registry["model_path"]).exists()
    assert (BASE / registry["booster_json_path"]).exists()


@pytest.mark.skipif(not FLOODS_GPKG.exists(), reason="data file not present")
def test_model_scores_real_wards(registry):
    model = joblib.load(BASE / registry["model_path"])
    wards = gpd.read_file(FLOODS_GPKG).head(50)
    wards = engineer_features(wards)
    probs = model.predict_proba(wards[registry["feature_cols"]])[:, 1]
    assert probs.shape == (50,)
    assert np.isfinite(probs).all()
    assert ((probs >= 0) & (probs <= 1)).all()
