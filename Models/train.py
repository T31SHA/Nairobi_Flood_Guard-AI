"""
Canonical training pipeline for the flood susceptibility model.

Replaces the per-notebook training flows as the single source of truth.
Design decisions (and why they differ from the original notebooks):

- **Spatial cross-validation.** Wards are grouped by county with GroupKFold,
  so a model is always evaluated on counties it never saw. A random ward
  split leaks spatially autocorrelated terrain/rainfall between train and
  test and materially inflates AUC; both numbers are reported side by side
  in the registry so the optimism is visible.
- **No leaked or fabricated features.** Features come exclusively from
  ``Utils.feature_engineering.engineer_features`` (see that module's
  docstring for what was removed and why).
- **Probability calibration.** Downstream consumers (route costs, risk maps)
  use the probabilities directly, so the final model is wrapped in isotonic
  calibration fitted with the same county-grouped splits.
- **Threshold from the PR curve.** The operating threshold is chosen from
  calibrated out-of-fold predictions as the highest-precision point that
  still meets the recall target, instead of a hardcoded constant.
- **Registry as the contract.** ``model_registry.json`` records the model
  path, exact feature list, threshold and metrics. The app and API read all
  of these from the registry so they can never drift from training.

Run:  python -m Models.train  (from the repository root)
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import sklearn
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Utils.feature_engineering import FEATS, engineer_features  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
FLOODS_GPKG = BASE / "Data" / "floods.gpkg"
MODEL_PATH = BASE / "Models" / "flood_model.joblib"
BOOSTER_JSON_PATH = BASE / "Models" / "flood_model_xgb.json"
REGISTRY_PATH = BASE / "Models" / "model_registry.json"

RANDOM_STATE = 2026
N_SPLITS = 5
TARGET_RECALL = 0.80

PARAM_GRID = [
    {"max_depth": d, "learning_rate": lr}
    for d in (3, 4, 6)
    for lr in (0.03, 0.1)
]

BASE_PARAMS = dict(
    n_estimators=400,
    subsample=0.9,
    colsample_bytree=0.9,
    tree_method="hist",
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)


def make_model(params: dict, y_train: np.ndarray) -> xgb.XGBClassifier:
    pos = max(int(y_train.sum()), 1)
    neg = len(y_train) - pos
    return xgb.XGBClassifier(
        **BASE_PARAMS, **params, scale_pos_weight=neg / pos
    )


def spatial_oof_predictions(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    params: dict,
    calibrate: bool,
) -> np.ndarray:
    """County-grouped out-of-fold probabilities.

    With ``calibrate=True`` each fold's model is itself an isotonic
    CalibratedClassifierCV fitted with inner county-grouped splits, so the
    OOF probabilities honestly reflect the deployed (calibrated) model.
    """
    oof = np.full(len(y), np.nan)
    for train_idx, test_idx in GroupKFold(n_splits=N_SPLITS).split(X, y, groups):
        model = make_model(params, y[train_idx])
        if calibrate:
            inner = list(
                GroupKFold(n_splits=3).split(
                    X[train_idx], y[train_idx], groups[train_idx]
                )
            )
            model = CalibratedClassifierCV(model, method="isotonic", cv=inner)
        model.fit(X[train_idx], y[train_idx])
        oof[test_idx] = model.predict_proba(X[test_idx])[:, 1]
    assert not np.isnan(oof).any()
    return oof


def random_cv_auc(X: np.ndarray, y: np.ndarray, params: dict) -> float:
    """ROC AUC under a plain random (non-spatial) split, reported only to
    quantify how much a random split flatters the model."""
    oof = np.full(len(y), np.nan)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    for train_idx, test_idx in skf.split(X, y):
        model = make_model(params, y[train_idx])
        model.fit(X[train_idx], y[train_idx])
        oof[test_idx] = model.predict_proba(X[test_idx])[:, 1]
    return float(roc_auc_score(y, oof))


def choose_threshold(y: np.ndarray, probs: np.ndarray) -> dict:
    """Highest-precision threshold that still achieves TARGET_RECALL."""
    precision, recall, thresholds = precision_recall_curve(y, probs)
    # precision_recall_curve returns len(thresholds) + 1 precision/recall
    # entries; align by dropping the final (recall=0) point.
    precision, recall = precision[:-1], recall[:-1]
    meets = recall >= TARGET_RECALL
    if meets.any():
        best = np.argmax(np.where(meets, precision, -1))
    else:  # fall back to max-F1 if the recall target is unreachable
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        best = int(np.argmax(f1))
    return {
        "threshold": float(thresholds[best]),
        "precision_at_threshold": float(precision[best]),
        "recall_at_threshold": float(recall[best]),
    }


def main() -> None:
    print(f"Loading {FLOODS_GPKG} ...")
    wards = gpd.read_file(FLOODS_GPKG)
    wards = engineer_features(wards)

    X = wards[FEATS].to_numpy(dtype=float)
    y = wards["flooded"].to_numpy(dtype=int)
    groups = wards["county"].to_numpy()
    print(f"{len(y)} wards, {y.mean():.1%} flooded, {len(set(groups))} counties")

    # --- Hyperparameter search under spatial CV (uncalibrated, PR AUC) ---
    best_params, best_pr_auc = None, -1.0
    for params in PARAM_GRID:
        oof = spatial_oof_predictions(X, y, groups, params, calibrate=False)
        pr_auc = average_precision_score(y, oof)
        print(f"  params={params}  spatial PR AUC={pr_auc:.4f}")
        if pr_auc > best_pr_auc:
            best_params, best_pr_auc = params, pr_auc
    print(f"Selected params: {best_params} (PR AUC {best_pr_auc:.4f})")

    # --- Honest metrics: calibrated, spatially out-of-fold ---
    oof_cal = spatial_oof_predictions(X, y, groups, best_params, calibrate=True)
    thr = choose_threshold(y, oof_cal)
    y_pred = (oof_cal >= thr["threshold"]).astype(int)
    metrics_spatial = {
        "roc_auc": float(roc_auc_score(y, oof_cal)),
        "pr_auc": float(average_precision_score(y, oof_cal)),
        "brier": float(brier_score_loss(y, oof_cal)),
        "recall": float(recall_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred)),
        "f1": float(f1_score(y, y_pred)),
    }
    print("Spatial OOF (calibrated):", json.dumps(metrics_spatial, indent=2))

    rnd_auc = random_cv_auc(X, y, best_params)
    print(
        f"Random-split ROC AUC {rnd_auc:.4f} vs spatial {metrics_spatial['roc_auc']:.4f}"
        f" (optimism {rnd_auc - metrics_spatial['roc_auc']:+.4f})"
    )

    # --- Final deployable model: calibrated, fitted on all data ---
    final_splits = list(GroupKFold(n_splits=N_SPLITS).split(X, y, groups))
    final_model = CalibratedClassifierCV(
        make_model(best_params, y), method="isotonic", cv=final_splits
    )
    final_model.fit(X, y)
    joblib.dump(final_model, MODEL_PATH)
    print(f"Saved calibrated model -> {MODEL_PATH}")

    # Portable, version-independent fallback artifact (uncalibrated booster).
    booster_only = make_model(best_params, y)
    booster_only.fit(X, y)
    booster_only.save_model(BOOSTER_JSON_PATH)
    print(f"Saved native XGBoost booster -> {BOOSTER_JSON_PATH}")

    registry = {
        "version": "3.0",
        "created_at": datetime.now(UTC).isoformat(),
        "model_path": str(MODEL_PATH.relative_to(BASE)),
        "booster_json_path": str(BOOSTER_JSON_PATH.relative_to(BASE)),
        "feature_cols": FEATS,
        "n_features": len(FEATS),
        "threshold": thr["threshold"],
        "threshold_policy": (
            f"max precision subject to recall >= {TARGET_RECALL} on calibrated "
            "spatial out-of-fold predictions"
        ),
        "precision_at_threshold": thr["precision_at_threshold"],
        "recall_at_threshold": thr["recall_at_threshold"],
        "calibration": "isotonic (CalibratedClassifierCV, county-grouped folds)",
        "cv": {
            "scheme": f"GroupKFold({N_SPLITS}) grouped by county",
            "n_groups": int(len(set(groups))),
        },
        "model_params": {**BASE_PARAMS, **best_params},
        "metrics_spatial_oof": metrics_spatial,
        "metrics_random_split": {
            "roc_auc": rnd_auc,
            "note": (
                "Random ward split leaks spatially autocorrelated signal; "
                "reported only to quantify optimism vs the spatial CV."
            ),
        },
        "labels": {
            "events": ["FL20240426KEN (UNOSAT, April 2024)"],
            "note": (
                "Single-event labels. To add events: stack rows per "
                "(ward, event) with event-specific rainfall columns and an "
                "'event_id' column, then group CV folds by county as before "
                "and additionally report per-event holdout metrics."
            ),
        },
        "library_versions": {
            "python": platform.python_version(),
            "xgboost": xgb.__version__,
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
        },
    }
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)
    print(f"Wrote registry -> {REGISTRY_PATH}")


if __name__ == "__main__":
    main()
