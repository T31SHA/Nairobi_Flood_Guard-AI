"""
Scheduled precompute: refresh live rainfall and rerouting results.

Fetches the latest Open-Meteo rainfall for Nairobi wards (writing the shared
``cache/rainfall_live.json`` used by the app), scores flood probabilities
with the registry model, runs the full Pareto rerouting pipeline, and writes
``cache/precomputed_reroutes.json`` for the API's /reroutes endpoint.

Run it on a schedule (e.g. a Render cron job or GitHub Actions cron) so user
requests never pay the graph-load + Dijkstra cost:

    python -m scripts.refresh_cache [--threshold 0.30] [--skip-rainfall]

The cache directory is ephemeral on most PaaS filesystems; on Render attach
a persistent disk at the repo's ``cache/`` path or point CACHE_DIR elsewhere.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import joblib
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from Utils.feature_engineering import engineer_features  # noqa: E402
from Utils.live_routing import load_road_graph, run_live_rerouting  # noqa: E402
from Utils.rainfall_fetcher import RAIN_COLS, apply_live_rainfall  # noqa: E402

FLOODS_GPKG = BASE / "Data" / "floods.gpkg"
REGISTRY_PATH = BASE / "Models" / "model_registry.json"
ROAD_GRAPH = BASE / "Data" / "nairobi_road_network.graphml"
GTFS_DIR = BASE / "Data" / "GTFS_FEED_2019"
OUT_PATH = BASE / "cache" / "precomputed_reroutes.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="High-risk ward threshold (default: registry operating point)",
    )
    parser.add_argument(
        "--skip-rainfall",
        action="store_true",
        help="Skip the Open-Meteo fetch and score with on-disk rainfall",
    )
    args = parser.parse_args()

    registry = json.load(open(REGISTRY_PATH, encoding="utf-8"))
    threshold = args.threshold if args.threshold is not None else registry["threshold"]
    model = joblib.load(BASE / registry["model_path"])

    wards = gpd.read_file(FLOODS_GPKG)

    if not args.skip_rainfall:
        print("Fetching live rainfall for Nairobi wards ...")
        nairobi_mask = wards["county"].str.lower() == "nairobi"
        try:
            live_nairobi, meta = apply_live_rainfall(wards[nairobi_mask])
            wards.loc[nairobi_mask, RAIN_COLS] = live_nairobi[RAIN_COLS].values
            print(f"  rainfall source: {meta['source']} ({meta['n_grid_points']} grid points)")
        except Exception as exc:
            print(f"  rainfall fetch failed ({exc}); scoring with historical values")

    wards = engineer_features(wards)
    feature_cols = registry["feature_cols"]
    X = wards[feature_cols].fillna(wards[feature_cols].median())
    wards["flood_prob"] = model.predict_proba(X)[:, 1]
    nairobi = wards[wards["county"].str.lower() == "nairobi"].copy()
    print(
        f"Scored {len(wards)} wards; {int((nairobi['flood_prob'] >= threshold).sum())} "
        f"Nairobi wards >= {threshold:.2f}"
    )

    print("Loading road graph ...")
    G = load_road_graph(ROAD_GRAPH)
    stops = pd.read_csv(GTFS_DIR / "stops.txt")
    stop_times = pd.read_csv(GTFS_DIR / "stop_times.txt")
    trips = pd.read_csv(GTFS_DIR / "trips.txt")

    print("Running rerouting pipeline ...")
    t0 = time.time()
    options_df, geoms, meta = run_live_rerouting(
        G, nairobi, stops, stop_times, trips, threshold=threshold
    )
    print(
        f"  {meta['rerouted_routes']} / {meta['total_affected_routes']} routes "
        f"rerouted in {time.time() - t0:.0f}s"
    )

    now = datetime.now(UTC)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": now.isoformat(),
                "generated_at_unix": now.timestamp(),
                "threshold": threshold,
                "meta": meta,
                "options": options_df.to_dict(orient="records"),
                "geometries": geoms,
            },
            f,
        )
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
