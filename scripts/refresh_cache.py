"""
Scheduled precompute: refresh live rainfall and rerouting results.

Fetches the latest Open-Meteo rainfall for Nairobi wards (writing the shared
``cache/rainfall_live.json`` used by the app), scores flood probabilities
with the registry model, runs the full Pareto rerouting pipeline, and writes
``cache/precomputed_reroutes.json`` for the API's /reroutes endpoint.

Run it on a schedule (e.g. a Render cron job or GitHub Actions cron) so user
requests never pay the graph-load + Dijkstra cost:

    python -m scripts.refresh_cache [--threshold 0.30] [--skip-rainfall]
                                    [--no-alerts] [--alert-on-baseline]

It is also the early-warning loop: after scoring, it diffs ward probabilities
against the previous run's snapshot (``cache/last_scored.json``) and sends
SMS alerts to subscribers of wards that *newly* crossed the threshold (see
Utils/alerting.py for the idempotency rules), logging every decision to the
alerts_sent table.

The cache directory is ephemeral on most PaaS filesystems; on Render attach
a persistent disk at the repo's ``cache/`` path or point CACHE_DIR elsewhere.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import joblib
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from scripts.verify_data_assets import check_assets  # noqa: E402
from Utils.alerting import process_alerts  # noqa: E402
from Utils.feature_engineering import engineer_features  # noqa: E402
from Utils.live_routing import (  # noqa: E402
    compute_affected_routes,
    load_road_graph,
    run_live_rerouting,
)
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
    parser.add_argument(
        "--no-alerts",
        action="store_true",
        help="Skip the threshold-crossing alert evaluation",
    )
    parser.add_argument(
        "--alert-on-baseline",
        action="store_true",
        help="Alert for all currently-high wards on the first ever run "
        "(default: first run only establishes the baseline snapshot)",
    )
    args = parser.parse_args()

    # Fail fast on a corrupt/placeholder asset (e.g. an un-pulled Git LFS
    # pointer) rather than 30s into the graph load with a cryptic XML error.
    problems = check_assets()
    if problems:
        raise SystemExit(
            "Data asset check FAILED:\n" + "\n".join(f"  - {p}" for p in problems)
        )

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

    if not args.no_alerts:
        summary = process_alerts(
            wards[["ward", "county", "flood_prob"]],
            threshold,
            alert_on_baseline=args.alert_on_baseline,
            username=os.environ.get("AT_USERNAME"),
            api_key=os.environ.get("AT_API_KEY"),
        )
        n_new = sum(1 for c in summary["crossings"] if c.kind == "new")
        n_esc = sum(1 for c in summary["crossings"] if c.kind == "escalation")
        print(
            f"Alert check: {n_new} new crossing(s), {n_esc} escalation(s); "
            f"{summary['sent']} sent, {summary['failed']} failed, "
            f"{summary['skipped']} logged without send"
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

    # Stop IDs inside high-risk wards, persisted so /reroutes/gtfs-rt can flag
    # SKIPPED stops consistently with the cached options (no recompute drift).
    stops_gdf = gpd.GeoDataFrame(
        stops,
        geometry=gpd.points_from_xy(stops["stop_lon"], stops["stop_lat"]),
        crs="EPSG:4326",
    )
    _affected_routes, affected_stop_ids = compute_affected_routes(
        nairobi, stops_gdf, stop_times, trips, threshold
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
                "affected_stop_ids": sorted(affected_stop_ids),
            },
            f,
        )
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
