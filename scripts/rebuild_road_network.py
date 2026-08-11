"""Regenerate Data/nairobi_road_network.graphml when Git LFS isn't available.

Run from repo root: python -m scripts.rebuild_road_network

API surface verified against the pinned osmnx==2.1.1: the edge-speed /
travel-time helpers live in ``ox.routing`` (they were top-level ``ox.*`` in
v1), and the bounding box is passed as a shapely polygon to sidestep the
bbox-tuple ordering changes across versions.

The bounding box is derived from the GTFS feed's own stops rather than a
place-name lookup on purpose: Nairobi County's official boundary is too
tight - it excludes real matatu termini like Ruiru (Kiambu County), which
appears in the README's rerouting examples. Buffering the stops' extent
guarantees every route terminal in the feed falls inside the graph.
"""

from __future__ import annotations

import time

import osmnx as ox
import pandas as pd
from shapely.geometry import box

from app_lib.config import GTFS_DIR, ROAD_GRAPH

MARGIN_DEG = 0.05  # ~5.5 km buffer past the furthest stop, for graph edges


def main() -> None:
    stops = pd.read_csv(GTFS_DIR / "stops.txt")

    # Derive programmatically (don't hardcode) so this stays correct if the
    # GTFS feed is ever refreshed (see README "Next Steps").
    west = stops["stop_lon"].min() - MARGIN_DEG
    east = stops["stop_lon"].max() + MARGIN_DEG
    south = stops["stop_lat"].min() - MARGIN_DEG
    north = stops["stop_lat"].max() + MARGIN_DEG
    print(f"Bounding box: ({west:.4f}, {south:.4f}) - ({east:.4f}, {north:.4f})")

    polygon = box(west, south, east, north)

    print("Downloading drive network from OpenStreetMap (this takes a while) ...")
    t0 = time.time()
    G = ox.graph_from_polygon(polygon, network_type="drive", simplify=True)
    G = ox.routing.add_edge_speeds(G)  # -> speed_kph edge attribute
    G = ox.routing.add_edge_travel_times(G)  # -> travel_time (seconds), required
    #                                            by Utils/live_routing.py
    print(
        f"  {G.number_of_nodes()} nodes / {G.number_of_edges()} edges "
        f"in {time.time() - t0:.0f}s"
    )

    ox.save_graphml(G, ROAD_GRAPH)
    print(f"Saved graph to {ROAD_GRAPH}")


if __name__ == "__main__":
    main()
