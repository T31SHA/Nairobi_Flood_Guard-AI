"""
Live flood-aware matatu rerouting for Nairobi Flood Guard.

Turns ward flood probabilities into alternative-route recommendations:

    ward flood_prob
        -> spatial join onto OSMnx road edges
        -> flood_cost = travel_time * (1 + alpha * flood_prob), per alpha
        -> weighted Dijkstra per affected route (terminal-to-terminal)
        -> Pareto option set + stop-coverage report per route

Design notes:

- **Exposure-weighted risk.** A path's flood risk is the travel-time-weighted
  mean of edge flood probabilities (plus the max edge probability and the
  share of travel time spent on high-risk edges). An unweighted mean would
  score a route that clips one flooded 50 m segment the same as one spending
  10 km in a flood zone.
- **Pareto options, not a single extreme.** Each affected route gets one
  alternative per alpha in ``ALPHA_OPTIONS`` (deduplicated when they collapse
  to the same path). Alpha = 1e6 blocks any flood-touched road outright and
  frequently costs 100+ extra minutes; the milder alphas surface "most of
  the risk reduction at a fraction of the detour" choices that operators can
  actually act on.
- **Stop coverage.** Terminal-to-terminal rerouting says nothing about the
  intermediate stops passengers depend on, so every option reports how many
  of the route's original stops remain within ``SERVICE_RADIUS_M`` of the
  alternative path, and which ones are dropped.
- **Cost/runtime.** The graph is large (~87k nodes / ~213k edges); loading it
  is the expensive part - always cache with st.cache_resource. One weighted
  graph copy carries a cost attribute per alpha, so the option set costs one
  extra Dijkstra per alpha per route (seconds overall), not extra graph
  copies.
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import LineString, Point

WGS84 = "EPSG:4326"
METRIC = "EPSG:32737"

# (label, alpha). Labels are stable identifiers used by the UI and API.
ALPHA_OPTIONS: tuple[tuple[str, float], ...] = (
    ("fastest", 5.0),
    ("balanced", 50.0),
    ("safest", 1_000_000.0),
)
# A stop still counts as served if the alternative path passes within this
# distance of it (walkable transfer to the diverted route).
SERVICE_RADIUS_M = 300.0


def load_road_graph(graphml_path) -> nx.MultiDiGraph:
    """Load the OSMnx road network. Callers should wrap this in
    st.cache_resource - it shouldn't be reloaded on every script rerun."""
    return ox.load_graphml(graphml_path)


def compute_edge_flood_map(
    G: nx.MultiDiGraph, wards_gdf: gpd.GeoDataFrame
) -> dict[tuple[Any, Any, int], float]:
    """Assign each road edge the flood_prob of the ward its midpoint falls in.
    Edges outside any ward default to 0.0."""
    edges_gdf = ox.graph_to_gdfs(G, nodes=False, edges=True)[["geometry"]].reset_index()
    # Interpolate midpoints in a metric CRS - doing it in geographic degrees
    # skews midpoints on long east-west edges.
    midpoints = (
        edges_gdf["geometry"].to_crs(METRIC).interpolate(0.5, normalized=True)
    ).to_crs(WGS84)
    edges_mid = gpd.GeoDataFrame(
        edges_gdf[["u", "v", "key"]], geometry=midpoints, crs=WGS84
    )

    edges_joined = gpd.sjoin(
        edges_mid,
        wards_gdf[["flood_prob", "ward", "geometry"]],
        how="left",
        predicate="within",
    )
    edges_joined = (
        edges_joined.groupby(["u", "v", "key"])["flood_prob"].max().reset_index()
    )
    edges_joined["flood_prob"] = edges_joined["flood_prob"].fillna(0.0)

    return {
        (row.u, row.v, row.key): row.flood_prob for row in edges_joined.itertuples()
    }


def _alpha_weight_key(label: str) -> str:
    return f"flood_cost_{label}"


def build_flood_weighted_graph(
    G: nx.MultiDiGraph,
    flood_prob_map: dict[tuple[Any, Any, int], float],
    alphas: tuple[tuple[str, float], ...] = ALPHA_OPTIONS,
) -> nx.MultiDiGraph:
    """Return a COPY of G carrying one 'flood_cost_<label>' attribute per
    alpha option. A single copy (rather than one per alpha) keeps memory flat
    on constrained deployments while still letting nx.shortest_path use a
    plain attribute name per option."""
    G = G.copy()
    for label, alpha in alphas:
        costs = {
            (u, v, key): data.get("travel_time", 60)
            * (1 + alpha * flood_prob_map.get((u, v, key), 0.0))
            for u, v, key, data in G.edges(keys=True, data=True)
        }
        nx.set_edge_attributes(G, costs, _alpha_weight_key(label))
    return G


def _get_route_terminals(
    route_id: str, trips: pd.DataFrame, stop_times: pd.DataFrame, stops: pd.DataFrame
) -> tuple:
    trip_id = trips[trips["route_id"] == route_id]["trip_id"].iloc[0]
    route_stops = (
        stop_times[stop_times["trip_id"] == trip_id]
        .sort_values("stop_sequence")
        .merge(stops, on="stop_id")
    )
    origin = route_stops.iloc[0]
    destination = route_stops.iloc[-1]
    return (
        (origin["stop_lat"], origin["stop_lon"], origin["stop_name"]),
        (destination["stop_lat"], destination["stop_lon"], destination["stop_name"]),
    )


def get_ordered_route_stops(
    route_id: str, trips: pd.DataFrame, stop_times: pd.DataFrame, stops: pd.DataFrame
) -> pd.DataFrame:
    """Ordered stops (first trip) for a route, with lat/lon and names."""
    trip_rows = trips[trips["route_id"] == route_id]
    if trip_rows.empty:
        return pd.DataFrame()
    trip_id = trip_rows.iloc[0]["trip_id"]
    return (
        stop_times[stop_times["trip_id"] == trip_id]
        .sort_values("stop_sequence")
        .merge(stops, on="stop_id")
    )


def _best_parallel_edge(G: nx.MultiDiGraph, u, v, weight_key: str):
    """(u, v) can have multiple parallel edges (keys). nx.shortest_path picks
    whichever parallel edge is cheapest at each hop but doesn't report which
    key it used, so reconstruct it here rather than assuming key 0 - which
    silently misreports metrics (or KeyErrors and drops the whole route) on
    any node pair with more than one edge between them."""
    edges = G[u][v]
    best_key = min(edges, key=lambda k: edges[k].get(weight_key, 0))
    return best_key, edges[best_key]


def path_metrics(
    path: list,
    G: nx.MultiDiGraph,
    flood_prob_map: dict[tuple[Any, Any, int], float],
    weight_key: str,
    risk_threshold: float,
) -> dict[str, float]:
    """Exposure-weighted risk metrics for a node path.

    ``flood_prob`` is the travel-time-weighted mean of edge probabilities;
    ``max_flood_prob`` the worst single edge; ``risk_time_frac`` the share of
    travel time spent on edges at or above ``risk_threshold``.
    """
    total_time = 0.0
    weighted_prob = 0.0
    risk_time = 0.0
    max_prob = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        key, edge_data = _best_parallel_edge(G, u, v, weight_key)
        t = float(edge_data.get("travel_time", 0.0))
        prob = float(flood_prob_map.get((u, v, key), 0.0))
        total_time += t
        weighted_prob += prob * t
        max_prob = max(max_prob, prob)
        if prob >= risk_threshold:
            risk_time += t
    return {
        "time_s": total_time,
        "flood_prob": (weighted_prob / total_time) if total_time > 0 else 0.0,
        "max_flood_prob": max_prob,
        "risk_time_frac": (risk_time / total_time) if total_time > 0 else 0.0,
    }


def _path_to_coords(path: list, G: nx.MultiDiGraph) -> list:
    return [[G.nodes[node]["y"], G.nodes[node]["x"]] for node in path]


def stop_coverage(
    path: list,
    G: nx.MultiDiGraph,
    route_stops: pd.DataFrame,
    radius_m: float = SERVICE_RADIUS_M,
) -> dict[str, Any]:
    """Which of the route's original stops the given path still serves.

    A stop is served if it lies within ``radius_m`` of the path polyline
    (computed in a metric CRS). Returns counts plus the dropped stop names.
    """
    if route_stops.empty or len(path) < 2:
        return {
            "stops_total": int(len(route_stops)),
            "stops_served": 0,
            "stops_dropped": int(len(route_stops)),
            "dropped_stop_names": route_stops.get("stop_name", pd.Series()).tolist(),
        }

    coords = [(G.nodes[n]["x"], G.nodes[n]["y"]) for n in path]  # lon, lat
    line = (
        gpd.GeoSeries([LineString(coords)], crs=WGS84).to_crs(METRIC).iloc[0]
    )
    pts = gpd.GeoSeries(
        [Point(xy) for xy in zip(route_stops["stop_lon"], route_stops["stop_lat"])],
        crs=WGS84,
    ).to_crs(METRIC)

    served_mask = pts.distance(line) <= radius_m
    dropped = route_stops.loc[~served_mask.values]
    return {
        "stops_total": int(len(route_stops)),
        "stops_served": int(served_mask.sum()),
        "stops_dropped": int((~served_mask).sum()),
        "dropped_stop_names": dropped.get("stop_name", pd.Series()).tolist(),
    }


def compute_route_options(
    route_id: str,
    G: nx.MultiDiGraph,
    trips: pd.DataFrame,
    stop_times: pd.DataFrame,
    stops: pd.DataFrame,
    flood_prob_map: dict[tuple[Any, Any, int], float],
    risk_threshold: float,
    alphas: tuple[tuple[str, float], ...] = ALPHA_OPTIONS,
    orig_node: Any = None,
    dest_node: Any = None,
    terminals: tuple | None = None,
) -> dict | None:
    """Pareto option set for one route: the original path plus one
    alternative per alpha (deduplicated when alphas produce the same path).

    Returns None if terminals can't be resolved or no path exists.
    `orig_node`/`dest_node`/`terminals` let a caller pass in values it
    already resolved in a batch, avoiding rebuilding OSMnx's spatial index
    once per route.
    """
    try:
        if terminals is not None:
            origin, destination = terminals
        else:
            origin, destination = _get_route_terminals(
                route_id, trips, stop_times, stops
            )
        if orig_node is None:
            orig_node = ox.nearest_nodes(G, X=origin[1], Y=origin[0])
        if dest_node is None:
            dest_node = ox.nearest_nodes(G, X=destination[1], Y=destination[0])

        original_path = nx.shortest_path(G, orig_node, dest_node, weight="travel_time")
        orig = path_metrics(
            original_path, G, flood_prob_map, "travel_time", risk_threshold
        )

        route_stops = get_ordered_route_stops(route_id, trips, stop_times, stops)

        options = []
        seen_paths: set[tuple] = {tuple(original_path)}
        for label, alpha in alphas:
            weight_key = _alpha_weight_key(label)
            alt_path = nx.shortest_path(G, orig_node, dest_node, weight=weight_key)
            path_key = tuple(alt_path)
            is_original = path_key == tuple(original_path)
            if path_key in seen_paths and not is_original:
                continue  # identical to a cheaper option already recorded
            alt = path_metrics(
                alt_path, G, flood_prob_map, weight_key, risk_threshold
            )
            coverage = stop_coverage(alt_path, G, route_stops)
            options.append(
                {
                    "option": label,
                    "alpha": alpha,
                    "path": alt_path,
                    "time_s": alt["time_s"],
                    "extra_time_min": round((alt["time_s"] - orig["time_s"]) / 60, 1),
                    "flood_prob": round(alt["flood_prob"], 3),
                    "max_flood_prob": round(alt["max_flood_prob"], 3),
                    "risk_time_frac": round(alt["risk_time_frac"], 3),
                    "risk_reduction": round(orig["flood_prob"] - alt["flood_prob"], 3),
                    "same_as_original": is_original,
                    **coverage,
                }
            )
            seen_paths.add(path_key)

        return {
            "route_id": route_id,
            "origin": origin[2],
            "destination": destination[2],
            "original_path": original_path,
            "original_time_s": orig["time_s"],
            "original_flood_prob": round(orig["flood_prob"], 3),
            "original_max_flood_prob": round(orig["max_flood_prob"], 3),
            "original_risk_time_frac": round(orig["risk_time_frac"], 3),
            "options": options,
        }
    except Exception:
        return None


def find_stop_preserving_route(
    route_id: str,
    G: nx.MultiDiGraph,
    trips: pd.DataFrame,
    stop_times: pd.DataFrame,
    stops: pd.DataFrame,
    wards_gdf: gpd.GeoDataFrame,
    flood_prob_map: dict[tuple[Any, Any, int], float],
    risk_threshold: float,
    option: str = "safest",
    max_waypoints: int = 8,
) -> dict | None:
    """Alternative that still visits the route's safe intermediate stops.

    Stops inside high-risk wards (flood_prob >= risk_threshold) are removed;
    the remaining ordered stops are subsampled to at most ``max_waypoints``
    and chained with flood-weighted Dijkstra between consecutive waypoints.
    Costs one Dijkstra per segment, so callers should invoke this on demand
    for a selected route rather than for every affected route.
    """
    route_stops = get_ordered_route_stops(route_id, trips, stop_times, stops)
    if route_stops.empty:
        return None

    stops_gdf = gpd.GeoDataFrame(
        route_stops,
        geometry=gpd.points_from_xy(route_stops["stop_lon"], route_stops["stop_lat"]),
        crs=WGS84,
    )
    high_risk = wards_gdf[wards_gdf["flood_prob"] >= risk_threshold][["geometry"]]
    if high_risk.empty:
        safe = route_stops
    else:
        joined = gpd.sjoin(stops_gdf, high_risk, how="left", predicate="within")
        # sjoin can duplicate rows when geometries overlap; collapse by index
        risky_idx = joined[joined["index_right"].notna()].index.unique()
        safe = route_stops.loc[~route_stops.index.isin(risky_idx)]

    if len(safe) < 2:
        return None  # not even two safe endpoints to connect

    dropped_unsafe = route_stops.loc[~route_stops.index.isin(safe.index)]

    # Always keep both terminals; subsample interior stops evenly.
    interior = safe.iloc[1:-1]
    if len(interior) > max_waypoints - 2:
        pick = np.linspace(0, len(interior) - 1, max_waypoints - 2).round().astype(int)
        interior = interior.iloc[sorted(set(pick))]
    waypoints = pd.concat([safe.iloc[[0]], interior, safe.iloc[[-1]]])

    nodes = ox.nearest_nodes(
        G, X=waypoints["stop_lon"].tolist(), Y=waypoints["stop_lat"].tolist()
    )
    weight_key = _alpha_weight_key(option)

    full_path: list = []
    try:
        for a, b in zip(nodes[:-1], nodes[1:]):
            if a == b:
                continue
            seg = nx.shortest_path(G, a, b, weight=weight_key)
            full_path.extend(seg if not full_path else seg[1:])
    except nx.NetworkXNoPath:
        return None
    if len(full_path) < 2:
        return None

    metrics = path_metrics(full_path, G, flood_prob_map, weight_key, risk_threshold)
    coverage = stop_coverage(full_path, G, route_stops)
    return {
        "route_id": route_id,
        "option": f"stop_preserving_{option}",
        "path": full_path,
        "waypoint_stop_names": waypoints.get("stop_name", pd.Series()).tolist(),
        "unsafe_stops_skipped": dropped_unsafe.get("stop_name", pd.Series()).tolist(),
        "time_s": metrics["time_s"],
        "flood_prob": round(metrics["flood_prob"], 3),
        "max_flood_prob": round(metrics["max_flood_prob"], 3),
        "risk_time_frac": round(metrics["risk_time_frac"], 3),
        **coverage,
    }


def compute_affected_routes(
    wards_gdf: gpd.GeoDataFrame,
    stops_gdf: gpd.GeoDataFrame,
    stop_times: pd.DataFrame,
    trips: pd.DataFrame,
    threshold: float,
) -> tuple[list[str], set[str]]:
    """Which routes serve a stop inside a high-risk ward, given current
    flood_prob and threshold."""
    high_risk_wards = wards_gdf[wards_gdf["flood_prob"] >= threshold]
    if high_risk_wards.empty:
        return [], set()
    stops_joined = gpd.sjoin(
        stops_gdf,
        high_risk_wards[["ward", "flood_prob", "geometry"]],
        how="left",
        predicate="within",
    )
    affected_stops = set(
        stops_joined[stops_joined["flood_prob"].notna()]["stop_id"].tolist()
    )

    affected_trip_ids = stop_times[stop_times["stop_id"].isin(affected_stops)][
        "trip_id"
    ].unique()
    affected_route_ids = (
        trips[trips["trip_id"].isin(affected_trip_ids)]["route_id"].unique().tolist()
    )
    return affected_route_ids, affected_stops


OPTION_ROW_COLS = [
    "route_id",
    "origin",
    "destination",
    "option",
    "alpha",
    "original_flood_prob",
    "original_max_flood_prob",
    "original_risk_time_frac",
    "alternative_flood_prob",
    "alternative_max_flood_prob",
    "alternative_risk_time_frac",
    "risk_reduction",
    "original_time_s",
    "alternative_time_s",
    "extra_time_min",
    "stops_total",
    "stops_served",
    "stops_dropped",
    "same_as_original",
]


def run_live_rerouting(
    G: nx.MultiDiGraph,
    wards_gdf: gpd.GeoDataFrame,
    stops_df: pd.DataFrame,
    stop_times: pd.DataFrame,
    trips: pd.DataFrame,
    threshold: float = 0.45,
    alphas: tuple[tuple[str, float], ...] = ALPHA_OPTIONS,
) -> tuple[pd.DataFrame, dict, dict]:
    """
    Full pipeline: current ward flood_prob -> flood-weighted graph ->
    affected routes -> Pareto option set per route.

    Returns ``(options_df, route_geometries, meta)`` where ``options_df`` has
    one row per (route, option) with exposure-weighted risk metrics and stop
    coverage, and ``route_geometries`` maps route_id -> {"original": coords,
    "<option>": coords, ...}.
    """
    stops_gdf = gpd.GeoDataFrame(
        stops_df,
        geometry=gpd.points_from_xy(stops_df["stop_lon"], stops_df["stop_lat"]),
        crs=WGS84,
    )

    flood_prob_map = compute_edge_flood_map(G, wards_gdf)
    G_weighted = build_flood_weighted_graph(G, flood_prob_map, alphas)

    affected_route_ids, affected_stops = compute_affected_routes(
        wards_gdf, stops_gdf, stop_times, trips, threshold
    )

    # Resolve every affected route's terminals up front, then look up
    # nearest road-network nodes for all of them in a single batched call.
    # ox.nearest_nodes rebuilds a spatial index over the graph's ~87k nodes
    # on every call, so doing this once instead of twice per route is the
    # difference between one index build and ~2N of them.
    terminals_by_route: dict[str, tuple] = {}
    for route_id in affected_route_ids:
        try:
            terminals_by_route[route_id] = _get_route_terminals(
                route_id, trips, stop_times, stops_df
            )
        except Exception:
            continue  # this route is skipped below

    node_lookup: dict[str, tuple] = {}
    if terminals_by_route:
        route_order = list(terminals_by_route)
        lats, lons = [], []
        for route_id in route_order:
            origin, destination = terminals_by_route[route_id]
            lats += [origin[0], destination[0]]
            lons += [origin[1], destination[1]]
        try:
            nearest = ox.nearest_nodes(G_weighted, X=lons, Y=lats)
            for i, route_id in enumerate(route_order):
                node_lookup[route_id] = (nearest[2 * i], nearest[2 * i + 1])
        except Exception:
            # A single malformed coordinate can fail a batched lookup outright.
            # Fall back to resolving one route at a time so that one bad
            # coordinate only drops that route, not every affected route.
            for route_id in route_order:
                origin, destination = terminals_by_route[route_id]
                try:
                    node_lookup[route_id] = (
                        ox.nearest_nodes(G_weighted, X=origin[1], Y=origin[0]),
                        ox.nearest_nodes(
                            G_weighted, X=destination[1], Y=destination[0]
                        ),
                    )
                except Exception:
                    pass  # this route is skipped below

    rows = []
    route_geometries: dict[str, dict[str, list]] = {}
    for route_id in affected_route_ids:
        if route_id not in node_lookup:
            continue
        orig_node, dest_node = node_lookup[route_id]
        result = compute_route_options(
            route_id,
            G_weighted,
            trips,
            stop_times,
            stops_df,
            flood_prob_map,
            risk_threshold=threshold,
            alphas=alphas,
            orig_node=orig_node,
            dest_node=dest_node,
            terminals=terminals_by_route[route_id],
        )
        if not result:
            continue

        geoms = {"original": _path_to_coords(result["original_path"], G_weighted)}
        for opt in result["options"]:
            geoms[opt["option"]] = _path_to_coords(opt["path"], G_weighted)
            rows.append(
                {
                    "route_id": result["route_id"],
                    "origin": result["origin"],
                    "destination": result["destination"],
                    "option": opt["option"],
                    "alpha": opt["alpha"],
                    "original_flood_prob": result["original_flood_prob"],
                    "original_max_flood_prob": result["original_max_flood_prob"],
                    "original_risk_time_frac": result["original_risk_time_frac"],
                    "alternative_flood_prob": opt["flood_prob"],
                    "alternative_max_flood_prob": opt["max_flood_prob"],
                    "alternative_risk_time_frac": opt["risk_time_frac"],
                    "risk_reduction": opt["risk_reduction"],
                    "original_time_s": result["original_time_s"],
                    "alternative_time_s": opt["time_s"],
                    "extra_time_min": opt["extra_time_min"],
                    "stops_total": opt["stops_total"],
                    "stops_served": opt["stops_served"],
                    "stops_dropped": opt["stops_dropped"],
                    "same_as_original": opt["same_as_original"],
                }
            )
        route_geometries[str(route_id)] = geoms

    options_df = pd.DataFrame(rows, columns=OPTION_ROW_COLS)

    meta = {
        "alphas": {label: alpha for label, alpha in alphas},
        "threshold": threshold,
        "total_affected_routes": len(affected_route_ids),
        "rerouted_routes": int(options_df["route_id"].nunique())
        if not options_df.empty
        else 0,
        "affected_stops": len(affected_stops),
        "service_radius_m": SERVICE_RADIUS_M,
    }

    return options_df, route_geometries, meta


def select_option(options_df: pd.DataFrame, preference: str) -> pd.DataFrame:
    """One row per route for a given option label, falling back to the
    closest available option when an alpha was deduplicated away.

    Fallback order: requested option, then progressively safer options,
    then progressively faster ones.
    """
    if options_df.empty:
        return options_df
    labels = [label for label, _ in ALPHA_OPTIONS]
    if preference not in labels:
        raise ValueError(f"preference must be one of {labels}; got {preference!r}")
    idx = labels.index(preference)
    priority = {
        label: rank
        for rank, label in enumerate(labels[idx:] + labels[:idx][::-1])
    }
    ranked = options_df.assign(_rank=options_df["option"].map(priority))
    picked = (
        ranked.sort_values(["route_id", "_rank"])
        .groupby("route_id", as_index=False)
        .first()
        .drop(columns="_rank")
    )
    return picked
