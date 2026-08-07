"""Routing engine tests on a small synthetic road network."""

import networkx as nx
import pandas as pd
import pytest

from Utils.live_routing import (
    _alpha_weight_key,
    build_flood_weighted_graph,
    path_metrics,
    select_option,
    stop_coverage,
)


@pytest.fixture
def graph_and_probs():
    """A -> B -> C direct (fast, flooded mid-edge) vs A -> D -> C (slow, dry),
    plus a parallel B->C edge that is slower but drier."""
    G = nx.MultiDiGraph()
    coords = {
        "A": (36.80, -1.30),
        "B": (36.82, -1.30),
        "C": (36.84, -1.30),
        "D": (36.82, -1.32),
    }
    for n, (x, y) in coords.items():
        G.add_node(n, x=x, y=y)
    G.add_edge("A", "B", key=0, travel_time=100.0)
    G.add_edge("B", "C", key=0, travel_time=100.0)
    G.add_edge("B", "C", key=1, travel_time=150.0)  # parallel, drier
    G.add_edge("A", "D", key=0, travel_time=300.0)
    G.add_edge("D", "C", key=0, travel_time=300.0)
    probs = {
        ("A", "B", 0): 0.0,
        ("B", "C", 0): 0.8,
        ("B", "C", 1): 0.1,
        ("A", "D", 0): 0.0,
        ("D", "C", 0): 0.0,
    }
    return build_flood_weighted_graph(G, probs), probs


def test_exposure_weighted_metrics(graph_and_probs):
    G, probs = graph_and_probs
    m = path_metrics(["A", "B", "C"], G, probs, "travel_time", risk_threshold=0.45)
    assert m["time_s"] == 200.0
    # 100s dry + 100s at prob 0.8 -> time-weighted mean 0.4
    assert m["flood_prob"] == pytest.approx(0.4)
    assert m["max_flood_prob"] == pytest.approx(0.8)
    assert m["risk_time_frac"] == pytest.approx(0.5)


def test_safest_alpha_blocks_flooded_edges(graph_and_probs):
    G, _ = graph_and_probs
    path = nx.shortest_path(G, "A", "C", weight=_alpha_weight_key("safest"))
    assert path == ["A", "D", "C"]


def test_mild_alpha_keeps_short_path_via_drier_parallel_edge(graph_and_probs):
    G, probs = graph_and_probs
    path = nx.shortest_path(G, "A", "C", weight=_alpha_weight_key("fastest"))
    assert path == ["A", "B", "C"]
    # Metrics under the fastest weight key must attribute the drier parallel
    # edge (key 1) that the router actually used, not silently assume key 0.
    m = path_metrics(path, G, probs, _alpha_weight_key("fastest"), 0.45)
    assert m["time_s"] == pytest.approx(250.0)
    assert m["max_flood_prob"] == pytest.approx(0.1)


def test_risk_never_increases_with_alpha(graph_and_probs):
    G, probs = graph_and_probs
    risks = []
    for label in ("fastest", "balanced", "safest"):
        key = _alpha_weight_key(label)
        path = nx.shortest_path(G, "A", "C", weight=key)
        risks.append(path_metrics(path, G, probs, key, 0.45)["flood_prob"])
    assert risks == sorted(risks, reverse=True)


def test_stop_coverage(graph_and_probs):
    G, _ = graph_and_probs
    route_stops = pd.DataFrame(
        {
            "stop_id": ["s1", "s2"],
            "stop_name": ["NearD", "NearB"],
            "stop_lon": [36.82, 36.82],
            "stop_lat": [-1.32, -1.30],
        }
    )
    cov = stop_coverage(["A", "D", "C"], G, route_stops)
    assert cov["stops_total"] == 2
    assert cov["stops_served"] == 1
    assert cov["dropped_stop_names"] == ["NearB"]


def test_stop_coverage_degenerate_path(graph_and_probs):
    G, _ = graph_and_probs
    route_stops = pd.DataFrame(
        {
            "stop_id": ["s1"],
            "stop_name": ["X"],
            "stop_lon": [36.82],
            "stop_lat": [-1.30],
        }
    )
    cov = stop_coverage(["A"], G, route_stops)
    assert cov["stops_served"] == 0
    assert cov["stops_dropped"] == 1


def test_select_option_fallback():
    df = pd.DataFrame(
        [
            {"route_id": "r1", "option": "fastest", "risk_reduction": 0.1},
            {"route_id": "r1", "option": "safest", "risk_reduction": 0.5},
            {"route_id": "r2", "option": "safest", "risk_reduction": 0.3},
        ]
    )
    picked = select_option(df, "balanced").set_index("route_id")
    # balanced missing everywhere: prefer the next-safer option
    assert picked.loc["r1", "option"] == "safest"
    assert picked.loc["r2", "option"] == "safest"

    picked_fast = select_option(df, "fastest").set_index("route_id")
    assert picked_fast.loc["r1", "option"] == "fastest"
    assert picked_fast.loc["r2", "option"] == "safest"  # only option available


def test_select_option_invalid_preference():
    df = pd.DataFrame([{"route_id": "r1", "option": "safest"}])
    with pytest.raises(ValueError):
        select_option(df, "yolo")
