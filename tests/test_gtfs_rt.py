"""GTFS-RT feed tests on a small synthetic GTFS fixture (no real feed needed)."""

import pandas as pd
import pytest
from google.transit import gtfs_realtime_pb2

from Utils.gtfs_rt import build_gtfs_rt_feed

ROUTE_1, ROUTE_2 = "R1", "R2"


@pytest.fixture
def gtfs_fixture():
    """Two routes: R1 has two trips, R2 one. R1/T1 stops s1-s2-s3."""
    trips = pd.DataFrame(
        {
            "route_id": [ROUTE_1, ROUTE_1, ROUTE_2],
            "trip_id": ["T1", "T2", "T3"],
        }
    )
    stop_times = pd.DataFrame(
        {
            "trip_id": ["T1", "T1", "T1", "T2", "T2", "T3"],
            "stop_id": ["s1", "s2", "s3", "s1", "s4", "s5"],
            "stop_sequence": [3, 1, 2, 1, 2, 1],
        }
    )
    options_df = pd.DataFrame(
        [
            {"route_id": ROUTE_1, "option": "fastest"},
            {"route_id": ROUTE_1, "option": "safest"},  # same route, two options
        ]
    )
    return options_df, trips, stop_times


def test_feed_parses_and_covers_each_trip_once(gtfs_fixture):
    options_df, trips, stop_times = gtfs_fixture
    blob = build_gtfs_rt_feed(options_df, trips, stop_times, affected_stops=set())

    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(blob)

    assert feed.header.gtfs_realtime_version == "2.0"
    assert feed.header.incrementality == gtfs_realtime_pb2.FeedHeader.FULL_DATASET
    assert feed.header.timestamp > 0
    # two trips on R1, emitted once each despite two option rows
    assert sorted(e.trip_update.trip.trip_id for e in feed.entity) == ["T1", "T2"]
    assert all(
        e.trip_update.trip.schedule_relationship
        == gtfs_realtime_pb2.TripDescriptor.ADDED
        for e in feed.entity
    )


def test_affected_stops_marked_skipped_in_stop_order(gtfs_fixture):
    options_df, trips, stop_times = gtfs_fixture
    blob = build_gtfs_rt_feed(
        options_df, trips, stop_times, affected_stops={"s2"}, timestamp=1_700_000_000
    )

    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(blob)
    assert feed.header.timestamp == 1_700_000_000

    t1 = next(e for e in feed.entity if e.trip_update.trip.trip_id == "T1")
    updates = list(t1.trip_update.stop_time_update)
    # stops are emitted in stop_sequence order: s2 (seq 1), s3 (seq 2), s1 (seq 3)
    assert [u.stop_id for u in updates] == ["s2", "s3", "s1"]
    assert [u.stop_sequence for u in updates] == [1, 2, 3]
    rel = gtfs_realtime_pb2.TripUpdate.StopTimeUpdate
    assert updates[0].schedule_relationship == rel.SKIPPED
    assert updates[1].schedule_relationship == rel.SCHEDULED
    assert updates[2].schedule_relationship == rel.SCHEDULED


def test_empty_options_produce_header_only_feed(gtfs_fixture):
    _, trips, stop_times = gtfs_fixture
    blob = build_gtfs_rt_feed(
        pd.DataFrame(columns=["route_id", "option"]), trips, stop_times, {"s1"}
    )
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(blob)
    assert len(feed.entity) == 0


def test_unknown_route_is_skipped(gtfs_fixture):
    _, trips, stop_times = gtfs_fixture
    options_df = pd.DataFrame([{"route_id": "NOPE", "option": "safest"}])
    blob = build_gtfs_rt_feed(options_df, trips, stop_times, set())
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(blob)
    assert len(feed.entity) == 0
