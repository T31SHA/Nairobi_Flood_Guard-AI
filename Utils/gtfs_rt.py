"""GTFS-Realtime feed generation for flood rerouting.

Extracted from Route_Optimization/route_optimization.ipynb so the feed is
producible from running code (API + app), not only from a notebook cell.

Semantics: every trip of an affected route gets a TripUpdate flagged
``ADDED`` (the rerouted alternative service), whose stops inside high-risk
wards are marked ``SKIPPED`` and all others ``SCHEDULED`` - matching the
original notebook implementation.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pandas as pd

try:
    from google.transit import gtfs_realtime_pb2

    GTFS_RT_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only on minimal installs
    GTFS_RT_AVAILABLE = False

if TYPE_CHECKING:
    from google.transit import gtfs_realtime_pb2 as _pb


def build_gtfs_rt_feed_message(
    options_df: pd.DataFrame,
    trips: pd.DataFrame,
    stop_times: pd.DataFrame,
    affected_stops: set | frozenset | list,
    timestamp: int | None = None,
) -> _pb.FeedMessage:
    """Build a GTFS-RT FeedMessage from the Pareto rerouting option set.

    ``options_df`` follows ``Utils.live_routing.OPTION_ROW_COLS`` (one row per
    (route, option)); only the distinct route IDs are used. ``affected_stops``
    is the set of stop_ids inside currently high-risk wards.
    """
    if not GTFS_RT_AVAILABLE:
        raise ImportError(
            "gtfs-realtime-bindings is not installed. "
            "Run: pip install gtfs-realtime-bindings"
        )

    feed = gtfs_realtime_pb2.FeedMessage()
    feed.header.gtfs_realtime_version = "2.0"
    feed.header.incrementality = gtfs_realtime_pb2.FeedHeader.FULL_DATASET
    feed.header.timestamp = int(timestamp if timestamp is not None else time.time())

    affected_stops = set(affected_stops)
    route_ids = (
        options_df["route_id"].unique().tolist() if not options_df.empty else []
    )
    trips_by_route = dict(tuple(trips.groupby("route_id")))
    stop_times_by_trip = dict(tuple(stop_times.groupby("trip_id")))

    for route_id in route_ids:
        route_trips = trips_by_route.get(route_id)
        if route_trips is None:
            continue
        for trip_id in route_trips["trip_id"].tolist():
            entity = feed.entity.add()
            entity.id = f"reroute_{trip_id}"

            entity.trip_update.trip.trip_id = str(trip_id)
            entity.trip_update.trip.route_id = str(route_id)
            entity.trip_update.trip.schedule_relationship = (
                gtfs_realtime_pb2.TripDescriptor.ADDED  # new alternative trip
            )

            trip_stop_times = stop_times_by_trip.get(trip_id)
            if trip_stop_times is None:
                continue
            for st_row in trip_stop_times.sort_values("stop_sequence").itertuples():
                stop_update = entity.trip_update.stop_time_update.add()
                stop_update.stop_sequence = int(st_row.stop_sequence)
                stop_update.stop_id = str(st_row.stop_id)
                stop_update.schedule_relationship = (
                    gtfs_realtime_pb2.TripUpdate.StopTimeUpdate.SKIPPED
                    if st_row.stop_id in affected_stops
                    else gtfs_realtime_pb2.TripUpdate.StopTimeUpdate.SCHEDULED
                )

    return feed


def build_gtfs_rt_feed(
    options_df: pd.DataFrame,
    trips: pd.DataFrame,
    stop_times: pd.DataFrame,
    affected_stops: set | frozenset | list,
    timestamp: int | None = None,
) -> bytes:
    """Serialized GTFS-RT feed (protobuf bytes), ready to serve or download."""
    return build_gtfs_rt_feed_message(
        options_df, trips, stop_times, affected_stops, timestamp
    ).SerializeToString()
