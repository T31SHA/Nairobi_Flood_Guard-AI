"""Folium map builders and GTFS shape helpers."""

from __future__ import annotations

import folium
import geopandas as gpd
import pandas as pd

from app_lib.theme import delta_color, risk_color


# Not cached since Folium maps with lambdas can't be pickled
def build_choropleth(map_df, centre_lat, centre_lon, zoom):
    fmap = folium.Map(
        location=[centre_lat, centre_lon],
        zoom_start=zoom,
        tiles="CartoDB dark_matter",
    )
    folium.GeoJson(
        map_df[["ward", "subcounty", "county", "flood_prob", "risk_label", "geometry"]],
        style_function=lambda feature: {
            "fillColor": risk_color(float(feature["properties"]["flood_prob"])),
            "fillOpacity": 0.55,
            "color": risk_color(float(feature["properties"]["flood_prob"])),
            "weight": 0.8,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["ward", "subcounty", "county", "flood_prob", "risk_label"],
            aliases=["Ward", "Sub-County", "County", "Flood Probability", "Risk Level"],
            localize=True,
            sticky=False,
        ),
    ).add_to(fmap)
    return fmap


def build_delta_choropleth(map_df, centre_lat, centre_lon, zoom):
    fmap = folium.Map(
        location=[centre_lat, centre_lon],
        zoom_start=zoom,
        tiles="CartoDB dark_matter",
    )
    folium.GeoJson(
        map_df[
            ["ward", "subcounty", "historical_prob", "live_prob", "delta", "geometry"]
        ],
        style_function=lambda feature: {
            "fillColor": delta_color(float(feature["properties"]["delta"])),
            "fillOpacity": 0.65,
            "color": delta_color(float(feature["properties"]["delta"])),
            "weight": 0.8,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["ward", "subcounty", "historical_prob", "live_prob", "delta"],
            aliases=[
                "Ward",
                "Sub-County",
                "Historical",
                "Live",
                "Δ (Live − Historical)",
            ],
            localize=True,
            sticky=False,
        ),
    ).add_to(fmap)
    return fmap


def get_route_shape(route_id, trips, shapes):
    """Return list of [lat, lon] for the first trip of a route."""
    trip_rows = trips[trips["route_id"] == route_id]
    if trip_rows.empty:
        return []
    shape_id = trip_rows.iloc[0]["shape_id"]
    pts = shapes[shapes["shape_id"] == shape_id].sort_values("shape_pt_sequence")
    return [[row["shape_pt_lat"], row["shape_pt_lon"]] for _, row in pts.iterrows()]


def get_route_stops(route_id, trips, stop_times, stops):
    """Return DataFrame of stops for the first trip of a route."""
    trip_rows = trips[trips["route_id"] == route_id]
    if trip_rows.empty:
        return pd.DataFrame()
    trip_id = trip_rows.iloc[0]["trip_id"]
    return (
        stop_times[stop_times["trip_id"] == trip_id]
        .sort_values("stop_sequence")
        .merge(stops, on="stop_id")
    )


def get_affected_stop_ids(nairobi_df, stops_df, flood_threshold):
    """Return set of stop_ids falling inside high-risk Nairobi wards."""
    high_risk = nairobi_df[nairobi_df["flood_prob"] >= flood_threshold][
        ["geometry"]
    ].copy()
    stops_gdf = gpd.GeoDataFrame(
        stops_df,
        geometry=gpd.points_from_xy(stops_df["stop_lon"], stops_df["stop_lat"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(stops_gdf, high_risk, how="inner", predicate="within")
    return set(joined["stop_id"].tolist())
