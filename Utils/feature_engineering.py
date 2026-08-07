"""
Leakage-free, ward-level feature engineering for the flood susceptibility model.

Every feature below is a deterministic, stateless transform of *measured*
inputs (SRTM terrain, CHIRPS/Open-Meteo rainfall, census population). This
module intentionally does NOT contain:

- ``ward_hist_rate``: previous versions set this directly from the ``flooded``
  label, which is target leakage. A genuine historical flood frequency can be
  reintroduced once labels from more than one flood event are available, and
  must then be computed strictly from *past* events relative to the event
  being predicted.
- Fabricated rainfall proxies (e.g. ``rain_max_7d = rain_max_daily * 1.10``,
  ``soil_moisture = rain_7d * 0.15``): deterministic rescalings of existing
  columns add no information and distort feature-importance analyses.
- Population-derived pseudo transport counts (``n_routes = pop / 5000``) and
  constant columns (``is_terminal = 0``): dead or misleading weight. Real
  GTFS-derived transport exposure lives in :func:`ward_transport_exposure`
  and is used for impact reporting, not flood prediction.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd

METRIC = "EPSG:32737"  # UTM zone 37S, metres, appropriate for Kenya
EPS = 1e-9

RAW_INPUT_COLS = [
    "pop2009",
    "rain_cumulative_mm",
    "rain_max_daily_mm",
    "rain_preflood_7d_mm",
    "elevation_mean_m",
    "elevation_min_m",
    "elevation_max_m",
    "slope_mean_deg",
]

FEATS = [
    # Terrain (dominant signal per EDA)
    "elevation_mean_m",
    "elevation_min_m",
    "elevation_max_m",
    "elev_range_m",
    "terrain_roughness",
    "slope_mean_deg",
    "twi_proxy",
    # Rainfall: the three measured aggregates plus two ratio features that
    # encode temporal structure trees cannot derive on their own
    "rain_cumulative_mm",
    "rain_max_daily_mm",
    "rain_preflood_7d_mm",
    "rain_recency_ratio",
    "rain_intensity_ratio",
    # Exposure / urbanisation
    "ward_area_km2",
    "pop_density",
]


def engineer_features(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Append the model's feature columns to a ward-level GeoDataFrame.

    Requires ``RAW_INPUT_COLS`` and a geometry column. Never reads the
    ``flooded`` label, so it is safe to call at inference time on wards whose
    outcome is unknown.
    """
    missing = [c for c in RAW_INPUT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required raw input columns: {missing}")

    df = df.copy()

    # --- Terrain ---
    metric_geom = df.geometry.to_crs(METRIC)
    df["ward_area_km2"] = metric_geom.area / 1e6
    df["elev_range_m"] = df["elevation_max_m"] - df["elevation_min_m"]
    df["terrain_roughness"] = df["elev_range_m"] / (df["elevation_mean_m"] + 1)
    # Topographic wetness proxy: ln(contributing area / tan(slope)). Uses ward
    # area as the contributing-area stand-in until a flow-accumulation raster
    # is integrated.
    slope_rad = np.radians(df["slope_mean_deg"].clip(lower=0.1))
    df["twi_proxy"] = np.log(df["ward_area_km2"] * 1e6 / (np.tan(slope_rad) + 1e-6))

    # --- Rainfall temporal structure ---
    # Share of the 90-day total that fell in the final week: distinguishes a
    # sudden deluge from steadily accumulated rain at equal totals.
    df["rain_recency_ratio"] = df["rain_preflood_7d_mm"] / (
        df["rain_cumulative_mm"] + EPS
    )
    # How concentrated the wettest day was within the final week.
    df["rain_intensity_ratio"] = df["rain_max_daily_mm"] / (
        df["rain_preflood_7d_mm"] + EPS
    )

    # --- Exposure / urbanisation ---
    df["pop_density"] = df["pop2009"].clip(lower=1) / (df["ward_area_km2"] + EPS)

    return df


def ward_transport_exposure(
    wards_gdf: gpd.GeoDataFrame,
    stops_df: pd.DataFrame,
    stop_times: pd.DataFrame,
    trips: pd.DataFrame,
) -> pd.DataFrame:
    """Real GTFS-derived transport exposure per ward, for impact reporting.

    Counts the actual matatu stops located in each ward and the number of
    distinct routes serving those stops. Wards outside the GTFS coverage area
    (all of Kenya except Nairobi for the 2019 feed) get zeros. These are
    *exposure* metrics - how much transit service is at stake if a ward
    floods - and are deliberately not model features, since GTFS coverage is
    Nairobi-only while the model is trained nationwide.

    Returns a DataFrame indexed like ``wards_gdf`` with columns
    ``n_stops``, ``n_routes``, ``stop_density`` and ``route_density``.
    """
    stops_gdf = gpd.GeoDataFrame(
        stops_df[["stop_id"]],
        geometry=gpd.points_from_xy(stops_df["stop_lon"], stops_df["stop_lat"]),
        crs="EPSG:4326",
    )
    wards = wards_gdf[["geometry"]].copy()
    wards["_ward_idx"] = wards.index

    joined = gpd.sjoin(
        stops_gdf, wards.to_crs(stops_gdf.crs), how="inner", predicate="within"
    )

    stop_to_routes = (
        stop_times[["stop_id", "trip_id"]]
        .merge(trips[["trip_id", "route_id"]], on="trip_id")
        .groupby("stop_id")["route_id"]
        .agg(set)
    )

    n_stops = joined.groupby("_ward_idx")["stop_id"].nunique()
    n_routes = joined.groupby("_ward_idx")["stop_id"].apply(
        lambda ids: len(set().union(*(stop_to_routes.get(s, set()) for s in ids)))
    )

    out = pd.DataFrame(index=wards_gdf.index)
    out["n_stops"] = n_stops.reindex(out.index).fillna(0).astype(int)
    out["n_routes"] = n_routes.reindex(out.index).fillna(0).astype(int)

    area_km2 = wards_gdf.geometry.to_crs(METRIC).area / 1e6
    out["stop_density"] = out["n_stops"] / (area_km2 + EPS)
    out["route_density"] = out["n_routes"] / (area_km2 + EPS)
    return out
