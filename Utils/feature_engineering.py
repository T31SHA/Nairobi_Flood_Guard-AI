import numpy as np
import geopandas as gpd

METRIC = "EPSG:32737"
LOW_ELEV = 1_620.0

FEATS = [
    "mean_elevation",
    "min_elevation",
    "slope_mean_deg",
    "twi_proxy",
    "elev_range",
    "terrain_roughness",
    "is_low_lying",
    "precip_mm",
    "rain_cum_1d",
    "rain_cum_2d",
    "rain_cum_3d",
    "rain_cum_6d",
    "rain_mean_3d",
    "rain_mean_7d",
    "rain_mean_14d",
    "rain_mean_30d",
    "rain_std_3d",
    "rain_std_7d",
    "rain_max_3d",
    "rain_max_7d",
    "rain_max_14d",
    "soil_moisture",
    "is_heavy_rain",
    "is_extreme_rain",
    "doy_sin",
    "doy_cos",
    "mon_sin",
    "mon_cos",
    "is_long_rain",
    "is_short_rain",
    "is_rainy",
    "month",
    "n_routes",
    "n_stops",
    "route_density",
    "stop_density",
    "is_terminal",
    "route_vuln",
    "route_vuln_n",
    "exp_disruption",
    "rain_low_elev",
    "rain_route_risk",
    "moisture_rain",
    "twi_rain",
    "compound_risk",
    "ward_hist_rate",
    "max_elevation",
]


def engineer_features(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Replicates the feature engineering pipeline from random_forest_notebook.ipynb.
    Accepts a GeoDataFrame of ward-level data and returns it with all 47
    engineered features appended, ready for prediction.
    """
    df = df.copy()

    # --- Terrain ---
    metric_geom = df.to_crs(METRIC)
    df["ward_area_km2"] = metric_geom.geometry.area / 1e6
    slope_rad = np.radians(df["slope_mean_deg"].clip(lower=0.1))
    df["twi_proxy"] = np.log(df["ward_area_km2"] * 1e6 / (np.tan(slope_rad) + 1e-6))
    df["elev_range"] = df["elevation_max_m"] - df["elevation_min_m"]
    df["terrain_roughness"] = df["elev_range"] / (df["elevation_mean_m"] + 1)
    df["is_low_lying"] = (df["elevation_min_m"] < LOW_ELEV).astype(int)
    df["mean_elevation"] = df["elevation_mean_m"]
    df["min_elevation"] = df["elevation_min_m"]
    df["max_elevation"] = df["elevation_max_m"]

    # --- Rainfall proxies ---
    d = df["rain_max_daily_mm"]
    r7 = df["rain_preflood_7d_mm"]
    rt = df["rain_cumulative_mm"]

    df["precip_mm"] = d
    df["rain_cum_1d"] = d
    df["rain_max_3d"] = d
    df["rain_cum_2d"] = r7 * (2 / 7)
    df["rain_cum_3d"] = r7 * (3 / 7)
    df["rain_cum_6d"] = r7 * (6 / 7)
    df["rain_mean_3d"] = df["rain_cum_3d"] / 3
    df["rain_mean_7d"] = r7 / 7
    df["rain_mean_14d"] = rt / 14
    df["rain_mean_30d"] = rt / 30
    df["rain_std_3d"] = df["rain_cum_3d"] * 0.25
    df["rain_std_7d"] = r7 * 0.25
    df["rain_max_7d"] = d * 1.10
    df["rain_max_14d"] = d * 1.20
    df["soil_moisture"] = r7 * 0.15
    df["is_heavy_rain"] = (d > 25).astype(int)
    df["is_extreme_rain"] = (d > 50).astype(int)

    # --- Transport proxies ---
    pop = df["pop2009"].clip(lower=1)
    df["n_routes"] = (pop / 5_000).clip(1, 50).round().astype(int)
    df["n_stops"] = (pop / 1_500).clip(2, 150).round().astype(int)
    df["route_density"] = df["n_routes"] / (df["ward_area_km2"] + 1e-6)
    df["stop_density"] = df["n_stops"] / (df["ward_area_km2"] + 1e-6)
    df["is_terminal"] = 0

    # --- Seasonality (fixed to April 26 flood date) ---
    doy, month = 117, 4
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365)
    df["mon_sin"] = np.sin(2 * np.pi * month / 12)
    df["mon_cos"] = np.cos(2 * np.pi * month / 12)
    df["is_long_rain"] = 1
    df["is_short_rain"] = 0
    df["is_rainy"] = 1
    df["month"] = month

    # --- Interaction features ---
    def minmax(s):
        return (s - s.min()) / (s.max() - s.min() + 1e-9)

    elev_pen = np.log1p((df["mean_elevation"] - LOW_ELEV).clip(lower=0)) + 1
    df["route_vuln"] = (df["route_density"] * (1 + df["is_terminal"])) / elev_pen
    df["route_vuln_n"] = minmax(df["route_vuln"])
    df["exp_disruption"] = df["route_vuln"] * df["precip_mm"]
    df["rain_low_elev"] = df["rain_cum_3d"] * df["is_low_lying"]
    df["rain_route_risk"] = df["rain_cum_3d"] * df["route_vuln_n"]
    df["moisture_rain"] = df["soil_moisture"] * df["precip_mm"]
    df["twi_rain"] = df["twi_proxy"] * df["rain_cum_6d"]
    df["compound_risk"] = (
        0.35 * minmax(df["rain_cum_3d"])
        + 0.25 * df["is_low_lying"]
        + 0.20 * minmax(df["soil_moisture"])
        + 0.10 * minmax(df["twi_proxy"])
        + 0.10 * df["route_vuln_n"]
    )

    # --- Historical flood rate ---
    df["ward_hist_rate"] = df["flooded"]

    return df
