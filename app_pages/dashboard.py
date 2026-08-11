"""Flood Risk Dashboard page."""

import geopandas as gpd
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium

from app_lib.data import load_gtfs, load_precomputed_reroutes
from app_lib.maps import build_choropleth
from app_lib.state import get_state
from app_lib.theme import PLOTLY_LAYOUT
from Utils.kmd_fetcher import fetch_kmd_advisory
from Utils.live_routing import compute_affected_routes, select_option
from Utils.rainfall_fetcher import rainfall_summary

state = get_state()
df = state["df"]
nairobi = state["nairobi"]
threshold = state["threshold"]

# -- Headline impact metrics (10-second read) --------------------------------
high_risk_nairobi = nairobi[nairobi["flood_prob"] >= threshold]

_routes_df, trips_df, _shapes_df, stops_df, stop_times_df = load_gtfs()
stops_gdf = gpd.GeoDataFrame(
    stops_df,
    geometry=gpd.points_from_xy(stops_df["stop_lon"], stops_df["stop_lat"]),
    crs="EPSG:4326",
)
affected_route_ids, affected_stops = compute_affected_routes(
    nairobi, stops_gdf, stop_times_df, trips_df, threshold
)

# Rerouting impact quoted from the latest scheduled refresh (the dashboard
# never pays the graph-load cost itself).
pre = load_precomputed_reroutes()
avg_reduction = stops_served_txt = None
pre_threshold = None
if pre:
    pre_options = pd.DataFrame(pre["options"])
    pre_threshold = pre["threshold"]
    if not pre_options.empty:
        balanced = select_option(pre_options, "balanced")
        avg_reduction = balanced["risk_reduction"].mean()
        if balanced["stops_served"].notna().any():
            served_pct = (
                balanced["stops_served"].sum() / balanced["stops_total"].sum()
            )
            stops_served_txt = f"{served_pct:.0%}"

h1, h2, h3, h4 = st.columns(4)
with h1:
    st.metric(
        "People in High-Risk Wards",
        f"{int(high_risk_nairobi['pop2009'].sum()):,}",
        help="Nairobi wards at or above the current threshold (2009 census).",
    )
with h2:
    st.metric(
        "Matatu Routes Affected",
        len(affected_route_ids),
        help="Routes serving a stop inside a currently high-risk Nairobi ward.",
    )
with h3:
    st.metric(
        "Avg Risk Reduction",
        f"{avg_reduction:.3f}" if avg_reduction is not None else "—",
        help=(
            f"Balanced option, latest scheduled refresh (threshold "
            f"{pre_threshold:.2f})."
            if pre_threshold is not None
            else "Run `make refresh-cache` to populate."
        ),
    )
with h4:
    st.metric(
        "Stops Still Served",
        stops_served_txt or "—",
        help=(
            "Share of original stops within 300 m of the balanced detour "
            "(latest scheduled refresh)."
            if stops_served_txt
            else "Run `make refresh-cache` to populate."
        ),
    )

if state["use_open_meteo"]:
    mode_label = (
        "Live mode"
        if state["use_live"]
        else f"{state['forecast_horizon_hours']}hr prediction mode"
    )
    st.info(
        f"**{mode_label}**: predictions use rainfall features from "
        f"{rainfall_summary(state['rainfall_meta'])}. "
        "Terrain features remain static (SRTM). "
        "Switch rainfall source in the sidebar to compare live, historical, "
        "24hr, and 48hr flood-risk maps."
    )

# -- Official KMD advisory (complement, not compete) -------------------------

kmd = fetch_kmd_advisory()
with st.expander("Official KMD Advisory (Kenya Meteorological Department)", expanded=False):
    st.markdown(
        f"**{kmd['title']}**  \n"
        f"{kmd['summary']}  \n"
        f"Source: {kmd['source']} · fetched {kmd['fetched_at'][:16].replace('T', ' ')} UTC"
    )
    st.markdown(
        f"- [Weather warnings]({kmd['urls']['weather_warnings']})  \n"
        f"- [Daily Flood Bulletin]({kmd['urls']['flood_bulletin']})"
    )
    st.caption(
        "Nairobi Flood Guard complements KMD's national forecasting — it does "
        "not replace official advisories. When KMD publishes a machine-readable "
        "CAP feed at a stable URL, this panel will ingest it automatically."
    )

st.markdown(
    '<div class="section-header">County Flood Risk Map</div>',
    unsafe_allow_html=True,
)

counties = sorted(df["county"].unique())
default_idx = counties.index("Nairobi") if "Nairobi" in counties else 0
selected_county = st.selectbox("Filter by county", counties, index=default_idx)

map_df = df[df["county"] == selected_county]

st.caption(f"{len(map_df)} wards · hover a ward for details")

centre_lat = float(map_df.geometry.centroid.y.mean())
centre_lon = float(map_df.geometry.centroid.x.mean())
zoom = 7 if selected_county == "All Kenya" else 10

with st.spinner("Rendering map..."):
    fmap = build_choropleth(map_df, centre_lat, centre_lon, zoom)
st_folium(fmap, width="stretch", height=520)

st.markdown(
    '<div class="section-header" style="margin-top:2rem">Flood Probability Distribution</div>',
    unsafe_allow_html=True,
)
fig = px.histogram(
    map_df,
    x="flood_prob",
    nbins=40,
    color_discrete_sequence=["#3FA66B"],
    labels={"flood_prob": "Flood Probability", "count": "Number of Wards"},
)
fig.add_vline(
    x=threshold,
    line_dash="dash",
    line_color="#C4622D",
    annotation_text=f"Threshold ({threshold:.2f})",
    annotation_font_color="#C4622D",
    annotation_position="top right",
)
fig.update_layout(
    **PLOTLY_LAYOUT,
    margin=dict(t=20, b=20, l=20, r=20),
    xaxis=dict(gridcolor="#1F4A32", tickformat=".0%"),
    yaxis=dict(gridcolor="#1F4A32"),
    bargap=0.05,
)
st.plotly_chart(fig, width="stretch")

st.markdown(
    '<div class="section-header">Highest Risk Wards</div>',
    unsafe_allow_html=True,
)
top10 = (
    map_df[["ward", "subcounty", "county", "flood_prob", "risk_label"]]
    .sort_values("flood_prob", ascending=False)
    .head(10)
    .reset_index(drop=True)
)
top10["flood_prob"] = top10["flood_prob"].map("{:.1%}".format)
top10.index += 1
st.dataframe(
    top10.rename(
        columns={
            "ward": "Ward",
            "subcounty": "Sub-County",
            "county": "County",
            "flood_prob": "Flood Probability",
            "risk_label": "Risk Level",
        }
    ),
    width="stretch",
)
