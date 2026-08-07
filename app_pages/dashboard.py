"""Flood Risk Dashboard page."""

import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium

from app_lib.maps import build_choropleth
from app_lib.state import get_state
from app_lib.theme import PLOTLY_LAYOUT
from Utils.rainfall_fetcher import rainfall_summary

state = get_state()
df = state["df"]
threshold = state["threshold"]

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
