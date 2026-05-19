"""
Nairobi Flood Guard - Streamlit UI
Run with: streamlit run app.py
"""

import warnings
import json
import pickle
import base64
from pathlib import Path

import pandas as pd
import geopandas as gpd
import folium
import streamlit as st
from streamlit_folium import st_folium
from groq import Groq
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")


# -- Paths --------------------------------------------------------------------
BASE = Path(__file__).parent
DATA = BASE / "Data"
MODELS = BASE / "Models"
GTFS_DIR = DATA / "GTFS_FEED_2019"
REPORTS = BASE / "Route_Optimization" / "Reports"
GROQ_MODEL = "llama-3.3-70b-versatile"

FLOODS_GPKG = DATA / "floods.gpkg"
XGB_MODEL = MODELS / "best_xgboost_model.pkl"
REROUTING_CSV = REPORTS / "rerouting_summary.csv"
TRADEOFF_PNG = REPORTS / "rerouting_tradeoff.png"
ROUTE_GEOMETRIES = REPORTS / "route_geometries.json"

NAIROBI_LAT, NAIROBI_LON = -1.286389, 36.817223

FEATURE_COLS = [
    "pop2009",
    "rain_cumulative_mm",
    "rain_max_daily_mm",
    "rain_preflood_7d_mm",
    "elevation_mean_m",
    "elevation_min_m",
    "elevation_max_m",
    "elevation_range_m",
    "slope_mean_deg",
]

# -- Page config --------------------------------------------------------------
st.set_page_config(
    page_title="Nairobi Flood Guard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Custom CSS ---------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Mono', monospace; }
h1, h2, h3, h4 { font-family: 'Syne', sans-serif !important; letter-spacing: -0.02em; }

.header-banner {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d2137 50%, #0a1628 100%);
    border-bottom: 2px solid #1a6fc4;
    padding: 2rem 2.5rem 1.5rem;
    margin: -1rem -1rem 2rem -1rem;
    position: relative; overflow: hidden;
}
.header-banner::before {
    content: ''; position: absolute; top: -50%; left: -10%;
    width: 120%; height: 200%;
    background: radial-gradient(ellipse at 30% 50%, rgba(26,111,196,0.15) 0%, transparent 60%),
                radial-gradient(ellipse at 70% 30%, rgba(226,75,74,0.08) 0%, transparent 50%);
    pointer-events: none;
}
.header-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 2.4rem; font-weight: 800;
    color: #ffffff; letter-spacing: -0.03em; margin: 0; line-height: 1.1;
}
.header-subtitle {
    font-family: 'DM Mono', monospace; font-size: 0.78rem; color: #1a6fc4;
    margin-top: 0.4rem; letter-spacing: 0.12em; text-transform: uppercase;
}
.badge {
    display: inline-block; padding: 0.25rem 0.75rem; border-radius: 2px;
    font-family: 'DM Mono', monospace; font-size: 0.75rem; font-weight: 500;
    letter-spacing: 0.08em; text-transform: uppercase;
}
.badge-low      { background: #1a3d2b; color: #4ade80; border: 1px solid #4ade80; }
.badge-moderate { background: #3d3010; color: #fbbf24; border: 1px solid #fbbf24; }
.badge-high     { background: #3d1a10; color: #fb923c; border: 1px solid #fb923c; }
.badge-critical { background: #2d0d0d; color: #f87171; border: 1px solid #f87171; }
.metric-card {
    background: #0d1117; border: 1px solid #1e2d3d;
    border-left: 3px solid #1a6fc4; border-radius: 4px;
    padding: 1rem 1.2rem; margin-bottom: 0.75rem;
}
.metric-label {
    font-size: 0.68rem; color: #6b7c93;
    letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.25rem;
}
.metric-value {
    font-family: 'Syne', sans-serif; font-size: 1.6rem;
    font-weight: 700; color: #e2e8f0; line-height: 1;
}
.metric-unit { font-size: 0.7rem; color: #4a5568; margin-left: 0.3rem; }
.section-header {
    font-family: 'Syne', sans-serif; font-size: 1.1rem; font-weight: 700;
    color: #e2e8f0; border-bottom: 1px solid #1e2d3d;
    padding-bottom: 0.5rem; margin-bottom: 1rem; letter-spacing: -0.01em;
}
section[data-testid="stSidebar"] { background: #080c14; border-right: 1px solid #1e2d3d; }
[data-testid="stSidebarCollapseButton"] { display: none !important; }
.material-symbols-rounded, [data-testid="stIconMaterial"] {
    font-family: 'Material Symbols Rounded' !important;
}
.ward-panel {
    background: #0a0f1e; border: 1px solid #1e2d3d;
    border-radius: 4px; padding: 1.25rem; margin-top: 1rem;
}
.ward-name {
    font-family: 'Syne', sans-serif; font-size: 1.3rem;
    font-weight: 700; color: #ffffff; margin-bottom: 0.2rem;
}
.ward-meta { font-size: 0.72rem; color: #4a6080; letter-spacing: 0.05em; }
.stSelectbox label, .stSlider label, .stTextInput label {
    font-family: 'DM Mono', monospace !important; font-size: 0.78rem !important;
    color: #6b7c93 !important; letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
div[data-testid="stMetric"] {
    background: #0d1117; border: 1px solid #1e2d3d;
    border-radius: 4px; padding: 0.75rem 1rem;
}
.route-stat-card {
    background: #0d1117; border: 1px solid #1e2d3d;
    border-radius: 4px; padding: 1rem 1.2rem; text-align: center;
}
.route-stat-label {
    font-size: 0.65rem; color: #6b7c93;
    letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.3rem;
}
.route-stat-value {
    font-family: 'Syne', sans-serif; font-size: 1.3rem;
    font-weight: 700; color: #e2e8f0;
}
</style>
""",
    unsafe_allow_html=True,
)


# -- Helpers ------------------------------------------------------------------
def risk_label(prob: float) -> tuple[str, str]:
    if prob >= 0.70:
        return "Critical", "badge-critical"
    if prob >= 0.45:
        return "High", "badge-high"
    if prob >= 0.20:
        return "Moderate", "badge-moderate"
    return "Low", "badge-low"


def risk_color(prob: float) -> str:
    if prob >= 0.70:
        return "#f87171"
    if prob >= 0.45:
        return "#fb923c"
    if prob >= 0.20:
        return "#fbbf24"
    return "#4ade80"


def normalize(col: str, df: pd.DataFrame) -> pd.Series:
    """Min-max normalise a DataFrame column to [0, 1]."""
    mn, mx = df[col].min(), df[col].max()
    return (df[col] - mn) / (mx - mn + 1e-9)


def highlight_best(s: pd.Series) -> list[str]:
    """Highlight the highest value in each column of the metrics table."""
    is_best = s == s.max()
    return [
        "background-color: #0d2137; color: #4ade80; font-weight:600" if v else ""
        for v in is_best
    ]


def render_message(role: str, content: str) -> None:
    """
    Render a chat message bubble with a clipboard icon that appears on hover.
    Content is base64-encoded to avoid all quote/backtick escaping issues.
    """
    b64 = base64.b64encode(content.encode("utf-8")).decode("utf-8")
    icon_color = "#4a6080"
    icon_hover = "#1a6fc4"
    align = "flex-end" if role == "user" else "flex-start"
    bubble_bg = "#0d1a2d" if role == "user" else "#0d1117"
    border_col = "#1a6fc4" if role == "user" else "#1e2d3d"
    avatar = "👤" if role == "user" else "🤖"
    btn_side = "left" if role == "user" else "right"

    html = f"""
<div style="display:flex; flex-direction:column; align-items:{align};
            margin-bottom:0.75rem; width:100%;">
  <div style="display:flex; align-items:flex-start; gap:0.5rem;
              flex-direction:{'row-reverse' if role == 'user' else 'row'}">
    <span style="font-size:1.2rem; margin-top:0.2rem;">{avatar}</span>
    <div class="msg-wrapper" style="position:relative; max-width:85%;">
      <div style="background:{bubble_bg}; border:1px solid {border_col};
                  border-radius:6px; padding:0.75rem 1rem;
                  font-family:'DM Mono',monospace; font-size:0.85rem;
                  color:#e2e8f0; line-height:1.6; white-space:pre-wrap;
                  word-break:break-word;">{content}</div>
      <button
        data-content="{b64}"
        onclick="
          var text = atob(this.getAttribute('data-content'));
          navigator.clipboard.writeText(text).then(() => {{
            var icon = this.querySelector('.icon-default');
            var check = this.querySelector('.icon-check');
            icon.style.display = 'none';
            check.style.display = 'inline';
            setTimeout(() => {{
              icon.style.display = 'inline';
              check.style.display = 'none';
            }}, 1500);
          }});
        "
        onmouseenter="this.style.color='{icon_hover}';this.style.borderColor='{icon_hover}';"
        onmouseleave="this.style.color='{icon_color}';this.style.borderColor='transparent';"
        style="position:absolute; top:0.4rem; {btn_side}:-2.2rem;
               background:transparent; border:1px solid transparent;
               border-radius:4px; padding:0.2rem 0.3rem;
               color:{icon_color}; cursor:pointer;
               font-size:0.9rem; line-height:1;
               opacity:0; transition:opacity 0.15s ease;"
        class="copy-icon-btn"
        title="Copy to clipboard">
        <span class="icon-default">📋</span>
        <span class="icon-check" style="display:none;">✓</span>
      </button>
    </div>
  </div>
</div>
<style>
  .msg-wrapper:hover .copy-icon-btn {{ opacity: 1 !important; }}
</style>
"""
    n_lines = content.count("\n") + len(content) // 80 + 1
    height = max(100, n_lines * 24 + 60)
    st.iframe(html, height=height, scrolling=True)


# -- Data loading -------------------------------------------------------------
@st.cache_data
def load_data():
    df = gpd.read_file(FLOODS_GPKG)
    df["elevation_range_m"] = df["elevation_max_m"] - df["elevation_min_m"]
    df["ward"] = df["ward"].str.title()
    df["county"] = df["county"].str.title()
    df["subcounty"] = df["subcounty"].str.title()
    return df


@st.cache_resource
def load_model():
    with open(XGB_MODEL, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_rerouting():
    return pd.read_csv(REROUTING_CSV)


@st.cache_resource
def get_groq_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])


@st.cache_data
def load_gtfs():
    routes = pd.read_csv(GTFS_DIR / "routes.txt")
    trips = pd.read_csv(GTFS_DIR / "trips.txt")
    shapes = pd.read_csv(GTFS_DIR / "shapes.txt")
    stops = pd.read_csv(GTFS_DIR / "stops.txt")
    stop_times = pd.read_csv(GTFS_DIR / "stop_times.txt")
    return routes, trips, shapes, stops, stop_times


@st.cache_data
def load_route_geometries():
    if not ROUTE_GEOMETRIES.exists():
        return {}
    with open(ROUTE_GEOMETRIES, "r") as f:
        return json.load(f)


@st.cache_data
def generate_predictions(_model, _df):
    X = _df[FEATURE_COLS].fillna(_df[FEATURE_COLS].median())
    return _model.predict_proba(X)[:, 1]


# Build choropleth - not cached since Folium maps with lambdas can't be pickled
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


# -- Load everything ----------------------------------------------------------
df = load_data()
model = load_model()
df["flood_prob"] = generate_predictions(model, df)
df["risk_label"], _ = zip(*df["flood_prob"].map(risk_label))
nairobi = df[df["county"].str.lower() == "nairobi"].copy()

# -- Header -------------------------------------------------------------------
st.markdown(
    """
<div class="header-banner">
    <div class="header-title">🌊 Nairobi Flood Guard</div>
    <div class="header-subtitle">Early Flood Warning &amp; Route Optimization System - Kenya</div>
</div>
""",
    unsafe_allow_html=True,
)

# -- Sidebar ------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Navigation")
    page = st.radio(
        "",
        [
            "Flood Risk Dashboard",
            "Ward Lookup",
            "Route Optimization",
            "Model Performance",
            "AI Assistant",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("### Risk Threshold")
    threshold = st.slider(
        "Flag wards above this probability as high-risk",
        min_value=0.10,
        max_value=0.90,
        value=0.45,
        step=0.05,
        format="%.2f",
    )
    st.markdown("---")
    n_critical = (df["flood_prob"] >= 0.70).sum()
    n_high = ((df["flood_prob"] >= threshold) & (df["flood_prob"] < 0.70)).sum()
    n_total = (df["flood_prob"] >= threshold).sum()

    st.markdown("### Kenya-Wide Summary")
    st.metric("Total Wards", len(df))
    st.metric("Critical Risk", int(n_critical))
    st.metric("High Risk", int(n_high))
    st.metric("Above Threshold", int(n_total))
    st.markdown("---")
    st.markdown(
        "<span style='font-size:0.65rem;color:#4a5568;'>Model: XGBoost · "
        "Data: UNOSAT April 2024 · Terrain: SRTM 90m · "
        "Rainfall: CHIRPS Feb-Apr 2024</span>",
        unsafe_allow_html=True,
    )


# =============================================================================
# PAGE 1 - FLOOD RISK DASHBOARD
# =============================================================================
if page == "Flood Risk Dashboard":

    st.markdown(
        '<div class="section-header">Kenya-Wide Flood Risk Map</div>',
        unsafe_allow_html=True,
    )

    col_filter, col_info = st.columns([2, 1])
    with col_filter:
        counties = sorted(df["county"].unique())
        selected_county = st.selectbox("Filter by county", ["All Kenya"] + counties)

    map_df = (
        df if selected_county == "All Kenya" else df[df["county"] == selected_county]
    )

    with col_info:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Showing</div>
            <div class="metric-value">{len(map_df)}<span class="metric-unit">wards</span></div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    centre_lat = float(map_df.geometry.centroid.y.mean())
    centre_lon = float(map_df.geometry.centroid.x.mean())
    zoom = 7 if selected_county == "All Kenya" else 10

    with st.spinner("Rendering map..."):
        fmap = build_choropleth(map_df, centre_lat, centre_lon, zoom)
    st_folium(fmap, use_container_width=True, height=520)

    st.markdown(
        '<div class="section-header" style="margin-top:2rem">Flood Probability Distribution</div>',
        unsafe_allow_html=True,
    )
    fig = px.histogram(
        map_df,
        x="flood_prob",
        nbins=40,
        color_discrete_sequence=["#1a6fc4"],
        labels={"flood_prob": "Flood Probability", "count": "Number of Wards"},
    )
    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="#f87171",
        annotation_text=f"Threshold ({threshold:.2f})",
        annotation_font_color="#f87171",
        annotation_position="top right",
    )
    fig.update_layout(
        paper_bgcolor="#0a0f1e",
        plot_bgcolor="#0d1117",
        font_color="#e2e8f0",
        font_family="DM Mono",
        margin=dict(t=20, b=20, l=20, r=20),
        xaxis=dict(gridcolor="#1e2d3d", tickformat=".0%"),
        yaxis=dict(gridcolor="#1e2d3d"),
        bargap=0.05,
    )
    st.plotly_chart(fig, use_container_width=True)

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
        use_container_width=True,
    )


# =============================================================================
# PAGE 2 - WARD LOOKUP
# =============================================================================
elif page == "Ward Lookup":

    st.markdown(
        '<div class="section-header">Ward Flood Risk Lookup</div>',
        unsafe_allow_html=True,
    )

    ward_names = sorted(df["ward"].unique())
    selected_ward = st.selectbox("Search for a ward", ward_names)

    ward_row = df[df["ward"] == selected_ward].iloc[0]
    prob = float(ward_row["flood_prob"])
    label, badge_class = risk_label(prob)

    st.markdown(
        f"""
    <div class="ward-panel">
        <div class="ward-name">{selected_ward}</div>
        <div class="ward-meta">{ward_row['subcounty']} &nbsp;·&nbsp; {ward_row['county']}</div>
        <div style="margin-top:1rem">
            <span class="badge {badge_class}">{label} Risk</span>
            &nbsp;
            <span style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:700;
                         color:{risk_color(prob)}">{prob:.1%}</span>
            <span style="font-size:0.7rem;color:#4a6080;margin-left:0.3rem">flood probability</span>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Feature Breakdown")
    feature_labels = {
        "pop2009": ("Population (2009)", "people"),
        "elevation_mean_m": ("Mean Elevation", "m"),
        "elevation_min_m": ("Min Elevation", "m"),
        "elevation_max_m": ("Max Elevation", "m"),
        "elevation_range_m": ("Elevation Range", "m"),
        "slope_mean_deg": ("Mean Slope", "°"),
        "rain_cumulative_mm": ("Cumulative Rainfall (90d)", "mm"),
        "rain_max_daily_mm": ("Max Daily Rainfall", "mm"),
        "rain_preflood_7d_mm": ("Pre-Flood 7-Day Rainfall", "mm"),
    }
    cols = st.columns(4)
    for i, (col_name, (label_text, unit)) in enumerate(feature_labels.items()):
        val = ward_row[col_name]
        with cols[i % 4]:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-label">{label_text}</div>
                <div class="metric-value">{val:,.0f}<span class="metric-unit">{unit}</span></div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    st.markdown("#### Ward vs. Kenya Average")
    features_for_radar = [
        "elevation_mean_m",
        "slope_mean_deg",
        "rain_cumulative_mm",
        "rain_max_daily_mm",
        "rain_preflood_7d_mm",
        "pop2009",
    ]
    radar_labels = [
        "Elevation",
        "Slope",
        "Cumul. Rain",
        "Max Daily Rain",
        "Pre-Flood Rain",
        "Population",
    ]

    ward_vals = [
        float(normalize(c, df)[df["ward"] == selected_ward].values[0])
        for c in features_for_radar
    ]
    avg_vals = [float(normalize(c, df).mean()) for c in features_for_radar]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=ward_vals + [ward_vals[0]],
            theta=radar_labels + [radar_labels[0]],
            fill="toself",
            name=selected_ward,
            line_color="#1a6fc4",
            fillcolor="rgba(26,111,196,0.2)",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=avg_vals + [avg_vals[0]],
            theta=radar_labels + [radar_labels[0]],
            fill="toself",
            name="Kenya Average",
            line_color="#4ade80",
            fillcolor="rgba(74,222,128,0.1)",
        )
    )
    fig.update_layout(
        polar=dict(
            bgcolor="#0d1117",
            radialaxis=dict(visible=True, gridcolor="#1e2d3d", color="#4a5568"),
            angularaxis=dict(gridcolor="#1e2d3d", color="#6b7c93"),
        ),
        paper_bgcolor="#0a0f1e",
        font_color="#e2e8f0",
        font_family="DM Mono",
        legend=dict(bgcolor="#0a0f1e", bordercolor="#1e2d3d", borderwidth=1),
        margin=dict(t=30, b=30, l=30, r=30),
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Ward Location")
    ward_lat = float(ward_row.geometry.centroid.y)
    ward_lon = float(ward_row.geometry.centroid.x)

    mini_map = folium.Map(
        location=[ward_lat, ward_lon], zoom_start=11, tiles="CartoDB dark_matter"
    )
    folium.GeoJson(
        ward_row.geometry.__geo_interface__,
        style_function=lambda _: {
            "fillColor": risk_color(prob),
            "fillOpacity": 0.55,
            "color": risk_color(prob),
            "weight": 2,
        },
        tooltip=f"{selected_ward} - {prob:.1%} flood probability",
    ).add_to(mini_map)
    st_folium(mini_map, use_container_width=True, height=320)


# =============================================================================
# PAGE 3 - ROUTE OPTIMIZATION
# =============================================================================
elif page == "Route Optimization":

    st.markdown(
        '<div class="section-header">Matatu Route Optimization - April 2024 Flood Event</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "The route optimization system uses XGBoost flood predictions to identify which "
        "Nairobi matatu routes pass through high-risk wards. Flooded road segments are "
        "blocked outright and Dijkstra's algorithm finds the safest available alternative "
        "path for each affected route. The GTFS-RT feed generated is immediately "
        "consumable by transit apps."
    )

    if not REROUTING_CSV.exists():
        st.warning(
            "Rerouting summary not found. "
            "Run Route_Optimization/route_optimization.ipynb first."
        )
        st.stop()

    rerouting_df = load_rerouting()
    routes, trips, shapes, stops, stop_times = load_gtfs()
    route_geoms = load_route_geometries()

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Affected Routes", len(rerouting_df))
    with c2:
        st.metric("Avg Risk Reduction", f"{rerouting_df['risk_reduction'].mean():.3f}")
    with c3:
        st.metric("Avg Extra Time", f"{rerouting_df['extra_time_min'].mean():.1f} min")
    with c4:
        st.metric("Routes Improved", int((rerouting_df["risk_reduction"] > 0).sum()))

    # Rerouting summary table
    st.markdown(
        '<div class="section-header" style="margin-top:1.5rem">Rerouting Summary</div>',
        unsafe_allow_html=True,
    )
    sort_col = st.selectbox(
        "Sort by",
        [
            "risk_reduction",
            "extra_time_min",
            "original_flood_prob",
            "alternative_flood_prob",
        ],
        format_func=lambda x: x.replace("_", " ").title(),
    )
    display_df = (
        rerouting_df[
            [
                "route_id",
                "origin",
                "destination",
                "original_flood_prob",
                "alternative_flood_prob",
                "risk_reduction",
                "extra_time_min",
            ]
        ]
        .sort_values(sort_col, ascending=False)
        .reset_index(drop=True)
    )
    display_df.index += 1
    display_df.columns = [
        "Route ID",
        "Origin",
        "Destination",
        "Original Risk",
        "Alternative Risk",
        "Risk Reduction",
        "Extra Time (min)",
    ]
    st.dataframe(display_df, use_container_width=True)
    st.download_button(
        label="⬇ Download Rerouting CSV",
        data=rerouting_df.to_csv(index=False),
        file_name="rerouting_summary.csv",
        mime="text/csv",
    )

    # Tradeoff chart
    st.markdown(
        '<div class="section-header" style="margin-top:1.5rem">Risk-Time Tradeoff</div>',
        unsafe_allow_html=True,
    )
    if TRADEOFF_PNG.exists():
        st.image(str(TRADEOFF_PNG), use_container_width=True)
    else:
        st.info(
            "Tradeoff chart not found. Run route_optimization.ipynb to generate it."
        )

    # Interactive map section
    st.markdown(
        '<div class="section-header" style="margin-top:1.5rem">Interactive Map</div>',
        unsafe_allow_html=True,
    )
    map_view = st.radio(
        "View",
        ["🗺 Flood Risk Map", "🚌 Route Explorer"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if map_view == "🗺 Flood Risk Map":
        st.caption("Nairobi ward flood risk · hover a ward for details")
        with st.spinner("Rendering flood risk map..."):
            risk_map = build_choropleth(nairobi, NAIROBI_LAT, NAIROBI_LON, zoom=11)
        st_folium(risk_map, use_container_width=True, height=520)

    else:
        affected_stop_ids = get_affected_stop_ids(nairobi, stops, threshold)
        affected_route_ids = rerouting_df["route_id"].tolist()
        n_routes = len(affected_route_ids)

        if "route_idx" not in st.session_state:
            st.session_state.route_idx = 0

        nav_left, nav_centre, nav_right = st.columns([1, 4, 1])

        with nav_left:
            if st.button("← Previous", use_container_width=True):
                st.session_state.route_idx = (st.session_state.route_idx - 1) % n_routes

        with nav_right:
            if st.button("Next →", use_container_width=True):
                st.session_state.route_idx = (st.session_state.route_idx + 1) % n_routes

        idx = st.session_state.route_idx
        route_row = rerouting_df.iloc[idx]
        route_id = route_row["route_id"]

        with nav_centre:
            st.markdown(
                f"<div style='text-align:center;padding:0.4rem 0;'>"
                f"<span style='font-family:Syne,sans-serif;font-size:1rem;"
                f"font-weight:700;color:#e2e8f0;'>Route {route_id}</span>"
                f"<span style='font-size:0.72rem;color:#4a6080;margin-left:0.6rem;'>"
                f"{route_row['origin']} → {route_row['destination']}</span>"
                f"<span style='font-size:0.65rem;color:#4a5568;margin-left:0.6rem;'>"
                f"({idx + 1} / {n_routes})</span></div>",
                unsafe_allow_html=True,
            )

        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.markdown(
                f"""<div class="route-stat-card">
                <div class="route-stat-label">Original Flood Risk</div>
                <div class="route-stat-value" style="color:#f87171;">
                    {route_row['original_flood_prob']:.1%}
                </div></div>""",
                unsafe_allow_html=True,
            )
        with s2:
            st.markdown(
                f"""<div class="route-stat-card">
                <div class="route-stat-label">Alternative Flood Risk</div>
                <div class="route-stat-value" style="color:#4ade80;">
                    {route_row['alternative_flood_prob']:.1%}
                </div></div>""",
                unsafe_allow_html=True,
            )
        with s3:
            st.markdown(
                f"""<div class="route-stat-card">
                <div class="route-stat-label">Risk Reduction</div>
                <div class="route-stat-value" style="color:#1a6fc4;">
                    {route_row['risk_reduction']:.3f}
                </div></div>""",
                unsafe_allow_html=True,
            )
        with s4:
            st.markdown(
                f"""<div class="route-stat-card">
                <div class="route-stat-label">Extra Travel Time</div>
                <div class="route-stat-value" style="color:#fbbf24;">
                    +{route_row['extra_time_min']:.1f} min
                </div></div>""",
                unsafe_allow_html=True,
            )

        st.markdown("<div style='margin-top:0.75rem'></div>", unsafe_allow_html=True)
        route_view = st.radio(
            "Route view",
            ["Original Route", "Alternative Route"],
            horizontal=True,
            label_visibility="collapsed",
        )

        route_map = folium.Map(
            location=[NAIROBI_LAT, NAIROBI_LON],
            zoom_start=12,
            tiles="CartoDB dark_matter",
        )
        route_coords = get_route_shape(route_id, trips, shapes)
        route_stops = get_route_stops(route_id, trips, stop_times, stops)

        if route_view == "Original Route":
            st.caption(
                "🔵 Original route path · 🔴 Affected stops (in flood-risk wards) · "
                "⚪ Safe stops"
            )
            if route_coords:
                folium.PolyLine(
                    route_coords,
                    color="#378ADD",
                    weight=4,
                    opacity=0.9,
                    tooltip=f"Route {route_id} - Original",
                ).add_to(route_map)
            if not route_stops.empty:
                for _, stop_row in route_stops.iterrows():
                    is_affected = stop_row["stop_id"] in affected_stop_ids
                    folium.CircleMarker(
                        location=[stop_row["stop_lat"], stop_row["stop_lon"]],
                        radius=5 if is_affected else 3,
                        color="#f87171" if is_affected else "#4a5568",
                        fill=True,
                        fill_color="#f87171" if is_affected else "#4a5568",
                        fill_opacity=0.9,
                        tooltip=(
                            f"⚠ Affected: {stop_row.get('stop_name', stop_row['stop_id'])}"
                            if is_affected
                            else str(stop_row.get("stop_name", stop_row["stop_id"]))
                        ),
                    ).add_to(route_map)
            if route_coords:
                lats = [c[0] for c in route_coords]
                lons = [c[1] for c in route_coords]
                route_map.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])

        else:
            alt_coords = route_geoms.get(str(route_id), {}).get("alternative", [])
            st.caption(
                "🟠 Alternative route (Dijkstra, flood-weighted) · "
                "🔵 Original route (faded reference) · "
                "🔴 Affected stops skipped · "
                "risk reduced by {:.3f} · +{:.1f} min".format(
                    route_row["risk_reduction"], route_row["extra_time_min"]
                )
            )
            if route_coords:
                folium.PolyLine(
                    route_coords,
                    color="#1e3a5f",
                    weight=3,
                    opacity=0.4,
                    tooltip=f"Route {route_id} - Original (reference)",
                    dash_array="6",
                ).add_to(route_map)
            if alt_coords:
                folium.PolyLine(
                    alt_coords,
                    color="#EF9F27",
                    weight=4,
                    opacity=0.9,
                    dash_array="8",
                    tooltip=f"Route {route_id} - Alternative",
                ).add_to(route_map)
            else:
                st.warning(
                    "Alternative path geometry not found. "
                    "Re-run route_optimization.ipynb."
                )
            if not route_stops.empty:
                for _, stop_row in route_stops.iterrows():
                    is_affected = stop_row["stop_id"] in affected_stop_ids
                    folium.CircleMarker(
                        location=[stop_row["stop_lat"], stop_row["stop_lon"]],
                        radius=5 if is_affected else 3,
                        color="#f87171" if is_affected else "#2d3748",
                        fill=True,
                        fill_color="#f87171" if is_affected else "#2d3748",
                        fill_opacity=0.85,
                        tooltip=(
                            f"🚫 Skipped: {stop_row.get('stop_name', stop_row['stop_id'])}"
                            if is_affected
                            else str(stop_row.get("stop_name", stop_row["stop_id"]))
                        ),
                    ).add_to(route_map)
            coords_for_bounds = alt_coords if alt_coords else route_coords
            if coords_for_bounds:
                lats = [c[0] for c in coords_for_bounds]
                lons = [c[1] for c in coords_for_bounds]
                route_map.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
            if alt_coords:
                mid = alt_coords[len(alt_coords) // 2]
                folium.Marker(
                    location=mid,
                    icon=folium.DivIcon(
                        html=(
                            f"<div style='background:#0d2137;border:1px solid #EF9F27;"
                            f"border-radius:4px;padding:4px 8px;font-family:monospace;"
                            f"font-size:11px;color:#e2e8f0;white-space:nowrap;'>"
                            f"Alternative · Risk ↓{route_row['risk_reduction']:.3f}"
                            f" · +{route_row['extra_time_min']:.1f} min</div>"
                        ),
                        icon_size=(270, 30),
                        icon_anchor=(135, 15),
                    ),
                ).add_to(route_map)

        st_folium(route_map, use_container_width=True, height=500)


# =============================================================================
# PAGE 4 - AI ASSISTANT (Mlinzi)
# =============================================================================
elif page == "AI Assistant":

    st.markdown(
        '<div class="section-header">Flood Guard AI Assistant</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Hi. My name is Mlinzi, an AI Chatbot specifically designed to help you "
        "with any questions you might have regarding flooding in Kenya. "
        "Ask anything about flood risk, affected wards, route recommendations, "
        "or how to interpret the model results."
    )

    # Input form at the top
    with st.form(key="chat_form", clear_on_submit=True):
        input_col, btn_col = st.columns([11, 1])
        with input_col:
            user_input = st.text_input(
                "",
                placeholder="Ask about flood risk, routes, or the model...",
                label_visibility="collapsed",
            )
        with btn_col:
            submitted = st.form_submit_button("➤")

    st.markdown(
        "<hr style='border:none;border-top:1px solid #1e2d3d;margin:0.5rem 0 1rem 0;'>",
        unsafe_allow_html=True,
    )

    # Build rich context
    wards_context = (
        df[df["county"].str.lower().str.strip() == "nairobi"][
            [
                "ward",
                "subcounty",
                "county",
                "flood_prob",
                "risk_label",
                "elevation_mean_m",
                "elevation_min_m",
                "elevation_max_m",
                "slope_mean_deg",
                "rain_cumulative_mm",
                "rain_max_daily_mm",
                "rain_preflood_7d_mm",
                "pop2009",
            ]
        ]
        .sort_values("flood_prob", ascending=False)
        .head(100)
        .assign(flood_prob=lambda x: x["flood_prob"].map("{:.1%}".format))
        .to_string(index=False)
    )

    model_perf_context = ""
    model_csv = DATA / "model_comparison.csv"
    if model_csv.exists():
        model_perf_context = "\nModel Performance Comparison:\n" + pd.read_csv(
            model_csv
        ).to_string(index=False)

    rerouting_context = ""
    if REROUTING_CSV.exists():
        r = load_rerouting()
        rerouting_context = (
            f"\nFull Rerouting Summary ({len(r)} affected routes):\n"
            + r.to_string(index=False)
            + "\n\nAggregate stats:"
            + f"\n  Average risk reduction : {r['risk_reduction'].mean():.3f}"
            + f"\n  Average extra time (min): {r['extra_time_min'].mean():.1f}"
            + f"\n  Routes with risk > 0   : {(r['risk_reduction'] > 0).sum()}"
        )

    system_prompt = (
        "You are Mlinzi, an AI assistant for Nairobi Flood Guard - a data science project\n"
        "that predicts flood susceptibility across Kenya's 1,450 administrative wards and\n"
        "recommends alternative matatu routes during flood events. Your name means\n"
        "'guardian' or 'protector' in Swahili, which reflects your purpose.\n\n"
        "--- PROJECT OVERVIEW ---\n"
        "The prediction model is XGBoost trained on the following features:\n"
        "  Terrain  : elevation (mean, min, max, range in metres), slope (degrees)\n"
        "  Rainfall : cumulative 90-day (mm), max single-day (mm), 7-day pre-flood (mm)\n"
        "  Population: 2009 Kenya census ward population\n\n"
        "Key insight: flooding in Kenya is primarily terrain-driven at ward scale.\n"
        "Low-lying wards flood not because they receive more rain but because water drains\n"
        "into them from surrounding higher ground. Elevation features dominate predictions;\n"
        "rainfall adds marginal predictive value at this spatial resolution.\n\n"
        "--- CURRENT RISK SUMMARY ---\n"
        f"Total wards         : {len(df)}\n"
        f"High-risk (>= {threshold:.0%}) : {int(n_total)}\n"
        f"Critical risk (>= 70%): {int(n_critical)}\n\n"
        "Risk thresholds:\n"
        "  Low      : flood probability < 20%\n"
        "  Moderate : 20% - 45%\n"
        "  High     : 45% - 70%\n"
        "  Critical : >= 70%\n\n"
        "--- ALL NAIROBI DATA (top 100 by flood probability) ---\n"
        f"{wards_context}\n\n"
        "--- MODEL PERFORMANCE ---\n"
        f"{model_perf_context if model_perf_context else 'Model comparison data not available.'}\n\n"
        "--- ROUTE OPTIMIZATION ---\n"
        "The route optimization system:\n"
        "  - Uses XGBoost flood probabilities to identify high-risk road segments\n"
        "  - Assigns flood cost = travel_time x (1 + alpha x flood_probability), "
        "alpha = 1,000,000\n"
        "    (effectively blocking all flood-affected roads outright)\n"
        "  - Runs weighted Dijkstra to find the safest alternative path\n"
        "  - Outputs a GTFS-RT feed consumable by transit apps\n\n"
        f"{rerouting_context if rerouting_context else 'Rerouting data not available.'}\n\n"
        "--- INSTRUCTIONS ---\n"
        "- Be concise, factual, and actionable.\n"
        "- When asked about a specific ward, look it up in the ward data above and quote\n"
        "  its exact flood probability, risk level, and key terrain/rainfall figures.\n"
        "- When asked about routes, reference the rerouting summary above by route ID.\n"
        "- If asked which wards are most at risk, list the top entries from the ward data.\n"
        "- If asked about the model, explain the XGBoost pipeline and feature importance.\n"
        "- Do not make up data - all ward and route figures are provided above.\n"
        "- Respond in English unless the user writes in another language.\n"
    )

    # Process new input
    if submitted and user_input.strip():
        if "messages" not in st.session_state:
            st.session_state.messages = []
        st.session_state.messages.append(
            {"role": "user", "content": user_input.strip()}
        )
        with st.spinner("Mlinzi is thinking..."):
            client = get_groq_client()
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    *st.session_state.messages,
                ],
                max_tokens=1024,
                temperature=0.4,
            )
            reply = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": reply})

    # Render message history below the input
    for msg in st.session_state.get("messages", []):
        render_message(msg["role"], msg["content"])

    if st.session_state.get("messages"):
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.rerun()


# =============================================================================
# PAGE 5 - MODEL PERFORMANCE
# =============================================================================
elif page == "Model Performance":

    st.markdown(
        '<div class="section-header">Model Evaluation Summary</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Four classification models were trained and evaluated on 1,450 Kenya "
        "ward-level samples. **Recall** is the primary metric - a missed flood "
        "prediction carries far greater consequences than a false alarm. The XGBoost "
        "model was selected as the best overall performer and powers both the flood "
        "risk map and the route optimization system."
    )

    metrics_df = pd.read_csv(DATA / "model_comparison.csv")
    styled = (
        metrics_df.set_index("Model")
        .style.apply(highlight_best, axis=0)
        .format("{:.3f}")
    )
    st.dataframe(styled, use_container_width=True)

    st.markdown(
        '<div class="section-header" style="margin-top:1.5rem">Feature Importance (XGBoost)</div>',
        unsafe_allow_html=True,
    )
    try:
        classifier = model.named_steps["classifier"]
        importances = classifier.feature_importances_
        feat_imp = pd.DataFrame(
            {"Feature": FEATURE_COLS, "Importance": importances}
        ).sort_values("Importance", ascending=True)
        fig = px.bar(
            feat_imp,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale=["#0d2137", "#1a6fc4", "#4ade80"],
            labels={"Importance": "Feature Importance", "Feature": ""},
        )
        fig.update_layout(
            paper_bgcolor="#0a0f1e",
            plot_bgcolor="#0d1117",
            font_color="#e2e8f0",
            font_family="DM Mono",
            coloraxis_showscale=False,
            margin=dict(t=10, b=10, l=10, r=10),
            xaxis=dict(gridcolor="#1e2d3d"),
            yaxis=dict(gridcolor="#1e2d3d"),
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)
    except KeyError:
        st.warning(
            "Could not find 'classifier' step in the pipeline. "
            "Check the step name with: print(model.named_steps.keys())"
        )
    except Exception as e:
        st.error(f"Feature importance error: {e}")

    st.markdown(
        '<div class="section-header" style="margin-top:1.5rem">Data Overview</div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Wards", len(df))
    with c2:
        st.metric("Flooded Wards", int(df["flooded"].sum()))
    with c3:
        st.metric("Counties Covered", df["county"].nunique())
    with c4:
        st.metric("Features Used", len(FEATURE_COLS))

    fig2 = px.pie(
        names=["Not Flooded", "Flooded"],
        values=[(df["flooded"] == 0).sum(), df["flooded"].sum()],
        color_discrete_sequence=["#1a6fc4", "#f87171"],
        hole=0.55,
    )
    fig2.update_layout(
        paper_bgcolor="#0a0f1e",
        font_color="#e2e8f0",
        font_family="DM Mono",
        legend=dict(bgcolor="#0a0f1e"),
        margin=dict(t=10, b=10, l=10, r=10),
        height=300,
    )
    st.plotly_chart(fig2, use_container_width=True)
