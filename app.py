"""
Nairobi Flood Guard — Streamlit UI
Run with: streamlit run app.py
"""

import warnings
import pickle
from pathlib import Path

import pandas as pd
import geopandas as gpd
import folium
import streamlit as st
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")


# ── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
DATA = BASE / "Data"
MODELS = BASE / "Models"
REPORTS = BASE / "Route_Optimization" / "Reports"

FLOODS_GPKG = DATA / "floods.gpkg"
XGB_MODEL = MODELS / "best_xgboost_model.pkl"
REROUTING_CSV = REPORTS / "rerouting_summary.csv"
TRADEOFF_PNG = REPORTS / "rerouting_tradeoff.png"
FOLIUM_MAP = REPORTS / "flood_rerouting_map.html"

# Nairobi centre coordinates
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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nairobi Flood Guard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
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
section[data-testid="stSidebar"] * { font-family: 'DM Mono', monospace !important; }
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
</style>
""",
    unsafe_allow_html=True,
)


# ── Helpers ───────────────────────────────────────────────────────────────────
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


def recenter_folium_html(html: str, lat: float, lon: float, zoom: int = 12) -> str:
    """Inject JS to re-centre a saved Folium map on the given coordinates."""
    inject = f"""
    <script>
    document.addEventListener("DOMContentLoaded", function() {{
        setTimeout(function() {{
            var maps = Object.values(window).filter(function(v) {{
                return v && typeof v === 'object' && typeof v.setView === 'function';
            }});
            if (maps.length > 0) {{ maps[0].setView([{lat}, {lon}], {zoom}); }}
        }}, 300);
    }});
    </script>
    """
    return html.replace("</body>", inject + "</body>")


# ── Data loading ──────────────────────────────────────────────────────────────
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


@st.cache_data
def generate_predictions(_model, _df):
    X = _df[FEATURE_COLS].fillna(_df[FEATURE_COLS].median())
    return _model.predict_proba(X)[:, 1]


@st.cache_data
def build_choropleth(_map_df, centre_lat, centre_lon, zoom):
    """
    Build Folium choropleth using a single GeoJson call on the full
    GeoDataFrame — avoids the slow iterrows() loop that caused the dashboard
    to appear frozen.
    """
    fmap = folium.Map(
        location=[centre_lat, centre_lon], zoom_start=zoom, tiles="CartoDB dark_matter"
    )
    folium.GeoJson(
        _map_df[
            ["ward", "subcounty", "county", "flood_prob", "risk_label", "geometry"]
        ],
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


# ── Load everything ───────────────────────────────────────────────────────────
df = load_data()
model = load_model()
df["flood_prob"] = generate_predictions(model, df)
df["risk_label"], _ = zip(*df["flood_prob"].map(risk_label))
nairobi = df[df["county"].str.lower() == "nairobi"].copy()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="header-banner">
    <div class="header-title">🌊 Nairobi Flood Guard</div>
    <div class="header-subtitle">Early Flood Warning & Route Optimization System — Kenya 2024</div>
</div>
""",
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Navigation")
    page = st.radio(
        "",
        [
            "Flood Risk Dashboard",
            "Ward Lookup",
            "Route Optimization",
            "Model Performance",
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
        "<span style='font-size:0.65rem;color:#4a5568;'>Model: XGBoost · Data: UNOSAT April 2024 · "
        "Terrain: SRTM 90m · Rainfall: CHIRPS Feb–Apr 2024</span>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — FLOOD RISK DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
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
        '<div class="section-header">Highest Risk Wards</div>', unsafe_allow_html=True
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


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — WARD LOOKUP
# ═══════════════════════════════════════════════════════════════════════════════
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

    def normalize(col):
        mn, mx = df[col].min(), df[col].max()
        return (df[col] - mn) / (mx - mn + 1e-9)

    ward_vals = [
        float(normalize(c)[df["ward"] == selected_ward].values[0])
        for c in features_for_radar
    ]
    avg_vals = [float(normalize(c).mean()) for c in features_for_radar]

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
        tooltip=f"{selected_ward} — {prob:.1%} flood probability",
    ).add_to(mini_map)
    st_folium(mini_map, use_container_width=True, height=320)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ROUTE OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Route Optimization":

    st.markdown(
        '<div class="section-header">Matatu Route Optimization — April 2024 Flood Event</div>',
        unsafe_allow_html=True,
    )
    st.markdown("""
    The route optimization system uses XGBoost flood predictions to identify which Nairobi
    matatu routes pass through high-risk wards. Flooded road segments are blocked outright
    and Dijkstra's algorithm finds the safest available alternative path for each affected
    route. The GTFS-RT feed generated is immediately consumable by transit apps.
    """)

    if REROUTING_CSV.exists():
        rerouting_df = load_rerouting()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Affected Routes", len(rerouting_df))
        with c2:
            st.metric(
                "Avg Risk Reduction", f"{rerouting_df['risk_reduction'].mean():.3f}"
            )
        with c3:
            st.metric(
                "Avg Extra Time", f"{rerouting_df['extra_time_min'].mean():.1f} min"
            )
        with c4:
            st.metric(
                "Routes Improved", int((rerouting_df["risk_reduction"] > 0).sum())
            )

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

        st.markdown(
            '<div class="section-header" style="margin-top:1.5rem">Risk-Time Tradeoff</div>',
            unsafe_allow_html=True,
        )
        if TRADEOFF_PNG.exists():
            st.image(str(TRADEOFF_PNG), width="content")
        else:
            st.info(
                "Tradeoff chart not found. Run route_optimization.ipynb to generate it."
            )
    else:
        st.warning(
            "Rerouting summary not found. Run Route_Optimization/route_optimization.ipynb first."
        )

    # Folium map — re-centred on Nairobi via JS injection
    st.markdown(
        '<div class="section-header" style="margin-top:1.5rem">Interactive Route Map</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Original routes shown in blue · Alternative routes shown in dashed orange · Affected stops in red"
    )
    if FOLIUM_MAP.exists():
        with open(FOLIUM_MAP, "r", encoding="utf-8") as f:
            map_html = f.read()
        map_html = recenter_folium_html(map_html, NAIROBI_LAT, NAIROBI_LON, zoom=12)
        st.components.v1.html(map_html, height=560, scrolling=False)
    else:
        st.info(
            "Folium map not found. Run Route_Optimization/route_optimization.ipynb to generate it."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":

    st.markdown(
        '<div class="section-header">Model Evaluation Summary</div>',
        unsafe_allow_html=True,
    )
    st.markdown("""
    Four classification models were trained and evaluated on 1,450 Kenya ward-level samples.
    **Recall** is the primary metric — a missed flood prediction carries far greater
    consequences than a false alarm. The XGBoost model was selected as the best overall
    performer and powers both the flood risk map and the route optimization system.
    """)

    metrics_df = pd.read_csv("./Data/model_comparison.csv")

    def highlight_best(s):
        is_best = s == s.max()
        return [
            "background-color: #0d2137; color: #4ade80; font-weight:600" if v else ""
            for v in is_best
        ]

    styled = (
        metrics_df.set_index("Model")
        .style.apply(highlight_best, axis=0)
        .format("{:.3f}")
    )
    st.dataframe(styled, use_container_width=True)

    # Feature importance — access classifier inside the pipeline via named_steps
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
            "Check the step name with: `print(model.named_steps.keys())`"
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
