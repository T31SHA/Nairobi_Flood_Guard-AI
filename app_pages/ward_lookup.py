"""Ward Lookup page: per-ward risk, features, and SHAP explanation."""

import folium
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

from app_lib.data import load_explainer_model
from app_lib.state import get_state
from app_lib.theme import PLOTLY_LAYOUT, normalize, risk_color, risk_label
from Utils.feature_engineering import engineer_features

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

state = get_state()
df = state["df"]
registry = state["registry"]
FEATURE_COLS = registry["feature_cols"]

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
    <div style="margin-top:1rem; position:relative;">
        <span class="badge {badge_class}">{label} Risk</span>
        &nbsp;
        <span style="font-family:'Fraunces',serif;font-size:1.5rem;font-weight:600;
                     color:{risk_color(prob)}">{prob:.1%}</span>
        <span style="font-size:0.7rem;color:#8FA894;margin-left:0.3rem">flood probability</span>
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
    "rain_preflood_7d_mm": ("Recent 7-Day Rainfall", "mm"),
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
        line_color="#D4A24C",
        fillcolor="rgba(212,162,76,0.2)",
    )
)
fig.add_trace(
    go.Scatterpolar(
        r=avg_vals + [avg_vals[0]],
        theta=radar_labels + [radar_labels[0]],
        fill="toself",
        name="Kenya Average",
        line_color="#3FA66B",
        fillcolor="rgba(63,166,107,0.1)",
    )
)
fig.update_layout(
    polar=dict(
        bgcolor="#0E2318",
        radialaxis=dict(visible=True, gridcolor="#1F4A32", color="#4E6357"),
        angularaxis=dict(gridcolor="#1F4A32", color="#8FA894"),
    ),
    paper_bgcolor="#07110D",
    font_color="#E8DFC8",
    font_family="Space Mono",
    legend=dict(bgcolor="#07110D", bordercolor="#1F4A32", borderwidth=1),
    margin=dict(t=30, b=30, l=30, r=30),
    height=380,
)
st.plotly_chart(fig, width="stretch")

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
st_folium(mini_map, width="stretch", height=320)

# -- SHAP Feature Explanation ---------------------------------------------
st.markdown("#### Why this flood risk score?")
st.caption(
    "Each bar shows how much a feature pushed the flood probability "
    "up (positive) or down (negative) from the baseline. Attributions come "
    "from the model's uncalibrated booster; calibration is monotonic so the "
    "direction and ranking of contributions are unchanged."
)

engineered = engineer_features(df)
X_all = engineered[FEATURE_COLS].fillna(engineered[FEATURE_COLS].median())
ward_idx = df[df["ward"] == selected_ward].index[0]

if SHAP_AVAILABLE:
    try:
        booster = load_explainer_model()
        explainer = shap.TreeExplainer(booster)
        shap_vals = explainer.shap_values(X_all)
        ward_shap = shap_vals[df.index.get_loc(ward_idx)]
        ward_data = X_all.loc[ward_idx]

        shap_df = pd.DataFrame(
            {
                "Feature": FEATURE_COLS,
                "SHAP Value": ward_shap,
                "Feature Value": ward_data.values,
            }
        ).sort_values("SHAP Value")
        shap_df["Label"] = shap_df.apply(
            lambda r: f"{r['Feature'].replace('_', ' ').title()}  ({r['Feature Value']:,.1f})",
            axis=1,
        )
        shap_df["Color"] = shap_df["SHAP Value"].apply(
            lambda v: "#C4622D" if v > 0 else "#3FA66B"
        )

        fig_shap = go.Figure(
            go.Bar(
                x=shap_df["SHAP Value"],
                y=shap_df["Label"],
                orientation="h",
                marker_color=shap_df["Color"],
                hovertemplate="<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>",
            )
        )
        fig_shap.add_vline(x=0, line_color="#4E6357", line_width=1)
        fig_shap.update_layout(
            **PLOTLY_LAYOUT,
            xaxis=dict(
                title="Impact on flood probability (log-odds)",
                gridcolor="#1F4A32",
                zerolinecolor="#4E6357",
            ),
            yaxis=dict(gridcolor="#1F4A32"),
            margin=dict(t=10, b=10, l=10, r=10),
            height=420,
            showlegend=False,
        )
        st.plotly_chart(fig_shap, width="stretch")

    except Exception as e:
        st.warning(f"SHAP explanation unavailable: {e}")

else:
    # Fallback: weighted feature importance bar chart (no shap needed)
    booster = load_explainer_model()
    raw_imp = booster.feature_importances_

    ward_data = X_all.loc[ward_idx]
    kenya_mean = X_all.mean()
    deviation = (ward_data - kenya_mean) / (X_all.std() + 1e-9)
    contribution = raw_imp * deviation.values

    imp_df = pd.DataFrame(
        {
            "Feature": FEATURE_COLS,
            "Contribution": contribution,
            "Value": ward_data.values,
        }
    ).sort_values("Contribution")
    imp_df["Label"] = imp_df.apply(
        lambda r: f"{r['Feature'].replace('_', ' ').title()}  ({r['Value']:,.1f})",
        axis=1,
    )
    imp_df["Color"] = imp_df["Contribution"].apply(
        lambda v: "#C4622D" if v > 0 else "#3FA66B"
    )

    fig_imp = go.Figure(
        go.Bar(
            x=imp_df["Contribution"],
            y=imp_df["Label"],
            orientation="h",
            marker_color=imp_df["Color"],
            hovertemplate="<b>%{y}</b><br>Contribution: %{x:.4f}<extra></extra>",
        )
    )
    fig_imp.add_vline(x=0, line_color="#4E6357", line_width=1)
    fig_imp.update_layout(
        **PLOTLY_LAYOUT,
        xaxis=dict(
            title="Feature contribution (importance × deviation from Kenya avg)",
            gridcolor="#1F4A32",
        ),
        yaxis=dict(gridcolor="#1F4A32"),
        margin=dict(t=10, b=10, l=10, r=10),
        height=420,
        showlegend=False,
    )
    st.plotly_chart(fig_imp, width="stretch")
    st.caption("Install `shap` for exact SHAP values: `pip install shap`")
