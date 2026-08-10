"""Visual theme: CSS injection, header banner, and risk display helpers."""

from __future__ import annotations

import pandas as pd
import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Space+Mono:wght@400;700&display=swap');

:root {
    --ground: #07110D; --panel: #0E2318; --panel-raised: #12301F;
    --line: #1F4A32; --line-soft: #17321F;
    --text: #E8DFC8; --text-dim: #8FA894; --text-faint: #4E6357;
    --accent: #D4A24C;
    --safe: #3FA66B; --moderate: #D4A24C; --high: #C4622D; --critical: #8B2E2E;
}

html, body, [class*="css"] { font-family: 'Space Mono', monospace; color: var(--text); }
.stApp { background: var(--ground); }
h1, h2, h3, h4 { font-family: 'Fraunces', serif !important; letter-spacing: -0.01em; font-weight: 600; }

/* Contour-line texture, reused wherever the terrain motif appears */
.contour-field {
    background-image:
        repeating-radial-gradient(ellipse 140% 100% at 15% 120%,
            transparent 0, transparent 22px, rgba(232,223,200,0.035) 23px, transparent 24px);
}

.header-banner {
    background: linear-gradient(180deg, #0B1F14 0%, #07110D 100%);
    border-bottom: 1px solid var(--line);
    padding: 2.75rem 2.5rem 2rem;
    margin: -1rem -1rem 2rem -1rem;
    position: relative; overflow: hidden;
}
.header-banner::before {
    content: ''; position: absolute; inset: 0;
    background-image:
        repeating-radial-gradient(ellipse 160% 120% at 88% 140%,
            transparent 0px, transparent 26px, rgba(212,162,76,0.05) 27px, transparent 28px),
        repeating-radial-gradient(ellipse 160% 120% at 88% 140%,
            transparent 0px, transparent 52px, rgba(232,223,200,0.04) 53px, transparent 54px);
    pointer-events: none;
}
.header-eyebrow {
    font-family: 'Space Mono', monospace; font-size: 0.68rem; color: var(--accent);
    letter-spacing: 0.22em; text-transform: uppercase; margin-bottom: 0.6rem;
    display: flex; align-items: center; gap: 0.5rem;
}
.header-eyebrow::before {
    content: ''; width: 6px; height: 6px; border-radius: 50%;
    background: var(--accent); box-shadow: 0 0 8px var(--accent);
}
.header-title {
    font-family: 'Fraunces', serif !important;
    font-size: 3rem; font-weight: 600; font-style: normal;
    color: var(--text); letter-spacing: -0.02em; margin: 0; line-height: 1.02;
    position: relative;
}
.header-subtitle {
    font-family: 'Space Mono', monospace; font-size: 0.76rem; color: var(--text-dim);
    margin-top: 0.65rem; letter-spacing: 0.08em; max-width: 640px; line-height: 1.6;
}
.badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.3rem 0.8rem; border-radius: 100px;
    font-family: 'Space Mono', monospace; font-size: 0.72rem; font-weight: 700;
    letter-spacing: 0.06em; text-transform: uppercase;
}
.badge::before { content: ''; width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
.badge-low      { background: rgba(63,166,107,0.12);  color: var(--safe);     border: 1px solid rgba(63,166,107,0.4); }
.badge-moderate { background: rgba(212,162,76,0.12);  color: var(--moderate); border: 1px solid rgba(212,162,76,0.4); }
.badge-high     { background: rgba(196,98,45,0.14);   color: var(--high);     border: 1px solid rgba(196,98,45,0.45); }
.badge-critical { background: rgba(139,46,46,0.18);   color: #E8A0A0;         border: 1px solid rgba(139,46,46,0.6); }

.metric-card {
    background: var(--panel); border: 1px solid var(--line-soft);
    border-left: 2px solid var(--accent); border-radius: 3px;
    padding: 1rem 1.2rem; margin-bottom: 0.75rem;
    transition: border-color 0.15s ease, background 0.15s ease;
}
.metric-card:hover { background: var(--panel-raised); border-left-color: var(--text); }
.metric-label {
    font-size: 0.65rem; color: var(--text-dim);
    letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.3rem;
}
.metric-value {
    font-family: 'Fraunces', serif; font-size: 1.7rem;
    font-weight: 600; color: var(--text); line-height: 1;
}
.metric-unit { font-family: 'Space Mono', monospace; font-size: 0.68rem; color: var(--text-faint); margin-left: 0.35rem; }

.section-header {
    font-family: 'Fraunces', serif; font-size: 1.35rem; font-weight: 600;
    color: var(--text); border-bottom: 1px solid var(--line);
    padding-bottom: 0.6rem; margin-bottom: 1.1rem; letter-spacing: -0.01em;
    display: flex; align-items: baseline; justify-content: space-between;
}

section[data-testid="stSidebar"] {
    background: #06100B; border-right: 1px solid var(--line-soft);
}
section[data-testid="stSidebar"] h3 {
    font-family: 'Space Mono', monospace !important; font-size: 0.7rem !important;
    font-weight: 700 !important; color: var(--accent) !important;
    letter-spacing: 0.16em !important; text-transform: uppercase !important;
    margin-bottom: 0.75rem !important;
}
[data-testid="stSidebarCollapseButton"] { display: none !important; }
.material-symbols-rounded, [data-testid="stIconMaterial"] {
    font-family: 'Material Symbols Rounded' !important;
}

.ward-panel {
    background: linear-gradient(160deg, var(--panel) 0%, var(--ground) 130%);
    border: 1px solid var(--line); border-radius: 4px;
    padding: 1.5rem 1.6rem; margin-top: 1rem; position: relative; overflow: hidden;
}
.ward-panel::after {
    content: ''; position: absolute; inset: 0;
    background-image: repeating-radial-gradient(ellipse 150% 130% at 105% 130%,
        transparent 0, transparent 20px, rgba(212,162,76,0.045) 21px, transparent 22px);
    pointer-events: none;
}
.ward-name {
    font-family: 'Fraunces', serif; font-size: 1.6rem;
    font-weight: 600; color: var(--text); margin-bottom: 0.25rem; position: relative;
}
.ward-meta {
    font-family: 'Space Mono', monospace; font-size: 0.7rem; color: var(--text-dim);
    letter-spacing: 0.06em; position: relative;
}

.stSelectbox label, .stSlider label, .stTextInput label, .stTextArea label, .stRadio label {
    font-family: 'Space Mono', monospace !important; font-size: 0.72rem !important;
    color: var(--text-dim) !important; letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}
div[data-testid="stMetric"] {
    background: var(--panel); border: 1px solid var(--line-soft);
    border-radius: 3px; padding: 0.75rem 1rem;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-family: 'Fraunces', serif; color: var(--text);
}
div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
    font-family: 'Space Mono', monospace; font-size: 0.68rem;
    letter-spacing: 0.08em; text-transform: uppercase; color: var(--text-dim);
}

.route-stat-card {
    background: var(--panel); border: 1px solid var(--line-soft);
    border-radius: 3px; padding: 1.1rem 1.2rem; text-align: center;
    border-top: 2px solid var(--line);
}
.route-stat-label {
    font-size: 0.63rem; color: var(--text-dim);
    letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.4rem;
}
.route-stat-value {
    font-family: 'Fraunces', serif; font-size: 1.5rem;
    font-weight: 600; color: var(--text);
}

/* Buttons */
.stButton > button, .stFormSubmitButton > button {
    background: var(--panel-raised) !important; color: var(--text) !important;
    border: 1px solid var(--line) !important; border-radius: 3px !important;
    font-family: 'Space Mono', monospace !important; font-size: 0.75rem !important;
    letter-spacing: 0.06em !important; text-transform: uppercase !important;
    transition: border-color 0.15s ease, background 0.15s ease !important;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
    border-color: var(--accent) !important; background: #16351F !important; color: var(--accent) !important;
}

/* Inputs, sliders, selects */
.stSelectbox [data-baseweb="select"] > div, .stTextInput input, .stTextArea textarea {
    background: var(--panel) !important; border-color: var(--line-soft) !important;
    color: var(--text) !important; font-family: 'Space Mono', monospace !important;
}
.stSlider [data-baseweb="slider"] div[role="slider"] { background: var(--accent) !important; }
div[data-baseweb="slider"] > div > div { background: var(--line) !important; }
div[data-baseweb="slider"] > div > div > div { background: var(--accent) !important; }

/* Expanders */
.streamlit-expanderHeader, div[data-testid="stExpander"] summary {
    background: var(--panel) !important; border: 1px solid var(--line-soft) !important;
    font-family: 'Space Mono', monospace !important; color: var(--text) !important;
}

/* Dataframe */
div[data-testid="stDataFrame"] { border: 1px solid var(--line-soft); border-radius: 3px; }

/* Map frame - the basemap tiles run cooler than the rest of the UI;
   a deliberate border makes that a frame rather than a seam. */
iframe[title="streamlit_folium.st_folium"] {
    border: 1px solid var(--line) !important;
    border-radius: 4px !important;
}

hr { border-top: 1px solid var(--line-soft) !important; }

/* Narrow viewports: the fixed-rem banner/type scale overflows phones, so
   step it down. Judges and audience frequently glance at the app on one. */
@media (max-width: 768px) {
    .header-banner { padding: 1.5rem 1rem 1.25rem; margin: -1rem -1rem 1.25rem; }
    .header-title { font-size: 1.9rem; }
    .header-subtitle { font-size: 0.68rem; }
    .metric-value { font-size: 1.25rem; }
    .route-stat-value { font-size: 1.1rem; }
    .route-stat-card { padding: 0.7rem 0.5rem; }
    .section-header { font-size: 1.05rem; }
    div[data-testid="stMetric"] { padding: 0.5rem 0.6rem; }
}
</style>
"""


def inject_theme() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def render_header() -> None:
    st.markdown(
        """
<div class="header-banner">
    <div class="header-eyebrow">Kenya &middot; 1,450 wards monitored</div>
    <div class="header-title">Nairobi Flood Guard</div>
    <div class="header-subtitle">Water follows elevation, not rainfall alone. This model reads
    the terrain each ward sits in &mdash; and where the surrounding highland will send the
    water next &mdash; to flag risk before it arrives, and to route matatus around it.</div>
</div>
""",
        unsafe_allow_html=True,
    )


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
        return "#8B2E2E"
    if prob >= 0.45:
        return "#C4622D"
    if prob >= 0.20:
        return "#D4A24C"
    return "#3FA66B"


def delta_color(delta: float) -> str:
    """Diverging color for live-vs-historical flood_prob shift. Buckets rather
    than a continuous scale, since a handful of pp of real movement should
    read clearly rather than blur into a gradient."""
    if delta >= 0.03:
        return "#C4622D"  # risk rose meaningfully
    if delta >= 0.01:
        return "#D4A24C"  # risk rose slightly
    if delta <= -0.03:
        return "#2E7D9E"  # risk fell meaningfully
    if delta <= -0.01:
        return "#5FA8C4"  # risk fell slightly
    return "#1F4A32"  # negligible change


def normalize(col: str, df: pd.DataFrame) -> pd.Series:
    """Min-max normalise a DataFrame column to [0, 1]."""
    mn, mx = df[col].min(), df[col].max()
    return (df[col] - mn) / (mx - mn + 1e-9)


def highlight_best(s: pd.Series) -> list[str]:
    """Highlight the highest value in each column of the metrics table."""
    is_best = s == s.max()
    return [
        "background-color: #12301F; color: #3FA66B; font-weight:600" if v else ""
        for v in is_best
    ]


PLOTLY_LAYOUT = dict(
    paper_bgcolor="#07110D",
    plot_bgcolor="#0E2318",
    font_color="#E8DFC8",
    font_family="Space Mono",
)
