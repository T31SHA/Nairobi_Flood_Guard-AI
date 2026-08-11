"""
Nairobi Flood Guard - Streamlit UI entrypoint.

Run with: streamlit run app.py

This file is a thin router: shared state (theme, sidebar controls, rainfall
source, scored ward predictions) is built once per rerun in
``app_lib.state.build_shared_state``, then the selected page renders from it.
Pages live in ``app_pages/``; shared code lives in ``app_lib/``.
"""

import warnings

import streamlit as st

from app_lib.state import build_shared_state
from app_lib.theme import inject_theme, render_header

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Nairobi Flood Guard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_theme()
render_header()

pages = [
    st.Page("app_pages/dashboard.py", title="Flood Risk Dashboard", default=True),
    st.Page("app_pages/ward_lookup.py", title="Ward Lookup"),
    st.Page("app_pages/route_optimization.py", title="Route Optimization"),
    st.Page("app_pages/live_alerts.py", title="Live Alerts"),
    st.Page("app_pages/alerts.py", title="Alert History"),
    st.Page("app_pages/model_card.py", title="Model Card"),
    st.Page("app_pages/ai_assistant.py", title="AI Assistant"),
]
nav = st.navigation(pages)

build_shared_state()

nav.run()
