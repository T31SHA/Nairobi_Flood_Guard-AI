"""Shared per-rerun state: sidebar controls, rainfall source, scored wards.

``build_shared_state`` runs once per rerun from app.py (before the selected
page executes) and stores everything pages need in st.session_state, so each
page stays a thin rendering layer.
"""

from __future__ import annotations

import streamlit as st

from app_lib.config import load_registry, registry_threshold
from app_lib.data import (
    add_risk_columns,
    get_open_meteo_ward_dataframe,
    load_data,
    load_model,
)
from app_lib.sms import render_sms_panel
from Utils.rainfall_fetcher import rainfall_summary

_STATE_KEY = "flood_guard_state"
SIM_KEY = "simulated_risk"  # {ward_name: probability} - in-memory only, never persisted


def render_simulator_panel(df) -> None:
    """'Simulate Scenario': dial up flood risk for a chosen ward and watch
    every page react (map, metrics, rerouting). Overrides model output purely
    in memory - never touches Data/floods.gpkg or the trained model."""
    with st.sidebar.expander("🧪 Simulate Scenario", expanded=False):
        st.caption(
            "Presenter tool: force one ward's flood probability and watch the "
            "map, metrics and rerouting react. In-memory only - nothing is "
            "written to data files or the model, and no SMS is sent."
        )
        nairobi_wards = sorted(
            df[df["county"].str.lower() == "nairobi"]["ward"].unique()
        )
        ward = st.selectbox("Ward to simulate", nairobi_wards, key="sim_ward")
        current = float(df.loc[df["ward"] == ward, "flood_prob"].iloc[0])
        prob = st.slider(
            "Simulated flood probability",
            min_value=0.0,
            max_value=1.0,
            value=max(0.8, round(current, 2)),
            step=0.05,
            key="sim_prob",
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Apply", use_container_width=True, key="sim_apply"):
                st.session_state[SIM_KEY] = {ward: prob}
        with c2:
            if st.button("Clear", use_container_width=True, key="sim_clear"):
                st.session_state.pop(SIM_KEY, None)


def apply_simulation(df) -> dict | None:
    """Overlay the simulated probabilities onto the scored frame. Returns the
    active simulation dict (or None)."""
    sim = st.session_state.get(SIM_KEY)
    if not sim:
        return None
    for ward, prob in sim.items():
        df.loc[df["ward"] == ward, "flood_prob"] = prob
    return sim


def get_state() -> dict:
    return st.session_state[_STATE_KEY]


def build_shared_state() -> dict:
    base_df = load_data()
    model = load_model()
    registry = load_registry()

    # -- Sidebar: rainfall source & threshold ---------------------------------
    with st.sidebar:
        st.markdown("### Rainfall Data")
        data_mode = st.radio(
            "Source",
            [
                "Live (Open-Meteo)",
                "Historical (Apr 2024)",
                "24hr Prediction",
                "48hr Prediction",
            ],
            help=(
                "Live mode uses the latest Open-Meteo rainfall. The 24hr and 48hr "
                "prediction modes roll forecast precipitation into the model's "
                "90-day, max-daily, and 7-day rainfall features before prediction."
            ),
        )
        forecast_horizon_hours = {
            "Live (Open-Meteo)": 0,
            "24hr Prediction": 24,
            "48hr Prediction": 48,
        }.get(data_mode)
        use_open_meteo = forecast_horizon_hours is not None
        use_live = forecast_horizon_hours == 0
        if use_open_meteo:
            if st.button("Refresh rainfall now", use_container_width=True):
                st.session_state["force_rainfall_refresh"] = True
                bust_key = f"rainfall_cache_bust_{forecast_horizon_hours}"
                st.session_state[bust_key] = st.session_state.get(bust_key, 0) + 1
            st.caption(
                "Open-Meteo rainfall applies to Nairobi's wards only (where route "
                "optimization and alerts operate). Other counties keep the "
                "historical Apr 2024 data. First fetch typically takes a few "
                "seconds; results are cached for 6 hours."
            )
        st.markdown("---")
        st.markdown("### Risk Threshold")
        default_threshold = round(registry_threshold(), 2)
        threshold = st.slider(
            "Flag wards above this probability as high-risk",
            min_value=0.01,
            max_value=0.90,
            value=default_threshold,
            step=0.01,
            format="%.2f",
        )
        st.caption(
            f"Default {default_threshold:.2f} is the trained operating point: "
            f"the highest-precision threshold with recall ≥ "
            f"{registry['recall_at_threshold']:.0%} under county-held-out "
            "validation (see model registry). Probabilities are calibrated, "
            "so lower values are meaningful during mild conditions."
        )

    # -- Apply rainfall source & generate predictions --------------------------
    force_refresh = st.session_state.pop("force_rainfall_refresh", False)
    cache_bust = (
        st.session_state.get(f"rainfall_cache_bust_{forecast_horizon_hours}", 0)
        if use_open_meteo
        else 0
    )
    rainfall_meta: dict = {"source": "historical", "label": "CHIRPS Feb-Apr 2024"}

    if use_open_meteo:
        try:
            spinner_label = (
                "Loading live rainfall..."
                if forecast_horizon_hours == 0
                else f"Loading {forecast_horizon_hours}hr forecast rainfall..."
            )
            with st.spinner(f"{spinner_label} (first fetch may take ~2 min)"):
                df, rainfall_meta = get_open_meteo_ward_dataframe(
                    cache_bust, force_refresh, int(forecast_horizon_hours)
                )
        except Exception as exc:
            st.warning(
                f"Could not fetch Open-Meteo rainfall ({exc}). "
                "Falling back to historical Apr 2024 data."
            )
            df = base_df.copy()
            rainfall_meta = {"source": "historical", "label": "CHIRPS Feb-Apr 2024"}
    else:
        df = base_df.copy()

    df = add_risk_columns(model, df)
    render_simulator_panel(df)
    sim = apply_simulation(df)
    if sim:
        from app_lib.theme import risk_label

        df["risk_label"], _ = zip(*df["flood_prob"].map(risk_label))
    nairobi = df[df["county"].str.lower() == "nairobi"].copy()

    n_critical = (df["flood_prob"] >= 0.70).sum()
    n_high = ((df["flood_prob"] >= threshold) & (df["flood_prob"] < 0.70)).sum()
    n_total = (df["flood_prob"] >= threshold).sum()

    with st.sidebar:
        st.markdown("---")
        st.markdown("### Kenya-Wide Summary")
        st.metric("Total Wards", len(df))
        st.metric("Critical Risk", int(n_critical))
        st.metric("High Risk", int(n_high))
        st.metric("Above Threshold", int(n_total))
        if use_open_meteo:
            st.caption(rainfall_summary(rainfall_meta))
        st.markdown("---")
        render_sms_panel(df, nairobi, threshold)
        st.markdown("---")
        rainfall_label = (
            rainfall_summary(rainfall_meta)
            if use_open_meteo
            else "Rainfall: CHIRPS Feb-Apr 2024"
        )
        oof = registry["metrics_spatial_oof"]
        st.markdown(
            f"<span style='font-size:0.65rem;color:#4E6357;'>Model: calibrated "
            f"XGBoost v{registry['version']} · county-held-out ROC AUC "
            f"{oof['roc_auc']:.2f}, recall {oof['recall']:.2f} · "
            f"Labels: UNOSAT Apr 2024 · Terrain: SRTM 90m · "
            f"{rainfall_label}</span>",
            unsafe_allow_html=True,
        )

    state = {
        "df": df,
        "nairobi": nairobi,
        "model": model,
        "registry": registry,
        "threshold": threshold,
        "use_open_meteo": use_open_meteo,
        "use_live": use_live,
        "forecast_horizon_hours": forecast_horizon_hours,
        "rainfall_meta": rainfall_meta,
        "simulation": sim,
        "n_critical": int(n_critical),
        "n_high": int(n_high),
        "n_total": int(n_total),
    }
    st.session_state[_STATE_KEY] = state
    if sim:
        for ward, prob in sim.items():
            st.warning(
                f"SIMULATION ACTIVE: **{ward}** forced to {prob:.0%} flood "
                "probability. Map, metrics and rerouting react; nothing is "
                "persisted and no SMS is sent. Clear it from the sidebar's "
                "'Simulate Scenario' panel."
            )
    return state
