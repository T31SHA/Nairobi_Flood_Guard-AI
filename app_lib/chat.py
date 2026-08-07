"""Mlinzi AI assistant: chat rendering and system-prompt construction."""

from __future__ import annotations

import base64

import streamlit as st
from groq import Groq

from app_lib.config import REROUTING_CSV, get_secret
from app_lib.data import load_model_comparison, load_rerouting, model_comparison_path
from Utils.rainfall_fetcher import rainfall_summary


@st.cache_resource
def get_groq_client():
    return Groq(api_key=get_secret("GROQ_API_KEY"))


def render_message(role: str, content: str) -> None:
    """
    Render a chat message bubble with a clipboard icon that appears on hover.
    Content is base64-encoded to avoid all quote/backtick escaping issues.
    """
    b64 = base64.b64encode(content.encode("utf-8")).decode("utf-8")
    icon_color = "#8FA894"
    icon_hover = "#D4A24C"
    align = "flex-end" if role == "user" else "flex-start"
    bubble_bg = "#12301F" if role == "user" else "#0E2318"
    border_col = "#D4A24C" if role == "user" else "#1F4A32"
    avatar = "👤" if role == "user" else "🌦️"
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
                  font-family:'Space Mono',monospace; font-size:0.85rem;
                  color:#E8DFC8; line-height:1.6; white-space:pre-wrap;
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
    st.iframe(html, height=height)


def build_system_prompt(state: dict) -> str:
    """Assemble Mlinzi's system prompt from the current app state.

    Built only when a message is actually being sent to the LLM, rather than
    on every rerun of the assistant page.
    """
    df = state["df"]
    registry = state["registry"]
    threshold = state["threshold"]
    use_open_meteo = state["use_open_meteo"]
    rainfall_meta = state["rainfall_meta"]

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
    model_csv = model_comparison_path()
    if model_csv.exists():
        model_perf_context = (
            "\nHistorical model-family comparison (random-split, for context):\n"
            + load_model_comparison(model_csv).to_string(index=False)
        )
    oof = registry["metrics_spatial_oof"]
    model_perf_context += (
        "\n\nProduction model (calibrated XGBoost, county-held-out spatial CV):\n"
        f"  ROC AUC {oof['roc_auc']:.3f} · PR AUC {oof['pr_auc']:.3f} · "
        f"recall {oof['recall']:.3f} · precision {oof['precision']:.3f} · "
        f"F1 {oof['f1']:.3f} · Brier {oof['brier']:.3f}\n"
        f"  Operating threshold {registry['threshold']:.3f} "
        f"({registry['threshold_policy']})."
    )

    rerouting_context = ""
    if REROUTING_CSV.exists():
        r = load_rerouting()
        rerouting_context = (
            f"\nRerouting summary from the April 2024 event ({len(r)} rows):\n"
            + r.to_string(index=False)
            + "\n\nAggregate stats:"
            + f"\n  Average risk reduction : {r['risk_reduction'].mean():.3f}"
            + f"\n  Average extra time (min): {r['extra_time_min'].mean():.1f}"
            + f"\n  Routes with risk > 0   : {(r['risk_reduction'] > 0).sum()}"
        )

    return (
        "You are Mlinzi, an AI assistant for Nairobi Flood Guard - a data science project\n"
        "that predicts flood susceptibility across Kenya's 1,450 administrative wards and\n"
        "recommends alternative matatu routes during flood events. Your name means\n"
        "'guardian' or 'protector' in Swahili, which reflects your purpose.\n\n"
        "--- PROJECT OVERVIEW ---\n"
        "The prediction model is a calibrated XGBoost classifier trained on leakage-free\n"
        "ward-level features and validated with county-grouped spatial cross-validation:\n"
        "  Terrain  : elevation (mean, min, max, range in metres), slope, roughness, TWI proxy\n"
        "  Rainfall : cumulative 90-day (mm), max single-day (mm), recent 7-day (mm),\n"
        "             plus recency/intensity ratio features\n"
        "  Exposure : ward area and population density (2009 census)\n"
        "Probabilities are isotonic-calibrated, so they can be read as real frequencies.\n\n"
        f"--- ACTIVE RAINFALL SOURCE ---\n"
        f"{rainfall_summary(rainfall_meta) if use_open_meteo else 'Historical CHIRPS data from the April 2024 flood event.'}\n\n"
        "Key insight: flooding in Kenya is primarily terrain-driven at ward scale.\n"
        "Low-lying wards flood not because they receive more rain but because water drains\n"
        "into them from surrounding higher ground. Elevation features dominate predictions;\n"
        "rainfall adds marginal predictive value at this spatial resolution.\n\n"
        "--- CURRENT RISK SUMMARY ---\n"
        f"Total wards         : {len(df)}\n"
        f"High-risk (>= {threshold:.0%}) : {state['n_total']}\n"
        f"Critical risk (>= 70%): {state['n_critical']}\n\n"
        "Risk thresholds:\n"
        "  Low      : flood probability < 20%\n"
        "  Moderate : 20% - 45%\n"
        "  High     : 45% - 70%\n"
        "  Critical : >= 70%\n\n"
        "--- ALL NAIROBI DATA (top 100 by flood probability) ---\n"
        f"{wards_context}\n\n"
        "--- MODEL PERFORMANCE ---\n"
        f"{model_perf_context}\n\n"
        "--- ROUTE OPTIMIZATION ---\n"
        "The route optimization system:\n"
        "  - Uses calibrated flood probabilities to identify high-risk road segments\n"
        "  - Assigns flood cost = travel_time x (1 + alpha x flood_probability) and\n"
        "    computes a Pareto set of alternatives per affected route:\n"
        "    fastest (alpha=5), balanced (alpha=50), safest (alpha=1e6, blocks all\n"
        "    flood-touched roads)\n"
        "  - Risk metrics are travel-time-weighted (exposure), and each option reports\n"
        "    how many of the route's original stops it still serves within 300 m\n"
        "  - Outputs a GTFS-RT feed consumable by transit apps\n\n"
        f"{rerouting_context if rerouting_context else 'Rerouting data not available.'}\n\n"
        "--- INSTRUCTIONS ---\n"
        "- Be concise, factual, and actionable.\n"
        "- When asked about a specific ward, look it up in the ward data above and quote\n"
        "  its exact flood probability, risk level, and key terrain/rainfall figures.\n"
        "- When asked about routes, reference the rerouting summary above by route ID.\n"
        "- If asked which wards are most at risk, list the top entries from the ward data.\n"
        "- If asked about the model, explain the calibrated XGBoost pipeline, the spatial\n"
        "  cross-validation, and feature importance.\n"
        "- Do not make up data - all ward and route figures are provided above.\n"
        "- Respond in English unless the user writes in another language.\n"
    )
