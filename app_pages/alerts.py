"""Alert History page: the early-warning system's actual audit trail.

Read-only view over the alerts_sent table written by the scheduled cache
refresh (Utils/alerting.py). "Here is every alert the system has sent, and
why" is a stronger claim than "here is a risk map".
"""

import pandas as pd
import streamlit as st

from app_lib.state import get_state
from Utils import alert_store
from Utils.alert_verification import compute_verification_stats

state = get_state()

st.markdown(
    '<div class="section-header">Alert History</div>',
    unsafe_allow_html=True,
)
st.markdown(
    "When the scheduled refresh detects a ward **newly crossing** the "
    "early-warning threshold (or escalating into the critical band), every "
    "active subscriber for that ward or county is sent an SMS - and every "
    "decision is logged here. A ward that stays above threshold across runs "
    "alerts **once**, not repeatedly."
)

alerts = alert_store.list_alerts(limit=500)
subscribers = alert_store.list_subscribers(active_only=True)

c1, c2, c3, c4 = st.columns(4)
sent = [a for a in alerts if a["status"] == "sent"]
with c1:
    st.metric("Alerts Sent", len(sent))
with c2:
    st.metric("Wards Alerted", len({a["ward"] for a in sent if a["ward"]}))
with c3:
    st.metric("Active Subscribers", len(subscribers))
with c4:
    st.metric("Last Alert", alerts[0]["timestamp"][:16].replace("T", " ") + " UTC"
              if alerts else "—")

stats = compute_verification_stats()
st.markdown(
    '<div class="section-header" style="margin-top:1rem">Warning performance</div>',
    unsafe_allow_html=True,
)
v1, v2, v3 = st.columns(3)
with v1:
    pod = stats["pod"]
    st.metric("POD (vs field reports)", f"{pod:.0%}" if pod is not None else "—")
with v2:
    far = stats["far"]
    st.metric("FAR (vs field reports)", f"{far:.0%}" if far is not None else "—")
with v3:
    st.metric(
        "2026 retrospective overlap",
        f"{stats['retro_2026_ward_overlap']}/{stats['retro_2026_ward_total']}",
    )
st.caption(stats["caveat"])

if not alerts:
    st.info(
        "No alerts logged yet. The audit trail fills in when the scheduled "
        "refresh (`make refresh-cache`) sees a ward newly cross the "
        "threshold - subscribe a phone in the sidebar to receive them."
    )
    st.stop()

st.markdown(
    '<div class="section-header" style="margin-top:1.5rem">Timeline</div>',
    unsafe_allow_html=True,
)
display = pd.DataFrame(alerts)
display["timestamp"] = display["timestamp"].str[:16].str.replace("T", " ")
display["phone"] = display["phone"].map(alert_store.mask_phone)
display = display.rename(
    columns={
        "timestamp": "Time (UTC)",
        "ward": "Ward",
        "phone": "Recipient",
        "message": "Message",
        "status": "Status",
    }
)[["Time (UTC)", "Ward", "Recipient", "Status", "Message"]]
st.dataframe(display, width="stretch", hide_index=True)

st.caption(
    "Statuses: **sent** (delivered to Africa's Talking), **failed** (AT "
    "rejected the send), **no_credentials** (crossing detected but no AT API "
    "key configured), **no_subscribers** (crossing detected, nobody "
    "subscribed to that ward/county). Recipient numbers are masked; full "
    "numbers never leave the SMS send path."
)

# -- Ground-truth loop: field reports become candidate retraining labels ----
st.markdown(
    '<div class="section-header" style="margin-top:1.5rem">Field Reports</div>',
    unsafe_allow_html=True,
)
st.markdown(
    "Submitted via `POST /reports` or by SMS to the Africa's Talking "
    "webhook. These accumulate as **candidate labels for retraining** - the "
    "scarcest asset in this system."
)
reports = alert_store.list_reports(limit=100)
if not reports:
    st.info("No field reports yet.")
else:
    rep = pd.DataFrame(reports)
    rep["created_at"] = rep["created_at"].str[:16].str.replace("T", " ")
    rep["phone"] = rep["phone"].map(alert_store.mask_phone)
    cols = {
        "created_at": "Time (UTC)",
        "source": "Source",
        "ward": "Ward",
        "county": "County",
        "text": "Report",
        "phone": "From",
    }
    st.dataframe(
        rep[list(cols)].rename(columns=cols), width="stretch", hide_index=True
    )
