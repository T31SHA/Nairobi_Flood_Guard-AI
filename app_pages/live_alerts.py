"""Live Alerts — public, no-login view of the alert feed."""

import streamlit as st

from Utils import alert_store
from Utils.alert_verification import compute_verification_stats

st.markdown(
    '<div class="section-header">Live Alerts</div>',
    unsafe_allow_html=True,
)
st.markdown(
    "Public alert feed — same data as `GET /alerts/feed` (JSON) and "
    "`GET /alerts/cap/feed` (CAP XML). Designed for pick-up by partners "
    "including Kenya Red Cross's SMS systems. **Complements** KMD's official "
    "advisories; does not replace them."
)

alerts = alert_store.list_alerts(limit=100)
sent = [a for a in alerts if a["status"] == "sent"]

if not sent:
    st.info(
        "No alerts issued yet. When the scheduled refresh detects a ward "
        "newly crossing the threshold, alerts appear here and in the CAP feed."
    )
else:
    for a in sent[:20]:
        sev = a.get("severity") or "—"
        ch = a.get("channel") or "sms"
        st.markdown(
            f"**{a['timestamp'][:16].replace('T', ' ')} UTC** · "
            f"{a.get('ward', '—')} · `{sev}` · {ch}"
        )
        st.caption(a["message"])
        st.divider()

st.markdown(
    '<div class="section-header" style="margin-top:1.5rem">Feed endpoints</div>',
    unsafe_allow_html=True,
)
st.code(
    "GET /alerts/feed          # JSON\n"
    "GET /alerts/feed/rss      # RSS 2.0\n"
    "GET /alerts/cap/feed      # CAP 1.2 XML",
    language="text",
)

stats = compute_verification_stats()
st.markdown(
    '<div class="section-header" style="margin-top:1.5rem">Warning performance</div>',
    unsafe_allow_html=True,
)
c1, c2, c3 = st.columns(3)
with c1:
    pod = stats["pod"]
    st.metric("POD (field reports)", f"{pod:.0%}" if pod is not None else "—")
with c2:
    far = stats["far"]
    st.metric("FAR (field reports)", f"{far:.0%}" if far is not None else "—")
with c3:
    st.metric(
        "2026 retrospective overlap",
        f"{stats['retro_2026_ward_overlap']}/{stats['retro_2026_ward_total']}",
    )
st.caption(stats["caveat"])
