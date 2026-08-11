"""SMS early-warning sidebar panel (Africa's Talking) + alert opt-in form."""

from __future__ import annotations

import streamlit as st

from app_lib.config import get_secret
from Utils import alert_store
from Utils.sms_sender import AT_AVAILABLE, SmsConfigError, send_sms


def render_subscribe_panel(df) -> None:
    """Opt-in form: 'Get SMS alerts for your ward'. Writes to the shared
    subscribers table (same SQLite store the API exposes via
    POST /subscribers for external integrations)."""
    with st.expander("📳 Get SMS alerts for your ward", expanded=False):
        counties = sorted(df["county"].unique())
        scope = st.radio(
            "Alert scope",
            ["A specific ward", "A whole county"],
            horizontal=True,
            key="subscribe_scope",
        )
        if scope == "A specific ward":
            default_idx = 0
            county = st.selectbox(
                "County", counties,
                index=counties.index("Nairobi") if "Nairobi" in counties else 0,
                key="subscribe_county",
            )
            ward_choices = sorted(df[df["county"] == county]["ward"].unique())
            target = st.selectbox("Ward", ward_choices, index=default_idx,
                                  key="subscribe_ward")
        else:
            target = st.selectbox(
                "County", counties,
                index=counties.index("Nairobi") if "Nairobi" in counties else 0,
                key="subscribe_county_wide",
            )
        phone = st.text_input(
            "Phone number (E.164, e.g. +254712345678)", key="subscribe_phone"
        )
        language = st.selectbox(
            "Alert language",
            ["English", "Kiswahili"],
            key="subscribe_language",
        )
        lang_code = "sw" if language == "Kiswahili" else "en"
        if st.button("Subscribe", use_container_width=True):
            try:
                alert_store.add_subscriber(phone, target, language=lang_code)
                st.success(
                    f"Subscribed {alert_store.mask_phone(phone)} to alerts for "
                    f"**{target}**. You'll be SMS'd when its flood risk newly "
                    "crosses the alert threshold."
                )
            except ValueError as exc:
                st.error(str(exc))
        n_active = len(alert_store.list_subscribers())
        st.caption(
            f"{n_active} active subscription(s). Alerts are sent automatically "
            "by the scheduled cache refresh when a ward's risk newly crosses "
            "the threshold - never repeatedly while it stays high."
        )


def render_sms_panel(df, nairobi, threshold) -> None:
    st.markdown("### 🔔 SMS Early Warning")
    render_subscribe_panel(df)

    with st.expander("Send flood alert via SMS (manual)", expanded=False):
        if not AT_AVAILABLE:
            st.error(
                "africastalking not installed.\n\n" "Run: `pip install africastalking`"
            )
            return

        import africastalking

        if st.button("💰 Check AT account balance"):
            try:
                api_key = get_secret("AT_API_KEY")
                if not api_key:
                    raise KeyError("AT_API_KEY")
                africastalking.initialize(
                    username=get_secret("AT_USERNAME", "sandbox"),
                    api_key=api_key,
                )
                app_data = africastalking.Application.fetch_application_data()
                st.info(f"Balance: {app_data['UserData']['balance']}")
            except Exception as e:
                st.error(f"Balance check failed: {e}")

        # ── Recipient input ──────────────────────────────────────────────
        sms_input_mode = st.radio(
            "Recipients",
            ["Enter numbers manually", "Auto (all critical wards)"],
            label_visibility="visible",
            horizontal=True,
        )

        if sms_input_mode == "Enter numbers manually":
            raw_numbers = st.text_area(
                "Phone numbers (one per line, include country code)",
                placeholder="+254712345678\n+254722345678",
                height=100,
            )
            recipient_list = [
                n.strip()
                for n in raw_numbers.splitlines()
                if n.strip().startswith("+")
            ]
        else:
            # Pull county from Ward Lookup selection if available,
            # otherwise default to Nairobi
            target_county = st.selectbox(
                "County to alert",
                sorted(df["county"].unique()),
                index=(
                    list(sorted(df["county"].unique())).index("Nairobi")
                    if "Nairobi" in df["county"].unique()
                    else 0
                ),
            )
            # In production: replace this with a real contacts DB keyed by county.
            # Here we show the count of wards that would be notified so the
            # operator understands the scope before sending.
            critical_wards = df[
                (df["county"] == target_county) & (df["flood_prob"] >= 0.70)
            ][["ward", "flood_prob"]]
            st.caption(
                f"{len(critical_wards)} critical ward(s) in {target_county}. "
                "Connect a contacts database to auto-populate numbers."
            )
            raw_numbers = st.text_area(
                "Phone numbers for this county (one per line)",
                placeholder="+254712345678",
                height=80,
            )
            recipient_list = [
                n.strip()
                for n in raw_numbers.splitlines()
                if n.strip().startswith("+")
            ]

        # ── Message composer ─────────────────────────────────────────────
        st.markdown("**Alert message**")

        # Build a smart default message from live model data
        top_wards = (
            nairobi[nairobi["flood_prob"] >= threshold]
            .nlargest(3, "flood_prob")[["ward", "flood_prob"]]
            .apply(lambda r: f"{r['ward']} ({r['flood_prob']:.0%})", axis=1)
            .tolist()
        )
        default_msg = (
            f"FLOOD ALERT: Nairobi Flood Guard\n"
            f"High-risk wards: {', '.join(top_wards) if top_wards else 'None at current threshold'}.\n"
            f"Threshold: {threshold:.0%}. Avoid low-lying areas & flooded routes.\n"
            f"Details: https://nairobi-flood-guard.streamlit.app"
        )
        sms_body = st.text_area(
            "Edit message before sending",
            value=default_msg,
            height=140,
        )
        char_count = len(sms_body)
        sms_pages = max(1, -(-char_count // 160))  # ceiling division
        st.caption(
            f"{char_count} characters · {sms_pages} SMS page(s) · "
            f"{len(recipient_list)} recipient(s)"
        )

        # ── Send button ──────────────────────────────────────────────────
        send_disabled = len(recipient_list) == 0 or len(sms_body.strip()) == 0
        if st.button(
            "📤 Send Alert",
            disabled=send_disabled,
            use_container_width=True,
        ):
            try:
                result = send_sms(
                    sms_body,
                    recipient_list,
                    username=get_secret("AT_USERNAME", "NgundoMuithya"),
                    api_key=get_secret("AT_API_KEY"),
                )
                success, failed = result["sent"], result["failed"]

                if success:
                    st.success(f"✅ Sent to {len(success)} recipient(s).")
                if failed:
                    st.warning(
                        "⚠️ Failed for:\n"
                        + "\n".join(
                            f"- {r.get('number')}: **{r.get('status')}**"
                            for r in failed
                        )
                    )
                    with st.expander("Raw AT response (debug)"):
                        st.json(result["raw"])

                # Log to session state so operator can review sends.
                # Trimmed to the last 5 - that's all the UI ever shows,
                # so a long session shouldn't keep every send in memory.
                if "sms_log" not in st.session_state:
                    st.session_state.sms_log = []
                st.session_state.sms_log.append(
                    {
                        "recipients": recipient_list,
                        "sent": len(success),
                        "failed": len(failed),
                        "message": sms_body[:80] + "...",
                    }
                )
                st.session_state.sms_log = st.session_state.sms_log[-5:]

            except SmsConfigError:
                st.error(
                    "AT_API_KEY not found in Streamlit secrets.\n\n"
                    "Add it to `.streamlit/secrets.toml`:\n"
                    '```\nAT_USERNAME = "your_username"\nAT_API_KEY  = "your_key"\n```'
                )
            except Exception as e:
                st.error(f"SMS send failed: {e}")

        # ── Send log ─────────────────────────────────────────────────────
        if st.session_state.get("sms_log"):
            st.markdown("**Send log (this session)**")
            for entry in reversed(st.session_state.sms_log[-5:]):
                st.caption(
                    f"✉ {entry['sent']} sent · {entry['failed']} failed: "
                    f"\"{entry['message']}\""
                )
