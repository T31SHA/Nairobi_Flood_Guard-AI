"""Shared Africa's Talking SMS sender.

Extracted from app_lib/sms.py so the manual UI panel and the automated
threshold-crossing alerts (scripts/refresh_cache.py) call one function
instead of duplicating the AT integration. Streamlit-free: callers pass
credentials explicitly (the app resolves them from st.secrets, scripts from
environment variables).
"""

from __future__ import annotations

try:
    import africastalking

    AT_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only on minimal installs
    AT_AVAILABLE = False


class SmsConfigError(RuntimeError):
    """Raised when the AT SDK or credentials are missing - callers decide
    whether that is a hard error (manual panel) or a logged skip (scheduled
    alerts)."""


def send_sms(
    message: str,
    recipients: list[str],
    username: str | None = None,
    api_key: str | None = None,
) -> dict:
    """Send ``message`` to E.164 ``recipients``.

    Returns {"sent": [numbers], "failed": [{"number", "status"}], "raw": ...}
    parsed from the Africa's Talking response.
    """
    if not AT_AVAILABLE:
        raise SmsConfigError("africastalking not installed: pip install africastalking")
    if not api_key:
        raise SmsConfigError("AT_API_KEY is not configured")
    if not recipients:
        raise ValueError("recipients must not be empty")

    africastalking.initialize(username=username or "sandbox", api_key=api_key)
    response = africastalking.SMS.send(message=message, recipients=recipients)

    results = response.get("SMSMessageData", {}).get("Recipients", [])
    return {
        "sent": [r["number"] for r in results if r.get("status") == "Success"],
        "failed": [
            {"number": r.get("number"), "status": r.get("status")}
            for r in results
            if r.get("status") != "Success"
        ],
        "raw": response,
    }


def send_whatsapp(
    message: str,
    recipients: list[str],
    username: str | None = None,
    api_key: str | None = None,
) -> dict:
    """Send WhatsApp message via Africa's Talking (same account as SMS)."""
    if not AT_AVAILABLE:
        raise SmsConfigError("africastalking not installed: pip install africastalking")
    if not api_key:
        raise SmsConfigError("AT_API_KEY is not configured")
    if not recipients:
        raise ValueError("recipients must not be empty")

    africastalking.initialize(username=username or "sandbox", api_key=api_key)
    # Africa's Talking WhatsApp API (channel parameter on SMS.send in recent SDKs)
    try:
        response = africastalking.SMS.send(
            message=message, recipients=recipients, channel="WhatsApp"
        )
    except TypeError:
        # Older SDK without channel kwarg — fall back to SMS transport.
        response = africastalking.SMS.send(message=message, recipients=recipients)

    results = response.get("SMSMessageData", {}).get("Recipients", [])
    return {
        "sent": [r["number"] for r in results if r.get("status") == "Success"],
        "failed": [
            {"number": r.get("number"), "status": r.get("status")}
            for r in results
            if r.get("status") != "Success"
        ],
        "raw": response,
    }
