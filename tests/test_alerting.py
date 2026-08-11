"""Threshold-crossing alert tests: synthetic scored wards, tmp SQLite store,
injected fake sender - no network, no real model."""

import pandas as pd
import pytest

from Utils import alert_store
from Utils.alerting import (
    build_alert_message,
    find_threshold_crossings,
    match_subscribers,
    process_alerts,
    ward_key,
)
from Utils.sms_sender import SmsConfigError

THRESHOLD = 0.30


def scored_df(probs: dict[str, float], county: str = "Nairobi") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ward": list(probs),
            "county": [county] * len(probs),
            "flood_prob": list(probs.values()),
        }
    )


@pytest.fixture
def store(tmp_path):
    return {
        "db": tmp_path / "alerts.db",
        "snapshot": tmp_path / "last_scored.json",
    }


def fake_sender_factory(calls, fail=False):
    def _send(message, recipients, username=None, api_key=None):
        calls.append({"message": message, "recipients": recipients})
        if fail:
            return {"sent": [], "failed": [{"number": r, "status": "Error"}
                                           for r in recipients]}
        return {"sent": list(recipients), "failed": []}

    return _send


def subscribe(store, phone="+254712345678", target="Kibera"):
    alert_store.add_subscriber(phone, target, db_path=store["db"])


class TestThresholdCrossingLifecycle:
    """The Phase 2 acceptance scenario: below -> above -> still above."""

    def test_crossing_fires_exactly_once(self, store):
        subscribe(store)
        calls = []
        sender = fake_sender_factory(calls)

        # Run 1 (baseline): ward below threshold -> no alert, snapshot taken.
        s1 = process_alerts(
            scored_df({"Kibera": 0.10}), THRESHOLD,
            snapshot_path=store["snapshot"], db_path=store["db"], sender=sender,
        )
        assert s1["crossings"] == [] and len(calls) == 0

        # Run 2: ward crosses above -> exactly one alert, logged once.
        s2 = process_alerts(
            scored_df({"Kibera": 0.55}), THRESHOLD,
            snapshot_path=store["snapshot"], db_path=store["db"], sender=sender,
        )
        assert len(s2["crossings"]) == 1 and s2["crossings"][0].kind == "new"
        assert len(calls) == 1
        assert calls[0]["recipients"] == ["+254712345678"]
        assert "Kibera" in calls[0]["message"]
        alerts = alert_store.list_alerts(db_path=store["db"])
        assert len(alerts) == 1 and alerts[0]["status"] == "sent"
        assert alerts[0]["ward"] == "Kibera"

        # Run 3: still above threshold -> zero additional alerts.
        s3 = process_alerts(
            scored_df({"Kibera": 0.60}), THRESHOLD,
            snapshot_path=store["snapshot"], db_path=store["db"], sender=sender,
        )
        assert s3["crossings"] == [] and len(calls) == 1
        assert len(alert_store.list_alerts(db_path=store["db"])) == 1

    def test_escalation_into_critical_band_re_alerts(self, store):
        subscribe(store)
        calls = []
        sender = fake_sender_factory(calls)
        process_alerts(scored_df({"Kibera": 0.10}), THRESHOLD,
                       snapshot_path=store["snapshot"], db_path=store["db"],
                       sender=sender)
        process_alerts(scored_df({"Kibera": 0.50}), THRESHOLD,
                       snapshot_path=store["snapshot"], db_path=store["db"],
                       sender=sender)
        s3 = process_alerts(scored_df({"Kibera": 0.75}), THRESHOLD,
                            snapshot_path=store["snapshot"], db_path=store["db"],
                            sender=sender)
        assert [c.kind for c in s3["crossings"]] == ["escalation"]
        assert len(calls) == 2
        assert "CRITICAL" in calls[1]["message"]

        # Staying critical does not re-alert.
        s4 = process_alerts(scored_df({"Kibera": 0.80}), THRESHOLD,
                            snapshot_path=store["snapshot"], db_path=store["db"],
                            sender=sender)
        assert s4["crossings"] == [] and len(calls) == 2

    def test_first_run_is_baseline_by_default(self, store):
        subscribe(store)
        calls = []
        s = process_alerts(
            scored_df({"Kibera": 0.90}), THRESHOLD,
            snapshot_path=store["snapshot"], db_path=store["db"],
            sender=fake_sender_factory(calls),
        )
        assert s["crossings"] == [] and len(calls) == 0

    def test_alert_on_baseline_override_alerts_currently_high_wards(self, store, tmp_path):
        subscribe(store)
        calls = []
        fresh = {"snapshot": tmp_path / "fresh.json", "db": tmp_path / "fresh.db"}
        alert_store.add_subscriber("+254712345678", "Kibera", db_path=fresh["db"])
        s = process_alerts(
            scored_df({"Kibera": 0.90, "Ruiru": 0.10}), THRESHOLD,
            snapshot_path=fresh["snapshot"], db_path=fresh["db"],
            sender=fake_sender_factory(calls), alert_on_baseline=True,
        )
        assert len(s["crossings"]) == 1 and len(calls) == 1
        assert "was" not in calls[0]["message"]  # no baseline prob to quote

    def test_county_subscription_matches_ward_crossing(self, store):
        subscribe(store, target="Nairobi")  # whole-county subscription
        calls = []
        process_alerts(scored_df({"Kibera": 0.10}), THRESHOLD,
                       snapshot_path=store["snapshot"], db_path=store["db"],
                       sender=fake_sender_factory(calls))
        process_alerts(scored_df({"Kibera": 0.50}), THRESHOLD,
                       snapshot_path=store["snapshot"], db_path=store["db"],
                       sender=fake_sender_factory(calls))
        assert len(calls) == 1

    def test_crossing_without_subscribers_is_logged_not_sent(self, store):
        calls = []
        process_alerts(scored_df({"Kibera": 0.10}), THRESHOLD,
                       snapshot_path=store["snapshot"], db_path=store["db"],
                       sender=fake_sender_factory(calls))
        process_alerts(scored_df({"Kibera": 0.50}), THRESHOLD,
                       snapshot_path=store["snapshot"], db_path=store["db"],
                       sender=fake_sender_factory(calls))
        assert len(calls) == 0
        alerts = alert_store.list_alerts(db_path=store["db"])
        assert len(alerts) == 1 and alerts[0]["status"] == "no_subscribers"

    def test_missing_credentials_logged_and_not_retried(self, store):
        subscribe(store)

        def no_creds(message, recipients, username=None, api_key=None):
            raise SmsConfigError("AT_API_KEY is not configured")

        process_alerts(scored_df({"Kibera": 0.10}), THRESHOLD,
                       snapshot_path=store["snapshot"], db_path=store["db"],
                       sender=no_creds)
        s = process_alerts(scored_df({"Kibera": 0.50}), THRESHOLD,
                           snapshot_path=store["snapshot"], db_path=store["db"],
                           sender=no_creds)
        assert s["skipped"] == 1
        alerts = alert_store.list_alerts(db_path=store["db"])
        assert [a["status"] for a in alerts] == ["no_credentials"]
        # Snapshot advanced despite the failed send: no later re-alert storm
        # once credentials appear.
        calls = []
        process_alerts(scored_df({"Kibera": 0.55}), THRESHOLD,
                       snapshot_path=store["snapshot"], db_path=store["db"],
                       sender=fake_sender_factory(calls))
        assert len(calls) == 0


class TestCrossingDetection:
    def test_ward_key_disambiguates_duplicate_ward_names(self):
        assert ward_key("Kibera", "Nairobi") != ward_key("Kibera", "Kiambu")

    def test_downward_movement_never_alerts(self, store):
        subscribe(store)
        calls = []
        process_alerts(scored_df({"Kibera": 0.80}), THRESHOLD,
                       snapshot_path=store["snapshot"], db_path=store["db"],
                       sender=fake_sender_factory(calls),
                       alert_on_baseline=True)
        calls.clear()
        s = process_alerts(scored_df({"Kibera": 0.10}), THRESHOLD,
                           snapshot_path=store["snapshot"], db_path=store["db"],
                           sender=fake_sender_factory(calls))
        assert s["crossings"] == [] and len(calls) == 0

    def test_find_threshold_crossings_ignores_unknown_wards(self):
        scored = scored_df({"NewWard": 0.99})
        assert find_threshold_crossings({}, scored, THRESHOLD) == []
        prev = {"wards": {ward_key("NewWard", "Nairobi"): {"prob": 0.1}}}
        crossings = find_threshold_crossings(prev, scored, THRESHOLD)
        assert len(crossings) == 1 and crossings[0].kind == "new"

    def test_match_subscribers_case_insensitive(self):
        c = find_threshold_crossings(
            {"wards": {ward_key("Kibera", "Nairobi"): {"prob": 0.1}}},
            scored_df({"Kibera": 0.5}), THRESHOLD,
        )
        pairs = match_subscribers(c, [{"phone": "+2541", "ward_or_county": "kibera"}])
        assert len(pairs) == 1

    def test_message_contents(self):
        c = find_threshold_crossings(
            {"wards": {ward_key("Kibera", "Nairobi"): {"prob": 0.10}}},
            scored_df({"Kibera": 0.55}), THRESHOLD,
        )[0]
        msg = build_alert_message(c, THRESHOLD)
        assert "Kibera" in msg and "55%" in msg and "30%" in msg


class TestSubscriberStore:
    def test_phone_validation(self):
        assert alert_store.valid_phone("+254712345678")
        assert not alert_store.valid_phone("0712345678")
        assert not alert_store.valid_phone("+254")
        assert not alert_store.valid_phone("hello")

    def test_resubscribe_is_idempotent(self, store):
        alert_store.add_subscriber("+254712345678", "Kibera", db_path=store["db"])
        alert_store.add_subscriber("+254712345678", "Kibera", db_path=store["db"])
        subs = alert_store.list_subscribers(db_path=store["db"])
        assert len(subs) == 1

    def test_unsubscribe_deactivates(self, store):
        alert_store.add_subscriber("+254712345678", "Kibera", db_path=store["db"])
        assert alert_store.unsubscribe("+254712345678", "Kibera",
                                       db_path=store["db"]) == 1
        assert alert_store.list_subscribers(db_path=store["db"]) == []
        assert len(alert_store.list_subscribers(active_only=False,
                                                db_path=store["db"])) == 1

    def test_mask_phone(self):
        assert alert_store.mask_phone("+254712345678") == "+254****5678"
        assert alert_store.mask_phone(None) == ""
