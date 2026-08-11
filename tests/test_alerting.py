"""Threshold-crossing alert tests: synthetic scored wards, tmp SQLite store,
injected fake dispatch - no network, no real model."""

import pandas as pd
import pytest

from Utils import alert_store
from Utils.alerting import (
    build_alert_message,
    crossing_to_alert,
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


def fake_dispatch_factory(calls, fail=False):
    def _dispatch(alert, phone, **kwargs):
        msg = alert.to_sms()
        calls.append({"message": msg, "recipients": [phone], "alert": alert})
        if fail:
            return False
        return True

    return _dispatch


def subscribe(store, phone="+254712345678", target="Kibera", language="en"):
    alert_store.add_subscriber(phone, target, db_path=store["db"], language=language)


class TestThresholdCrossingLifecycle:
    def test_crossing_fires_exactly_once(self, store):
        subscribe(store)
        calls = []
        dispatch = fake_dispatch_factory(calls)

        s1 = process_alerts(
            scored_df({"Kibera": 0.10}),
            THRESHOLD,
            snapshot_path=store["snapshot"],
            db_path=store["db"],
            dispatch=dispatch,
        )
        assert s1["crossings"] == [] and len(calls) == 0

        s2 = process_alerts(
            scored_df({"Kibera": 0.55}),
            THRESHOLD,
            snapshot_path=store["snapshot"],
            db_path=store["db"],
            dispatch=dispatch,
        )
        assert len(s2["crossings"]) == 1 and s2["crossings"][0].kind == "new"
        assert len(calls) == 1
        assert calls[0]["recipients"] == ["+254712345678"]
        assert "Kibera" in calls[0]["message"]
        alerts = alert_store.list_alerts(db_path=store["db"])
        assert len(alerts) == 1 and alerts[0]["status"] == "sent"

        s3 = process_alerts(
            scored_df({"Kibera": 0.60}),
            THRESHOLD,
            snapshot_path=store["snapshot"],
            db_path=store["db"],
            dispatch=dispatch,
        )
        assert s3["crossings"] == [] and len(calls) == 1

    def test_escalation_into_critical_band_re_alerts(self, store):
        subscribe(store)
        calls = []
        dispatch = fake_dispatch_factory(calls)
        process_alerts(
            scored_df({"Kibera": 0.10}),
            THRESHOLD,
            snapshot_path=store["snapshot"],
            db_path=store["db"],
            dispatch=dispatch,
        )
        process_alerts(
            scored_df({"Kibera": 0.50}),
            THRESHOLD,
            snapshot_path=store["snapshot"],
            db_path=store["db"],
            dispatch=dispatch,
        )
        s3 = process_alerts(
            scored_df({"Kibera": 0.75}),
            THRESHOLD,
            snapshot_path=store["snapshot"],
            db_path=store["db"],
            dispatch=dispatch,
        )
        assert [c.kind for c in s3["crossings"]] == ["escalation"]
        assert len(calls) == 2
        assert "critical band" in calls[1]["message"].lower() or "CRITICAL" in calls[1]["message"]

        s4 = process_alerts(
            scored_df({"Kibera": 0.80}),
            THRESHOLD,
            snapshot_path=store["snapshot"],
            db_path=store["db"],
            dispatch=dispatch,
        )
        assert s4["crossings"] == [] and len(calls) == 2

    def test_first_run_is_baseline_by_default(self, store):
        subscribe(store)
        calls = []
        s = process_alerts(
            scored_df({"Kibera": 0.90}),
            THRESHOLD,
            snapshot_path=store["snapshot"],
            db_path=store["db"],
            dispatch=fake_dispatch_factory(calls),
        )
        assert s["crossings"] == [] and len(calls) == 0

    def test_alert_on_baseline_override(self, store, tmp_path):
        subscribe(store)
        calls = []
        fresh = {"snapshot": tmp_path / "fresh.json", "db": tmp_path / "fresh.db"}
        alert_store.add_subscriber("+254712345678", "Kibera", db_path=fresh["db"])
        s = process_alerts(
            scored_df({"Kibera": 0.90, "Ruiru": 0.10}),
            THRESHOLD,
            snapshot_path=fresh["snapshot"],
            db_path=fresh["db"],
            dispatch=fake_dispatch_factory(calls),
            alert_on_baseline=True,
        )
        assert len(s["crossings"]) == 1 and len(calls) == 1

    def test_county_subscription_matches_ward_crossing(self, store):
        subscribe(store, target="Nairobi")
        calls = []
        dispatch = fake_dispatch_factory(calls)
        process_alerts(
            scored_df({"Kibera": 0.10}),
            THRESHOLD,
            snapshot_path=store["snapshot"],
            db_path=store["db"],
            dispatch=dispatch,
        )
        process_alerts(
            scored_df({"Kibera": 0.50}),
            THRESHOLD,
            snapshot_path=store["snapshot"],
            db_path=store["db"],
            dispatch=dispatch,
        )
        assert len(calls) == 1

    def test_crossing_without_subscribers_is_logged_not_sent(self, store):
        calls = []
        dispatch = fake_dispatch_factory(calls)
        process_alerts(
            scored_df({"Kibera": 0.10}),
            THRESHOLD,
            snapshot_path=store["snapshot"],
            db_path=store["db"],
            dispatch=dispatch,
        )
        process_alerts(
            scored_df({"Kibera": 0.50}),
            THRESHOLD,
            snapshot_path=store["snapshot"],
            db_path=store["db"],
            dispatch=dispatch,
        )
        assert len(calls) == 0
        alerts = alert_store.list_alerts(db_path=store["db"])
        assert len(alerts) == 1 and alerts[0]["status"] == "no_subscribers"

    def test_missing_credentials_queues_and_advances_snapshot(self, store):
        subscribe(store)

        def no_creds(alert, phone, **kwargs):
            raise SmsConfigError("AT_API_KEY is not configured")

        process_alerts(
            scored_df({"Kibera": 0.10}),
            THRESHOLD,
            snapshot_path=store["snapshot"],
            db_path=store["db"],
            dispatch=no_creds,
        )
        s = process_alerts(
            scored_df({"Kibera": 0.50}),
            THRESHOLD,
            snapshot_path=store["snapshot"],
            db_path=store["db"],
            dispatch=no_creds,
        )
        assert s["skipped"] == 1
        from Utils import alert_queue

        pending = alert_queue.list_pending(db_path=store["db"])
        assert len(pending) == 1

        calls = []
        process_alerts(
            scored_df({"Kibera": 0.55}),
            THRESHOLD,
            snapshot_path=store["snapshot"],
            db_path=store["db"],
            dispatch=fake_dispatch_factory(calls),
        )
        assert len(calls) == 0


class TestCrossingDetection:
    def test_ward_key_disambiguates_duplicate_ward_names(self):
        assert ward_key("Kibera", "Nairobi") != ward_key("Kibera", "Kiambu")

    def test_find_threshold_crossings_ignores_unknown_wards(self):
        scored = scored_df({"NewWard": 0.99})
        assert find_threshold_crossings({}, scored, THRESHOLD) == []
        prev = {"wards": {ward_key("NewWard", "Nairobi"): {"prob": 0.1}}}
        crossings = find_threshold_crossings(prev, scored, THRESHOLD)
        assert len(crossings) == 1 and crossings[0].kind == "new"

    def test_match_subscribers_case_insensitive(self):
        c = find_threshold_crossings(
            {"wards": {ward_key("Kibera", "Nairobi"): {"prob": 0.1}}},
            scored_df({"Kibera": 0.5}),
            THRESHOLD,
        )
        pairs = match_subscribers(c, [{"phone": "+2541", "ward_or_county": "kibera"}])
        assert len(pairs) == 1

    def test_message_contents(self):
        from Utils.alerting import Crossing

        c = Crossing("Kibera", "Nairobi", 0.10, 0.55, "new")
        msg = build_alert_message(c, THRESHOLD)
        assert "Kibera" in msg and "55%" in msg

    def test_swahili_alert(self):
        from Utils.alerting import Crossing

        c = Crossing("Kibera", "Nairobi", 0.10, 0.55, "new")
        alert = crossing_to_alert(c, THRESHOLD, language="sw")
        assert "MAFURIKO" in alert.to_sms("sw")


class TestSubscriberStore:
    def test_phone_validation(self):
        assert alert_store.valid_phone("+254712345678")
        assert not alert_store.valid_phone("0712345678")

    def test_language_stored(self, store):
        alert_store.add_subscriber(
            "+254712345678", "Kibera", db_path=store["db"], language="sw"
        )
        subs = alert_store.list_subscribers(db_path=store["db"])
        assert subs[0]["language"] == "sw"

    def test_mask_phone(self):
        assert alert_store.mask_phone("+254712345678") == "+254****5678"
