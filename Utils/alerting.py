"""Autonomous threshold-crossing alerts: the half of "early warning" that
reaches out instead of waiting to be looked at.

Flow (driven by ``scripts/refresh_cache.py`` on its schedule):

    score wards -> diff against previous run's snapshot (cache/last_scored.json)
    -> wards that NEWLY crossed from below-threshold to at-or-above-threshold
       (or escalated into the critical band while already above threshold)
    -> active subscribers for those wards / their counties
    -> send SMS via Utils.sms_sender.send_sms
    -> log every decision into the alerts_sent table

Idempotency: only a genuine *new* crossing fires (never a ward that stays
above threshold across runs), and the snapshot is updated even when sending
is impossible (no credentials), so a crossing is never re-alerted later.
The first run against an empty snapshot establishes the baseline without
alerting - pass ``alert_on_baseline=True`` to override (e.g. in a demo).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from Utils import alert_store
from Utils.sms_sender import SmsConfigError, send_sms

BASE = Path(__file__).resolve().parent.parent
SNAPSHOT_PATH = BASE / "cache" / "last_scored.json"
CRITICAL_PROB = 0.70  # mirrors app_lib.theme.risk_label's critical band


@dataclass
class Crossing:
    ward: str
    county: str
    prev_prob: float | None
    curr_prob: float
    kind: str  # "new" | "escalation"


def ward_key(ward: str, county: str) -> str:
    """Ward names are not nationally unique; key snapshots by (ward, county)."""
    return f"{ward.strip().title()}|{county.strip().title()}"


def load_snapshot(path: Path | str = SNAPSHOT_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_snapshot(scored: pd.DataFrame, threshold: float, path=SNAPSHOT_PATH) -> None:
    """Atomic write (tmp + rename) so a crashed run can't leave a half-written
    snapshot that would either re-fire or swallow crossings."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "threshold": threshold,
        "wards": {
            ward_key(row.ward, row.county): {"prob": float(row.flood_prob)}
            for row in scored.itertuples()
        },
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp, path)


def find_threshold_crossings(
    prev: dict,
    scored: pd.DataFrame,
    threshold: float,
    critical: float = CRITICAL_PROB,
) -> list[Crossing]:
    """Wards that newly crossed upward since the previous snapshot.

    - ``new``: prev < threshold <= curr
    - ``escalation``: already above threshold, but crossed into the critical
      band (>= ``critical``) since the previous run

    Compares both sides against the *current* threshold, so a threshold
    change can't retroactively manufacture or hide crossings.
    """
    prev_wards = prev.get("wards", {})
    crossings = []
    for row in scored.itertuples():
        key = ward_key(row.ward, row.county)
        curr = float(row.flood_prob)
        prev_entry = prev_wards.get(key)
        prev_prob = float(prev_entry["prob"]) if prev_entry else None
        if prev_prob is None:
            continue  # no baseline for this ward - never alert blind
        if prev_prob < threshold <= curr:
            crossings.append(
                Crossing(row.ward, row.county, prev_prob, curr, "new")
            )
        elif (
            threshold <= prev_prob < critical <= curr
        ):
            crossings.append(
                Crossing(row.ward, row.county, prev_prob, curr, "escalation")
            )
    return crossings


def match_subscribers(
    crossings: list[Crossing], subscribers: list[dict]
) -> list[tuple[dict, Crossing]]:
    """A subscriber matches a crossing on their ward OR their whole-county
    subscription (case-insensitive)."""
    pairs = []
    for sub in subscribers:
        target = sub["ward_or_county"].strip().lower()
        for c in crossings:
            if target in (c.ward.strip().lower(), c.county.strip().lower()):
                pairs.append((sub, c))
    return pairs


def _baseline_crossings(scored: pd.DataFrame, threshold: float) -> list[Crossing]:
    """First-run crossings for --alert-on-baseline: every ward currently at
    or above threshold, with no previous probability to quote."""
    return [
        Crossing(row.ward, row.county, None, float(row.flood_prob), "new")
        for row in scored.itertuples()
        if float(row.flood_prob) >= threshold
    ]


def build_alert_message(crossing: Crossing, threshold: float) -> str:
    if crossing.kind == "escalation":
        return (
            f"FLOOD ALERT (CRITICAL): {crossing.ward} ({crossing.county}) flood "
            f"risk has escalated to {crossing.curr_prob:.0%} - in the critical "
            f"band. Avoid low-lying areas & flooded routes. "
            f"- Nairobi Flood Guard"
        )
    was = (
        f" (was {crossing.prev_prob:.0%})" if crossing.prev_prob is not None else ""
    )
    return (
        f"FLOOD ALERT: {crossing.ward} ({crossing.county}) flood risk is now "
        f"{crossing.curr_prob:.0%}, above the {threshold:.0%} early-warning "
        f"threshold{was}. Avoid low-lying areas & "
        f"flooded routes. - Nairobi Flood Guard"
    )


def process_alerts(
    scored: pd.DataFrame,
    threshold: float,
    snapshot_path=SNAPSHOT_PATH,
    db_path=None,
    alert_on_baseline: bool = False,
    sender=send_sms,
    username: str | None = None,
    api_key: str | None = None,
) -> dict:
    """Diff -> notify -> log -> snapshot. Returns a run summary.

    ``scored`` needs ``ward``, ``county`` and ``flood_prob`` columns.
    ``sender`` is injectable for tests; the default hits Africa's Talking.
    The snapshot is always advanced, even when sending is impossible, so a
    crossing handled once is never re-alerted.
    """
    prev = load_snapshot(snapshot_path)
    is_baseline = not prev.get("wards")

    if is_baseline:
        crossings = _baseline_crossings(scored, threshold) if alert_on_baseline else []
    else:
        crossings = find_threshold_crossings(prev, scored, threshold)

    subscribers = alert_store.list_subscribers(active_only=True, db_path=db_path)
    pairs = match_subscribers(crossings, subscribers)

    summary = {"crossings": crossings, "sent": 0, "failed": 0, "skipped": 0}

    with alert_store.get_conn(db_path) as conn:
        for crossing in crossings:
            message = build_alert_message(crossing, threshold)
            targets = [sub for sub, c in pairs if c is crossing]
            if not targets:
                # A crossing nobody subscribed to: still audit-worthy.
                alert_store.log_alert(
                    crossing.ward, None, message, "no_subscribers", conn=conn
                )
                summary["skipped"] += 1
                continue
            for sub in targets:
                phone = sub["phone"]
                try:
                    result = sender(
                        message, [phone], username=username, api_key=api_key
                    )
                    ok = phone in result["sent"] or not result["failed"]
                    status = "sent" if ok else "failed"
                except SmsConfigError:
                    status = "no_credentials"
                except Exception:
                    status = "failed"
                alert_store.log_alert(
                    crossing.ward, phone, message, status, conn=conn
                )
                summary["sent" if status == "sent" else
                        "skipped" if status == "no_credentials" else
                        "failed"] += 1

    save_snapshot(scored, threshold, snapshot_path)
    return summary
