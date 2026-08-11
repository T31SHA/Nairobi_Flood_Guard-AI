"""Warning performance verification: POD/FAR-style metrics from alert audit log.

Compares issued alerts against field reports (``/reports``) and the March 2026
ground-truth list from ``scripts/validate_against_2026_event.py``. Small-sample
caveats apply — surfaced honestly in the UI.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from Utils import alert_store

# Wards reported flooded in March 2026 (mappable subset from validation script).
REPORTED_2026_WARDS = {
  "mathare north ward",
  "eastleigh north ward",
  "eastleigh south ward",
  "kwa reuben ward",
  "korogocho ward",
  "dandora area i ward",
  "kariobangi north ward",
  "kayole central ward",
  "ruai ward",
  "parklands/highridge ward",
  "kilimani ward",
}


def compute_verification_stats(
  *,
  db_path=None,
  lookback_days: int = 90,
) -> dict:
  """Return POD/FAR-style summary for alerts vs field reports."""
  alerts = alert_store.list_alerts(limit=500, db_path=db_path)
  reports = alert_store.list_reports(limit=500, db_path=db_path)
  cutoff = datetime.now(UTC) - timedelta(days=lookback_days)

  sent = [
    a
    for a in alerts
    if a["status"] == "sent"
    and datetime.fromisoformat(a["timestamp"]) >= cutoff
  ]
  alert_wards = {a["ward"].strip().lower() for a in sent if a.get("ward")}
  report_wards = {
    r["ward"].strip().lower()
    for r in reports
    if r.get("ward") and datetime.fromisoformat(r["created_at"]) >= cutoff
  }

  hits = len(alert_wards & report_wards)
  false_alarms = len(alert_wards - report_wards)
  misses = len(report_wards - alert_wards)

  # Retrospective check against March 2026 reported wards.
  retro_hits = len(alert_wards & REPORTED_2026_WARDS)

  pod = hits / (hits + misses) if (hits + misses) else None
  far = false_alarms / (hits + false_alarms) if (hits + false_alarms) else None

  return {
    "lookback_days": lookback_days,
    "n_alerts_sent": len(sent),
    "n_field_reports": len(
      [r for r in reports if datetime.fromisoformat(r["created_at"]) >= cutoff]
    ),
    "hits": hits,
    "false_alarms": false_alarms,
    "misses": misses,
    "pod": pod,
    "far": far,
    "retro_2026_ward_overlap": retro_hits,
    "retro_2026_ward_total": len(REPORTED_2026_WARDS),
    "caveat": (
      "Small-sample metrics. POD/FAR here compare alert wards to voluntary "
      "field reports and a coarse March 2026 retrospective list — not a "
      "meteorological verification dataset."
    ),
  }
