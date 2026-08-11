"""Pending alert retry queue for transient SMS/WhatsApp provider outages."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from Utils import alert_store
from Utils.alerts import Alert

DEFAULT_DB = alert_store.DEFAULT_DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pending_alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    phone TEXT NOT NULL,
    channel TEXT NOT NULL,
    payload TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    status TEXT NOT NULL DEFAULT 'pending'
);
"""


def _ensure_table(conn: sqlite3.Connection) -> None:
  conn.executescript(_SCHEMA)


def enqueue(
  phone: str,
  channel: str,
  alert: Alert,
  *,
  db_path: Path | str | None = None,
  conn: sqlite3.Connection | None = None,
) -> int:
  own = conn is None
  conn = conn or alert_store.get_conn(db_path)
  _ensure_table(conn)
  cur = conn.execute(
    "INSERT INTO pending_alerts (created_at, phone, channel, payload, status) "
    "VALUES (?, ?, ?, ?, 'pending')",
    (datetime.now(UTC).isoformat(), phone, channel, alert.to_json()),
  )
  if own:
    conn.commit()
    conn.close()
  return int(cur.lastrowid)


def list_pending(
  limit: int = 100, db_path: Path | str | None = None
) -> list[dict]:
  with alert_store.get_conn(db_path) as conn:
    _ensure_table(conn)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
      "SELECT * FROM pending_alerts WHERE status = 'pending' "
      "ORDER BY id ASC LIMIT ?",
      (limit,),
    ).fetchall()
  return [dict(r) for r in rows]


def mark_sent(row_id: int, db_path: Path | str | None = None) -> None:
  with alert_store.get_conn(db_path) as conn:
    _ensure_table(conn)
    conn.execute(
      "UPDATE pending_alerts SET status = 'sent' WHERE id = ?", (row_id,)
    )


def mark_failed(
  row_id: int, error: str, db_path: Path | str | None = None, max_attempts: int = 5
) -> None:
  with alert_store.get_conn(db_path) as conn:
    _ensure_table(conn)
    conn.execute(
      "UPDATE pending_alerts SET attempts = attempts + 1, last_error = ?, "
      "status = CASE WHEN attempts + 1 >= ? THEN 'dead' ELSE 'pending' END "
      "WHERE id = ?",
      (error[:500], max_attempts, row_id),
    )


def payload_to_alert(payload: str) -> Alert:
  data = json.loads(payload)
  from Utils.alerts import Certainty, Severity, Urgency

  return Alert(
    event=data.get("event", "Flood"),
    severity=Severity(data["severity"]),
    urgency=Urgency(data["urgency"]),
    certainty=Certainty(data["certainty"]),
    area=data.get("area", ""),
    county=data.get("county", ""),
    ward=data.get("ward", ""),
    flood_prob=float(data.get("flood_prob", 0)),
    description_en=data.get("description_en", ""),
    description_sw=data.get("description_sw", ""),
    instruction_en=data.get("instruction_en", ""),
    instruction_sw=data.get("instruction_sw", ""),
    identifier=data.get("identifier", ""),
    language=data.get("language", "en"),
    horizon_hours=int(data.get("horizon_hours", 0)),
    kind=data.get("kind", "new"),
  )


def process_pending(
  sender,
  *,
  db_path: Path | str | None = None,
  username: str | None = None,
  api_key: str | None = None,
) -> dict:
  """Retry pending alerts. ``sender`` is ``dispatch_alert`` from alerting."""
  pending = list_pending(db_path=db_path)
  summary = {"processed": 0, "sent": 0, "failed": 0, "dead": 0}
  for row in pending:
    summary["processed"] += 1
    alert = payload_to_alert(row["payload"])
    try:
      ok = sender(
        alert,
        row["phone"],
        channel=row["channel"],
        username=username,
        api_key=api_key,
        queue_on_failure=False,
        db_path=db_path,
      )
      if ok:
        mark_sent(row["id"], db_path=db_path)
        summary["sent"] += 1
      else:
        mark_failed(row["id"], "send returned false", db_path=db_path)
        summary["failed"] += 1
    except Exception as exc:
      mark_failed(row["id"], str(exc), db_path=db_path)
      summary["failed"] += 1
  return summary
