"""Shared SQLite storage for the early-warning loop: SMS subscribers and the
alert audit log, alongside the existing flood_reports table.

Used by the API (``api/main.py``), the Streamlit app (sidebar opt-in form,
Alert History page) and the scheduled alerting script
(``scripts/refresh_cache.py``) so the schema and the queries live in exactly
one place. Deliberately streamlit-free so scripts can import it.

On platforms with ephemeral filesystems point ``REPORTS_DB_PATH`` at a
mounted persistent disk; the schema stays three flat tables to keep a later
swap to a managed database trivial.
"""

from __future__ import annotations

import os
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = Path(
    os.environ.get("REPORTS_DB_PATH", BASE / "cache" / "flood_reports.db")
)

# E.164-ish: leading '+', non-zero country code digit, 7-15 digits total.
PHONE_RE = re.compile(r"^\+[1-9]\d{6,14}$")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS flood_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    source TEXT NOT NULL,
    phone TEXT,
    ward TEXT,
    county TEXT,
    lat REAL,
    lon REAL,
    text TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS subscribers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    phone TEXT NOT NULL,
    ward_or_county TEXT NOT NULL,
    language TEXT NOT NULL DEFAULT 'en',
    created_at TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    UNIQUE (phone, ward_or_county)
);
CREATE TABLE IF NOT EXISTS alerts_sent (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    ward TEXT,
    phone TEXT,
    message TEXT NOT NULL,
    status TEXT NOT NULL,
    channel TEXT DEFAULT 'sms',
    severity TEXT,
    alert_id TEXT
);
"""


def get_conn(db_path: Path | str | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(_SCHEMA)
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Additive migrations for existing databases."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(subscribers)")}
    if "language" not in cols:
        conn.execute(
            "ALTER TABLE subscribers ADD COLUMN language TEXT NOT NULL DEFAULT 'en'"
        )
    acols = {row[1] for row in conn.execute("PRAGMA table_info(alerts_sent)")}
    if "channel" not in acols:
        conn.execute("ALTER TABLE alerts_sent ADD COLUMN channel TEXT DEFAULT 'sms'")
    if "severity" not in acols:
        conn.execute("ALTER TABLE alerts_sent ADD COLUMN severity TEXT")
    if "alert_id" not in acols:
        conn.execute("ALTER TABLE alerts_sent ADD COLUMN alert_id TEXT")
    conn.commit()


def valid_phone(phone: str) -> bool:
    return bool(PHONE_RE.match(phone.strip()))


def mask_phone(phone: str | None) -> str:
    """'+254712345678' -> '+254****5678' for display surfaces; the full
    number never leaves the send path."""
    if not phone:
        return ""
    phone = phone.strip()
    return phone[:4] + "****" + phone[-4:] if len(phone) > 8 else "****"


def add_subscriber(
    phone: str,
    ward_or_county: str,
    db_path: Path | str | None = None,
    language: str = "en",
) -> tuple[int, bool]:
    """Insert (or reactivate) a subscription. Returns (row id, newly created).

    Idempotent on (phone, ward_or_county): re-subscribing the same pair
    flips ``active`` back to 1 rather than duplicating the row.
    """
    phone = phone.strip()
    ward_or_county = ward_or_county.strip()
    if not valid_phone(phone):
        raise ValueError(f"Invalid phone number {phone!r}: expected E.164 (+...)")
    if not ward_or_county:
        raise ValueError("ward_or_county must not be empty")
    if language not in ("en", "sw"):
        raise ValueError(f"language must be 'en' or 'sw', got {language!r}")
    with get_conn(db_path) as conn:
        conn.execute(
            "INSERT INTO subscribers (phone, ward_or_county, language, created_at, active) "
            "VALUES (?, ?, ?, ?, 1) "
            "ON CONFLICT (phone, ward_or_county) DO UPDATE SET active = 1, language = excluded.language",
            (phone, ward_or_county, language, datetime.now(UTC).isoformat()),
        )
        row = conn.execute(
            "SELECT id FROM subscribers WHERE phone = ? AND ward_or_county = ?",
            (phone, ward_or_county),
        ).fetchone()
        created = conn.total_changes > 0
        return int(row[0]), created


def unsubscribe(phone: str, ward_or_county: str, db_path=None) -> int:
    """Deactivate (never delete) so the audit trail stays intact."""
    with get_conn(db_path) as conn:
        cur = conn.execute(
            "UPDATE subscribers SET active = 0 WHERE phone = ? AND ward_or_county = ?",
            (phone.strip(), ward_or_county.strip()),
        )
        return cur.rowcount


def list_subscribers(active_only: bool = True, db_path=None) -> list[dict]:
    query = (
        "SELECT id, phone, ward_or_county, language, created_at, active "
        "FROM subscribers"
    )
    if active_only:
        query += " WHERE active = 1"
    query += " ORDER BY id DESC"
    with get_conn(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(query).fetchall()]


def log_alert(
    ward: str | None,
    phone: str | None,
    message: str,
    status: str,
    db_path=None,
    conn: sqlite3.Connection | None = None,
    channel: str = "sms",
    severity: str | None = None,
    alert_id: str | None = None,
) -> None:
    """One row per alert decision: 'sent', 'failed', 'no_credentials' or
    'no_subscribers' (a crossing with nobody to notify - still worth an
    audit trail)."""
    own_conn = conn is None
    conn = conn or get_conn(db_path)
    try:
        conn.execute(
            "INSERT INTO alerts_sent "
            "(timestamp, ward, phone, message, status, channel, severity, alert_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                datetime.now(UTC).isoformat(),
                ward,
                phone,
                message,
                status,
                channel,
                severity,
                alert_id,
            ),
        )
        if own_conn:
            conn.commit()
    finally:
        if own_conn:
            conn.close()


def list_alerts(limit: int = 200, db_path=None) -> list[dict]:
    with get_conn(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM alerts_sent ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def list_reports(limit: int = 100, db_path=None) -> list[dict]:
    with get_conn(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM flood_reports ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]
