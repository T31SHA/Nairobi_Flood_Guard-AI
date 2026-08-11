"""Kenya Meteorological Department (KMD) advisory fetcher.

KMD runs national flood forecasting (Daily Flood Bulletin, weather warnings).
As of 2026 their CAP RSS is listed on meteo.go.ke/weather-warnings/ but no
stable machine-readable feed URL was verified — this module tries a short list
of candidate endpoints, then falls back to a clearly-labelled link-out.

Mirrors ``Utils/rainfall_fetcher.py`` resilience: timeouts, retries, graceful
fallback. Nairobi Flood Guard complements KMD; it does not replace it.
"""

from __future__ import annotations

import re
import time
from datetime import UTC, datetime
from typing import Any

import requests

KMD_WARNINGS_URL = "https://meteo.go.ke/weather-warnings/"
KMD_FLOOD_BULLETIN_URL = "https://meteo.go.ke/our-products/flood-bulletin/"
CANDIDATE_FEEDS = (
  "https://meteo.go.ke/cap/rss",
  "https://meteo.go.ke/weather-warnings/feed/",
)
MAX_RETRIES = 2
TIMEOUT_SEC = 15
_CACHE: dict[str, Any] | None = None
_CACHE_TS: float = 0.0
CACHE_TTL_SEC = 900  # 15 min


def _looks_like_xml(text: str) -> bool:
  t = text.lstrip()[:200].lower()
  return t.startswith("<?xml") or "<rss" in t or "<feed" in t or "<alert" in t


def _parse_warnings_html(html: str) -> dict[str, Any] | None:
  """Best-effort scrape of the weather-warnings page (no stable API)."""
  active = re.search(
    r"Active Alerts\s*</[^>]+>\s*<[^>]+>([^<]+)",
    html,
    re.IGNORECASE | re.DOTALL,
  )
  if active and "no active" not in active.group(1).lower():
    return {
      "title": active.group(1).strip(),
      "summary": "Active KMD weather warning (scraped from official page).",
      "mode": "scraped",
    }
  past = re.search(
    r"Expired:\s*([^<]+?)\s*(Heavy Rainfall|Large waves|Strong [Ww]inds)",
    html,
  )
  if past:
    return {
      "title": past.group(0).strip()[:200],
      "summary": "Most recent past advisory on KMD weather-warnings page.",
      "mode": "scraped",
    }
  return None


def fetch_kmd_advisory(*, force: bool = False) -> dict[str, Any]:
  """Return the latest KMD advisory context for dashboard display."""
  global _CACHE, _CACHE_TS
  now = time.time()
  if not force and _CACHE and (now - _CACHE_TS) < CACHE_TTL_SEC:
    return _CACHE

  result: dict[str, Any] = {
    "source": "Kenya Meteorological Department (KMD)",
    "fetched_at": datetime.now(UTC).isoformat(),
    "mode": "linkout",
    "title": "Official KMD advisories",
    "summary": (
      "No machine-readable CAP/RSS feed verified at a stable URL. "
      "Consult KMD's official weather warnings and flood bulletins."
    ),
    "urls": {
      "weather_warnings": KMD_WARNINGS_URL,
      "flood_bulletin": KMD_FLOOD_BULLETIN_URL,
    },
    "active_count": 0,
  }

  for url in CANDIDATE_FEEDS:
    for attempt in range(MAX_RETRIES):
      try:
        resp = requests.get(url, timeout=TIMEOUT_SEC, allow_redirects=True)
        if resp.status_code != 200:
          continue
        if _looks_like_xml(resp.text):
          result.update(
            {
              "mode": "feed",
              "title": "KMD CAP/RSS feed",
              "summary": resp.text[:500],
              "feed_url": url,
            }
          )
          _CACHE, _CACHE_TS = result, now
          return result
      except requests.RequestException:
        if attempt < MAX_RETRIES - 1:
          time.sleep(1.0)

  try:
    resp = requests.get(KMD_WARNINGS_URL, timeout=TIMEOUT_SEC)
    if resp.status_code == 200:
      scraped = _parse_warnings_html(resp.text)
      if scraped:
        result.update(scraped)
        if "no active" in resp.text.lower():
          result["active_count"] = 0
          result["summary"] = (
            "KMD reports no active alerts at this time. Past advisories and "
            "the Daily Flood Bulletin remain the authoritative source."
          )
  except requests.RequestException:
    pass

  _CACHE, _CACHE_TS = result, now
  return result
