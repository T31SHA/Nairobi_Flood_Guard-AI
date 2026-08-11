# Judge Brief — What Changed Since the Original Submission

This documents the hardening pass on the original hackathon submission, in the
same spirit as the README: what changed, where, and why. The short version is
that the model and routing engineering are unchanged — the work closed the
gaps between "impresses on first click" and "survives opening the terminal."

## What was already true (and still is)

- Spatially cross-validated, isotonic-calibrated XGBoost (spatial OOF ROC AUC
  0.889; the leaky random-split 0.923 is quoted in the registry only to
  quantify optimism), a leakage-audited feature pipeline, and an
  exposure-weighted Pareto routing engine with stop-coverage reporting.
- Every number above is read from `Models/model_registry.json` at runtime.
  Nothing in this pass hardcodes a metric, threshold, or feature list.

## What was broken or only claimed, and where it was fixed

**The road network could silently be a Git LFS pointer.** A ZIP download or a
clone without LFS receives 134 bytes instead of the ~101 MB graph, and the
default UI path (Live routing) then opened with a warning banner.
`scripts/verify_data_assets.py` now fails loudly with the exact remediation
(`git lfs pull`, or `scripts/rebuild_road_network.py` to regenerate from
OpenStreetMap — its bounding box is derived from the GTFS stops' own extent
so termini like Ruiru stay inside the graph). It runs at the start of
`scripts/refresh_cache.py` and as a fast, non-blocking CI step.

**The "production-ready GTFS-RT feed" existed only in a notebook cell.** It is
now `Utils/gtfs_rt.py` (a tested, pure function), served live at
`GET /reroutes/gtfs-rt` (`application/x-protobuf`), and downloadable from the
Route Optimization page. Verified: the served bytes parse back into a
`FeedMessage` with one `ADDED` trip update per trip of each affected route
and stops in high-risk wards flagged `SKIPPED`.

**"Early warning" was a dashboard you had to look at.** It is now a loop.
`scripts/refresh_cache.py` diffs each run's scored wards against the previous
snapshot (`cache/last_scored.json`), and a ward that *newly* crosses the
threshold (or escalates into the critical band) triggers an SMS to every
active subscriber of that ward or county, via the Africa's Talking helper now
shared between the UI panel and the script (`Utils/sms_sender.py`). Every
decision lands in an `alerts_sent` table and on the new **Alert History**
page. Idempotency is the design, not a hope: a ward that stays high alerts
once; the first run establishes a baseline; the snapshot advances even when
credentials are missing so nothing re-fires later. Subscriptions come in
through the sidebar form or `POST /subscribers` (`Utils/alert_store.py`).

**"No new hardware" was an assertion.** `Dockerfile` + `docker-compose.yml`
make it runnable: `docker compose up` brings up the API and the app, and the
image build itself fails on an LFS pointer. `make demo` is the one-command
local equivalent; `DEMO_SCRIPT.md` is the 60–90 second run-of-show with named
fallbacks (including a fully offline path via the Historical rainfall mode).

**The methodology was buried in the repo.** The new **Model Card** page
surfaces it in-app — spatial-vs-random validation, calibration, the
recall-first threshold policy, and the limitations (single-event labels,
ward resolution, 2009 census exposure) — read live from the registry.

**The demo couldn't show the system deciding anything on a dry day.** The
sidebar's **Simulate Scenario** control forces one ward's flood probability
in memory; the map, headline metrics, and live rerouting all react. It is
clearly labeled, persists nothing, and never sends an SMS.

## What to verify in five minutes

1. `make demo` from a fresh clone → app on :8501, API on :8000, no warnings.
2. Sidebar → Simulate Scenario → Ruai Ward → 0.85 → Apply → watch the map
   and Route Optimization recompute.
3. `curl localhost:8000/reroutes/gtfs-rt -o feed.pb` → valid protobuf.
4. `make refresh-cache` after a crossing → exactly one new row on Alert
   History; run it again → zero.

## Known limits, stated plainly

- The deployed Streamlit link gated behind a login redirect when probed from
  an unauthenticated datacenter client during this pass; the team should
  confirm it is public and deployed from the branch containing these fixes.
- SMS delivery requires Africa's Talking credentials; without them, crossings
  are still detected and logged (`no_credentials`), which is visible on the
  Alert History page.
- The LICENSE is MIT ("Nairobi Flood Guard AI Contributors") — the
  conventional default for a public hackathon repo; override if the team
  prefers otherwise.

## EW4All upgrade (second pass — early warning system, not just a model)

**Out-of-time March 2026 validation** (`scripts/validate_against_2026_event.py`):
source-cited ground truth from Ushahidi, Interior Ministry lists, and news
coverage. Honest finding: with static April 2024 rainfall features, only Ruai
Ward scores above threshold among mappable reported-flooded neighborhoods —
surfaced on the Model Card page, not spun as a second spatial-CV metric.

**KMD complementarity** (`Utils/kmd_fetcher.py`): dashboard panel links to
official KMD weather warnings and flood bulletins; no stable machine-readable
feed verified, so we link out rather than scrape.

**CAP-aligned alerts** (`Utils/alerts.py`): severity/urgency/certainty/
instruction fields; `GET /alerts/cap/feed`, `/alerts/feed`, `/alerts/feed/rss`.
Multilingual EN/Swahili templates; subscriber language preference.

**Multi-channel dissemination**: SMS + WhatsApp (Africa's Talking), public
Live Alerts page, pending-alert retry queue (`Utils/alert_queue.py`).

**Warning performance tracking** (`Utils/alert_verification.py`): POD/FAR-style
summary on Alert History with small-sample caveat.

**Response protocol** (`RESPONSE_PROTOCOL.md`): who acts at each severity tier;
complements KMD and Kenya Red Cross, does not claim their endorsement.
