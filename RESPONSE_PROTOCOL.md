# Response Protocol — Nairobi Flood Guard

Plain-language guide for who acts at each severity tier. This system
**complements** the Kenya Meteorological Department (KMD) and Kenya Red Cross;
it does not replace their official channels.

## Severity tiers (CAP-aligned)

| Tier | Model signal | CAP severity | Who acts |
|------|--------------|--------------|----------|
| **Moderate** | `flood_prob` ≥ registry threshold (~0.30) | Moderate | Informational: dashboard, public JSON/CAP feeds. No SMS unless subscriber opt-in and threshold newly crossed. |
| **High** | `flood_prob` ≥ 0.45 | Severe | Ward/county subscribers alerted (SMS + optional WhatsApp). Matatu reroute options published (`GET /reroutes`, GTFS-RT). |
| **Critical** | `flood_prob` ≥ 0.70 | Extreme | Escalation alert to existing subscribers. County disaster desk notification (manual step — integrate with NDMU contacts). Kenya Red Cross partnership channel per pitch deck. |
| **Extreme / observed flooding** | Field reports + KMD advisory | Extreme + Observed | All of the above plus manual broadcast via presenter SMS panel if AT credentials configured. |

## Channels (redundant by design)

1. **SMS** — primary low-bandwidth channel (Africa's Talking).
2. **WhatsApp** — richer formatting, same `Alert` object.
3. **Public feeds** — `GET /alerts/feed`, `/alerts/feed/rss`, `/alerts/cap/feed` for partners (e.g. Kenya Red Cross SMS relay).
4. **GTFS-RT** — matatu stop skip updates for transit riders/operators.
5. **IVR/voice** — documented stretch goal for lower-literacy reach.

## Idempotency and trust

- Alerts fire on **new** threshold crossings only (not while risk stays elevated).
- Failed sends queue in `pending_alerts` and retry on the next refresh cycle.
- Every decision is logged; POD/FAR-style stats surface on Alert History (small-sample caveat).

## Official sources to monitor alongside this system

- KMD weather warnings: https://meteo.go.ke/weather-warnings/
- KMD Daily Flood Bulletin: https://meteo.go.ke/our-products/flood-bulletin/
- Kenya Red Cross flood early-warning (reactivated 2026 short-rains season)

## What we do not claim

- Endorsement by KMD or Kenya Red Cross (complement, not compete).
- Doorstep-level precision (ward granularity only).
- Replacement for national basin-scale flood forecasting (Nzoia, Tana, etc.).
