# Demo Run-of-Show (60–90 seconds)

One command setup, then a rehearsed path that cannot show a stack trace.

## Before going on stage

```bash
make demo
```

This checks the data assets (fails loudly on an un-pulled Git LFS pointer),
warms the rerouting cache, then starts both services:

- App: http://localhost:8501
- API: http://localhost:8000 (docs at `/docs`, GTFS-RT feed at `/reroutes/gtfs-rt`)

No secrets are required for the path below. If you have Africa's Talking
credentials in the environment, the alert in step 4 sends a real SMS; without
them it is still logged to the audit trail with status `no_credentials`.

**Fallback if venue wifi fails:** everything above is localhost - the demo
works fully offline except the "Live (Open-Meteo)" rainfall fetch, so switch
the sidebar Source to **Historical (Apr 2024)** before you start. If the
machine itself fails, play the recorded walkthrough: _(record after final
QA - see JUDGE_BRIEF.md)_.

## The path

1. **Dashboard (0:00–0:20).** Hero numbers up top: people in high-risk
   wards, matatu routes affected, average achievable risk reduction, stops
   still served. Point at the map: "Calibrated probabilities, validated with
   county-held-out spatial CV - not a random split."

2. **Simulate Scenario (0:20–0:40) - the differentiator moment.** Sidebar →
   *Simulate Scenario* → pick a ward (e.g. Ruai) → probability 0.85 →
   **Apply**. The map recolors immediately. "Nothing is written to disk or
   the model - this is what the system does when a real storm arrives."

3. **Route Optimization (0:40–1:05).** Same simulated risk now drives
   rerouting: each affected route gets a Pareto set (fastest / balanced /
   safest), exposure-weighted risk, stop coverage. Open one route in the
   Route Explorer. Click **Download GTFS-RT Feed**: "this drops straight into
   existing transit infrastructure."

4. **Alert History (1:05–1:20).** Sidebar → *Get SMS alerts for your ward* →
   subscribe a phone to the ward's county. Alerts fire on the *scheduled*
   refresh when a ward newly crosses; to trigger one deterministically on
   stage, seed the previous-run snapshot low for one ward, then refresh:

   ```bash
   python - <<'EOF'
   import json
   p = "cache/last_scored.json"
   s = json.load(open(p))
   s["wards"]["Ruai Ward|Nairobi"]["prob"] = 0.10   # "last run it was calm"
   json.dump(s, open(p, "w"))
   EOF
   make refresh-cache   # -> "1 new crossing(s); 1 logged" (real SMS if AT keys set)
   ```

   Reload the Alert History page: exactly one new entry for that ward, phone
   masked. Run `make refresh-cache` again: zero - "a ward that stays high
   alerts once, not every run."

5. **Model Card (1:20–1:30).** Close on rigor: spatial OOF ROC AUC 0.889 vs
   leaky random 0.923, isotonic calibration, recall-first threshold policy,
   single-event label caveat - all read live from the registry.

## If something goes wrong

| Symptom | Recovery |
|---|---|
| Live rainfall fetch fails | Sidebar → Historical (Apr 2024); everything else unchanged |
| Routing recompute errors | Page falls back to historical Apr 2024 results by design |
| `make demo` asset check fails | `git lfs pull`, or `python -m scripts.rebuild_road_network` |
| Total venue failure | Recorded walkthrough video (link above) |
