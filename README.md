<img src='./Images/idealized_flood_guard.jpeg' style='width: 100%; height: 400px; object-fit: cover;' />

---

<h1 align='center'>
NAIROBI FLOOD GUARD
</h1>

> **Authors**: The Unscripted Engineers

---

<h2 align='center'>
1. OVERVIEW
</h2>

Nairobi Flood Guard is a data science project that addresses the growing threat of flooding across Kenya, motivated by the devastating April 2024 floods and the recent 2026 floods

It has two components:

- **A flood susceptibility model** and

- **A matatu route optimization system**

It is built using open data and reproducible tools

---

<h2 align='center'>
2. BUSINESS UNDERSTANDING
</h2>

### _Problem Statement_

Flooding in Nairobi is extremely disruptive and leads to loss of life, displacement, and infrastructure damage. Current flood response is largely reactive than predictive

### _Objectives_

- **Flood Susceptibility Prediction**

- **Matatu Route Optimization**

## _Stakeholders_

- Kenya Red Cross / National Disaster Management Unit

- Nairobi City County

- Matatu operators and SACCOs

- General public in flood-prone wards

## _Success Metrics_

- High **recall**

- Route recommendations that successfully avoid confirmed flood zones

- Ward-level risk scores that align with known historically flooded areas

## _Scope and Limitations_

- Labels are based on a single flood event

- GTFS data is from 2019

- Ward-level predictions are coarse

- The model predicts **susceptibility** not exact flood timing or depth

---

 <h2 align='center'>
 3. DATA UNDERSTANDING
 </h2>

This project utilises five datasets, each contributing a different dimension to the flood prediction and route optimization pipeline

### a) SRTM Digital Elevation Model (DEM)

The Shuttle Radar Topography Mission (SRTM) DEM provides elevation data at 90 metre resolution. It was used to derive four terrain features per ward: mean elevation, minimum elevation, maximum elevation, and slope

**Source**: OpenTopography (SRTM GL3 product)

### b) CHIRPS Rainfall Data

The Climate Hazards Group InfraRed Precipitation with Station Data (CHIRPS) provides daily rainfall estimates at approximately 5km resolution. Ninety daily rasters covering February-April 2024 were used to derive three rainfall features per ward: cumulative rainfall, maximum single-day rainfall, and total rainfall in the seven days preceding the April 26 flood event

**Source**: UCSB Climate Hazards Group

### c) UNOSAT Flood Extent - FL20240426KEN

A satellite-derived flood extent geodatabase produced by UNOSAT following the April 2024 Kenya floods. The Kenya-wide maximum flood water extent polygon was used to generate binary flood labels for each ward - flooded (1) or not flooded (0).

**Source:** UNOSAT / UNITAR

### d) Kenya Wards Shapefile

A polygon shapefile of Kenya's 1450 administrative wards including ward name, sub-county, county, and 2009 census population. This served as the spatial backbone of the project - all raster datasets were aggregated to ward level through spatial joins and zonal statistics.

**Source:** Regional Centre for Mapping of Resources for Development (RGMRD)

### e) GTFS Feed 2019 - Nairobi Matatu Network

A General Transit Feed Specification (GTFS) dataset describing Nairobi's matatu public transport network as of 2019, including 136 routes, 4,284 stops, and 36,483 route shape points. This dataset underpins the route optimization component of the project.

**Source:** Digital Matatus Project

### f) Compiled Feature Matrix - floods.gpkg

All datasets were processed and merged into a single GeoPackage file (`floods.gpkg`) containing one row per ward with all features and the flood label. More information about about the compiled feature matrix can be found [here](./Data/floods_description.md).

### _EDA_

After loading and examining the dataset (checking for null values and duplicates), the following visualizations were developed:

#### i) Class Imbalance visualization

<img src='./Images/class_distribution.png' />

The not flooded class accounts for ~79% of the data in the dataset. This confirms that the dataset suffers from class imbalance which was addressed.

#### ii) Feature distributions

<img src='./Images/feature_distributions_by_flood_label.png' />

The feature distribution plots reveal that flooded wards receive less rainfall than non-flooded ones suggesting that, at ward scale, rainfall intensity is a weak standalone predictor of flooding. The elevation features show the clearest separation and are better predictors.

#### iii) Correlation heatmap

<img src='./Images/feature_correlation_matrix.png' />

The correlation heatmap confirms the previous observations. Elevation features dominate - `elevation_min_m` and `elevation_mean_m` carry the strongest negative correlations with flooding (-0.50 and -0.50 respectively), followed by `elevation_max_m` (-0.39) and `slope_mean_deg` (-0.26).

All rainfall features correlate weakly with flooding, with `rain_max_daily_mm` showing virtually no linear relationship (-0.001). The heatmap also reveals high inter-correlation between the three elevation features.

#### iv) Top 10 most flooded counties

<img src='./Images/top_10_flooded_counties.png' />

Again, some of the top 10 counties are ones that do not receive a lot of rainfall e.g. Turkana and Garissa. They experience flooding due to their terrain which does not allow the water to drain effectively during those rare seasons when it does rain.

#### _Key Takeaway_

The dataset reveals that in Kenya, flooding is primarily a terrain-driven phenomenon at the ward scale. Low-lying wards flood not necessarily because they receive more rain, but because water from surrounding higher ground drains into them. Terrain features will dominate predictions, and rainfall adds marginal value at this spatial scale.

---

<h2 align='center'>
4. MODEL BUILDING AND EVALUATION
</h2>

Four classification model families were independently developed and tuned by the project team, each in its own dedicated notebook located in the `Model/Notebooks/` directory:

a) [Logistic Regression](./Models/Notebooks/logistic_notebook.ipynb) (baseline) - saved [here](./Models/best_logistic_model.pkl)

b) [Random Forest Classifier](./Models/Notebooks/random_forest_notebook.ipynb) - saved [here](./Models/best_random_forest_model.joblib)

c) [XGBoost Classifier](./Models/Notebooks/XGBoost_notebook.ipynb) - saved [here](./Models/best_xgboost_model.pkl)

d) [Neural Network](./Models/Notebooks/neural_notebook.ipynb) - saved [here](./Models/best_neural_model.keras)

Each model was iteratively improved through hyperparameter tuning, regularisation, and class imbalance handling before the best version was saved.

The following results were obtained:

### a) Logistic Regression (Baseline)

<img src='./Images/logistic_confusion_matrix.png' />

The baseline logistic regression model did not show strong recall on the flooded class, with more than half of its predicted positives being false positives. It showed strong overall recall by keeping false negatives low. This set a solid performance floor for the more complex models to beat.

### b) Random Forest Model

<img src='./Images/random_forest_confusion_matrix.png' />

The Random Forest improved on the baseline across all metrics. Its ensemble nature - aggregating predictions from many decision trees - allowed it to capture non-linear relationships between terrain features and flood risk that the logistic regression cannot.

### c) XGBoost Classifier Model

<img src='./Images/xgboost_confusion_matrix.png' />

The XGBoost model performed better relative to the Random Forest on this dataset in terms of recall. It also had a high accuracy, precision and f1-score. This is likely attributable to its ensemble nature which, like the Random Forest, allowed it to capture non-linear relationships between terrain features and flood risk

### d) Neural Network

<img src='./Images/neural_network_confusion_matrix.png' />

The Neural Network significantly underperformed relative to the Random Forest and XGBoost model. Neural networks typically require large amounts of training data to generalise well - with only 1,450 ward-level samples, the model had limited capacity to learn complex spatial patterns compared to tree-based ensembles.

### _Final Evaluation_

Comparing the metrics of all the models:

| Model          |      AUC | accuracy | precision |   recall | f1-score | support |
| :------------- | -------: | -------: | --------: | -------: | -------: | ------: |
| Logistic       |  0.69898 | 0.689655 |   0.60063 | 0.632194 | 0.604335 |     435 |
| Neural_Network | 0.777919 | 0.737931 |   0.65103 | 0.694622 | 0.661215 |     435 |
| Random_Forest  | 0.881322 | 0.822989 |  0.742775 | 0.792306 | 0.760649 |     435 |
| XGBoost        | 0.896913 | 0.813793 |  0.742293 | 0.818291 | 0.762601 |     435 |

The **XGBoost model** achieved some of the highest metrics among all four models

Given that we were looking for the model with the best recall, and, combined with the fact that it had the best AUC and F1-Score, the **XGBoost model** was selected as the final model for flood susceptibility prediction.

The models' ROC curves reinforce this decision with XGBoost achieving the highest AUC (0.9):

<img src='./Images/roc_curves.png' />

### _Production Model v3 - Leakage-Free, Spatially Validated, Calibrated_

The notebook comparison above used a random ward split, which leaks spatially autocorrelated terrain and rainfall between train and test. The production model is trained by the canonical pipeline in `Models/train.py`, which fixes three integrity issues:

1. **No leaked or fabricated features.** Feature engineering (`Utils/feature_engineering.py`) no longer contains `ward_hist_rate` (previously set directly from the flood label), deterministic rainfall rescalings, or population-derived pseudo transport counts. Every feature is a measured or honestly derived signal.

2. **Spatial cross-validation.** Wards are grouped by county with `GroupKFold`, so the model is always evaluated on counties it never saw. The random split flatters ROC AUC by ~0.03-0.04 on this dataset; the registry records both numbers.

3. **Calibrated probabilities and a principled threshold.** The final model is isotonic-calibrated (probabilities can be read as real frequencies - important because route costs consume them directly), and the operating threshold is chosen from the precision-recall curve as the highest-precision point with recall ≥ 0.80.

Honest county-held-out metrics (calibrated, out-of-fold): **ROC AUC 0.889 · PR AUC 0.690 · recall 0.80 · precision 0.57 · Brier 0.103**.

`Models/model_registry.json` is the single source of truth: it records the model artifact path, exact feature list, operating threshold, CV scheme, metrics and library versions. The Streamlit app and the API read all of these from the registry, so serving can never drift from training. Retrain with:

```bash
python -m Models.train
```

---

<h2 align='center'>
5. ROUTE OPTIMIZATION
</h2>

### Overview

With the XGBoost model identified as the best performer, its flood probability predictions were used to power a matatu route optimization system for Nairobi. The full implementation is in `Route_Optimization/route_optimization.ipynb` ([here](./Route_Optimization/route_optimization.ipynb)). This section summarises the methodology, key outputs, and findings.

The system works in four stages:

1. **Flood probabilities given to road edges** - each road segment in Nairobi's OpenStreetMap network is assigned the flood probability of the ward it passes through via a spatial join

2. **Pareto set of flood-weighted alternatives** - each edge is penalized using the formula `cost = travel_time × (1 + α × flood_probability)`, and every affected route gets one alternative per α level (`Utils/live_routing.py`):

   - **fastest** (α = 5) - mild penalty, short detours, may retain some exposure
   - **balanced** (α = 50) - most of the risk reduction at a fraction of the detour
   - **safest** (α = 1,000,000) - practical infinity; any flood-touched road becomes impassable, so a route is only returned if a completely flood-free path exists

   Alternatives that collapse to the same path are deduplicated. Risk metrics are **exposure-weighted**: a path's flood risk is the travel-time-weighted mean of edge probabilities (plus the worst single edge and the share of travel time on high-risk edges), so clipping one flooded 50 m segment no longer scores the same as driving 10 km through a flood zone.

   Each option also reports **stop coverage** - how many of the route's original stops remain within 300 m of the alternative path, and which ones are dropped. An experimental **stop-preserving mode** chains flood-weighted Dijkstra through the route's safe intermediate stops so passengers along the way are still picked up.

3. **GTFS-RT feed** — rerouting decisions are packaged as a production-ready GTFS-RT protobuf feed with `TripUpdate` messages for each affected trip, consumable by transit apps such as Google Maps and Transit App.

4. **Folium map** — an interactive map that visualises ward flood risk, affected stops, and original vs. alternative route paths side by side.

| route_id    | origin        | destination        | original_flood_prob | alternative_flood_prob | risk_reduction | original_time_s | alternative_time_s | extra_time_min |
| :---------- | :------------ | :----------------- | ------------------: | ---------------------: | -------------: | --------------: | -----------------: | -------------: |
| 20104003910 | Super Highway | Transami           |               0.067 |                  0.003 |          0.064 |         1469.63 |             7772.5 |            105 |
| 30603373812 | Dune          | Rounda             |               0.306 |                  0.022 |          0.284 |         868.318 |            8671.07 |            130 |
| 40705383911 | Quickmart     | Muthurwa           |               0.342 |                  0.028 |          0.313 |         1479.57 |            8216.55 |          112.3 |
| 50700003311 | Utawala       | Kencom/Ambassadeur |               0.315 |                  0.009 |          0.306 |         1494.09 |            7898.88 |          106.7 |
| 50700014501 | Ruiru         | Ruai Bypass        |               0.122 |                  0.122 |              0 |         1108.48 |            1108.48 |              0 |
| 50700033H01 | By Pass       | Cabanas            |               0.317 |                  0.009 |          0.308 |         1083.11 |             8475.9 |          123.2 |
| 50703033J01 | Githunguri    | Cabanas            |               0.378 |                  0.035 |          0.343 |         881.788 |            1704.78 |           13.7 |

The table above shows the top 10 most improved matatu routes ranked by flood risk reduction. Each row represents one route and shows:

- **Original flood risk** - the average flood probability across road segments on the standard route

- **Alternative flood risk** - the same metric for the recommended alternative path

- **Risk reduction** - the absolute improvement; higher is better

- **Extra travel time** - the additional journey time the alternative route adds in minutes, representing the safety-convenience tradeoff

Routes with high risk reduction and low extra travel time are the most actionable recommendations - they offer meaningful safety improvements at minimal inconvenience to operators and commuters.

<img src='./Route_Optimization/Reports/rerouting_tradeoff.png' />

The scatter plot (left) shows the tradeoff between flood risk reduction and extra travel time for each rerouted route. Routes in the upper-left quadrant are ideal — they achieve large risk reductions with little added journey time. Routes in the lower-right represent cases where the algorithm found an alternative path, but the safety gain is marginal relative to the detour cost.

The histogram (right) shows the distribution of extra travel time across all rerouted routes. The majority of alternatives add a significant amount of time, suggesting that for most affected matatu routes, there does not exist a safer path that is not significantly longer than the original. These options, while not convenient, offer a lot more safety. The mean extra travel time is marked by the red dashed line.

> To view the folium map run the streamlit website in `app.py` by typing `streamlit run app.py` in your terminal. More information on this is provided in the section `For More Information` below

### Conclusion

The route optimization system demonstrates that for the majority of Nairobi's flood-affected matatu routes, safer alternatives exist. However, most of them add significant travel time. The GTFS-RT feed produced by this system is immediately compatible with existing transit infrastructure, requiring no changes to operator hardware or passenger apps to deploy.

---

<h2 align='center'>
6. CONCLUSION AND RECOMMENDATION
</h2>

### Conclusion

Nairobi Flood Guard set out to address two problems: predicting which areas of Kenya are most susceptible to flooding, and recommending safer matatu routes when flood events occur. Both objectives were successfully achieved.

The data understanding phase revealed an important and counterintuitive insight: flooding in Kenya at ward scale is primarily a **terrain-driven phenomenon**, not a rainfall-driven one. Low-lying wards flood not because they receive more rain, but because water from surrounding higher ground drains into them. This meant that elevation features dominated model performance while rainfall features contributed marginally, a finding that shaped feature engineering decisions across all four models.

Among the four model families evaluated - Logistic Regression, Random Forest, XGBoost, and Neural Network - the **XGBoost model emerged as the best overall performer**, achieving the highest AUC (0.90) and recall among all models. Its ability to handle non-linear relationships, class imbalance, and noisy features made it well-suited to this dataset. The Neural Network underperformed relative to the tree-based models, consistent with its need for larger datasets than the 1,450 ward-level samples available here.

The route optimization system translated XGBoost's flood probability predictions into actionable rerouting recommendations for Nairobi's matatu network. By assigning prohibitively high costs to flood-affected road segments and running weighted Dijkstra across the real OpenStreetMap road network, the system identified safer alternative paths for affected routes - packaged in a production-ready GTFS-RT feed compatible with existing transit infrastructure.

### Recommendations

#### 1. Running the Flood Prediction Model

Load the calibrated production model and score wards through the shared feature pipeline - always take the model path, feature list and threshold from the registry:

```python
import json, joblib, geopandas as gpd
from Utils.feature_engineering import engineer_features

registry = json.load(open("Models/model_registry.json"))
model = joblib.load(registry["model_path"])
wards = engineer_features(gpd.read_file("Data/floods.gpkg"))
wards["flood_prob"] = model.predict_proba(wards[registry["feature_cols"]])[:, 1]
high_risk = wards[wards["flood_prob"] >= registry["threshold"]]
```

To retrain (spatial CV, calibration and threshold selection included): `python -m Models.train`.

#### 2. Running the Route Optimization System

The live engine is `Utils/live_routing.py` (`run_live_rerouting` returns the full Pareto option set; `select_option` picks one option per route by preference). It requires `Data/nairobi_road_network.graphml` (Git LFS), `Data/GTFS_FEED_2019/`, and scored ward probabilities. The original notebook at `Route_Optimization/route_optimization.ipynb` documents the April 2024 event analysis.

#### 3. Tuning the Flood Risk Threshold

The default high-risk threshold is **read from the registry** (currently ≈ 0.30): the highest-precision operating point that keeps recall ≥ 0.80 under county-held-out validation. Because probabilities are now calibrated, thresholds are interpretable as real frequencies - lower the slider (app) or `threshold` parameter (API) during extreme events to increase sensitivity.

#### 4. Choosing a Detour Preference

Instead of a single alpha, pick a preference per use case: `fastest` (α=5) for minor flooding where roads stay passable, `balanced` (α=50) for most events, `safest` (α=1e6) when flooded roads must be avoided outright. The tradeoff chart in the app shows all three per route.

#### 5. Familiarizing Yourself With the Project

You should begin by reading and running `notebook.ipynb` for a full project overview. Feature engineering logic is centralised in `Utils/feature_engineering.py` - any changes to features must be reflected there and the model retrained (`python -m Models.train`) to keep the registry contract intact. Historical model notebooks are in `Models/Notebooks/`.

---

<h2 align='center'>
7. SERVING, AUTOMATION AND QUALITY
</h2>

### One-Command Demo

```bash
make demo          # asset preflight -> cache warm-up -> API + Streamlit together
docker compose up  # or fully containerized: API on :8000, app on :8501
```

`make demo` first runs `scripts/verify_data_assets.py`, which fails loudly with a remediation hint if a large asset is an un-pulled Git LFS pointer (run `git lfs pull`, or regenerate the road network with `python -m scripts.rebuild_road_network`). See `DEMO_SCRIPT.md` for the rehearsed 60–90 second run-of-show.

### Streamlit App

```bash
streamlit run app.py    # or: make app
```

`app.py` is a thin router; pages live in `app_pages/` (Dashboard, Ward Lookup, Route Optimization, Live Alerts, Alert History, Model Card, AI Assistant) and shared code in `app_lib/`. The sidebar switches rainfall between historical CHIRPS, live Open-Meteo, and 24/48hr forecast modes, hosts the SMS alert opt-in form, and includes a clearly-labeled **Simulate Scenario** control that forces one ward's risk in memory so the map, metrics and rerouting visibly react.

### REST API

Model serving decoupled from the UI (FastAPI):

```bash
uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

| Endpoint | Description |
| :--- | :--- |
| `GET /health`, `GET /registry` | Liveness and the full model contract |
| `GET /wards/risk?county=Nairobi` | Scored wards, sorted by risk |
| `GET /wards/{ward}/risk` | One ward's probability + features |
| `GET /reroutes?preference=balanced` | Pareto rerouting (precomputed cache or on demand) |
| `GET /reroutes/gtfs-rt` | The same option set as a GTFS-Realtime v2.0 protobuf feed - immediately consumable by existing transit infrastructure |
| `POST /reports` / `POST /reports/sms` | Flood report intake (JSON / Africa's Talking webhook) |
| `POST /subscribers` / `DELETE /subscribers` | Opt a phone number into SMS/WhatsApp alerts for a ward or county (EN/Swahili) |
| `GET /alerts` | The alert audit log (recipient numbers masked) |
| `GET /alerts/feed` | Public JSON alert feed |
| `GET /alerts/feed/rss` | RSS 2.0 alert feed |
| `GET /alerts/cap/feed` | CAP 1.2 XML alert feed (complements KMD; does not replace) |

Field reports accumulate in SQLite as **candidate labels for retraining** - point `REPORTS_DB_PATH` at persistent storage in production. The app's Alert History page surfaces both the alert audit trail and recent field reports.

### Scheduled Precompute & Early-Warning Loop

```bash
python -m scripts.refresh_cache    # or: make refresh-cache
```

Refreshes Open-Meteo rainfall, rescores wards, and precomputes the rerouting option set into `cache/precomputed_reroutes.json` so API requests never pay the graph-load cost. Run it on a cron (e.g. every 6 hours).

It is also the early-warning loop: after scoring, it diffs ward probabilities against the previous run's snapshot (`cache/last_scored.json`) and sends an SMS to every active subscriber of a ward (or its county) that **newly crossed** the threshold or escalated into the critical band - logging every decision to the `alerts_sent` table. A ward that stays above threshold alerts once, never repeatedly; the first run establishes the baseline without alerting (`--alert-on-baseline` overrides).

### Tests & CI

```bash
pip install -r requirements.txt -r requirements-dev.txt
ruff check .
pytest tests
```

The suite covers leakage regressions in feature engineering (label-invariance test), the routing engine on synthetic graphs, rainfall windowing/retries/caching, the registry contract, the API (including the GTFS-RT protobuf contract), GTFS-RT feed generation on a synthetic feed, and the alert engine's crossing/de-dup/baseline lifecycle. GitHub Actions runs lint + tests on every push and pull request, plus a fast non-blocking data-asset pre-check that catches an un-pulled Git LFS pointer without downloading the ~100 MB graph.

---

<h2 align='center'>
8. NEXT STEPS
</h2>

Shipped since the original submission: the GTFS-RT feed is served live at `/reroutes/gtfs-rt` (no longer notebook-only); SMS early warning is autonomous (threshold-crossing alerts with idempotent de-dup, subscriber opt-in, and a visible audit trail); `make demo` / `docker compose up` prove the "no new infrastructure" claim; a data-asset preflight guards the Git LFS road network. The EW4All upgrade adds CAP-aligned alerts (`/alerts/cap/feed`, `/alerts/feed`), multilingual EN/Swahili messaging, KMD complementarity panel, March 2026 out-of-time validation, pending-alert retry queue, and `RESPONSE_PROTOCOL.md`. See `JUDGE_BRIEF.md` for the full list.

Still open:

1. Expand the flood label dataset - the training pipeline already supports multi-event labels (see the `labels` section of the model registry); UNOSAT extents for the 2023 El Niño season are the natural next addition, enabling per-event holdout validation

2. Update the GTFS feed (the 2019 Digital Matatus feed predates several route changes)

3. Add flood depth estimation

4. Ground-truth loop, second half - inbound SMS flood reports are stored via `POST /reports/sms` and visible on the Alert History page; remaining work is connecting the Africa's Talking shortcode for inbound keyword opt-in (e.g. text `JOIN <ward>`) and folding confirmed reports into the label set at retraining time

5. Replace the ward-area TWI proxy with real flow accumulation (HydroSHEDS) and add land-cover imperviousness (ESA WorldCover) as features

---

<h2 align='center'>
9. FOR MORE INFORMATION
</h2>

For more information visit the:

- [Main Notebook](./notebook.ipynb)
- [Model Notebooks](./Models/Notebooks/)
- [Route Optimization Notebook](./Route_Optimization/route_optimization.ipynb)
- [Presentation](./presentation.pdf)
- [Tableau Dashboard](https://public.tableau.com/app/profile/carl.collins/viz/NairobiFloodGuardVisualisations/Story1)
- [Deployed Streamlit App](https://nairobi-flood-guard.streamlit.app/)
