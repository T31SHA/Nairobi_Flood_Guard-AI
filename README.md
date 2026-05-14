<img src='./Images/idealized_flood_guard.jpeg' style='width: 100%; height: 400px; object-fit: cover;' />

---

<h1 align='center'>
NAIROBI FLOOD GUARD
</h1>

---

<h2 align='center'>
1. OVERVIEW
</h2>

Nairobi Flood Guard is a data science project that addresses the growing threat of flooding across Kenya, motivated by the devastating April 2024 floods abd the recent 2026 floods

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

A satellite-derived flood extent geodatabase produced by UNOSAT following the April 2024 Kenya floods. The Kenya-wide maximum flood water extent polygon was used to generate binary flood labels for each ward — flooded (1) or not flooded (0).

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

###
