This document describes the columns in the [floods](./Data/floods.gpkg) dataset enginnered using CHIRPS and UNOSAT data.

| **Column**            | **Type** | **Description**                                                                                     |
| --------------------- | -------- | --------------------------------------------------------------------------------------------------- |
| `ward`                | Text     | Name of the administrative ward                                                                     |
| `subcounty`           | Text     | Sub-county the ward belongs to                                                                      |
| `county`              | Text     | County the ward belongs to                                                                          |
| `pop2009`             | Float    | Ward population from the 2009 Kenya national census                                                 |
| `rain_cumulative_mm`  | Float    | Total rainfall in millimetres accumulated over the 90-day period February-April 2024                |
| `rain_max_daily_mm`   | Float    | The single highest daily rainfall recorded in that ward across the 90-day period                    |
| `rain_preflood_7d_mm` | Float    | Total rainfall in the 7 days immediately before the April 26 2024 flood event                       |
| `elevation_mean_m`    | Float    | Mean elevation of the ward in metres, derived from the SRTM 90m DEM                                 |
| `elevation_min_m`     | Float    | Lowest elevation point within the ward in metres, derived from the SRTM 90m DEM                     |
| `elevation_max_m`     | Float    | Highest elevation point within the ward in metres, derived from the SRTM 90m DEM                    |
| `slope_mean_deg`      | Float    | Average slope steepness across the ward in degrees, derived from the SRTM DEM using NumPy gradients |
| `flooded`             | Integer  | Binary flood label - 1 if the ward intersected the UNOSAT April 2024 flood extent, 0 if it did not  |
| `geometry`            | Geometry | Polygon boundary of the ward in EPSG:4326 (WGS84), used for spatial operations and mapping          |
