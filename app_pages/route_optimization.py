"""Route Optimization page: Pareto rerouting options per affected route."""

import folium
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium

from app_lib.config import NAIROBI_LAT, NAIROBI_LON, REROUTING_CSV, TRADEOFF_PNG
from app_lib.data import (
    OSMNX_AVAILABLE,
    flood_prob_fingerprint,
    get_live_routes,
    get_weighted_graph,
    load_gtfs,
    load_rerouting,
    load_road_graph,
    load_route_geometries,
)
from app_lib.maps import (
    build_choropleth,
    get_affected_stop_ids,
    get_route_shape,
    get_route_stops,
)
from app_lib.state import get_state
from app_lib.theme import PLOTLY_LAYOUT
from Utils.gtfs_rt import GTFS_RT_AVAILABLE, build_gtfs_rt_feed
from Utils.live_routing import find_stop_preserving_route, select_option

state = get_state()
nairobi = state["nairobi"]
threshold = state["threshold"]
use_open_meteo = state["use_open_meteo"]
use_live = state["use_live"]
forecast_horizon_hours = state["forecast_horizon_hours"]

OPTION_LABELS = {
    "fastest": "Fastest (α=5)",
    "balanced": "Balanced (α=50)",
    "safest": "Safest (α=∞)",
}

st.markdown(
    '<div class="section-header">Matatu Route Optimization</div>',
    unsafe_allow_html=True,
)
st.markdown(
    "Calibrated flood probabilities identify which Nairobi matatu routes pass "
    "through high-risk wards. Instead of a single all-or-nothing detour, each "
    "affected route gets a **Pareto set of alternatives** - from *fastest* "
    "(mild flood penalty) to *safest* (flooded roads blocked outright) - so "
    "operators can trade risk against travel time. Risk metrics are "
    "travel-time-weighted (exposure), and each option reports how many of the "
    "route's original stops it still serves within 300 m."
)

routing_source = "historical"
routing_meta: dict = {}
routes, trips, shapes, stops, stop_times = load_gtfs()

mode_label = (
    "live"
    if use_live
    else f"{forecast_horizon_hours}hr forecast" if use_open_meteo else "historical"
)

if use_open_meteo:
    if not OSMNX_AVAILABLE:
        # Expected-and-handled on minimal installs, not a failure: the page
        # falls back to the historical rerouting results by design.
        st.info(
            "osmnx isn't installed, so recomputed rerouting isn't "
            "available here. Showing the historical (Apr 2024) "
            "rerouting results below. Install with `pip install osmnx`."
        )
    else:
        st.info(
            f"**{mode_label.capitalize()} routing**: automatically "
            "recomputed using the current Nairobi flood risk from the "
            "sidebar. Results are cached by the underlying risk data, so "
            "this only takes a few seconds the first time risk actually "
            "changes; repeat visits with the same data are instant."
        )
        if st.button("Force refresh routing"):
            get_live_routes.clear()

        with st.spinner("Loading road network & running flood-weighted Dijkstra..."):
            try:
                G = load_road_graph()
                fingerprint = flood_prob_fingerprint(
                    nairobi, threshold, forecast_horizon_hours
                )
                options_df, route_geoms, routing_meta = get_live_routes(
                    G,
                    nairobi,
                    stops,
                    stop_times,
                    trips,
                    threshold,
                    fingerprint,
                )
                routing_source = "live"
                routing_meta["mode_label"] = mode_label
            except Exception as exc:
                # Genuinely unexpected (e.g. corrupt graph file) - keep this
                # a warning, but make clear the page still has full results.
                st.warning(
                    f"Could not recompute {mode_label} routing ({exc}). "
                    "Showing the historical (Apr 2024) results below."
                )

if routing_source == "historical":
    if not REROUTING_CSV.exists():
        st.info(
            "No historical rerouting data on disk yet. "
            + (
                "Live/forecast routing was attempted automatically above; "
                "if it also failed, check the warning shown."
                if use_open_meteo and OSMNX_AVAILABLE
                else "Run Route_Optimization/route_optimization.ipynb first."
            )
        )
        st.stop()
    options_df = load_rerouting()
    route_geoms = load_route_geometries()

if routing_source == "live":
    st.success(
        f"Showing **{routing_meta.get('mode_label', mode_label)}** rerouting "
        f"(threshold={routing_meta['threshold']:.2f}) · "
        f"{routing_meta['rerouted_routes']} of "
        f"{routing_meta['total_affected_routes']} affected routes rerouted · "
        f"{routing_meta['affected_stops']} stops in high-risk wards."
    )
else:
    st.caption(
        "Showing **historical** rerouting results from the April 2024 "
        "flood event (Route_Optimization/route_optimization.ipynb)."
    )

if options_df.empty:
    st.success(
        "No routes currently need rerouting. Flood risk is below the "
        f"{threshold:.2f} threshold across all monitored wards."
    )
    st.stop()

# -- Preference selector -------------------------------------------------------
preference = st.radio(
    "Detour preference",
    list(OPTION_LABELS),
    format_func=OPTION_LABELS.get,
    horizontal=True,
    index=1,
    help=(
        "Fastest keeps detours short but may retain some flood exposure. "
        "Safest avoids every flood-touched road, often at a large time cost. "
        "Balanced sits between. Routes where a milder option already achieves "
        "the same path are deduplicated to the closest available option."
    ),
)
picked = select_option(options_df, preference)

# Summary metrics
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Affected Routes", int(picked["route_id"].nunique()))
with c2:
    st.metric("Avg Risk Reduction", f"{picked['risk_reduction'].mean():.3f}")
with c3:
    st.metric("Avg Extra Time", f"{picked['extra_time_min'].mean():.1f} min")
with c4:
    if picked["stops_served"].notna().any():
        served = picked["stops_served"].sum() / picked["stops_total"].sum()
        st.metric("Stops Still Served", f"{served:.0%}")
    else:
        st.metric("Routes Improved", int((picked["risk_reduction"] > 0).sum()))

# Rerouting summary table
st.markdown(
    '<div class="section-header" style="margin-top:1.5rem">Rerouting Summary</div>',
    unsafe_allow_html=True,
)
sort_col = st.selectbox(
    "Sort by",
    [
        "risk_reduction",
        "extra_time_min",
        "original_flood_prob",
        "alternative_flood_prob",
        "stops_dropped",
    ],
    format_func=lambda x: x.replace("_", " ").title(),
)
display_cols = {
    "route_id": "Route ID",
    "origin": "Origin",
    "destination": "Destination",
    "option": "Option",
    "original_flood_prob": "Original Risk",
    "alternative_flood_prob": "Alternative Risk",
    "risk_reduction": "Risk Reduction",
    "extra_time_min": "Extra Time (min)",
    "stops_served": "Stops Served",
    "stops_dropped": "Stops Dropped",
}
display_df = (
    picked[list(display_cols)]
    .sort_values(sort_col, ascending=False)
    .reset_index(drop=True)
)
display_df.index += 1
display_df.columns = list(display_cols.values())
st.dataframe(display_df, width="stretch")
dl_csv, dl_gtfs = st.columns(2)
with dl_csv:
    st.download_button(
        label="⬇ Download All Options CSV",
        data=options_df.to_csv(index=False),
        file_name="rerouting_options.csv",
        mime="text/csv",
        width="stretch",
    )
with dl_gtfs:
    if GTFS_RT_AVAILABLE:
        # The same option set as a GTFS-Realtime feed: one ADDED TripUpdate
        # per trip of each affected route, stops in high-risk wards SKIPPED.
        affected_stop_ids = get_affected_stop_ids(nairobi, stops, threshold)
        st.download_button(
            label="⬇ Download GTFS-RT Feed",
            data=build_gtfs_rt_feed(options_df, trips, stop_times, affected_stop_ids),
            file_name="flood_rerouting_feed.pb",
            mime="application/x-protobuf",
            width="stretch",
            help=(
                "GTFS-Realtime v2.0 protobuf feed of the current rerouting "
                "options - immediately consumable by existing transit "
                "infrastructure. Also served live at /reroutes/gtfs-rt."
            ),
        )
    else:
        st.caption(
            "GTFS-RT download unavailable: `pip install gtfs-realtime-bindings`."
        )

# Tradeoff chart
st.markdown(
    '<div class="section-header" style="margin-top:1.5rem">Risk-Time Tradeoff</div>',
    unsafe_allow_html=True,
)
if routing_source == "live":
    # Every option for every route, coloured by option - shows the Pareto
    # frontier rather than a single point per route.
    fig_tradeoff = px.scatter(
        options_df,
        x="extra_time_min",
        y="risk_reduction",
        color="option",
        hover_data=["route_id", "origin", "destination", "stops_dropped"],
        labels={
            "extra_time_min": "Extra Travel Time (minutes)",
            "risk_reduction": "Flood Risk Reduction (exposure-weighted)",
            "option": "Option",
        },
        color_discrete_map={
            "fastest": "#3FA66B",
            "balanced": "#D4A24C",
            "safest": "#C4622D",
        },
    )
    fig_tradeoff.update_traces(marker=dict(size=9, opacity=0.8))
    fig_tradeoff.add_hline(y=0, line_dash="dash", line_color="#4E6357")
    fig_tradeoff.add_vline(x=0, line_dash="dash", line_color="#4E6357")
    fig_tradeoff.update_layout(
        **PLOTLY_LAYOUT,
        xaxis=dict(gridcolor="#1F4A32"),
        yaxis=dict(gridcolor="#1F4A32"),
        margin=dict(t=20, b=20, l=20, r=20),
        legend=dict(bgcolor="#07110D", bordercolor="#1F4A32", borderwidth=1),
    )
    st.plotly_chart(fig_tradeoff, width="stretch")
elif TRADEOFF_PNG.exists():
    st.image(str(TRADEOFF_PNG), width="stretch")
else:
    st.info("Tradeoff chart not found. Run route_optimization.ipynb to generate it.")

# Interactive map section
st.markdown(
    '<div class="section-header" style="margin-top:1.5rem">Interactive Map</div>',
    unsafe_allow_html=True,
)
map_view = st.radio(
    "View",
    ["Flood Risk Map", "Route Explorer"],
    horizontal=True,
    label_visibility="collapsed",
)

if map_view == "Flood Risk Map":
    st.caption("Nairobi ward flood risk · hover a ward for details")
    with st.spinner("Rendering flood risk map..."):
        risk_map = build_choropleth(nairobi, NAIROBI_LAT, NAIROBI_LON, zoom=11)
    st_folium(risk_map, width="stretch", height=520)

else:
    affected_stop_ids = get_affected_stop_ids(nairobi, stops, threshold)
    affected_route_ids = picked["route_id"].tolist()
    n_routes = len(affected_route_ids)

    if "route_idx" not in st.session_state:
        st.session_state.route_idx = 0
    st.session_state.route_idx %= n_routes

    nav_left, nav_centre, nav_right = st.columns([1, 4, 1])

    with nav_left:
        if st.button("← Previous", width="stretch"):
            st.session_state.route_idx = (st.session_state.route_idx - 1) % n_routes

    with nav_right:
        if st.button("Next →", width="stretch"):
            st.session_state.route_idx = (st.session_state.route_idx + 1) % n_routes

    idx = st.session_state.route_idx
    route_row = picked.iloc[idx]
    route_id = route_row["route_id"]

    with nav_centre:
        st.markdown(
            f"<div style='text-align:center;padding:0.4rem 0;'>"
            f"<span style='font-family:Fraunces,serif;font-size:1.05rem;"
            f"font-weight:600;color:#E8DFC8;'>Route {route_id}</span>"
            f"<span style='font-size:0.72rem;color:#8FA894;margin-left:0.6rem;'>"
            f"{route_row['origin']} → {route_row['destination']}</span>"
            f"<span style='font-size:0.65rem;color:#4E6357;margin-left:0.6rem;'>"
            f"({idx + 1} / {n_routes})</span></div>",
            unsafe_allow_html=True,
        )

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(
            f"""<div class="route-stat-card">
            <div class="route-stat-label">Original Flood Risk</div>
            <div class="route-stat-value" style="color:#C4622D;">
                {route_row['original_flood_prob']:.1%}
            </div></div>""",
            unsafe_allow_html=True,
        )
    with s2:
        st.markdown(
            f"""<div class="route-stat-card">
            <div class="route-stat-label">Alternative Flood Risk</div>
            <div class="route-stat-value" style="color:#3FA66B;">
                {route_row['alternative_flood_prob']:.1%}
            </div></div>""",
            unsafe_allow_html=True,
        )
    with s3:
        st.markdown(
            f"""<div class="route-stat-card">
            <div class="route-stat-label">Extra Travel Time</div>
            <div class="route-stat-value" style="color:#5FA8C4;">
                +{route_row['extra_time_min']:.1f} min
            </div></div>""",
            unsafe_allow_html=True,
        )
    with s4:
        stops_note = (
            f"{int(route_row['stops_served'])} / {int(route_row['stops_total'])}"
            if pd.notna(route_row.get("stops_served"))
            else "n/a"
        )
        st.markdown(
            f"""<div class="route-stat-card">
            <div class="route-stat-label">Stops Still Served</div>
            <div class="route-stat-value" style="color:#D4A24C;">
                {stops_note}
            </div></div>""",
            unsafe_allow_html=True,
        )

    # All options for this route (the Pareto set)
    route_options = options_df[options_df["route_id"] == route_id]
    if len(route_options) > 1:
        with st.expander("All detour options for this route"):
            opt_view = route_options[
                [
                    "option",
                    "alternative_flood_prob",
                    "alternative_max_flood_prob",
                    "risk_reduction",
                    "extra_time_min",
                    "stops_served",
                    "stops_dropped",
                ]
            ].rename(
                columns={
                    "option": "Option",
                    "alternative_flood_prob": "Risk (weighted)",
                    "alternative_max_flood_prob": "Worst Edge",
                    "risk_reduction": "Risk Reduction",
                    "extra_time_min": "Extra Time (min)",
                    "stops_served": "Stops Served",
                    "stops_dropped": "Stops Dropped",
                }
            )
            st.dataframe(opt_view, width="stretch", hide_index=True)

    st.markdown("<div style='margin-top:0.75rem'></div>", unsafe_allow_html=True)
    route_view = st.radio(
        "Route view",
        ["Original Route", "Alternative Route"],
        horizontal=True,
        label_visibility="collapsed",
    )

    route_map = folium.Map(
        location=[NAIROBI_LAT, NAIROBI_LON],
        zoom_start=12,
        tiles="CartoDB dark_matter",
    )
    route_coords = get_route_shape(route_id, trips, shapes)
    route_stops = get_route_stops(route_id, trips, stop_times, stops)

    if route_view == "Original Route":
        st.caption(
            "🔵 Original route path · 🔴 Affected stops (in flood-risk wards) · "
            "⚪ Safe stops"
        )
        if route_coords:
            folium.PolyLine(
                route_coords,
                color="#378ADD",
                weight=4,
                opacity=0.9,
                tooltip=f"Route {route_id} - Original",
            ).add_to(route_map)
        if not route_stops.empty:
            for _, stop_row in route_stops.iterrows():
                is_affected = stop_row["stop_id"] in affected_stop_ids
                folium.CircleMarker(
                    location=[stop_row["stop_lat"], stop_row["stop_lon"]],
                    radius=5 if is_affected else 3,
                    color="#C4622D" if is_affected else "#4E6357",
                    fill=True,
                    fill_color="#C4622D" if is_affected else "#4E6357",
                    fill_opacity=0.9,
                    tooltip=(
                        f"⚠ Affected: {stop_row.get('stop_name', stop_row['stop_id'])}"
                        if is_affected
                        else str(stop_row.get("stop_name", stop_row["stop_id"]))
                    ),
                ).add_to(route_map)
        if route_coords:
            lats = [c[0] for c in route_coords]
            lons = [c[1] for c in route_coords]
            route_map.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])

    else:
        geoms_for_route = route_geoms.get(str(route_id), {})
        alt_coords = geoms_for_route.get(
            route_row["option"], geoms_for_route.get("alternative", [])
        )
        st.caption(
            "🟡 Alternative route ({} option) · "
            "🟢 Original route (faded reference) · "
            "🔴 Affected stops · "
            "risk reduced by {:.3f} · +{:.1f} min".format(
                route_row["option"],
                route_row["risk_reduction"],
                route_row["extra_time_min"],
            )
        )
        if route_coords:
            folium.PolyLine(
                route_coords,
                color="#2E5C42",
                weight=3,
                opacity=0.5,
                tooltip=f"Route {route_id} - Original (reference)",
                dash_array="6",
            ).add_to(route_map)
        if alt_coords:
            folium.PolyLine(
                alt_coords,
                color="#D4A24C",
                weight=4,
                opacity=0.95,
                dash_array="8",
                tooltip=f"Route {route_id} - Alternative ({route_row['option']})",
            ).add_to(route_map)
        else:
            st.warning(
                "Alternative path geometry not found. "
                "Force refresh routing or re-run the pipeline."
            )
        if not route_stops.empty:
            for _, stop_row in route_stops.iterrows():
                is_affected = stop_row["stop_id"] in affected_stop_ids
                folium.CircleMarker(
                    location=[stop_row["stop_lat"], stop_row["stop_lon"]],
                    radius=5 if is_affected else 3,
                    color="#C4622D" if is_affected else "#2E4038",
                    fill=True,
                    fill_color="#C4622D" if is_affected else "#2E4038",
                    fill_opacity=0.85,
                    tooltip=(
                        f"🚫 Skipped: {stop_row.get('stop_name', stop_row['stop_id'])}"
                        if is_affected
                        else str(stop_row.get("stop_name", stop_row["stop_id"]))
                    ),
                ).add_to(route_map)
        coords_for_bounds = alt_coords if alt_coords else route_coords
        if coords_for_bounds:
            lats = [c[0] for c in coords_for_bounds]
            lons = [c[1] for c in coords_for_bounds]
            route_map.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
        if alt_coords:
            mid = alt_coords[len(alt_coords) // 2]
            folium.Marker(
                location=mid,
                icon=folium.DivIcon(
                    html=(
                        f"<div style='background:#0E2318;border:1px solid #D4A24C;"
                        f"border-radius:4px;padding:4px 8px;font-family:monospace;"
                        f"font-size:11px;color:#E8DFC8;white-space:nowrap;'>"
                        f"{route_row['option'].title()} · Risk ↓{route_row['risk_reduction']:.3f}"
                        f" · +{route_row['extra_time_min']:.1f} min</div>"
                    ),
                    icon_size=(270, 30),
                    icon_anchor=(135, 15),
                ),
            ).add_to(route_map)

    st_folium(route_map, width="stretch", height=500)

    # -- Stop-preserving detour (on demand: costs one Dijkstra per segment) ----
    if routing_source == "live":
        with st.expander("Stop-preserving detour (experimental)"):
            st.caption(
                "Instead of routing terminal-to-terminal, chain the detour "
                "through the route's intermediate stops that sit *outside* "
                "high-risk wards (up to 8 waypoints), so passengers along the "
                "way are still picked up. Computed on demand for this route."
            )
            if st.button("Compute stop-preserving detour", key=f"sp_{route_id}"):
                with st.spinner("Chaining flood-weighted Dijkstra through safe stops..."):
                    G = load_road_graph()
                    fingerprint = flood_prob_fingerprint(
                        nairobi, threshold, forecast_horizon_hours
                    )
                    Gw, fmap = get_weighted_graph(G, nairobi, fingerprint)
                    sp = find_stop_preserving_route(
                        route_id,
                        Gw,
                        trips,
                        stop_times,
                        stops,
                        nairobi,
                        fmap,
                        risk_threshold=threshold,
                    )
                if sp is None:
                    st.warning(
                        "No stop-preserving path exists for this route at the "
                        "current threshold."
                    )
                else:
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Risk (weighted)", f"{sp['flood_prob']:.1%}")
                    m2.metric("Travel Time", f"{sp['time_s'] / 60:.0f} min")
                    m3.metric(
                        "Stops Served",
                        f"{sp['stops_served']} / {sp['stops_total']}",
                    )
                    if sp["unsafe_stops_skipped"]:
                        st.caption(
                            "Skipped (in high-risk wards): "
                            + ", ".join(sp["unsafe_stops_skipped"])
                        )
                    sp_map = folium.Map(
                        location=[NAIROBI_LAT, NAIROBI_LON],
                        zoom_start=12,
                        tiles="CartoDB dark_matter",
                    )
                    from Utils.live_routing import _path_to_coords

                    sp_coords = _path_to_coords(sp["path"], Gw)
                    if route_coords:
                        folium.PolyLine(
                            route_coords, color="#2E5C42", weight=3, opacity=0.5,
                            dash_array="6", tooltip="Original (reference)",
                        ).add_to(sp_map)
                    folium.PolyLine(
                        sp_coords, color="#5FA8C4", weight=4, opacity=0.95,
                        tooltip="Stop-preserving detour",
                    ).add_to(sp_map)
                    lats = [c[0] for c in sp_coords]
                    lons = [c[1] for c in sp_coords]
                    sp_map.fit_bounds(
                        [[min(lats), min(lons)], [max(lats), max(lons)]]
                    )
                    st_folium(sp_map, width="stretch", height=420)
