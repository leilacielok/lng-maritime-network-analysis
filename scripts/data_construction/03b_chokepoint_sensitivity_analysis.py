#!/usr/bin/env python3
"""Sensitivity analysis for LNG route--chokepoint assignments.

Primary classification remains exact geometric intersection. This script tests
whether assignments change when each chokepoint geometry is expanded by 5, 10
and 25 km in a local azimuthal-equidistant projection. It never overwrites the
primary route file.

Run: python 03b_chokepoint_sensitivity_analysis.py
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from shapely.geometry import mapping, shape
from shapely.ops import transform
from shapely.validation import explain_validity

ROUTES_FILENAME = "LNG_1037_routes_searoute.geojson"
CHOKEPOINTS_FILENAME = "PortWatch_28_chokepoints_geometry.geojson"
BUFFER_KM = (0, 5, 10, 25)
EXPECTED_ROUTES = 1037
EXPECTED_CHOKEPOINTS = 28


def code_root() -> Path:
    here = Path(__file__).resolve().parent
    return next((p for p in (here, *here.parents) if p.name.lower() == "code"), here)


def find_file(root: Path, filename: str) -> Path:
    preferred = [root / "data" / filename, root / "data" / "processed" / filename,
                 root / "data" / "marnet" / filename, root / "marnet" / filename,
                 root / filename]
    preferred.extend(sorted(root.rglob(filename)))
    found = next((p for p in preferred if p.is_file()), None)
    if found is None:
        raise FileNotFoundError(f"Could not find {filename!r} under {root}")
    return found


def read_geojson(path: Path) -> dict:
    with path.open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def local_transformer(geometry):
    centre = geometry.representative_point()
    local = CRS.from_proj4(
        f"+proj=aeqd +lat_0={centre.y} +lon_0={centre.x} +datum=WGS84 +units=m +no_defs"
    )
    return Transformer.from_crs("OGC:CRS84", local, always_xy=True).transform


def validate_inputs(routes: dict, chokepoints: dict) -> tuple[list, list, list[dict]]:
    route_features = routes.get("features", [])
    cp_features = chokepoints.get("features", [])
    if len(route_features) != EXPECTED_ROUTES:
        raise ValueError(f"Expected {EXPECTED_ROUTES} routes, got {len(route_features)}")
    if len(cp_features) != EXPECTED_CHOKEPOINTS:
        raise ValueError(f"Expected {EXPECTED_CHOKEPOINTS} chokepoints, got {len(cp_features)}")

    route_ids = [f.get("properties", {}).get("route_id") for f in route_features]
    if any(not rid for rid in route_ids) or len(set(route_ids)) != len(route_ids):
        raise ValueError("Missing or duplicated route_id")
    cp_ids = [f.get("properties", {}).get("node_id") for f in cp_features]
    if any(not node for node in cp_ids) or len(set(cp_ids)) != len(cp_ids):
        raise ValueError("Missing or duplicated chokepoint node_id")

    route_geometries = [shape(f["geometry"]) for f in route_features]
    cp_geometries = [shape(f["geometry"]) for f in cp_features]
    geometry_qa = []
    for feature, geometry in zip(route_features, route_geometries):
        geometry_qa.append({
            "feature_type": "route", "feature_id": feature["properties"]["route_id"],
            "feature_name": f"{feature['properties'].get('from_terminal')} -> {feature['properties'].get('to_terminal')}",
            "geometry_type": geometry.geom_type, "is_valid": geometry.is_valid,
            "validity_reason": explain_validity(geometry), "is_empty": geometry.is_empty,
        })
    for feature, geometry in zip(cp_features, cp_geometries):
        geometry_qa.append({
            "feature_type": "chokepoint", "feature_id": feature["properties"]["node_id"],
            "feature_name": feature["properties"].get("chokepoint"),
            "geometry_type": geometry.geom_type, "is_valid": geometry.is_valid,
            "validity_reason": explain_validity(geometry), "is_empty": geometry.is_empty,
        })
    return route_geometries, cp_geometries, geometry_qa


def analyse(routes: dict, chokepoints: dict, route_geometries: list, cp_geometries: list):
    details, summaries = [], []
    assignments = {distance: {f["properties"]["route_id"]: [] for f in routes["features"]}
                   for distance in BUFFER_KM}

    for cp_feature, cp_geometry in zip(chokepoints["features"], cp_geometries):
        cp = cp_feature["properties"]
        project = local_transformer(cp_geometry)
        projected_cp = transform(project, cp_geometry)
        distances = []
        exact_flags = []
        for route_feature, route_geometry in zip(routes["features"], route_geometries):
            route = route_feature["properties"]
            projected_route = transform(project, route_geometry)
            distance_km = projected_route.distance(projected_cp) / 1000
            distances.append(distance_km)
            exact = route_geometry.intersects(cp_geometry)
            exact_flags.append(exact)
            row = {
                "route_id": route["route_id"], "from_node_id": route.get("from_node_id"),
                "to_node_id": route.get("to_node_id"), "from_terminal": route.get("from_terminal"),
                "to_terminal": route.get("to_terminal"), "chokepoint_node_id": cp["node_id"],
                "chokepoint": cp.get("chokepoint"), "minimum_distance_km": distance_km,
                "exact_intersection": exact,
            }
            for buffer_km in BUFFER_KM[1:]:
                row[f"within_{buffer_km}km"] = distance_km <= buffer_km
                row[f"new_at_{buffer_km}km"] = (not exact) and distance_km <= buffer_km
            details.append(row)
            for buffer_km in BUFFER_KM:
                if exact or distance_km <= buffer_km:
                    assignments[buffer_km][route["route_id"]].append(cp.get("chokepoint"))

        summary = {
            "chokepoint_node_id": cp["node_id"], "chokepoint": cp.get("chokepoint"),
            "geometry_status": cp.get("geometry_status"),
            "exact_route_count": int(sum(exact_flags)),
            "minimum_route_distance_km": float(min(distances)),
        }
        exact_count = summary["exact_route_count"]
        for buffer_km in BUFFER_KM[1:]:
            count = int(sum(d <= buffer_km for d in distances))
            summary[f"route_count_{buffer_km}km"] = count
            summary[f"additional_routes_{buffer_km}km"] = count - exact_count
        summary["sensitivity_status"] = (
            "exactly_observed" if exact_count else
            "near_miss_within_5km" if summary["route_count_5km"] else
            "near_miss_within_10km" if summary["route_count_10km"] else
            "near_miss_within_25km" if summary["route_count_25km"] else
            "not_observed_within_25km"
        )
        summaries.append(summary)
    return pd.DataFrame(details), pd.DataFrame(summaries), assignments


def sensitivity_geojson(routes: dict, assignments: dict) -> dict:
    result = deepcopy(routes)
    for feature in result["features"]:
        route_id = feature["properties"]["route_id"]
        for distance in BUFFER_KM:
            label = "exact" if distance == 0 else f"within_{distance}km"
            names = assignments[distance][route_id]
            feature["properties"][f"n_chokepoints_{label}"] = len(names)
            feature["properties"][f"chokepoints_{label}"] = "; ".join(names)
    return result


def mindoro_map_data(routes: dict, chokepoints: dict, detail: pd.DataFrame) -> dict:
    """Return Mindoro, its metric buffers and every candidate route within 25 km."""
    cp_feature = next(
        f for f in chokepoints["features"]
        if f["properties"].get("chokepoint") == "Mindoro Strait"
    )
    cp_geometry = shape(cp_feature["geometry"])
    centre = cp_geometry.representative_point()
    local = CRS.from_proj4(
        f"+proj=aeqd +lat_0={centre.y} +lon_0={centre.x} +datum=WGS84 +units=m +no_defs"
    )
    forward = Transformer.from_crs("OGC:CRS84", local, always_xy=True).transform
    inverse = Transformer.from_crs(local, "OGC:CRS84", always_xy=True).transform
    projected = transform(forward, cp_geometry)

    features = []
    for distance, color in [(25, "#F59E0B"), (10, "#EF4444")]:
        geometry = transform(inverse, projected.buffer(distance * 1000))
        features.append({
            "type": "Feature", "geometry": mapping(geometry),
            "properties": {"feature_type": "buffer", "buffer_km": distance,
                           "label": f"Mindoro + {distance} km", "color": color},
        })
    features.append({
        "type": "Feature", "geometry": cp_feature["geometry"],
        "properties": {"feature_type": "chokepoint", "label": "Mindoro Strait geometry",
                       "color": "#111827"},
    })

    mindoro = detail.loc[
        detail["chokepoint"].eq("Mindoro Strait") & detail["within_25km"]
    ].set_index("route_id")
    for route in routes["features"]:
        route_id = route["properties"]["route_id"]
        if route_id not in mindoro.index:
            continue
        audit = mindoro.loc[route_id]
        distance = float(audit["minimum_distance_km"])
        band = "within_10km" if distance <= 10 else "between_10_and_25km"
        color = "#DC2626" if distance <= 10 else "#F59E0B"
        features.append({
            "type": "Feature", "geometry": route["geometry"],
            "properties": {
                "feature_type": "route", "route_id": route_id,
                "from_terminal": route["properties"].get("from_terminal"),
                "to_terminal": route["properties"].get("to_terminal"),
                "from_node_id": route["properties"].get("from_node_id"),
                "to_node_id": route["properties"].get("to_node_id"),
                "minimum_distance_km": round(distance, 3), "sensitivity_band": band,
                "color": color,
            },
        })
    return {"type": "FeatureCollection", "features": features}


def write_interactive_map(data: dict, path: Path) -> None:
    """Write a standalone Leaflet viewer with routes embedded in the HTML."""
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mindoro Strait sensitivity routes</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>html,body,#map{height:100%;margin:0} .legend{background:white;padding:10px 12px;
font:13px Arial;line-height:20px;box-shadow:0 1px 5px #777}.sw{display:inline-block;
width:18px;height:4px;margin-right:7px;vertical-align:middle}</style></head>
<body><div id="map"></div><script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>const data=__DATA__; const map=L.map('map',{attributionControl:false});
function style(f){const p=f.properties;if(p.feature_type==='buffer')return{
 color:p.color,weight:2,fillColor:p.color,fillOpacity:p.buffer_km===10?.12:.06,dashArray:'6 4'};
 if(p.feature_type==='chokepoint')return{color:p.color,weight:3,fillOpacity:.18};
 return{color:p.color,weight:p.sensitivity_band==='within_10km'?3:1.5,opacity:.75};}
function popup(f,l){const p=f.properties;if(p.feature_type==='route')l.bindPopup(
 `<b>${p.route_id}</b><br>${p.from_terminal} &rarr; ${p.to_terminal}<br>`+
 `Minimum distance: <b>${p.minimum_distance_km} km</b><br>${p.sensitivity_band}`);
 else l.bindPopup(p.label)}
const layer=L.geoJSON(data,{style:style,onEachFeature:popup}).addTo(map);map.fitBounds(layer.getBounds());
const legend=L.control({position:'topright'});legend.onAdd=function(){const d=L.DomUtil.create('div','legend');
 d.innerHTML='<b>Mindoro sensitivity</b><br><span class="sw" style="background:#DC2626"></span>Routes within 10 km<br>'+
 '<span class="sw" style="background:#F59E0B"></span>Routes 10–25 km<br>'+
 '<span class="sw" style="background:#111827"></span>Original geometry';return d};legend.addTo(map);
</script></body></html>""".replace("__DATA__", payload)
    path.write_text(html, encoding="utf-8")


def all_sensitive_map_data(routes: dict, chokepoints: dict, detail: pd.DataFrame,
                           summary: pd.DataFrame) -> tuple[dict, list[str]]:
    """Build offline-map layers for every chokepoint gaining routes by 25 km."""
    names = summary.loc[
        summary["additional_routes_10km"].gt(0) | summary["additional_routes_25km"].gt(0),
        "chokepoint",
    ].tolist()
    route_by_id = {f["properties"]["route_id"]: f for f in routes["features"]}
    cp_by_name = {f["properties"]["chokepoint"]: f for f in chokepoints["features"]}
    features = []
    for name in names:
        cp_feature = cp_by_name[name]
        cp_geometry = shape(cp_feature["geometry"])
        centre = cp_geometry.representative_point()
        local = CRS.from_proj4(
            f"+proj=aeqd +lat_0={centre.y} +lon_0={centre.x} +datum=WGS84 +units=m +no_defs"
        )
        forward = Transformer.from_crs("OGC:CRS84", local, always_xy=True).transform
        inverse = Transformer.from_crs(local, "OGC:CRS84", always_xy=True).transform
        projected = transform(forward, cp_geometry)
        for distance, color in [(25, "#F59E0B"), (10, "#DC2626")]:
            features.append({
                "type": "Feature", "geometry": mapping(transform(inverse, projected.buffer(distance * 1000))),
                "properties": {"focus_chokepoint": name, "feature_type": "buffer",
                               "label": f"{name} + {distance} km", "buffer_km": distance,
                               "color": color},
            })
        features.append({
            "type": "Feature", "geometry": cp_feature["geometry"],
            "properties": {"focus_chokepoint": name, "feature_type": "chokepoint",
                           "label": name, "color": "#111827"},
        })
        rows = detail.loc[detail["chokepoint"].eq(name) &
                          (detail["exact_intersection"] | detail["within_25km"])]
        for row in rows.itertuples(index=False):
            if row.exact_intersection:
                band, color = "exact_intersection", "#64748B"
            elif row.within_10km:
                band, color = "new_within_10km", "#DC2626"
            else:
                band, color = "new_between_10_and_25km", "#F59E0B"
            source = route_by_id[row.route_id]
            features.append({
                "type": "Feature", "geometry": source["geometry"],
                "properties": {"focus_chokepoint": name, "feature_type": "route",
                               "route_id": row.route_id, "from_terminal": row.from_terminal,
                               "to_terminal": row.to_terminal,
                               "minimum_distance_km": round(float(row.minimum_distance_km), 3),
                               "sensitivity_band": band, "color": color},
            })
    return {"type": "FeatureCollection", "features": features}, names


def write_all_sensitive_map(data: dict, names: list[str], path: Path) -> None:
    """Interactive, tile-free Leaflet map with a chokepoint selector."""
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    options = "".join(f'<option value="{name}">{name}</option>' for name in names)
    html = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Chokepoint sensitivity routes</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>html,body,#map{height:100%;margin:0}#map{background:#eaf4f8}.panel{position:absolute;z-index:1000;
top:10px;right:10px;background:#fff;padding:12px;font:13px Arial;box-shadow:0 1px 5px #777;
max-width:310px}.panel select{width:100%;margin:7px 0 9px}.sw{display:inline-block;width:18px;height:4px;
margin-right:7px;vertical-align:middle}.leaflet-control-attribution{display:none}</style></head><body>
<div id="map"></div><div class="panel"><b>Chokepoint sensitivity</b><br><select id="choice">__OPTIONS__</select><br>
<span class="sw" style="background:#64748B"></span>Exact routes<br>
<span class="sw" style="background:#DC2626"></span>New within 10 km<br>
<span class="sw" style="background:#F59E0B"></span>New between 10–25 km<br>
<small>No external basemap: the viewer works without OpenStreetMap tiles.</small></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script><script>
const data=__DATA__;const map=L.map('map',{attributionControl:false});let layer;
function style(f){const p=f.properties;if(p.feature_type==='buffer')return{color:p.color,weight:2,
fillColor:p.color,fillOpacity:p.buffer_km===10?.10:.04,dashArray:'6 4'};if(p.feature_type==='chokepoint')
return{color:'#111827',weight:4,fillOpacity:.18};return{color:p.color,
weight:p.sensitivity_band==='exact_intersection'?1.2:3,opacity:p.sensitivity_band==='exact_intersection'?.35:.9};}
function popup(f,l){const p=f.properties;if(p.feature_type==='route')l.bindPopup(`<b>${p.route_id}</b><br>`+
`${p.from_terminal} &rarr; ${p.to_terminal}<br>Minimum distance: <b>${p.minimum_distance_km} km</b><br>${p.sensitivity_band}`);
else l.bindPopup(p.label)}
function draw(name){if(layer)map.removeLayer(layer);layer=L.geoJSON(data,{filter:f=>f.properties.focus_chokepoint===name,
style:style,onEachFeature:popup}).addTo(map);map.fitBounds(layer.getBounds(),{padding:[20,20]});}
const choice=document.getElementById('choice');choice.onchange=()=>draw(choice.value);draw(choice.value);
</script></body></html>""".replace("__DATA__", payload).replace("__OPTIONS__", options)
    path.write_text(html, encoding="utf-8")


def plot_summary(summary: pd.DataFrame, path: Path) -> None:
    plot = summary.sort_values("route_count_25km", ascending=True)
    y = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(13, 11), constrained_layout=True)
    ax.barh(y, plot["route_count_25km"], color="#C4B5FD", label="Within 25 km")
    ax.barh(y, plot["route_count_10km"], color="#818CF8", label="Within 10 km")
    ax.barh(y, plot["route_count_5km"], color="#3B82F6", label="Within 5 km")
    ax.barh(y, plot["exact_route_count"], color="#1E3A8A", label="Exact intersection")
    ax.set_yticks(y, plot["chokepoint"], fontsize=8)
    ax.set_xlabel("Number of Searoute geometries classified")
    ax.set_title("Chokepoint assignment sensitivity", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    ax.grid(axis="x", alpha=.2); ax.spines[["top", "right", "left"]].set_visible(False)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    root = code_root()
    routes_path = find_file(root, ROUTES_FILENAME)
    cp_path = find_file(root, CHOKEPOINTS_FILENAME)
    output = root / "eda_outputs" / "chokepoint_sensitivity"; output.mkdir(parents=True, exist_ok=True)
    routes, chokepoints = read_geojson(routes_path), read_geojson(cp_path)
    route_geometries, cp_geometries, qa = validate_inputs(routes, chokepoints)
    detail, summary, assignments = analyse(routes, chokepoints, route_geometries, cp_geometries)

    pd.DataFrame(qa).to_csv(output / "geometry_validity_audit.csv", index=False)
    detail.to_csv(output / "route_chokepoint_distance_audit.csv", index=False)
    summary.to_csv(output / "chokepoint_sensitivity_summary.csv", index=False)
    candidates = detail.loc[~detail.exact_intersection & detail.within_25km].sort_values(
        ["chokepoint", "minimum_distance_km"])
    candidates.to_csv(output / "new_assignments_within_25km.csv", index=False)
    with (output / "LNG_1037_routes_chokepoint_sensitivity.geojson").open("w", encoding="utf-8") as stream:
        json.dump(sensitivity_geojson(routes, assignments), stream, ensure_ascii=False, separators=(",", ":"))
    plot_summary(summary, output / "chokepoint_sensitivity.png")
    mindoro_data = mindoro_map_data(routes, chokepoints, detail)
    with (output / "mindoro_25km_candidate_routes.geojson").open("w", encoding="utf-8") as stream:
        json.dump(mindoro_data, stream, ensure_ascii=False, separators=(",", ":"))
    write_interactive_map(mindoro_data, output / "mindoro_25km_candidate_routes_map.html")
    sensitive_data, sensitive_names = all_sensitive_map_data(routes, chokepoints, detail, summary)
    with (output / "all_sensitive_chokepoint_routes.geojson").open("w", encoding="utf-8") as stream:
        json.dump(sensitive_data, stream, ensure_ascii=False, separators=(",", ":"))
    write_all_sensitive_map(
        sensitive_data, sensitive_names,
        output / "all_sensitive_chokepoint_routes_map.html",
    )

    print("CHOKEPOINT SENSITIVITY ANALYSIS COMPLETED")
    print(f"Routes: {len(routes['features']):,}; chokepoints: {len(chokepoints['features']):,}")
    for distance in BUFFER_KM:
        field = "exact_route_count" if distance == 0 else f"route_count_{distance}km"
        print(f"Chokepoints observed at {distance} km: {(summary[field] > 0).sum()} of {len(summary)}")
    print("Unobserved under exact intersection:", ", ".join(summary.loc[summary.exact_route_count.eq(0), "chokepoint"]))
    print("Near misses within 25 km:", ", ".join(summary.loc[summary.sensitivity_status.str.startswith('near_miss'), "chokepoint"]) or "none")
    mindoro_routes = detail.loc[detail.chokepoint.eq("Mindoro Strait") & detail.within_25km]
    print(f"Mindoro candidate routes within 25 km: {len(mindoro_routes):,}")
    print(f"  within 10 km: {int(mindoro_routes.within_10km.sum()):,}")
    print(f"  between 10 and 25 km: {int((~mindoro_routes.within_10km).sum()):,}")
    print("Mapped sensitive chokepoints:", ", ".join(sensitive_names))
    print(f"Outputs: {output.resolve()}")


if __name__ == "__main__":
    main()
