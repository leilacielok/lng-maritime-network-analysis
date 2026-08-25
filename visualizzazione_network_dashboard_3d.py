#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def find_one(folder: Path, patterns: list[str]) -> Path:
    for pattern in patterns:
        matches = sorted(folder.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        "File not found. Searched patterns:\n  - " + "\n  - ".join(patterns)
    )


def discover_inputs(folder: Path) -> dict[str, Path]:
    data_folder = folder / "data"

    if not data_folder.exists():
        raise FileNotFoundError(
            f"Data folder not found: {data_folder}\n"
            "Create a 'data' folder in the project root and place the datasets inside it."
        )

    return {
        "nodes": find_one(data_folder, [
            "LNG_multilayer_nodes_v1.csv",
            "LNG_multilayer_nodes_v1*.csv",
        ]),
        "edges": find_one(data_folder, [
            "LNG_multilayer_edges_monthly_v1.csv",
            "LNG_multilayer_edges_monthly_v1*.csv",
        ]),
        "routes": find_one(data_folder, [
            "LNG_1037_routes_with_final_chokepoints_v1.geojson",
            "LNG_1037_routes_with_final_chokepoints_v1*.geojson",
        ]),
    }


def parse_route_ids(value) -> list[str]:
    if pd.isna(value):
        return []
    return sorted({x.strip() for x in re.split(r"\s*;\s*", str(value)) if x.strip()})


def clean_scalar(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def prepare_nodes(nodes: pd.DataFrame) -> dict[str, dict]:
    required = {"node_id", "node_name", "node_type", "latitude", "longitude"}
    missing = required - set(nodes.columns)
    if missing:
        raise ValueError(f"Missing columns in the nodes file: {sorted(missing)}")

    if nodes[["latitude", "longitude"]].isna().any().any():
        bad = nodes[nodes[["latitude", "longitude"]].isna().any(axis=1)]
        raise ValueError(
            "Nodes without coordinates: " + ", ".join(bad["node_id"].astype(str).tolist())
        )

    optional = ["layer", "terminal_role", "country", "region", "coordinate_source", "UN_LOCODE"]
    out = {}
    for _, row in nodes.iterrows():
        node_id = str(row["node_id"])
        record = {
            "node_id": node_id,
            "node_name": str(row["node_name"]),
            "node_type": str(row["node_type"]),
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
        }
        for col in optional:
            if col in nodes.columns:
                record[col] = clean_scalar(row[col])
        out[node_id] = record
    return out


def infer_global_monthly_lng(df: pd.DataFrame) -> float | None:
    if "share_global_monthly_lng" not in df.columns:
        return None
    valid = df[
        df["share_global_monthly_lng"].notna()
        & (df["share_global_monthly_lng"] > 0)
        & df["lng_flow_cmb"].notna()
    ]
    if valid.empty:
        return None
    ratios = valid["lng_flow_cmb"].astype(float) / valid["share_global_monthly_lng"].astype(float)
    return float(ratios.median())


def prepare_month_data(edges: pd.DataFrame, node_ids: set[str]):
    required = {
        "period_month", "from_node_id", "to_node_id", "edge_type",
        "route_count", "voyage_count", "lng_flow_cmb", "route_ids",
    }
    missing = required - set(edges.columns)
    if missing:
        raise ValueError(f"Missing columns in the edges file: {sorted(missing)}")

    edges = edges.copy()
    edges["period_month"] = edges["period_month"].astype(str)
    endpoint_ids = set(edges["from_node_id"].astype(str)) | set(edges["to_node_id"].astype(str))
    missing_nodes = endpoint_ids - node_ids
    if missing_nodes:
        raise ValueError(
            f"{len(missing_nodes)} endpoints are not present in the nodes file. "
            f"Examples: {sorted(missing_nodes)[:10]}"
        )

    months = sorted(edges["period_month"].dropna().unique().tolist())
    month_data = {}

    for month in months:
        df = edges[edges["period_month"] == month].copy()
        active_nodes = set()
        active_routes = set()
        records = []

        for _, row in df.iterrows():
            from_id = str(row["from_node_id"])
            to_id = str(row["to_node_id"])
            route_ids = parse_route_ids(row["route_ids"])
            active_nodes.update([from_id, to_id])
            active_routes.update(route_ids)

            rec = {
                "from_node_id": from_id,
                "to_node_id": to_id,
                "edge_type": str(row["edge_type"]),
                "route_count": float(row["route_count"]) if pd.notna(row["route_count"]) else 0.0,
                "voyage_count": float(row["voyage_count"]) if pd.notna(row["voyage_count"]) else 0.0,
                "lng_flow_cmb": float(row["lng_flow_cmb"]) if pd.notna(row["lng_flow_cmb"]) else 0.0,
                "route_ids": route_ids,
                "weight_row_normalized": float(row["weight_row_normalized"]) if "weight_row_normalized" in df.columns and pd.notna(row["weight_row_normalized"]) else 0.0,
                "share_global_monthly_lng": float(row["share_global_monthly_lng"]) if "share_global_monthly_lng" in df.columns and pd.notna(row["share_global_monthly_lng"]) else 0.0,
            }
            records.append(rec)

        month_data[month] = {
            "edges": records,
            "active_node_ids": sorted(active_nodes),
            "active_route_ids": sorted(active_routes),
            "stats": {
                "active_nodes": len(active_nodes),
                "active_edges": len(records),
                "active_routes": len(active_routes),
                "global_monthly_lng_cmb": infer_global_monthly_lng(df),
            },
        }

    return months, month_data


def validate_routes(routes_geojson: dict, month_data: dict) -> None:
    route_geo_ids = {
        str(feature.get("properties", {}).get("route_id"))
        for feature in routes_geojson.get("features", [])
    }
    used_ids = set()
    for data in month_data.values():
        used_ids.update(data["active_route_ids"])
    missing = used_ids - route_geo_ids
    if missing:
        raise ValueError(
            f"{len(missing)} route_id values used in the network have no geometry. "
            f"Examples: {sorted(missing)[:10]}"
        )


HTML_TEMPLATE = r'''<!doctype html>
<html lang="it">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LNG Dynamic Multilayer Network 3D</title>
<style>
html,body{margin:0;width:100%;height:100%;overflow:hidden;font-family:Arial,Helvetica,sans-serif;background:#07111f;color:#eef4f8}
*{box-sizing:border-box}
#app{display:grid;grid-template-columns:320px 1fr;width:100%;height:100%}
#sidebar{background:#0d1826;border-right:1px solid rgba(255,255,255,.08);padding:18px;overflow:auto;z-index:20}
#globeWrap{position:relative;min-width:0;height:100%;background:#03070d}
#globeViz{width:100%;height:100%}
h1{font-size:20px;margin:0 0 4px}.subtitle{font-size:12px;color:#9eb1c4;margin-bottom:18px}
.section{border-top:1px solid rgba(255,255,255,.08);padding-top:14px;margin-top:14px}.section:first-of-type{border-top:0;padding-top:0;margin-top:0}
.section-title{font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:#8da0b4;font-weight:700;margin-bottom:9px}
label{display:block;font-size:12px;font-weight:600;margin-bottom:5px}select,button,input[type=range]{width:100%}
select,button{min-height:36px;border:1px solid rgba(255,255,255,.12);border-radius:6px;background:#152232;color:#eef4f8;padding:7px 9px;font-size:13px}
button{cursor:pointer;font-weight:600}.nav{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-top:8px}.play{margin-top:7px}
.range-labels{display:flex;justify-content:space-between;font-size:10px;color:#8094a8}.stats{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.card{border:1px solid rgba(255,255,255,.08);border-radius:7px;padding:10px;background:rgba(255,255,255,.035)}.value{font-size:18px;font-weight:700}.label{font-size:9px;color:#88a0b6;text-transform:uppercase;margin-top:4px}
.toggle{display:flex;gap:8px;align-items:center;margin:9px 0;font-size:12px;font-weight:400}.toggle input{width:auto;margin:0}
#monthBadge{position:absolute;top:14px;left:50%;transform:translateX(-50%);z-index:10;background:rgba(8,17,29,.82);border:1px solid rgba(255,255,255,.12);border-radius:7px;padding:8px 14px;font-weight:700;pointer-events:none}
#hint{position:absolute;right:14px;top:14px;z-index:10;background:rgba(8,17,29,.82);border:1px solid rgba(255,255,255,.12);border-radius:7px;padding:10px 12px;font-size:10px;line-height:1.5;color:#a9bac9}
.note{font-size:10px;color:#8194a7;line-height:1.4;margin-top:10px}
@media(max-width:800px){#app{grid-template-columns:1fr;grid-template-rows:350px 1fr}#sidebar{border-right:0;border-bottom:1px solid rgba(255,255,255,.08)}#hint{display:none}}
</style>
</head>
<body>
<div id="app">
<aside id="sidebar">
<h1>LNG Dynamic Multilayer Network</h1><div class="subtitle">3D Geographic Dashboard</div>
<div class="section"><div class="section-title">Period</div><label for="monthSelect">Month</label><select id="monthSelect"></select><input id="monthSlider" type="range" min="0" max="0" value="0" step="1"><div class="range-labels"><span id="firstMonthLabel"></span><span id="lastMonthLabel"></span></div><div class="nav"><button id="prevBtn">◀ Previous</button><button id="nextBtn">Next ▶</button></div><div class="play"><button id="playBtn">▶ Play</button></div></div>
<div class="section"><div class="section-title">Monthly snapshot</div><div class="stats"><div class="card"><div id="statNodes" class="value">-</div><div class="label">Active nodes</div></div><div class="card"><div id="statEdges" class="value">-</div><div class="label">Active edges</div></div><div class="card"><div id="statRoutes" class="value">-</div><div class="label">Active routes</div></div><div class="card"><div id="statLng" class="value">-</div><div class="label">Global LNG flow</div></div></div></div>
<div class="section"><div class="section-title">Edge size</div><label for="metricSelect">Metric</label><select id="metricSelect"><option value="lng_flow_cmb">LNG flow</option><option value="voyage_count">Voyage count</option><option value="route_count">Route count</option><option value="weight_row_normalized">Normalized weight</option><option value="share_global_monthly_lng">Share global monthly LNG</option></select></div>
<div class="section"><div class="section-title">Layers</div><label class="toggle"><input id="toggleTerminals" type="checkbox" checked> LNG terminals</label><label class="toggle"><input id="toggleCpNodes" type="checkbox" checked> Chokepoints</label><label class="toggle"><input id="toggleEdges" type="checkbox" checked> Network edges</label><label class="toggle"><input id="toggleRoutes" type="checkbox"> Physical SeaRoutes</label></div>
</aside>
<main id="globeWrap"><div id="monthBadge"></div><div id="hint"><b>Controls</b><br>Drag: rotate<br>Mouse wheel: zoom<br>Double-click: reset view</div><div id="globeViz"></div></main>
</div>
<script src="https://unpkg.com/three@0.160.0/build/three.min.js"></script>
<script src="https://unpkg.com/globe.gl@2.32.2/dist/globe.gl.min.js"></script>
<script>
const MONTHS=__MONTHS__;
const NODES=__NODES__;
const MONTH_DATA=__MONTH_DATA__;
const ROUTES_GEOJSON=__ROUTES_GEOJSON__;
const DEFAULT_MONTH=__DEFAULT_MONTH__;

const monthSelect=document.getElementById('monthSelect');
const monthSlider=document.getElementById('monthSlider');
const prevBtn=document.getElementById('prevBtn');
const nextBtn=document.getElementById('nextBtn');
const playBtn=document.getElementById('playBtn');
const metricSelect=document.getElementById('metricSelect');
const toggleTerminals=document.getElementById('toggleTerminals');
const toggleCpNodes=document.getElementById('toggleCpNodes');
const toggleEdges=document.getElementById('toggleEdges');
const toggleRoutes=document.getElementById('toggleRoutes');
const statNodes=document.getElementById('statNodes');
const statEdges=document.getElementById('statEdges');
const statRoutes=document.getElementById('statRoutes');
const statLng=document.getElementById('statLng');
const monthBadge=document.getElementById('monthBadge');

document.getElementById('firstMonthLabel').textContent=MONTHS[0];
document.getElementById('lastMonthLabel').textContent=MONTHS[MONTHS.length-1];
for(const m of MONTHS){const o=document.createElement('option');o.value=m;o.textContent=formatMonth(m);monthSelect.appendChild(o)}
monthSlider.max=String(MONTHS.length-1);

let currentMonthIndex=Math.max(0,MONTHS.indexOf(DEFAULT_MONTH));
let playTimer=null;

function formatMonth(month){const [y,m]=month.split('-').map(Number);const n=['January','February','March','April','May','June','July','August','September','October','November','December'];return `${n[m-1]} ${y}`}
function formatNumber(v,d=0){if(v==null||!Number.isFinite(Number(v)))return '–';return new Intl.NumberFormat('en-US',{maximumFractionDigits:d,minimumFractionDigits:d}).format(Number(v))}
function formatCompact(v){if(v==null||!Number.isFinite(Number(v)))return '–';return new Intl.NumberFormat('en-US',{notation:'compact',maximumFractionDigits:1}).format(Number(v))}
function nodeColor(n){if(n.node_type==='chokepoint')return '#e64b3c';const r=String(n.terminal_role??'').toLowerCase();if(r.includes('export'))return '#2f80ed';if(r.includes('import'))return '#48b36b';return '#7b8a97'}
function edgeColor(t){return {'terminal-terminal':'#97a3ad','terminal-chokepoint':'#a65ad7','chokepoint-terminal':'#a65ad7','chokepoint-chokepoint':'#e67e22'}[String(t)]||'#8f9ba5'}
function widthScale(edges,metric){const vals=edges.map(e=>Math.max(0,Number(e[metric]??0)||0));if(!vals.length)return()=>0.15;const log=['lng_flow_cmb','voyage_count','route_count'].includes(metric);const t=vals.map(v=>log?Math.log1p(v):v);const min=Math.min(...t),max=Math.max(...t);return v=>{let x=Math.max(0,Number(v??0)||0);if(log)x=Math.log1p(x);if(max<=min)return 0.25;const z=Math.max(0,Math.min(1,(x-min)/(max-min)));return 0.10+0.55*z}}

const routeLookup=new Map();
for(const f of ROUTES_GEOJSON.features||[]){const rid=String(f?.properties?.route_id??'');if(rid)routeLookup.set(rid,f)}
function flattenRouteFeature(feature){const g=feature.geometry||{};const out=[];if(g.type==='LineString'){out.push({points:g.coordinates.map(c=>({lat:c[1],lng:c[0]})),properties:feature.properties||{}})}else if(g.type==='MultiLineString'){for(const line of g.coordinates){out.push({points:line.map(c=>({lat:c[1],lng:c[0]})),properties:feature.properties||{}})}}return out}

const globe=Globe()(document.getElementById('globeViz'))
  .backgroundColor('#03070d')
  .globeImageUrl('https://unpkg.com/three-globe/example/img/earth-blue-marble.jpg')
  .bumpImageUrl('https://unpkg.com/three-globe/example/img/earth-topology.png')
  .showAtmosphere(true)
  .atmosphereColor('#4ca8ff')
  .atmosphereAltitude(0.15)
  .pointLat('lat').pointLng('lng')
  .pointAltitude(d=>d.node_type==='chokepoint'?0.025:0.015)
  .pointRadius(d=>d.node_type==='chokepoint'?0.42:0.28)
  .pointColor('color')
  .pointLabel(d=>`<b>${d.node_name}</b><br>Node ID: ${d.node_id}<br>Type: ${d.node_type}<br>${d.country?`Country: ${d.country}<br>`:''}Lat/Lon: ${Number(d.lat).toFixed(4)}, ${Number(d.lng).toFixed(4)}`)
  .arcStartLat('startLat').arcStartLng('startLng').arcEndLat('endLat').arcEndLng('endLng')
  .arcColor('color').arcStroke('stroke').arcAltitudeAutoScale(0.35)
  .arcLabel(d=>`<b>${d.from_name} → ${d.to_name}</b><br>Edge type: ${d.edge_type}<br>LNG flow: ${formatNumber(d.lng_flow_cmb)} cmb<br>Voyages: ${formatNumber(d.voyage_count)}<br>Route count: ${formatNumber(d.route_count)}`)
  .pathPoints('points').pathPointLat('lat').pathPointLng('lng')
  .pathColor(()=> '#8aa0b3').pathStroke(0.18).pathDashLength(0.7).pathDashGap(0.3)
  .pathLabel(d=>{const p=d.properties||{};return `<b>${p.route_id??''}</b><br>${p.from_terminal??''} → ${p.to_terminal??''}<br>Chokepoints: ${p.chokepoints_final??''}`});

globe.controls().autoRotate=false;
globe.controls().enableDamping=true;
globe.pointOfView({lat:18,lng:15,altitude:2.15},0);
document.getElementById('globeWrap').addEventListener('dblclick',()=>globe.pointOfView({lat:18,lng:15,altitude:2.15},700));

function makePointData(data){const pts=[];for(const id of data.active_node_ids){const n=NODES[id];if(!n)continue;if(n.node_type==='terminal'&&!toggleTerminals.checked)continue;if(n.node_type==='chokepoint'&&!toggleCpNodes.checked)continue;pts.push({...n,lat:n.latitude,lng:n.longitude,color:nodeColor(n)})}return pts}
function makeArcData(data){if(!toggleEdges.checked)return[];const metric=metricSelect.value;const scale=widthScale(data.edges,metric);const arcs=[];for(const e of data.edges){const a=NODES[e.from_node_id],b=NODES[e.to_node_id];if(!a||!b)continue;arcs.push({...e,startLat:a.latitude,startLng:a.longitude,endLat:b.latitude,endLng:b.longitude,from_name:a.node_name,to_name:b.node_name,color:edgeColor(e.edge_type),stroke:scale(e[metric])})}return arcs}
function makeRouteData(data){if(!toggleRoutes.checked)return[];const out=[];for(const rid of data.active_route_ids){const f=routeLookup.get(rid);if(f)out.push(...flattenRouteFeature(f))}return out}
function updateStats(month,data){statNodes.textContent=formatNumber(data.stats.active_nodes);statEdges.textContent=formatNumber(data.stats.active_edges);statRoutes.textContent=formatNumber(data.stats.active_routes);statLng.textContent=data.stats.global_monthly_lng_cmb==null?'–':formatCompact(data.stats.global_monthly_lng_cmb);monthBadge.textContent=`LNG Dynamic Multilayer Network — ${formatMonth(month)}`}
function applyMonth(index){currentMonthIndex=Math.max(0,Math.min(MONTHS.length-1,index));const month=MONTHS[currentMonthIndex],data=MONTH_DATA[month];monthSelect.value=month;monthSlider.value=String(currentMonthIndex);globe.pointsData(makePointData(data));globe.arcsData(makeArcData(data));globe.pathsData(makeRouteData(data));updateStats(month,data);prevBtn.disabled=currentMonthIndex===0;nextBtn.disabled=currentMonthIndex===MONTHS.length-1}
function stopPlay(){if(playTimer!==null){clearInterval(playTimer);playTimer=null}playBtn.textContent='▶ Play'}
function startPlay(){if(playTimer!==null)return;playBtn.textContent='⏸ Pause';playTimer=setInterval(()=>{let n=currentMonthIndex+1;if(n>=MONTHS.length)n=0;applyMonth(n)},1200)}

monthSelect.addEventListener('change',()=>{stopPlay();const i=MONTHS.indexOf(monthSelect.value);if(i>=0)applyMonth(i)});
monthSlider.addEventListener('input',()=>{stopPlay();applyMonth(Number(monthSlider.value))});
prevBtn.addEventListener('click',()=>{stopPlay();applyMonth(currentMonthIndex-1)});
nextBtn.addEventListener('click',()=>{stopPlay();applyMonth(currentMonthIndex+1)});
playBtn.addEventListener('click',()=>playTimer===null?startPlay():stopPlay());
metricSelect.addEventListener('change',()=>applyMonth(currentMonthIndex));
toggleTerminals.addEventListener('change',()=>applyMonth(currentMonthIndex));
toggleCpNodes.addEventListener('change',()=>applyMonth(currentMonthIndex));
toggleEdges.addEventListener('change',()=>applyMonth(currentMonthIndex));
toggleRoutes.addEventListener('change',()=>applyMonth(currentMonthIndex));

function resizeGlobe(){const w=document.getElementById('globeWrap');globe.width(w.clientWidth).height(w.clientHeight)}
window.addEventListener('resize',resizeGlobe);
resizeGlobe();
applyMonth(currentMonthIndex);
</script>
</body>
</html>
'''


def build_dashboard(folder: Path, output: Path, default_month: str | None = None) -> Path:
    paths = discover_inputs(folder)
    print("Files:")
    for key, path in paths.items():
        print(f"  {key:12s}: {path.relative_to(folder)}")

    nodes_df = pd.read_csv(paths["nodes"])
    edges_df = pd.read_csv(paths["edges"], dtype={"period_month": str})
    with open(paths["routes"], "r", encoding="utf-8") as f:
        routes_geojson = json.load(f)

    nodes = prepare_nodes(nodes_df)
    months, month_data = prepare_month_data(edges_df, set(nodes.keys()))
    validate_routes(routes_geojson, month_data)

    if not months:
        raise ValueError("No months found.")
    if default_month is None:
        default_month = months[-1]
    if default_month not in months:
        raise ValueError(
            f"Mese {default_month!r} not available. Range: {months[0]} - {months[-1]}"
        )

    html = HTML_TEMPLATE
    replacements = {
        "__MONTHS__": json.dumps(months, ensure_ascii=False, separators=(",", ":")),
        "__NODES__": json.dumps(nodes, ensure_ascii=False, separators=(",", ":")),
        "__MONTH_DATA__": json.dumps(month_data, ensure_ascii=False, separators=(",", ":")),
        "__ROUTES_GEOJSON__": json.dumps(routes_geojson, ensure_ascii=False, separators=(",", ":")),
        "__DEFAULT_MONTH__": json.dumps(default_month),
    }
    for placeholder, value in replacements.items():
        html = html.replace(placeholder, value)

    output.write_text(html, encoding="utf-8")

    print("\nDashboard 3D summary:")
    print(f"  Months: {len(months)} ({months[0]} -> {months[-1]})")
    print(f"  Nodes: {len(nodes)}")
    print(f"  Edge-month rows: {len(edges_df)}")
    print(f"  Route geometries: {len(routes_geojson.get('features', []))}")
    print(f"  Default month: {default_month}")
    print(f"  Output: {output}")
    print("\nNote: the 3D globe uses Globe.gl/Three.js e texture online; serve connessione internet.")
    return output


def main():
    parser = argparse.ArgumentParser(description="Create the 3D geographic dashboard del LNG network.")
    parser.add_argument("--folder", default=".")
    parser.add_argument("--output", default="LNG_network_dashboard_3d.html")
    parser.add_argument("--default-month", default=None)
    args = parser.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = folder / output
    output = output.resolve()

    build_dashboard(folder, output, args.default_month)


if __name__ == "__main__":
    main()
