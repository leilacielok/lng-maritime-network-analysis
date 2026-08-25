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
        "File non trovato. Cercati i pattern:\n  - " + "\n  - ".join(patterns)
    )


def discover_inputs(folder: Path) -> dict[str, Path]:
    return {
        "nodes": find_one(folder, [
            "LNG_multilayer_nodes_v1.csv",
            "LNG_multilayer_nodes_v1*.csv",
        ]),
        "edges": find_one(folder, [
            "LNG_multilayer_edges_monthly_v1.csv",
            "LNG_multilayer_edges_monthly_v1*.csv",
        ]),
        "routes": find_one(folder, [
            "LNG_1037_routes_with_final_chokepoints_v1.geojson",
            "LNG_1037_routes_with_final_chokepoints_v1*.geojson",
        ]),
        "chokepoints": find_one(folder, [
            "PortWatch_28_chokepoints_geometry_working_v10.geojson",
            "PortWatch_28_chokepoints_geometry_working_v10*.geojson",
        ]),
    }


def parse_route_ids(value) -> list[str]:
    if pd.isna(value):
        return []
    return sorted({
        x.strip()
        for x in re.split(r"\s*;\s*", str(value))
        if x.strip()
    })


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
        raise ValueError(f"Colonne mancanti nel file nodi: {sorted(missing)}")

    if nodes[["latitude", "longitude"]].isna().any().any():
        bad = nodes[nodes[["latitude", "longitude"]].isna().any(axis=1)]
        raise ValueError(
            "Nodi senza coordinate: "
            + ", ".join(bad["node_id"].astype(str).tolist())
        )

    out = {}
    optional = [
        "layer", "terminal_role", "country", "region",
        "coordinate_source", "UN_LOCODE", "is_operating", "geometry_status",
    ]

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


def infer_global_monthly_lng(month_edges: pd.DataFrame) -> float | None:
    if "share_global_monthly_lng" not in month_edges.columns:
        return None

    valid = month_edges[
        month_edges["share_global_monthly_lng"].notna()
        & (month_edges["share_global_monthly_lng"] > 0)
        & month_edges["lng_flow_cmb"].notna()
    ]

    if valid.empty:
        return None

    ratios = (
        valid["lng_flow_cmb"].astype(float)
        / valid["share_global_monthly_lng"].astype(float)
    )
    return float(ratios.median())


def prepare_month_data(edges: pd.DataFrame, node_ids: set[str]):
    required = {
        "period_month", "from_node_id", "to_node_id", "edge_type",
        "route_count", "voyage_count", "lng_flow_cmb", "route_ids",
    }
    missing = required - set(edges.columns)
    if missing:
        raise ValueError(f"Colonne mancanti nel file archi: {sorted(missing)}")

    endpoint_ids = (
        set(edges["from_node_id"].astype(str))
        | set(edges["to_node_id"].astype(str))
    )
    missing_nodes = endpoint_ids - node_ids
    if missing_nodes:
        raise ValueError(
            f"{len(missing_nodes)} endpoint non presenti nel file nodi. "
            f"Esempi: {sorted(missing_nodes)[:10]}"
        )

    edges = edges.copy()
    edges["period_month"] = edges["period_month"].astype(str)
    months = sorted(edges["period_month"].dropna().unique().tolist())
    month_data = {}

    for month in months:
        df = edges[edges["period_month"] == month].copy()

        edge_records = []
        active_nodes = set()
        active_routes = set()

        for _, row in df.iterrows():
            from_id = str(row["from_node_id"])
            to_id = str(row["to_node_id"])
            route_ids = parse_route_ids(row["route_ids"])

            active_nodes.add(from_id)
            active_nodes.add(to_id)
            active_routes.update(route_ids)

            record = {
                "edge_period_id": clean_scalar(row.get("edge_period_id")),
                "from_node_id": from_id,
                "to_node_id": to_id,
                "edge_type": str(row["edge_type"]),
                "route_count": float(row["route_count"]) if pd.notna(row["route_count"]) else 0.0,
                "voyage_count": float(row["voyage_count"]) if pd.notna(row["voyage_count"]) else 0.0,
                "lng_flow_cmb": float(row["lng_flow_cmb"]) if pd.notna(row["lng_flow_cmb"]) else 0.0,
                "route_ids": route_ids,
            }

            for col in ["share_global_monthly_lng", "weight_row_normalized", "qa_watch_routes"]:
                if col in df.columns:
                    val = row[col]
                    if col in {"share_global_monthly_lng", "weight_row_normalized"}:
                        record[col] = float(val) if pd.notna(val) else 0.0
                    else:
                        record[col] = clean_scalar(val)

            edge_records.append(record)

        month_data[month] = {
            "edges": edge_records,
            "active_node_ids": sorted(active_nodes),
            "active_route_ids": sorted(active_routes),
            "stats": {
                "active_nodes": len(active_nodes),
                "active_edges": len(edge_records),
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
            f"{len(missing)} route_id usate nel network non hanno geometria. "
            f"Esempi: {sorted(missing)[:10]}"
        )


HTML_TEMPLATE = r'''<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LNG Dynamic Multilayer Network Dashboard</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>
html,body{margin:0;width:100%;height:100%;font-family:Arial,Helvetica,sans-serif;background:#eef1f4;overflow:hidden}
*{box-sizing:border-box}
#app{display:grid;grid-template-columns:310px 1fr;width:100%;height:100%}
#sidebar{background:#fff;border-right:1px solid #d9dde2;padding:18px;overflow-y:auto;z-index:1000}
#map{width:100%;height:100%;background:#dfe7ec}
h1{font-size:20px;line-height:1.25;margin:0 0 4px;color:#1f2933}
.subtitle{font-size:12px;color:#69757f;margin-bottom:18px;line-height:1.4}
.section{border-top:1px solid #e3e7ea;padding-top:15px;margin-top:15px}
.section-title{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:#7b8790;font-weight:700;margin-bottom:9px}
label{display:block;font-size:12px;font-weight:600;color:#37434d;margin-bottom:5px}
select,button,input[type="range"]{width:100%}
select,button{min-height:36px;border:1px solid #cfd6dc;border-radius:6px;background:#fff;color:#26313a;padding:7px 9px;font-size:13px}
button{cursor:pointer;font-weight:600}
button:hover{background:#f2f5f7}
button:disabled{opacity:.45;cursor:default}
.nav-row{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-top:8px}
.play-row{margin-top:7px}
#monthSlider{margin-top:10px;cursor:pointer}
.range-labels{display:flex;justify-content:space-between;font-size:10px;color:#8a949c;margin-top:1px}
.stats{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.stat-card{border:1px solid #e1e5e8;border-radius:7px;padding:10px;background:#fafbfc;min-height:69px}
.stat-value{font-size:18px;line-height:1.15;color:#202b33;font-weight:700}
.stat-label{font-size:9px;line-height:1.25;margin-top:5px;color:#75818a;text-transform:uppercase;letter-spacing:.06em}
.toggle{display:flex;align-items:center;gap:8px;margin:9px 0;color:#37434d;font-size:12px}
.toggle input{width:auto;margin:0}
.legend-item{display:flex;align-items:center;gap:8px;font-size:11px;color:#4d5962;margin:6px 0}
.legend-dot{width:10px;height:10px;border-radius:50%;flex:0 0 auto}
.legend-line{width:25px;height:4px;border-radius:2px;flex:0 0 auto}
.note,.metric-help{font-size:10px;color:#7d8891;line-height:1.4;margin-top:8px}
#monthBadge{position:absolute;top:12px;left:50%;transform:translateX(-50%);z-index:900;background:rgba(255,255,255,.94);border:1px solid #cfd5da;border-radius:6px;padding:8px 14px;font-size:13px;font-weight:700;color:#26313a;box-shadow:0 1px 4px rgba(0,0,0,.12);pointer-events:none}
.leaflet-popup-content{font-size:12px;line-height:1.45}
@media(max-width:760px){#app{grid-template-columns:1fr;grid-template-rows:330px 1fr}#sidebar{border-right:none;border-bottom:1px solid #d9dde2}#monthBadge{top:342px}}
</style>
</head>
<body>
<div id="app">
<aside id="sidebar">
<h1>LNG Dynamic Multilayer Network</h1>
<div class="subtitle">Geographic dashboard · 2020-01 to 2024-12</div>

<div class="section" style="border-top:none;padding-top:0;margin-top:0">
<div class="section-title">Periodo</div>
<label for="monthSelect">Month</label>
<select id="monthSelect"></select>
<input id="monthSlider" type="range" min="0" max="0" value="0" step="1">
<div class="range-labels"><span id="firstMonthLabel"></span><span id="lastMonthLabel"></span></div>
<div class="nav-row">
<button id="prevBtn" type="button">◀ Previous</button>
<button id="nextBtn" type="button">Next ▶</button>
</div>
<div class="play-row"><button id="playBtn" type="button">▶ Play</button></div>
</div>

<div class="section">
<div class="section-title">Monthly snapshot</div>
<div class="stats">
<div class="stat-card"><div id="statNodes" class="stat-value">-</div><div class="stat-label">Active nodes</div></div>
<div class="stat-card"><div id="statEdges" class="stat-value">-</div><div class="stat-label">Active edges</div></div>
<div class="stat-card"><div id="statRoutes" class="stat-value">-</div><div class="stat-label">Active routes</div></div>
<div class="stat-card"><div id="statLng" class="stat-value">-</div><div class="stat-label">Global LNG flow</div></div>
</div>
</div>

<div class="section">
<div class="section-title">Network</div>
<label for="metricSelect">Edge width metric</label>
<select id="metricSelect">
<option value="lng_flow_cmb">LNG flow</option>
<option value="voyage_count">Voyage count</option>
<option value="route_count">Route count</option>
<option value="weight_row_normalized">Normalized weight</option>
<option value="share_global_monthly_lng">Share of global monthly LNG</option>
</select>
<div class="metric-help">Lo spessore viene normalizzato nel mese selezionato.</div>
</div>

<div class="section">
<div class="section-title">Layers</div>
<label class="toggle"><input id="toggleTerminals" type="checkbox" checked> LNG terminals</label>
<label class="toggle"><input id="toggleCpNodes" type="checkbox" checked> Chokepoints</label>
<label class="toggle"><input id="toggleEdges" type="checkbox" checked> Network edges</label>
<label class="toggle"><input id="toggleRoutes" type="checkbox"> Physical SeaRoutes</label>
<label class="toggle"><input id="toggleCpGeom" type="checkbox"> Chokepoint geometries</label>
</div>

<div class="section">
<div class="section-title">Legend</div>
<div class="legend-item"><span class="legend-dot" style="background:#2471a3"></span>LNG export terminal</div>
<div class="legend-item"><span class="legend-dot" style="background:#148f77"></span>LNG import terminal</div>
<div class="legend-item"><span class="legend-dot" style="background:#c0392b"></span>Chokepoint</div>
<div class="legend-item"><span class="legend-line" style="background:#7f8c8d"></span>Terminal → terminal</div>
<div class="legend-item"><span class="legend-line" style="background:#8e44ad"></span>Terminal ↔ chokepoint</div>
<div class="legend-item"><span class="legend-line" style="background:#d35400"></span>Chokepoint → chokepoint</div>
<div class="note">Network edges e SeaRoute fisiche sono layer distinti: il flusso LNG è associato all'edge aggregato.</div>
</div>
</aside>

<main style="position:relative;min-width:0">
<div id="monthBadge"></div>
<div id="map"></div>
</main>
</div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const MONTHS=__MONTHS__;
const NODES=__NODES__;
const MONTH_DATA=__MONTH_DATA__;
const ROUTES_GEOJSON=__ROUTES_GEOJSON__;
const CHOKEPOINTS_GEOJSON=__CHOKEPOINTS_GEOJSON__;
const DEFAULT_MONTH=__DEFAULT_MONTH__;

const map=L.map("map",{worldCopyJump:true,preferCanvas:true}).setView([20,10],2);
L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",{
  attribution:"&copy; OpenStreetMap contributors &copy; CARTO",subdomains:"abcd",maxZoom:19
}).addTo(map);

const edgeLayer=L.layerGroup().addTo(map);
const terminalLayer=L.layerGroup().addTo(map);
const cpNodeLayer=L.layerGroup().addTo(map);
const routeLayer=L.layerGroup();
const cpGeomLayer=L.layerGroup();

const routeLookup=new Map();
for(const feature of ROUTES_GEOJSON.features||[]){
  const rid=String(feature?.properties?.route_id??"");
  if(rid)routeLookup.set(rid,feature);
}

L.geoJSON(CHOKEPOINTS_GEOJSON,{
  style:()=>({color:"#c0392b",weight:1.5,fillColor:"#e67e22",fillOpacity:.16,opacity:.75}),
  onEachFeature:(feature,layer)=>{
    const p=feature.properties||{};
    layer.bindTooltip(`<b>${esc(p.chokepoint??"")}</b><br>Node ID: ${esc(p.node_id??"")}<br>Geometry: ${esc(p.geometry_status??"")}`);
  }
}).addTo(cpGeomLayer);

const monthSelect=document.getElementById("monthSelect");
const monthSlider=document.getElementById("monthSlider");
const prevBtn=document.getElementById("prevBtn");
const nextBtn=document.getElementById("nextBtn");
const playBtn=document.getElementById("playBtn");
const metricSelect=document.getElementById("metricSelect");
const toggleTerminals=document.getElementById("toggleTerminals");
const toggleCpNodes=document.getElementById("toggleCpNodes");
const toggleEdges=document.getElementById("toggleEdges");
const toggleRoutes=document.getElementById("toggleRoutes");
const toggleCpGeom=document.getElementById("toggleCpGeom");
const statNodes=document.getElementById("statNodes");
const statEdges=document.getElementById("statEdges");
const statRoutes=document.getElementById("statRoutes");
const statLng=document.getElementById("statLng");
const monthBadge=document.getElementById("monthBadge");

document.getElementById("firstMonthLabel").textContent=MONTHS[0];
document.getElementById("lastMonthLabel").textContent=MONTHS[MONTHS.length-1];

for(const month of MONTHS){
  const option=document.createElement("option");
  option.value=month;
  option.textContent=formatMonth(month);
  monthSelect.appendChild(option);
}
monthSlider.max=String(MONTHS.length-1);

let currentMonthIndex=Math.max(0,MONTHS.indexOf(DEFAULT_MONTH));
let playTimer=null;

function esc(value){
  return String(value??"")
    .replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;")
    .replaceAll('"',"&quot;").replaceAll("'","&#039;");
}

function formatMonth(month){
  const [year,m]=month.split("-").map(Number);
  const names=["January","February","March","April","May","June","July","August","September","October","November","December"];
  return `${names[m-1]} ${year}`;
}

function formatNumber(value,digits=0){
  if(value===null||value===undefined||!Number.isFinite(Number(value)))return "–";
  return new Intl.NumberFormat("en-US",{maximumFractionDigits:digits,minimumFractionDigits:digits}).format(Number(value));
}

function formatCompact(value){
  if(value===null||value===undefined||!Number.isFinite(Number(value)))return "–";
  return new Intl.NumberFormat("en-US",{notation:"compact",maximumFractionDigits:1}).format(Number(value));
}

function nodeColor(node){
  if(node.node_type==="chokepoint")return "#c0392b";
  const role=String(node.terminal_role??"").toLowerCase();
  if(role.includes("export"))return "#2471a3";
  if(role.includes("import"))return "#148f77";
  return "#5d6d7e";
}

function edgeColor(type){
  return {
    "terminal-terminal":"#7f8c8d",
    "terminal-chokepoint":"#8e44ad",
    "chokepoint-terminal":"#8e44ad",
    "chokepoint-chokepoint":"#d35400"
  }[String(type)]||"#566573";
}

function scaleWidths(edges,metric){
  const values=edges.map(e=>Math.max(0,Number(e[metric]??0)||0));
  if(!values.length)return ()=>2;
  const logMetric=["lng_flow_cmb","voyage_count","route_count"].includes(metric);
  const t=values.map(v=>logMetric?Math.log1p(v):v);
  const min=Math.min(...t),max=Math.max(...t);
  return value=>{
    let x=Math.max(0,Number(value??0)||0);
    if(logMetric)x=Math.log1p(x);
    if(max<=min)return 3;
    const z=Math.max(0,Math.min(1,(x-min)/(max-min)));
    return 1.2+7*z;
  };
}

function normalizeLon(lon){
  lon=Number(lon);
  while(lon>180)lon-=360;
  while(lon<-180)lon+=360;
  return lon;
}

function splitAntimeridian(a,b){
  const lat1=Number(a[0]),lon1=normalizeLon(a[1]);
  const lat2=Number(b[0]),lon2=normalizeLon(b[1]);
  if(Math.abs(lon2-lon1)<=180)return [[[lat1,lon1],[lat2,lon2]]];
  let lon2a,boundary,other;
  if(lon1>0&&lon2<0){lon2a=lon2+360;boundary=180;other=-180;}
  else if(lon1<0&&lon2>0){lon2a=lon2-360;boundary=-180;other=180;}
  else return [[[lat1,lon1],[lat2,lon2]]];
  const denom=lon2a-lon1;
  if(denom===0)return [[[lat1,lon1],[lat2,lon2]]];
  const t=(boundary-lon1)/denom;
  const crossLat=lat1+t*(lat2-lat1);
  return [[[lat1,lon1],[crossLat,boundary]],[[crossLat,other],[lat2,lon2]]];
}

function metricLabel(metric){
  return {
    lng_flow_cmb:"LNG flow (cmb)",
    voyage_count:"Voyage count",
    route_count:"Route count",
    weight_row_normalized:"Normalized weight",
    share_global_monthly_lng:"Share global monthly LNG"
  }[metric]||metric;
}

function formatMetric(metric,value){
  value=Number(value??0);
  if(metric==="share_global_monthly_lng")return `${formatNumber(value*100,2)}%`;
  if(metric==="weight_row_normalized")return formatNumber(value,4);
  return formatNumber(value,0);
}

function drawNodes(data){
  terminalLayer.clearLayers();
  cpNodeLayer.clearLayers();
  for(const nodeId of data.active_node_ids){
    const node=NODES[nodeId];
    if(!node)continue;
    const isCp=node.node_type==="chokepoint";
    const color=nodeColor(node);
    const parts=[
      `<b>${esc(node.node_name)}</b>`,
      `Node ID: ${esc(node.node_id)}`,
      `Type: ${esc(node.node_type)}`
    ];
    if(node.terminal_role)parts.push(`Role: ${esc(node.terminal_role)}`);
    if(node.country)parts.push(`Country: ${esc(node.country)}`);
    if(node.region)parts.push(`Region: ${esc(node.region)}`);
    parts.push(`Lat/Lon: ${Number(node.latitude).toFixed(4)}, ${Number(node.longitude).toFixed(4)}`);
    const marker=L.circleMarker([node.latitude,node.longitude],{
      radius:isCp?6.5:3.8,color,weight:1.2,fillColor:color,fillOpacity:.9
    });
    marker.bindTooltip(`${esc(node.node_name)} (${esc(node.node_id)})`);
    marker.bindPopup(parts.join("<br>"));
    marker.addTo(isCp?cpNodeLayer:terminalLayer);
  }
}

function drawEdges(data){
  edgeLayer.clearLayers();
  const metric=metricSelect.value;
  const widthScale=scaleWidths(data.edges,metric);
  for(const edge of data.edges){
    const a=NODES[edge.from_node_id],b=NODES[edge.to_node_id];
    if(!a||!b)continue;
    const tooltip=
      `<b>${esc(a.node_name)} → ${esc(b.node_name)}</b><br>`+
      `Edge type: ${esc(edge.edge_type)}<br>`+
      `LNG flow (cmb): ${formatNumber(edge.lng_flow_cmb,0)}<br>`+
      `Voyages: ${formatNumber(edge.voyage_count,0)}<br>`+
      `Route count: ${formatNumber(edge.route_count,0)}<br>`+
      `${esc(metricLabel(metric))}: ${esc(formatMetric(metric,edge[metric]))}<br>`+
      `Route IDs: ${esc((edge.route_ids||[]).join("; "))}`;
    for(const segment of splitAntimeridian([a.latitude,a.longitude],[b.latitude,b.longitude])){
      const line=L.polyline(segment,{
        color:edgeColor(edge.edge_type),weight:widthScale(edge[metric]),opacity:.58
      });
      line.bindTooltip(tooltip,{sticky:true});
      line.addTo(edgeLayer);
    }
  }
}

function drawRoutes(data){
  routeLayer.clearLayers();
  const features=[];
  for(const routeId of data.active_route_ids){
    const f=routeLookup.get(routeId);
    if(f)features.push(f);
  }
  L.geoJSON({type:"FeatureCollection",features},{
    style:()=>({color:"#607d8b",weight:1.15,opacity:.34}),
    onEachFeature:(feature,layer)=>{
      const p=feature.properties||{};
      layer.bindTooltip(
        `<b>${esc(p.route_id??"")}</b><br>`+
        `${esc(p.from_terminal??"")} → ${esc(p.to_terminal??"")}<br>`+
        `Chokepoints: ${esc(p.chokepoints_final??"")}<br>`+
        `QA: ${esc(p.qa_status??"")}`
      );
    }
  }).addTo(routeLayer);
}

function updateStats(month,data){
  statNodes.textContent=formatNumber(data.stats.active_nodes,0);
  statEdges.textContent=formatNumber(data.stats.active_edges,0);
  statRoutes.textContent=formatNumber(data.stats.active_routes,0);
  statLng.textContent=data.stats.global_monthly_lng_cmb==null?"–":formatCompact(data.stats.global_monthly_lng_cmb);
  monthBadge.textContent=`LNG Dynamic Multilayer Network — ${formatMonth(month)}`;
}

function setLayerVisibility(layer,checkbox){
  if(checkbox.checked){
    if(!map.hasLayer(layer))map.addLayer(layer);
  }else{
    if(map.hasLayer(layer))map.removeLayer(layer);
  }
}

function applyMonth(index){
  currentMonthIndex=Math.max(0,Math.min(MONTHS.length-1,index));
  const month=MONTHS[currentMonthIndex];
  const data=MONTH_DATA[month];
  monthSelect.value=month;
  monthSlider.value=String(currentMonthIndex);
  drawNodes(data);
  drawEdges(data);
  drawRoutes(data);
  updateStats(month,data);
  setLayerVisibility(terminalLayer,toggleTerminals);
  setLayerVisibility(cpNodeLayer,toggleCpNodes);
  setLayerVisibility(edgeLayer,toggleEdges);
  setLayerVisibility(routeLayer,toggleRoutes);
  setLayerVisibility(cpGeomLayer,toggleCpGeom);
  prevBtn.disabled=currentMonthIndex===0;
  nextBtn.disabled=currentMonthIndex===MONTHS.length-1;
}

function stopPlay(){
  if(playTimer!==null){
    clearInterval(playTimer);
    playTimer=null;
  }
  playBtn.textContent="▶ Play";
}

function startPlay(){
  if(playTimer!==null)return;
  playBtn.textContent="⏸ Pause";
  playTimer=setInterval(()=>{
    let next=currentMonthIndex+1;
    if(next>=MONTHS.length)next=0;
    applyMonth(next);
  },1100);
}

monthSelect.addEventListener("change",()=>{
  stopPlay();
  const idx=MONTHS.indexOf(monthSelect.value);
  if(idx>=0)applyMonth(idx);
});
monthSlider.addEventListener("input",()=>{
  stopPlay();
  applyMonth(Number(monthSlider.value));
});
prevBtn.addEventListener("click",()=>{stopPlay();applyMonth(currentMonthIndex-1);});
nextBtn.addEventListener("click",()=>{stopPlay();applyMonth(currentMonthIndex+1);});
playBtn.addEventListener("click",()=>{playTimer===null?startPlay():stopPlay();});
metricSelect.addEventListener("change",()=>drawEdges(MONTH_DATA[MONTHS[currentMonthIndex]]));
toggleTerminals.addEventListener("change",()=>setLayerVisibility(terminalLayer,toggleTerminals));
toggleCpNodes.addEventListener("change",()=>setLayerVisibility(cpNodeLayer,toggleCpNodes));
toggleEdges.addEventListener("change",()=>setLayerVisibility(edgeLayer,toggleEdges));
toggleRoutes.addEventListener("change",()=>setLayerVisibility(routeLayer,toggleRoutes));
toggleCpGeom.addEventListener("change",()=>setLayerVisibility(cpGeomLayer,toggleCpGeom));

applyMonth(currentMonthIndex);
setTimeout(()=>map.invalidateSize(),50);
</script>
</body>
</html>
'''


def build_dashboard(folder: Path, output: Path, default_month: str | None = None) -> Path:
    paths = discover_inputs(folder)

    print("Files:")
    for key, path in paths.items():
        print(f"  {key:12s}: {path.name}")

    nodes_df = pd.read_csv(paths["nodes"])
    edges_df = pd.read_csv(paths["edges"], dtype={"period_month": str})

    with open(paths["routes"], "r", encoding="utf-8") as f:
        routes_geojson = json.load(f)

    with open(paths["chokepoints"], "r", encoding="utf-8") as f:
        chokepoints_geojson = json.load(f)

    nodes = prepare_nodes(nodes_df)
    months, month_data = prepare_month_data(edges_df, set(nodes.keys()))

    if not months:
        raise ValueError("Nessun mese trovato nel file degli archi.")

    validate_routes(routes_geojson, month_data)

    if default_month is None:
        default_month = months[-1]

    if default_month not in months:
        raise ValueError(
            f"Default month {default_month!r} non presente. "
            f"Intervallo disponibile: {months[0]} - {months[-1]}"
        )

    html = HTML_TEMPLATE
    replacements = {
        "__MONTHS__": json.dumps(months, ensure_ascii=False, separators=(",", ":")),
        "__NODES__": json.dumps(nodes, ensure_ascii=False, separators=(",", ":")),
        "__MONTH_DATA__": json.dumps(month_data, ensure_ascii=False, separators=(",", ":")),
        "__ROUTES_GEOJSON__": json.dumps(routes_geojson, ensure_ascii=False, separators=(",", ":")),
        "__CHOKEPOINTS_GEOJSON__": json.dumps(chokepoints_geojson, ensure_ascii=False, separators=(",", ":")),
        "__DEFAULT_MONTH__": json.dumps(default_month),
    }

    for placeholder, value in replacements.items():
        html = html.replace(placeholder, value)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")

    print("\nDashboard summary:")
    print(f"  Months: {len(months)} ({months[0]} -> {months[-1]})")
    print(f"  Nodes: {len(nodes)}")
    print(f"  Edge-month rows: {len(edges_df)}")
    print(f"  Route geometries: {len(routes_geojson.get('features', []))}")
    print(f"  Chokepoint geometries: {len(chokepoints_geojson.get('features', []))}")
    print(f"  Default month: {default_month}")
    print(f"  Output: {output}")
    print("\nLa basemap e Leaflet sono caricati da internet, quindi l'HTML richiede connessione.")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Crea la dashboard geografica interattiva del LNG network."
    )
    parser.add_argument(
        "--folder",
        default=".",
        help="Cartella contenente CSV e GeoJSON (default: cartella corrente).",
    )
    parser.add_argument(
        "--output",
        default="LNG_network_dashboard.html",
        help="Nome/percorso HTML di output.",
    )
    parser.add_argument(
        "--default-month",
        default=None,
        help="Mese mostrato all'apertura, YYYY-MM. Se omesso: ultimo mese disponibile.",
    )

    args = parser.parse_args()
    folder = Path(args.folder).expanduser().resolve()
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = folder / output
    output = output.resolve()

    build_dashboard(folder, output, args.default_month)


if __name__ == "__main__":
    main()
