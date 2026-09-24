"""Plot the observed LNG network in June 2021 on a geographic map.

Run from the repository root:
    python scripts/analysis/plot_monthly_network_maps.py

Or from any folder:
    python plot_monthly_network_maps.py --project-root PATH_TO_REPOSITORY

Requires pandas, numpy, matplotlib. The optional land outline is cached from
Natural Earth on the first run. Supply --land-geojson PATH to use a local copy.
The edges are schematic links between node coordinates, NOT sailed routes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd


LAND_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "master/geojson/ne_110m_land.geojson"
)
MONTH = "2021-06"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--land-geojson", type=Path, default=None)
    return parser.parse_args()


def load_inputs(root):
    data = root / "data" / "processed"
    nodes_path = data / "LNG_multilayer_nodes_observed.csv"
    edges_path = data / "LNG_multilayer_edges_monthly.csv"
    for path in (nodes_path, edges_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required input is missing: {path}")

    nodes = pd.read_csv(nodes_path, dtype={"node_id": str})
    edges = pd.read_csv(
        edges_path, dtype={"from_node_id": str, "to_node_id": str}
    )
    required_nodes = {"node_id", "node_type", "latitude", "longitude"}
    required_edges = {
        "period_month", "from_node_id", "to_node_id", "lng_flow_cmb"
    }
    for frame, required, path in (
        (nodes, required_nodes, nodes_path),
        (edges, required_edges, edges_path),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} lacks columns: {sorted(missing)}")
    if nodes.node_id.isna().any() or nodes.node_id.duplicated().any():
        raise ValueError("Node identifiers must be nonmissing and unique.")
    nodes["longitude"] = pd.to_numeric(nodes.longitude, errors="coerce")
    nodes["latitude"] = pd.to_numeric(nodes.latitude, errors="coerce")
    if nodes[["longitude", "latitude"]].isna().any().any():
        raise ValueError("Every observed node needs numeric coordinates.")
    if (~nodes.longitude.between(-180, 180)).any() or (
        ~nodes.latitude.between(-90, 90)
    ).any():
        raise ValueError("Node coordinates are outside longitude/latitude bounds.")
    edges["period_month"] = pd.to_datetime(edges.period_month, errors="raise")
    edges["lng_flow_cmb"] = pd.to_numeric(edges.lng_flow_cmb, errors="raise")
    if (edges.lng_flow_cmb <= 0).any() or edges.lng_flow_cmb.isna().any():
        raise ValueError("Monthly network edges must have positive LNG volume.")
    if not set(edges.from_node_id).union(edges.to_node_id) <= set(nodes.node_id):
        raise ValueError("An edge endpoint has no matching observed node.")
    return nodes.set_index("node_id"), edges


def land_geometry(root, supplied_path):
    path = supplied_path or root / "eda_outputs" / "network_level" / "land_110m.geojson"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            urlretrieve(LAND_URL, path)
        except Exception as exc:
            raise RuntimeError(
                "Could not download the land outline. Download the Natural Earth "
                "ne_110m_land.geojson file and pass --land-geojson PATH."
            ) from exc
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def add_land(ax, geojson):
    for feature in geojson["features"]:
        geometry = feature["geometry"]
        if geometry is None:
            continue
        if geometry["type"] == "Polygon":
            polygons = [geometry["coordinates"]]
        elif geometry["type"] == "MultiPolygon":
            polygons = geometry["coordinates"]
        else:
            continue
        for polygon in polygons:
            outer_ring = np.asarray(polygon[0], dtype=float)
            ax.fill(
                outer_ring[:, 0], outer_ring[:, 1],
                facecolor="#e9ecef", edgecolor="#b9c2cb", linewidth=0.35,
                zorder=0,
            )


def segment_at_dateline(lon1, lat1, lon2, lat2):
    """Draw the shorter schematic connection without a line across the map."""
    delta = lon2 - lon1
    if abs(delta) <= 180:
        return [[(lon1, lat1), (lon2, lat2)]]
    adjusted_lon2 = lon2 - 360 if delta > 180 else lon2 + 360
    boundary = 180 if adjusted_lon2 > lon1 else -180
    frac = (boundary - lon1) / (adjusted_lon2 - lon1)
    boundary_lat = lat1 + frac * (lat2 - lat1)
    opposite = -boundary
    return [
        [(lon1, lat1), (boundary, boundary_lat)],
        [(opposite, boundary_lat), (lon2, lat2)],
    ]


def draw_panel(ax, nodes, edges, land):
    add_land(ax, land)
    selected = edges.loc[edges.period_month.dt.strftime("%Y-%m").eq(MONTH)]
    if selected.empty:
        raise ValueError(f"There are no edges for {MONTH}.")
    # Keep counting compatible with monthly_network_structure.csv: one edge
    # per directed pair in each month. Source data are already aggregated.
    if selected.duplicated(["from_node_id", "to_node_id"]).any():
        raise ValueError(f"Duplicate directed edges found in {MONTH}.")
    active_ids = set(selected.from_node_id).union(selected.to_node_id)
    active_nodes = nodes.loc[sorted(active_ids)].copy()
    segments, widths = [], []
    arrows = []
    max_flow = selected.lng_flow_cmb.max()
    for edge in selected.itertuples(index=False):
        source = nodes.loc[edge.from_node_id]
        target = nodes.loc[edge.to_node_id]
        parts = segment_at_dateline(
            float(source.longitude), float(source.latitude),
            float(target.longitude), float(target.latitude),
        )
        width = 0.45 + 1.3 * np.sqrt(edge.lng_flow_cmb / max_flow)
        segments.extend(parts)
        widths.extend([width] * len(parts))
        # One arrow per directed edge, placed on its longest displayed segment.
        # For a link crossing the date line the two visible pieces retain the
        # source-to-target order and direction.
        longest = max(
            parts,
            key=lambda part: np.hypot(
                part[1][0] - part[0][0], part[1][1] - part[0][1]
            ),
        )
        start, end = np.asarray(longest[0]), np.asarray(longest[1])
        if np.hypot(*(end - start)) >= 3:
            arrows.append((start, end, width))
    ax.add_collection(LineCollection(
        segments, linewidths=widths, colors="#285f82",
        alpha=0.38, zorder=1,
    ))
    for start, end, width in arrows:
        p1 = start + 0.58 * (end - start)
        p2 = start + 0.73 * (end - start)
        ax.add_patch(FancyArrowPatch(
            p1, p2, arrowstyle="-|>", mutation_scale=9,
            linewidth=width, color="#285f82", alpha=0.65, zorder=1.3,
        ))
    for node_type, color, size in (
        ("terminal", "#e06c36", 15),
        ("chokepoint", "#4b3e9d", 28),
    ):
        group = active_nodes.loc[
            active_nodes.node_type.astype(str).str.lower().eq(node_type)
        ]
        ax.scatter(
            group.longitude, group.latitude, s=size, c=color,
            edgecolors="white", linewidths=0.3, zorder=2,
        )
    ax.set_xlim(-180, 180)
    ax.set_ylim(-65, 85)
    ax.set_aspect("auto")
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_yticks([-60, 0, 60])
    ax.tick_params(labelsize=8, colors="#48515c")
    ax.grid(color="white", linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_title(
        f"June 2021  |  {len(active_ids)} active nodes, "
        f"{len(selected)} directed edges", fontsize=11, loc="left", pad=8,
    )
    return len(active_ids), len(selected)


def main():
    args = parse_args()
    root = args.project_root.resolve()
    nodes, edges = load_inputs(root)
    land = land_geometry(root, args.land_geojson)
    output = root / "eda_outputs" / "network_level" / "maps"
    output.mkdir(parents=True, exist_ok=True)
    legend = [
        Line2D([0], [0], marker="o", linestyle="", markersize=6,
               markerfacecolor="#e06c36", markeredgecolor="white", label="Terminal"),
        Line2D([0], [0], marker="o", linestyle="", markersize=7,
               markerfacecolor="#4b3e9d", markeredgecolor="white", label="Chokepoint"),
        Line2D([0], [0], marker=">", color="#285f82", markersize=5,
               label="Directed connection"),
    ]
    fig, ax = plt.subplots(figsize=(12, 5.1), layout="constrained")
    counts = draw_panel(ax, nodes, edges, land)
    fig.legend(handles=legend, loc="outside lower center", ncol=3, frameon=False)
    target = output / "lng_network_2021-06.png"
    fig.savefig(target, dpi=300)
    plt.close(fig)
    print(f"June 2021: {counts[0]} active nodes, {counts[1]} directed edges")
    print(f"Saved {target}")
    print("Arrows show edge direction; lines are schematic, not sailed routes.")


if __name__ == "__main__":
    main()
