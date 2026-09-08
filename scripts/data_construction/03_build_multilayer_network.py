"""Build the monthly LNG terminal-chokepoint network from matched voyages.

Inputs (by default, in a ``data`` folder beside this script):
  - LNG_voyage_node_matching.xlsx, sheet ``Matched Voyages``
  - LNG_multilayer_nodes.csv
  - LNG_1037_routes_with_final_chokepoints.geojson

Outputs:
  - LNG_multilayer_edges_monthly.csv
  - LNG_multilayer_monthly_QA.csv

Month assignment uses the voyage departure date. Return/ballast movements are
excluded. Exact duplicate laden voyages are removed before aggregation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


DEFAULT_VOYAGES = "LNG_voyage_node_matching.xlsx"
DEFAULT_NODES = "LNG_multilayer_nodes.csv"
DEFAULT_ROUTES = "LNG_1037_routes_with_final_chokepoints.geojson"
DEFAULT_EDGES = "LNG_multilayer_edges_monthly.csv"
DEFAULT_MONTHLY_QA = "LNG_multilayer_monthly_QA.csv"

DATE_COLUMN = "start_date"
DUPLICATE_COLUMNS = [
    "start_date",
    "end_date",
    "IMO",
    "voyage",
    "from_terminal",
    "to_terminal",
    "amount_cmb",
]


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"

    parser = argparse.ArgumentParser(
        description=(
            "Construct directed monthly LNG network edges by propagating each "
            "laden voyage along its ordered terminal-chokepoint path."
        )
    )
    parser.add_argument(
        "--voyages",
        type=Path,
        default=data_dir / DEFAULT_VOYAGES,
        help="Matched-voyage workbook.",
    )
    parser.add_argument(
        "--nodes",
        type=Path,
        default=data_dir / DEFAULT_NODES,
        help="Multilayer node table.",
    )
    parser.add_argument(
        "--routes",
        type=Path,
        default=data_dir / DEFAULT_ROUTES,
        help="Route GeoJSON with final chokepoint assignments.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=data_dir,
        help="Directory for the two output CSV files.",
    )
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(sorted(missing))}")


def haversine_radians(point_a: list[float] | tuple[float, float],
                      point_b: list[float] | tuple[float, float]) -> float:
    """Angular great-circle distance; sufficient for ordering route geometry."""
    lon1, lat1 = map(math.radians, point_a[:2])
    lon2, lat2 = map(math.radians, point_b[:2])
    delta_lon = (lon2 - lon1 + math.pi) % (2 * math.pi) - math.pi
    delta_lat = lat2 - lat1
    value = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
    )
    return 2 * math.asin(min(1.0, math.sqrt(value)))


def stitch_multiline(
    lines: list[list[list[float]]],
    origin: tuple[float, float],
) -> list[list[float]]:
    """Orient and join MultiLineString components from origin to destination.

    SeaRoute splits antimeridian-crossing paths at -180/+180 degrees. Each
    component can have its own orientation, so reversing the complete flattened
    coordinate array is not sufficient. Components are attached one at a time
    using the closest endpoint under great-circle distance.
    """
    remaining = [list(line) for line in lines if line]
    if not remaining:
        raise ValueError("Route contains no coordinates.")

    ordered: list[list[float]] = []
    current: list[float] | tuple[float, float] = origin

    while remaining:
        candidates: list[tuple[float, int, bool]] = []
        for index, line in enumerate(remaining):
            candidates.append((haversine_radians(current, line[0]), index, False))
            candidates.append((haversine_radians(current, line[-1]), index, True))

        _, index, reverse = min(candidates)
        component = remaining.pop(index)
        if reverse:
            component.reverse()

        if ordered and haversine_radians(ordered[-1], component[0]) < 1e-12:
            ordered.extend(component[1:])
        else:
            ordered.extend(component)
        current = ordered[-1]

    return ordered


def closest_vertex_index(
    route_coordinates: list[list[float]],
    point: tuple[float, float],
) -> int:
    distances = [haversine_radians(point, vertex) for vertex in route_coordinates]
    return min(range(len(distances)), key=distances.__getitem__)


def load_voyages(path: Path) -> tuple[pd.DataFrame, dict[str, int | float]]:
    voyages = pd.read_excel(path, sheet_name="Matched Voyages")
    raw_rows = len(voyages)

    required = {
        DATE_COLUMN,
        "end_date",
        "IMO",
        "voyage",
        "amount_cmb",
        "from_node_id",
        "to_node_id",
        "from_terminal",
        "to_terminal",
    }
    require_columns(voyages, required, "Matched Voyages")

    voyages["start_date"] = pd.to_datetime(voyages["start_date"], errors="coerce")
    voyages["end_date"] = pd.to_datetime(voyages["end_date"], errors="coerce")
    voyages["amount_cmb"] = pd.to_numeric(voyages["amount_cmb"], errors="coerce")

    voyages = voyages.loc[
        voyages["voyage"].astype(str).str.lower().eq("export")
        & voyages[DATE_COLUMN].notna()
        & voyages["from_node_id"].notna()
        & voyages["to_node_id"].notna()
        & voyages["amount_cmb"].gt(0)
    ].copy()

    rows_before_deduplication = len(voyages)
    duplicate_mask = voyages.duplicated(subset=DUPLICATE_COLUMNS, keep=False)
    possible_duplicate_rows = int(duplicate_mask.sum())

    voyages = voyages.drop_duplicates(
        subset=DUPLICATE_COLUMNS,
        keep="first",
    ).copy()

    voyages["from_node_id"] = voyages["from_node_id"].astype(str)
    voyages["to_node_id"] = voyages["to_node_id"].astype(str)
    voyages["period_month"] = voyages[DATE_COLUMN].dt.to_period("M").astype(str)

    qa = {
        "raw_workbook_rows": raw_rows,
        "positive_export_rows_before_deduplication": rows_before_deduplication,
        "possible_duplicate_export_rows": possible_duplicate_rows,
        "duplicate_export_rows_removed": rows_before_deduplication - len(voyages),
        "positive_export_rows_used": len(voyages),
        "total_lng_volume_cmb": int(voyages["amount_cmb"].sum()),
    }
    return voyages, qa


def load_nodes(path: Path) -> pd.DataFrame:
    nodes = pd.read_csv(path)
    require_columns(
        nodes,
        {"node_id", "node_name", "node_type", "latitude", "longitude"},
        "Node table",
    )
    if nodes["node_id"].duplicated().any():
        raise ValueError("Node table contains duplicated node_id values.")
    nodes["node_id"] = nodes["node_id"].astype(str)
    return nodes


def load_routes(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as source:
        collection = json.load(source)
    features = collection.get("features", [])
    if not features:
        raise ValueError("Route GeoJSON contains no features.")
    return features


def build_route_table(
    features: list[dict],
    nodes: pd.DataFrame,
) -> pd.DataFrame:
    node_by_name = nodes.set_index("node_name")
    route_records: list[dict] = []
    seen_route_ids: set[str] = set()
    seen_od_pairs: set[tuple[str, str]] = set()

    for feature in features:
        properties = feature.get("properties", {})
        geometry = feature.get("geometry", {})
        route_id = str(properties.get("route_id"))
        from_node_id = str(properties.get("from_node_id"))
        to_node_id = str(properties.get("to_node_id"))
        od_pair = (from_node_id, to_node_id)

        if route_id in seen_route_ids:
            raise ValueError(f"Duplicated route_id: {route_id}")
        if od_pair in seen_od_pairs:
            raise ValueError(f"Duplicated route for OD pair: {od_pair}")
        seen_route_ids.add(route_id)
        seen_od_pairs.add(od_pair)

        if geometry.get("type") != "MultiLineString":
            raise ValueError(f"{route_id}: expected MultiLineString geometry.")

        origin = (float(properties["fromLon"]), float(properties["fromLat"]))
        route_coordinates = stitch_multiline(geometry["coordinates"], origin)

        chokepoint_names = [
            item.strip()
            for item in str(properties.get("chokepoints_final", "")).split(";")
            if item.strip()
        ]
        ordered_chokepoints: list[tuple[int, str]] = []
        for name in chokepoint_names:
            if name not in node_by_name.index:
                raise ValueError(f"{route_id}: unknown chokepoint name {name!r}.")
            chokepoint = node_by_name.loc[name]
            if chokepoint["node_type"] != "chokepoint":
                raise ValueError(f"{route_id}: {name!r} is not a chokepoint node.")
            point = (float(chokepoint["longitude"]), float(chokepoint["latitude"]))
            ordered_chokepoints.append(
                (closest_vertex_index(route_coordinates, point), str(chokepoint["node_id"]))
            )

        ordered_chokepoints.sort(key=lambda item: item[0])
        path_nodes = [from_node_id]
        path_nodes.extend(node_id for _, node_id in ordered_chokepoints)
        path_nodes.append(to_node_id)

        if len(path_nodes) != len(set(path_nodes)):
            raise ValueError(f"{route_id}: repeated node in ordered route path {path_nodes}.")

        route_records.append(
            {
                "route_id": route_id,
                "from_node_id": from_node_id,
                "to_node_id": to_node_id,
                "path_nodes": path_nodes,
                "qa_status": str(properties.get("qa_status", "")).upper(),
            }
        )

    return pd.DataFrame(route_records)


def attach_routes(voyages: pd.DataFrame, routes: pd.DataFrame) -> pd.DataFrame:
    merged = voyages.merge(
        routes,
        on=["from_node_id", "to_node_id"],
        how="left",
        validate="many_to_one",
    )
    missing = merged["route_id"].isna()
    if missing.any():
        examples = (
            merged.loc[missing, ["from_node_id", "to_node_id"]]
            .drop_duplicates()
            .head(20)
            .to_dict("records")
        )
        raise ValueError(f"Voyages without a reconstructed route: {examples}")
    return merged


def expand_voyages(voyages: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for voyage in voyages.itertuples(index=False):
        for from_node_id, to_node_id in zip(voyage.path_nodes, voyage.path_nodes[1:]):
            records.append(
                {
                    "period_month": voyage.period_month,
                    "from_node_id": from_node_id,
                    "to_node_id": to_node_id,
                    "route_id": voyage.route_id,
                    "amount_cmb": voyage.amount_cmb,
                    "qa_watch": int(voyage.qa_status != "PASS"),
                }
            )
    return pd.DataFrame(records)


def classify_edge(from_node: str, to_node: str, node_types: dict[str, str]) -> str:
    from_type = node_types[from_node]
    to_type = node_types[to_node]
    if from_type == "chokepoint" and to_type == "chokepoint":
        return "chokepoint-chokepoint"
    if from_type == "terminal" and to_type == "terminal":
        return "terminal-terminal"
    return "terminal-chokepoint"


def build_edges(
    traversals: pd.DataFrame,
    voyages: pd.DataFrame,
    nodes: pd.DataFrame,
) -> pd.DataFrame:
    def join_route_ids(series: pd.Series) -> str:
        # Lexical sorting reproduces the established R001...R1037 convention.
        return "; ".join(sorted(set(series.astype(str))))

    edges = (
        traversals.groupby(
            ["period_month", "from_node_id", "to_node_id"],
            sort=True,
            as_index=False,
        )
        .agg(
            route_count=("route_id", "nunique"),
            voyage_count=("route_id", "size"),
            lng_flow_cmb=("amount_cmb", "sum"),
            qa_watch_routes=("qa_watch", "sum"),
            route_ids=("route_id", join_route_ids),
        )
    )

    node_types = nodes.set_index("node_id")["node_type"].to_dict()
    unknown_nodes = (
        set(edges["from_node_id"]) | set(edges["to_node_id"])
    ).difference(node_types)
    if unknown_nodes:
        raise ValueError(f"Edges refer to unknown nodes: {sorted(unknown_nodes)[:20]}")

    edges["edge_type"] = [
        classify_edge(from_node, to_node, node_types)
        for from_node, to_node in zip(edges["from_node_id"], edges["to_node_id"])
    ]

    monthly_volume = voyages.groupby("period_month")["amount_cmb"].sum()
    edges["share_global_monthly_lng"] = (
        edges["lng_flow_cmb"] / edges["period_month"].map(monthly_volume)
    )
    outgoing_flow = edges.groupby(
        ["period_month", "from_node_id"]
    )["lng_flow_cmb"].transform("sum")
    edges["weight_row_normalized"] = edges["lng_flow_cmb"] / outgoing_flow

    edges = edges.sort_values(
        ["period_month", "from_node_id", "to_node_id"]
    ).reset_index(drop=True)
    edges.insert(0, "edge_period_id", [f"ME{i:06d}" for i in range(1, len(edges) + 1)])
    edges.insert(2, "year", edges["period_month"].str[:4].astype(int))
    edges.insert(3, "month", edges["period_month"].str[5:7].astype(int))

    return edges[
        [
            "edge_period_id",
            "period_month",
            "year",
            "month",
            "from_node_id",
            "to_node_id",
            "edge_type",
            "route_count",
            "voyage_count",
            "lng_flow_cmb",
            "qa_watch_routes",
            "route_ids",
            "share_global_monthly_lng",
            "weight_row_normalized",
        ]
    ]


def build_monthly_qa(
    voyages: pd.DataFrame,
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
) -> pd.DataFrame:
    node_types = nodes.set_index("node_id")["node_type"].to_dict()
    total_nodes = len(nodes)
    density_denominator = total_nodes * (total_nodes - 1)
    records: list[dict] = []

    for period_month, month_voyages in voyages.groupby("period_month", sort=True):
        month_edges = edges.loc[edges["period_month"].eq(period_month)]
        out_nodes = set(month_edges["from_node_id"])
        in_nodes = set(month_edges["to_node_id"])
        active_nodes = out_nodes | in_nodes
        row_sums = month_edges.groupby("from_node_id")["weight_row_normalized"].sum()

        records.append(
            {
                "period_month": period_month,
                "year": int(period_month[:4]),
                "month": int(period_month[5:7]),
                "export_voyages": len(month_voyages),
                "total_export_lng_cmb": int(month_voyages["amount_cmb"].sum()),
                "active_od_routes": month_voyages["route_id"].nunique(),
                "active_origin_terminals": month_voyages["from_node_id"].nunique(),
                "active_destination_terminals": month_voyages["to_node_id"].nunique(),
                "active_edges": len(month_edges),
                "active_edge_types": month_edges["edge_type"].nunique(),
                "nodes_with_outflow": len(out_nodes),
                "nodes_with_inflow": len(in_nodes),
                "active_nodes": len(active_nodes),
                "active_chokepoints": sum(
                    node_types[node_id] == "chokepoint" for node_id in active_nodes
                ),
                "active_terminals": sum(
                    node_types[node_id] == "terminal" for node_id in active_nodes
                ),
                "network_density_directed": len(month_edges) / density_denominator,
                "active_node_share": len(active_nodes) / total_nodes,
                "min_row_sum": row_sums.min(),
                "max_row_sum": row_sums.max(),
            }
        )

    return pd.DataFrame(records)


def validate_outputs(
    voyages: pd.DataFrame,
    routes: pd.DataFrame,
    edges: pd.DataFrame,
    monthly_qa: pd.DataFrame,
    nodes: pd.DataFrame,
) -> None:
    if len(voyages) != 8_642:
        raise ValueError(f"Expected 8,642 deduplicated voyages, got {len(voyages):,}.")
    if len(routes) != 1_037:
        raise ValueError(f"Expected 1,037 routes, got {len(routes):,}.")
    if len(nodes) != 187:
        raise ValueError(f"Expected 187 nodes, got {len(nodes):,}.")
    if monthly_qa["period_month"].nunique() != 60:
        raise ValueError("Expected 60 monthly observations.")
    if edges.duplicated(["period_month", "from_node_id", "to_node_id"]).any():
        raise ValueError("Duplicated edge-month keys remain in output.")
    if not (edges["lng_flow_cmb"] > 0).all():
        raise ValueError("Non-positive edge flow found.")

    row_sums = edges.groupby(
        ["period_month", "from_node_id"]
    )["weight_row_normalized"].sum()
    if not ((row_sums - 1.0).abs() < 1e-10).all():
        raise ValueError("Outgoing row-normalized weights do not sum to one.")


def main() -> None:
    args = parse_args()
    for path in [args.voyages, args.nodes, args.routes]:
        if not path.exists():
            raise FileNotFoundError(path)

    voyages, input_qa = load_voyages(args.voyages)
    nodes = load_nodes(args.nodes)
    route_features = load_routes(args.routes)
    routes = build_route_table(route_features, nodes)
    voyages = attach_routes(voyages, routes)
    traversals = expand_voyages(voyages)
    edges = build_edges(traversals, voyages, nodes)
    monthly_qa = build_monthly_qa(voyages, edges, nodes)
    validate_outputs(voyages, routes, edges, monthly_qa, nodes)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    edges_path = args.output_dir / DEFAULT_EDGES
    qa_path = args.output_dir / DEFAULT_MONTHLY_QA
    edges.to_csv(edges_path, index=False)
    monthly_qa.to_csv(qa_path, index=False)

    print("\n" + "=" * 72)
    print("INPUT QA")
    print("=" * 72)
    for check, value in input_qa.items():
        print(f"{check}: {value:,}")
    print(f"observed_terminals: {len(set(voyages['from_node_id']) | set(voyages['to_node_id'])):,}")
    print(f"months: {voyages['period_month'].nunique():,}")

    print("\n" + "=" * 72)
    print("NETWORK OUTPUT")
    print("=" * 72)
    print(f"routes: {len(routes):,}")
    print(f"edge-month rows: {len(edges):,}")
    print(f"monthly QA rows: {len(monthly_qa):,}")
    print(f"edges output: {edges_path}")
    print(f"monthly QA output: {qa_path}")


if __name__ == "__main__":
    main()
