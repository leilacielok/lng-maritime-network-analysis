"""Prepare matched LNG voyages and the final multilayer node table.

The script reconstructs the two upstream datasets used by the network pipeline:

* ``LNG_voyage_node_matching.xlsx``
* ``LNG_multilayer_nodes.csv``

Raw LNG-T3 inputs remain unchanged. The workbook preserves all original voyage
rows (including return legs); filtering and voyage deduplication are performed
later by ``03_build_multilayer_network.py`` and the terminal-month analysis.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


VOYAGES_NAME = "LNG_tanker_voyage.csv"
TERMINALS_NAME = "LNG_terminal.csv"
REGISTRY_NAME = "PortWatch_28_geometry_registry.xlsx"
CHOKEPOINTS_NAME = "PortWatch_28_chokepoints_geometry.geojson"
MATCHED_OUTPUT_NAME = "LNG_voyage_node_matching.xlsx"
NODES_OUTPUT_NAME = "LNG_multilayer_nodes.csv"

STATUS_PRIORITY = {
    "operating": 0,
    "construction": 1,
    "idle": 2,
    "mothballed": 3,
    "proposed": 4,
}

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
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"
    
    parser = argparse.ArgumentParser(
        description="Match LNG-T3 voyages to terminal nodes and build node data."
    )
    parser.add_argument("--voyages", type=Path, default=data_dir / VOYAGES_NAME)
    parser.add_argument("--terminals", type=Path, default=data_dir / TERMINALS_NAME)
    parser.add_argument("--registry", type=Path, default=data_dir / REGISTRY_NAME)
    parser.add_argument("--chokepoints", type=Path, default=data_dir / CHOKEPOINTS_NAME)
    parser.add_argument("--output-dir", type=Path, default=data_dir)
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(sorted(missing))}")


def load_inputs(args: argparse.Namespace):
    for path in [args.voyages, args.terminals, args.registry, args.chokepoints]:
        if not path.exists():
            raise FileNotFoundError(path)

    voyages = pd.read_csv(args.voyages)
    terminals = pd.read_csv(args.terminals)
    registry = pd.read_excel(args.registry, sheet_name="Geometry Registry")
    with args.chokepoints.open(encoding="utf-8-sig") as source:
        chokepoint_geojson = json.load(source)

    require_columns(
        voyages,
        {
            "start_date", "end_date", "IMO", "voyage", "from_terminal",
            "to_terminal", "amount_cmb", "from_country", "to_country",
            "confidence_score", "voyage_distance",
        },
        "LNG voyage table",
    )
    require_columns(
        terminals,
        {
            "name", "status", "terminal_type", "region", "areas", "lat",
            "lon", "UN_LOCODE",
        },
        "LNG terminal table",
    )
    require_columns(
        registry,
        {
            "node_id", "portwatch_id", "chokepoint", "portwatch_lat",
            "portwatch_lon", "status",
        },
        "Chokepoint geometry registry",
    )
    return voyages, terminals, registry, chokepoint_geojson


def prepare_terminal_map(terminals: pd.DataFrame):
    terminals = terminals.copy()
    terminals["source_row"] = terminals.index + 2  # CSV header occupies row 1.

    # LNGN identifiers were assigned to distinct name-coordinate variants in
    # terminal-name and coordinate order. Repeated status records at identical
    # coordinates share one ID.
    variants = (
        terminals.drop_duplicates(["name", "lat", "lon"], keep="first")
        .sort_values(["name", "lat", "lon"], kind="stable")
        .copy()
    )
    variants["variant_node_id"] = [
        f"LNGN{number:04d}" for number in range(1, len(variants) + 1)
    ]

    variant_id_lookup = {
        (row.name, row.lat, row.lon): row.variant_node_id
        for row in variants.itertuples(index=False)
    }
    terminals["variant_node_id"] = [
        variant_id_lookup[(name, lat, lon)]
        for name, lat, lon in zip(terminals["name"], terminals["lat"], terminals["lon"])
    ]
    canonical_ids = variants.groupby("name", sort=False)["variant_node_id"].first().to_dict()
    terminals["canonical_node_id"] = terminals["name"].map(canonical_ids)

    terminals["status_priority"] = (
        terminals["status"].astype(str).str.lower().map(STATUS_PRIORITY).fillna(99)
    )
    selected = (
        terminals.sort_values(["status_priority", "source_row"])
        .drop_duplicates("name", keep="first")
        .sort_values("canonical_node_id")
        .copy()
    )

    terminal_map = selected.rename(
        columns={
            "canonical_node_id": "node_id",
            "name": "node_name",
            "areas": "country",
            "lat": "latitude",
            "lon": "longitude",
        }
    )[
        [
            "node_id", "node_name", "country", "region", "latitude",
            "longitude", "UN_LOCODE", "status",
        ]
    ]
    terminal_map["is_operating"] = (
        terminal_map["status"].astype(str).str.lower().eq("operating").astype(int)
    )
    terminal_map = terminal_map.drop(columns="status")

    coordinate_variants = []
    variant_counts = variants.groupby("name").size()
    for name in variant_counts[variant_counts > 1].index:
        name_variants = variants.loc[variants["name"].eq(name)].copy()
        canonical_id = canonical_ids[name]
        chosen = terminal_map.loc[terminal_map["node_name"].eq(name)].iloc[0]
        for variant in name_variants.itertuples(index=False):
            source_rows = terminals.loc[
                terminals["variant_node_id"].eq(variant.variant_node_id), "source_row"
            ]
            coordinate_variants.append(
                {
                    "canonical_node_id": canonical_id,
                    "node_name": name,
                    "variant_node_id": variant.variant_node_id,
                    "latitude": variant.lat,
                    "longitude": variant.lon,
                    "status": variant.status,
                    "UN_LOCODE": variant.UN_LOCODE,
                    "source_row_min": int(source_rows.min()),
                    "source_row_max": int(source_rows.max()),
                    "selected_as_canonical": int(
                        variant.lat == chosen["latitude"]
                        and variant.lon == chosen["longitude"]
                    ),
                    "rule": (
                        "Same terminal name + same UN/LOCODE + geographically "
                        "close; one canonical node retained."
                    ),
                }
            )

    coordinate_variants = pd.DataFrame(coordinate_variants)
    return terminal_map.reset_index(drop=True), coordinate_variants, terminals


def match_voyages(voyages: pd.DataFrame, terminal_map: pd.DataFrame) -> pd.DataFrame:
    lookup = terminal_map.set_index("node_name")
    matched = voyages.copy()

    for prefix, terminal_column in [("from", "from_terminal"), ("to", "to_terminal")]:
        matched[f"{prefix}_node_id"] = matched[terminal_column].map(lookup["node_id"])
        matched[f"{prefix}_node_latitude"] = matched[terminal_column].map(lookup["latitude"])
        matched[f"{prefix}_node_longitude"] = matched[terminal_column].map(lookup["longitude"])
        matched[f"{prefix}_match"] = matched[f"{prefix}_node_id"].notna()

    matched["both_matched"] = matched["from_match"] & matched["to_match"]
    return matched[
        [
            "start_date", "end_date", "IMO", "voyage", "from_terminal",
            "to_terminal", "amount_cmb", "from_country", "to_country",
            "confidence_score", "voyage_distance", "from_node_id",
            "to_node_id", "from_node_latitude", "from_node_longitude",
            "to_node_latitude", "to_node_longitude", "from_match", "to_match",
            "both_matched",
        ]
    ]


def build_od_pairs(matched: pd.DataFrame) -> pd.DataFrame:
    data = matched.copy()
    data["start_date"] = pd.to_datetime(data["start_date"], errors="coerce")
    data["end_date"] = pd.to_datetime(data["end_date"], errors="coerce")
    data["amount_cmb"] = pd.to_numeric(data["amount_cmb"], errors="coerce")

    od_pairs = (
        data.groupby(
            ["from_node_id", "to_node_id", "from_terminal", "to_terminal"],
            sort=False,
            as_index=False,
        )
        .agg(
            voyage_rows=("voyage", "size"),
            export_rows=("voyage", lambda values: values.astype(str).str.lower().eq("export").sum()),
            return_rows=("voyage", lambda values: values.astype(str).str.lower().eq("return").sum()),
            first_start=("start_date", "min"),
            last_end=("end_date", "max"),
            median_voyage_distance=("voyage_distance", "median"),
        )
    )
    return od_pairs.sort_values(
        ["voyage_rows", "from_node_id", "to_node_id"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def build_pilot_routes(od_pairs: pd.DataFrame) -> pd.DataFrame:
    pilot = od_pairs.loc[od_pairs["export_rows"].gt(0)].copy()
    pilot = pilot.sort_values(
        "export_rows",
        ascending=False,
        kind="stable",
    ).head(50)
    pilot["voyage_rows"] = pilot["export_rows"]
    pilot = pilot[
        [
            "from_node_id", "to_node_id", "from_terminal", "to_terminal",
            "voyage_rows", "first_start", "last_end", "median_voyage_distance",
        ]
    ]
    pilot["route_status"] = "ready_for_routing"
    pilot["baseline_route_distance_nm"] = pd.NA
    pilot["detected_chokepoints"] = pd.NA
    pilot["routing_notes"] = pd.NA
    return pilot.reset_index(drop=True)


def validate_chokepoints(registry: pd.DataFrame, geojson: dict) -> None:
    registry_ids = set(registry["node_id"].astype(str))
    features = geojson.get("features", [])
    geometry_ids = {str(item.get("properties", {}).get("node_id")) for item in features}
    expected_ids = {f"CP{number:03d}" for number in range(1, 29)}
    if len(registry) != 28 or registry_ids != expected_ids:
        raise ValueError("Registry must contain exactly CP001 through CP028.")
    if len(features) != 28 or geometry_ids != expected_ids:
        raise ValueError("GeoJSON must contain exactly CP001 through CP028.")


def build_final_nodes(
    matched: pd.DataFrame,
    terminal_map: pd.DataFrame,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    voyages = matched.copy()
    voyages["start_date"] = pd.to_datetime(voyages["start_date"], errors="coerce")
    voyages["end_date"] = pd.to_datetime(voyages["end_date"], errors="coerce")
    voyages["amount_cmb"] = pd.to_numeric(voyages["amount_cmb"], errors="coerce")
    voyages = voyages.loc[
        voyages["voyage"].astype(str).str.lower().eq("export")
        & voyages["start_date"].notna()
        & voyages["from_node_id"].notna()
        & voyages["to_node_id"].notna()
        & voyages["amount_cmb"].gt(0)
    ].copy()
    voyages = voyages.drop_duplicates(subset=DUPLICATE_COLUMNS, keep="first")

    outgoing = voyages.groupby("from_node_id").agg(
        origin_export_voyages=("amount_cmb", "size"),
        origin_lng_volume_cmb=("amount_cmb", "sum"),
    )
    incoming = voyages.groupby("to_node_id").agg(
        destination_export_voyages=("amount_cmb", "size"),
        destination_lng_volume_cmb=("amount_cmb", "sum"),
    )
    observed_ids = set(outgoing.index) | set(incoming.index)

    terminal_nodes = terminal_map.loc[terminal_map["node_id"].isin(observed_ids)].copy()
    terminal_nodes = terminal_nodes.merge(outgoing, left_on="node_id", right_index=True, how="left")
    terminal_nodes = terminal_nodes.merge(incoming, left_on="node_id", right_index=True, how="left")
    count_and_flow = [
        "origin_export_voyages", "destination_export_voyages",
        "origin_lng_volume_cmb", "destination_lng_volume_cmb",
    ]
    terminal_nodes[count_and_flow] = terminal_nodes[count_and_flow].fillna(0)
    terminal_nodes["node_type"] = "terminal"
    terminal_nodes["layer"] = "LNG_terminal"
    terminal_nodes["terminal_role"] = ""
    terminal_nodes.loc[terminal_nodes["origin_export_voyages"].gt(0), "terminal_role"] = "export_origin"
    terminal_nodes.loc[terminal_nodes["destination_export_voyages"].gt(0), "terminal_role"] = "import_destination"
    both = terminal_nodes["origin_export_voyages"].gt(0) & terminal_nodes["destination_export_voyages"].gt(0)
    terminal_nodes.loc[both, "terminal_role"] = "bidirectional"
    terminal_nodes["coordinate_source"] = "LNG-T3 Terminal Map"
    terminal_nodes["geometry_status"] = pd.NA

    chokepoint_nodes = registry.rename(
        columns={
            "chokepoint": "node_name",
            "portwatch_lat": "latitude",
            "portwatch_lon": "longitude",
            "status": "geometry_status",
        }
    )[["node_id", "node_name", "latitude", "longitude", "geometry_status"]].copy()
    chokepoint_nodes["node_type"] = "chokepoint"
    chokepoint_nodes["layer"] = "maritime_chokepoint"
    chokepoint_nodes["terminal_role"] = pd.NA
    chokepoint_nodes["country"] = pd.NA
    chokepoint_nodes["region"] = pd.NA
    chokepoint_nodes["coordinate_source"] = "IMF PortWatch reference location"
    chokepoint_nodes["UN_LOCODE"] = pd.NA
    chokepoint_nodes["is_operating"] = pd.NA
    for column in count_and_flow:
        chokepoint_nodes[column] = pd.NA

    columns = [
        "node_id", "node_name", "node_type", "layer", "terminal_role",
        "country", "region", "latitude", "longitude", "coordinate_source",
        "UN_LOCODE", "is_operating", "origin_export_voyages",
        "destination_export_voyages", "origin_lng_volume_cmb",
        "destination_lng_volume_cmb", "geometry_status",
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        nodes = pd.concat(
            [terminal_nodes[columns], chokepoint_nodes[columns]],
            ignore_index=True,
        )
    return nodes.sort_values("node_id").reset_index(drop=True)


def build_summary(
    matched: pd.DataFrame,
    terminal_map: pd.DataFrame,
    od_pairs: pd.DataFrame,
    nodes: pd.DataFrame,
) -> pd.DataFrame:
    dates = pd.concat(
        [pd.to_datetime(matched["start_date"]), pd.to_datetime(matched["end_date"])]
    )
    terminal_names = set(matched["from_terminal"]) | set(matched["to_terminal"])
    possible_duplicates = int(
        matched.loc[matched["voyage"].astype(str).str.lower().eq("export")]
        .duplicated(subset=DUPLICATE_COLUMNS, keep=False)
        .sum()
    )
    metrics = [
        ("Voyage rows", len(matched), "Raw LNG_tanker_voyage.csv rows"),
        ("Date range", f"{dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d}", "Observed voyage records"),
        ("Unique terminal names used in voyages", len(terminal_names), "Across origin and destination"),
        ("Unique terminal names matched", len(terminal_names), "Matched to consolidated Terminal Map"),
        ("Terminal match rate", float(matched["both_matched"].mean()), "Expected 100%"),
        ("Unique directed OD pairs", len(od_pairs), "Origin -> destination"),
        ("Unique export OD pairs", int(od_pairs["export_rows"].gt(0).sum()), "Laden/export legs"),
        ("Export voyage rows", int(matched["voyage"].astype(str).str.lower().eq("export").sum()), "Before deduplication"),
        ("Return voyage rows", int(matched["voyage"].astype(str).str.lower().eq("return").sum()), "Zero-cargo movements"),
        ("Possible duplicate export rows", possible_duplicates, "Five rows represent two unique voyages"),
        ("Consolidated LNG terminal nodes", len(terminal_map), "After merging coordinate variants"),
        ("Observed LNG terminal nodes", int(nodes["node_type"].eq("terminal").sum()), "Included in final network"),
        ("PortWatch chokepoints", int(nodes["node_type"].eq("chokepoint").sum()), "Included in final network"),
        ("Final multilayer nodes", len(nodes), "Observed terminals plus chokepoints"),
    ]
    return pd.DataFrame(metrics, columns=["Metric", "Value", "Notes"])


def format_workbook(path: Path) -> None:
    workbook = load_workbook(path)
    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)
    for sheet in workbook.worksheets:
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for cell in sheet[1]:
            cell.fill = header_fill
            cell.font = header_font
        for column in range(1, sheet.max_column + 1):
            values = [
                str(sheet.cell(row, column).value or "")
                for row in range(1, min(sheet.max_row, 250) + 1)
            ]
            width = min(max(10, max(map(len, values)) + 2), 45)
            sheet.column_dimensions[get_column_letter(column)].width = width
        for row in sheet.iter_rows():
            for cell in row:
                cell.alignment = Alignment(vertical="top", wrap_text=True)
    workbook.save(path)


def validate_outputs(
    matched: pd.DataFrame,
    terminal_map: pd.DataFrame,
    coordinate_variants: pd.DataFrame,
    od_pairs: pd.DataFrame,
    nodes: pd.DataFrame,
) -> None:
    expected = {
        "voyages": (len(matched), 17_592),
        "terminal map": (len(terminal_map), 471),
        "coordinate-variant rows": (len(coordinate_variants), 6),
        "OD pairs": (len(od_pairs), 2_153),
        "export OD pairs": (int(od_pairs["export_rows"].gt(0).sum()), 1_037),
        "final nodes": (len(nodes), 187),
        "observed terminals": (int(nodes["node_type"].eq("terminal").sum()), 159),
        "chokepoints": (int(nodes["node_type"].eq("chokepoint").sum()), 28),
    }
    errors = [f"{name}: expected {target:,}, got {actual:,}" for name, (actual, target) in expected.items() if actual != target]
    if errors:
        raise ValueError("Output validation failed: " + "; ".join(errors))
    if not matched["both_matched"].all():
        raise ValueError("Some voyage endpoints were not matched to terminal nodes.")
    if nodes["node_id"].duplicated().any():
        raise ValueError("Final node table contains duplicated node IDs.")


def main() -> None:
    args = parse_args()
    voyages, terminals, registry, chokepoint_geojson = load_inputs(args)
    validate_chokepoints(registry, chokepoint_geojson)

    terminal_map, coordinate_variants, _ = prepare_terminal_map(terminals)
    matched = match_voyages(voyages, terminal_map)
    od_pairs = build_od_pairs(matched)
    pilot_routes = build_pilot_routes(od_pairs)
    nodes = build_final_nodes(matched, terminal_map, registry)
    summary = build_summary(matched, terminal_map, od_pairs, nodes)
    validate_outputs(matched, terminal_map, coordinate_variants, od_pairs, nodes)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    workbook_path = args.output_dir / MATCHED_OUTPUT_NAME
    nodes_path = args.output_dir / NODES_OUTPUT_NAME

    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        matched.to_excel(writer, sheet_name="Matched Voyages", index=False)
        od_pairs.to_excel(writer, sheet_name="OD Pairs", index=False)
        pilot_routes.to_excel(writer, sheet_name="Pilot 50 export routes", index=False)
        terminal_map.to_excel(writer, sheet_name="Terminal Map", index=False)
        coordinate_variants.to_excel(writer, sheet_name="Coordinate variants", index=False)
    format_workbook(workbook_path)
    nodes.to_csv(nodes_path, index=False)

    print("\n" + "=" * 72)
    print("VOYAGE AND NODE PREPARATION COMPLETE")
    print("=" * 72)
    print(f"Raw voyage rows: {len(matched):,}")
    print(f"Matched endpoint share: {matched['both_matched'].mean():.1%}")
    print(f"Consolidated terminal map: {len(terminal_map):,}")
    print(f"Coordinate-variant rows documented: {len(coordinate_variants):,}")
    print(f"Directed OD pairs: {len(od_pairs):,}")
    print(f"Export OD pairs: {int(od_pairs['export_rows'].gt(0).sum()):,}")
    print(f"Final observed terminal nodes: {nodes['node_type'].eq('terminal').sum():,}")
    print(f"Final chokepoint nodes: {nodes['node_type'].eq('chokepoint').sum():,}")
    print(f"Workbook output: {workbook_path}")
    print(f"Node output: {nodes_path}")


if __name__ == "__main__":
    main()
