#!/usr/bin/env python3
"""Generate maritime routes for the 1,037 observed LNG export OD pairs.

The script reads ``data/LNG_voyage_node_matching.xlsx``, prepares the unique
export origin-destination pairs, and runs the Eurostat SeaRoute JAR stored in
``tools/searoute``. The resulting unclassified route geometries are written to
``data/LNG_1037_routes_searoute.geojson``.

Default routing settings reproduce the configuration used for the existing
route dataset: 5 km network resolution and the Panama Canal disabled.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd


WORKBOOK_NAME = "LNG_voyage_node_matching.xlsx"
OUTPUT_NAME = "LNG_1037_routes_searoute.geojson"
EXPECTED_ROUTE_COUNT = 1_037


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"
    searoute_dir = project_root / "tools" / "searoute"

    parser = argparse.ArgumentParser(
        description=(
            "Generate Eurostat SeaRoute geometries for the LNG export "
            "origin-destination pairs."
        )
    )
    parser.add_argument(
        "--workbook",
        type=Path,
        default=data_dir / WORKBOOK_NAME,
        help="Voyage matching workbook produced by script 01.",
    )
    parser.add_argument(
        "--searoute-dir",
        type=Path,
        default=searoute_dir,
        help="Directory containing searoute.jar and the marnet directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=data_dir / OUTPUT_NAME,
        help="Output GeoJSON containing the unclassified SeaRoute routes.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        choices=[5, 10, 20, 50, 100],
        default=5,
        help="SeaRoute maritime-network resolution in kilometres.",
    )
    parser.add_argument(
        "--panama",
        type=int,
        choices=[0, 1],
        default=0,
        help="Whether SeaRoute may use the Panama Canal.",
    )
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing: {', '.join(sorted(missing))}")


def prepare_route_input(workbook: Path) -> pd.DataFrame:
    od_pairs = pd.read_excel(workbook, sheet_name="OD Pairs")
    terminals = pd.read_excel(workbook, sheet_name="Terminal Map")

    require_columns(
        od_pairs,
        {
            "from_node_id",
            "to_node_id",
            "from_terminal",
            "to_terminal",
            "voyage_rows",
            "export_rows",
            "first_start",
            "last_end",
            "median_voyage_distance",
        },
        "OD Pairs sheet",
    )
    require_columns(
        terminals,
        {"node_id", "latitude", "longitude"},
        "Terminal Map sheet",
    )

    routes = od_pairs.loc[
        pd.to_numeric(od_pairs["export_rows"], errors="coerce").gt(0)
    ].copy()
    routes["export_rows"] = pd.to_numeric(
        routes["export_rows"], errors="raise"
    ).astype(int)
    routes["voyage_rows"] = pd.to_numeric(
        routes["voyage_rows"], errors="raise"
    ).astype(int)

    # Stable and explicit ordering makes route IDs reproducible.
    routes = routes.sort_values(
        ["export_rows", "from_node_id", "to_node_id"],
        ascending=[False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    routes.insert(
        0,
        "route_id",
        [f"R{number:03d}" for number in range(1, len(routes) + 1)],
    )

    terminal_coordinates = terminals.set_index("node_id")[["latitude", "longitude"]]
    routes = routes.join(
        terminal_coordinates.rename(
            columns={"latitude": "fromLat", "longitude": "fromLon"}
        ),
        on="from_node_id",
        validate="many_to_one",
    )
    routes = routes.join(
        terminal_coordinates.rename(
            columns={"latitude": "toLat", "longitude": "toLon"}
        ),
        on="to_node_id",
        validate="many_to_one",
    )

    coordinate_columns = ["fromLat", "fromLon", "toLat", "toLon"]
    if routes[coordinate_columns].isna().any().any():
        affected = routes.loc[
            routes[coordinate_columns].isna().any(axis=1),
            ["route_id", "from_node_id", "to_node_id"],
        ]
        raise ValueError(
            "Some OD pairs have missing terminal coordinates:\n"
            + affected.head(20).to_string(index=False)
        )

    routes = routes.rename(
        columns={"median_voyage_distance": "voyage_distance_km_observed"}
    )
    routes["first_start"] = pd.to_datetime(
        routes["first_start"], errors="raise"
    ).dt.strftime("%Y-%m-%d")
    routes["last_end"] = pd.to_datetime(
        routes["last_end"], errors="raise"
    ).dt.strftime("%Y-%m-%d")
    routes["benchmark_50"] = routes.index < 50

    columns = [
        "route_id",
        "from_node_id",
        "to_node_id",
        "from_terminal",
        "to_terminal",
        "voyage_rows",
        "export_rows",
        "first_start",
        "last_end",
        "voyage_distance_km_observed",
        "fromLat",
        "fromLon",
        "toLat",
        "toLon",
        "benchmark_50",
    ]
    routes = routes[columns]

    if len(routes) != EXPECTED_ROUTE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_ROUTE_COUNT:,} export OD pairs, got {len(routes):,}."
        )
    if routes["route_id"].duplicated().any():
        raise ValueError("Route IDs are not unique.")
    if routes[["from_node_id", "to_node_id"]].duplicated().any():
        raise ValueError("Export origin-destination pairs are not unique.")

    return routes


def validate_searoute_installation(searoute_dir: Path) -> Path:
    jar = searoute_dir / "searoute.jar"
    marnet = searoute_dir / "marnet"

    if shutil.which("java") is None:
        raise RuntimeError(
            "Java was not found. Install Java and make sure the 'java' command "
            "is available from the terminal."
        )
    if not jar.is_file():
        raise FileNotFoundError(
            f"Missing {jar}. Extract the Eurostat SeaRoute ZIP so that "
            "tools/searoute contains searoute.jar."
        )
    if not marnet.is_dir():
        raise FileNotFoundError(
            f"Missing {marnet}. Keep the marnet directory beside searoute.jar."
        )
    return jar


def run_searoute(
    routes: pd.DataFrame,
    searoute_dir: Path,
    jar: Path,
    output: Path,
    resolution: int,
    panama: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="lng_searoute_") as temporary_dir:
        temporary_dir = Path(temporary_dir)
        input_csv = temporary_dir / "lng_export_od_pairs.csv"
        temporary_output = temporary_dir / "LNG_1037_routes_searoute.geojson"
        routes.to_csv(input_csv, index=False)

        command = [
            "java",
            "-jar",
            str(jar),
            "-i",
            str(input_csv),
            "-o",
            str(temporary_output),
            "-res",
            str(resolution),
            "-panama",
            str(panama),
            "-olatCol",
            "fromLat",
            "-olonCol",
            "fromLon",
            "-dlatCol",
            "toLat",
            "-dlonCol",
            "toLon",
        ]
        subprocess.run(command, cwd=searoute_dir, check=True)

        if not temporary_output.is_file():
            raise RuntimeError("SeaRoute completed without creating its GeoJSON output.")
        shutil.copyfile(temporary_output, output)


def validate_output(output: Path, expected_pairs: pd.DataFrame) -> dict:
    with output.open(encoding="utf-8-sig") as source:
        geojson = json.load(source)

    features = geojson.get("features", [])
    if len(features) != len(expected_pairs):
        raise ValueError(
            f"Expected {len(expected_pairs):,} route features, got {len(features):,}."
        )

    expected_ids = set(expected_pairs["route_id"])
    actual_ids = {
        str(feature.get("properties", {}).get("route_id")) for feature in features
    }
    if actual_ids != expected_ids:
        missing = sorted(expected_ids.difference(actual_ids))
        unexpected = sorted(actual_ids.difference(expected_ids))
        raise ValueError(
            f"Route-ID mismatch. Missing: {missing[:10]}; "
            f"unexpected: {unexpected[:10]}."
        )

    distances = []
    origin_snaps = []
    destination_snaps = []
    missing_distances = 0
    degenerate_geometries = 0
    for feature in features:
        properties = feature.get("properties", {})
        geometry = feature.get("geometry") or {}
        coordinates = geometry.get("coordinates") or []
        coordinate_pairs = []
        if geometry.get("type") == "LineString":
            coordinate_pairs = coordinates
        elif geometry.get("type") == "MultiLineString":
            coordinate_pairs = [pair for part in coordinates for pair in part]
        if len({tuple(pair) for pair in coordinate_pairs}) < 2:
            degenerate_geometries += 1

        try:
            distances.append(float(properties["distKM"]))
        except (KeyError, TypeError, ValueError):
            missing_distances += 1

        for collection, key in [
            (origin_snaps, "dFromKM"),
            (destination_snaps, "dToKM"),
        ]:
            try:
                collection.append(float(properties[key]))
            except (KeyError, TypeError, ValueError):
                pass

    return {
        "routes": len(features),
        "total_route_distance_km": sum(distances),
        "missing_distances": missing_distances,
        "degenerate_geometries": degenerate_geometries,
        "large_origin_snaps": sum(value > 100 for value in origin_snaps),
        "large_destination_snaps": sum(value > 100 for value in destination_snaps),
    }


def main() -> None:
    args = parse_args()

    if not args.workbook.is_file():
        raise FileNotFoundError(
            f"Missing {args.workbook}. Run 01_prepare_voyages_and_nodes.py first."
        )

    jar = validate_searoute_installation(args.searoute_dir)
    routes = prepare_route_input(args.workbook)

    print("\n" + "=" * 72)
    print("GENERATING SEAROUTE ROUTES")
    print("=" * 72)
    print(f"Export OD pairs: {len(routes):,}")
    print(f"Resolution: {args.resolution} km")
    print(f"Panama Canal enabled: {'yes' if args.panama else 'no'}")

    run_searoute(
        routes=routes,
        searoute_dir=args.searoute_dir,
        jar=jar,
        output=args.output,
        resolution=args.resolution,
        panama=args.panama,
    )
    qa = validate_output(args.output, routes)

    print("\n" + "=" * 72)
    print("SEAROUTE GENERATION COMPLETE")
    print("=" * 72)
    print(f"Routes generated: {qa['routes']:,}")
    print(f"Routes with missing SeaRoute distance: {qa['missing_distances']:,}")
    print(f"Degenerate route geometries: {qa['degenerate_geometries']:,}")
    print(f"Large origin snaps (>100 km): {qa['large_origin_snaps']:,}")
    print(f"Large destination snaps (>100 km): {qa['large_destination_snaps']:,}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
