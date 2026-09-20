"""Build the three model-ready monthly panels used by the thesis models.

The script combines the outputs produced by ``node_month_analysis.py``,
``terminal_month_analysis.py`` and ``network_structure_diagnostics.py``.
Run those scripts before this one.

Neighbour activity is based on the cumulative undirected neighbourhood known
by the end of t-1. Defining neighbours only from active edges in t-1 would make
the share of active neighbours mechanically equal to one. The cumulative
definition also avoids using edges first observed after the modelled month.
"""

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
NODE_MONTH_FILE = (
    BASE_DIR / "eda_outputs" / "node_month" / "data" / "node_month_metrics.csv"
)
TERMINAL_MONTH_FILE = (
    BASE_DIR
    / "eda_outputs"
    / "terminal_month"
    / "data"
    / "terminal_month_metrics.csv"
)
STRUCTURE_FILE = (
    BASE_DIR
    / "eda_outputs"
    / "network_level"
    / "data"
    / "node_month_structure_metrics.csv"
)
NODES_FILE = PROCESSED_DIR / "LNG_multilayer_nodes.csv"
EDGES_FILE = PROCESSED_DIR / "LNG_multilayer_edges_monthly.csv"

OUTPUT_DIR = PROCESSED_DIR / "model_ready"
ACTIVITY_OUTPUT = OUTPUT_DIR / "activity_model_panel.csv"
TERMINAL_OUTPUT = OUTPUT_DIR / "terminal_criticality_model_panel.csv"
CHOKEPOINT_OUTPUT = OUTPUT_DIR / "chokepoint_criticality_model_panel.csv"


def require_columns(frame, required, dataset_name):
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: " + ", ".join(missing)
        )


def read_monthly_csv(path, id_columns):
    if not path.exists():
        raise FileNotFoundError(
            f"Required input not found: {path}. Run its generating script first."
        )

    frame = pd.read_csv(path)
    require_columns(frame, {"period_month", *id_columns}, path.name)
    frame["period_month"] = pd.to_datetime(frame["period_month"], errors="raise")

    for column in id_columns:
        frame[column] = frame[column].astype(str)

    if frame.duplicated([*id_columns, "period_month"]).any():
        raise ValueError(f"{path.name} contains duplicate node-month keys.")

    return frame


def load_inputs():
    node_month = read_monthly_csv(NODE_MONTH_FILE, ["node_id"])
    terminal_month = read_monthly_csv(TERMINAL_MONTH_FILE, ["terminal_id"])
    structure = read_monthly_csv(STRUCTURE_FILE, ["node_id"])

    nodes = pd.read_csv(NODES_FILE)
    edges = pd.read_csv(EDGES_FILE)
    require_columns(
        nodes,
        {"node_id", "node_type", "country", "capacity_mtpa"},
        NODES_FILE.name,
    )
    require_columns(
        edges,
        {"period_month", "from_node_id", "to_node_id"},
        EDGES_FILE.name,
    )

    nodes["node_id"] = nodes["node_id"].astype(str)
    edges["from_node_id"] = edges["from_node_id"].astype(str)
    edges["to_node_id"] = edges["to_node_id"].astype(str)
    edges["period_month"] = pd.to_datetime(edges["period_month"], errors="raise")

    return node_month, terminal_month, structure, nodes, edges


def add_seasonality(frame):
    result = frame.copy()
    result["year"] = result["period_month"].dt.year
    result["month"] = result["period_month"].dt.month
    angle = 2 * np.pi * (result["month"] - 1) / 12
    result["month_sin"] = np.sin(angle)
    result["month_cos"] = np.cos(angle)
    return result


def add_node_metadata(node_month, nodes):
    metadata_columns = [
        column
        for column in [
            "node_id",
            "node_type",
            "country",
            "infrastructure_type",
            "capacity_mtpa",
            "start_year",
        ]
        if column in nodes.columns
    ]
    metadata = nodes[metadata_columns].drop_duplicates("node_id")
    result = node_month.merge(
        metadata,
        on="node_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_metadata"),
    )

    for column in metadata_columns:
        metadata_column = f"{column}_metadata"
        if metadata_column in result.columns:
            result[column] = result[column].combine_first(result[metadata_column])
            result = result.drop(columns=metadata_column)

    return result


def build_historical_neighbor_activity(node_month, edges):
    """Return neighbour activity at t-1 over neighbours observed through t-1."""
    activity_lookup = node_month.set_index(["period_month", "node_id"])["active"]
    months = sorted(node_month["period_month"].unique())
    edges_by_month = {
        month: group for month, group in edges.groupby("period_month", sort=False)
    }
    historical_neighbors = {
        node_id: set() for node_id in node_month["node_id"].unique()
    }
    rows = []

    for month_index, month in enumerate(months):
        if month_index == 0:
            previous_month = None
        else:
            previous_month = months[month_index - 1]
            previous_edges = edges_by_month.get(previous_month)
            if previous_edges is not None:
                for edge in previous_edges.itertuples(index=False):
                    source = edge.from_node_id
                    target = edge.to_node_id
                    historical_neighbors.setdefault(source, set()).add(target)
                    historical_neighbors.setdefault(target, set()).add(source)

        month_nodes = node_month.loc[
            node_month["period_month"].eq(month), "node_id"
        ]
        for node_id in month_nodes:
            neighbors = historical_neighbors.get(node_id, set())
            neighbor_values = []

            if previous_month is not None:
                for neighbor_id in neighbors:
                    key = (previous_month, neighbor_id)
                    if key in activity_lookup.index:
                        neighbor_values.append(float(activity_lookup.loc[key]))

            rows.append(
                {
                    "node_id": node_id,
                    "period_month": month,
                    "historical_neighbor_count_lag1": len(neighbor_values),
                    "active_neighbor_count_lag1": (
                        float(np.sum(neighbor_values)) if neighbor_values else np.nan
                    ),
                    "neighbor_activity_lag1": (
                        float(np.mean(neighbor_values)) if neighbor_values else np.nan
                    ),
                }
            )

    return pd.DataFrame(rows)


def build_activity_panel(node_month, nodes, edges):
    require_columns(
        node_month,
        {"node_id", "period_month", "active", "node_throughput", "total_voyages"},
        NODE_MONTH_FILE.name,
    )
    panel = add_node_metadata(node_month, nodes)
    panel = panel.sort_values(["node_id", "period_month"]).reset_index(drop=True)

    grouped = panel.groupby("node_id", sort=False)
    panel["activity_lag1"] = grouped["active"].shift(1)
    panel["throughput_lag1"] = grouped["node_throughput"].shift(1)
    panel["log1p_throughput_lag1"] = np.log1p(panel["throughput_lag1"])
    panel["voyage_count_lag1"] = grouped["total_voyages"].shift(1)
    panel["log1p_voyage_count_lag1"] = np.log1p(
        panel["voyage_count_lag1"])
    panel["lag_available"] = panel["activity_lag1"].notna().astype(int)

    neighbor_activity = build_historical_neighbor_activity(panel, edges)
    panel = panel.merge(
        neighbor_activity,
        on=["node_id", "period_month"],
        how="left",
        validate="one_to_one",
    )
    panel = add_seasonality(panel)

    if "capacity_mtpa" in panel.columns:
        capacity = pd.to_numeric(panel["capacity_mtpa"], errors="coerce")
        panel["log1p_capacity_mtpa"] = np.log1p(capacity)

    return panel.sort_values(["period_month", "node_id"]).reset_index(drop=True)


def add_structural_indicators(structure, edges):
    require_columns(
        structure,
        {"node_id", "period_month", "is_articulation_point", "global_bridge_count"},
        STRUCTURE_FILE.name,
    )
    result = structure.copy()
    result["has_incident_global_bridge"] = (
        result["global_bridge_count"].gt(0).astype(int)
    )

    articulation_by_month = (
        result.loc[result["is_articulation_point"].eq(1)]
        .groupby("period_month")["node_id"]
        .agg(set)
        .to_dict()
    )
    connected_rows = []
    for month, month_edges in edges.groupby("period_month", sort=False):
        articulation_nodes = articulation_by_month.get(month, set())
        connected = set()
        for edge in month_edges.itertuples(index=False):
            if edge.to_node_id in articulation_nodes:
                connected.add(edge.from_node_id)
            if edge.from_node_id in articulation_nodes:
                connected.add(edge.to_node_id)
        connected_rows.extend(
            {
                "node_id": node_id,
                "period_month": month,
                "connected_to_articulation_point": int(node_id in connected),
            }
            for node_id in result.loc[
                result["period_month"].eq(month), "node_id"
            ]
        )

    connected_frame = pd.DataFrame(connected_rows)
    result = result.merge(
        connected_frame,
        on=["node_id", "period_month"],
        how="left",
        validate="one_to_one",
    )
    result["connected_to_articulation_point"] = (
        result["connected_to_articulation_point"].fillna(0).astype(int)
    )
    return result


def build_terminal_panel(terminal_month, activity_panel, structure, nodes):
    require_columns(
        terminal_month,
        {
            "terminal_id",
            "period_month",
            "active",
            "throughput",
            "capacity_mtpa",
            "role_specific_pagerank",
            "counterparty_hhi_terminal",
            "role_specific_dependence",
        },
        TERMINAL_MONTH_FILE.name,
    )
    terminal = terminal_month.rename(columns={"terminal_id": "node_id"})

    metadata = nodes[
        [column for column in ["node_id", "country"] if column in nodes.columns]
    ].drop_duplicates("node_id")
    terminal = terminal.merge(
        metadata,
        on="node_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_metadata"),
    )
    if "country_metadata" in terminal.columns:
        terminal["country"] = terminal["country"].combine_first(
            terminal["country_metadata"]
        )
        terminal = terminal.drop(columns="country_metadata")

    activity_covariates = activity_panel[
        [
            "node_id",
            "period_month",
            "activity_lag1",
            "voyage_count_lag1",
            "log1p_voyage_count_lag1",
            "neighbor_activity_lag1",
            "active_neighbor_count_lag1",
            "historical_neighbor_count_lag1",
            "lag_available",
            "month_sin",
            "month_cos"
        ]
    ]
    terminal = terminal.merge(
        activity_covariates,
        on=["node_id", "period_month"],
        how="left",
        validate="one_to_one",
    ).merge(
        structure,
        on=["node_id", "period_month"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_structure"),
    )

    terminal = terminal.sort_values(["node_id", "period_month"]).reset_index(drop=True)
    terminal["throughput_lag1"] = terminal.groupby("node_id")["throughput"].shift(1)
    terminal["log1p_throughput_lag1"] = np.log1p(terminal["throughput_lag1"])
    capacity = pd.to_numeric(terminal["capacity_mtpa"], errors="coerce")
    terminal["log1p_capacity_mtpa"] = np.log1p(capacity)
    terminal = add_seasonality(terminal)
    terminal = terminal.rename(columns={"node_id": "terminal_id"})
    return terminal.sort_values(["period_month", "terminal_id"]).reset_index(drop=True)


def build_chokepoint_panel(node_month, activity_panel, structure):
    require_columns(
        node_month,
        {
            "node_id",
            "node_type",
            "period_month",
            "betweenness",
            "pagerank",
            "share_monthly_network_flow",
        },
        NODE_MONTH_FILE.name,
    )
    chokepoints = node_month.loc[
        node_month["node_type"].astype(str).str.lower().eq("chokepoint")
    ].copy()
    activity_covariates = activity_panel[
        [
            "node_id",
            "period_month",
            "activity_lag1",
            "throughput_lag1",
            "log1p_throughput_lag1",
            "voyage_count_lag1",
            "log1p_voyage_count_lag1",
            "neighbor_activity_lag1",
            "active_neighbor_count_lag1",
            "historical_neighbor_count_lag1",
            "lag_available",
            "month_sin",
            "month_cos",
        ]
    ]
    chokepoints = chokepoints.merge(
        activity_covariates,
        on=["node_id", "period_month"],
        how="left",
        validate="one_to_one",
    ).merge(
        structure,
        on=["node_id", "period_month"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_structure"),
    )
    return add_seasonality(chokepoints).sort_values(
        ["period_month", "node_id"]
    ).reset_index(drop=True)


def validate_outputs(activity, terminals, chokepoints):
    for name, frame, node_column in [
        (ACTIVITY_OUTPUT.name, activity, "node_id"),
        (TERMINAL_OUTPUT.name, terminals, "terminal_id"),
        (CHOKEPOINT_OUTPUT.name, chokepoints, "node_id"),
    ]:
        if frame.empty:
            raise ValueError(f"{name} would be empty.")
        if frame.duplicated([node_column, "period_month"]).any():
            raise ValueError(f"{name} contains duplicate node-month keys.")


def main():
    node_month, terminal_month, structure, nodes, edges = load_inputs()
    structure = add_structural_indicators(structure, edges)
    activity = build_activity_panel(node_month, nodes, edges)
    terminals = build_terminal_panel(
        terminal_month, activity, structure, nodes
    )
    chokepoints = build_chokepoint_panel(node_month, activity, structure)
    validate_outputs(activity, terminals, chokepoints)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    activity.to_csv(ACTIVITY_OUTPUT, index=False, float_format="%.12g")
    terminals.to_csv(TERMINAL_OUTPUT, index=False, float_format="%.12g")
    chokepoints.to_csv(CHOKEPOINT_OUTPUT, index=False, float_format="%.12g")


if __name__ == "__main__":
    main()
