from pathlib import Path

import networkx as nx
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"

NETWORK_LEVEL_DIR = BASE_DIR / "eda_outputs" / "network_level"
DATA_OUTPUT_DIR = NETWORK_LEVEL_DIR / "data"
TEMPORAL_OUTPUT_DIR = NETWORK_LEVEL_DIR / "temporal"

NODES_FILE = DATA_DIR / "LNG_multilayer_nodes_observed.csv"
EDGES_FILE = DATA_DIR / "LNG_multilayer_edges_monthly.csv"

NODE_MONTH_OUTPUT = DATA_OUTPUT_DIR / "node_month_structure_metrics.csv"
MONTHLY_NETWORK_OUTPUT = (
    TEMPORAL_OUTPUT_DIR / "monthly_network_structure.csv"
)


# ============================================================
# DATA VALIDATION AND LOADING
# ============================================================

def validate_columns(df, required, dataset_name):
    missing = sorted(set(required).difference(df.columns))

    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: "
            + ", ".join(missing)
        )


def load_data():
    nodes = pd.read_csv(NODES_FILE)
    edges = pd.read_csv(EDGES_FILE)

    validate_columns(
        nodes,
        {"node_id", "node_type"},
        "Nodes dataset",
    )
    validate_columns(
        edges,
        {"period_month", "from_node_id", "to_node_id"},
        "Edges dataset",
    )

    nodes = nodes.copy()
    edges = edges.copy()

    nodes["node_id"] = nodes["node_id"].astype(str)
    nodes["node_type"] = (
        nodes["node_type"].astype(str).str.lower()
    )

    edges["from_node_id"] = edges["from_node_id"].astype(str)
    edges["to_node_id"] = edges["to_node_id"].astype(str)
    edges["period_month"] = pd.to_datetime(
        edges["period_month"],
        errors="raise",
    ).dt.to_period("M").dt.to_timestamp()

    conflicting_types = (
        nodes.groupby("node_id")["node_type"].nunique(dropna=False)
    )
    conflicting_types = conflicting_types[conflicting_types > 1]

    if not conflicting_types.empty:
        raise ValueError(
            "Some node_id values have multiple node_type values: "
            + ", ".join(conflicting_types.index.astype(str))
        )

    node_metadata = (
        nodes[["node_id", "node_type"]]
        .drop_duplicates("node_id")
        .sort_values("node_id")
        .reset_index(drop=True)
    )

    known_nodes = set(node_metadata["node_id"])
    edge_nodes = set(edges["from_node_id"]).union(
        edges["to_node_id"]
    )
    unknown_nodes = sorted(edge_nodes.difference(known_nodes))

    if unknown_nodes:
        raise ValueError(
            "Edges contain node identifiers absent from the nodes file: "
            + ", ".join(unknown_nodes)
        )

    return node_metadata, edges


# ============================================================
# MONTHLY GRAPH
# ============================================================

def build_monthly_graph(month_edges):
    """Build one active directed monthly graph.

    Repeated records for the same ordered pair are collapsed. Self-loops are
    excluded because they do not contribute to connectivity or bridge status.
    """
    active_pairs = (
        month_edges[["from_node_id", "to_node_id"]]
        .drop_duplicates()
    )
    active_pairs = active_pairs[
        active_pairs["from_node_id"]
        != active_pairs["to_node_id"]
    ]

    graph = nx.DiGraph()
    graph.add_edges_from(
        active_pairs.itertuples(index=False, name=None)
    )

    return graph


# ============================================================
# MODEL-RELEVANT NETWORK-LEVEL MEASURES
# ============================================================

def calculate_monthly_network_metrics(period_month, directed_graph):
    n_nodes = directed_graph.number_of_nodes()
    n_edges = directed_graph.number_of_edges()

    if n_nodes == 0:
        return {
            "period_month": period_month,
            "active_nodes": 0,
            "active_directed_edges": 0,
            "directed_density": 0.0,
            "weak_components": 0,
            "largest_weak_component_share": 0.0,
            "is_weakly_connected": 0,
        }

    weak_components = list(
        nx.weakly_connected_components(directed_graph)
    )
    largest_weak_component = max(map(len, weak_components))

    return {
        "period_month": period_month,
        "active_nodes": n_nodes,
        "active_directed_edges": n_edges,
        "directed_density": nx.density(directed_graph),
        "weak_components": len(weak_components),
        "largest_weak_component_share": (
            largest_weak_component / n_nodes
        ),
        "is_weakly_connected": int(len(weak_components) == 1),
    }


# ============================================================
# MODEL-RELEVANT NODE-LEVEL MEASURES
# ============================================================

def calculate_active_node_metrics(period_month, directed_graph):
    """Calculate structural measures for active nodes in one month.

    Bridge and articulation concepts are evaluated on the undirected
    projection. Global bridges are removed from the local-bridge set so that
    local_bridge_only_count represents a distinct structural mechanism.
    """
    undirected_graph = directed_graph.to_undirected()

    global_bridges = {
        frozenset((u, v))
        for u, v in nx.bridges(undirected_graph)
    }
    local_bridges = {
        frozenset((u, v))
        for u, v in nx.local_bridges(
            undirected_graph,
            with_span=False,
        )
    }
    local_bridges_only = local_bridges.difference(global_bridges)
    articulation_points = set(
        nx.articulation_points(undirected_graph)
    )

    rows = []

    for node_id in undirected_graph.nodes:
        degree = undirected_graph.degree(node_id)
        incident_edges = {
            frozenset((node_id, neighbor))
            for neighbor in undirected_graph.neighbors(node_id)
        }

        global_bridge_count = len(
            incident_edges.intersection(global_bridges)
        )
        local_bridge_only_count = len(
            incident_edges.intersection(local_bridges_only)
        )

        rows.append(
            {
                "period_month": period_month,
                "node_id": node_id,
                "is_active": 1,
                "degree_undirected": degree,
                "is_articulation_point": int(
                    node_id in articulation_points
                ),
                "global_bridge_count": global_bridge_count,
                "global_bridge_share": (
                    global_bridge_count / degree
                ),
                "local_bridge_only_count": local_bridge_only_count,
                "local_bridge_only_share": (
                    local_bridge_only_count / degree
                ),
            }
        )

    return rows


def build_complete_node_month_panel(
    node_metadata,
    months,
    active_node_metrics,
):
    panel = (
        pd.MultiIndex.from_product(
            [node_metadata["node_id"], months],
            names=["node_id", "period_month"],
        )
        .to_frame(index=False)
        .merge(
            node_metadata,
            on="node_id",
            how="left",
            validate="many_to_one",
        )
        .merge(
            active_node_metrics,
            on=["node_id", "period_month"],
            how="left",
            validate="one_to_one",
        )
    )

    integer_columns = [
        "is_active",
        "degree_undirected",
        "is_articulation_point",
        "global_bridge_count",
        "local_bridge_only_count",
    ]
    share_columns = [
        "global_bridge_share",
        "local_bridge_only_share",
    ]

    panel[integer_columns] = (
        panel[integer_columns].fillna(0).astype(int)
    )
    panel[share_columns] = panel[share_columns].fillna(0.0)

    return panel[
        [
            "period_month",
            "node_id",
            "node_type",
            "is_active",
            "degree_undirected",
            "is_articulation_point",
            "global_bridge_count",
            "global_bridge_share",
            "local_bridge_only_count",
            "local_bridge_only_share",
        ]
    ].sort_values(["period_month", "node_id"]).reset_index(drop=True)


# ============================================================
# MAIN
# ============================================================

def main():
    node_metadata, edges = load_data()

    months = pd.date_range(
        edges["period_month"].min(),
        edges["period_month"].max(),
        freq="MS",
    )

    network_rows = []
    node_rows = []

    for period_month in months:
        month_edges = edges.loc[
            edges["period_month"].eq(period_month)
        ]
        directed_graph = build_monthly_graph(month_edges)

        network_rows.append(
            calculate_monthly_network_metrics(
                period_month,
                directed_graph,
            )
        )
        node_rows.extend(
            calculate_active_node_metrics(
                period_month,
                directed_graph,
            )
        )

    monthly_network = pd.DataFrame(network_rows).sort_values(
        "period_month"
    )
    active_node_metrics = pd.DataFrame(node_rows)

    node_month_panel = build_complete_node_month_panel(
        node_metadata,
        months,
        active_node_metrics,
    )

    DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMPORAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    monthly_network.to_csv(
        MONTHLY_NETWORK_OUTPUT,
        index=False,
    )
    node_month_panel.to_csv(
        NODE_MONTH_OUTPUT,
        index=False,
    )


if __name__ == "__main__":
    main()
