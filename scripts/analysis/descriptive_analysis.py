from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"
EDA_DIR = BASE_DIR / "eda_outputs"
OUTPUT_DIR = EDA_DIR / "network_level"
DATA_OUTPUT_DIR = OUTPUT_DIR / "data"
DISTRIBUTIONS_DIR = OUTPUT_DIR / "distributions"
CORRELATIONS_DIR = OUTPUT_DIR / "correlations"
TEMPORAL_DIR = OUTPUT_DIR / "temporal"
RANKINGS_DIR = OUTPUT_DIR / "rankings"

for directory in [
    OUTPUT_DIR,
    DATA_OUTPUT_DIR,
    DISTRIBUTIONS_DIR,
    CORRELATIONS_DIR,
    TEMPORAL_DIR,
    RANKINGS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# FILES
# ============================================================

NODES_FILE = DATA_DIR / "LNG_multilayer_nodes.csv"
EDGES_FILE = DATA_DIR / "LNG_multilayer_edges_monthly.csv"
MONTHLY_QA_FILE = DATA_DIR / "LNG_multilayer_monthly_QA.csv"

# Add other files here later if needed.
# ROUTES_FILE = DATA_DIR / "LNG_1037_routes_with_final_chokepoints.geojson"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    nodes = pd.read_csv(NODES_FILE)
    edges = pd.read_csv(EDGES_FILE)
    monthly_qa = pd.read_csv(MONTHLY_QA_FILE)

    return nodes, edges, monthly_qa


# ============================================================
# NODE ANALYSIS
# ============================================================

def analyze_nodes(nodes):
    # --------------------------------------------------------
    # Node types
    # --------------------------------------------------------

    type_col = "node_type"

    node_types = (
        nodes[type_col]
        .value_counts(dropna=False)
        .rename_axis(type_col)
        .reset_index(name="count")
    )

    plt.figure(figsize=(8, 5))
    node_types.set_index(type_col)["count"].plot(kind="bar")

    plt.title("Number of nodes by type")
    plt.xlabel("Node type")
    plt.ylabel("Number of nodes")
    plt.tight_layout()

    plt.savefig(
        DISTRIBUTIONS_DIR / "nodes_by_type.png",
        dpi=300
    )
    plt.close()

    # --------------------------------------------------------
    # Countries
    # --------------------------------------------------------

    country_col = "country"

    nodes_with_country = nodes[nodes[country_col].notna()].copy()

    countries = (
        nodes_with_country[country_col]
        .value_counts()
        .rename_axis(country_col)
        .reset_index(name="node_count")
    )

    # --------------------------------------------------------
    # Terminal processing capacity
    # --------------------------------------------------------

    terminal_nodes = (
        nodes[nodes["node_type"].eq("terminal")]
        .copy()
    )

    terminal_nodes["capacity_mtpa"] = pd.to_numeric(
        terminal_nodes["capacity_mtpa"],
        errors="raise"
    )


    # --------------------------------------------------------
    # Capacity by infrastructure type
    # --------------------------------------------------------

    capacity_by_type = (
        terminal_nodes
        .groupby(
            "infrastructure_type",
            dropna=False
        )
        .agg(
            terminal_count=("node_id", "nunique"),
            total_capacity_mtpa=("capacity_mtpa", "sum"),
            mean_capacity_mtpa=("capacity_mtpa", "mean"),
            median_capacity_mtpa=("capacity_mtpa", "median"),
            min_capacity_mtpa=("capacity_mtpa", "min"),
            max_capacity_mtpa=("capacity_mtpa", "max"),
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Capacity by country
    # --------------------------------------------------------

    capacity_by_country = (
        terminal_nodes
        .groupby(
            "country",
            dropna=False
        )
        .agg(
            terminal_count=("node_id", "nunique"),
            total_capacity_mtpa=("capacity_mtpa", "sum"),
            mean_capacity_mtpa=("capacity_mtpa", "mean"),
            median_capacity_mtpa=("capacity_mtpa", "median"),
        )
        .reset_index()
        .sort_values(
            "total_capacity_mtpa",
            ascending=False
        )
    )

    # Consolidate related summaries to keep the data output
    # directory compact while preserving the aggregation level.
    node_type_summary = (
        node_types
        .rename(
            columns={
                type_col: "group",
                "count": "node_count",
            }
        )
        .assign(grouping_dimension="node_type")
    )

    capacity_by_type = capacity_by_type.rename(
        columns={"infrastructure_type": "group", "terminal_count": "node_count"}
    ).assign(grouping_dimension="infrastructure_type")

    capacity_by_country = capacity_by_country.rename(
        columns={"country": "group", "terminal_count": "node_count",}
    ).assign(grouping_dimension="country")

    node_group_summary = pd.concat(
        [node_type_summary, capacity_by_type, capacity_by_country],
        ignore_index=True,
    )

    column_order = [
        "grouping_dimension",
        "group",
        "node_count",
        "total_capacity_mtpa",
        "mean_capacity_mtpa",
        "median_capacity_mtpa",
        "min_capacity_mtpa",
        "max_capacity_mtpa",
    ]

    node_group_summary = node_group_summary.reindex(
        columns=column_order
    )
    node_group_summary.to_csv(
        DATA_OUTPUT_DIR / "node_group_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Capacity distribution
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.hist(
        terminal_nodes["capacity_mtpa"],
        bins=30,
        edgecolor="black"
    )

    plt.title("Distribution of LNG terminal processing capacity")
    plt.xlabel("Processing capacity (MTPA)")
    plt.ylabel("Number of terminals")
    plt.tight_layout()

    plt.savefig(
        DISTRIBUTIONS_DIR /
        "terminal_capacity_distribution.png",
        dpi=300
    )
    plt.close()


# ============================================================
# EDGE ANALYSIS
# ============================================================

def analyze_edges(edges):
    # --------------------------------------------------------
    # Edge types
    # --------------------------------------------------------

    edge_types = (
        edges["edge_type"]
        .value_counts(dropna=False)
        .rename_axis("edge_type")
        .reset_index(name="count")
    )

    edge_types.to_csv(
        DATA_OUTPUT_DIR / "edge_types.csv",
        index=False
    )

    # --------------------------------------------------------
    # Extreme-value QA
    # --------------------------------------------------------

    qa_variables = [
        "lng_flow_cmb",
        "share_global_monthly_lng",
        "voyage_count",
        "route_count",
    ]

    qa_variables = [
        col for col in qa_variables
        if col in edges.columns
    ]

    qa_columns = [
        "period_month",
        "from_node_id",
        "to_node_id",
        "edge_type",
        "lng_flow_cmb",
        "voyage_count",
        "route_count",
        "share_global_monthly_lng",
        "weight_row_normalized",
    ]

    qa_columns = [
        col for col in qa_columns
        if col in edges.columns
    ]

    extreme_rows = []

    for variable in qa_variables:
        top = (
            edges
            .sort_values(variable, ascending=False)
            .head(20)
            .copy()
        )

        top["qa_variable"] = variable
        top["qa_rank"] = range(1, len(top) + 1)

        extreme_rows.append(
            top[["qa_variable", "qa_rank"] + qa_columns]
        )

    extreme_qa = pd.concat(
        extreme_rows,
        ignore_index=True
    )

    extreme_qa.to_csv(
        DATA_OUTPUT_DIR / "extreme_values_qa.csv",
        index=False
    )

# ============================================================
# TEMPORAL ANALYSIS
# ============================================================

def analyze_temporal_network(edges, monthly_qa):
    if "period_month" not in edges.columns:
        return

    required_qa = {
        "period_month",
        "export_voyages",
        "total_export_lng_cmb",
    }

    if not required_qa.issubset(monthly_qa.columns):
        return

    edges = edges.copy()
    monthly_qa = monthly_qa.copy()

    edges["period_month"] = pd.to_datetime(
        edges["period_month"],
        errors="coerce"
    )

    monthly_qa["period_month"] = pd.to_datetime(
        monthly_qa["period_month"],
        errors="coerce"
    )

    # ========================================================
    # 1. BASIC MONTHLY NETWORK MEASURES
    # ========================================================

    monthly_edges = (
        edges.groupby("period_month")
        .agg(
            active_edges=(
                "edge_period_id",
                "count"
            ) if "edge_period_id" in edges.columns
            else ("from_node_id", "count"),

            voyage_edge_traversals=(
                "voyage_count",
                "sum"
            ),

            total_edge_flow_exposure=(
                "lng_flow_cmb",
                "sum"
            ),
        )
        .reset_index()
    )

    # ========================================================
    # 2. ORIGINAL LNG TRADE MEASURES
    # ========================================================

    monthly_trade = (
        monthly_qa[
            [
                "period_month",
                "export_voyages",
                "total_export_lng_cmb",
            ]
        ]
        .drop_duplicates("period_month")
        .rename(
            columns={
                "export_voyages":
                    "unique_export_voyages",

                "total_export_lng_cmb":
                    "global_export_lng_volume",
            }
        )
    )

    # ========================================================
    # 3. EDGE-FLOW CONCENTRATION
    # ========================================================

    concentration_rows = []

    for period_month, group in edges.groupby("period_month"):

        flow = (
            group["lng_flow_cmb"]
            .fillna(0)
        )

        total_flow = flow.sum()

        if total_flow > 0:

            shares = flow / total_flow

            # HHI across directed edges
            edge_flow_hhi = (
                shares ** 2
            ).sum()

            sorted_shares = (
                shares
                .sort_values(
                    ascending=False
                )
            )

            top_1_edge_share = (
                sorted_shares
                .iloc[:1]
                .sum()
            )

            top_5_edge_share = (
                sorted_shares
                .iloc[:5]
                .sum()
            )

            top_10_edge_share = (
                sorted_shares
                .iloc[:10]
                .sum()
            )

        else:

            edge_flow_hhi = np.nan
            top_1_edge_share = np.nan
            top_5_edge_share = np.nan
            top_10_edge_share = np.nan

        concentration_rows.append(
            {
                "period_month":
                    period_month,

                "edge_flow_hhi":
                    edge_flow_hhi,

                "top_1_edge_flow_share":
                    top_1_edge_share,

                "top_5_edge_flow_share":
                    top_5_edge_share,

                "top_10_edge_flow_share":
                    top_10_edge_share,
            }
        )

    monthly_concentration = pd.DataFrame(
        concentration_rows
    )

    # ========================================================
    # 4. TRUE CHOKEPOINT-LEVEL MONTHLY FLOW
    # ========================================================

    #
    # Important:
    #
    # We do NOT calculate concentration across
    # "chokepoint-related edges".
    #
    # Instead, we reconstruct throughput for each chokepoint
    # node in each month.
    #
    # Since LNG flowing through a chokepoint appears once on
    # an incoming edge and once on an outgoing edge:
    #
    # chokepoint throughput =
    # (incoming flow + outgoing flow) / 2
    #
    # This avoids double counting transit cargo.
    #

    outgoing_cp = (
        edges[
            edges["from_node_id"]
            .astype(str)
            .str.startswith("CP")
        ]
        .groupby(
            [
                "period_month",
                "from_node_id",
            ]
        )["lng_flow_cmb"]
        .sum()
        .reset_index()
        .rename(
            columns={
                "from_node_id":
                    "chokepoint_id",

                "lng_flow_cmb":
                    "outgoing_cp_flow",
            }
        )
    )

    incoming_cp = (
        edges[
            edges["to_node_id"]
            .astype(str)
            .str.startswith("CP")
        ]
        .groupby(
            [
                "period_month",
                "to_node_id",
            ]
        )["lng_flow_cmb"]
        .sum()
        .reset_index()
        .rename(
            columns={
                "to_node_id":
                    "chokepoint_id",

                "lng_flow_cmb":
                    "incoming_cp_flow",
            }
        )
    )

    chokepoint_monthly = (
        outgoing_cp
        .merge(
            incoming_cp,
            on=[
                "period_month",
                "chokepoint_id",
            ],
            how="outer"
        )
        .fillna(0)
    )

    chokepoint_monthly["chokepoint_throughput"] = (
        chokepoint_monthly["incoming_cp_flow"]
        + chokepoint_monthly["outgoing_cp_flow"]
    ) / 2

    chokepoint_monthly["flow_imbalance"] = abs(
        chokepoint_monthly["incoming_cp_flow"]
        - chokepoint_monthly["outgoing_cp_flow"]
    )

    # Save node-month chokepoint throughput
    chokepoint_monthly.to_csv(
        DATA_OUTPUT_DIR /
        "monthly_chokepoint_throughput.csv",
        index=False
    )

    # ========================================================
    # 5. TRUE CHOKEPOINT HHI
    # ========================================================

    chokepoint_concentration_rows = []

    for period_month, group in (
        chokepoint_monthly
        .groupby("period_month")
    ):

        throughput = (
            group["chokepoint_throughput"]
            .fillna(0)
        )

        total_cp_throughput = (
            throughput.sum()
        )

        active_chokepoints = (
            throughput > 0
        ).sum()

        if total_cp_throughput > 0:

            cp_shares = (
                throughput
                / total_cp_throughput
            )

            chokepoint_flow_hhi = (
                cp_shares ** 2
            ).sum()

            sorted_cp_shares = (
                cp_shares
                .sort_values(
                    ascending=False
                )
            )

            top_1_chokepoint_share = (
                sorted_cp_shares
                .iloc[:1]
                .sum()
            )

            top_5_chokepoint_share = (
                sorted_cp_shares
                .iloc[:5]
                .sum()
            )

        else:

            chokepoint_flow_hhi = np.nan
            top_1_chokepoint_share = np.nan
            top_5_chokepoint_share = np.nan

        chokepoint_concentration_rows.append(
            {
                "period_month":
                    period_month,

                "active_chokepoints":
                    active_chokepoints,

                "chokepoint_flow_hhi":
                    chokepoint_flow_hhi,

                "top_1_chokepoint_flow_share":
                    top_1_chokepoint_share,

                "top_5_chokepoint_flow_share":
                    top_5_chokepoint_share,
            }
        )

    monthly_chokepoint_concentration = (
        pd.DataFrame(
            chokepoint_concentration_rows
        )
    )

    # ========================================================
    # 6. MERGE MONTHLY MEASURES
    # ========================================================

    monthly = (
        monthly_edges
        .merge(
            monthly_trade,
            on="period_month",
            how="left"
        )
        .merge(
            monthly_concentration,
            on="period_month",
            how="left"
        )
        .merge(
            monthly_chokepoint_concentration,
            on="period_month",
            how="left"
        )
        .sort_values("period_month")
        .reset_index(drop=True)
    )

    # ========================================================
    # 7. ROUTE-STRUCTURE INDICATOR
    # ========================================================

    monthly["mean_edge_traversals_per_voyage"] = (
        monthly["voyage_edge_traversals"]
        / monthly["unique_export_voyages"]
    )

    # ========================================================
    # 8. TEMPORAL ACTIVITY QA
    # ========================================================

    #
    # These indicators do NOT determine whether a month has
    # incomplete data coverage.
    #
    # They compare each month with the historical median and
    # flag unusually low observed network activity for manual
    # inspection.
    #

    median_voyages = (
        monthly["unique_export_voyages"]
        .median()
    )

    median_volume = (
        monthly["global_export_lng_volume"]
        .median()
    )

    median_active_edges = (
        monthly["active_edges"]
        .median()
    )

    monthly["voyages_vs_median"] = (
        monthly["unique_export_voyages"]
        / median_voyages
    )

    monthly["volume_vs_median"] = (
        monthly["global_export_lng_volume"]
        / median_volume
    )

    monthly["active_edges_vs_median"] = (
        monthly["active_edges"]
        / median_active_edges
    )

    # Flag individual low-activity indicators.
    monthly["qa_low_voyages"] = (
        monthly["voyages_vs_median"] < 0.70
    )

    monthly["qa_low_volume"] = (
        monthly["volume_vs_median"] < 0.70
    )

    monthly["qa_low_active_edges"] = (
        monthly["active_edges_vs_median"] < 0.70
    )

    # Flag months in which at least two of the three activity
    # indicators fall below 70% of their historical median.
    #
    # This is an activity flag, not evidence of incomplete
    # temporal data coverage.
    monthly["qa_low_activity_watch"] = (
        monthly[
            [
                "qa_low_voyages",
                "qa_low_volume",
                "qa_low_active_edges",
            ]
        ]
        .sum(axis=1)
        >= 2
    )

    monthly.to_csv(
        TEMPORAL_DIR /
        "monthly_network_statistics.csv",
        index=False
    )

    # ========================================================
    # PLOT 1 — GLOBAL LNG EXPORT VOLUME
    # ========================================================

    plt.figure(figsize=(10, 5))

    plt.plot(
        monthly["period_month"],
        monthly["global_export_lng_volume"],
        marker="o",
        markersize=3
    )

    plt.title(
        "Global LNG export volume over time"
    )

    plt.xlabel("Month")

    plt.ylabel(
        "Exported LNG volume (cmb)"
    )

    plt.tight_layout()

    plt.savefig(
        TEMPORAL_DIR /
        "global_lng_export_volume_over_time.png",
        dpi=300
    )

    plt.close()

    # ========================================================
    # PLOT 2 — ACTIVE EDGES
    # ========================================================

    plt.figure(figsize=(10, 5))

    plt.plot(
        monthly["period_month"],
        monthly["active_edges"],
        marker="o",
        markersize=3
    )

    plt.title(
        "Active network edges over time"
    )

    plt.xlabel("Month")

    plt.ylabel(
        "Number of active edges"
    )

    plt.tight_layout()

    plt.savefig(
        TEMPORAL_DIR /
        "active_edges_over_time.png",
        dpi=300
    )

    plt.close()

    # ========================================================
    # PLOT 3 — MEAN EDGE TRAVERSALS PER VOYAGE
    # ========================================================

    plt.figure(figsize=(10, 5))

    plt.plot(
        monthly["period_month"],
        monthly[
            "mean_edge_traversals_per_voyage"
        ],
        marker="o",
        markersize=3
    )

    plt.title(
        "Mean network edge traversals "
        "per LNG voyage"
    )

    plt.xlabel("Month")

    plt.ylabel(
        "Edge traversals per voyage"
    )

    plt.tight_layout()

    plt.savefig(
        TEMPORAL_DIR /
        "mean_edge_traversals_per_voyage.png",
        dpi=300
    )

    plt.close()

    # ========================================================
    # PLOT 5 — TOP-10 EDGE SHARE
    # ========================================================

    plt.figure(figsize=(10, 5))

    plt.plot(
        monthly["period_month"],
        monthly[
            "top_10_edge_flow_share"
        ],
        marker="o",
        markersize=3
    )

    plt.title(
        "Share of LNG edge-flow "
        "carried by top 10 edges"
    )

    plt.xlabel("Month")

    plt.ylabel(
        "Share of monthly edge-flow"
    )

    plt.tight_layout()

    plt.savefig(
        TEMPORAL_DIR /
        "top_10_edge_flow_share_over_time.png",
        dpi=300
    )

    plt.close()

    # ========================================================
    # PLOT 7 — ACTIVE CHOKEPOINTS
    # ========================================================

    plt.figure(figsize=(10, 5))

    plt.plot(
        monthly["period_month"],
        monthly[
            "active_chokepoints"
        ],
        marker="o",
        markersize=3
    )

    plt.title(
        "Active LNG chokepoints over time"
    )

    plt.xlabel("Month")

    plt.ylabel(
        "Number of active chokepoints"
    )

    plt.tight_layout()

    plt.savefig(
        TEMPORAL_DIR /
        "active_chokepoints_over_time.png",
        dpi=300
    )

    plt.close()

# ============================================================
# FLOW DISTRIBUTION
# ============================================================

def analyze_flow_distribution(edges):
    flow = edges["lng_flow_cmb"].dropna()

    statistics = flow.describe(
        percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    )

    statistics.to_csv(
        DATA_OUTPUT_DIR / "lng_flow_descriptive_statistics.csv"
    )

    # --------------------------------------------------------
    # Histogram
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.hist(
        flow,
        bins=50
    )

    plt.title("Distribution of LNG flow across edge-month observations")
    plt.xlabel("LNG flow")
    plt.ylabel("Frequency")
    plt.tight_layout()

    plt.savefig(
        DISTRIBUTIONS_DIR / "lng_flow_distribution.png",
        dpi=300
    )
    plt.close()

    # --------------------------------------------------------
    # Log histogram
    # --------------------------------------------------------

    positive_flow = flow[flow > 0]

    if len(positive_flow) > 0:
        plt.figure(figsize=(8, 5))

        plt.hist(
            np.log10(positive_flow),
            bins=50
        )

        plt.title("Distribution of LNG flow - log10 scale")
        plt.xlabel("log10 LNG flow")
        plt.ylabel("Frequency")
        plt.tight_layout()

        plt.savefig(
            DISTRIBUTIONS_DIR / "lng_flow_distribution_log.png",
            dpi=300
        )
        plt.close()

# ============================================================
# TOP EDGES
# ============================================================

def analyze_top_edges(edges):
    required = {
        "from_node_id",
        "to_node_id",
        "lng_flow_cmb",
    }

    top_edges = (
        edges.groupby(
            ["from_node_id", "to_node_id"],
            as_index=False
        )
        .agg(
            cumulative_edge_flow=("lng_flow_cmb", "sum"),
            mean_monthly_edge_flow=("lng_flow_cmb", "mean"),
            active_months=("period_month", "nunique")
            if "period_month" in edges.columns
            else ("lng_flow_cmb", "count")
        )
        .sort_values(
            "cumulative_edge_flow",
            ascending=False
        )
    )

    voyages = (
        edges.groupby(
            ["from_node_id", "to_node_id"],
            as_index=False
        )["voyage_count"]
        .sum()
        .rename(
            columns={
                "voyage_count": "cumulative_voyage_traversals"
            }
        )
    )

    top_edges = top_edges.merge(
        voyages,
        on=["from_node_id", "to_node_id"],
        how="left"
    )

    top_edges.to_csv(
        RANKINGS_DIR / "edges_ranked_by_cumulative_flow.csv",
        index=False
    )

# ============================================================
# NODE ACTIVITY
# ============================================================

def analyze_node_activity(nodes, edges):
    required = {
        "from_node_id",
        "to_node_id",
        "lng_flow_cmb",
    }

    # --------------------------------------------------------
    # Outgoing activity
    # --------------------------------------------------------

    outgoing = (
        edges.groupby("from_node_id")
        .agg(
            outgoing_flow=("lng_flow_cmb", "sum"),
            outgoing_edge_records=("to_node_id", "count"),
            outgoing_neighbors=("to_node_id", "nunique"),
        )
        .reset_index()
        .rename(columns={"from_node_id": "node_id"})
    )

    # --------------------------------------------------------
    # Incoming activity
    # --------------------------------------------------------

    incoming = (
        edges.groupby("to_node_id")
        .agg(
            incoming_flow=("lng_flow_cmb", "sum"),
            incoming_edge_records=("from_node_id", "count"),
            incoming_neighbors=("from_node_id", "nunique"),
        )
        .reset_index()
        .rename(columns={"to_node_id": "node_id"})
    )

    # --------------------------------------------------------
    # Merge incoming and outgoing activity
    # --------------------------------------------------------

    activity = outgoing.merge(
        incoming,
        on="node_id",
        how="outer"
    ).fillna(0)

    # --------------------------------------------------------
    # Add node type
    # --------------------------------------------------------

    node_info = (
        nodes[
            ["node_id", "node_type"]
        ]
        .drop_duplicates("node_id")
    )

    activity = activity.merge(
        node_info,
        on="node_id",
        how="left"
    )

    # --------------------------------------------------------
    # Flow indicators
    # --------------------------------------------------------

    # Sum of all incoming and outgoing edge flows associated with the node.
    # For chokepoints, this double-counts transit cargo because the same LNG appears once on the incoming edge and once on the outgoing edge.
    activity["incident_flow"] = (
        activity["incoming_flow"]
        + activity["outgoing_flow"]
    )

    # Observed LNG throughput.
    # For chokepoints dividing incident flow by two avoids double counting.
     # For LNG terminals the total incident flow is the appropriate observed throughput.
    activity["node_throughput"] = np.where(
        activity["node_type"]
        .astype(str)
        .str.lower()
        .eq("chokepoint"),

        (
            activity["incoming_flow"]
            + activity["outgoing_flow"]
        ) / 2,

        activity["incoming_flow"]
        + activity["outgoing_flow"]
    )

    activity["flow_imbalance"] = abs(
        activity["outgoing_flow"]
        - activity["incoming_flow"]
    )

    # --------------------------------------------------------
    # Connectivity indicators
    # --------------------------------------------------------

    # Directed degree
    activity["in_degree"] = activity["incoming_neighbors"]
    activity["out_degree"] = activity["outgoing_neighbors"]

    # Unique neighbors regardless of edge direction
    neighbor_pairs = pd.concat(
        [
            edges[["from_node_id", "to_node_id"]]
            .rename(
                columns={
                    "from_node_id": "node_id",
                    "to_node_id": "neighbor_id",
                }
            ),

            edges[["to_node_id", "from_node_id"]]
            .rename(
                columns={
                    "to_node_id": "node_id",
                    "from_node_id": "neighbor_id",
                }
            ),
        ],
        ignore_index=True,
    )

    total_degree = (
        neighbor_pairs
        .drop_duplicates(["node_id", "neighbor_id"])
        .groupby("node_id")["neighbor_id"]
        .nunique()
        .reset_index(name="total_degree")
    )

    activity = activity.merge(
        total_degree,
        on="node_id",
        how="left"
    )

    activity["total_degree"] = (
        activity["total_degree"]
        .fillna(0)
        .astype(int)
    )

    # --------------------------------------------------------
    # Rank nodes by observed throughput
    # --------------------------------------------------------

    activity = activity.sort_values(
        "node_throughput",
        ascending=False
    )
    
    activity.to_csv(
        DATA_OUTPUT_DIR / "node_activity.csv",
        index=False
    )

# ============================================================
# CONCENTRATION
# ============================================================

def analyze_flow_concentration(edges):
    edge_flow = (
        edges.groupby(
            ["from_node_id", "to_node_id"],
            as_index=False
        )["lng_flow_cmb"]
        .sum()
        .sort_values(
            "lng_flow_cmb",
            ascending=False
        )
    )

    total_flow = edge_flow["lng_flow_cmb"].sum()

    edge_flow["flow_share"] = (
        edge_flow["lng_flow_cmb"]
        / total_flow
    )

    edge_flow["cumulative_flow_share"] = (
        edge_flow["flow_share"].cumsum()
    )

    edge_flow.to_csv(
        DATA_OUTPUT_DIR / "edge_flow_concentration.csv",
        index=False
    )


# ============================================================
# OUTLIERS
# ============================================================

def analyze_outliers(edges):
    numeric_columns = [
        "lng_flow_cmb",
        "voyage_count",
        "route_count",
        "share_global_monthly_lng",
        "weight_row_normalized",
    ]

    available = [
        col
        for col in numeric_columns
        if col in edges.columns
    ]

    results = []

    for col in available:
        series = edges[col].dropna()

        if len(series) == 0:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = series[
            (series < lower)
            | (series > upper)
        ]

        results.append(
            {
                "variable": col,
                "n": len(series),
                "mean": series.mean(),
                "median": series.median(),
                "min": series.min(),
                "max": series.max(),
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower,
                "upper_bound": upper,
                "outlier_count": len(outliers),
                "outlier_share": (
                    len(outliers) / len(series)
                ),
            }
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        DATA_OUTPUT_DIR / "outlier_summary.csv",
        index=False
    )


# ============================================================
# CORRELATIONS
# ============================================================

def analyze_correlations(edges):
    candidate_columns = [
        "lng_flow_cmb",
        "voyage_count",
        "route_count",
        "share_global_monthly_lng",
        "weight_row_normalized",
    ]

    columns = [
        col
        for col in candidate_columns
        if col in edges.columns
    ]

    if len(columns) < 2:
        return

    corr = edges[columns].corr(
        method="spearman"
    )

    corr.to_csv(
        CORRELATIONS_DIR / "spearman_correlations.csv"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    nodes, edges, monthly_qa = load_data()

    analyze_nodes(nodes)
    analyze_edges(edges)

    analyze_temporal_network(edges, monthly_qa)

    analyze_flow_distribution(edges)

    analyze_top_edges(edges)

    analyze_node_activity(nodes, edges)

    analyze_flow_concentration(edges)

    analyze_outliers(edges)

    analyze_correlations(edges)

if __name__ == "__main__":
    main()
