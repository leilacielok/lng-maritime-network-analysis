from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "eda_outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# FILES
# ============================================================

NODES_FILE = DATA_DIR / "LNG_multilayer_nodes_v1.csv"
EDGES_FILE = DATA_DIR / "LNG_multilayer_edges_monthly_v1.csv"
MONTHLY_QA_FILE = DATA_DIR / "LNG_multilayer_monthly_QA_v1.csv"

# Add other files here later if needed.
# ROUTES_FILE = DATA_DIR / "LNG_1037_routes_with_final_chokepoints_v1.geojson"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    nodes = pd.read_csv(NODES_FILE)
    edges = pd.read_csv(EDGES_FILE)
    monthly_qa = pd.read_csv(MONTHLY_QA_FILE)

    print(f"Nodes file: {NODES_FILE.name}")
    print(f"Rows: {len(nodes):,}")
    print(f"Columns: {len(nodes.columns)}")

    print()

    print(f"Edges file: {EDGES_FILE.name}")
    print(f"Rows: {len(edges):,}")
    print(f"Columns: {len(edges.columns)}")

    print()

    print(f"Monthly QA file: {MONTHLY_QA_FILE.name}")
    print(f"Rows: {len(monthly_qa):,}")
    print(f"Columns: {len(monthly_qa.columns)}")

    return nodes, edges, monthly_qa


# ============================================================
# BASIC DATASET OVERVIEW
# ============================================================

def dataset_overview(df, name):
    print("\n" + "=" * 70)
    print(f"{name.upper()} DATASET OVERVIEW")
    print("=" * 70)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    for col in df.columns:
        print(f"  - {col}")

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    missing = (
        df.isna()
        .sum()
        .sort_values(ascending=False)
        .to_frame("missing_count")
    )

    missing["missing_share"] = missing["missing_count"] / len(df)

    print(missing[missing["missing_count"] > 0])

    missing.to_csv(
        OUTPUT_DIR / f"{name.lower()}_missing_values.csv"
    )

    print("\nDuplicated rows:")
    print(df.duplicated().sum())


# ============================================================
# NODE ANALYSIS
# ============================================================

def analyze_nodes(nodes):
    print("\n" + "=" * 70)
    print("NODE ANALYSIS")
    print("=" * 70)

    print(f"\nTotal node records: {len(nodes):,}")

    # --------------------------------------------------------
    # Node types
    # --------------------------------------------------------

    possible_type_columns = [
        "node_type",
        "type",
        "layer",
        "category",
    ]

    type_col = next(
        (col for col in possible_type_columns if col in nodes.columns),
        None
    )

    if type_col:
        node_types = (
            nodes[type_col]
            .value_counts(dropna=False)
            .rename_axis(type_col)
            .reset_index(name="count")
        )

        print(f"\nNodes by {type_col}:")
        print(node_types)

        node_types.to_csv(
            OUTPUT_DIR / "nodes_by_type.csv",
            index=False
        )

        plt.figure(figsize=(8, 5))
        node_types.set_index(type_col)["count"].plot(kind="bar")

        plt.title("Number of nodes by type")
        plt.xlabel("Node type")
        plt.ylabel("Number of nodes")
        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "nodes_by_type.png",
            dpi=300
        )
        plt.close()

    # --------------------------------------------------------
    # Countries
    # --------------------------------------------------------

    possible_country_columns = [
        "country",
        "country_name",
        "iso3",
    ]

    country_col = next(
        (col for col in possible_country_columns if col in nodes.columns),
        None
    )

    if country_col:

        # ----------------------------------------------------
        # All nodes with country information
        # ----------------------------------------------------

        nodes_with_country = nodes[nodes[country_col].notna()].copy()

        countries = (
            nodes_with_country[country_col]
            .value_counts()
            .rename_axis(country_col)
            .reset_index(name="node_count")
        )

        countries.to_csv(
            OUTPUT_DIR / "nodes_by_country.csv",
            index=False
        )

        print("\nTop countries by number of nodes:")
        print(countries.head(20))

        # ----------------------------------------------------
        # Missing country values
        # ----------------------------------------------------

        missing_country = nodes[nodes[country_col].isna()].copy()

        print(
            f"\nNodes without country information: "
            f"{len(missing_country):,}"
        )

        if len(missing_country) > 0:
            columns_to_show = ["node_id"]

            if type_col:
                columns_to_show.append(type_col)

            available_columns = [
                col for col in columns_to_show
                if col in missing_country.columns
            ]

            print(
                missing_country[
                    available_columns
                ].head(50)
            )

            missing_country.to_csv(
                OUTPUT_DIR / "nodes_missing_country.csv",
                index=False
            )

        # ----------------------------------------------------
        # Country counts by node type
        # ----------------------------------------------------

        if type_col:

            country_by_type = (
                nodes_with_country
                .groupby([type_col, country_col])
                .size()
                .reset_index(name="node_count")
                .sort_values(
                    ["node_count"],
                    ascending=False
                )
            )

            country_by_type.to_csv(
                OUTPUT_DIR / "nodes_by_country_and_type.csv",
                index=False
            )


# ============================================================
# EDGE ANALYSIS
# ============================================================

def analyze_edges(edges):
    print("\n" + "=" * 70)
    print("EDGE ANALYSIS")
    print("=" * 70)

    print(f"\nTotal edge-period observations: {len(edges):,}")

    # --------------------------------------------------------
    # Unique physical/directed edges
    # --------------------------------------------------------

    required_cols = {"from_node_id", "to_node_id"}

    if required_cols.issubset(edges.columns):
        unique_edges = (
            edges[["from_node_id", "to_node_id"]]
            .drop_duplicates()
        )

        print(
            f"Unique directed node pairs: "
            f"{len(unique_edges):,}"
        )

    # --------------------------------------------------------
    # Edge types
    # --------------------------------------------------------

    if "edge_type" in edges.columns:
        edge_types = (
            edges["edge_type"]
            .value_counts(dropna=False)
            .rename_axis("edge_type")
            .reset_index(name="count")
        )

        print("\nEdge types:")
        print(edge_types)

        edge_types.to_csv(
            OUTPUT_DIR / "edge_types.csv",
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
        OUTPUT_DIR / "extreme_values_qa.csv",
        index=False
    )

    print("\nTop extreme observations:")
    print(extreme_qa.head(40).to_string(index=False))

# ============================================================
# TEMPORAL ANALYSIS
# ============================================================

def analyze_temporal_network(edges, monthly_qa):
    if "period_month" not in edges.columns:
        print("\nNo period_month column found. Temporal analysis skipped.")
        return

    required_qa = {
        "period_month",
        "export_voyages",
        "total_export_lng_cmb",
    }

    if not required_qa.issubset(monthly_qa.columns):
        missing = required_qa.difference(monthly_qa.columns)

        print(
            "\nMonthly QA file is missing required columns: "
            + ", ".join(sorted(missing))
        )
        return

    print("\n" + "=" * 70)
    print("TEMPORAL NETWORK ANALYSIS")
    print("=" * 70)

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
        OUTPUT_DIR /
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

    activity_detail = (
        monthly[
            [
                "period_month",
                "unique_export_voyages",
                "global_export_lng_volume",
                "active_edges",
                "voyages_vs_median",
                "volume_vs_median",
                "active_edges_vs_median",
                "qa_low_activity_watch",
            ]
        ]
        .sort_values("period_month")
    )

    activity_detail.to_csv(
        OUTPUT_DIR / "temporal_activity_detail.csv",
        index=False
    )
    
    # ========================================================
    # 9. SAVE MONTHLY OUTPUT
    # ========================================================

    print(
        "\nMonthly structural network statistics:"
    )

    print(
        monthly.to_string(
            index=False
        )
    )

    monthly.to_csv(
        OUTPUT_DIR /
        "monthly_network_statistics.csv",
        index=False
    )

    # --------------------------------------------------------
    # Print months requiring low-activity inspection
    # --------------------------------------------------------

    activity_watch = (
        monthly[
            monthly["qa_low_activity_watch"]
        ][
            [
                "period_month",
                "unique_export_voyages",
                "global_export_lng_volume",
                "active_edges",
                "voyages_vs_median",
                "volume_vs_median",
                "active_edges_vs_median",
            ]
        ]
    )

    print(
        "\nMonths flagged for low temporal activity review:"
    )

    if activity_watch.empty:
        print("None")

    else:
        print(
            activity_watch.to_string(
                index=False
            )
        )

    activity_watch.to_csv(
        OUTPUT_DIR / "temporal_activity_watch.csv",
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
        OUTPUT_DIR /
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
        OUTPUT_DIR /
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
        OUTPUT_DIR /
        "mean_edge_traversals_per_voyage.png",
        dpi=300
    )

    plt.close()

    # ========================================================
    # PLOT 4 — EDGE-FLOW HHI
    # ========================================================

    plt.figure(figsize=(10, 5))

    plt.plot(
        monthly["period_month"],
        monthly["edge_flow_hhi"],
        marker="o",
        markersize=3
    )

    plt.title(
        "Monthly concentration of LNG flow "
        "across edges"
    )

    plt.xlabel("Month")

    plt.ylabel(
        "Edge-flow HHI"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        "edge_flow_hhi_over_time.png",
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
        OUTPUT_DIR /
        "top_10_edge_flow_share_over_time.png",
        dpi=300
    )

    plt.close()

    # ========================================================
    # PLOT 6 — TRUE CHOKEPOINT HHI
    # ========================================================

    plt.figure(figsize=(10, 5))

    plt.plot(
        monthly["period_month"],
        monthly[
            "chokepoint_flow_hhi"
        ],
        marker="o",
        markersize=3
    )

    plt.title(
        "Monthly concentration of LNG throughput "
        "across chokepoints"
    )

    plt.xlabel("Month")

    plt.ylabel(
        "Chokepoint throughput HHI"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        "chokepoint_flow_hhi_over_time.png",
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
        OUTPUT_DIR /
        "active_chokepoints_over_time.png",
        dpi=300
    )

    plt.close()

# ============================================================
# FLOW DISTRIBUTION
# ============================================================

def analyze_flow_distribution(edges):
    if "lng_flow_cmb" not in edges.columns:
        print("\nNo lng_flow_cmb column found. Flow analysis skipped.")
        return

    print("\n" + "=" * 70)
    print("LNG FLOW DISTRIBUTION")
    print("=" * 70)

    flow = edges["lng_flow_cmb"].dropna()

    print("\nDescriptive statistics:")
    print(flow.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]))

    statistics = flow.describe(
        percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    )

    statistics.to_csv(
        OUTPUT_DIR / "lng_flow_descriptive_statistics.csv"
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
        OUTPUT_DIR / "lng_flow_distribution.png",
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
            OUTPUT_DIR / "lng_flow_distribution_log.png",
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

    if not required.issubset(edges.columns):
        return

    print("\n" + "=" * 70)
    print("TOP EDGES BY CUMULATIVE LNG FLOW")
    print("=" * 70)

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

    if "voyage_count" in edges.columns:
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

    print("\nTop 20 edges:")
    print(top_edges.head(20))

    top_edges.to_csv(
        OUTPUT_DIR / "edges_ranked_by_cumulative_flow.csv",
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

    if not required.issubset(edges.columns):
        return

    print("\n" + "=" * 70)
    print("NODE ACTIVITY")
    print("=" * 70)

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

    if "node_type" in nodes.columns:

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

    else:
        activity["node_type"] = "unknown"

    # Avoid problems with missing node types
    activity["node_type"] = (
        activity["node_type"]
        .fillna("unknown")
    )

    # --------------------------------------------------------
    # Flow indicators
    # --------------------------------------------------------

    # Sum of all incoming and outgoing edge flows associated
    # with the node.
    #
    # For chokepoints, this double-counts transit cargo because
    # the same LNG appears once on the incoming edge and once
    # on the outgoing edge.
    activity["incident_flow"] = (
        activity["incoming_flow"]
        + activity["outgoing_flow"]
    )

    # Observed LNG throughput.
    #
    # For chokepoints:
    # incoming and outgoing flows represent the same physical
    # cargo, so dividing incident flow by two avoids double
    # counting.
    #
    # For LNG terminals:
    # the terminal is normally an origin or destination, so the
    # total incident flow is the appropriate observed throughput.
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

    # Difference between incoming and outgoing flow.
    #
    # For pure transit chokepoints this should generally be
    # close to zero.
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

    print("\nTop 20 nodes by observed LNG throughput:")

    print(
        activity[
            [
                "node_id",
                "node_type",
                "node_throughput",
                "incident_flow",
                "flow_imbalance",
                "incoming_flow",
                "outgoing_flow",
                "in_degree",
                "out_degree",
                "total_degree",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Save output
    # --------------------------------------------------------

    activity.to_csv(
        OUTPUT_DIR / "node_activity.csv",
        index=False
    )


# ============================================================
# CONCENTRATION
# ============================================================

def analyze_flow_concentration(edges):
    if "lng_flow_cmb" not in edges.columns:
        return

    print("\n" + "=" * 70)
    print("FLOW CONCENTRATION")
    print("=" * 70)

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

    if total_flow <= 0:
        return

    edge_flow["flow_share"] = (
        edge_flow["lng_flow_cmb"]
        / total_flow
    )

    edge_flow["cumulative_flow_share"] = (
        edge_flow["flow_share"].cumsum()
    )

    n_edges = len(edge_flow)

    for share in [0.50, 0.75, 0.90]:
        n_required = (
            edge_flow["cumulative_flow_share"]
            < share
        ).sum() + 1

        print(
            f"Edges required to account for "
            f"{share:.0%} of total flow: "
            f"{n_required:,} / {n_edges:,} "
            f"({n_required / n_edges:.1%})"
        )

    edge_flow.to_csv(
        OUTPUT_DIR / "edge_flow_concentration.csv",
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

    if not available:
        return

    print("\n" + "=" * 70)
    print("OUTLIER CHECK")
    print("=" * 70)

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

    print(results_df)

    results_df.to_csv(
        OUTPUT_DIR / "outlier_summary.csv",
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

    print("\n" + "=" * 70)
    print("CORRELATIONS")
    print("=" * 70)

    corr = edges[columns].corr(
        method="spearman"
    )

    print("\nSpearman correlation matrix:")
    print(corr.round(3))

    corr.to_csv(
        OUTPUT_DIR / "spearman_correlations.csv"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    nodes, edges, monthly_qa = load_data()

    dataset_overview(nodes, "nodes")
    dataset_overview(edges, "edges")

    analyze_nodes(nodes)
    analyze_edges(edges)

    analyze_temporal_network(edges, monthly_qa)

    analyze_flow_distribution(edges)

    analyze_top_edges(edges)

    analyze_node_activity(nodes, edges)

    analyze_flow_concentration(edges)

    analyze_outliers(edges)

    analyze_correlations(edges)

    print("\n" + "=" * 70)
    print("EDA COMPLETED")
    print("=" * 70)

    print(
        f"\nResults saved in:\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()