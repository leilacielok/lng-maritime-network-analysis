from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import networkx as nx
except ImportError as exc:
    raise ImportError(
        "This analysis requires networkx. Install it with: pip install networkx"
    ) from exc


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "eda_node_month"

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

NODES_FILE = DATA_DIR / "LNG_multilayer_nodes_v1.csv"
EDGES_FILE = DATA_DIR / "LNG_multilayer_edges_monthly_v1.csv"


# ============================================================
# SETTINGS
# ============================================================

CENTRALITY_WEIGHT = "lng_flow_cmb"

CORE_METRICS = [
    "node_throughput",
    "total_voyages",
    "total_route_exposure",
    "total_degree",
    "total_strength",
    "betweenness",
    "pagerank",
    "eigenvector",
    "share_monthly_network_flow",
]


# ============================================================
# HELPERS
# ============================================================

def parse_route_ids(value):
    """Return a set of route IDs from the edge-level route_ids field."""
    if pd.isna(value):
        return set()

    value = str(value).strip()

    if not value:
        return set()

    # The source file may store route IDs using different separators.
    for separator in [";", "|", ","]:
        if separator in value:
            return {
                item.strip()
                for item in value.split(separator)
                if item.strip()
            }

    return {value}


def safe_cv(series):
    mean = series.mean()
    if pd.isna(mean) or np.isclose(mean, 0):
        return np.nan
    return series.std(ddof=1) / mean

def lag1_autocorrelation(series):
    series = series.astype(float)

    paired = pd.concat(
        [
            series.shift(1).rename("previous"),
            series.rename("current"),
        ],
        axis=1,
    ).dropna()

    if len(paired) < 3:
        return np.nan

    if paired["previous"].nunique() <= 1:
        return np.nan

    if paired["current"].nunique() <= 1:
        return np.nan

    return paired["previous"].corr(
        paired["current"],
        method="pearson"
    )


def safe_spearman(x, y):
    valid = pd.concat([x, y], axis=1).dropna()

    if len(valid) < 3:
        return np.nan

    if valid.iloc[:, 0].nunique() <= 1:
        return np.nan

    if valid.iloc[:, 1].nunique() <= 1:
        return np.nan

    return valid.iloc[:, 0].corr(valid.iloc[:, 1], method="spearman")


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    print("\n" + "=" * 70)
    print("LOADING NODE-MONTH INPUT DATA")
    print("=" * 70)

    nodes = pd.read_csv(NODES_FILE)
    edges = pd.read_csv(EDGES_FILE)

    required_nodes = {"node_id"}
    required_edges = {
        "period_month",
        "from_node_id",
        "to_node_id",
        "lng_flow_cmb",
    }

    missing_nodes = required_nodes.difference(nodes.columns)
    missing_edges = required_edges.difference(edges.columns)

    if missing_nodes:
        raise ValueError(
            "Nodes file is missing required columns: "
            + ", ".join(sorted(missing_nodes))
        )

    if missing_edges:
        raise ValueError(
            "Edges file is missing required columns: "
            + ", ".join(sorted(missing_edges))
        )

    edges["period_month"] = pd.to_datetime(
        edges["period_month"],
        errors="coerce"
    )

    if edges["period_month"].isna().any():
        raise ValueError(
            "Some period_month values could not be parsed as dates."
        )

    print(f"Nodes: {len(nodes):,}")
    print(f"Edge-month observations: {len(edges):,}")
    print(
        "Months: "
        f"{edges['period_month'].min():%Y-%m} to "
        f"{edges['period_month'].max():%Y-%m}"
    )

    return nodes, edges


# ============================================================
# COMPLETE NODE-MONTH PANEL
# ============================================================

def build_panel_index(nodes, edges):
    """
    Create the complete node x month panel.

    Inactive node-month observations are retained. Activity metrics are
    later set to zero where conceptually appropriate.
    """
    node_ids = (
        nodes["node_id"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values()
    )

    months = (
        edges["period_month"]
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    panel = pd.MultiIndex.from_product(
        [node_ids, months],
        names=["node_id", "period_month"]
    ).to_frame(index=False)

    node_metadata_columns = ["node_id"]

    for col in [
        "node_type",
        "name",
        "node_name",
        "country",
        "country_name",
        "iso3",
    ]:
        if col in nodes.columns and col not in node_metadata_columns:
            node_metadata_columns.append(col)

    node_metadata = (
        nodes[node_metadata_columns]
        .copy()
        .drop_duplicates("node_id")
    )

    node_metadata["node_id"] = node_metadata["node_id"].astype(str)

    panel = panel.merge(
        node_metadata,
        on="node_id",
        how="left"
    )

    return panel


# ============================================================
# FLOW, VOYAGE, ROUTE AND DEGREE MEASURES
# ============================================================

def calculate_basic_node_month_metrics(edges):
    print("\n" + "=" * 70)
    print("CALCULATING NODE-MONTH ACTIVITY METRICS")
    print("=" * 70)

    edges = edges.copy()
    edges["from_node_id"] = edges["from_node_id"].astype(str)
    edges["to_node_id"] = edges["to_node_id"].astype(str)

    # --------------------------------------------------------
    # Incoming measures
    # --------------------------------------------------------

    incoming_agg = {
        "incoming_flow": ("lng_flow_cmb", "sum"),
        "in_degree": ("from_node_id", "nunique"),
    }

    if "voyage_count" in edges.columns:
        incoming_agg["in_voyages"] = ("voyage_count", "sum")

    if "route_count" in edges.columns:
        incoming_agg["in_route_count_sum"] = ("route_count", "sum")

    incoming = (
        edges.groupby(["period_month", "to_node_id"])
        .agg(**incoming_agg)
        .reset_index()
        .rename(columns={"to_node_id": "node_id"})
    )

    # --------------------------------------------------------
    # Outgoing measures
    # --------------------------------------------------------

    outgoing_agg = {
        "outgoing_flow": ("lng_flow_cmb", "sum"),
        "out_degree": ("to_node_id", "nunique"),
    }

    if "voyage_count" in edges.columns:
        outgoing_agg["out_voyages"] = ("voyage_count", "sum")

    if "route_count" in edges.columns:
        outgoing_agg["out_route_count_sum"] = ("route_count", "sum")

    outgoing = (
        edges.groupby(["period_month", "from_node_id"])
        .agg(**outgoing_agg)
        .reset_index()
        .rename(columns={"from_node_id": "node_id"})
    )

    metrics = incoming.merge(
        outgoing,
        on=["node_id", "period_month"],
        how="outer"
    )

    # --------------------------------------------------------
    # Undirected unique degree
    # --------------------------------------------------------

    neighbor_pairs = pd.concat(
        [
            edges[
                ["period_month", "from_node_id", "to_node_id"]
            ].rename(
                columns={
                    "from_node_id": "node_id",
                    "to_node_id": "neighbor_id",
                }
            ),
            edges[
                ["period_month", "to_node_id", "from_node_id"]
            ].rename(
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
        .drop_duplicates(
            ["period_month", "node_id", "neighbor_id"]
        )
        .groupby(["period_month", "node_id"])
        .size()
        .reset_index(name="total_degree")
    )

    metrics = metrics.merge(
        total_degree,
        on=["period_month", "node_id"],
        how="outer"
    )

    # --------------------------------------------------------
    # Unique route exposure
    # --------------------------------------------------------

    if "route_ids" in edges.columns:
        route_rows = []

        for row in edges[
            ["period_month", "from_node_id", "to_node_id", "route_ids"]
        ].itertuples(index=False):

            routes = parse_route_ids(row.route_ids)

            for route_id in routes:
                route_rows.append(
                    {
                        "period_month": row.period_month,
                        "node_id": row.from_node_id,
                        "direction": "out",
                        "route_id": route_id,
                    }
                )
                route_rows.append(
                    {
                        "period_month": row.period_month,
                        "node_id": row.to_node_id,
                        "direction": "in",
                        "route_id": route_id,
                    }
                )

        if route_rows:
            route_long = pd.DataFrame(route_rows)

            in_routes = (
                route_long[
                    route_long["direction"].eq("in")
                ]
                .groupby(["period_month", "node_id"])["route_id"]
                .nunique()
                .reset_index(name="in_route_exposure")
            )

            out_routes = (
                route_long[
                    route_long["direction"].eq("out")
                ]
                .groupby(["period_month", "node_id"])["route_id"]
                .nunique()
                .reset_index(name="out_route_exposure")
            )

            total_routes = (
                route_long
                .groupby(["period_month", "node_id"])["route_id"]
                .nunique()
                .reset_index(name="total_route_exposure")
            )

            metrics = (
                metrics
                .merge(
                    in_routes,
                    on=["period_month", "node_id"],
                    how="outer"
                )
                .merge(
                    out_routes,
                    on=["period_month", "node_id"],
                    how="outer"
                )
                .merge(
                    total_routes,
                    on=["period_month", "node_id"],
                    how="outer"
                )
            )

    # Fallback when route_ids is unavailable.
    if "total_route_exposure" not in metrics.columns:
        if "route_count" in edges.columns:
            incident_routes = pd.concat(
                [
                    edges[
                        ["period_month", "from_node_id", "route_count"]
                    ].rename(columns={"from_node_id": "node_id"}),
                    edges[
                        ["period_month", "to_node_id", "route_count"]
                    ].rename(columns={"to_node_id": "node_id"}),
                ],
                ignore_index=True,
            )

            total_routes = (
                incident_routes
                .groupby(["period_month", "node_id"])["route_count"]
                .sum()
                .reset_index(name="total_route_exposure")
            )

            metrics = metrics.merge(
                total_routes,
                on=["period_month", "node_id"],
                how="outer"
            )

    return metrics


# ============================================================
# MONTHLY NETWORK CENTRALITIES
# ============================================================

def calculate_monthly_centralities(edges):
    print("\n" + "=" * 70)
    print("CALCULATING MONTHLY CENTRALITIES")
    print("=" * 70)

    rows = []

    for period_month, month_edges in edges.groupby("period_month"):
        print(f"  {period_month:%Y-%m}")

        graph = nx.DiGraph()

        for row in month_edges.itertuples(index=False):
            source = str(row.from_node_id)
            target = str(row.to_node_id)

            raw_weight = getattr(row, CENTRALITY_WEIGHT, 0)
            weight = 0.0 if pd.isna(raw_weight) else float(raw_weight)

            if graph.has_edge(source, target):
                graph[source][target]["weight"] += weight
            else:
                graph.add_edge(
                    source,
                    target,
                    weight=weight
                )

        # Strength = weighted degree using LNG flow as edge weight.
        in_strength = dict(graph.in_degree(weight="weight"))
        out_strength = dict(graph.out_degree(weight="weight"))

        # For shortest-path centrality, high LNG flow should imply
        # stronger/closer connectivity, not a longer path. Therefore
        # distance is defined as the inverse of positive flow.
        for source, target, data in graph.edges(data=True):
            weight = data.get("weight", 0.0)

            if weight > 0:
                data["distance"] = 1.0 / weight
            else:
                data["distance"] = np.inf

        try:
            betweenness = nx.betweenness_centrality(
                graph,
                weight="distance",
                normalized=True
            )
        except Exception:
            betweenness = {
                node: np.nan for node in graph.nodes
            }

        try:
            pagerank = nx.pagerank(
                graph,
                weight="weight"
            )
        except Exception:
            pagerank = {
                node: np.nan for node in graph.nodes
            }

        # Eigenvector centrality is computed on the undirected weighted
        # projection to obtain one structural score per node and avoid
        # ambiguity between left/right directed eigenvectors.
        undirected = nx.Graph()

        for source, target, data in graph.edges(data=True):
            weight = data.get("weight", 0.0)

            if undirected.has_edge(source, target):
                undirected[source][target]["weight"] += weight
            else:
                undirected.add_edge(
                    source,
                    target,
                    weight=weight
                )

        try:
            eigenvector = nx.eigenvector_centrality_numpy(
                undirected,
                weight="weight"
            )
        except Exception:
            try:
                eigenvector = nx.eigenvector_centrality(
                    undirected,
                    weight="weight",
                    max_iter=5000
                )
            except Exception:
                eigenvector = {
                    node: np.nan for node in graph.nodes
                }

        for node in graph.nodes:
            rows.append(
                {
                    "node_id": node,
                    "period_month": period_month,
                    "in_strength": in_strength.get(node, 0.0),
                    "out_strength": out_strength.get(node, 0.0),
                    "total_strength": (
                        in_strength.get(node, 0.0)
                        + out_strength.get(node, 0.0)
                    ),
                    "betweenness": betweenness.get(node, np.nan),
                    "pagerank": pagerank.get(node, np.nan),
                    "eigenvector": eigenvector.get(node, np.nan),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# FINAL NODE-MONTH TABLE
# ============================================================

def build_node_month_dataset(nodes, edges):
    panel = build_panel_index(nodes, edges)

    basic = calculate_basic_node_month_metrics(edges)
    centrality = calculate_monthly_centralities(edges)

    node_month = (
        panel
        .merge(
            basic,
            on=["node_id", "period_month"],
            how="left"
        )
        .merge(
            centrality,
            on=["node_id", "period_month"],
            how="left"
        )
    )

    # --------------------------------------------------------
    # Fill structurally meaningful inactive values with zero
    # --------------------------------------------------------

    zero_columns = [
        "incoming_flow",
        "outgoing_flow",
        "in_voyages",
        "out_voyages",
        "in_route_count_sum",
        "out_route_count_sum",
        "in_route_exposure",
        "out_route_exposure",
        "total_route_exposure",
        "in_degree",
        "out_degree",
        "total_degree",
        "in_strength",
        "out_strength",
        "total_strength",
        "betweenness",
        "pagerank",
        "eigenvector",
    ]

    for col in zero_columns:
        if col in node_month.columns:
            node_month[col] = node_month[col].fillna(0)

    # --------------------------------------------------------
    # Node type
    # --------------------------------------------------------

    if "node_type" not in node_month.columns:
        node_month["node_type"] = np.where(
            node_month["node_id"].str.startswith("CP"),
            "chokepoint",
            "terminal"
        )

    node_month["node_type"] = (
        node_month["node_type"]
        .fillna("unknown")
        .astype(str)
    )

    # --------------------------------------------------------
    # Throughput
    # --------------------------------------------------------

    node_month["incident_flow"] = (
        node_month.get("incoming_flow", 0)
        + node_month.get("outgoing_flow", 0)
    )

    is_chokepoint = (
        node_month["node_type"]
        .str.lower()
        .eq("chokepoint")
    )

    # For chokepoints the same physical cargo is represented on
    # incoming and outgoing edges, so incident flow is divided by two.
    # For terminals, total incident flow is the observed throughput.
    node_month["node_throughput"] = np.where(
        is_chokepoint,
        node_month["incident_flow"] / 2,
        node_month["incident_flow"]
    )

    node_month["flow_imbalance"] = (
        node_month.get("incoming_flow", 0)
        - node_month.get("outgoing_flow", 0)
    ).abs()

    # --------------------------------------------------------
    # Voyages
    # --------------------------------------------------------

    node_month["incident_voyage_traversals"] = (
        node_month.get("in_voyages", 0)
        + node_month.get("out_voyages", 0)
    )

    # As with flow, chokepoint incoming/outgoing traversal counts
    # represent the same voyage passing through the node.
    node_month["total_voyages"] = np.where(
        is_chokepoint,
        node_month["incident_voyage_traversals"] / 2,
        node_month["incident_voyage_traversals"]
    )

    # --------------------------------------------------------
    # Monthly network-flow share
    # --------------------------------------------------------

    monthly_throughput = (
        node_month
        .groupby("period_month")["node_throughput"]
        .transform("sum")
    )

    node_month["share_monthly_network_flow"] = np.where(
        monthly_throughput > 0,
        node_month["node_throughput"] / monthly_throughput,
        np.nan
    )

    # --------------------------------------------------------
    # Activity indicator
    # --------------------------------------------------------

    node_month["active"] = (
        (node_month["node_throughput"] > 0)
        | (node_month["total_degree"] > 0)
    ).astype(int)

    # --------------------------------------------------------
    # Months since last active observation
    # --------------------------------------------------------

    node_month = node_month.sort_values(
        ["node_id", "period_month"]
    ).reset_index(drop=True)

    months_since = []

    for _, group in node_month.groupby("node_id", sort=False):
        counter = np.nan

        for active in group["active"]:
            if active == 1:
                counter = 0
            elif pd.isna(counter):
                counter = np.nan
            else:
                counter += 1

            months_since.append(counter)

    node_month["months_since_last_active"] = months_since

    # --------------------------------------------------------
    # Internal consistency QA
    # --------------------------------------------------------

    if {"total_strength", "incident_flow"}.issubset(node_month.columns):
        node_month["qa_strength_minus_incident_flow"] = (
            node_month["total_strength"]
            - node_month["incident_flow"]
        )

    node_month.to_csv(
        DATA_OUTPUT_DIR / "node_month_metrics.csv",
        index=False
    )

    print(
        f"\nNode-month observations: {len(node_month):,}"
    )
    print(
        f"Active node-month observations: "
        f"{node_month['active'].sum():,}"
    )

    return node_month


# ============================================================
# DISTRIBUTIONS
# ============================================================

def analyze_distributions(node_month):
    print("\n" + "=" * 70)
    print("NODE-MONTH DISTRIBUTIONS")
    print("=" * 70)

    metrics = [
        metric
        for metric in CORE_METRICS
        if metric in node_month.columns
    ]

    summary_rows = []

    groups = [("all", node_month)]

    if "node_type" in node_month.columns:
        for node_type, group in node_month.groupby("node_type"):
            groups.append((str(node_type), group))

    for group_name, group in groups:
        for metric in metrics:
            series = group[metric].dropna()

            if series.empty:
                continue

            summary_rows.append(
                {
                    "node_group": group_name,
                    "metric": metric,
                    "n": len(series),
                    "mean": series.mean(),
                    "std": series.std(),
                    "min": series.min(),
                    "p01": series.quantile(0.01),
                    "p05": series.quantile(0.05),
                    "p25": series.quantile(0.25),
                    "median": series.median(),
                    "p75": series.quantile(0.75),
                    "p95": series.quantile(0.95),
                    "p99": series.quantile(0.99),
                    "max": series.max(),
                    "zero_share": (series == 0).mean(),
                }
            )

    summary = pd.DataFrame(summary_rows)

    summary.to_csv(
        DISTRIBUTIONS_DIR / "node_month_summary_statistics.csv",
        index=False
    )

    for metric in metrics:
        values = node_month[metric].dropna()

        plt.figure(figsize=(8, 5))
        plt.hist(values, bins=50)
        plt.title(f"Node-month distribution: {metric}")
        plt.xlabel(metric)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(
            DISTRIBUTIONS_DIR / f"{metric}_distribution.png",
            dpi=300
        )
        plt.close()

        positive = values[values > 0]

        if not positive.empty:
            plt.figure(figsize=(8, 5))
            plt.hist(np.log10(positive), bins=50)
            plt.title(f"Node-month distribution: {metric} - log10 scale")
            plt.xlabel(f"log10({metric})")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(
                DISTRIBUTIONS_DIR / f"{metric}_distribution_log10.png",
                dpi=300
            )
            plt.close()

    return summary


# ============================================================
# CORRELATIONS AND REDUNDANCY
# ============================================================

def plot_correlation_matrix(corr, title, output_file):
    fig, ax = plt.subplots(figsize=(10, 8))

    image = ax.imshow(
        corr.values,
        vmin=-1,
        vmax=1,
        aspect="auto"
    )

    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(
        corr.columns,
        rotation=45,
        ha="right"
    )

    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)

    ax.set_title(title)

    fig.colorbar(image, ax=ax, label="Correlation")
    fig.tight_layout()
    fig.savefig(output_file, dpi=300)
    plt.close(fig)


def analyze_correlations(node_month):
    print("\n" + "=" * 70)
    print("NODE-MONTH CORRELATIONS")
    print("=" * 70)

    metrics = [
        metric
        for metric in CORE_METRICS
        if metric in node_month.columns
    ]

    if len(metrics) < 2:
        return

    pearson = node_month[metrics].corr(method="pearson")
    spearman = node_month[metrics].corr(method="spearman")

    pearson.to_csv(
        CORRELATIONS_DIR / "pearson_correlations.csv"
    )

    spearman.to_csv(
        CORRELATIONS_DIR / "spearman_correlations.csv"
    )

    plot_correlation_matrix(
        pearson,
        "Pearson correlations across node-month observations",
        CORRELATIONS_DIR / "pearson_correlation_heatmap.png"
    )

    plot_correlation_matrix(
        spearman,
        "Spearman correlations across node-month observations",
        CORRELATIONS_DIR / "spearman_correlation_heatmap.png"
    )

    # Correlations separately by node type.
    if "node_type" in node_month.columns:
        for node_type, group in node_month.groupby("node_type"):
            safe_name = (
                str(node_type)
                .lower()
                .replace(" ", "_")
            )

            if len(group) < 3:
                continue

            group[metrics].corr(
                method="pearson"
            ).to_csv(
                CORRELATIONS_DIR /
                f"pearson_correlations_{safe_name}.csv"
            )

            group[metrics].corr(
                method="spearman"
            ).to_csv(
                CORRELATIONS_DIR /
                f"spearman_correlations_{safe_name}.csv"
            )

    # --------------------------------------------------------
    # Month-specific correlation stability
    # --------------------------------------------------------

    monthly_rows = []

    for period_month, group in node_month.groupby("period_month"):
        if len(group) < 3:
            continue

        for i, metric_a in enumerate(metrics):
            for metric_b in metrics[i + 1:]:
                monthly_rows.append(
                    {
                        "period_month": period_month,
                        "metric_a": metric_a,
                        "metric_b": metric_b,
                        "spearman_correlation": safe_spearman(
                            group[metric_a],
                            group[metric_b]
                        ),
                    }
                )

    monthly_corr = pd.DataFrame(monthly_rows)

    monthly_corr.to_csv(
        CORRELATIONS_DIR /
        "monthly_spearman_correlation_stability.csv",
        index=False
    )

    if not monthly_corr.empty:
        correlation_stability = (
            monthly_corr
            .groupby(["metric_a", "metric_b"])
            ["spearman_correlation"]
            .agg(
                mean_correlation="mean",
                std_correlation="std",
                min_correlation="min",
                max_correlation="max",
                months_observed="count",
            )
            .reset_index()
        )

        correlation_stability.to_csv(
            CORRELATIONS_DIR /
            "correlation_stability_summary.csv",
            index=False
        )


# ============================================================
# TEMPORAL STABILITY
# ============================================================

def analyze_temporal_stability(node_month):
    print("\n" + "=" * 70)
    print("TEMPORAL STABILITY")
    print("=" * 70)

    metrics = [
        metric
        for metric in CORE_METRICS
        if metric in node_month.columns
    ]

    rows = []

    for node_id, group in node_month.groupby("node_id"):
        group = group.sort_values("period_month")

        node_type = (
            group["node_type"].iloc[0]
            if "node_type" in group.columns
            else "unknown"
        )

        for metric in metrics:
            series = group[metric]

            rows.append(
                {
                    "node_id": node_id,
                    "node_type": node_type,
                    "metric": metric,
                    "months": len(series),
                    "active_months": int(group["active"].sum()),
                    "mean": series.mean(),
                    "std": series.std(),
                    "coefficient_of_variation": safe_cv(series),
                    "min": series.min(),
                    "max": series.max(),
                    "lag1_autocorrelation": lag1_autocorrelation(series),
                    "mean_absolute_monthly_change": (
                        series.diff().abs().mean()
                    ),
                }
            )

    stability = pd.DataFrame(rows)

    stability.to_csv(
        DATA_OUTPUT_DIR / "node_month_temporal_stability.csv",
        index=False
    )

    # --------------------------------------------------------
    # Cross-sectional month-to-month persistence by metric
    # --------------------------------------------------------

    persistence_rows = []

    months = sorted(node_month["period_month"].unique())

    for metric in metrics:
        for previous_month, current_month in zip(months[:-1], months[1:]):
            previous = (
                node_month[
                    node_month["period_month"].eq(previous_month)
                ][["node_id", metric]]
                .rename(columns={metric: "previous_value"})
            )

            current = (
                node_month[
                    node_month["period_month"].eq(current_month)
                ][["node_id", metric]]
                .rename(columns={metric: "current_value"})
            )

            paired = previous.merge(
                current,
                on="node_id",
                how="inner"
            )

            persistence_rows.append(
                {
                    "metric": metric,
                    "previous_month": previous_month,
                    "current_month": current_month,
                    "spearman_month_to_month": safe_spearman(
                        paired["previous_value"],
                        paired["current_value"]
                    ),
                }
            )

    persistence = pd.DataFrame(persistence_rows)

    persistence.to_csv(
        TEMPORAL_DIR / "month_to_month_metric_persistence.csv",
        index=False
    )

    # --------------------------------------------------------
    # Aggregate monthly means by node type
    # --------------------------------------------------------

    if "node_type" in node_month.columns:
        for metric in metrics:
            monthly_type = (
                node_month
                .groupby(["period_month", "node_type"])[metric]
                .mean()
                .unstack("node_type")
            )

            if monthly_type.empty:
                continue

            plt.figure(figsize=(10, 5))

            for column in monthly_type.columns:
                plt.plot(
                    monthly_type.index,
                    monthly_type[column],
                    marker="o",
                    markersize=2,
                    label=str(column)
                )

            plt.title(
                f"Mean monthly {metric} by node type"
            )
            plt.xlabel("Month")
            plt.ylabel(metric)
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                TEMPORAL_DIR /
                f"{metric}_monthly_mean_by_node_type.png",
                dpi=300
            )
            plt.close()

    return stability


# ============================================================
# MONTHLY RANKS AND RANK STABILITY
# ============================================================

def analyze_rank_stability(node_month):
    print("\n" + "=" * 70)
    print("MONTHLY RANK STABILITY")
    print("=" * 70)

    metrics = [
        metric
        for metric in CORE_METRICS
        if metric in node_month.columns
    ]

    rank_long = []

    for period_month, group in node_month.groupby("period_month"):
        for metric in metrics:
            temp = group[
                ["node_id", "node_type", metric]
            ].copy()

            temp["rank"] = temp[metric].rank(
                method="min",
                ascending=False
            )

            temp["percentile_rank"] = temp[metric].rank(
                method="average",
                pct=True,
                ascending=True
            )

            n_nodes = len(temp)

            temp["top_5pct"] = (
                temp["rank"] <= max(1, int(np.ceil(n_nodes * 0.05)))
            )

            temp["top_10pct"] = (
                temp["rank"] <= max(1, int(np.ceil(n_nodes * 0.10)))
            )

            temp["top_20pct"] = (
                temp["rank"] <= max(1, int(np.ceil(n_nodes * 0.20)))
            )

            temp["period_month"] = period_month
            temp["metric"] = metric
            temp = temp.rename(columns={metric: "metric_value"})

            rank_long.append(temp)

    ranks = pd.concat(rank_long, ignore_index=True)

    ranks.to_csv(
        RANKINGS_DIR / "monthly_node_ranks.csv",
        index=False
    )

    rank_stability = (
        ranks
        .groupby(["node_id", "node_type", "metric"])
        .agg(
            mean_rank=("rank", "mean"),
            median_rank=("rank", "median"),
            best_rank=("rank", "min"),
            worst_rank=("rank", "max"),
            rank_std=("rank", "std"),
            mean_percentile_rank=("percentile_rank", "mean"),
            top_5pct_months=("top_5pct", "sum"),
            top_10pct_months=("top_10pct", "sum"),
            top_20pct_months=("top_20pct", "sum"),
            months_observed=("period_month", "nunique"),
        )
        .reset_index()
    )

    rank_stability["top_5pct_share"] = (
        rank_stability["top_5pct_months"]
        / rank_stability["months_observed"]
    )

    rank_stability["top_10pct_share"] = (
        rank_stability["top_10pct_months"]
        / rank_stability["months_observed"]
    )

    rank_stability["top_20pct_share"] = (
        rank_stability["top_20pct_months"]
        / rank_stability["months_observed"]
    )

    rank_stability.to_csv(
        DATA_OUTPUT_DIR / "node_month_rank_stability.csv",
        index=False
    )

    # --------------------------------------------------------
    # Rank correlation between consecutive months
    # --------------------------------------------------------

    rank_persistence_rows = []

    for metric in metrics:
        metric_ranks = ranks[
            ranks["metric"].eq(metric)
        ]

        months = sorted(metric_ranks["period_month"].unique())

        for previous_month, current_month in zip(months[:-1], months[1:]):
            previous = (
                metric_ranks[
                    metric_ranks["period_month"].eq(previous_month)
                ][["node_id", "rank"]]
                .rename(columns={"rank": "previous_rank"})
            )

            current = (
                metric_ranks[
                    metric_ranks["period_month"].eq(current_month)
                ][["node_id", "rank"]]
                .rename(columns={"rank": "current_rank"})
            )

            paired = previous.merge(
                current,
                on="node_id",
                how="inner"
            )

            rank_persistence_rows.append(
                {
                    "metric": metric,
                    "previous_month": previous_month,
                    "current_month": current_month,
                    "spearman_rank_correlation": safe_spearman(
                        paired["previous_rank"],
                        paired["current_rank"]
                    ),
                }
            )

    rank_persistence = pd.DataFrame(rank_persistence_rows)

    rank_persistence.to_csv(
        RANKINGS_DIR / "month_to_month_rank_persistence.csv",
        index=False
    )

    return ranks, rank_stability


# ============================================================
# TOP-NODE TIME SERIES
# ============================================================

def plot_top_node_timeseries(node_month, top_n=10):
    """
    Plot the nodes with the highest mean value over the full period.
    Separate plots are produced for terminals and chokepoints.
    """
    metrics = [
        metric
        for metric in CORE_METRICS
        if metric in node_month.columns
    ]

    if "node_type" not in node_month.columns:
        return

    for metric in metrics:
        for node_type, type_group in node_month.groupby("node_type"):
            ranking = (
                type_group
                .groupby("node_id")[metric]
                .mean()
                .sort_values(ascending=False)
            )

            top_nodes = ranking.head(top_n).index

            plot_data = type_group[
                type_group["node_id"].isin(top_nodes)
            ]

            if plot_data.empty:
                continue

            plt.figure(figsize=(11, 6))

            for node_id, node_group in plot_data.groupby("node_id"):
                node_group = node_group.sort_values("period_month")

                plt.plot(
                    node_group["period_month"],
                    node_group[metric],
                    label=node_id
                )

            plt.title(
                f"{metric} over time - top {top_n} {node_type} nodes"
            )
            plt.xlabel("Month")
            plt.ylabel(metric)
            plt.legend(
                fontsize=7,
                ncol=2
            )
            plt.tight_layout()

            safe_type = (
                str(node_type)
                .lower()
                .replace(" ", "_")
            )

            plt.savefig(
                TEMPORAL_DIR /
                f"{metric}_top_{top_n}_{safe_type}_timeseries.png",
                dpi=300
            )

            plt.close()


# ============================================================
# QA SUMMARY
# ============================================================

def save_qa_summary(node_month):
    qa_rows = []

    qa_rows.append(
        {
            "check": "node_month_rows",
            "value": len(node_month),
        }
    )

    qa_rows.append(
        {
            "check": "unique_nodes",
            "value": node_month["node_id"].nunique(),
        }
    )

    qa_rows.append(
        {
            "check": "unique_months",
            "value": node_month["period_month"].nunique(),
        }
    )

    qa_rows.append(
        {
            "check": "active_node_months",
            "value": int(node_month["active"].sum()),
        }
    )

    qa_rows.append(
        {
            "check": "inactive_node_months",
            "value": int((node_month["active"] == 0).sum()),
        }
    )

    qa_rows.append(
        {
            "check": "duplicate_node_month_keys",
            "value": int(
                node_month.duplicated(
                    ["node_id", "period_month"]
                ).sum()
            ),
        }
    )

    if "qa_strength_minus_incident_flow" in node_month.columns:
        qa_rows.append(
            {
                "check": "max_abs_strength_minus_incident_flow",
                "value": node_month[
                    "qa_strength_minus_incident_flow"
                ].abs().max(),
            }
        )

    pd.DataFrame(qa_rows).to_csv(
        DATA_OUTPUT_DIR / "node_month_qa_summary.csv",
        index=False
    )


# ============================================================
# MAIN
# ============================================================

def main():
    nodes, edges = load_data()

    node_month = build_node_month_dataset(
        nodes,
        edges
    )

    save_qa_summary(node_month)

    analyze_distributions(node_month)

    analyze_correlations(node_month)

    analyze_temporal_stability(node_month)

    analyze_rank_stability(node_month)

    plot_top_node_timeseries(node_month)

    print("\n" + "=" * 70)
    print("NODE-MONTH EDA COMPLETED")
    print("=" * 70)

    print(f"\nResults saved in:\n{OUTPUT_DIR}")

    print(
        "\nMain panel dataset:\n"
        f"{DATA_OUTPUT_DIR / 'node_month_metrics.csv'}"
    )


if __name__ == "__main__":
    main()
