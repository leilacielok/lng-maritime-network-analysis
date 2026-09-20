from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx


# ============================================================
# PATHS AND SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "eda_outputs" / "terminal_month"

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


MATCHED_VOYAGES_FILE = DATA_DIR / "LNG_voyage_node_matching.xlsx"
NODES_FILE = DATA_DIR / "LNG_multilayer_nodes_observed.csv"

# Month assignment is based on voyage departure
DATE_COLUMN = "start_date"
PAGERANK_ALPHA = 0.85
TOP_N = 20


METRICS = [
    "throughput",
    "voyage_count",
    "counterparty_count",
    "counterparty_hhi_terminal",
    "counterparty_hhi_country",
    "effective_counterparties_terminal",
    "effective_counterparties_country",
    "max_dependence_generated",
    "weighted_dependence_generated",
    "country_terminal_dependence",
    "import_pagerank",
    "export_pagerank",
    "role_specific_pagerank",
    "role_specific_dependence",
]


# ============================================================
# HELPERS
# ============================================================

def safe_divide(numerator, denominator):
    """Vectorised division that returns NaN when the denominator is zero."""
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def hhi(values):
    """Herfindahl-Hirschman concentration index for non-negative flows."""
    values = pd.to_numeric(values, errors="coerce").fillna(0)
    total = values.sum()
    if total <= 0:
        return np.nan
    shares = values / total
    return float((shares**2).sum())


def safe_spearman(x, y):
    valid = pd.concat([x, y], axis=1).dropna()
    if len(valid) < 3:
        return np.nan
    if valid.iloc[:, 0].nunique() <= 1 or valid.iloc[:, 1].nunique() <= 1:
        return np.nan
    return valid.iloc[:, 0].corr(valid.iloc[:, 1], method="spearman")


def lag1_autocorrelation(series):
    series = series.astype(float)
    paired = pd.concat(
        [series.shift(1).rename("previous"), series.rename("current")], axis=1
    ).dropna()
    if len(paired) < 3:
        return np.nan
    if paired["previous"].nunique() <= 1 or paired["current"].nunique() <= 1:
        return np.nan
    return paired["previous"].corr(paired["current"], method="pearson")

def write_csv(frame, path, index=False):
    frame.to_csv(path, index=index, float_format="%.12g")

def flow_pattern(outgoing_flow, incoming_flow):
    if outgoing_flow <= 0 and incoming_flow <= 0:
        return "inactive"
    if outgoing_flow > 0 and incoming_flow <= 0:
        return "exporter"
    if incoming_flow > 0 and outgoing_flow <= 0:
        return "importer"
    return "bidirectional"

def net_trade_position(outgoing_flow, incoming_flow):
    if outgoing_flow <= 0 and incoming_flow <= 0:
        return "inactive"
    if np.isclose(outgoing_flow, incoming_flow):
        return "balanced"
    if outgoing_flow > incoming_flow:
        return "net_exporter"
    return "net_importer"
# ============================================================
# LOAD AND VALIDATE DATA
# ============================================================

def load_data():

    voyages = pd.read_excel(MATCHED_VOYAGES_FILE, sheet_name="Matched Voyages")
    nodes = pd.read_csv(NODES_FILE)

    required_voyages = {
        DATE_COLUMN,
        "start_date",
        "end_date",
        "IMO",
        "voyage",
        "amount_cmb",
        "from_node_id",
        "to_node_id",
        "from_terminal",
        "to_terminal",
        "from_country",
        "to_country",
    }
    required_nodes = {
        "node_id",
        "node_name",
        "node_type",
        "country",
        "region",
        "infrastructure_type",
        "capacity_mtpa",
        "unit_count",
        "start_year",
    }

    missing_voyages = required_voyages.difference(voyages.columns)
    missing_nodes = required_nodes.difference(nodes.columns)
    
    if missing_voyages:
        raise ValueError(
            "Matched Voyages is missing: " + ", ".join(sorted(missing_voyages))
        )
    if missing_nodes:
        raise ValueError("Nodes file is missing: " + ", ".join(sorted(missing_nodes)))

    raw_rows = len(voyages)
    
    voyages["start_date"] = pd.to_datetime(
        voyages["start_date"],
        errors="coerce",
    )

    voyages["end_date"] = pd.to_datetime(
        voyages["end_date"],
        errors="coerce",
    )

    voyages["amount_cmb"] = pd.to_numeric(
        voyages["amount_cmb"],
        errors="coerce",
    )

    # Return voyages contain zero cargo and must not enter the trade network.
    voyages = voyages.loc[
        voyages["voyage"].astype(str).str.lower().eq("export")
        & voyages[DATE_COLUMN].notna()
        & voyages["from_node_id"].notna()
        & voyages["to_node_id"].notna()
        & voyages["amount_cmb"].gt(0)
    ].copy()

    # Identify and remove duplicated export-voyage observations.
    duplicate_columns = [
        "start_date",
        "end_date",
        "IMO",
        "voyage",
        "from_terminal",
        "to_terminal",
        "amount_cmb",
    ]

    duplicate_mask = voyages.duplicated(
        subset=duplicate_columns,
        keep=False,
    )

    possible_duplicate_rows = int(duplicate_mask.sum())
    rows_before_deduplication = len(voyages)

    voyages = voyages.drop_duplicates(
        subset=duplicate_columns,
        keep="first",
    ).copy()

    duplicates_removed = rows_before_deduplication - len(voyages)
    
    voyages["from_node_id"] = voyages["from_node_id"].astype(str)
    voyages["to_node_id"] = voyages["to_node_id"].astype(str)
    voyages["period_month"] = voyages[DATE_COLUMN].dt.to_period("M").dt.to_timestamp()

    nodes = nodes.loc[nodes["node_type"].eq("terminal")].copy()
    nodes["node_id"] = nodes["node_id"].astype(str)
    
    observed_ids = set(voyages["from_node_id"]) | set(voyages["to_node_id"])
    nodes = nodes.loc[nodes["node_id"].isin(observed_ids)].drop_duplicates("node_id")

    missing_metadata = observed_ids.difference(nodes["node_id"])
    if missing_metadata:
        raise ValueError(
            "Some voyage terminal IDs are absent from the nodes file: "
            + ", ".join(sorted(missing_metadata)[:20])
        )

    qa = pd.DataFrame(
        {
            "check": [
                "raw_workbook_rows",
                "positive_export_rows_before_deduplication",
                "possible_duplicate_export_rows",
                "duplicate_export_rows_removed",
                "positive_export_rows_used",
                "observed_terminals",
                "months",
                "total_lng_volume_cmb",
            ],
            "value": [
                raw_rows,
                rows_before_deduplication,
                possible_duplicate_rows,
                duplicates_removed,
                len(voyages),
                len(observed_ids),
                voyages["period_month"].nunique(),
                voyages["amount_cmb"].sum(),
            ],
        }
    )
    write_csv(qa, DATA_OUTPUT_DIR / "input_qa_summary.csv")

    return voyages, nodes


# ============================================================
# TERMINAL-MONTH METRICS
# ============================================================

def build_complete_panel(voyages, nodes):
    months = pd.date_range(
        voyages["period_month"].min(),
        voyages["period_month"].max(),
        freq="MS",
    )
    
    panel = (
        pd.MultiIndex.from_product(
            [nodes["node_id"].sort_values(), months],
            names=["terminal_id", "period_month"],
        )
        .to_frame(index=False)
    )

    metadata_columns = [
        "node_id",
        "node_name",
        "country",
        "latitude",
        "longitude",
        "infrastructure_type",
        "capacity_mtpa",
        "unit_count",
        "start_year",
    ]

    terminal_metadata = (
        nodes[metadata_columns]
        .drop_duplicates(subset=["node_id"])
        .rename(columns={"node_id": "terminal_id"})
    )

    panel = panel.merge(
        terminal_metadata,
        on="terminal_id",
        how="left",
        validate="many_to_one",
    )
    
    panel["terminal_age"] = (
        panel["period_month"].dt.year - panel["start_year"]
    )

    panel.loc[
        panel["start_year"].isna()
        | (panel["terminal_age"] < 0),
        "terminal_age",
    ] = np.nan
    
    return panel

def calculate_basic_activity(voyages):
    outgoing = (
        voyages.groupby(["period_month", "from_node_id"], as_index=False)
        .agg(outgoing_flow=("amount_cmb", "sum"), outgoing_voyages=("amount_cmb", "size"))
        .rename(columns={"from_node_id": "terminal_id"})
    )
    incoming = (
        voyages.groupby(["period_month", "to_node_id"], as_index=False)
        .agg(incoming_flow=("amount_cmb", "sum"), incoming_voyages=("amount_cmb", "size"))
        .rename(columns={"to_node_id": "terminal_id"})
    )
    return outgoing, incoming


def build_country_role_tables(voyages):
    """Classify countries from their own incoming and outgoing LNG flows.

    The monthly table preserves time variation. The full-period table provides
    one structural classification based on all observed voyages.
    """
    country_voyages = voyages.loc[
        voyages["from_country"].notna()
        & voyages["to_country"].notna()
        & voyages["from_country"].ne(voyages["to_country"])
    ].copy()
    
    country_values = pd.concat(
        [
            country_voyages["from_country"],
            country_voyages["to_country"],
        ],
        ignore_index=True,
    ).dropna()
    
    countries = sorted(country_values.astype(str).str.strip().unique())
    
    months = pd.date_range(
        country_voyages["period_month"].min(),
        country_voyages["period_month"].max(),
        freq="MS",
    )
    
    panel = pd.MultiIndex.from_product(
        [countries, months], names=["country", "period_month"]
    ).to_frame(index=False)

    outgoing = (
        country_voyages.groupby(
            ["period_month", "from_country"],
            as_index=False,
        )
        .agg(
            outgoing_flow=("amount_cmb", "sum"),
            outgoing_voyages=("amount_cmb", "size"),
            active_export_terminals=("from_node_id", "nunique"),
            destination_countries=("to_country", "nunique"),
        )
        .rename(columns={"from_country": "country"})
    )
    incoming = (
        country_voyages.groupby(
            ["period_month", "to_country"],
            as_index=False,
        )
        .agg(
            incoming_flow=("amount_cmb", "sum"),
            incoming_voyages=("amount_cmb", "size"),
            active_import_terminals=("to_node_id", "nunique"),
            origin_countries=("from_country", "nunique"),
        )
        .rename(columns={"to_country": "country"})
    )
    
    panel = panel.merge(
        outgoing, on=["country", "period_month"], how="left"
    ).merge(incoming, on=["country", "period_month"], how="left")

    count_columns = [
        "outgoing_voyages",
        "incoming_voyages",
        "active_export_terminals",
        "active_import_terminals",
        "destination_countries",
        "origin_countries",
    ]
    flow_columns = ["outgoing_flow", "incoming_flow"]
    panel[flow_columns + count_columns] = panel[flow_columns + count_columns].fillna(0)
    panel["throughput"] = panel["outgoing_flow"] + panel["incoming_flow"]
    panel["voyage_count"] = panel["outgoing_voyages"] + panel["incoming_voyages"]
    panel["export_share"] = safe_divide(panel["outgoing_flow"], panel["throughput"])
    panel["country_flow_pattern"] = [
        flow_pattern(outgoing, incoming)
        for outgoing, incoming in zip(
            panel["outgoing_flow"],
            panel["incoming_flow"],
        )
    ]

    panel["country_role"] = [
        net_trade_position(outgoing, incoming)
        for outgoing, incoming in zip(
            panel["outgoing_flow"],
            panel["incoming_flow"],
        )
    ]

    panel["normalized_trade_balance"] = safe_divide(
        panel["outgoing_flow"] - panel["incoming_flow"],
        panel["throughput"],
    )
    panel["active"] = panel["throughput"].gt(0).astype(int)
    panel["year"] = panel["period_month"].dt.year
    panel["month"] = panel["period_month"].dt.month

    full_period = (
        panel.groupby("country", as_index=False)
        .agg(
            outgoing_flow=("outgoing_flow", "sum"),
            incoming_flow=("incoming_flow", "sum"),
            outgoing_voyages=("outgoing_voyages", "sum"),
            incoming_voyages=("incoming_voyages", "sum"),
            active_months=("active", "sum"),
        )
    )
    full_period["throughput"] = (
        full_period["outgoing_flow"] + full_period["incoming_flow"]
    )
    full_period["voyage_count"] = (
        full_period["outgoing_voyages"] + full_period["incoming_voyages"]
    )
    full_period["export_share"] = safe_divide(
        full_period["outgoing_flow"], full_period["throughput"]
    )
    
    full_period["country_flow_pattern"] = [
        flow_pattern(outgoing, incoming)
        for outgoing, incoming in zip(
            full_period["outgoing_flow"],
            full_period["incoming_flow"],
        )
    ]
    
    full_period["country_role"] = [
        net_trade_position(outgoing, incoming)
        for outgoing, incoming in zip(
            full_period["outgoing_flow"],
            full_period["incoming_flow"],
        )
    ]
    
    full_period["normalized_trade_balance"] = safe_divide(
        full_period["outgoing_flow"] - full_period["incoming_flow"],
        full_period["throughput"],
    )
    
    return (
        panel.sort_values(["period_month", "country"]).reset_index(drop=True),
        full_period.sort_values(["country"]).reset_index(drop=True),
    )


def calculate_concentration(voyages):
    """Calculate role-aware counterparty HHI at terminal and country levels."""
    out_terminal = (
        voyages.groupby(["period_month", "from_node_id", "to_node_id"], as_index=False)["amount_cmb"]
        .sum()
        .groupby(["period_month", "from_node_id"])
        .agg(out_hhi_terminal=("amount_cmb", hhi), out_counterparties=("to_node_id", "nunique"))
        .reset_index()
        .rename(columns={"from_node_id": "terminal_id"})
    )
    in_terminal = (
        voyages.groupby(["period_month", "to_node_id", "from_node_id"], as_index=False)["amount_cmb"]
        .sum()
        .groupby(["period_month", "to_node_id"])
        .agg(in_hhi_terminal=("amount_cmb", hhi), in_counterparties=("from_node_id", "nunique"))
        .reset_index()
        .rename(columns={"to_node_id": "terminal_id"})
    )
    out_country = (
        voyages.groupby(["period_month", "from_node_id", "to_country"], as_index=False)["amount_cmb"]
        .sum()
        .groupby(["period_month", "from_node_id"])
        .agg(out_hhi_country=("amount_cmb", hhi), out_counterparty_countries=("to_country", "nunique"))
        .reset_index()
        .rename(columns={"from_node_id": "terminal_id"})
    )
    in_country = (
        voyages.groupby(["period_month", "to_node_id", "from_country"], as_index=False)["amount_cmb"]
        .sum()
        .groupby(["period_month", "to_node_id"])
        .agg(in_hhi_country=("amount_cmb", hhi), in_counterparty_countries=("from_country", "nunique"))
        .reset_index()
        .rename(columns={"to_node_id": "terminal_id"})
    )
    result = out_terminal
    for frame in [in_terminal, out_country, in_country]:
        result = result.merge(frame, on=["terminal_id", "period_month"], how="outer")
    return result


def calculate_dependence(voyages):
    """Calculate exporter-generated dependence and terminal shares of national LNG receipts."""
    
    od = voyages.groupby(
        ["period_month", "from_node_id", "to_node_id"], as_index=False
    )["amount_cmb"].sum()
    recipient_imports = od.groupby(["period_month", "to_node_id"])["amount_cmb"].transform("sum")
    exporter_total = od.groupby(["period_month", "from_node_id"])["amount_cmb"].transform("sum")
    od["recipient_dependence"] = od["amount_cmb"] / recipient_imports
    od["exporter_flow_share"] = od["amount_cmb"] / exporter_total
    od["weighted_component"] = od["recipient_dependence"] * od["exporter_flow_share"]

    export_dep = (
        od.groupby(["period_month", "from_node_id"], as_index=False)
        .agg(
            max_dependence_generated=("recipient_dependence", "max"),
            weighted_dependence_generated=("weighted_component", "sum"),
        )
        .rename(columns={"from_node_id": "terminal_id"})
    )

    country_receipts = voyages.groupby(
        ["period_month", "to_country", "to_node_id"],
        as_index=False,
    )["amount_cmb"].sum()

    total_country_receipts = country_receipts.groupby(
        ["period_month", "to_country"]
    )["amount_cmb"].transform("sum")

    country_receipts["country_terminal_dependence"] = (
        country_receipts["amount_cmb"] / total_country_receipts
    )
    
    receipt_dep = country_receipts.rename(
        columns={"to_node_id": "terminal_id", "to_country": "recipient_country"}
    )[["period_month", "terminal_id", "recipient_country", "country_terminal_dependence"]]
    return export_dep, receipt_dep


def calculate_directional_pagerank(voyages, all_terminal_ids):
    records = []
    for month, month_data in voyages.groupby("period_month", sort=True):
        edge_data = month_data.groupby(
            ["from_node_id", "to_node_id"], as_index=False
        )["amount_cmb"].sum()
        graph = nx.DiGraph()
        # PageRank is computed on the active monthly network. Inactive terminals
        # are added to the completed output panel afterwards with a value of 0.
        active_ids = set(edge_data["from_node_id"]) | set(edge_data["to_node_id"])
        graph.add_nodes_from(active_ids)
        graph.add_weighted_edges_from(
            edge_data[["from_node_id", "to_node_id", "amount_cmb"]].itertuples(
                index=False, name=None
            )
        )
        import_pr = nx.pagerank(graph, alpha=PAGERANK_ALPHA, weight="weight")
        export_pr = nx.pagerank(graph.reverse(copy=True), alpha=PAGERANK_ALPHA, weight="weight")
        for terminal_id in all_terminal_ids:
            records.append(
                {
                    "period_month": month,
                    "terminal_id": terminal_id,
                    "import_pagerank": import_pr.get(terminal_id, 0.0),
                    "export_pagerank": export_pr.get(terminal_id, 0.0),
                }
            )
    return pd.DataFrame(records)

def calculate_voyage_distance_metrics(voyages):
    """
    Calculate the arithmetic mean voyage distance for each terminal-month.
    """

    distance_data = voyages[
        [
            "period_month",
            "from_node_id",
            "to_node_id",
            "voyage_distance",
        ]
    ].copy()

    distance_data["voyage_distance"] = pd.to_numeric(
        distance_data["voyage_distance"],
        errors="coerce",
    )

    distance_data = distance_data.loc[
        distance_data["period_month"].notna()
        & distance_data["voyage_distance"].gt(0)
    ]

    # Mean distance of voyages departing from each terminal
    outgoing_distance = (
        distance_data.dropna(subset=["from_node_id"])
        .groupby(
            ["period_month", "from_node_id"],
            as_index=False,
        )
        .agg(
            mean_outgoing_voyage_distance=(
                "voyage_distance",
                "mean",
            )
        )
        .rename(columns={"from_node_id": "terminal_id"})
    )

    # Mean distance of voyages arriving at each terminal
    incoming_distance = (
        distance_data.dropna(subset=["to_node_id"])
        .groupby(
            ["period_month", "to_node_id"],
            as_index=False,
        )
        .agg(
            mean_incoming_voyage_distance=(
                "voyage_distance",
                "mean",
            )
        )
        .rename(columns={"to_node_id": "terminal_id"})
    )

    return outgoing_distance.merge(
        incoming_distance,
        on=["terminal_id", "period_month"],
        how="outer",
        validate="one_to_one",
    )

def assemble_metrics(voyages, nodes):
    panel = build_complete_panel(voyages, nodes)

    outgoing_activity, incoming_activity = calculate_basic_activity(voyages)
    distance_metrics = calculate_voyage_distance_metrics(voyages)
    concentration = calculate_concentration(voyages)
    export_dep, receipt_dep = calculate_dependence(voyages)
    pagerank = calculate_directional_pagerank(
        voyages,
        nodes["node_id"].tolist(),
    )
    
    metric_frames = [
        outgoing_activity,
        incoming_activity,
        distance_metrics,
        concentration,
        export_dep,
        receipt_dep,
        pagerank,
    ]
    
    for frame in metric_frames:
        panel = panel.merge(frame, on=["terminal_id", "period_month"], how="left", validate="one_to_one")
    
    activity_columns = [
    "outgoing_flow",
    "incoming_flow",
    "outgoing_voyages",
    "incoming_voyages",
]

    panel[activity_columns] = panel[activity_columns].fillna(0)

    panel["throughput"] = (
        panel["outgoing_flow"]
        + panel["incoming_flow"]
    )

    panel["voyage_count"] = (
        panel["outgoing_voyages"]
        + panel["incoming_voyages"]
    )

    panel["export_share"] = safe_divide(
        panel["outgoing_flow"],
        panel["throughput"],
    )

    panel["terminal_role"] = [
        flow_pattern(out_flow, in_flow)
        for out_flow, in_flow in zip(
            panel["outgoing_flow"],
            panel["incoming_flow"],
        )
    ]

    panel["active"] = panel["throughput"].gt(0).astype(int)

    exporter = panel["terminal_role"].eq("exporter")
    importer = panel["terminal_role"].eq("importer")
    bidirectional = panel["terminal_role"].eq("bidirectional")

    panel["mean_voyage_distance"] = np.select(
        [exporter, importer],
        [
            panel["mean_outgoing_voyage_distance"],
            panel["mean_incoming_voyage_distance"],
        ],
        default=np.nan,
    )
    
    # For bidirectional terminals, combine inbound and outbound measures using the corresponding flow shares.
    out_share = panel["export_share"].fillna(0)
    in_share = 1 - out_share
    panel["counterparty_hhi_terminal"] = np.select(
        [exporter, importer, bidirectional],
        [
            panel["out_hhi_terminal"],
            panel["in_hhi_terminal"],
            out_share * panel["out_hhi_terminal"].fillna(0)
            + in_share * panel["in_hhi_terminal"].fillna(0),
        ],
        default=np.nan,
    )
    panel["counterparty_hhi_country"] = np.select(
        [exporter, importer, bidirectional],
        [
            panel["out_hhi_country"],
            panel["in_hhi_country"],
            out_share * panel["out_hhi_country"].fillna(0)
            + in_share * panel["in_hhi_country"].fillna(0),
        ],
        default=np.nan,
    )
    panel["counterparty_count"] = np.select(
        [exporter, importer, bidirectional],
        [
            panel["out_counterparties"],
            panel["in_counterparties"],
            panel["out_counterparties"].fillna(0) + panel["in_counterparties"].fillna(0),
        ],
        default=0,
    )
    panel["counterparty_country_count"] = np.select(
        [exporter, importer, bidirectional],
        [
            panel["out_counterparty_countries"],
            panel["in_counterparty_countries"],
            panel["out_counterparty_countries"].fillna(0)
            + panel["in_counterparty_countries"].fillna(0),
        ],
        default=0,
    )
    panel["effective_counterparties_terminal"] = 1 / panel["counterparty_hhi_terminal"]
    panel["effective_counterparties_country"] = 1 / panel["counterparty_hhi_country"]
    panel["role_specific_pagerank"] = np.select(
        [exporter, importer, bidirectional],
        [
            panel["export_pagerank"],
            panel["import_pagerank"],
            out_share * panel["export_pagerank"] + in_share * panel["import_pagerank"],
        ],
        default=np.nan,
    )
    panel["role_specific_dependence"] = np.select(
        [exporter, importer, bidirectional],
        [
            panel["weighted_dependence_generated"],
            panel["country_terminal_dependence"],
            out_share * panel["weighted_dependence_generated"].fillna(0)
            + in_share * panel["country_terminal_dependence"].fillna(0),
        ],
        default=np.nan,
    )
    panel["year"] = panel["period_month"].dt.year
    panel["month"] = panel["period_month"].dt.month
    return panel.sort_values(["period_month", "terminal_id"]).reset_index(drop=True)


# ============================================================
# EDA
# ============================================================

def make_summary(panel):
    active = panel.loc[panel["active"].eq(1)].copy()
    summary = active[METRICS + ["role_specific_dependence"]].describe(
        percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
    ).T
    summary.insert(0, "missing", active[summary.index].isna().sum())
    summary.insert(1, "zero", active[summary.index].eq(0).sum())
    write_csv(summary.reset_index(names="metric"), DATA_OUTPUT_DIR / "metric_summary_active.csv")

    roles = panel.groupby(["period_month", "terminal_role"], as_index=False).agg(
        entities=("terminal_id", "nunique"),
        total_throughput=("throughput", "sum"),
    )
    roles = (
        roles
        .rename(columns={"terminal_role": "role"})
        .assign(entity_level="terminal")
    )

    return roles[
        [
            "period_month",
            "entity_level",
            "role",
            "entities",
            "total_throughput",
        ]
    ]


def make_correlations(panel):
    rows = []
    for role in ["exporter", "importer", "bidirectional"]:
        subset = panel.loc[panel["terminal_role"].eq(role)]
        for metric in METRICS:
            rows.append(
                {
                    "terminal_role": role,
                    "metric": metric,
                    "spearman_with_throughput": safe_spearman(
                        subset[metric], subset["throughput"]
                    ),
                    "observations": subset[[metric, "throughput"]].dropna().shape[0],
                }
            )
    write_csv(pd.DataFrame(rows), CORRELATIONS_DIR / "correlation_with_throughput_by_role.csv")

    # Terminal-level correlations with capacity

    terminal_metadata = (
        panel.groupby(
            "terminal_id",
            as_index=False,
        )
        .agg(
            node_name=("node_name", "first"),
            infrastructure_type=(
                "infrastructure_type",
                "first",
            ),
            capacity_mtpa=("capacity_mtpa", "first"),
            active_share=("active", "mean"),
        )
    )

    terminal_means = (
        panel.loc[panel["active"].eq(1)]
        .groupby("terminal_id")[METRICS]
        .mean()
        .add_prefix("mean_")
        .reset_index()
    )

    terminal_level = terminal_metadata.merge(
        terminal_means,
        on="terminal_id",
        how="left",
        validate="one_to_one",
    )

    capacity_variables = [
        "capacity_mtpa",
        "active_share",
        *[
            f"mean_{metric}"
            for metric in METRICS
        ],
    ]

    capacity_correlations = (
        terminal_level[capacity_variables]
        .corr(method="spearman")
    )

    write_csv(
        capacity_correlations,
        CORRELATIONS_DIR /
        "capacity_spearman_terminal_level.csv",
        index=True,
    )
    
    # Capacity correlations by infrastructure type

    capacity_rows = []

    for infrastructure_type, subset in terminal_level.groupby(
        "infrastructure_type"
    ):
        for variable in capacity_variables[1:]:
            capacity_rows.append(
                {
                    "infrastructure_type":
                        infrastructure_type,
                    "variable": variable,
                    "spearman_with_capacity":
                        safe_spearman(
                            subset["capacity_mtpa"],
                            subset[variable],
                        ),
                    "terminals": (
                        subset[
                            ["capacity_mtpa", variable]
                        ]
                        .dropna()
                        .shape[0]
                    ),
                }
            )

    write_csv(
        pd.DataFrame(capacity_rows),
        CORRELATIONS_DIR /
        "capacity_spearman_by_infrastructure_type.csv",
    )

def make_temporal_stability(panel):
    rows = []
    for terminal_id, group in panel.groupby("terminal_id"):
        group = group.sort_values("period_month")
        active_roles = group.loc[group["active"].eq(1),"terminal_role"]
        role = (active_roles.mode().iloc[0]
            if not active_roles.empty
            else "inactive")
        for metric in METRICS:
            rows.append(
                {
                    "terminal_id": terminal_id,
                    "modal_role": role,
                    "metric": metric,
                    "active_months": int(group["active"].sum()),
                    "lag1_autocorrelation": lag1_autocorrelation(group[metric]),
                }
            )
    stability = pd.DataFrame(rows)
    write_csv(stability, TEMPORAL_DIR / "lag1_autocorrelation_by_terminal.csv")

    rank_rows = []
    for metric in METRICS:
        ranks = panel.loc[panel["active"].eq(1), ["period_month", "terminal_id", metric]].dropna()
        ranks["rank"] = ranks.groupby("period_month")[metric].rank(method="average", ascending=False)
        wide = ranks.pivot(index="terminal_id", columns="period_month", values="rank")
        months = sorted(wide.columns)
        for previous, current in zip(months[:-1], months[1:]):
            rank_rows.append(
                {
                    "metric": metric,
                    "previous_month": previous,
                    "current_month": current,
                    "spearman_rank_persistence": safe_spearman(wide[previous], wide[current]),
                }
            )
    write_csv(pd.DataFrame(rank_rows), TEMPORAL_DIR / "month_to_month_rank_persistence.csv")


def make_rankings(panel):
    active = panel.loc[panel["active"].eq(1)].copy()
    ranking_metrics = [
        "throughput",
        "counterparty_hhi_terminal",
        "weighted_dependence_generated",
        "country_terminal_dependence",
        "role_specific_pagerank",
        "role_specific_dependence",
    ]
    rows = []
    for (month, role), group in active.groupby(["period_month", "terminal_role"]):
        for metric in ranking_metrics:
            ranked = group.dropna(subset=[metric]).nlargest(TOP_N, metric)
            for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
                rows.append(
                    {
                        "period_month": month,
                        "terminal_role": role,
                        "metric": metric,
                        "rank": rank,
                        "terminal_id": row["terminal_id"],
                        "node_name": row.get("node_name", np.nan),
                        "country": row.get("country", np.nan),
                        "value": row[metric],
                    }
                )
    write_csv(pd.DataFrame(rows), RANKINGS_DIR / "monthly_top_terminals.csv")


def make_plots(panel):
    active = panel.loc[panel["active"].eq(1)].copy()
    plot_metrics = [
        "throughput",
        "counterparty_hhi_terminal",
        "counterparty_hhi_country",
        "role_specific_dependence",
        "role_specific_pagerank",
    ]
    for metric in plot_metrics:
        values = active[metric].dropna()
        if values.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(values, bins=40, color="#2878B5", edgecolor="white")
        ax.set_title(f"Distribution of {metric} (active terminal-months)")
        ax.set_xlabel(metric)
        ax.set_ylabel("Frequency")
        fig.tight_layout()
        fig.savefig(DISTRIBUTIONS_DIR / f"{metric}.png", dpi=180)
        plt.close(fig)


def main():
    voyages, nodes = load_data()
    panel = assemble_metrics(voyages, nodes)
    country_month, country_full_period = build_country_role_tables(voyages)

    output_columns = [
        # Identifiers and metadata
        "terminal_id",
        "node_name",
        "period_month",
        "year",
        "month",
        "infrastructure_type",
        "capacity_mtpa",
        "unit_count",
        "start_year",
        "terminal_age",

        # Activity and role
        "terminal_role",
        "active",
        "export_share",
        "outgoing_flow",
        "incoming_flow",
        "throughput",
        "outgoing_voyages",
        "incoming_voyages",
        "voyage_count",

        # Voyage distance
        "mean_outgoing_voyage_distance",
        "mean_incoming_voyage_distance",
        "mean_voyage_distance",

        # Counterparties
        "counterparty_count",
        "counterparty_country_count",
        "counterparty_hhi_terminal",
        "counterparty_hhi_country",
        "effective_counterparties_terminal",
        "effective_counterparties_country",

        # Dependence
        "max_dependence_generated",
        "weighted_dependence_generated",
        "country_terminal_dependence",
        "role_specific_dependence",

        # PageRank
        "import_pagerank",
        "export_pagerank",
        "role_specific_pagerank",
    ]
    
    write_csv(
        panel[output_columns],
        DATA_OUTPUT_DIR / "terminal_month_metrics.csv",
    )
    
    write_csv(
        country_month,
        DATA_OUTPUT_DIR / "country_month_roles.csv",
    )

    write_csv(
        country_full_period,
        DATA_OUTPUT_DIR / "country_roles_full_period.csv",
    )
    
    country_role_summary = (
        country_month
        .groupby(["period_month", "country_role"], as_index=False)
        .agg(
            entities=("country", "nunique"),
            total_throughput=("throughput", "sum"),
        )
        .rename(columns={"country_role": "role"})
        .assign(entity_level="country")
    )

    country_role_summary = country_role_summary[
        [
            "period_month",
            "entity_level",
            "role",
            "entities",
            "total_throughput",
        ]
    ]

    terminal_role_summary = make_summary(panel)

    monthly_role_summary = (
        pd.concat(
            [terminal_role_summary, country_role_summary],
            ignore_index=True,
        )
        .sort_values(["period_month", "entity_level", "role"])
        .reset_index(drop=True)
    )

    write_csv(
        monthly_role_summary,
        DATA_OUTPUT_DIR / "monthly_role_summary.csv",
    )

    make_correlations(panel)
    make_temporal_stability(panel)
    make_rankings(panel)
    make_plots(panel)


if __name__ == "__main__":
    main()
