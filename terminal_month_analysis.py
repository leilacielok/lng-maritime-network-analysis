from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx


# ============================================================
# PATHS AND SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "eda_terminal_month"

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


# The matched-voyage workbook is preferred because it contains the canonical
# LNGN node IDs already used in the multilayer network.
MATCHED_VOYAGES_FILE = DATA_DIR / "LNG_voyage_node_matching_v1.xlsx"
NODES_FILE = DATA_DIR / "LNG_multilayer_nodes_v1.csv"

# Month assignment is based on voyage departure. Change to "end_date" if the
# research design should assign a cargo to its delivery month instead.
DATE_COLUMN = "start_date"
ROLE_EXPORT_THRESHOLD = 0.90
ROLE_IMPORT_THRESHOLD = 0.10
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


def role_from_share(export_share, throughput):
    if throughput <= 0:
        return "inactive"
    if export_share >= ROLE_EXPORT_THRESHOLD:
        return "exporter"
    if export_share <= ROLE_IMPORT_THRESHOLD:
        return "importer"
    return "bidirectional"


def write_csv(frame, path, index=False):
    frame.to_csv(path, index=index, float_format="%.12g")


# ============================================================
# LOAD AND VALIDATE DATA
# ============================================================

def load_data():
    print("\n" + "=" * 72)
    print("LOADING TERMINAL-MONTH INPUT DATA")
    print("=" * 72)

    if not MATCHED_VOYAGES_FILE.exists():
        raise FileNotFoundError(
            f"Missing {MATCHED_VOYAGES_FILE}. Place the matching workbook "
            "inside the data folder."
        )
    if not NODES_FILE.exists():
        raise FileNotFoundError(
            f"Missing {NODES_FILE}. Place the multilayer node table inside "
            "the data folder."
        )

    voyages = pd.read_excel(MATCHED_VOYAGES_FILE, sheet_name="Matched Voyages")
    nodes = pd.read_csv(NODES_FILE)

    required_voyages = {
        DATE_COLUMN,
        "voyage",
        "amount_cmb",
        "from_node_id",
        "to_node_id",
        "from_terminal",
        "to_terminal",
        "from_country",
        "to_country",
    }
    required_nodes = {"node_id", "node_name", "node_type", "country", "region"}

    missing_voyages = required_voyages.difference(voyages.columns)
    missing_nodes = required_nodes.difference(nodes.columns)
    if missing_voyages:
        raise ValueError(
            "Matched Voyages is missing: " + ", ".join(sorted(missing_voyages))
        )
    if missing_nodes:
        raise ValueError("Nodes file is missing: " + ", ".join(sorted(missing_nodes)))

    raw_rows = len(voyages)
    voyages[DATE_COLUMN] = pd.to_datetime(voyages[DATE_COLUMN], errors="coerce")
    voyages["amount_cmb"] = pd.to_numeric(voyages["amount_cmb"], errors="coerce")

    # Return voyages contain zero cargo and must not enter the trade network.
    voyages = voyages.loc[
        voyages["voyage"].astype(str).str.lower().eq("export")
        & voyages[DATE_COLUMN].notna()
        & voyages["from_node_id"].notna()
        & voyages["to_node_id"].notna()
        & voyages["amount_cmb"].gt(0)
    ].copy()

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

    duplicate_key = voyages.duplicated(
        [DATE_COLUMN, "from_node_id", "to_node_id", "amount_cmb", "IMO"],
        keep=False,
    ) if "IMO" in voyages.columns else pd.Series(False, index=voyages.index)

    qa = pd.DataFrame(
        {
            "check": [
                "raw_workbook_rows",
                "positive_export_rows_used",
                "observed_terminals",
                "months",
                "possible_duplicate_rows",
                "total_lng_volume_cmb",
            ],
            "value": [
                raw_rows,
                len(voyages),
                len(observed_ids),
                voyages["period_month"].nunique(),
                int(duplicate_key.sum()),
                voyages["amount_cmb"].sum(),
            ],
        }
    )
    write_csv(qa, DATA_OUTPUT_DIR / "input_qa_summary.csv")

    print(f"Positive export voyages: {len(voyages):,}")
    print(f"Observed terminals: {len(observed_ids):,}")
    print(
        f"Months: {voyages['period_month'].min():%Y-%m} to "
        f"{voyages['period_month'].max():%Y-%m}"
    )
    return voyages, nodes


# ============================================================
# TERMINAL-MONTH METRICS
# ============================================================

def build_complete_panel(voyages, nodes):
    months = pd.date_range(
        voyages["period_month"].min(), voyages["period_month"].max(), freq="MS"
    )
    panel = pd.MultiIndex.from_product(
        [nodes["node_id"].sort_values(), months],
        names=["terminal_id", "period_month"],
    ).to_frame(index=False)

    metadata_columns = [
        column
        for column in [
            "node_id",
            "node_name",
            "country",
            "region",
            "terminal_role",
            "latitude",
            "longitude",
        ]
        if column in nodes.columns
    ]
    metadata = nodes[metadata_columns].rename(
        columns={"node_id": "terminal_id", "terminal_role": "structural_role"}
    )
    return panel.merge(metadata, on="terminal_id", how="left", validate="many_to_one")


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
    """Calculate dependence generated by exporters and country dependence on import terminals."""
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

    imports = voyages.groupby(
        ["period_month", "to_country", "to_node_id"], as_index=False
    )["amount_cmb"].sum()
    country_total = imports.groupby(["period_month", "to_country"])["amount_cmb"].transform("sum")
    imports["country_terminal_dependence"] = imports["amount_cmb"] / country_total
    import_dep = imports.rename(
        columns={"to_node_id": "terminal_id", "to_country": "observed_import_country"}
    )[["period_month", "terminal_id", "observed_import_country", "country_terminal_dependence"]]
    return export_dep, import_dep


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


def assemble_metrics(voyages, nodes):
    panel = build_complete_panel(voyages, nodes)
    outgoing, incoming = calculate_basic_activity(voyages)
    concentration = calculate_concentration(voyages)
    export_dep, import_dep = calculate_dependence(voyages)
    pagerank = calculate_directional_pagerank(voyages, nodes["node_id"].tolist())

    for frame in [outgoing, incoming, concentration, export_dep, import_dep, pagerank]:
        panel = panel.merge(frame, on=["terminal_id", "period_month"], how="left")

    activity_columns = ["outgoing_flow", "incoming_flow", "outgoing_voyages", "incoming_voyages"]
    panel[activity_columns] = panel[activity_columns].fillna(0)
    panel["throughput"] = panel["outgoing_flow"] + panel["incoming_flow"]
    panel["voyage_count"] = panel["outgoing_voyages"] + panel["incoming_voyages"]
    panel["export_share"] = safe_divide(panel["outgoing_flow"], panel["throughput"])
    panel["terminal_role"] = [
        role_from_share(export_share, throughput)
        for export_share, throughput in zip(panel["export_share"].fillna(0), panel["throughput"])
    ]
    panel["active"] = panel["throughput"].gt(0).astype(int)

    exporter = panel["terminal_role"].eq("exporter")
    importer = panel["terminal_role"].eq("importer")
    bidirectional = panel["terminal_role"].eq("bidirectional")

    # For bidirectional terminals, combine inbound and outbound measures using
    # the corresponding flow shares.
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
        terminal_months=("terminal_id", "size"),
        terminals=("terminal_id", "nunique"),
        total_throughput=("throughput", "sum"),
    )
    write_csv(roles, DATA_OUTPUT_DIR / "monthly_role_summary.csv")


def make_correlations(panel):
    active = panel.loc[panel["active"].eq(1), METRICS + ["role_specific_dependence"]]
    write_csv(active.corr(method="pearson"), CORRELATIONS_DIR / "pearson_active.csv", index=True)
    write_csv(active.corr(method="spearman"), CORRELATIONS_DIR / "spearman_active.csv", index=True)

    rows = []
    for role in ["exporter", "importer", "bidirectional"]:
        subset = panel.loc[panel["terminal_role"].eq(role)]
        for metric in METRICS + ["role_specific_dependence"]:
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


def make_temporal_stability(panel):
    active = panel.loc[panel["active"].eq(1)].copy()
    rows = []
    for terminal_id, group in active.groupby("terminal_id"):
        group = group.sort_values("period_month")
        role = group["terminal_role"].mode().iloc[0]
        for metric in METRICS + ["role_specific_dependence"]:
            rows.append(
                {
                    "terminal_id": terminal_id,
                    "modal_role": role,
                    "metric": metric,
                    "active_months": group[metric].notna().sum(),
                    "lag1_autocorrelation": lag1_autocorrelation(group[metric]),
                }
            )
    stability = pd.DataFrame(rows)
    write_csv(stability, TEMPORAL_DIR / "lag1_autocorrelation_by_terminal.csv")

    rank_rows = []
    for metric in METRICS + ["role_specific_dependence"]:
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


def print_diagnostics(panel):
    active = panel.loc[panel["active"].eq(1)]
    print("\n" + "=" * 72)
    print("TERMINAL-MONTH PANEL")
    print("=" * 72)
    print(f"Rows: {len(panel):,}")
    print(f"Active terminal-months: {len(active):,}")
    print(f"Terminals: {panel['terminal_id'].nunique():,}")
    print("\nMonthly role counts (all panel rows):")
    print(panel["terminal_role"].value_counts().to_string())
    print("\nActive-observation missing shares:")
    print(active[METRICS + ["role_specific_dependence"]].isna().mean().sort_values(ascending=False).to_string())
    print("\nPageRank monthly sum range (should be approximately 1):")
    pr_sums = panel.groupby("period_month")[["import_pagerank", "export_pagerank"]].sum()
    print(pr_sums.agg(["min", "max"]).to_string())


def main():
    voyages, nodes = load_data()
    panel = assemble_metrics(voyages, nodes)

    output_columns = [
        "terminal_id", "node_name", "country", "region", "period_month", "year", "month",
        "structural_role", "terminal_role", "active", "export_share",
        "outgoing_flow", "incoming_flow", "throughput", "outgoing_voyages",
        "incoming_voyages", "voyage_count", "counterparty_count",
        "counterparty_country_count", "counterparty_hhi_terminal",
        "counterparty_hhi_country", "effective_counterparties_terminal",
        "effective_counterparties_country", "max_dependence_generated",
        "weighted_dependence_generated", "country_terminal_dependence",
        "role_specific_dependence", "import_pagerank", "export_pagerank",
        "role_specific_pagerank", "latitude", "longitude",
    ]
    output_columns = [column for column in output_columns if column in panel.columns]
    write_csv(panel[output_columns], DATA_OUTPUT_DIR / "terminal_month_metrics.csv")

    make_summary(panel)
    make_correlations(panel)
    make_temporal_stability(panel)
    make_rankings(panel)
    make_plots(panel)
    print_diagnostics(panel)

    print("\nAnalysis complete.")
    print(f"Main output: {DATA_OUTPUT_DIR / 'terminal_month_metrics.csv'}")


if __name__ == "__main__":
    main()
