r"""
Forest plots of node-specific random intercepts.

Outputs
-------
1. terminal_node_effects_selected.pdf
   Five largest positive and five most negative terminal effects.

2. chokepoint_node_effects_all.pdf
   All chokepoint effects.

Example
-------
python scripts/visualization/plot_node_effects.py \
    --chokepoint-csv "C:\Users\Leila\thesis_code\outputs\model\node_effects\chokepoint\chokepoint_lowram_corrected_20000_30000\chokepoint\review_20261110T161054451592Z\node_effect_intervals.csv" \
    --terminal-csv "C:\Users\Leila\thesis_code\outputs\model\node_effects\terminal\terminal_lowram_corrected_20000_30000\terminal\review_20261110T163253314426Z\node_effect_intervals.csv"
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OUTPUT = (
    PROJECT_ROOT / "thesis" / "images"
)

DPI = 300

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------
# CSV utilities
# ---------------------------------------------------------

COLUMN_CANDIDATES = {
    "name": [
        "node_name",
        "terminal_name",
        "chokepoint_name",
        "name",
        "node",
        "label",
    ],
    "mean": [
        "mean",
        "posterior_mean",
        "mu_mean",
        "effect_mean",
        "mu_node_mean",
    ],
    "lower": [
        "q025",
        "q2.5",
        "q2.5%",
        "lower",
        "lower_95",
        "ci_lower",
        "hdi_2.5%",
    ],
    "upper": [
        "q975",
        "q97.5",
        "q97.5%",
        "upper",
        "upper_95",
        "ci_upper",
        "hdi_97.5%",
    ],
}


def find_column(df, candidates, override=None):
    """Find a column using case-insensitive matching."""

    if override is not None:
        if override not in df.columns:
            raise ValueError(
                f"Column '{override}' not found.\n"
                f"Available columns: {list(df.columns)}"
            )
        return override

    normalized = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]

    return None


def load_node_effects(
    csv_path,
    name_col=None,
    mean_col=None,
    lower_col=None,
    upper_col=None,
):
    """Load posterior node-effect summaries."""

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    columns = {
        "name": find_column(
            df, COLUMN_CANDIDATES["name"], name_col
        ),
        "mean": find_column(
            df, COLUMN_CANDIDATES["mean"], mean_col
        ),
        "lower": find_column(
            df, COLUMN_CANDIDATES["lower"], lower_col
        ),
        "upper": find_column(
            df, COLUMN_CANDIDATES["upper"], upper_col
        ),
    }

    missing = [
        key for key, value in columns.items()
        if value is None
    ]

    if missing:
        raise ValueError(
            f"Could not identify columns: {missing}\n"
            f"Available columns: {list(df.columns)}\n"
            "Specify the column names using the CLI options."
        )

    result = df[
        [
            columns["name"],
            columns["mean"],
            columns["lower"],
            columns["upper"],
        ]
    ].copy()

    result.columns = [
        "node_name",
        "mean",
        "lower",
        "upper",
    ]

    for col in ["mean", "lower", "upper"]:
        result[col] = pd.to_numeric(
            result[col], errors="raise"
        )

    if result.isna().any().any():
        raise ValueError(
            f"Missing values found in {csv_path}"
        )

    if not (
        (result["lower"] <= result["mean"])
        & (result["mean"] <= result["upper"])
    ).all():
        raise ValueError(
            "Some posterior means fall outside their "
            "reported credible intervals."
        )

    if result["node_name"].duplicated().any():
        raise ValueError(
            "Duplicate node names found. Check whether "
            "the CSV contains multiple rows per node."
        )

    return result


# ---------------------------------------------------------
# Node selection
# ---------------------------------------------------------

def select_extreme_nodes(df, n=5):
    """
    Select n lowest and n highest posterior mean effects.
    """

    ordered = df.sort_values("mean")

    negative = ordered.head(n)
    positive = ordered.tail(n)

    selected = pd.concat(
        [negative, positive]
    ).drop_duplicates(subset="node_name")

    return selected.sort_values("mean")


# ---------------------------------------------------------
# Plotting
# ---------------------------------------------------------

def forest_plot(
    df,
    output_dir,
    filename,
    title,
    figsize=None,
):
    """
    Horizontal forest plot of posterior random intercepts.
    """

    df = df.sort_values("mean").reset_index(drop=True)

    n_nodes = len(df)

    if figsize is None:
        figsize = (
            10,
            max(4.5, 0.35 * n_nodes + 1.5),
        )

    fig, ax = plt.subplots(figsize=figsize)

    y = np.arange(n_nodes)

    means = df["mean"].to_numpy()
    lower = df["lower"].to_numpy()
    upper = df["upper"].to_numpy()

    xerr = np.vstack([
        means - lower,
        upper - means,
    ])

    # Credible intervals and posterior means
    ax.errorbar(
        means,
        y,
        xerr=xerr,
        fmt="o",
        markersize=4.5,
        color="#254B70",
        ecolor="#7290AA",
        elinewidth=1.5,
        capsize=2.5,
        capthick=1.2,
        zorder=3,
    )

    # Reference line
    ax.axvline(
        0,
        color="#555555",
        linestyle="--",
        linewidth=1,
        alpha=0.8,
        zorder=1,
    )

    # Subtle horizontal grid
    ax.grid(
        axis="x",
        linestyle=":",
        linewidth=0.6,
        alpha=0.35,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(df["node_name"])

    ax.set_xlabel(
        r"Node-specific intercept $\mu_i$"
    )

    ax.set_title(
        title,
        fontsize=12,
        pad=14,
    )

    ax.set_ylim(-0.7, n_nodes - 0.3)

    # Automatic x limits with padding
    xmin = min(lower.min(), 0)
    xmax = max(upper.max(), 0)

    padding = max(
        0.05 * (xmax - xmin),
        0.15,
    )

    ax.set_xlim(
        xmin - padding,
        xmax + padding,
    )

    ax.tick_params(
        axis="y",
        length=0,
        pad=8,
    )

    fig.tight_layout()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination = output_dir / f"{filename}.pdf"

    fig.savefig(
        destination,
        bbox_inches="tight",
    )

    print(f"Saved: {destination}")
        

    plt.close(fig)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Create forest plots of posterior "
            "node-specific intercepts."
        )
    )

    parser.add_argument(
        "--terminal-csv",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--chokepoint-csv",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--name-col",
        default=None,
    )

    parser.add_argument(
        "--mean-col",
        default=None,
    )

    parser.add_argument(
        "--lower-col",
        default=None,
    )

    parser.add_argument(
        "--upper-col",
        default=None,
    )

    args = parser.parse_args()

    # Load terminal results
    terminal_df = load_node_effects(
        args.terminal_csv,
        name_col=args.name_col,
        mean_col=args.mean_col,
        lower_col=args.lower_col,
        upper_col=args.upper_col,
    )

    # Load chokepoint results
    chokepoint_df = load_node_effects(
        args.chokepoint_csv,
        name_col=args.name_col,
        mean_col=args.mean_col,
        lower_col=args.lower_col,
        upper_col=args.upper_col,
    )

    # -----------------------------------------------------
    # Terminal figure: five lowest + five highest >= 20 observations
    # -----------------------------------------------------

    # Count observations actually included in the fitted model
    latent = pd.read_csv(args.terminal_csv.parent / "latent_summary.csv")

    node_counts = (
        latent.groupby("node_name")
        .size()
        .rename("n_observations")
        .reset_index()
    )

    terminal_df = terminal_df.merge(
        node_counts,
        on="node_name",
        how="left",
        validate="one_to_one",
    )

    if terminal_df["n_observations"].isna().any():
        raise ValueError("Some terminal names could not be matched.")

    terminal_df = terminal_df[
        terminal_df["n_observations"] >= 20
    ].copy()

    print(
        f"Terminals with at least 20 observations: {len(terminal_df)}"
    )

    selected_terminals = select_extreme_nodes(
        terminal_df,
        n=5,
    )

    print("\nSelected terminals:")
    print(
        selected_terminals[
            ["node_name", "mean", "lower", "upper"]
        ].to_string(index=False)
    )

    forest_plot(
        selected_terminals,
        output_dir=args.output_dir,
        filename="terminal_node_effects_selected",
        title="Terminal-specific random intercepts",
        figsize=(10, 5.5),
    )

    # -----------------------------------------------------
    # Chokepoint figure: all nodes
    # -----------------------------------------------------

    print(
        f"\nChokepoints included: "
        f"{len(chokepoint_df)}"
    )

    forest_plot(
        chokepoint_df,
        output_dir=args.output_dir,
        filename="chokepoint_node_effects_all",
        title="Chokepoint-specific random intercepts",
        figsize=(10, 9.5),
    )

    print("\nAll figures generated successfully.")


if __name__ == "__main__":
    main()