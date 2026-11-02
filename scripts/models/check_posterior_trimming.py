from pathlib import Path
import arviz as az
import pandas as pd
import json
from diagnostics import save_classical_diagnostics
# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

# Inserisci il percorso del posterior.nc dell'ultima
# esecuzione con target_accept=0.95

source = Path(
    r"C:\Users\Leila\OneDrive\Desktop\Thesis\code\outputs\model\node_effects_1\terminal\20261031T100248066626Z_terminal_bounded_original\terminal\posterior.nc"
)

output_dir = source.parent / "trimming_diagnostics"
output_dir.mkdir(parents=True, exist_ok=True)

trim_values = [0, 1000, 2000]

# --------------------------------------------------
# LOAD POSTERIOR
# --------------------------------------------------

print("Loading posterior...")

idata = az.from_netcdf(source)

print(f"Chains: {idata.posterior.sizes['chain']}")
print(f"Draws: {idata.posterior.sizes['draw']}")

parameters = [
    "beta",
    "intercept_binary",
    "intercept_binary_centered",
    "intercept_cont",
    "intercept_cont_centered",
    "intercept_one",
    "intercept_one_centered",
    "loading_binary",
    "loading_cont",
    "loading_one",
    "node_contrast_raw",
    "node_mean",
    "other_loadings",
    "pagerank_loading",
    "precision_cont",
    "sigma_mu",
]

missing = [
    name for name in parameters
    if name not in idata.posterior
]

if missing:
    raise ValueError(f"Missing parameters: {missing}")

n_parameters = sum(
    idata.posterior[name].size
    // (
        idata.posterior.sizes["chain"]
        * idata.posterior.sizes["draw"]
    )
    for name in parameters
)

n_tests = n_parameters * idata.posterior.sizes["chain"]

print(f"Selected scalar parameters: {n_parameters}")
print(f"Expected Geweke tests: {n_tests}")

# --------------------------------------------------
# COMPUTE DIAGNOSTICS
# --------------------------------------------------

results = []

for trim in trim_values:

    print(f"\nAnalyzing trim = {trim}...")

    posterior = idata.posterior.isel(
        draw=slice(trim, None)
    )

    # Geweke and Heidelberger-Welch diagnostics
    classical_dir = output_dir / f"classical_trim_{trim}"

    save_classical_diagnostics(
        posterior,
        classical_dir,
        parameters=parameters,
        alpha=0.05,
        eps=0.10,
    )

    geweke_df = pd.read_csv(
        classical_dir / "geweke_diagnostics.csv"
    )

    hw_df = pd.read_csv(
        classical_dir / "heidelberger_welch_diagnostics.csv"
    )
    
    geweke_failures = int(
    (geweke_df["passed"].astype(str).str.lower() == "false").sum()
    )

    hw_stationarity_failures = int(
        (
            hw_df["stationarity_passed"]
            .astype(str)
            .str.lower() == "false"
        ).sum()
    )

    hw_halfwidth_failures = int(
        (
            hw_df["half_width_passed"]
            .astype(str)
            .str.lower() == "false"
        ).sum()
    )

    print(
        f"Trim {trim}: "
        f"Geweke failures = {geweke_failures}/{len(geweke_df)}, "
        f"HW stationarity failures = {hw_stationarity_failures}/{len(hw_df)}, "
        f"HW half-width failures = {hw_halfwidth_failures}/{len(hw_df)}"
    )

    summary = az.summary(
        posterior,
        kind="diagnostics"
    )

    summary.to_csv(
        output_dir / f"diagnostics_trim_{trim}.csv"
    )

    rhat = summary["r_hat"]
    ess_bulk = summary["ess_bulk"]
    ess_tail = summary["ess_tail"]

    results.append({
        "trim": trim,
        "draws_per_chain": posterior.sizes["draw"],
        "max_rhat": rhat.max(),
        "min_ess_bulk": ess_bulk.min(),
        "min_ess_tail": ess_tail.min(),
        "n_rhat_above_1_01": (rhat > 1.01).sum(),
        "n_ess_bulk_below_400": (ess_bulk < 400).sum(),
        "n_ess_tail_below_400": (ess_tail < 400).sum(),
        "n_geweke_tests": len(geweke_df),
        "n_geweke_failures": geweke_failures,
        "n_hw_stationarity_failures": hw_stationarity_failures,
        "n_hw_halfwidth_failures": hw_halfwidth_failures,
    })

# --------------------------------------------------
# SAVE COMPARISON
# --------------------------------------------------

comparison = pd.DataFrame(results)

comparison.to_csv(
    output_dir / "trimming_comparison.csv",
    index=False
)

print("\n" + "=" * 60)
print("TRIMMING COMPARISON")
print("=" * 60)

print(comparison.to_string(index=False))

print("\nResults saved to:")
print(output_dir)