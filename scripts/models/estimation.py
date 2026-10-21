"""Static rank-one criticality models conditional on consecutive activity.

Select terminal or chokepoint with --node-type.
Chokepoints: Beta PageRank/flow share; zero-inflated Beta betweenness.
Chokepoint bridge indicator omitted (near duplicate of articulation).

PageRank: Beta; HHI and dependence: one-inflated Beta, on original scale.
Interior means and probabilities of one have distinct factor loadings.
Structural responses: Bernoulli. Factor variance is one; PageRank loading positive.
No activity component, country, throughput, AR(1), or repeated-node effects.
Default priors use the regularized profile; --prior-profile original reproduces
the previous priors. Both profiles are provisional: check prior predictions
and posterior sensitivity before interpreting scores.
--prior-profile response_precision keeps regularized priors but uses
response-specific Beta precision medians as a third sensitivity candidate.
Place this file in scripts/models/ to resolve the existing default panel paths.
"""

import argparse
import inspect
import hashlib
import json
from pathlib import Path
import warnings
import sys
import platform
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError

from common import BASE_DIR, PANEL_DIR, OUTPUT_DIR, SPECS, prepare_panel
from baseline import prior_specification, build_model, initial_values, chokepoint_initial_values
import numpy as np
import pandas as pd


























from diagnostics import predictive_checks




def fit_model(data, kind, args, destination):
    try:
        import pymc as pm
        import arviz as az
    except ImportError as exc:
        raise RuntimeError("Install scripts/model/requirements.txt to estimate the model") from exc
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = build_model(data)
    with model:
        prior_count_key = ("draws" if "draws" in inspect.signature(pm.sample_prior_predictive).parameters
                           else "samples")
        prior = pm.sample_prior_predictive(**{prior_count_key: args.prior_draws}, random_seed=args.seed)
        predictive_checks(prior, data, "prior_predictive").to_csv(destination / "prior_checks.csv", index=False)
        prior.to_netcdf(destination / "prior.nc")
        if args.prior_only:
            return
        idata = pm.sample(draws=args.draws, tune=args.tune, chains=args.chains,
                          cores=args.cores, target_accept=args.target_accept,
                          init="adapt_diag",
                          initvals=(initial_values(data, args.chains, args.seed) if kind == "terminal"
                                    else chokepoint_initial_values(data, args.chains, args.seed)),
                          random_seed=args.seed, return_inferencedata=True)
        if (idata.posterior.sizes["chain"] != args.chains
                or idata.posterior.sizes["draw"] != args.draws):
            idata.to_netcdf(destination / "interrupted_posterior.nc")
            raise SystemExit("Incomplete sampling: saved interrupted_posterior.nc; "
                             "stopping without replacing posterior.nc or fitting another model")
        # Preserve completed sampling even if predictive simulation fails.
        idata.to_netcdf(destination / "posterior.nc")
        pm.sample_posterior_predictive(idata, var_names=["y_cont", "y_binary"],
                                       random_seed=args.seed, extend_inferencedata=True)
    idata.to_netcdf(destination / "posterior.nc")
    from diagnostics import run_diagnostics
    from findings import export_findings
    parameters = ["beta", "loading_cont", "loading_binary", "intercept_cont",
                  "intercept_binary", "precision_cont"] + (
                  ["intercept_one", "loading_one"] if kind == "terminal"
                  else ["intercept_zero", "loading_zero"])
    diagnostic = run_diagnostics(idata, data, kind, destination, parameters)
    export_findings(idata, data, kind, destination, parameters, diagnostic, SPECS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terminal-panel", type=Path, default=PANEL_DIR / "terminal_criticality_model_panel.csv")
    parser.add_argument("--chokepoint-panel", type=Path, default=PANEL_DIR / "chokepoint_criticality_model_panel.csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--prior-profile", choices=["original", "regularized", "response_precision"],
                        default=None, help="Automatically chosen from --model-group; optional matching explicit profile")
    parser.add_argument("--model-group", choices=["baseline", "CON", "RS"], default="baseline")
    parser.add_argument("--precision-prior-sd", type=float, default=.7,
                        help="Common log SD of Beta precision priors (medians depend on profile); try 1.0 for sensitivity")
    parser.add_argument("--run-name", help="Optional new subfolder name; existing folders are never overwritten")
    parser.add_argument("--node-type", choices=["terminal", "chokepoint"], default="terminal")
    parser.add_argument("--check-only", action="store_true", help="Validate/filter panels without PyMC or writing datasets")
    parser.add_argument("--prior-only", action="store_true", help="Save prior predictive checks without fitting")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=2000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=1, help="One core is portable to Windows; increase if desired")
    parser.add_argument("--prior-draws", type=int, default=200)
    parser.add_argument("--target-accept", type=float, default=.95)
    parser.add_argument("--seed", type=int, default=20261008)
    args = parser.parse_args()
    profiles = {"baseline": "original", "CON": "regularized", "RS": "response_precision"}
    expected_profile = profiles[args.model_group]
    if args.prior_profile is not None and args.prior_profile != expected_profile:
        parser.error("--prior-profile conflicts with --model-group")
    args.prior_profile = expected_profile
    if min(args.draws, args.tune, args.chains, args.cores, args.prior_draws) < 1:
        parser.error("Sampling counts must be positive")
    if not 0 < args.target_accept < 1:
        parser.error("target-accept must be between 0 and 1")
    if args.run_name and (args.run_name in (".", "..") or
                          any(c in args.run_name for c in '/\\:')):
        parser.error("run-name must be a single folder name")
    if not np.isfinite(args.precision_prior_sd) or args.precision_prior_sd <= 0:
        parser.error("precision-prior-sd must be finite and positive")
    kinds = [args.node_type]
    # Validate all selected inputs before starting any potentially expensive fit.
    data = {kind: prepare_panel(getattr(args, f"{kind}_panel"), kind) for kind in kinds}
    for kind, item in data.items():
        item["precision_prior_sd"] = args.precision_prior_sd
        item["prior_profile"] = args.prior_profile
        item["audit"]["prior_specification"] = {**prior_specification(kind, args.prior_profile),
                                                  "precision_log_sd": args.precision_prior_sd}
        item["audit"]["response_likelihood_scale"] = "original (0,1]"
        item["audit"]["response_scaling_purpose"] = "initial values only; not used in likelihood"
        y = item["frame"][item["audit"]["continuous_responses"]]
        if kind == "terminal":
            if y.eq(0).any().any() or y.iloc[:, 0].ge(1).any():
                raise ValueError("Terminal model requires positive responses and PageRank below one")
            if any(not 0 < v < 1 for v in y.iloc[:, 1:].eq(1).mean()):
                raise ValueError("Each inflated response needs boundary and interior observations")
        else:
            item["audit"]["response_likelihood_scale"] = "original [0,1)"
            if y.ge(1).any().any() or y[["pagerank", "share_monthly_network_flow"]].le(0).any().any():
                raise ValueError("Chokepoint PageRank/flow must be interior; no response may equal one")
            if not 0 < y["betweenness"].eq(0).mean() < 1:
                raise ValueError("Betweenness requires both zeros and interior observations")
            item["audit"]["excluded_binary_response"] = "has_incident_global_bridge: near duplicate of articulation"
        item["audit"]["boundary_counts"] = {c: {"zeros": int(y[c].eq(0).sum()),
                                                       "ones": int(y[c].eq(1).sum())} for c in y}
        print(f"{kind}: {item['audit']['estimation_rows']} observations, "
              f"{item['audit']['estimation_nodes']} nodes")
    if args.check_only:
        print(json.dumps({kind: item["audit"] for kind, item in data.items()}, indent=2))
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    group_dir = (args.output_dir / "baseline" if args.model_group == "baseline"
                 else args.output_dir / "sensitivity_analysis" / args.model_group)
    run = group_dir / args.node_type / (args.run_name or f"{stamp}_{args.node_type}_bounded_{args.prior_profile}")
    run.mkdir(parents=True, exist_ok=False)
    print(f"Run directory: {run.resolve()}", flush=True)
    script_path = Path(__file__).resolve()
    packages = {}
    for package in ["numpy", "pandas", "pymc", "pytensor", "arviz", "matplotlib"]:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    config = {**vars(args), "likelihood": ("Beta PageRank; one-inflated Beta HHI/dependence" if args.node_type == "terminal"
                                       else "Beta PageRank/flow; zero-inflated Beta betweenness"),
              "prior_specification": prior_specification(args.node_type, args.prior_profile),
              "precision_prior": {"distribution": "LogNormal",
                                  "median_by_response": dict(zip(SPECS[args.node_type]["continuous"],
                                      prior_specification(args.node_type, args.prior_profile)["precision_median"])),
                                  "log_sd": args.precision_prior_sd},
              "started_utc": stamp, "command_argv": sys.argv,
              "python": platform.python_version(), "packages": packages,
              "script_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest()}
    (run / "run_config.json").write_text(json.dumps(config, default=str, indent=2), encoding="utf-8")
    for kind, item in data.items():
        destination = run / kind
        destination.mkdir(exist_ok=True)
        (destination / "sample_audit.json").write_text(json.dumps(item["audit"], indent=2), encoding="utf-8")
        item["x_scaling"].to_csv(destination / "predictor_scaling.csv")
        item["y_scaling"].to_csv(destination / "response_scaling.csv")
        item["frame"][item["audit"]["binary_responses"]].corr().to_csv(destination / "binary_correlations.csv")
        status_path = destination / "run_status.json"
        def write_status(status, **details):
            status_path.write_text(json.dumps({"status": status, **details}, indent=2), encoding="utf-8")
        write_status("running")
        try:
            fit_model(item, kind, args, destination)
        except BaseException as exc:
            write_status("incomplete_or_failed", error_type=type(exc).__name__, error=str(exc))
            raise
        if args.prior_only:
            write_status("prior_only_complete")
        else:
            diagnostics = json.loads((destination / "sampling_diagnostics.json").read_text())
            write_status("sampling_complete", passes_basic_sampling_checks=diagnostics["passes_basic_checks"],
                         note="Passing sampling checks does not establish model adequacy.")
    print(f"Results saved to {run}")


if __name__ == "__main__":
    main()
