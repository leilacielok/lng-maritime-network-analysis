"""Static rank-one criticality models conditional on consecutive activity.

Select terminal or chokepoint with --node-type.
Chokepoint bridge indicator omitted (near duplicate of articulation).

Optional --model node_effects adds hierarchical repeated-node intercepts.
Default --model-group baseline uses the original priors; CON and RS select
the alternative sensitivity profiles.
--prior-profile response_precision keeps regularized priors but uses
response-specific Beta precision medians as a third sensitivity candidate.
--low-ram for using less memory.
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
from baseline import prior_specification, build_model, initial_values, chokepoint_initial_values, rank_two_initial_values
import numpy as np
import pandas as pd

def fit_model(data, kind, args, destination):
    try:
        import pymc as pm
        import arviz as az
    except ImportError as exc:
        raise RuntimeError("Install scripts/models/requirements.txt to estimate the model") from exc
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from diagnostics import predictive_checks
    if args.model == "node_effects":
        from node_effects import build_factor_mean
        model = build_model(data, factor_builder=build_factor_mean, rank=args.rank)
    else:
        model = build_model(data, rank=args.rank)
    with model:
        prior_count_key = ("draws" if "draws" in inspect.signature(pm.sample_prior_predictive).parameters
                           else "samples")
        prior = pm.sample_prior_predictive(**{prior_count_key: args.prior_draws}, random_seed=args.seed)
        predictive_checks(prior, data, "prior_predictive").to_csv(destination / "prior_checks.csv", index=False)
        prior.to_netcdf(destination / "prior.nc")
        if args.prior_only:
            return
        if args.low_ram:
            from low_ram import run_chains, merge_chains, posterior_predictive_subset
            chain_dir = run_chains(model, data, kind, args, destination)
            merge_chains(chain_dir, destination / "posterior.nc", args.chains, args.draws)
            n_ppc = posterior_predictive_subset(model, destination / "posterior.nc", args)
            (destination / "low_ram_metadata.json").write_text(json.dumps({
                "chain_files": str(chain_dir), "chains": args.chains,
                "tune_per_chain": args.tune, "retained_per_chain": args.draws,
                "posterior_predictive_draws_per_chain": n_ppc,
                "observation_level_deterministics_not_stored": True,
                "posterior_draws_thinned": False,
                "note": "PPC is a subset. Global diagnostics are computed separately; run review.py for detailed plots."
            }, indent=2), encoding="utf-8")
            print("Saved full posterior and sample_stats; PPC uses a subset.", flush=True)
            return
        idata = pm.sample(draws=args.draws, tune=args.tune, chains=args.chains,
                          cores=args.cores, target_accept=args.target_accept,
                          init="adapt_diag",
                          initvals=(rank_two_initial_values(data, args.chains, args.seed) if args.rank == 2
                                     else initial_values(data, args.chains, args.seed) if kind == "terminal"
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
    if args.model == "node_effects":
        parameters += ["sigma_mu"]
    diagnostic = run_diagnostics(idata, data, kind, destination, parameters)
    if args.rank == 1:
        export_findings(idata, data, kind, destination, parameters, diagnostic, SPECS)
    else:
        # findings.py is rank-one-specific; retain rank-two summaries without mislabelling scores.
        az.summary(idata, var_names=parameters).to_csv(destination / "parameter_summary.csv")

def main(default_model="baseline"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["baseline", "node_effects"], default=default_model)
    parser.add_argument("--rank", type=int, choices=[1, 2], default=1,
                        help="Latent rank (rank 1 preserves original baseline)")
    parser.add_argument("--node-sd-prior", type=float, default=1.0,
                        help="HalfNormal scale for sigma_mu in node_effects model")
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
    parser.add_argument("--low-ram", action="store_true", help="Resumable chains; --cores controls parallel chains; chunked NetCDF assembly")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=2000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=1, help="Maximum concurrent chains in --low-ram mode (1 or 2 recommended with 12 GB RAM)")
    parser.add_argument("--prior-draws", type=int, default=200)
    parser.add_argument("--target-accept", type=float, default=.80)
    parser.add_argument("--seed", type=int, default=20261008)
    args = parser.parse_args()
    if args.rank == 2 and args.model != "baseline":
        parser.error("--rank 2 currently supports --model baseline only")
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
    if not np.isfinite(args.node_sd_prior) or args.node_sd_prior <= 0:
        parser.error("node-sd-prior must be finite and positive")
    kinds = [args.node_type]
    # Validate all selected inputs before starting any potentially expensive fit.
    data = {kind: prepare_panel(getattr(args, f"{kind}_panel"), kind) for kind in kinds}
    for kind, item in data.items():
        if args.model == "node_effects":
            from node_effects import prepare_node_effects
            prepare_node_effects(item, kind, args.node_sd_prior)
        item["precision_prior_sd"] = args.precision_prior_sd
        item["prior_profile"] = args.prior_profile
        item["audit"]["rank"] = args.rank
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
    if args.rank == 2:
        group_dir = (args.output_dir / "rank_two" if args.model_group == "baseline"
                     else args.output_dir / "rank_two" / "sensitivity_analysis" / args.model_group)
    if args.model == "node_effects":
        group_dir = (args.output_dir / "node_effects" if args.model_group == "baseline"
                     else args.output_dir / "node_effects" / "sensitivity_analysis" / args.model_group)
    run = group_dir / args.node_type / (args.run_name or f"{stamp}_{args.node_type}_bounded_{args.prior_profile}")
    if args.low_ram and run.exists():
        config_file = run / "run_config.json"
        if not config_file.exists():
            raise RuntimeError("Existing run directory lacks run_config.json; refusing to resume")
        previous = json.loads(config_file.read_text())
        for key in ("draws", "tune", "chains", "seed", "rank", "model", "node_type", "target_accept"):
            if str(previous[key]) != str(getattr(args, key)):
                raise RuntimeError(f"Cannot resume: {key} differs from original run")
        print(f"Resuming run: {run.resolve()}", flush=True)
    else:
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
              "node_effect_specification": data[args.node_type]["audit"].get("node_effect_specification"),
              "source_sha256": {name: hashlib.sha256((script_path.parent / name).read_bytes()).hexdigest()
                                for name in ["common.py", "baseline.py", "estimation.py", "node_effects.py",
                                             "diagnostics.py", "findings.py"]
                                if (script_path.parent / name).is_file()},
              "started_utc": stamp, "command_argv": sys.argv,
              "python": platform.python_version(), "packages": packages,
              "script_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest()}
    if not (run / "run_config.json").exists():
        (run / "run_config.json").write_text(json.dumps(config, default=str, indent=2), encoding="utf-8")
    for kind, item in data.items():
        destination = run / kind
        destination.mkdir(exist_ok=True)
        (destination / "sample_audit.json").write_text(json.dumps(item["audit"], indent=2), encoding="utf-8")
        if args.model == "node_effects":
            pd.DataFrame({"node_index": np.arange(len(item["node_ids"])),
                          "node_id": item["node_ids"], "node_name": item["node_names"]}).to_csv(
                              destination / "node_index.csv", index=False)
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
            if args.low_ram:
                write_status("sampling_complete_pending_diagnostics", note="Run low_ram_review.py to compute global diagnostics")
                continue
            diagnostics = json.loads((destination / "sampling_diagnostics.json").read_text())
            write_status("sampling_complete", passes_basic_sampling_checks=diagnostics["passes_basic_checks"],
                         note="Passing sampling checks does not establish model adequacy.")
    print(f"Results saved to {run}")

if __name__ == "__main__":
    main()
