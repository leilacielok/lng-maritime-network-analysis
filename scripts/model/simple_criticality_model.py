"""First Bayesian rank-one criticality model, conditional on consecutive activity.

Read existing panels; filter only in memory. Separate fits for terminals and
chokepoints. No country effects, throughput predictor, activity model or AR(1).

f_it = X_it beta + z_it, z_it ~ Normal(0, 1)
E[Y_cont,j | f] = alpha_cont,j + lambda_cont,j f (standardized scale)
logit P(Y_binary,j = 1 | f) = alpha_binary,j + lambda_binary,j f

Thus the covariate coefficient matrix has rank at most one. Unit residual
factor variance fixes scale; a positive PageRank loading fixes orientation.
Other loadings are unrestricted: a criticality interpretation requires review.
Gaussian continuous likelihoods are a preliminary approximation: they do not
respect [0, 1] bounds or reproduce boundary masses. Predictive checks explicitly
report these limitations. Repeated node/month dependence is not modelled yet.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
PANEL_DIR = BASE_DIR / "data" / "processed" / "model_ready"
OUTPUT_DIR = BASE_DIR / "outputs" / "model" / "simple_criticality"
BINARY_RESPONSES = [
    "is_articulation_point", "has_incident_global_bridge",
    "connected_to_articulation_point",
]
COMMON_PREDICTORS = [
    "neighbor_activity_lag1", "log1p_mean_voyage_distance_lag1",
    "month_sin", "month_cos",
]
SPECS = {
    "terminal": {
        "id": "terminal_id",
        "continuous": ["role_specific_pagerank", "counterparty_hhi_terminal",
                       "role_specific_dependence"],
        "predictors": COMMON_PREDICTORS + ["log1p_capacity_mtpa", "terminal_age"],
    },
    "chokepoint": {
        "id": "node_id",
        "continuous": ["pagerank", "betweenness", "share_monthly_network_flow"],
        "predictors": COMMON_PREDICTORS,
    },
}


def standardize(frame):
    """Fit scaling on the estimation sample; save it for interpretation/reuse."""
    mean, sd = frame.mean(), frame.std(ddof=0)
    if (sd <= 0).any() or not np.isfinite(sd).all():
        raise ValueError(f"Constant/invalid columns: {sd.index[sd <= 0].tolist()}")
    values = ((frame - mean) / sd).to_numpy(dtype=float)
    return values, pd.DataFrame({"mean": mean, "sd": sd})


def prepare_panel(path, kind):
    spec = SPECS[kind]
    frame = pd.read_csv(path, dtype={spec["id"]: "string"})
    needed = [spec["id"], "node_name", "period_month", "active",
              "activity_lag1", "lag_available", *spec["predictors"],
              *spec["continuous"], *BINARY_RESPONSES]
    if kind == "terminal":
        needed += ["terminal_role_lag1"]
    missing = sorted(set(needed) - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    keys = [spec["id"], "period_month"]
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError(f"{path.name}: missing or duplicate node-month keys")
    frame["period_month"] = pd.to_datetime(frame["period_month"], errors="raise")
    if not frame["period_month"].dt.is_month_start.all():
        raise ValueError("period_month must contain first-of-month dates")
    for col in ["active", "lag_available"]:
        if not frame[col].isin([0, 1]).all():
            raise ValueError(f"{col} must be binary without missing values")
    if not frame["activity_lag1"].dropna().isin([0, 1]).all():
        raise ValueError("activity_lag1 must be binary where observed")
    # Verify the supplied lag refers to the preceding CALENDAR month, not
    # the previous selected active row. Do not recalculate any panel metric.
    ordered = frame.sort_values(keys)
    grouped = ordered.groupby(spec["id"], sort=False)
    previous_month = grouped["period_month"].shift()
    previous_active = grouped["active"].shift()
    expected = previous_month.eq(ordered["period_month"] - pd.offsets.MonthBegin(1))
    if not ordered["lag_available"].eq(expected.astype(int)).all():
        raise ValueError("lag_available disagrees with calendar-month availability")
    if not ordered.loc[expected, "activity_lag1"].eq(previous_active[expected]).all():
        raise ValueError("activity_lag1 disagrees with previous calendar activity")
    audit = {"panel_rows": len(frame), "panel_nodes": frame[spec["id"]].nunique()}
    selected = frame.loc[frame["active"].eq(1)].copy()
    audit["excluded_inactive_current"] = len(frame) - len(selected)
    valid_lag = selected["lag_available"].eq(1)
    audit["excluded_no_calendar_lag"] = int((~valid_lag).sum())
    selected = selected.loc[valid_lag].copy()
    previous_active_mask = selected["activity_lag1"].eq(1)
    audit["excluded_inactive_previous"] = int((~previous_active_mask).sum())
    selected = selected.loc[previous_active_mask].copy()
    audit["consecutive_active_rows"] = len(selected)
    audit["excluded_missing_age"] = 0
    if kind == "terminal":
        audit["excluded_missing_age"] = int(selected["terminal_age"].isna().sum())
        selected = selected.loc[selected["terminal_age"].notna()].copy()
        if selected["terminal_age"].lt(0).any():
            raise ValueError("Negative terminal age: review the panel metadata")
        if not selected["terminal_role_lag1"].isin(["importer", "exporter"]).all():
            raise ValueError("Unexpected lagged role in consecutive-active sample")
        selected["exporter_lag1"] = selected["terminal_role_lag1"].eq("exporter").astype(int)
    selected = selected.sort_values(["period_month", spec["id"]]).reset_index(drop=True)
    if selected.empty:
        raise ValueError(f"No observations remain for {kind}")
    predictors = spec["predictors"] + (["exporter_lag1"] if kind == "terminal" else [])
    model_columns = predictors + spec["continuous"] + BINARY_RESPONSES
    values = selected[model_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Unexpected missing/nonfinite model values: "
                         f"{selected[model_columns].isna().sum().to_dict()}")
    if not selected[BINARY_RESPONSES].isin([0, 1]).all().all():
        raise ValueError("Structural responses must be binary")
    if not selected[spec["continuous"]].ge(0).all().all() or not selected[spec["continuous"]].le(1).all().all():
        raise ValueError("Continuous responses must lie in [0, 1]")
    if selected[BINARY_RESPONSES].nunique().lt(2).any():
        raise ValueError("A binary response is constant in the estimation sample")
    # Center/scale numeric covariates; keep role dummy on its 0/1 scale.
    x, x_scaling = standardize(selected[spec["predictors"]])
    if kind == "terminal":
        x = np.column_stack([x, selected["exporter_lag1"].to_numpy()])
        x_scaling.loc["exporter_lag1"] = [0.0, 1.0]
    yc, y_scaling = standardize(selected[spec["continuous"]])
    audit.update(estimation_rows=len(selected), estimation_nodes=selected[spec["id"]].nunique(),
                 predictors=predictors, continuous_responses=spec["continuous"],
                 binary_responses=BINARY_RESPONSES,
                 input_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return {"frame": selected, "x": x, "yc": yc,
            "yb": selected[BINARY_RESPONSES].to_numpy(dtype=int),
            "x_scaling": x_scaling, "y_scaling": y_scaling, "audit": audit}


def build_model(data):
    import pymc as pm
    import pytensor.tensor as pt

    coords = {"obs": np.arange(len(data["frame"])),
              "predictor": data["audit"]["predictors"],
              "continuous": data["audit"]["continuous_responses"],
              "other_continuous": data["audit"]["continuous_responses"][1:],
              "binary": BINARY_RESPONSES}
    with pm.Model(coords=coords) as model:
        beta = pm.Normal("beta", 0, 0.5, dims="predictor")
        factor_mean = pm.Deterministic("factor_mean", pt.dot(data["x"], beta), dims="obs")
        z = pm.Normal("factor_residual", 0, 1, dims="obs")
        factor = pm.Deterministic("factor", factor_mean + z, dims="obs")
        # PageRank is the first response in both specifications.
        anchor = pm.HalfNormal("pagerank_loading", 1)
        other = pm.Normal("other_loadings", 0, 1, dims="other_continuous")
        lc = pm.Deterministic("loading_cont", pt.concatenate([anchor[None], other]), dims="continuous")
        lb = pm.Normal("loading_binary", 0, 1, dims="binary")
        ac = pm.Normal("intercept_cont", 0, 1, dims="continuous")
        ab = pm.Normal("intercept_binary", 0, 2, dims="binary")
        sigma = pm.HalfNormal("sigma_cont", 1, dims="continuous")
        pm.Normal("y_cont", mu=ac + factor[:, None] * lc, sigma=sigma,
                  observed=data["yc"], dims=("obs", "continuous"))
        pm.Bernoulli("y_binary", logit_p=ab + factor[:, None] * lb,
                     observed=data["yb"], dims=("obs", "binary"))
    return model


def predictive_checks(idata, data, group):
    """Compare in-sample replicated distributions, on the original scale."""
    draws = getattr(idata, group)
    scale = data["y_scaling"]
    yc = draws["y_cont"].transpose("chain", "draw", "obs", "continuous").values
    yc = yc * scale["sd"].to_numpy() + scale["mean"].to_numpy()
    yb = draws["y_binary"].transpose("chain", "draw", "obs", "binary").values
    rows = []
    for j, col in enumerate(data["audit"]["continuous_responses"]):
        observed = data["frame"][col].to_numpy()
        replicated = yc[..., j]
        rows.append({"response": col, "observed_mean": observed.mean(),
                     "replicated_mean": replicated.mean(),
                     "observed_sd": observed.std(), "replicated_sd": replicated.std(),
                     "observed_zero_share": np.mean(observed == 0),
                     "observed_one_share": np.mean(observed == 1),
                     "replicated_outside_support_share": np.mean((replicated < 0) | (replicated > 1))})
    for j, col in enumerate(BINARY_RESPONSES):
        proportions = yb[..., j].mean(axis=-1)
        rows.append({"response": col, "observed_mean": data["yb"][:, j].mean(),
                     "replicated_mean": proportions.mean(),
                     "replicated_mean_q025": np.quantile(proportions, .025),
                     "replicated_mean_q975": np.quantile(proportions, .975)})
    return pd.DataFrame(rows)


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
        prior = pm.sample_prior_predictive(samples=args.prior_draws, random_seed=args.seed)
        predictive_checks(prior, data, "prior_predictive").to_csv(destination / "prior_checks.csv", index=False)
        prior.to_netcdf(destination / "prior.nc")
        if args.prior_only:
            return
        idata = pm.sample(draws=args.draws, tune=args.tune, chains=args.chains,
                          cores=args.cores, target_accept=args.target_accept,
                          random_seed=args.seed, return_inferencedata=True)
        pm.sample_posterior_predictive(idata, var_names=["y_cont", "y_binary"],
                                       random_seed=args.seed, extend_inferencedata=True)
    idata.to_netcdf(destination / "posterior.nc")
    parameters = ["beta", "loading_cont", "loading_binary", "intercept_cont",
                  "intercept_binary", "sigma_cont"]
    summary = az.summary(idata, var_names=parameters, hdi_prob=.95)
    summary.to_csv(destination / "parameter_summary.csv")
    # Include latent residual mixing: good global parameters alone are insufficient.
    mixing = az.summary(idata, var_names=parameters + ["factor_residual"], kind="diagnostics")
    diagnostic = {"divergences": int(idata.sample_stats["diverging"].sum()),
                  "max_rhat": float(mixing["r_hat"].max()),
                  "min_ess_bulk": float(mixing["ess_bulk"].min()),
                  "min_ess_tail": float(mixing["ess_tail"].min())}
    diagnostic["passes_basic_checks"] = bool(diagnostic["divergences"] == 0
        and np.isfinite(diagnostic["max_rhat"]) and diagnostic["max_rhat"] <= 1.01
        and diagnostic["min_ess_bulk"] >= 400 and diagnostic["min_ess_tail"] >= 400)
    (destination / "sampling_diagnostics.json").write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")
    if not diagnostic["passes_basic_checks"]:
        warnings.warn(f"{kind}: sampling checks did not pass; do not interpret scores yet")
    scores = data["frame"][[SPECS[kind]["id"], "node_name", "period_month"]].copy()
    for variable in ["factor", "factor_mean"]:
        samples = idata.posterior[variable].transpose("obs", "chain", "draw").values.reshape(len(scores), -1)
        scores[f"{variable}_mean"] = samples.mean(axis=1)
        scores[f"{variable}_q025"] = np.quantile(samples, .025, axis=1)
        scores[f"{variable}_q975"] = np.quantile(samples, .975, axis=1)
    scores.to_csv(destination / "factor_scores.csv", index=False)
    predictive_checks(idata, data, "posterior_predictive").to_csv(destination / "posterior_checks.csv", index=False)
    az.plot_trace(idata, var_names=parameters)
    plt.savefig(destination / "trace.png", dpi=150, bbox_inches="tight")
    plt.close("all")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terminal-panel", type=Path, default=PANEL_DIR / "terminal_criticality_model_panel.csv")
    parser.add_argument("--chokepoint-panel", type=Path, default=PANEL_DIR / "chokepoint_criticality_model_panel.csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--node-type", choices=["both", "terminal", "chokepoint"], default="both")
    parser.add_argument("--check-only", action="store_true", help="Validate/filter panels without PyMC or writing datasets")
    parser.add_argument("--prior-only", action="store_true", help="Save prior predictive checks without fitting")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=1, help="One core is portable to Windows; increase if desired")
    parser.add_argument("--prior-draws", type=int, default=200)
    parser.add_argument("--target-accept", type=float, default=.95)
    parser.add_argument("--seed", type=int, default=20261008)
    args = parser.parse_args()
    if min(args.draws, args.tune, args.chains, args.cores, args.prior_draws) < 1:
        parser.error("Sampling counts must be positive")
    if not 0 < args.target_accept < 1:
        parser.error("target-accept must be between 0 and 1")
    kinds = list(SPECS) if args.node_type == "both" else [args.node_type]
    # Validate all selected inputs before starting any potentially expensive fit.
    data = {kind: prepare_panel(getattr(args, f"{kind}_panel"), kind) for kind in kinds}
    for kind, item in data.items():
        print(f"{kind}: {item['audit']['estimation_rows']} observations, "
              f"{item['audit']['estimation_nodes']} nodes")
    if args.check_only:
        print(json.dumps({kind: item["audit"] for kind, item in data.items()}, indent=2))
        return
    # A unique run folder prevents accidental replacement of earlier results.
    run = args.output_dir / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run.mkdir(parents=True, exist_ok=False)
    (run / "run_config.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    for kind, item in data.items():
        destination = run / kind
        destination.mkdir()
        (destination / "sample_audit.json").write_text(json.dumps(item["audit"], indent=2), encoding="utf-8")
        item["x_scaling"].to_csv(destination / "predictor_scaling.csv")
        item["y_scaling"].to_csv(destination / "response_scaling.csv")
        item["frame"][BINARY_RESPONSES].corr().to_csv(destination / "binary_correlations.csv")
        fit_model(item, kind, args, destination)
    print(f"Results saved to {run}")


if __name__ == "__main__":
    main()
