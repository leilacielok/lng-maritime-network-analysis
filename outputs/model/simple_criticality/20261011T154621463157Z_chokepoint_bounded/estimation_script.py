"""Static rank-one criticality models conditional on consecutive activity.

Select terminal or chokepoint with --node-type.
Chokepoints: Beta PageRank/flow share; zero-inflated Beta betweenness.
Chokepoint bridge indicator omitted (near duplicate of articulation).

PageRank: Beta; HHI and dependence: one-inflated Beta, on original scale.
Interior means and probabilities of one have distinct factor loadings.
Structural responses: Bernoulli. Factor variance is one; PageRank loading positive.
No activity component, country, throughput, AR(1), or repeated-node effects.
Priors are provisional and must be checked through predictive/sensitivity analysis.
Place this file in scripts/model/ to resolve the existing default panel paths.
"""

import argparse
import inspect
import hashlib
import json
from pathlib import Path
import warnings
import sys
import shutil
import platform
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError

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
        "binary": BINARY_RESPONSES.copy(),
        "predictors": COMMON_PREDICTORS + ["log1p_capacity_mtpa", "terminal_age"],
    },
    "chokepoint": {
        "id": "node_id",
        "continuous": ["pagerank", "betweenness", "share_monthly_network_flow"],
        "binary": ["is_articulation_point", "connected_to_articulation_point"],
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
              *spec["continuous"], *spec["binary"]]
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
    model_columns = predictors + spec["continuous"] + spec["binary"]
    values = selected[model_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Unexpected missing/nonfinite model values: "
                         f"{selected[model_columns].isna().sum().to_dict()}")
    if not selected[spec["binary"]].isin([0, 1]).all().all():
        raise ValueError("Structural responses must be binary")
    if not selected[spec["continuous"]].ge(0).all().all() or not selected[spec["continuous"]].le(1).all().all():
        raise ValueError("Continuous responses must lie in [0, 1]")
    if selected[spec["binary"]].nunique().lt(2).any():
        raise ValueError("A binary response is constant in the estimation sample")
    # Center/scale numeric covariates; keep role dummy on its 0/1 scale.
    x, x_scaling = standardize(selected[spec["predictors"]])
    if kind == "terminal":
        x = np.column_stack([x, selected["exporter_lag1"].to_numpy()])
        x_scaling.loc["exporter_lag1"] = [0.0, 1.0]
    yc, y_scaling = standardize(selected[spec["continuous"]])
    audit.update(estimation_rows=len(selected), estimation_nodes=selected[spec["id"]].nunique(),
                 predictors=predictors, continuous_responses=spec["continuous"],
                 binary_responses=spec["binary"],
                 input_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return {"frame": selected, "x": x, "yc": yc,
            "yb": selected[spec["binary"]].to_numpy(dtype=int),
            "x_scaling": x_scaling, "y_scaling": y_scaling, "audit": audit}


def one_inflated_beta_logp(value, alpha, beta, boundary_logit, inflated):
    """Mixed measure: Lebesgue density on (0,1), point probability at one.

    Substitute 0.5 ONLY in the unselected Beta branch for exact boundaries;
    observations are never clipped, shifted, or replaced in the likelihood.
    """
    import pymc as pm
    import pytensor.tensor as pt
    inside = (value > 0) & (value < 1)
    safe_value = pt.switch(inside, value, 0.5)
    log_beta = pm.logp(pm.Beta.dist(alpha=alpha, beta=beta), safe_value)
    log_one = -pt.logaddexp(0.0, -boundary_logit)
    log_interior = -pt.logaddexp(0.0, boundary_logit)
    continuous = log_beta + pt.switch(inflated, log_interior, 0.0)
    return pt.switch(inside, continuous,
                     pt.switch(pt.eq(value, 1) & inflated, log_one, -np.inf))


def one_inflated_beta_random(alpha, beta, boundary_logit, inflated, rng=None, size=None):
    rng = np.random.default_rng() if rng is None else rng
    # Stable sigmoid, including extreme logits in predictive draws.
    prob = np.exp(-np.logaddexp(0.0, -np.asarray(boundary_logit)))
    prob = np.where(inflated, prob, 0.0)
    interior = rng.beta(alpha, beta, size=size)
    # Floating-point beta routines may round to 0/1; keep interior draws interior.
    interior = np.clip(interior, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))
    return np.where(rng.uniform(size=np.shape(interior)) < prob, 1.0, interior)


def build_terminal_model(data):
    import pymc as pm
    import pytensor.tensor as pt

    coords = {"obs": np.arange(len(data["frame"])),
              "predictor": data["audit"]["predictors"],
              "continuous": data["audit"]["continuous_responses"],
              "other_continuous": data["audit"]["continuous_responses"][1:],
              "binary": data["audit"]["binary_responses"],
              "boundary": data["audit"]["continuous_responses"][1:]}
    with pm.Model(coords=coords) as model:
        beta = pm.Normal("beta", 0, 0.5, dims="predictor")
        factor_mean = pm.Deterministic("factor_mean", pt.dot(data["x"], beta), dims="obs")
        factor = pm.Normal("factor", mu=factor_mean, sigma=1, dims="obs")
        pm.Deterministic("factor_residual", factor - factor_mean, dims="obs")
        # PageRank is the first response in both specifications.
        anchor = pm.HalfNormal("pagerank_loading", 1)
        other = pm.Normal("other_loadings", 0, 1, dims="other_continuous")
        lc = pm.Deterministic("loading_cont", pt.concatenate([anchor[None], other]), dims="continuous")
        lb = pm.Normal("loading_binary", 0, 1, dims="binary")
        ac = pm.Normal("intercept_cont", mu=[-4.0, 0.0, 0.0], sigma=1.5, dims="continuous")
        ab = pm.Normal("intercept_binary", 0, 2, dims="binary")
        # Fixed-response precision prior: median 10, log SD .7 by default.
        phi = pm.LogNormal("precision_cont", mu=np.log(10.0),
                           sigma=data["precision_prior_sd"], dims="continuous")
        boundary_intercept = pm.Normal("intercept_one", 0, 1.5, dims="boundary")
        boundary_loading = pm.Normal("loading_one", 0, 1, dims="boundary")
        eta_one = boundary_intercept + factor[:, None] * boundary_loading
        logits = pt.concatenate([pt.zeros_like(factor[:, None]), eta_one], axis=1)
        mu = pm.math.sigmoid(ac + factor[:, None] * lc)
        pm.CustomDist("y_cont", mu * phi, (1 - mu) * phi, logits,
                      np.array([False, True, True]),
                      logp=one_inflated_beta_logp, random=one_inflated_beta_random,
                      observed=data["frame"][data["audit"]["continuous_responses"]].to_numpy(),
                      dims=("obs", "continuous"))
        pm.Bernoulli("y_binary", logit_p=ab + factor[:, None] * lb,
                     observed=data["yb"], dims=("obs", "binary"))
    return model



def zero_inflated_beta_logp(value, alpha, beta, boundary_logit, inflated):
    # Reflecting the Beta and boundary gives exactly a zero-inflated Beta.
    return one_inflated_beta_logp(1 - value, beta, alpha, boundary_logit, inflated)


def zero_inflated_beta_random(alpha, beta, boundary_logit, inflated, rng=None, size=None):
    rng = np.random.default_rng() if rng is None else rng
    prob = np.exp(-np.logaddexp(0.0, -np.asarray(boundary_logit)))
    prob = np.where(inflated, prob, 0.0)
    interior = np.clip(rng.beta(alpha, beta, size=size),
                       np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))
    return np.where(rng.uniform(size=np.shape(interior)) < prob, 0.0, interior)


def build_chokepoint_model(data):
    import pymc as pm
    import pytensor.tensor as pt
    coords = {"obs": np.arange(len(data["frame"])),
              "predictor": data["audit"]["predictors"],
              "continuous": data["audit"]["continuous_responses"],
              "other_continuous": data["audit"]["continuous_responses"][1:],
              "binary": data["audit"]["binary_responses"],
              "boundary": ["betweenness"]}
    with pm.Model(coords=coords) as model:
        beta = pm.Normal("beta", 0, .5, dims="predictor")
        mean = pm.Deterministic("factor_mean", pt.dot(data["x"], beta), dims="obs")
        factor = pm.Normal("factor", mean, 1, dims="obs")
        pm.Deterministic("factor_residual", factor - mean, dims="obs")
        anchor = pm.HalfNormal("pagerank_loading", 1)
        other = pm.Normal("other_loadings", 0, 1, dims="other_continuous")
        lc = pm.Deterministic("loading_cont", pt.concatenate([anchor[None], other]), dims="continuous")
        lb = pm.Normal("loading_binary", 0, 1, dims="binary")
        ac = pm.Normal("intercept_cont", [-4., -5., -4.], 1.5, dims="continuous")
        ab = pm.Normal("intercept_binary", 0, 2, dims="binary")
        phi = pm.LogNormal("precision_cont", np.log(10.), data["precision_prior_sd"], dims="continuous")
        azero = pm.Normal("intercept_zero", -5., 1.5, dims="boundary")
        lzero = pm.Normal("loading_zero", 0, 1, dims="boundary")
        eta = azero + factor[:, None] * lzero
        logits = pt.concatenate([pt.zeros_like(eta), eta, pt.zeros_like(eta)], axis=1)
        mu = pm.math.sigmoid(ac + factor[:, None] * lc)
        pm.CustomDist("y_cont", mu * phi, (1-mu)*phi, logits,
                      np.array([False, True, False]),
                      logp=zero_inflated_beta_logp, random=zero_inflated_beta_random,
                      observed=data["frame"][data["audit"]["continuous_responses"]].to_numpy(),
                      dims=("obs", "continuous"))
        pm.Bernoulli("y_binary", logit_p=ab + factor[:, None]*lb,
                     observed=data["yb"], dims=("obs", "binary"))
    return model


def build_model(data):
    return (build_terminal_model(data) if "exporter_lag1" in data["audit"]["predictors"]
            else build_chokepoint_model(data))


def chokepoint_initial_values(data, chains, seed):
    rng = np.random.default_rng(seed)
    f = data["yc"][:, 0]
    b = np.linalg.lstsq(data["x"], f, rcond=None)[0]
    y = data["frame"][data["audit"]["continuous_responses"]]
    means = np.array([y[c].loc[y[c].gt(0)].mean() for c in y])
    p = np.clip(data["yb"].mean(axis=0), .01, .99)
    pzero = y["betweenness"].eq(0).mean()
    starts = []
    for _ in range(chains):
        starts.append({"beta": b+rng.normal(0,.03,b.shape),
                       "factor": f+rng.normal(0,.05,f.shape),
                       "pagerank_loading": float(rng.uniform(.6,.8)),
                       "other_loadings": rng.normal(.2,.05,2),
                       "loading_binary": rng.normal(0,.05,len(p)),
                       "intercept_cont": np.log(means/(1-means))+rng.normal(0,.03,3),
                       "intercept_binary": np.log(p/(1-p))+rng.normal(0,.03,len(p)),
                       "precision_cont": 10*np.exp(rng.normal(0,.03,3)),
                       "intercept_zero": np.array([np.log(pzero/(1-pzero))])+rng.normal(0,.03,1),
                       "loading_zero": rng.normal(0,.05,1)})
    return starts

def predictive_checks(idata, data, group):
    """Compare in-sample replicated distributions, on the original scale."""
    draws = getattr(idata, group)
    yc = draws["y_cont"].transpose("chain", "draw", "obs", "continuous").values
    # Bounded responses are modelled directly on their original scale.
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
    for j, col in enumerate(data["audit"]["binary_responses"]):
        proportions = yb[..., j].mean(axis=-1)
        rows.append({"response": col, "observed_mean": data["yb"][:, j].mean(),
                     "replicated_mean": proportions.mean(),
                     "replicated_mean_q025": np.quantile(proportions, .025),
                     "replicated_mean_q975": np.quantile(proportions, .975)})
    # Quantities calculated separately for each replicated dataset, not pooled.
    masks = {"all": np.ones(len(data["frame"]), dtype=bool)}
    for role in (["importer", "exporter"] if "terminal_role_lag1" in data["frame"] else []):
        masks[role] = data["frame"]["terminal_role_lag1"].eq(role).to_numpy()
    for role, mask in masks.items():
        if not mask.any():
            continue
        for j, col in enumerate(data["audit"]["continuous_responses"]):
            obs = data["frame"][col].to_numpy()[mask]
            rep = yc[..., mask, j]
            statistics = {"mean": (obs.mean(), rep.mean(axis=-1)),
                          "sd": (obs.std(), rep.std(axis=-1)),
                          "one_share": ((obs == 1).mean(), (rep == 1).mean(axis=-1)),
                          "zero_share": ((obs == 0).mean(), (rep == 0).mean(axis=-1))}
            for q in [.05, .5, .95]:
                statistics[f"q{int(q*100):02d}"] = (np.quantile(obs, q), np.quantile(rep, q, axis=-1))
            for stat, (actual, replicated) in statistics.items():
                rows.append({"response": col, "role_lag1": role, "statistic": stat,
                             "observed": actual, "replicated_mean": replicated.mean(),
                             "replicated_q025": np.quantile(replicated, .025),
                             "replicated_q975": np.quantile(replicated, .975)})
    return pd.DataFrame(rows)


def initial_values(data, chains, seed):
    """Data-informed starting points only; priors and likelihoods stay unchanged.

    Orient the initial factor toward PageRank, with small independent chain
    perturbations. This does not prove that other posterior modes are absent.
    """
    rng = np.random.default_rng(seed)
    factor_start = data["yc"][:, 0].copy()
    beta_start = np.linalg.lstsq(data["x"], factor_start, rcond=None)[0]
    loading = data["yc"].T @ factor_start / (factor_start @ factor_start)
    # PageRank initializes the factor but must not start with zero noise.
    probability = np.clip(data["yb"].mean(axis=0), 0.01, 0.99)
    intercept_binary = np.log(probability / (1 - probability))
    binary_loading = np.array([
        (np.corrcoef(factor_start, data["yb"][:, j])[0, 1]
         if data["yb"][:, j].std() > 0 else 0.0)
        for j in range(data["yb"].shape[1])
    ])
    starts = []
    for _ in range(chains):
        starts.append({
            "beta": beta_start + rng.normal(0, 0.03, beta_start.shape),
            "factor": factor_start + rng.normal(0, 0.05, factor_start.shape),
            "pagerank_loading": float(rng.uniform(0.6, 0.8)),
            "other_loadings": loading[1:] + rng.normal(0, 0.03, loading[1:].shape),
            "loading_binary": binary_loading + rng.normal(0, 0.03, binary_loading.shape),
            "intercept_cont": np.array([
                np.log(v / (1-v)) for v in [
                    data["frame"][col].loc[data["frame"][col].lt(1)].mean()
                    for col in data["audit"]["continuous_responses"]]]) + rng.normal(0, .03, 3),
            "intercept_one": np.array([
                np.log(v / (1-v)) for v in
                data["frame"][data["audit"]["continuous_responses"][1:]].eq(1).mean()
            ]) + rng.normal(0, .03, 2),
            "loading_one": rng.normal(0, .05, 2),
            "intercept_binary": intercept_binary + rng.normal(0, 0.03, probability.shape),
            "precision_cont": 10 * np.exp(rng.normal(0, .03, 3)),
        })
    return starts


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
    parameters = ["beta", "loading_cont", "loading_binary", "intercept_cont",
                  "intercept_binary", "precision_cont"] + (
                      ["intercept_one", "loading_one"] if kind == "terminal"
                      else ["intercept_zero", "loading_zero"])
    summary = az.summary(idata, var_names=parameters, hdi_prob=.95)
    summary.to_csv(destination / "parameter_summary.csv")
    for chain in idata.posterior.coords["chain"].values:
        az.summary(idata.sel(chain=[chain]), var_names=parameters,
                   kind="stats", hdi_prob=.95).to_csv(
                       destination / f"parameter_summary_chain_{chain}.csv")
    # Include latent residual mixing: good global parameters alone are insufficient.
    mixing = az.summary(idata, var_names=parameters + ["factor", "factor_mean", "factor_residual"], kind="diagnostics")
    diagnostic = {"divergences": int(idata.sample_stats["diverging"].sum()),
                  "max_rhat": float(mixing["r_hat"].max()),
                  "min_ess_bulk": float(mixing["ess_bulk"].min()),
                  "min_ess_tail": float(mixing["ess_tail"].min())}
    diagnostic["nonfinite_diagnostic_entries"] = int(
        (~np.isfinite(mixing[["r_hat", "ess_bulk", "ess_tail"]].to_numpy())).sum())
    mixing.to_csv(destination / "mixing_diagnostics.csv")
    diagnostic["passes_basic_checks"] = bool(
        diagnostic["nonfinite_diagnostic_entries"] == 0 and diagnostic["divergences"] == 0
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
    scores["passes_basic_sampling_checks"] = diagnostic["passes_basic_checks"]
    score_filename = ("factor_scores.csv" if diagnostic["passes_basic_checks"]
                      else "factor_scores_UNVALIDATED.csv")
    scores.to_csv(destination / score_filename, index=False)
    predictive_checks(idata, data, "posterior_predictive").to_csv(destination / "posterior_checks.csv", index=False)
    with az.rc_context({"plot.max_subplots": 100}):
        az.plot_trace(idata, var_names=parameters, compact=False)
    plt.savefig(destination / "trace.png", dpi=150, bbox_inches="tight")
    plt.close("all")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terminal-panel", type=Path, default=PANEL_DIR / "terminal_criticality_model_panel.csv")
    parser.add_argument("--chokepoint-panel", type=Path, default=PANEL_DIR / "chokepoint_criticality_model_panel.csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--precision-prior-sd", type=float, default=.7,
                        help="Log SD of Beta precision prior (median 10); try 1.0 for sensitivity")
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
    run = args.output_dir / (args.run_name or f"{stamp}_{args.node_type}_bounded")
    run.mkdir(parents=True, exist_ok=False)
    print(f"Run directory: {run.resolve()}", flush=True)
    script_path = Path(__file__).resolve()
    shutil.copy2(script_path, run / "estimation_script.py")
    packages = {}
    for package in ["numpy", "pandas", "pymc", "pytensor", "arviz", "matplotlib"]:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    config = {**vars(args), "likelihood": ("Beta PageRank; one-inflated Beta HHI/dependence" if args.node_type == "terminal"
                                       else "Beta PageRank/flow; zero-inflated Beta betweenness"),
              "precision_prior": {"distribution": "LogNormal", "median": 10, "log_sd": args.precision_prior_sd},
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
