"""Rank-one baseline and simple rank-two extension; CON/RS change priors."""
import numpy as np
from common import (SPECS, one_inflated_beta_logp, one_inflated_beta_random, zero_inflated_beta_logp, zero_inflated_beta_random)

def prior_specification(kind, profile="regularized"):
    """Fixed candidate priors, not fitted to empirical response frequencies.

    Keep the original intercept centres. Concentrate
    small-response intercepts/loadings and boundary logits to reduce extreme
    prior predictions. This is a sensitivity candidate, not a validated prior.
    All scales below are standard deviations (HalfNormal: scale).
    """
    if kind not in SPECS or profile not in ("original", "regularized", "response_precision"):
        raise ValueError("Unknown node type or prior profile")
    original = profile == "original"
    return {
        "profile": profile,
        "beta_sd": 0.5,
        "pagerank_loading_sd": 1.0 if original else 0.5,
        "other_loading_sd": ([1.0, 1.0] if original or kind == "terminal"
                             else [0.5, 0.5]),
        "binary_loading_sd": 1.0,
        "interior_intercept_mean": ([-4.0, 0.0, 0.0] if kind == "terminal"
                                    else [-4.0, -5.0, -4.0]),
        "interior_intercept_sd": ([1.5, 1.5, 1.5] if original else
                                  ([0.7, 1.5, 1.5] if kind == "terminal"
                                   else [0.7, 0.7, 0.7])),
        "binary_intercept_sd": 2.0,
        "boundary_intercept_mean": 0.0 if kind == "terminal" else -5.0,
        "boundary_intercept_sd": 1.5 if original else 1.0,
        "boundary_loading_sd": 1.0 if original else 0.5,
        # Candidate scales: at mu=.018, phi=300/500 gives conditional
        # Beta SD approximately .0077/.0059. At mu=.01/.02, phi=100
        # gives SD approximately .0099/.0139. These are illustrative
        # conditional scales, not matches to marginal empirical SDs.
        # Keep HHI/dependence unchanged to isolate small-response precision.
        "precision_median": (([300.0, 10.0, 10.0] if kind == "terminal"
                              else [500.0, 100.0, 100.0])
                             if profile == "response_precision"
                             else [10.0, 10.0, 10.0]),
        "precision_response_order": SPECS[kind]["continuous"].copy(),
    }


def build_terminal_model(data, factor_builder=None):
    import pymc as pm
    import pytensor.tensor as pt

    priors = prior_specification("terminal", data.get("prior_profile", "regularized"))
    coords = {"obs": np.arange(len(data["frame"])),
              "predictor": data["audit"]["predictors"],
              "continuous": data["audit"]["continuous_responses"],
              "other_continuous": data["audit"]["continuous_responses"][1:],
              "binary": data["audit"]["binary_responses"],
              "boundary": data["audit"]["continuous_responses"][1:]}
    with pm.Model(coords=coords) as model:
        beta = pm.Normal("beta", 0, priors["beta_sd"], dims="predictor")
        factor_mean = (pm.Deterministic("factor_mean", pt.dot(data["x"], beta), dims="obs")
                       if factor_builder is None else factor_builder(data, beta))
        factor = pm.Normal("factor", mu=factor_mean, sigma=1, dims="obs")
        pm.Deterministic("factor_residual", factor - factor_mean, dims="obs")
        # PageRank is the first response in both specifications.
        anchor = pm.HalfNormal("pagerank_loading", priors["pagerank_loading_sd"])
        other = pm.Normal("other_loadings", 0, priors["other_loading_sd"], dims="other_continuous")
        lc = pm.Deterministic("loading_cont", pt.concatenate([anchor[None], other]), dims="continuous")
        lb = pm.Normal("loading_binary", 0, priors["binary_loading_sd"], dims="binary")
        ac = pm.Normal("intercept_cont", mu=priors["interior_intercept_mean"], sigma=priors["interior_intercept_sd"], dims="continuous")
        ab = pm.Normal("intercept_binary", 0, priors["binary_intercept_sd"], dims="binary")
        # Each response has its own precision; profile determines its prior median.
        phi = pm.LogNormal("precision_cont", mu=np.log(priors["precision_median"]),
                           sigma=data["precision_prior_sd"], dims="continuous")
        boundary_intercept = pm.Normal("intercept_one", priors["boundary_intercept_mean"], priors["boundary_intercept_sd"], dims="boundary")
        boundary_loading = pm.Normal("loading_one", 0, priors["boundary_loading_sd"], dims="boundary")
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


def build_chokepoint_model(data, factor_builder=None):
    import pymc as pm
    import pytensor.tensor as pt
    priors = prior_specification("chokepoint", data.get("prior_profile", "regularized"))
    coords = {"obs": np.arange(len(data["frame"])),
              "predictor": data["audit"]["predictors"],
              "continuous": data["audit"]["continuous_responses"],
              "other_continuous": data["audit"]["continuous_responses"][1:],
              "binary": data["audit"]["binary_responses"],
              "boundary": ["betweenness"]}
    with pm.Model(coords=coords) as model:
        beta = pm.Normal("beta", 0, priors["beta_sd"], dims="predictor")
        mean = (pm.Deterministic("factor_mean", pt.dot(data["x"], beta), dims="obs")
                if factor_builder is None else factor_builder(data, beta))
        factor = pm.Normal("factor", mean, 1, dims="obs")
        pm.Deterministic("factor_residual", factor - mean, dims="obs")
        anchor = pm.HalfNormal("pagerank_loading", priors["pagerank_loading_sd"])
        other = pm.Normal("other_loadings", 0, priors["other_loading_sd"], dims="other_continuous")
        lc = pm.Deterministic("loading_cont", pt.concatenate([anchor[None], other]), dims="continuous")
        lb = pm.Normal("loading_binary", 0, priors["binary_loading_sd"], dims="binary")
        ac = pm.Normal("intercept_cont", priors["interior_intercept_mean"], priors["interior_intercept_sd"], dims="continuous")
        ab = pm.Normal("intercept_binary", 0, priors["binary_intercept_sd"], dims="binary")
        phi = pm.LogNormal("precision_cont", np.log(priors["precision_median"]), data["precision_prior_sd"], dims="continuous")
        azero = pm.Normal("intercept_zero", priors["boundary_intercept_mean"], priors["boundary_intercept_sd"], dims="boundary")
        lzero = pm.Normal("loading_zero", 0, priors["boundary_loading_sd"], dims="boundary")
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


def build_model(data, factor_builder=None, rank=1):
    if rank == 2:
        if factor_builder is not None:
            raise ValueError("rank=2 with node effects is not implemented")
        return build_rank_two_model(data)
    if rank != 1:
        raise ValueError("rank must be 1 or 2")
    return (build_terminal_model(data, factor_builder)
            if "exporter_lag1" in data["audit"]["predictors"]
            else build_chokepoint_model(data, factor_builder))


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
                       "precision_cont": np.asarray(prior_specification("chokepoint", data.get("prior_profile", "regularized"))["precision_median"])*np.exp(rng.normal(0,.03,3)),
                       "intercept_zero": np.array([np.log(pzero/(1-pzero))])+rng.normal(0,.03,1),
                       "loading_zero": rng.normal(0,.05,1)})
    return starts


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
            "precision_cont": np.asarray(prior_specification("terminal", data.get("prior_profile", "regularized"))["precision_median"]) * np.exp(rng.normal(0, .03, 3)),
        })
    return starts




def build_rank_two_model(data):
    """Unconstrained second factor except the original positive PageRank first loading.

    No triangular constraints or posterior rotation. Other priors and likelihoods
    are those of the rank-one baseline. Raw factor coordinates are not identified.
    """
    import pymc as pm
    import pytensor.tensor as pt

    kind = "terminal" if "exporter_lag1" in data["audit"]["predictors"] else "chokepoint"
    priors = prior_specification(kind, data.get("prior_profile", "regularized"))
    coords = {"obs": np.arange(len(data["frame"])),
              "predictor": data["audit"]["predictors"],
              "factor_dim": ["factor_1", "factor_2"],
              "continuous": data["audit"]["continuous_responses"],
              "other_continuous": data["audit"]["continuous_responses"][1:],
              "binary": data["audit"]["binary_responses"],
              "boundary": (data["audit"]["continuous_responses"][1:]
                           if kind == "terminal" else ["betweenness"])}
    with pm.Model(coords=coords) as model:
        beta = pm.Normal("beta", 0, priors["beta_sd"], dims=("predictor", "factor_dim"))
        mean = pm.Deterministic("factor_mean", pt.dot(data["x"], beta),
                                dims=("obs", "factor_dim"))
        factor = pm.Normal("factor", mu=mean, sigma=1, dims=("obs", "factor_dim"))
        pm.Deterministic("factor_residual", factor - mean, dims=("obs", "factor_dim"))

        anchor = pm.HalfNormal("pagerank_loading", priors["pagerank_loading_sd"])
        pagerank_second = pm.Normal("pagerank_second_loading", 0, priors["pagerank_loading_sd"])
        other = pm.Normal("other_loadings", 0,
                          np.asarray(priors["other_loading_sd"])[:, None],
                          dims=("other_continuous", "factor_dim"))
        lc = pm.Deterministic("loading_cont",
                              pt.concatenate([pt.stack([anchor, pagerank_second])[None, :], other], axis=0),
                              dims=("continuous", "factor_dim"))
        lb = pm.Normal("loading_binary", 0, priors["binary_loading_sd"],
                       dims=("binary", "factor_dim"))
        ac = pm.Normal("intercept_cont", priors["interior_intercept_mean"],
                       priors["interior_intercept_sd"], dims="continuous")
        ab = pm.Normal("intercept_binary", 0, priors["binary_intercept_sd"], dims="binary")
        phi = pm.LogNormal("precision_cont", np.log(priors["precision_median"]),
                           data["precision_prior_sd"], dims="continuous")
        bc = pm.Normal("intercept_one" if kind == "terminal" else "intercept_zero",
                       priors["boundary_intercept_mean"], priors["boundary_intercept_sd"],
                       dims="boundary")
        bl = pm.Normal("loading_one" if kind == "terminal" else "loading_zero",
                       0, priors["boundary_loading_sd"], dims=("boundary", "factor_dim"))
        boundary_eta = bc + pt.dot(factor, bl.T)
        if kind == "terminal":
            logits = pt.concatenate([pt.zeros_like(boundary_eta[:, :1]), boundary_eta], axis=1)
            logp, random = one_inflated_beta_logp, one_inflated_beta_random
            inflated = np.array([False, True, True])
        else:
            logits = pt.concatenate([pt.zeros_like(boundary_eta), boundary_eta,
                                     pt.zeros_like(boundary_eta)], axis=1)
            logp, random = zero_inflated_beta_logp, zero_inflated_beta_random
            inflated = np.array([False, True, False])
        mu = pm.math.sigmoid(ac + pt.dot(factor, lc.T))
        pm.CustomDist("y_cont", mu * phi, (1 - mu) * phi, logits, inflated,
                      logp=logp, random=random,
                      observed=data["frame"][data["audit"]["continuous_responses"]].to_numpy(),
                      dims=("obs", "continuous"))
        pm.Bernoulli("y_binary", logit_p=ab + pt.dot(factor, lb.T),
                     observed=data["yb"], dims=("obs", "binary"))
    return model


def rank_two_initial_values(data, chains, seed):
    """Initialise both factors with distinct small perturbations, no rotation."""
    rng = np.random.default_rng(seed)
    kind = "terminal" if "exporter_lag1" in data["audit"]["predictors"] else "chokepoint"
    n, q = data["x"].shape
    k = len(data["audit"]["binary_responses"])
    f1 = data["yc"][:, 0]
    f2 = data["yc"][:, 1]
    factors = np.column_stack([f1, f2])
    b = np.linalg.lstsq(data["x"], factors, rcond=None)[0]
    y = data["frame"][data["audit"]["continuous_responses"]]
    interior_means = np.array([y[c].loc[y[c].between(0, 1, inclusive="neither")].mean()
                               for c in y])
    interior_means = np.clip(interior_means, .0001, .9999)
    binary_probs = np.clip(data["yb"].mean(axis=0), .01, .99)
    boundary_probs = (y.iloc[:, 1:].eq(1).mean().to_numpy() if kind == "terminal"
                      else np.array([y["betweenness"].eq(0).mean()]))
    boundary_probs = np.clip(boundary_probs, .01, .99)
    starts = []
    for _ in range(chains):
        init = {"beta": b + rng.normal(0, .03, (q, 2)),
                "factor": factors + rng.normal(0, .05, (n, 2)),
                "pagerank_loading": float(rng.uniform(.6, .8)),
                "pagerank_second_loading": float(rng.normal(0, .05)),
                "other_loadings": rng.normal(0, .05, (2, 2)),
                "loading_binary": rng.normal(0, .05, (k, 2)),
                "intercept_cont": np.log(interior_means / (1 - interior_means)) + rng.normal(0, .03, 3),
                "intercept_binary": np.log(binary_probs / (1 - binary_probs)) + rng.normal(0, .03, k),
                "precision_cont": np.asarray(prior_specification(kind, data.get("prior_profile", "regularized"))["precision_median"]) * np.exp(rng.normal(0, .03, 3)),
                "intercept_one" if kind == "terminal" else "intercept_zero":
                    np.log(boundary_probs / (1 - boundary_probs)) + rng.normal(0, .03, len(boundary_probs)),
                "loading_one" if kind == "terminal" else "loading_zero":
                    rng.normal(0, .05, (len(boundary_probs), 2))}
        starts.append(init)
    return starts
