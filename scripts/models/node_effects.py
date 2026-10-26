"""Rank-one model with persistent, hierarchical node intercepts.

Run from the repository root:
    python scripts/models/node_effects.py --node-type terminal --check-only
    python scripts/models/node_effects.py --node-type terminal --prior-only

Uses estimation.py and the baseline likelihoods; no additional latent rank.
mu_raw ~ Normal(0,1); sigma_mu ~ HalfNormal(node_sd_prior);
mu_node = sigma_mu * mu_raw; factor ~ Normal(X beta + mu_node[node_index], 1).
The population mean of node effects is fixed to zero; the realised node effects
are not constrained to sum to zero. This is an independent random-effects
specification: it does not model dependence between mu_node and covariates.
"""
import numpy as np
import pandas as pd
from common import SPECS


def prepare_node_effects(data, kind, prior_scale=1.0):
    """Construct a deterministic ID mapping on the filtered estimation sample."""
    if not np.isfinite(prior_scale) or prior_scale <= 0:
        raise ValueError("Node-effect prior scale must be finite and positive")
    id_column = SPECS[kind]["id"]
    frame = data["frame"]
    ids = frame[id_column].astype("string")
    if ids.isna().any():
        raise ValueError("Missing node identifiers")
    node_ids = sorted(ids.unique().tolist())
    node_index = pd.Index(node_ids).get_indexer(ids).astype("int64")
    if (node_index < 0).any():
        raise ValueError("Unmapped node identifier")
    labels = frame.assign(_node_id=ids).groupby("_node_id")["node_name"].first()
    data.update(node_ids=node_ids, node_index=node_index,
                node_names=labels.reindex(node_ids).fillna("").astype(str).tolist(),
                node_sd_prior=float(prior_scale))
    data["audit"]["node_effect_specification"] = {
        "factor": "Normal(X beta + mu_node[node_index], 1)",
        "mu_raw": "Normal(0, 1)",
        "mu_node": "sigma_mu * mu_raw (non-centred)",
        "sigma_mu": {"distribution": "HalfNormal", "scale": float(prior_scale)},
        "population_mean": 0.0,
        "sum_to_zero_constraint": False,
        "estimation_nodes": len(node_ids),
        "node_id_column": id_column,
        "node_id_order": node_ids,
    }
    return data


def build_factor_mean(data, beta):
    """Called inside the same pm.Model context as the baseline likelihoods."""
    import pymc as pm
    import pytensor.tensor as pt
    model = pm.modelcontext(None)
    model.add_coord("node", values=data["node_ids"])
    sigma_mu = pm.HalfNormal("sigma_mu", sigma=data["node_sd_prior"])
    mu_raw = pm.Normal("mu_raw", mu=0, sigma=1, dims="node")
    mu_node = pm.Deterministic("mu_node", sigma_mu * mu_raw, dims="node")
    covariate_mean = pm.Deterministic("factor_covariate_mean", pt.dot(data["x"], beta), dims="obs")
    return pm.Deterministic("factor_mean", covariate_mean + mu_node[data["node_index"]], dims="obs")


if __name__ == "__main__":
    from estimation import main
    main(default_model="node_effects")
