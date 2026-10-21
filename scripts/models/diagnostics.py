"""Sampling diagnostics, trace plots and predictive checks."""
import json
import warnings
import numpy as np
import pandas as pd

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


def run_diagnostics(idata, data, kind, destination, parameters):
    import arviz as az
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
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
    predictive_checks(idata, data, "posterior_predictive").to_csv(destination / "posterior_checks.csv", index=False)
    with az.rc_context({"plot.max_subplots": 100}):
        az.plot_trace(idata, var_names=parameters, compact=False)
    plt.savefig(destination / "trace.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    return diagnostic
