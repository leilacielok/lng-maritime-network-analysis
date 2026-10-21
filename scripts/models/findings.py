"""Posterior summaries and latent factor scores."""
import numpy as np

def export_findings(idata, data, kind, destination, parameters, diagnostic, specs):
    import arviz as az
    summary = az.summary(idata, var_names=parameters, hdi_prob=.95)
    summary.to_csv(destination / "parameter_summary.csv")
    for chain in idata.posterior.coords["chain"].values:
        az.summary(idata.sel(chain=[chain]), var_names=parameters,
                   kind="stats", hdi_prob=.95).to_csv(
                       destination / f"parameter_summary_chain_{chain}.csv")
    scores = data["frame"][[specs[kind]["id"], "node_name", "period_month"]].copy()
    for variable in ["factor", "factor_mean"]:
        samples = idata.posterior[variable].transpose("obs", "chain", "draw").values.reshape(len(scores), -1)
        scores[f"{variable}_mean"] = samples.mean(axis=1)
        scores[f"{variable}_q025"] = np.quantile(samples, .025, axis=1)
        scores[f"{variable}_q975"] = np.quantile(samples, .975, axis=1)
    scores["passes_basic_sampling_checks"] = diagnostic["passes_basic_checks"]
    score_filename = ("factor_scores.csv" if diagnostic["passes_basic_checks"]
                      else "factor_scores_UNVALIDATED.csv")
    scores.to_csv(destination / score_filename, index=False)
