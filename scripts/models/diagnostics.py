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
    save_classical_diagnostics(idata.posterior, destination, parameters)
    return diagnostic

# Additional single-chain diagnostics. Python implementation of the procedures
# documented in CRAN coda (geweke.diag, heidel.diag, spectrum0.ar).
# The AR spectrum is selected by AIC using Yule-Walker estimates; numerical
# results need not be identical to R/coda. No thinning or chain concatenation.
def _spectrum0_ar(values):
    """Long-run variance from an AIC-selected autoregression (orders include 0)."""
    from scipy.signal import correlate
    x = np.asarray(values, dtype=float)
    x = x - x.mean()
    n = len(x)
    order_max = min(int(10 * np.log10(n)), n - 2)
    acov = correlate(x, x, mode="full", method="fft")[n-1:n+order_max] / n
    if acov[0] <= 0:
        return np.nan
    coefficients = np.empty(0)
    variance = acov[0]
    best_aic = n * np.log(variance)
    best_spectrum = variance
    for order in range(1, order_max + 1):
        reflection = (acov[order] - np.dot(coefficients, acov[1:order][::-1])) / variance
        if not np.isfinite(reflection) or abs(reflection) >= 1:
            break
        coefficients = np.r_[coefficients - reflection * coefficients[::-1], reflection]
        variance *= 1 - reflection**2
        aic = n * np.log(variance) + 2 * order
        if aic < best_aic:
            best_aic = aic
            best_spectrum = variance / (1 - coefficients.sum())**2
    return float(best_spectrum)


def _bridge_cdf(statistic):
    """Asymptotic Cramer-von Mises CDF, evaluated by its Bessel series."""
    from scipy.special import gammaln, kv
    if statistic <= 0:
        return 0.0
    if statistic >= 4:
        return 1.0  # upper tail is negligible at the diagnostic's 5% threshold
    total = 0.0
    for k in range(100):
        y = 4 * k + 1
        q = y*y / (16 * statistic)
        term = (np.exp(gammaln(k+.5) - gammaln(k+1) - q)
                * np.sqrt(y) * kv(.25, q) / (np.pi**1.5 * np.sqrt(statistic)))
        total += term
        if term < 1e-12:
            break
    return float(np.clip(total, 0, 1))


def _single_chain_tests(x, alpha=.05, eps=.1):
    from scipy.stats import norm
    x = np.asarray(x, dtype=float)
    n = len(x)
    g = dict(n_draws=n, first_fraction=.1, last_fraction=.5,
             z_score=np.nan, p_value=np.nan, passed=None, status="ok")
    h = dict(n_draws=n, stationarity_passed=None, stationarity_p_value=np.nan,
             discarded_draws=None, retained_start_position=None, retained_draws=None,
             retained_mean=np.nan, half_width=np.nan, relative_half_width=np.nan,
             half_width_passed=None, status="ok")
    if n < 100 or not np.isfinite(x).all() or np.ptp(x) == 0:
        status = "too_short" if n < 100 else ("nonfinite" if not np.isfinite(x).all() else "constant")
        g["status"] = h["status"] = status
        return g, h
    a, b = x[:int(.1*n)], x[n-int(.5*n):]
    va, vb = _spectrum0_ar(a), _spectrum0_ar(b)
    se2 = va / len(a) + vb / len(b)
    if np.isfinite(se2) and se2 > 0:
        z = (a.mean() - b.mean()) / np.sqrt(se2)
        p = 2 * norm.sf(abs(z))
        g.update(z_score=float(z), p_value=float(p), passed=bool(p >= alpha))
    else:
        g["status"] = "invalid_spectrum"
    # As in coda, stationarity uses a fixed spectrum from the last half.
    s0 = _spectrum0_ar(x[n//2:])
    if not np.isfinite(s0) or s0 <= 0:
        h["status"] = "invalid_spectrum"
        return g, h
    for fraction in np.arange(6) / 10:
        discarded = int(fraction * n)
        y = x[discarded:]
        m = len(y)
        bridge = np.cumsum(y - y.mean())
        statistic = np.sum(bridge**2) / (m*m*s0)
        p = 1 - _bridge_cdf(statistic)
        h.update(stationarity_p_value=float(p), stationarity_passed=bool(p > alpha))
        if p > alpha:
            s_ci = _spectrum0_ar(y)
            half = norm.ppf(.975) * np.sqrt(s_ci / m)
            if not np.isfinite(half) or half <= 0:
                h["status"] = "invalid_spectrum"
                return g, h
            mean = float(y.mean())
            relative = half / abs(mean) if mean != 0 else np.inf
            h.update(discarded_draws=discarded, retained_start_position=discarded+1,
                     retained_draws=m, retained_mean=mean, half_width=float(half),
                     relative_half_width=float(relative), half_width_passed=bool(relative <= eps))
            return g, h
    h["status"] = "stationarity_failed"
    return g, h


def save_classical_diagnostics(posterior, destination, parameters=None, alpha=.05, eps=.1):
    """Save per-coordinate, per-chain results for explicitly selected variables.

    Defaults to global parameters, excluding observation/node effects and
    large deterministic arrays. Pass parameters explicitly to change scope.
    Missing variables and invalid chains are recorded, never silently passed.
    Half-width is a 95% interval and eps is its target relative precision.
    P-values are unadjusted: interpret with multiplicity and R-hat/ESS.
    """
    from pathlib import Path
    if not 0 < alpha < 1 or eps <= 0:
        raise ValueError("Require 0 < alpha < 1 and eps > 0")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if parameters is None:
        parameters = [name for name, da in posterior.data_vars.items()
                      if 'chain' in da.dims and 'draw' in da.dims
                      and not any(d in da.dims for d in ('obs', 'node'))
                      and name not in ('factor', 'factor_mean', 'factor_residual', 'mu_node')]
    parameters = list(dict.fromkeys(parameters))
    if not parameters:
        raise ValueError("No parameters selected")
    missing = [name for name in parameters if name not in posterior]
    if missing:
        raise ValueError(f"Parameters missing from posterior: {missing}")
    grows, hrows = [], []
    for name in parameters:
        da = posterior[name]
        if 'chain' not in da.dims or 'draw' not in da.dims:
            raise ValueError(f"{name} must have chain and draw dimensions")
        extra = [dim for dim in da.dims if dim not in ('chain', 'draw')]
        values = da.transpose('chain', 'draw', *extra).values
        for index in np.ndindex(values.shape[2:]):
            coords = {dim: str(da.coords[dim].values[i]) if dim in da.coords else str(i)
                      for dim, i in zip(extra, index)}
            for chain in range(values.shape[0]):
                key = dict(variable=name, coordinates=json.dumps(coords, sort_keys=True),
                           chain=str(da.coords['chain'].values[chain]), alpha=alpha)
                g, h = _single_chain_tests(values[(chain, slice(None)) + index], alpha, eps)
                grows.append(dict(key, **g))
                hrows.append(dict(key, eps=eps, **h))
    pd.DataFrame(grows).to_csv(destination / 'geweke_diagnostics.csv', index=False)
    pd.DataFrame(hrows).to_csv(destination / 'heidelberger_welch_diagnostics.csv', index=False)
    metadata = dict(parameters=parameters, rows=len(grows), alpha=alpha, eps=eps,
                    geweke_segments=[.1, .5], half_width_confidence=.95,
                    spectrum='AIC-selected Yule-Walker autoregression',
                    scope='Selected variables only; each coordinate and chain separately',
                    warmup='Posterior group only; warm-up excluded',
                    notes=['Python implementation; not guaranteed numerically identical to R coda',
                           'Unadjusted p-values; interpret alongside R-hat, ESS and divergences',
                           'Relative half-width is unstable for means near zero',
                           'HW discarded draws affect this diagnostic only, not saved posterior'])
    (destination / 'classical_diagnostics_settings.json').write_text(
        json.dumps(metadata, indent=2), encoding='utf-8')
    return metadata


def export_sampling_diagnostics(post, nc, out):
    """Recompute diagnostics and divergence plots from the saved run."""
    import arviz as az
    import xarray as xr
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mixing = az.summary(post, kind='diagnostics')
    mixing.to_csv(out / 'mixing_diagnostics.csv')
    az.summary(post).to_csv(out / 'parameter_summary.csv')
    for chain in post.chain.values:
        az.summary(post.sel(chain=[chain]), kind='stats').to_csv(
            out / f'parameter_summary_chain_{chain}.csv')
    try:
        stats = xr.open_dataset(nc, group='sample_stats')
    except (OSError, ValueError, KeyError) as exc:
        (out / 'divergence_plots_status.json').write_text(json.dumps(
            dict(status='unavailable', reason=f'Cannot read sample_stats: {exc}'), indent=2))
        return
    with stats:
        if 'diverging' not in stats:
            (out / 'divergence_plots_status.json').write_text(json.dumps(
                dict(status='unavailable', reason='sample_stats.diverging is missing'), indent=2))
            return
        # Exact alignment prevents associating a divergence with the wrong draw.
        post, stats = xr.align(post, stats, join='exact')
        diverging = stats.diverging.transpose('chain', 'draw').astype(bool)
        counts = diverging.sum('draw')
        pd.DataFrame({'chain':stats.chain.values, 'draws':stats.sizes['draw'],
                      'divergences':counts.values,
                      'divergence_rate':counts.values/stats.sizes['draw']}).to_csv(
                          out / 'divergences_by_chain.csv', index=False)
        report = dict(divergences=int(diverging.sum()), max_rhat=float(mixing.r_hat.max()),
                      min_ess_bulk=float(mixing.ess_bulk.min()), min_ess_tail=float(mixing.ess_tail.min()),
                      nonfinite_diagnostic_entries=int((~np.isfinite(
                          mixing[['r_hat','ess_bulk','ess_tail']].to_numpy())).sum()))
        report['passes_basic_checks'] = bool(report['divergences']==0 and
            report['nonfinite_diagnostic_entries']==0 and report['max_rhat']<=1.01 and
            report['min_ess_bulk']>=400 and report['min_ess_tail']>=400)
        (out / 'sampling_diagnostics.json').write_text(json.dumps(report, indent=2))
        fig, ax = plt.subplots(figsize=(8,3.5))
        ax.bar([str(c) for c in stats.chain.values], counts.values, color='#bb625b')
        ax.set(xlabel='Chain', ylabel='Divergent transitions', title='Post-warm-up divergences by chain')
        fig.tight_layout(); fig.savefig(out / 'divergences_by_chain.png', dpi=160); plt.close(fig)
        n_divergent = report['divergences']
        mask = diverging.values.ravel()
        n_regular = int((~mask).sum())
        # Remove stale figures if this helper is called again in the same folder.
        for filename in ('divergences_parallel.png', 'divergences_pair.png'):
            (out / filename).unlink(missing_ok=True)
        status = dict(divergences=n_divergent, nondivergent_draws=n_regular,
                      selected_parameters=[],
                      selection_method='Descending two-sample Kolmogorov-Smirnov distance',
                      candidate_scope='Global parameters and node effects; observation-level arrays excluded',
                      note='Exploratory marginal association, not a cause or attribution of divergences; no hypothesis-test p-values.')
        columns = ['rank','parameter','variable','coordinates','ks_distance',
                   'standardized_mean_difference','selected']
        if n_divergent == 0 or n_regular == 0:
            status.update(status='skipped_no_divergences' if n_divergent == 0 else 'skipped_no_nondivergent_draws',
                          warning=None if n_divergent == 0 else 'No comparison group is available.')
            pd.DataFrame(columns=columns).to_csv(out / 'divergence_plot_selection.csv', index=False)
            (out / 'divergence_plots_status.json').write_text(json.dumps(status, indent=2))
            return
        from scipy.stats import ks_2samp
        import hashlib
        ranked = []
        seen = set()
        # Prefer complete, canonical loading arrays to their component aliases.
        variables = list(post.data_vars)
        variables.sort(key=lambda name: (name in ('pagerank_loading','other_loadings'), name))
        for family in variables:
            da = post[family]
            if not {'chain','draw'}.issubset(da.dims):
                continue
            dims = [d for d in da.dims if d not in ('chain','draw')]
            if any(d in ('obs','observation','obs_id') for d in dims) or family in (
                    'factor','factor_mean','factor_residual'):
                continue
            for idx in np.ndindex(tuple(da.sizes[d] for d in dims)):
                values = np.asarray(da.isel(dict(zip(dims,idx))).transpose('chain','draw').values, dtype=float)
                if not np.isfinite(values).all() or np.ptp(values)==0:
                    continue
                digest = hashlib.sha256(np.ascontiguousarray(values).tobytes()).digest()
                if digest in seen:
                    continue
                seen.add(digest)
                flat = values.ravel()
                score = float(ks_2samp(flat[mask], flat[~mask], method='asymp').statistic)
                difference = float((flat[mask].mean()-flat[~mask].mean())/flat.std())
                coords = {d:str(da[d].values[i]) if d in da.coords else str(i) for d,i in zip(dims,idx)}
                name = family + (' ['+', '.join(coords.values())+']' if coords else '')
                ranked.append(dict(parameter=name, variable=family, coordinates=json.dumps(coords),
                                   ks_distance=score, standardized_mean_difference=difference,
                                   index=idx, dimensions=dims))
        ranked.sort(key=lambda row: (-row['ks_distance'], row['parameter']))
        selected = {}
        selection = []
        for rank, row in enumerate(ranked, 1):
            include = rank <= 8
            if include:
                da = post[row['variable']].isel(dict(zip(row['dimensions'], row['index'])))
                selected[row['parameter']] = xr.DataArray(da.transpose('chain','draw').values,
                    dims=('chain','draw'), coords={'chain':post.chain,'draw':post.draw})
            selection.append({**{k:v for k,v in row.items() if k not in ('index','dimensions')},
                              'rank':rank, 'selected':include})
        pd.DataFrame(selection, columns=columns).to_csv(out / 'divergence_plot_selection.csv', index=False)
        status.update(selected_parameters=list(selected), candidate_count=len(ranked),
                      warning='Exploratory ranking; uncertainty is greater when divergent draws are few. No count threshold is applied.')
        if len(selected)>=2:
            idata = az.InferenceData(posterior=xr.Dataset(selected), sample_stats=stats.load())
            axes = az.plot_parallel(idata, var_names=list(selected), norm_method='normal',
                                    show=False, figsize=(12,5))
            fig = np.asarray(axes,dtype=object).flat[0].figure
            fig.subplots_adjust(bottom=.35)
            for ax in fig.axes:
                plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=8)
            fig.savefig(out / 'divergences_parallel.png', dpi=160, bbox_inches='tight'); plt.close(fig)
            # Pair the two highest-ranked marginal associations.
            scale, effect = list(selected)[:2]
            axes = az.plot_pair(idata, var_names=[scale,effect], kind='scatter',
                                divergences=True, marginals=True, show=False, figsize=(7,6))
            fig = next(ax for ax in np.asarray(axes,dtype=object).flat if ax is not None).figure
            fig.savefig(out / 'divergences_pair.png', dpi=160, bbox_inches='tight'); plt.close(fig)
        status['status'] = 'created' if len(selected)>=2 else 'skipped_insufficient_parameters'
        (out / 'divergence_plots_status.json').write_text(json.dumps(status, indent=2))
