"""Export a small review ZIP from an existing bounded-model run; never refit.

Place in scripts/model. Requires the existing lng-model environment.
Posterior predictive checks are conditional, in-sample checks, not validation
on unseen nodes/months. Role groups always refer to terminal_role_lag1.
"""
import argparse
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xarray as xr
from scipy.special import expit
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def samples(ds, name, dimension):
    return ds[name].transpose('chain', 'draw', dimension).values.reshape(-1, ds.sizes[dimension])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--node-type', choices=['terminal', 'chokepoint'], default='terminal')
    parser.add_argument('--panel', type=Path)
    parser.add_argument('--ppc-draws', type=int, default=200)
    args = parser.parse_args()
    if args.ppc_draws < 20:
        parser.error('--ppc-draws must be at least 20')
    if args.panel is None:
        args.panel = Path(__file__).resolve().parents[2] / 'data/processed/model_ready' / f'{args.node_type}_criticality_model_panel.csv'
    run = args.run_dir.resolve()
    terminal = run if (run / 'posterior.nc').exists() else run / args.node_type
    nc = terminal / 'posterior.nc'
    if not nc.exists():
        raise FileNotFoundError(f'Cannot find {nc}')
    archived = terminal.parent / 'estimation_script.py'
    if not archived.exists():
        archived = Path(__file__).with_name('simple_criticality_bounded.py')
    spec = importlib.util.spec_from_file_location('archived_bounded', archived)
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    audit = json.loads((terminal / 'sample_audit.json').read_text())
    if hashlib.sha256(args.panel.read_bytes()).hexdigest() != audit['input_sha256']:
        raise ValueError('Panel differs from the original run; use the exact original panel.')
    data = model.prepare_panel(args.panel, args.node_type)
    frame = data['frame']
    masks = [('all', np.ones(len(frame), bool))]
    if args.node_type == 'terminal':
        masks += [(r, frame.terminal_role_lag1.eq(r).to_numpy()) for r in ['importer', 'exporter']]
    if data['audit']['continuous_responses'] != audit['continuous_responses'] or data['audit']['binary_responses'] != audit['binary_responses']:
        raise ValueError('Response specification differs from the archived run.')
    with xr.open_dataset(nc, group='posterior') as post:
        if len(frame) != post.sizes['obs'] or not np.array_equal(post.obs.values, np.arange(len(frame))):
            raise ValueError('Observation coordinates do not match the estimation sample.')
        out = terminal / ('review_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
        out.mkdir()
        print(f'Exporting to {out}', flush=True)
        key_columns = [model.SPECS[args.node_type]['id'], 'node_name', 'period_month']
        if args.node_type == 'terminal':
            key_columns += ['terminal_role_lag1']
        keys = frame[key_columns].copy()
        # Load one latent array at a time, rather than the whole NetCDF.
        for name in ['factor', 'factor_mean', 'factor_residual']:
            values = samples(post, name, 'obs')
            keys[name + '_mean'] = values.mean(axis=0)
            keys[name + '_q025'] = np.quantile(values, .025, axis=0)
            keys[name + '_q975'] = np.quantile(values, .975, axis=0)
            del values
        keys.to_csv(out / 'latent_summary.csv', index=False)
        associations = []
        for role, mask in masks:
            for response in audit['continuous_responses'] + audit['binary_responses']:
                a = keys.loc[mask, 'factor_mean'].to_numpy()
                b = frame.loc[mask, response].to_numpy()
                rho = np.corrcoef(rankdata(a), rankdata(b))[0, 1] if np.std(b) else np.nan
                associations.append(dict(role_lag1=role, response=response, spearman_with_posterior_factor_mean=rho))
        pd.DataFrame(associations).to_csv(out / 'factor_associations.csv', index=False)
        # Response functions use paired posterior draws, preserving dependence.
        ac = samples(post, 'intercept_cont', 'continuous')
        lc = samples(post, 'loading_cont', 'continuous')
        boundary_suffix = 'one' if args.node_type == 'terminal' else 'zero'
        ao = samples(post, 'intercept_' + boundary_suffix, 'boundary')
        lo = samples(post, 'loading_' + boundary_suffix, 'boundary')
        ab = samples(post, 'intercept_binary', 'binary')
        lb = samples(post, 'loading_binary', 'binary')
        grid = np.linspace(keys.factor_mean.quantile(.01), keys.factor_mean.quantile(.99), 101)
        rows = []
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        for j, response in enumerate(audit['continuous_responses']):
            mu = expit(ac[:, j, None] + lc[:, j, None] * grid)
            if args.node_type == 'terminal':
                p = np.zeros_like(mu) if j == 0 else expit(ao[:, j-1, None] + lo[:, j-1, None] * grid)
                expected = p + (1-p)*mu
            else:
                p = expit(ao[:, 0, None] + lo[:, 0, None] * grid) if response == 'betweenness' else np.zeros_like(mu)
                expected = (1-p)*mu
            for quantity, v in [('interior_mean', mu), ('probability_' + boundary_suffix, p), ('marginal_mean', expected)]:
                low, mid, high = np.quantile(v, [.025, .5, .975], axis=0)
                rows.extend(dict(response=response, quantity=quantity, factor=f, median=m, q025=l, q975=h) for f,m,l,h in zip(grid, mid,low,high))
                if quantity == 'marginal_mean':
                    axes.flat[j].plot(grid, mid)
                    axes.flat[j].fill_between(grid, low, high, alpha=.25)
            axes.flat[j].set_title(response, fontsize=9)
        for j, response in enumerate(audit['binary_responses']):
            v = expit(ab[:, j, None] + lb[:, j, None] * grid)
            low, mid, high = np.quantile(v, [.025, .5, .975], axis=0)
            rows.extend(dict(response=response, quantity='probability', factor=f, median=m, q025=l, q975=h) for f,m,l,h in zip(grid,mid,low,high))
            axes.flat[j+3].plot(grid, mid)
            axes.flat[j+3].fill_between(grid, low, high, alpha=.25)
            axes.flat[j+3].set_title(response, fontsize=9)
        for ax in list(axes.flat)[3 + len(audit['binary_responses']):]:
            ax.set_visible(False)
        for ax in axes.flat:
            ax.set_xlabel('Latent factor')
            ax.set_ylabel('Expected response / probability')
        fig.tight_layout()
        fig.savefig(out / 'response_curves.png', dpi=130)
        plt.close(fig)
        pd.DataFrame(rows).to_csv(out / 'response_curves.csv', index=False)
    print('Checking replicated response associations...', flush=True)
    with xr.open_dataset(nc, group='posterior_predictive') as ppc:
        # Equal spacing within every chain; only selected predictive draws load.
        per_chain = max(1, args.ppc_draws // ppc.sizes['chain'])
        chosen = np.unique(np.linspace(0, ppc.sizes['draw']-1, min(per_chain, ppc.sizes['draw']), dtype=int))
        small = ppc.isel(draw=chosen).load()
        actual_draws = small.sizes['chain'] * small.sizes['draw']
    observed = frame[audit['continuous_responses'] + audit['binary_responses']].to_numpy()
    rep = np.concatenate([small.y_cont.transpose('chain','draw','obs','continuous').values, small.y_binary.transpose('chain','draw','obs','binary').values], axis=-1).reshape(actual_draws, len(frame), -1)
    names = audit['continuous_responses'] + audit['binary_responses']
    rows = []
    for role, mask in masks:
        obs_corr = pd.DataFrame(observed[mask]).corr(method='spearman').to_numpy()
        corr = np.array([pd.DataFrame(v[mask]).corr(method='spearman').to_numpy() for v in rep])
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                valid = corr[:,i,j][np.isfinite(corr[:,i,j])]
                low, mid, high = np.quantile(valid, [.025,.5,.975]) if len(valid) else [np.nan]*3
                rows.append(dict(role_lag1=role, response_a=names[i], response_b=names[j], observed=obs_corr[i,j], replicated_median=mid, replicated_q025=low, replicated_q975=high, valid_replicates=len(valid)))
    pd.DataFrame(rows).to_csv(out / 'joint_predictive_checks.csv', index=False)
    for name in ['parameter_summary.csv','sampling_diagnostics.json','posterior_checks.csv','sample_audit.json','predictor_scaling.csv']:
        if (terminal / name).exists():
            shutil.copy2(terminal / name, out / name)
    (out / 'README.txt').write_text('Existing posterior only: no refitting.\nPredictive checks are conditional and in-sample, not out-of-sample validation.\nTerminal role groups use LAGGED role; chokepoints use the full sample.\nFactor associations use posterior mean scores informed by these same responses; they are descriptive, not independent validation or causal effects.\nCurve bands are equal-tailed 95% parameter uncertainty, not predictive intervals.\nCurves span the 1st-99th percentiles of posterior mean latent scores.\nNode/month scores are conditional on consecutive activity (and available age for terminals); interpret alongside sampling diagnostics and model limitations.\n', encoding='utf-8')
    (out / 'export_metadata.json').write_text(json.dumps(dict(source=str(nc), node_type=args.node_type, predictive_replicates=actual_draws, panel_sha256=audit['input_sha256']), indent=2))
    archive = shutil.make_archive(str(out), 'zip', out)
    print(f'Done. Send this small ZIP: {archive}', flush=True)


if __name__ == '__main__':
    main()
