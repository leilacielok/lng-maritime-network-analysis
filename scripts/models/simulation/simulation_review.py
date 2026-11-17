"""Full simulation diagnostics, posterior summaries, parameter recovery and PPC.

Run from thesis_code:
 python scripts/models/simulation/simulation_review.py --dataset outputs/simulation/terminal_s1 --output-name run_5000_10000

Three variants are reviewed sequentially. No model refitting is performed.
"""
import argparse
import gc
import json
import shutil
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from simulation_utils import load_dataset, load_truth, save_json, sigmoid
from rank_two_variants import VARIANTS

EXCLUDE_GLOBAL = {'factor', 'factor_mean', 'factor_residual', 'factor_covariate_mean'}
TRUTH_MAP = {'beta': 'beta', 'loading_cont': 'loading_cont',
             'loading_binary': 'loading_binary', 'loading_one': 'loading_boundary',
             'loading_zero': 'loading_boundary', 'intercept_cont': 'intercept_cont',
             'intercept_binary': 'intercept_binary', 'intercept_one': 'intercept_boundary',
             'intercept_zero': 'intercept_boundary', 'precision_cont': 'precision_cont'}


def global_variables(post):
    return [name for name in post.data_vars if name not in EXCLUDE_GLOBAL
            and 'obs' not in post[name].dims and name not in
            ('pagerank_loading', 'pagerank_second_loading', 'other_loadings',
             'other_first_loadings', 'second_diagonal_loading', 'remaining_second_loading')]


def scalar_dataset(post, names):
    """Flatten parameter coordinates into scalar traces, preserving chain and draw."""
    variables = {}
    for name in names:
        da = post[name]
        dims = [d for d in da.dims if d not in ('chain', 'draw')]
        for idx in np.ndindex(*(da.sizes[d] for d in dims)):
            coords = [str(da.coords[d].values[i]) if d in da.coords else str(i)
                      for d, i in zip(dims, idx)]
            label = name + (' [' + ', '.join(coords) + ']' if coords else '')
            variables[label] = xr.DataArray(
                da.isel(dict(zip(dims, idx))).transpose('chain', 'draw').values,
                dims=('chain', 'draw')
            )
    return xr.Dataset(variables)


def plot_trace_acf(post, out, names, acf_count=8):
    import arviz as az
    scalar = scalar_dataset(post, names)
    labels = list(scalar.data_vars)
    for start in range(0, len(labels), 4):
        selected = labels[start:start+4]
        with az.rc_context({'plot.max_subplots': 100}):
            axes = az.plot_trace(scalar[selected], compact=False,
                                 figsize=(12, 2.5*len(selected)), show=False)
        fig = np.asarray(axes).ravel()[0].figure
        fig.tight_layout()
        fig.savefig(out / f'trace_{start//4+1:02d}.png', dpi=145)
        plt.close(fig)
    diag = az.summary(scalar, kind='diagnostics', round_to=6)
    selection = list(dict.fromkeys(
        diag['ess_bulk'].nsmallest(acf_count).index.tolist() +
        diag['ess_tail'].nsmallest(3).index.tolist()))
    diag.loc[selection].to_csv(out / 'acf_selection.csv')
    for i, name in enumerate(selection, 1):
        values = scalar[name].values
        lag = min(100, values.shape[1]-1)
        fig, ax = plt.subplots(figsize=(8, 3.4))
        for chain in range(values.shape[0]):
            ax.plot(np.arange(lag+1), az.autocorr(values[chain])[:lag+1],
                    label=f'Chain {chain+1}', lw=1.1)
        ax.axhline(0, color='grey', lw=.6)
        ax.set(xlabel='Lag', ylabel='Autocorrelation', title=name, xlim=(0, lag))
        ax.legend(fontsize=8, ncol=min(4, len(values)))
        fig.tight_layout()
        fig.savefig(out / f'acf_{i:02d}.png', dpi=150)
        plt.close(fig)
    del scalar


def sample_diagnostics(idata, out, global_names, include_local=True):
    import arviz as az
    post = idata.posterior
    stats = idata.sample_stats
    # Global summaries and chain-specific summaries, as in the LNG review.
    az.summary(post, var_names=global_names, round_to=6).to_csv(
        out / 'parameter_summary.csv', float_format='%.3f')
    az.summary(post, var_names=global_names, kind='diagnostics', round_to=6).to_csv(
        out / 'mixing_diagnostics.csv', float_format='%.3f')
    for chain in post.chain.values:
        az.summary(post.sel(chain=[chain]), var_names=global_names,
                   kind='stats', round_to=6).to_csv(
            out / f'parameter_summary_chain_{int(chain)+1:02d}.csv', float_format='%.3f')
    diag = az.summary(post, var_names=global_names, kind='diagnostics', round_to=6)
    local = None
    if include_local and 'factor' in post:
        local = az.summary(post, var_names=['factor'], kind='diagnostics', round_to=6)
        local.to_csv(out / 'latent_factor_mixing_diagnostics.csv', float_format='%.3f')
    diverging = stats['diverging'].transpose('chain', 'draw').values.astype(bool)
    div_by_chain = diverging.sum(axis=1)
    pd.DataFrame({'chain': np.arange(1, len(div_by_chain)+1),
                  'draws': diverging.shape[1], 'divergences': div_by_chain,
                  'divergence_rate': div_by_chain / diverging.shape[1]}).to_csv(
                      out / 'divergences_by_chain.csv', index=False)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(np.arange(1, len(div_by_chain)+1), div_by_chain)
    ax.set(xlabel='Chain', ylabel='Divergences', title='Post-warm-up divergences')
    fig.tight_layout(); fig.savefig(out / 'divergences_by_chain.png', dpi=150); plt.close(fig)
    bfmi = az.bfmi(idata) if 'energy' in stats else np.full(len(div_by_chain), np.nan)
    pd.DataFrame({'chain': np.arange(1, len(div_by_chain)+1), 'bfmi': bfmi}).to_csv(
        out / 'bfmi_by_chain.csv', index=False)
    sampler_cols = [k for k in ('step_size', 'step_size_bar', 'n_steps', 'tree_depth',
                                 'acceptance_rate', 'energy', 'lp') if k in stats]
    if sampler_cols:
        rows = []
        for c in range(stats.sizes['chain']):
            row = {'chain': c+1}
            for key in sampler_cols:
                v = stats[key].isel(chain=c).values
                row[key+'_mean'] = float(np.nanmean(v))
                row[key+'_median'] = float(np.nanmedian(v))
            rows.append(row)
        pd.DataFrame(rows).to_csv(out / 'nuts_statistics_by_chain.csv', index=False)
    report = {'divergences': int(diverging.sum()),
              'divergence_pct': float(100*diverging.mean()),
              'max_rhat_global': float(diag.r_hat.max()),
              'min_ess_bulk_global': float(diag.ess_bulk.min()),
              'min_ess_tail_global': float(diag.ess_tail.min()),
              'bfmi_by_chain': np.asarray(bfmi).tolist(),
              'nonfinite_global_diagnostics': int((~np.isfinite(
                  diag[['r_hat', 'ess_bulk', 'ess_tail']].to_numpy())).sum())}
    if local is not None:
        report.update(max_rhat_factor=float(local.r_hat.max()),
                      min_ess_bulk_factor=float(local.ess_bulk.min()),
                      min_ess_tail_factor=float(local.ess_tail.min()))
    report['passes_basic_checks'] = bool(
        report['divergences'] == 0 and report['nonfinite_global_diagnostics'] == 0
        and report['max_rhat_global'] <= 1.01
        and report['min_ess_bulk_global'] >= 400
        and report['min_ess_tail_global'] >= 400)
    save_json(out / 'sampling_diagnostics.json', report)
    return report


def classical_diagnostics(post, out, global_names):
    # Reuse the same implementation used by the user's production review.py.
    from diagnostics import save_classical_diagnostics
    save_classical_diagnostics(post, out, parameters=global_names)
    g = pd.read_csv(out / 'geweke_diagnostics.csv')
    h = pd.read_csv(out / 'heidelberger_welch_diagnostics.csv')
    summary = {'geweke_tests': len(g),
               'geweke_passes': int((g['z_score'].abs() <= 1.96).sum())}
    for col in ('stationarity_passed', 'half_width_passed'):
        if col in h:
            summary[col+'_count'] = int(h[col].fillna(False).astype(bool).sum())
    save_json(out / 'classical_diagnostics_summary.json', summary)
    for filename, column, title in [('geweke_diagnostics.csv', 'z_score', 'Geweke Z'),
                                     ('heidelberger_welch_diagnostics.csv', 'stationarity_passed',
                                      'Heidelberger–Welch stationarity')]:
        table = pd.read_csv(out / filename)
        if column == 'z_score':
            fig, ax = plt.subplots(figsize=(10, 4))
            for chain, group in table.groupby('chain'):
                ax.scatter(np.arange(len(group)), group[column], s=7, alpha=.65,
                           label=f'Chain {chain}')
            ax.axhline(1.96, color='red', ls='--', lw=.8)
            ax.axhline(-1.96, color='red', ls='--', lw=.8)
            ax.set(xlabel='Parameter coordinate (within chain)', ylabel='Z-score', title=title)
            ax.legend(fontsize=8)
        else:
            fig, ax = plt.subplots(figsize=(8, 3.5))
            passed = table[column].fillna(False).astype(bool)
            ax.bar(['Pass', 'Fail'], [int(passed.sum()), int((~passed).sum())])
            ax.set(ylabel='Parameter–chain tests', title=title)
        fig.tight_layout()
        fig.savefig(out / filename.replace('.csv', '.png'), dpi=150)
        plt.close(fig)


def recovery(post, truth, meta, out, max_gram_draws=4000):
    import arviz as az
    rows = []
    for name, key in TRUTH_MAP.items():
        if name not in post or key not in truth:
            continue
        da = post[name]
        actual = np.asarray(truth[key])
        # Loadings are NOT individually identified in the unconstrained model.
        values = da.transpose('chain', 'draw', *[d for d in da.dims if d not in ('chain','draw')]).values
        if values.shape[2:] != actual.shape:
            raise ValueError(f'Truth shape mismatch for {name}: {values.shape[2:]} != {actual.shape}')
        mean = values.mean(axis=(0,1))
        lo, hi = np.quantile(values, [.025, .975], axis=(0,1))
        dims = [d for d in da.dims if d not in ('chain','draw')]
        for idx in np.ndindex(actual.shape):
            coords = {d: str(da.coords[d].values[i]) for d, i in zip(dims, idx)}
            rows.append({'parameter': name, 'coordinates': json.dumps(coords),
                         'true': float(actual[idx]), 'posterior_mean': float(mean[idx]),
                         'q025': float(lo[idx]), 'q975': float(hi[idx]),
                         'covered_95pct': bool(lo[idx] <= actual[idx] <= hi[idx]),
                         'absolute_error': float(abs(mean[idx]-actual[idx])),
                         'rotation_sensitive': name.startswith('loading_') or name == 'beta'})
    table = pd.DataFrame(rows)
    table.to_csv(out / 'parameter_recovery.csv', index=False)
    for family, subset in table.groupby('parameter', sort=False):
        fig, ax = plt.subplots(figsize=(9, max(3.2, .33*len(subset)+1.5)))
        y = np.arange(len(subset))
        ax.errorbar(subset.posterior_mean, y,
                    xerr=np.vstack([subset.posterior_mean-subset.q025,
                                    subset.q975-subset.posterior_mean]), fmt='o', capsize=2,
                    label='Posterior mean and 95% interval')
        ax.scatter(subset['true'], y, marker='x', s=35, label='True value')
        ax.set_yticks(y, subset.coordinates.to_list(), fontsize=7)
        ax.set_title(f'{family} — recovery' + (' (orientation-dependent)' if subset.rotation_sensitive.any() else ''))
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f'recovery_{family}.png', dpi=150)
        plt.close(fig)
    # Invariant Gram matrix, sampled using evenly spaced paired draws per chain.
    boundary = 'loading_one' if meta['kind'] == 'terminal' else 'loading_zero'
    n_chain, n_draw = post.sizes['chain'], post.sizes['draw']
    sel = np.unique(np.linspace(0, n_draw-1, min(n_draw, max(1, max_gram_draws//n_chain)), dtype=int))
    names = ['loading_cont', 'loading_binary', boundary]
    loading = np.concatenate([post[name].isel(draw=sel).values for name in names], axis=-2)
    true_loading = np.vstack([truth['loading_cont'], truth['loading_binary'], truth['loading_boundary']])
    gram_true = true_loading @ true_loading.T
    gram = loading @ np.swapaxes(loading, -1, -2)
    gram_mean = gram.mean(axis=(0,1))
    lo, hi = np.quantile(gram, [.025, .975], axis=(0,1))
    response_names = meta['continuous_responses'] + meta['binary_responses'] + (
        [name + '_boundary' for name in meta['continuous_responses'][1:]] if meta['kind'] == 'terminal' else ['betweenness_boundary'])
    grow = []
    for i in range(len(response_names)):
        for j in range(len(response_names)):
            grow.append({'response_i': response_names[i], 'response_j': response_names[j],
                         'true': gram_true[i,j], 'posterior_mean': gram_mean[i,j],
                         'q025': lo[i,j], 'q975': hi[i,j],
                         'covered_95pct': bool(lo[i,j] <= gram_true[i,j] <= hi[i,j])})
    pd.DataFrame(grow).to_csv(out / 'gram_recovery.csv', index=False)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(gram_mean-gram_true)
    ax.set(title='Posterior mean Gram − true Gram', xlabel='Response index', ylabel='Response index')
    fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(out / 'gram_error_heatmap.png', dpi=150); plt.close(fig)
    return {'gram_rmse': float(np.sqrt(np.mean((gram_mean-gram_true)**2))),
            'gram_element_95pct_coverage': float(np.mean((lo<=gram_true)&(gram_true<=hi))),
            'direct_parameter_95pct_coverage': float(table.covered_95pct.mean()),
            'nonrotation_parameter_95pct_coverage': float(table.loc[~table.rotation_sensitive, 'covered_95pct'].mean())}


def posterior_predictive(post, data, meta, out, draws=200, seed=2026):
    """Conditional in-sample PPC from saved latent factors and global parameters.

    No need to rebuild the PyMC model or to have a posterior_predictive group.
    Uses paired posterior draws and a separate RNG for replicates.
    """
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    frame = data['frame']
    c_names = meta['continuous_responses']; b_names = meta['binary_responses']
    n = len(frame)
    n_chain = post.sizes['chain']; n_draw = post.sizes['draw']
    n_reps = min(draws, n_chain*n_draw)
    # Stratified, deterministic chain/draw selection.
    indices = np.linspace(0, n_chain*n_draw-1, n_reps, dtype=int)
    observed = frame[c_names+b_names].to_numpy(dtype=float)
    reps = np.empty((n_reps, n, len(c_names)+len(b_names)), dtype=np.float32)
    for r, flat in enumerate(indices):
        chain, draw = divmod(int(flat), n_draw)
        def at(name):
            return post[name].isel(chain=chain, draw=draw).values
        f = at('factor')
        mu = sigmoid(at('intercept_cont') + f @ at('loading_cont').T)
        phi = at('precision_cont')
        yc = rng.beta(np.maximum(mu*phi, 1e-8), np.maximum((1-mu)*phi, 1e-8))
        boundary = 'one' if meta['kind'] == 'terminal' else 'zero'
        p_boundary = sigmoid(at('intercept_'+boundary) + f @ at('loading_'+boundary).T)
        events = rng.random(p_boundary.shape) < p_boundary
        if meta['kind'] == 'terminal':
            yc[:,1:] = np.where(events, 1., yc[:,1:])
        else:
            yc[:,1:2] = np.where(events, 0., yc[:,1:2])
        pb = sigmoid(at('intercept_binary') + f @ at('loading_binary').T)
        yb = rng.binomial(1, pb)
        reps[r] = np.concatenate([yc, yb], axis=1)
    rows = []
    for j, name in enumerate(c_names+b_names):
        x = observed[:,j]; y = reps[:,:,j]
        for stat, func in [('mean', lambda a,axis: np.mean(a,axis=axis)),
                           ('sd', lambda a,axis: np.std(a,axis=axis)),
                           ('zero_share', lambda a,axis: np.mean(a==0,axis=axis)),
                           ('one_share', lambda a,axis: np.mean(a==1,axis=axis)),
                           ('q05', lambda a,axis: np.quantile(a,.05,axis=axis)),
                           ('q50', lambda a,axis: np.quantile(a,.5,axis=axis)),
                           ('q95', lambda a,axis: np.quantile(a,.95,axis=axis))]:
            v = func(y, axis=1)
            rows.append({'response':name,'statistic':stat,'observed':float(func(x,axis=0)),
                         'replicated_median':float(np.median(v)),
                         'replicated_q025':float(np.quantile(v,.025)),
                         'replicated_q975':float(np.quantile(v,.975))})
    pd.DataFrame(rows).to_csv(out/'posterior_checks.csv',index=False)
    corr_rows = []
    names = c_names+b_names
    # Pearson correlation of ranks = Spearman, using pandas for tied/binary values.
    obs_corr = pd.DataFrame(observed).corr(method='spearman').to_numpy()
    corr = np.array([pd.DataFrame(rep).corr(method='spearman').to_numpy() for rep in reps])
    for i in range(len(names)):
        for j in range(i+1,len(names)):
            valid = corr[:,i,j][np.isfinite(corr[:,i,j])]
            if not len(valid):
                continue
            corr_rows.append({'response_a':names[i], 'response_b':names[j],
                              'observed':float(obs_corr[i,j]),
                              'replicated_median':float(np.median(valid)),
                              'replicated_q025':float(np.quantile(valid,.025)),
                              'replicated_q975':float(np.quantile(valid,.975)),
                              'valid_replicates':len(valid)})
    pd.DataFrame(corr_rows).to_csv(out/'joint_predictive_checks.csv',index=False)
    for j, name in enumerate(names):
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        if j < len(c_names):
            ax.hist(observed[:,j], bins=25, density=True, alpha=.55, label='Observed')
            ax.hist(reps[:min(30,n_reps),:,j].ravel(), bins=25, density=True,
                    alpha=.45, label='Posterior replicated')
        else:
            ax.bar([0,1], [np.mean(observed[:,j]==0),np.mean(observed[:,j]==1)],
                   width=.35, label='Observed')
            ax.bar([.35,1.35], [np.mean(reps[:,:,j]==0),np.mean(reps[:,:,j]==1)],
                   width=.35, label='Posterior replicated')
            ax.set_xticks([.175,1.175], ['0','1'])
        ax.set_title(name); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out/f'ppc_{j+1:02d}.png',dpi=150); plt.close(fig)
    return {'predictive_replicates':n_reps, 'predictive_seed':seed}


def latent_summary(post, truth, out):
    if 'factor' not in post:
        return
    factor = post['factor']
    n = factor.sizes['obs']
    rows = []
    # Calculate per factor, so we do not materialize the whole tensor at once.
    for h in range(factor.sizes['factor_dim']):
        v = factor.isel(factor_dim=h).values
        m = v.mean(axis=(0,1)); lo,hi=np.quantile(v,[.025,.975],axis=(0,1))
        for i in range(n):
            rows.append({'obs':i, 'factor':h+1, 'true':float(truth['factor'][i,h]),
                         'posterior_mean':float(m[i]), 'q025':float(lo[i]), 'q975':float(hi[i])})
    pd.DataFrame(rows).to_csv(out/'latent_summary.csv',index=False)
    (out/'latent_summary_README.txt').write_text(
        'Raw factor coordinates are rotation-sensitive; compare across models only after alignment.\n',
        encoding='utf-8')


def review(dataset, output_name=None, ppc_draws=200, skip_classical=False,
           skip_ppc=False, skip_local=False, zip_reviews=True):
    import arviz as az
    data, meta = load_dataset(dataset)
    truth = load_truth(dataset)
    run_dir = Path(dataset)/'fits'/output_name if output_name else Path(dataset)/'fits'
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    comparisons=[]
    for variant in VARIANTS:
        directory=run_dir/variant
        cfg_file=directory/'config.json'
        if not cfg_file.exists():
            print(f'Skipping {variant}: no config',flush=True);continue
        cfg=json.loads(cfg_file.read_text(encoding='utf-8'))
        files=[directory/f'chain_{i+1:02d}.nc' for i in range(cfg['chains'])]
        if not all(p.exists() for p in files):
            print(f'Skipping {variant}: incomplete chains',flush=True);continue
        out=directory/('review_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
        out.mkdir()
        print(f'\nReviewing {variant} -> {out}',flush=True)
        idata=None
        try:
            idata=az.concat([az.from_netcdf(p) for p in files],dim='chain')
            post=idata.posterior
            names=global_variables(post)
            if not names:
                raise ValueError('No global variables')
            print('  Posterior summaries and NUTS diagnostics...',flush=True)
            diagnostics=sample_diagnostics(idata,out,names,include_local=not skip_local)
            print('  Trace plots and autocorrelation...',flush=True)
            plot_trace_acf(post,out,names)
            if not skip_classical:
                print('  Geweke and Heidelberger–Welch...',flush=True)
                classical_diagnostics(post,out,names)
            print('  Parameter and invariant Gram recovery...',flush=True)
            rec=recovery(post,truth,meta,out)
            print('  Latent factor summaries...',flush=True)
            latent_summary(post,truth,out)
            ppc_info={}
            if not skip_ppc:
                print('  Conditional posterior predictive checks...',flush=True)
                ppc_info=posterior_predictive(post,data,meta,out,draws=ppc_draws)
            row={'variant':variant,'scenario':meta['scenario'],'n':meta['n'],
                 'chains':cfg['chains'],'tune':cfg['tune'],'draws_per_chain':cfg['draws'],
                 'target_accept':cfg['target_accept'],**diagnostics,**rec,**ppc_info}
            comparisons.append(row)
            save_json(out/'review_metadata.json',{'source_chains':[str(p) for p in files],
                'run_config':cfg,'dataset_metadata':meta,'review_options':{
                    'ppc_draws':ppc_draws,'skip_classical':skip_classical,
                    'skip_ppc':skip_ppc,'skip_local':skip_local}})
            (out/'README.txt').write_text(
                'Simulation review (no refitting).\n'
                'parameter_summary.csv: posterior means, HDIs and MCMC diagnostics for global parameters.\n'
                'parameter_recovery.csv: true values and equal-tailed 95% posterior intervals.\n'
                'Raw beta, loadings and factors are rotation-sensitive, including in shrinkage.\n'
                'gram_recovery.csv: rotation-invariant loading recovery; 95% equal-tailed intervals.\n'
                'posterior_checks.csv / joint_predictive_checks.csv: conditional in-sample PPCs.\n'
                'PPCs condition on latent factors inferred from the same observations, not out-of-sample predictions.\n'
                'Geweke/HW use production diagnostics.py; not proof of convergence.\n'
                'Interpret posterior recovery cautiously if divergences or poor mixing are present.\n',
                encoding='utf-8')
            if zip_reviews:
                archive=shutil.make_archive(str(out),'zip',out)
                print(f'  Saved ZIP: {archive}',flush=True)
            print(f'  Completed {variant}: {diagnostics["divergences"]} divergences',flush=True)
        finally:
            del idata
            gc.collect()
    if not comparisons:
        raise RuntimeError(f'No complete fits found under {run_dir}')
    dest=run_dir/'comparison.csv' if output_name else Path(dataset)/'comparison.csv'
    pd.DataFrame(comparisons).to_csv(dest,index=False)
    print(f'\nComparison saved: {dest}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True,type=Path)
    parser.add_argument('--output-name',default=None)
    parser.add_argument('--ppc-draws',type=int,default=200)
    parser.add_argument('--skip-classical',action='store_true')
    parser.add_argument('--skip-ppc',action='store_true')
    parser.add_argument('--skip-local',action='store_true',help='Skip observation-level factor mixing diagnostics')
    parser.add_argument('--no-zip',action='store_true')
    args=parser.parse_args()
    if args.ppc_draws<20:
        parser.error('--ppc-draws must be >= 20')
    review(args.dataset,args.output_name,args.ppc_draws,args.skip_classical,
           args.skip_ppc,args.skip_local,not args.no_zip)
