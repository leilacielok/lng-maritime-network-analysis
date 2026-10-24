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



# Display labels shared by posterior traces, ACFs and response curves.
DISPLAY_NAMES = {
    'terminal_age': 'Terminal age',
    'exporter_lag1': 'Lagged exporter status',
    'log1p_capacity_mtpa': 'Terminal capacity',
    'log1p_mean_voyage_distance_lag1': 'Lagged voyage distance',
    'neighbor_activity_lag1': 'Lagged neighbour activity',
    'month_sin': 'Seasonality (sine)', 'month_cos': 'Seasonality (cosine)',
    'role_specific_pagerank': 'PageRank', 'pagerank': 'PageRank',
    'role_specific_hhi_terminal': 'HHI', 'hhi_terminal': 'HHI',
    'role_specific_dependence': 'Dependence', 'dependence': 'Dependence',
    'betweenness': 'Betweenness centrality',
    'share_monthly_network_flow': 'Monthly flow share',
    'articulation_point': 'Articulation point',
    'incident_global_bridge': 'Incident global bridge',
    'connected_to_articulation_point': 'Adjacent to articulation point',
    'exporter_current': 'Current exporter status',
}
PARAMETER_FAMILIES = {
    'beta': 'Regression coefficient', 'loading_cont': 'Continuous-response loading',
    'loading_binary': 'Binary-response loading', 'loading_one': 'One-mass loading',
    'loading_zero': 'Zero-mass loading', 'intercept_cont': 'Continuous-response intercept',
    'intercept_binary': 'Binary-response intercept', 'intercept_one': 'One-mass intercept',
    'intercept_zero': 'Zero-mass intercept', 'precision_cont': 'Beta precision',
}
CHAIN_COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color'][:4]

def pretty_parameter(family, labels):
    subject = ', '.join(DISPLAY_NAMES.get(str(x), str(x).replace('_', ' ').title()) for x in labels)
    return f'{subject} — {PARAMETER_FAMILIES.get(family, family)}' if subject else PARAMETER_FAMILIES.get(family, family)

def add_chain_legend(fig, chain_values):
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=CHAIN_COLORS[i % len(CHAIN_COLORS)], lw=1.7,
                      label=f'Chain {i + 1}') for i, _ in enumerate(chain_values)]
    fig.legend(handles=handles, loc='lower center', ncol=len(handles),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.005))

def samples(ds, name, dimension):
    return ds[name].transpose('chain', 'draw', dimension).values.reshape(-1, ds.sizes[dimension])

def export_trace_pages(post, out, node_type):
    import arviz as az

    names = [
        'beta', 'loading_cont', 'loading_binary',
        'intercept_cont', 'intercept_binary', 'precision_cont',
        'intercept_one', 'loading_one',
        'intercept_zero', 'loading_zero',
    ]

    # Una variabile scalare per ciascun parametro.
    variables = {}

    for name in names:
        if name not in post:
            continue

        array = post[name]
        extra_dims = [
            dim for dim in array.dims
            if dim not in ('chain', 'draw')
        ]
        shape = tuple(array.sizes[dim] for dim in extra_dims)

        for index in np.ndindex(shape):
            selection = dict(zip(extra_dims, index))
            scalar = array.isel(selection).transpose('chain', 'draw')

            labels = [
                str(array.coords[dim].values[i])
                for dim, i in selection.items()
            ]
            title = pretty_parameter(name, labels)

            variables[title] = xr.DataArray(
                scalar.values, dims=('chain', 'draw')
            )

    dataset = xr.Dataset(variables)
    labels = list(variables)
    rows_per_page = 4

    for start in range(0, len(labels), rows_per_page):
        selected = labels[start:start + rows_per_page]
        page = start // rows_per_page + 1

        with az.rc_context({'plot.max_subplots': 100}):
            axes = az.plot_trace(
                dataset[selected],
                compact=False,
                figsize=(11, 2.3 * len(selected)),
            )

        fig = np.asarray(axes).ravel()[0].figure

        for row, title in enumerate(selected):
            for column in range(2):
                ax = axes[row, column]
                ax.set_title(title, fontsize=9, pad=10)
                ax.tick_params(axis='both', labelsize=8)

        # ArviZ colors chains consistently across density and trace panels.
        for ax in np.asarray(axes).ravel():
            ax.set_xlabel('Iteration' if ax in np.asarray(axes)[:, 1] else 'Parameter value', fontsize=9)
        add_chain_legend(fig, post.chain.values)
        fig.subplots_adjust(hspace=0.85, wspace=0.30, top=0.96, bottom=0.12)

        filename = f'{node_type}_trace_{page:02d}'
        fig.savefig(out / f'{filename}.png', dpi=180)
        plt.close(fig)
        
def export_selected_acf(post, out):
    import arviz as az

    # Solo parametri globali: esclude i fattori nodo-mese.
    names = [
        'beta', 'loading_cont', 'loading_binary',
        'intercept_cont', 'intercept_binary', 'precision_cont',
        'intercept_one', 'loading_one',
        'intercept_zero', 'loading_zero',
    ]

    variables = {}
    families = {}

    for name in names:
        if name not in post:
            continue

        array = post[name]
        extra_dims = [
            dim for dim in array.dims if dim not in ('chain', 'draw')
        ]

        shape = tuple(array.sizes[dim] for dim in extra_dims)
        for index in np.ndindex(shape):
            selection = dict(zip(extra_dims, index))
            scalar = array.isel(selection).transpose('chain', 'draw')

            labels = [
                str(array.coords[dim].values[i])
                for dim, i in selection.items()
            ]
            label = name + (
                '[' + ', '.join(labels) + ']' if labels else ''
            )

            variables[label] = xr.DataArray(
                scalar.values, dims=('chain', 'draw')
            )
            families[label] = name

    dataset = xr.Dataset(variables)
    diagnostics = az.summary(dataset, kind='diagnostics')

    selected = {}

    def select_lowest(labels, column, reason):
        values = diagnostics.loc[labels, column].dropna()
        if not values.empty:
            label = values.idxmin()
            selected.setdefault(label, []).append(reason)

    select_lowest(
        [label for label in variables if families[label] == 'beta'],
        'ess_bulk', 'Lowest bulk ESS among regression coefficients'
    )
    select_lowest(
        [label for label in variables
         if families[label].startswith('loading_')],
        'ess_bulk', 'Lowest bulk ESS among response loadings'
    )
    select_lowest(
        list(variables), 'ess_bulk',
        'Lowest bulk ESS among global parameters'
    )
    select_lowest(
        list(variables), 'ess_tail',
        'Lowest tail ESS among global parameters'
    )

    for label in variables:
        if families[label] == 'loading_cont' and 'pagerank' in label:
            selected.setdefault(label, []).append('PageRank anchor loading')

    selection_table = diagnostics.loc[list(selected)].copy()
    selection_table['selection_reason'] = [
        '; '.join(selected[label]) for label in selection_table.index
    ]
    selection_table.to_csv(out / 'acf_selection.csv')

    max_lag = min(100, post.sizes['draw'] - 1)

    for number, label in enumerate(selected, start=1):
        values = variables[label].values
        fig, ax = plt.subplots(figsize=(8.5, 3.0))
        lags = np.arange(max_lag + 1)
        for chain in range(values.shape[0]):
            acf = az.autocorr(values[chain])[:max_lag + 1]
            ax.plot(lags, acf, color=CHAIN_COLORS[chain % len(CHAIN_COLORS)],
                    linewidth=1.25, alpha=0.85, label=f'Chain {chain + 1}')
        ax.axhline(0, color='black', linewidth=0.7)
        ax.set_xlim(0, max_lag)
        ax.set_ylim(-0.25, 1.0)
        ax.set_xlabel('Lag')
        ax.set_ylabel('Autocorrelation')
        # Recover original parameter family and coordinate labels.
        family = families[label]
        coordinate_labels = label[len(family):].strip('[]').split(', ') if label != family else []
        ax.set_title(pretty_parameter(family, coordinate_labels), fontsize=11, pad=10)
        ax.legend(loc='upper right', ncol=min(4, values.shape[0]), frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f'acf_{number:02d}.png', dpi=180, bbox_inches='tight')
        plt.close(fig)

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
        archived = Path(__file__).with_name('estimation.py')
    import sys
    sys.path.insert(0, str(archived.parent))
    try:
        spec = importlib.util.spec_from_file_location('archived_bounded', archived)
        model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model)
    finally:
        sys.path.pop(0)
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
        export_selected_acf(post, out)
        export_trace_pages(post, out, args.node_type)
        
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
        # Paired posterior contrasts on the original response scale.
        # These include the point mass at one (terminal HHI/dependence)
        # or zero (chokepoint betweenness), rather than only the Beta mean.
        factor_low = float(keys.factor_mean.quantile(.01))
        factor_high = float(keys.factor_mean.quantile(.99))
        if not (factor_low <= 0.0 and 1.0 <= factor_high):
            print(f'WARNING: factor values 0 and 1 are not both within '
                  f'the plotted 1%-99% range [{factor_low:.3f}, {factor_high:.3f}]')
        contrast_rows = []
        for j, response in enumerate(audit['continuous_responses']):
            def expected_at(f):
                mu = expit(ac[:, j] + lc[:, j] * f)
                if args.node_type == 'terminal':
                    p = (np.zeros_like(mu) if j == 0 else
                         expit(ao[:, j-1] + lo[:, j-1] * f))
                    return p + (1.0 - p) * mu
                p = (expit(ao[:, 0] + lo[:, 0] * f)
                     if response == 'betweenness' else np.zeros_like(mu))
                return (1.0 - p) * mu

            at_zero = expected_at(0.0)
            at_one = expected_at(1.0)
            difference = at_one - at_zero  # paired draws, not differences of quantiles
            row = {'response': response, 'factor_from': 0.0, 'factor_to': 1.0,
                   'scale': 'original_response', 'draws': len(difference),
                   'within_plotted_factor_range': factor_low <= 0.0 and 1.0 <= factor_high}
            for label, values in [('expected_f0', at_zero), ('expected_f1', at_one),
                                  ('difference', difference)]:
                row[f'{label}_median'] = float(np.median(values))
                row[f'{label}_q025'] = float(np.quantile(values, .025))
                row[f'{label}_q975'] = float(np.quantile(values, .975))
            contrast_rows.append(row)
        pd.DataFrame(contrast_rows).to_csv(out / 'factor_response_contrasts.csv', index=False)
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
            axes.flat[j].set_title(DISPLAY_NAMES.get(response, response.replace('_', ' ').title()), fontsize=11)
        for j, response in enumerate(audit['binary_responses']):
            v = expit(ab[:, j, None] + lb[:, j, None] * grid)
            low, mid, high = np.quantile(v, [.025, .5, .975], axis=0)
            rows.extend(dict(response=response, quantity='probability', factor=f, median=m, q025=l, q975=h) for f,m,l,h in zip(grid,mid,low,high))
            axes.flat[j+3].plot(grid, mid)
            axes.flat[j+3].fill_between(grid, low, high, alpha=.25)
            axes.flat[j+3].set_title(DISPLAY_NAMES.get(response, response.replace('_', ' ').title()), fontsize=11)
        for ax in list(axes.flat)[3 + len(audit['binary_responses']):]:
            ax.set_visible(False)
        for j, ax in enumerate(axes.flat):
            if not ax.get_visible():
                continue
            ax.set_xlabel('Latent factor')
            ax.set_ylabel('Expected response' if j < len(audit['continuous_responses'])
                          else 'Probability')
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
    (out / 'README.txt').write_text('factor_response_contrasts.csv: paired posterior comparisons of expected bounded responses at factor 0 vs 1; intervals are equal-tailed 95% credible intervals for the difference, not predictive intervals.\nExisting posterior only: no refitting.\nPredictive checks are conditional and in-sample, not out-of-sample validation.\nTerminal role groups use LAGGED role; chokepoints use the full sample.\nFactor associations use posterior mean scores informed by these same responses; they are descriptive, not independent validation or causal effects.\nCurve bands are equal-tailed 95% parameter uncertainty, not predictive intervals.\nCurves span the 1st-99th percentiles of posterior mean latent scores.\nNode/month scores are conditional on consecutive activity (and available age for terminals); interpret alongside sampling diagnostics and model limitations.\n', encoding='utf-8')
    (out / 'export_metadata.json').write_text(json.dumps(dict(source=str(nc), node_type=args.node_type, predictive_replicates=actual_draws, panel_sha256=audit['input_sha256']), indent=2))
    archive = shutil.make_archive(str(out), 'zip', out)
    print(f'Done. Send this small ZIP: {archive}', flush=True)


if __name__ == '__main__':
    main()
