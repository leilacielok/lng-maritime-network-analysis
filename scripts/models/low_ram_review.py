"""Low-memory review for posterior.nc created by --low-ram.

Computes global/node diagnostics and exact posterior summaries for all observation
factors in small observation batches. Does not pretend to diagnose every latent
coordinate; reports a fixed, reproducible observation subset separately.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--node-type', choices=['terminal', 'chokepoint'], default='terminal')
    parser.add_argument('--panel', type=Path)
    args = parser.parse_args()
    import arviz as az
    from common import prepare_panel, SPECS
    from diagnostics import predictive_checks, save_classical_diagnostics
    from review import export_selected_acf, export_trace_pages, export_node_effects
    terminal = args.run_dir if (args.run_dir / 'posterior.nc').exists() else args.run_dir / args.node_type
    nc = terminal / 'posterior.nc'
    if not nc.exists():
        raise FileNotFoundError(nc)
    if args.panel is None:
        args.panel = Path(__file__).resolve().parents[2] / 'data/processed/model_ready' / f'{args.node_type}_criticality_model_panel.csv'
    data = prepare_panel(args.panel, args.node_type)
    if (terminal / 'node_index.csv').exists():
        from node_effects import prepare_node_effects
        prepare_node_effects(data, args.node_type)
    out = terminal / 'low_ram_review'
    out.mkdir(exist_ok=True)
    with xr.open_dataset(nc, group='posterior', engine='h5netcdf') as post:
        globals_ = post.drop_vars([n for n, a in post.data_vars.items() if 'obs' in a.dims])
        mixing = az.summary(globals_, kind='diagnostics')
        mixing.to_csv(out / 'global_mixing_diagnostics.csv')
        az.summary(globals_).to_csv(out / 'global_parameter_summary.csv')
        save_classical_diagnostics(globals_, out)
        export_selected_acf(globals_, out)
        export_trace_pages(globals_, out, args.node_type)
        if 'mu_node' in post:
            export_node_effects(globals_, data, out)
        # Exact full-draw factor summaries; at most 8 obs columns in RAM.
        frame = data['frame']
        keys = frame[[SPECS[args.node_type]['id'], 'node_name', 'period_month']].copy()
        if 'terminal_role_lag1' in frame:
            keys['terminal_role_lag1'] = frame['terminal_role_lag1'].values
        beta = post['beta'].transpose('chain', 'draw', 'predictor').values
        mu = (post['mu_node'].transpose('chain', 'draw', 'node').values
              if 'mu_node' in post else None)
        for start in range(0, len(keys), 8):
            stop = min(start + 8, len(keys))
            factor = post['factor'].isel(obs=slice(start, stop)).transpose('chain','draw','obs').values
            cov = np.einsum('cdp,op->cdo', beta, data['x'][start:stop])
            mean = cov + (mu[:, :, data['node_index'][start:stop]] if mu is not None else 0)
            for name, array in [('factor', factor), ('factor_mean', mean),
                                ('factor_residual', factor-mean), ('factor_covariate_mean', cov)]:
                vals = array.reshape(-1, stop-start)
                keys.loc[start:stop-1, name+'_mean'] = vals.mean(axis=0)
                qs = np.quantile(vals, [.025, .975], axis=0)
                keys.loc[start:stop-1, name+'_q025'] = qs[0]
                keys.loc[start:stop-1, name+'_q975'] = qs[1]
        keys.to_csv(out / 'latent_summary.csv', index=False)
        # Representative latent diagnostics, explicitly NOT full-coordinate coverage.
        positions = np.unique(np.linspace(0, len(keys)-1, min(24,len(keys)), dtype=int))
        factor_subset = post['factor'].isel(obs=positions).load()
        latent_mix = az.summary(xr.Dataset({'factor': factor_subset}), kind='diagnostics')
        latent_mix.to_csv(out / 'sampled_latent_mixing_diagnostics.csv')
        with xr.open_dataset(nc, group='sample_stats', engine='h5netcdf') as stats:
            divergences = int(stats['diverging'].sum())
        report = {'divergences': divergences,
                  'max_rhat_global': float(mixing['r_hat'].max()),
                  'min_ess_bulk_global': float(mixing['ess_bulk'].min()),
                  'min_ess_tail_global': float(mixing['ess_tail'].min()),
                  'latent_diagnostics_scope': f'{len(positions)} evenly spaced observations, not all',
                  'max_rhat_latent_subset': float(latent_mix['r_hat'].max()),
                  'min_ess_bulk_latent_subset': float(latent_mix['ess_bulk'].min()),
                  'min_ess_tail_latent_subset': float(latent_mix['ess_tail'].min()),
                  'passes_basic_checks': False,
                  'note': 'Full latent-coordinate mixing not evaluated; no blanket convergence claim.'}
        (terminal / 'sampling_diagnostics.json').write_text(json.dumps(report, indent=2))
    with xr.open_dataset(nc, group='posterior_predictive', engine='h5netcdf') as ppc:
        import arviz as az
        from diagnostics import predictive_checks
        idata = az.InferenceData(posterior_predictive=ppc.load())
        predictive_checks(idata, data, 'posterior_predictive').to_csv(out / 'posterior_checks.csv', index=False)
    print('Review saved:', out)


if __name__ == '__main__':
    main()
