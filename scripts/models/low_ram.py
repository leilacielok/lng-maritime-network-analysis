"""Low-memory, resumable sequential PyMC sampling for PyMC 5.20.x.

Requires: dask[array], h5netcdf. 
One chain per file.
"""
import gc
import json
from pathlib import Path
import numpy as np


# Versioned provenance marker prevents accidentally reusing chains from the
# previous custom-NDArray implementation, which produced prior-like samples.
SAMPLER_VERSION = "pymc_default_trace_v2"


def _sampling_deterministics(model):
    """Keep only non-observation-level deterministics in the default PyMC trace.

    This changes only which deterministic outputs PyMC records. It does not
    remove stochastic variables, observed RVs, or any likelihood terms.
    """
    dims = model.named_vars_to_dims
    return [v for v in model.deterministics
            if "obs" not in dims.get(v.name, ())]


def _chain_marker(path):
    return path.with_name(path.stem + ".sampler.json")


def _sample_one_chain(data, kind, args, index, seed, start, path):
    """Spawn-safe worker: constructs its own model; never shares PyMC model objects."""
    import pymc as pm
    from baseline import build_model
    if args.model == 'node_effects':
        from node_effects import build_factor_mean
        model = build_model(data, factor_builder=build_factor_mean, rank=args.rank)
    else:
        model = build_model(data, rank=args.rank)
    print(f'Sampling chain {index+1}/{args.chains}; saving to {path}', flush=True)
    # IMPORTANT: Never pass a manually created NDArray as `trace=` to pm.sample.
    # That code path produced prior-like draws in the verified synthetic test.
    # Use PyMC's own trace backend and exclude only bulky deterministic outputs.
    original_deterministics = list(model.deterministics)
    model.deterministics = _sampling_deterministics(model)
    try:
        with model:
            idata = pm.sample(
                draws=args.draws, tune=args.tune, chains=1, cores=1,
                target_accept=args.target_accept, init='adapt_diag',
                initvals=start, random_seed=int(seed),
                return_inferencedata=True,
                compute_convergence_checks=False,
                idata_kwargs={'log_likelihood': False},
            )
    finally:
        model.deterministics = original_deterministics
    if idata.posterior.sizes['draw'] != args.draws:
        raise RuntimeError(f'Chain {index+1} returned incomplete draws')
    idata.posterior = idata.posterior.assign_coords(chain=[index])
    idata.sample_stats = idata.sample_stats.assign_coords(chain=[index])
    temporary = path.with_name(path.stem + '.partial.nc')
    idata.to_netcdf(temporary, engine='h5netcdf')
    temporary.replace(path)
    # Written only after the NetCDF has been successfully finalized.
    _chain_marker(path).write_text(json.dumps({
        'sampler_version': SAMPLER_VERSION,
        'draws': args.draws,
        'tune': args.tune,
        'seed': int(seed),
        'target_accept': args.target_accept,
        'model': args.model,
        'rank': args.rank,
        'node_type': kind,
    }, indent=2), encoding='utf-8')
    del idata
    gc.collect()
    return str(path)


def run_chains(model, data, kind, args, destination):
    from concurrent.futures import ProcessPoolExecutor
    from baseline import initial_values, chokepoint_initial_values, rank_two_initial_values
    import xarray as xr

    chain_dir = destination / 'chains'
    chain_dir.mkdir(exist_ok=True)
    starts = (rank_two_initial_values(data, args.chains, args.seed) if args.rank == 2
              else initial_values(data, args.chains, args.seed) if kind == 'terminal'
              else chokepoint_initial_values(data, args.chains, args.seed))
    seeds = np.random.SeedSequence(args.seed).generate_state(args.chains).tolist()
    pending = []
    for index in range(args.chains):
        path = chain_dir / f'chain_{index + 1:02d}.nc'
        if path.exists():
            marker = _chain_marker(path)
            if not marker.exists():
                raise RuntimeError(
                    f'Unverified/legacy chain: {path}. Do not reuse it. '
                    'Choose a NEW --run-name or move the old run aside.'
                )
            metadata = json.loads(marker.read_text(encoding='utf-8'))
            expected = {
                'sampler_version': SAMPLER_VERSION,
                'draws': args.draws, 'tune': args.tune,
                'seed': int(seeds[index]),
                'target_accept': args.target_accept,
                'model': args.model, 'rank': args.rank, 'node_type': kind,
            }
            if metadata != expected:
                raise RuntimeError(f'Chain metadata mismatch: {path}. Use a new --run-name.')
            with xr.open_dataset(path, group='posterior') as ds:
                if ds.sizes.get('draw') != args.draws or ds.sizes.get('chain') != 1:
                    raise RuntimeError(f'Incomplete chain file: {path}. Move it aside before retrying.')
            print(f'Reusing verified chain {index+1}: {path}', flush=True)
        else:
            pending.append((index, path))
    workers = min(args.cores, args.chains)
    if workers == 1:
        for index, path in pending:
            _sample_one_chain(data, kind, args, index, seeds[index], starts[index], path)
    else:
        # On Windows, ProcessPoolExecutor uses spawn. Workers reconstruct models
        # inside their own processes; no PyMC model is passed between processes.
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for offset in range(0, len(pending), workers):
                batch = pending[offset:offset + workers]
                futures = [pool.submit(_sample_one_chain, data, kind, args, i,
                                       seeds[i], starts[i], path) for i, path in batch]
                for future in futures:
                    future.result()  # propagate errors; completed chains remain saved
    return chain_dir

def merge_chains(chain_dir, output, count, draws):
    import xarray as xr
    import dask.array  # noqa: F401 -- enables chunked xarray reads
    paths = [chain_dir / f'chain_{i+1:02d}.nc' for i in range(count)]
    if not all(p.exists() for p in paths):
        raise RuntimeError('Missing chain files; cannot combine')
    temporary = output.with_name('posterior.partial.nc')
    if temporary.exists():
        temporary.unlink()
    for group in ('posterior', 'sample_stats'):
        opened = [xr.open_dataset(p, group=group, engine='h5netcdf',
                                  chunks={'draw': min(250, draws)}) for p in paths]
        try:
            combined = xr.concat(opened, dim='chain', join='exact', compat='equals')
            if combined.sizes['chain'] != count or combined.sizes['draw'] != draws:
                raise RuntimeError('Combined posterior has wrong dimensions')
            encoding = {name: {'zlib': True, 'complevel': 2,
                               'chunksizes': tuple(min(1 if dim == 'chain' else
                                                       250 if dim == 'draw' else 32, size)
                                                   for dim, size in zip(var.dims, var.shape))}
                        for name, var in combined.data_vars.items() if var.ndim > 0}
            combined.to_netcdf(temporary, engine='h5netcdf',
                               group=group, mode='w' if group == 'posterior' else 'a',
                               encoding=encoding)
        finally:
            for ds in opened:
                ds.close()
    temporary.replace(output)


def posterior_predictive_subset(model, output, args, draws_per_chain=100):
    """Sample a small, documented PPC subset, not all 120k posterior draws."""
    import pymc as pm
    import arviz as az
    import xarray as xr
    with xr.open_dataset(output, group='posterior', engine='h5netcdf') as post:
        chosen = np.linspace(0, post.sizes['draw'] - 1,
                             min(draws_per_chain, post.sizes['draw']), dtype=int)
        subset = post.isel(draw=chosen).load()
    small = az.InferenceData(posterior=subset)
    with model:
        ppc = pm.sample_posterior_predictive(small, var_names=['y_cont', 'y_binary'],
                                              random_seed=args.seed, return_inferencedata=True)
    ppc.posterior_predictive.to_netcdf(output, engine='h5netcdf',
                                      group='posterior_predictive', mode='a')
    return len(chosen)
