"""Fit rank-two variants to the same simulated dataset.

Variants run sequentially; within each variant up to --cores chains run in
separate processes. Each chain is saved atomically as an individual NetCDF.
Existing completed chains are reused only when the run config matches.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import numpy as np
from simulation_utils import load_dataset, save_json
from rank_two_variants import build_model, VARIANTS


def fit_chain(dataset, variant, shrink_sd, draws, tune, target_accept,
              chain_idx, seed, out, chains):
    """Run in a fresh process, so parallel chains do not share PyTensor state."""
    import gc
    import pymc as pm

    data, _ = load_dataset(dataset)
    print(f'Fitting {variant}: chain {chain_idx + 1}/{chains}', flush=True)
    model = build_model(data, variant=variant, shrink_sd=shrink_sd)
    # Preserve original low-RAM behavior: exclude observation deterministics.
    model.deterministics = [
        v for v in model.deterministics
        if 'obs' not in model.named_vars_to_dims.get(v.name, ())
    ]
    with model:
        idata = pm.sample(
            draws=draws, tune=tune, chains=1, cores=1,
            random_seed=int(seed), target_accept=target_accept,
            init='adapt_diag', compute_convergence_checks=False,
            idata_kwargs={'log_likelihood': False},
            return_inferencedata=True,
        )
    idata.posterior = idata.posterior.assign_coords(chain=[chain_idx])
    idata.sample_stats = idata.sample_stats.assign_coords(chain=[chain_idx])
    chain_path = out / f'chain_{chain_idx + 1:02d}.nc'
    tmp = out / f'chain_{chain_idx + 1:02d}.partial.nc'
    try:
        idata.to_netcdf(tmp, engine='h5netcdf')
        tmp.replace(chain_path)
    finally:
        if tmp.exists():
            tmp.unlink()
    del model, idata
    gc.collect()
    return str(chain_path)


def sample(args):
    # Read once in parent to validate dataset before launching workers.
    load_dataset(args.dataset)
    variants = VARIANTS if args.variant == 'all' else (args.variant,)
    fingerprint = hashlib.sha256()
    for name in ('metadata.json', 'observations.csv', 'x.npy'):
        fingerprint.update((args.dataset / name).read_bytes())
    dataset_hash = fingerprint.hexdigest()

    for variant in variants:
        fits_dir = args.dataset / 'fits'
        out = fits_dir / args.output_name / variant if args.output_name else fits_dir / variant
        out.mkdir(parents=True, exist_ok=True)
        # cores controls scheduling, not statistical configuration: changing
        # --cores does not invalidate already completed chains.
        config = dict(dataset_hash=dataset_hash, variant=variant, draws=args.draws,
                      tune=args.tune, chains=args.chains, seed=args.seed,
                      target_accept=args.target_accept, shrink_sd=args.shrink_sd)
        cfg_path = out / 'config.json'
        if cfg_path.exists():
            if json.loads(cfg_path.read_text(encoding='utf-8')) != config:
                raise RuntimeError(
                    f'Configuration mismatch at {out}; choose another --output-name'
                )
        else:
            if list(out.glob('chain_*.nc')):
                raise RuntimeError(f'Chains exist without config in {out}')
            save_json(cfg_path, config)

        seeds = np.random.SeedSequence(args.seed).generate_state(args.chains)
        pending = []
        for idx in range(args.chains):
            chain_path = out / f'chain_{idx + 1:02d}.nc'
            if chain_path.exists():
                print(f'Skipping existing {chain_path}', flush=True)
            else:
                pending.append((args.dataset, variant, args.shrink_sd,
                                args.draws, args.tune, args.target_accept,
                                idx, int(seeds[idx]), out, args.chains))

        if pending:
            if args.cores == 1:
                for job in pending:
                    print(f'Saved {fit_chain(*job)}', flush=True)
            else:
                # Spawn-safe on Windows. At most --cores chains of ONE variant
                # are active; the next variant starts after all are complete.
                with ProcessPoolExecutor(max_workers=min(args.cores, len(pending))) as pool:
                    futures = [pool.submit(fit_chain, *job) for job in pending]
                    for future in as_completed(futures):
                        print(f'Saved {future.result()}', flush=True)
        print(f'Completed {variant}: {out}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True, type=Path)
    p.add_argument('--variant', choices=[*VARIANTS, 'all'], default='all')
    p.add_argument('--tune', type=int, default=500)
    p.add_argument('--draws', type=int, default=500)
    p.add_argument('--chains', type=int, default=2)
    p.add_argument('--cores', type=int, default=1,
                   help='Maximum concurrent chains within each variant (default: 1)')
    p.add_argument('--output-name', default=None,
                   help='Optional run subfolder under dataset/fits (default: existing layout)')
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--target-accept', type=float, default=.9)
    p.add_argument('--shrink-sd', type=float, default=.25)
    args = p.parse_args()
    if args.chains < 1 or args.cores < 1:
        p.error('--chains and --cores must both be >= 1')
    if args.output_name and (args.output_name in ('.', '..') or
                             Path(args.output_name).name != args.output_name or
                             '/' in args.output_name or '\\' in args.output_name):
        p.error('--output-name must be a single folder name, not a path')
    sample(args)
