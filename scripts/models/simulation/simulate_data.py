"""Simulate fully synthetic observations from the original rank-two LNG model.

No empirical panel or fitted posterior estimates are used. Raw covariates respect
natural domains; continuous predictors are standardized exactly as in common.py.
Output files are compatible with simulation_estimation.py and simulation_review.py.

Example:
 python scripts/models/simulation/simulate_data.py --kind terminal \
   --scenario nontriangular --n 1000 --seed 42 \
   --output outputs/simulation/terminal_s1
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from simulation_utils import sigmoid, save_json
from common import SPECS


def generate_covariates(rng, n, kind):
    # Draw a calendar month first: sine and cosine must describe the SAME month.
    month = rng.integers(1, 13, size=n)
    activity = rng.poisson(5, size=n).astype(float)  # nonnegative count
    voyage_km = rng.lognormal(mean=np.log(4500), sigma=0.55, size=n)
    raw = pd.DataFrame({
        'neighbor_activity_lag1': activity,
        'log1p_mean_voyage_distance_lag1': np.log1p(voyage_km),
        'month_sin': np.sin(2 * np.pi * month / 12),
        'month_cos': np.cos(2 * np.pi * month / 12),
    })
    if kind == 'terminal':
        capacity_mtpa = rng.lognormal(mean=np.log(5), sigma=0.75, size=n)
        age_years = rng.uniform(0, 45, size=n)
        raw['log1p_capacity_mtpa'] = np.log1p(capacity_mtpa)
        raw['terminal_age'] = age_years
        raw['exporter_lag1'] = rng.binomial(1, 0.45, size=n)
    spec = SPECS[kind]
    numeric = spec['predictors']
    means = raw[numeric].mean()
    sds = raw[numeric].std(ddof=0)
    if (sds <= 0).any():
        raise ValueError('At least one continuous predictor is constant; increase --n')
    x = ((raw[numeric] - means) / sds).to_numpy(dtype=float)
    if kind == 'terminal':
        x = np.column_stack([x, raw['exporter_lag1'].to_numpy(dtype=float)])
    scaling = pd.DataFrame({'mean': means, 'sd': sds})
    if kind == 'terminal':
        scaling.loc['exporter_lag1'] = [0.0, 1.0]
    raw.insert(0, 'month', month)
    return x, raw, scaling


def simulate(args):
    rng = np.random.default_rng(args.seed)
    spec = SPECS[args.kind]
    predictors = spec['predictors'] + (['exporter_lag1'] if args.kind == 'terminal' else [])
    x, raw, scaling = generate_covariates(rng, args.n, args.kind)
    n, q = x.shape

    # Fixed, explicitly known coefficients, NOT taken from empirical estimates.
    beta = np.zeros((q, 2))
    beta[:4] = [[0.22, -0.10], [0.12, 0.20], [0.08, 0.04], [-0.06, 0.08]]
    if args.kind == 'terminal':
        beta[4:] = [[0.18, 0.12], [0.05, -0.08], [0.55, 0.20]]
    factor_mean = x @ beta
    factor = factor_mean + rng.normal(size=(n, 2))

    loading_cont = np.array([[0.70, 0.50], [-0.45, 0.75], [0.35, -0.30]])
    if args.scenario == 'triangular':
        loading_cont[0, 1] = 0.0
    loading_binary = (np.array([[0.80, 0.70], [-0.60, 0.90], [0.15, -0.35]])
                      if args.kind == 'terminal' else np.array([[0.80, 0.70], [0.15, -0.35]]))
    loading_boundary = (np.array([[-0.50, 0.35], [0.30, -0.40]])
                        if args.kind == 'terminal' else np.array([[-0.40, 0.50]]))
    # --------------------------------------------------
    # Check the strength of the two latent dimensions
    # --------------------------------------------------

    loading_matrix = np.vstack([
        loading_cont,
        loading_binary,
        loading_boundary
    ])

    singular_values = np.linalg.svd(
        loading_matrix,
        compute_uv=False
    )

    rank = np.linalg.matrix_rank(loading_matrix)

    sigma1, sigma2 = singular_values

    strength_ratio = sigma2 / sigma1

    second_factor_share = (
        sigma2**2 / (sigma1**2 + sigma2**2)
    )

    print("\nLatent factor diagnostics")
    print("-------------------------")
    print(f"Loading matrix rank: {rank}")
    print(f"First singular value: {sigma1:.4f}")
    print(f"Second singular value: {sigma2:.4f}")
    print(f"Strength ratio: {strength_ratio:.4f}")
    print(f"Second dimension share: {second_factor_share:.2%}")

    if rank < 2:
        print("WARNING: The loading matrix is not rank two.")
    elif strength_ratio < 0.30:
        print("WARNING: The second latent dimension may be weak.")
    else:
        print("Both latent dimensions have substantial loadings.")
        
    intercept_cont = (np.array([-3.5, 0.0, 0.0]) if args.kind == 'terminal'
                      else np.array([-4.0, -4.0, -3.5]))
    intercept_binary = (np.array([-0.5, -0.5, -0.7]) if args.kind == 'terminal'
                        else np.array([-0.5, -0.7]))
    intercept_boundary = (np.array([-1.3, -1.2]) if args.kind == 'terminal'
                          else np.array([-3.0]))
    precision_cont = (np.array([120.0, 12.0, 12.0]) if args.kind == 'terminal'
                      else np.array([150.0, 100.0, 100.0]))

    mu = sigmoid(intercept_cont + factor @ loading_cont.T)
    alpha = np.maximum(mu * precision_cont, 1e-8)
    beta_shape = np.maximum((1.0 - mu) * precision_cont, 1e-8)
    y_cont = rng.beta(alpha, beta_shape)
    # Floating point guard, not a change to the generative distribution.
    y_cont = np.clip(y_cont, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))
    boundary_prob = sigmoid(intercept_boundary + factor @ loading_boundary.T)
    boundary_event = rng.random(size=boundary_prob.shape) < boundary_prob
    if args.kind == 'terminal':
        y_cont[:, 1:] = np.where(boundary_event, 1.0, y_cont[:, 1:])
    else:
        y_cont[:, 1:2] = np.where(boundary_event, 0.0, y_cont[:, 1:2])
    binary_prob = sigmoid(intercept_binary + factor @ loading_binary.T)
    y_binary = rng.binomial(1, binary_prob)

    frame = pd.DataFrame(np.column_stack([y_cont, y_binary]),
                         columns=spec['continuous'] + spec['binary'])
    for col in spec['binary']:
        frame[col] = frame[col].astype(int)
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {out}')
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / 'observations.csv', index=False)
    raw.to_csv(out / 'covariates_raw.csv', index=False)
    scaling.to_csv(out / 'x_scaling.csv', index_label='predictor')
    np.save(out / 'x.npy', x)
    np.savez_compressed(out / 'truth.npz', beta=beta, factor=factor,
                        factor_mean=factor_mean, loading_cont=loading_cont,
                        loading_binary=loading_binary, loading_boundary=loading_boundary,
                        intercept_cont=intercept_cont, intercept_binary=intercept_binary,
                        intercept_boundary=intercept_boundary, precision_cont=precision_cont)
    save_json(out / 'metadata.json', {
        'kind': args.kind, 'scenario': args.scenario, 'seed': args.seed, 'n': n,
        'predictors': predictors, 'continuous_responses': spec['continuous'],
        'binary_responses': spec['binary'], 'prior_profile': args.prior_profile,
        'precision_prior_sd': args.precision_prior_sd, 'panel': None,
        'generative_parameters': 'fixed independently of empirical estimates',
        'covariates': 'fully synthetic; numeric predictors standardized',
    })
    summary = pd.DataFrame({
        'response': frame.columns,
        'mean': frame.mean().to_numpy(),
        'sd': frame.std(ddof=0).to_numpy(),
        'min': frame.min().to_numpy(),
        'max': frame.max().to_numpy(),
        'share_equal_zero': frame.eq(0).mean().to_numpy(),
        'share_equal_one': frame.eq(1).mean().to_numpy(),
    })
    summary.to_csv(out / 'response_summary.csv', index=False)
    print(f'Generated {n} fully synthetic {args.kind} observations in {out}')
    print(summary.to_string(index=False, float_format=lambda v: f'{v:.3f}'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--node-type', dest='kind', choices=['terminal', 'chokepoint'], default='terminal')
    p.add_argument('--scenario', choices=['triangular', 'nontriangular'], default='nontriangular')
    p.add_argument('--n', type=int, default=1000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--prior-profile', choices=['original', 'regularized', 'response_precision'],
                   default='original')
    p.add_argument('--precision-prior-sd', type=float, default=0.7)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    if a.n < 30:
        p.error('--n must be at least 30')
    simulate(a)
