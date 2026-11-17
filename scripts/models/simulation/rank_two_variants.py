"""Three rank-two model variants with the existing LNG likelihoods.

unconstrained: original baseline.py rank-two model.
triangular: PageRank factor-2 loading = 0, second continuous response
            factor-2 loading > 0 (positive diagonal).
shrinkage: factor-2 loading priors tightened, including boundary loadings.

The shrinkage is on *loadings*, not on factor variance (fixed at one).
"""
import numpy as np
from simulation_utils import MODEL_DIR  # noqa: F401 (sets import path)
from baseline import build_rank_two_model, prior_specification
from common import (one_inflated_beta_logp, one_inflated_beta_random,
                    zero_inflated_beta_logp, zero_inflated_beta_random)

VARIANTS = ('unconstrained', 'triangular', 'shrinkage')


def build_model(data, variant='unconstrained', shrink_sd=0.25):
    if variant not in VARIANTS:
        raise ValueError(f'variant must be one of {VARIANTS}')
    if shrink_sd <= 0:
        raise ValueError('shrink_sd must be positive')
    if variant == 'unconstrained':
        return build_rank_two_model(data)

    import pymc as pm
    import pytensor.tensor as pt

    kind = 'terminal' if 'exporter_lag1' in data['audit']['predictors'] else 'chokepoint'
    priors = prior_specification(kind, data.get('prior_profile', 'regularized'))
    c = data['audit']['continuous_responses']
    b = data['audit']['binary_responses']
    boundary = c[1:] if kind == 'terminal' else ['betweenness']
    coords = {'obs': np.arange(len(data['frame'])),
              'predictor': data['audit']['predictors'],
              'factor_dim': ['factor_1', 'factor_2'],
              'continuous': c, 'other_continuous': c[1:],
              'binary': b, 'boundary': boundary}

    with pm.Model(coords=coords) as model:
        beta = pm.Normal('beta', 0, priors['beta_sd'], dims=('predictor', 'factor_dim'))
        mean = pm.Deterministic('factor_mean', pt.dot(data['x'], beta), dims=('obs', 'factor_dim'))
        factor = pm.Normal('factor', mu=mean, sigma=1, dims=('obs', 'factor_dim'))
        pm.Deterministic('factor_residual', factor - mean, dims=('obs', 'factor_dim'))

        anchor = pm.HalfNormal('pagerank_loading', priors['pagerank_loading_sd'])
        if variant == 'triangular':
            second_pr = pt.as_tensor_variable(0.0)
            # Lower triangular leading 2x2 block: [[+, 0], [free, +]].
            other_first = pm.Normal('other_first_loadings', 0,
                                    np.asarray(priors['other_loading_sd']),
                                    dims='other_continuous')
            diagonal = pm.HalfNormal('second_diagonal_loading',
                                     float(priors['other_loading_sd'][0]))
            remaining_second = pm.Normal('remaining_second_loading', 0,
                                         float(priors['other_loading_sd'][1]))
            other = pt.stack([other_first, pt.stack([diagonal, remaining_second])], axis=1)
            pm.Deterministic('other_loadings', other,
                             dims=('other_continuous', 'factor_dim'))
        else:
            second_pr = pm.Normal('pagerank_second_loading', 0, shrink_sd)
            other_sd = np.asarray(priors['other_loading_sd'])[:, None] * np.array([1.0, shrink_sd])
            other = pm.Normal('other_loadings', 0, other_sd,
                              dims=('other_continuous', 'factor_dim'))
        lc = pm.Deterministic('loading_cont',
                              pt.concatenate([pt.stack([anchor, second_pr])[None, :], other], axis=0),
                              dims=('continuous', 'factor_dim'))
        binary_sd = (priors['binary_loading_sd'] if variant == 'triangular'
                     else np.array([priors['binary_loading_sd'], shrink_sd]))
        lb = pm.Normal('loading_binary', 0, binary_sd, dims=('binary', 'factor_dim'))
        ac = pm.Normal('intercept_cont', priors['interior_intercept_mean'],
                       priors['interior_intercept_sd'], dims='continuous')
        ab = pm.Normal('intercept_binary', 0, priors['binary_intercept_sd'], dims='binary')
        phi = pm.LogNormal('precision_cont', np.log(priors['precision_median']),
                           data['precision_prior_sd'], dims='continuous')
        bc = pm.Normal('intercept_one' if kind == 'terminal' else 'intercept_zero',
                       priors['boundary_intercept_mean'], priors['boundary_intercept_sd'],
                       dims='boundary')
        boundary_sd = (priors['boundary_loading_sd'] if variant == 'triangular'
                       else np.array([priors['boundary_loading_sd'], shrink_sd]))
        bl = pm.Normal('loading_one' if kind == 'terminal' else 'loading_zero',
                       0, boundary_sd, dims=('boundary', 'factor_dim'))
        eta = bc + pt.dot(factor, bl.T)
        if kind == 'terminal':
            logits = pt.concatenate([pt.zeros_like(eta[:, :1]), eta], axis=1)
            logp, random = one_inflated_beta_logp, one_inflated_beta_random
            inflated = np.array([False, True, True])
        else:
            logits = pt.concatenate([pt.zeros_like(eta), eta, pt.zeros_like(eta)], axis=1)
            logp, random = zero_inflated_beta_logp, zero_inflated_beta_random
            inflated = np.array([False, True, False])
        mu = pm.math.sigmoid(ac + pt.dot(factor, lc.T))
        pm.CustomDist('y_cont', mu * phi, (1-mu) * phi, logits, inflated,
                      logp=logp, random=random,
                      observed=data['frame'][c].to_numpy(), dims=('obs', 'continuous'))
        pm.Bernoulli('y_binary', logit_p=ab + pt.dot(factor, lb.T),
                     observed=data['yb'], dims=('obs', 'binary'))
    return model
