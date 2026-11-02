"""Equivalent rank-one node-effects model in mean/contrast coordinates.

Place beside estimation.py, baseline.py, common.py and node_effects.py.
Only this runner changes; existing source files are not modified.
Uses the existing estimation CLI, prior profiles, predictive checks and exports.
Default target_accept remains 0.80. Results default to
outputs/model/node_effects_1/<kind>/<run>/<kind>/.

For N nodes choose an orthonormal N x (N-1) contrast matrix Q:
  sigma ~ HalfNormal(original_scale)
  node_mean | sigma ~ Normal(0, sigma/sqrt(N))
  u ~ Normal(0,I); delta = sigma Q u; mu_node = node_mean + delta.
Thus Cov(mu_node | sigma) = sigma^2 I, exactly as in the original.
Sample g = factor - node_mean and A = intercept + loading*node_mean.
Conditional priors A | loading,node_mean preserve the original intercept priors.
All original parameter names are reconstructed for existing review/export code.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np


def contrast_matrix(n):
    """Helmert orthonormal columns: Q.T Q=I, Q.T 1=0."""
    if n < 2:
        raise ValueError('At least two estimation nodes are required')
    q = np.zeros((n, n-1))
    for k in range(1, n):
        q[:k, k-1] = 1 / np.sqrt(k*(k+1))
        q[k, k-1] = -k / np.sqrt(k*(k+1))
    return q


def build_model(data, factor_builder=None):
    import pymc as pm
    import pytensor.tensor as pt
    from baseline import prior_specification
    from common import (one_inflated_beta_logp, one_inflated_beta_random,
                        zero_inflated_beta_logp, zero_inflated_beta_random)
    kind = 'terminal' if 'exporter_lag1' in data['audit']['predictors'] else 'chokepoint'
    terminal = kind == 'terminal'
    prior = prior_specification(kind, data.get('prior_profile', 'original'))
    continuous = data['audit']['continuous_responses']
    boundary = continuous[1:] if terminal else ['betweenness']
    n = len(data['node_ids'])
    coords = dict(obs=np.arange(len(data['frame'])), node=data['node_ids'],
                  node_contrast=np.arange(n-1), predictor=data['audit']['predictors'],
                  continuous=continuous, other_continuous=continuous[1:],
                  binary=data['audit']['binary_responses'], boundary=boundary)
    with pm.Model(coords=coords) as model:
        beta = pm.Normal('beta', 0, prior['beta_sd'], dims='predictor')
        sigma = pm.HalfNormal('sigma_mu', data['node_sd_prior'])
        m = pm.Normal('node_mean', 0, sigma/np.sqrt(n))
        u = pm.Normal('node_contrast_raw', 0, 1, dims='node_contrast')
        delta = pm.Deterministic('node_deviation', sigma*pt.dot(contrast_matrix(n), u), dims='node')
        mu_node = pm.Deterministic('mu_node', m+delta, dims='node')
        pm.Deterministic('mu_raw', mu_node/sigma, dims='node')
        covmean = pm.Deterministic('factor_covariate_mean', pt.dot(data['x'], beta), dims='obs')
        gmean = covmean + delta[data['node_index']]
        g = pm.Normal('factor_centered', gmean, 1, dims='obs')
        pm.Deterministic('factor', g+m, dims='obs')
        pm.Deterministic('factor_mean', gmean+m, dims='obs')
        pm.Deterministic('factor_residual', g-gmean, dims='obs')
        anchor = pm.HalfNormal('pagerank_loading', prior['pagerank_loading_sd'])
        other = pm.Normal('other_loadings', 0, prior['other_loading_sd'], dims='other_continuous')
        lc = pm.Deterministic('loading_cont', pt.concatenate([anchor[None], other]), dims='continuous')
        lb = pm.Normal('loading_binary', 0, prior['binary_loading_sd'], dims='binary')
        ac = pm.Normal('intercept_cont_centered',
                       np.asarray(prior['interior_intercept_mean'])+lc*m,
                       prior['interior_intercept_sd'], dims='continuous')
        ab = pm.Normal('intercept_binary_centered', lb*m,
                       prior['binary_intercept_sd'], dims='binary')
        pm.Deterministic('intercept_cont', ac-lc*m, dims='continuous')
        pm.Deterministic('intercept_binary', ab-lb*m, dims='binary')
        phi = pm.LogNormal('precision_cont', np.log(prior['precision_median']),
                           data['precision_prior_sd'], dims='continuous')
        suffix = 'one' if terminal else 'zero'
        loading = pm.Normal('loading_'+suffix, 0, prior['boundary_loading_sd'], dims='boundary')
        intercept = pm.Normal('intercept_'+suffix+'_centered',
                              prior['boundary_intercept_mean']+loading*m,
                              prior['boundary_intercept_sd'], dims='boundary')
        pm.Deterministic('intercept_'+suffix, intercept-loading*m, dims='boundary')
        eta = intercept+g[:,None]*loading
        logits = (pt.concatenate([pt.zeros_like(g[:,None]), eta], axis=1) if terminal else
                  pt.concatenate([pt.zeros_like(eta),eta,pt.zeros_like(eta)],axis=1))
        means = pm.math.sigmoid(ac+g[:,None]*lc)
        pm.CustomDist('y_cont', means*phi, (1-means)*phi, logits,
                      np.array([False,True,True] if terminal else [False,True,False]),
                      logp=one_inflated_beta_logp if terminal else zero_inflated_beta_logp,
                      random=one_inflated_beta_random if terminal else zero_inflated_beta_random,
                      observed=data['frame'][continuous].to_numpy(), dims=('obs','continuous'))
        pm.Bernoulli('y_binary', logit_p=ab+g[:,None]*lb,
                     observed=data['yb'], dims=('obs','binary'))
    return model


def convert_initial_values(starts, data):
    """Same original-coordinate starts with initial node mean/effects zero."""
    converted = []
    for original in starts:
        point = dict(original)
        point['factor_centered'] = point.pop('factor')
        for name in ('intercept_cont','intercept_binary','intercept_one','intercept_zero'):
            if name in point:
                point[name+'_centered'] = point.pop(name)
        point.update(node_mean=0., node_contrast_raw=np.zeros(len(data['node_ids'])-1), sigma_mu=1.)
        converted.append(point)
    return converted


def main():
    import estimation
    import node_effects
    from common import OUTPUT_DIR
    if '--model' in sys.argv or any(arg.startswith('--model=') for arg in sys.argv):
        raise SystemExit('This runner always uses node_effects; omit --model.')
    # The original CLI remains responsible for validation, unique run folders,
    # posterior serialization, diagnostics and findings. Overrides are in memory.
    original_prepare = node_effects.prepare_node_effects
    def prepare(data, kind, prior_scale=1.):
        original_prepare(data, kind, prior_scale)
        data['audit']['node_effect_specification']['sampling_parameterization'] = {
            'name':'mean_contrasts_shifted_intercepts', 'equivalent_original_prior':True,
            'node_mean':'Normal(0, sigma_mu/sqrt(N))',
            'deviations':'sigma_mu * Q * Normal(0,I), Q orthonormal contrasts',
            'intercepts':'A | m, loading ~ Normal(original_mean + loading*m, original_sd)',
            'original_mu_node_sum_to_zero':False}
        return data
    node_effects.prepare_node_effects = prepare
    estimation.build_model = build_model
    for name in ('initial_values','chokepoint_initial_values'):
        original = getattr(estimation,name)
        def wrapper(data, chains, seed, _original=original):
            return convert_initial_values(_original(data,chains,seed),data)
        setattr(estimation,name,wrapper)
    original_fit = estimation.fit_model
    def fit(data, kind, args, destination):
        # Record source identity without copying scripts into the output directory.
        source = Path(__file__).resolve()
        config_path = destination.parent/'run_config.json'
        config = json.loads(config_path.read_text(encoding='utf-8'))
        config['sampling_parameterization'] = 'mean_contrasts_shifted_intercepts'
        config['source_sha256'][source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
        config_path.write_text(json.dumps(config,indent=2),encoding='utf-8')
        return original_fit(data,kind,args,destination)
    estimation.fit_model = fit
    # Reuse the existing CLI with a distinct model directory. Guard the
    # source replacement so an incompatible estimation.py fails explicitly.
    import inspect
    source = inspect.getsource(estimation.main)
    old_route = 'args.output_dir / "node_effects"'
    if old_route not in source:
        raise RuntimeError('Cannot locate node_effects output routing in estimation.py')
    source = source.replace(old_route, 'args.output_dir / "node_effects_1"')
    namespace = dict(estimation.__dict__)
    exec(compile(source, estimation.__file__, 'exec'), namespace)
    namespace['main'](default_model='node_effects')


if __name__ == '__main__':
    main()
