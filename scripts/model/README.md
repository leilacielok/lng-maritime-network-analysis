# Simple criticality model

Run from the repository root. The existing panels are read from
`data/processed/model_ready/`, matching `build_model_ready_panels.py`.
They are filtered in memory; no new modelling panel is written.

```bash
python -m pip install -r scripts/model/requirements.txt
python scripts/model/simple_criticality_model.py --check-only
python scripts/model/simple_criticality_model.py --prior-only
python scripts/model/simple_criticality_model.py
```

Existing project dependencies include NumPy, pandas and matplotlib. Use
`--terminal-panel` / `--chokepoint-panel` for other input locations. Use
`--node-type terminal` or `--node-type chokepoint` to run one fit. The script
resolves default paths relative to itself, independently of the working directory.

## Selection and variables

Keep `active == 1`, `activity_lag1 == 1` and `lag_available == 1`. Exclude
missing terminal age, without changing the original start years. Other missing
model values cause an error. Supplied activity lags are checked against the
preceding calendar month in the full panel before selection. On the October 8
uploaded panels, the expected counts are 3,453 terminal observations and 1,163
chokepoint observations (these are not hard-coded constraints).

Terminal responses: role-specific PageRank, terminal counterparty HHI,
role-specific dependence and the three structural binary indicators.
Chokepoint responses: PageRank, weighted betweenness, flow share and the same
binary indicators. HHI is continuous, not categorical.

Predictors: lagged neighbour activity share, lagged log1p mean voyage distance,
month sine/cosine; additionally terminal log1p capacity, age and a lagged exporter
dummy (importer reference). No country or throughput effects. A dummy for an
empty historical neighbourhood is unnecessary in these selected panels; an
unexpected undefined neighbour share raises an error rather than being imputed.

## Preliminary statistical specification

Separate Bayesian single-factor regressions are estimated for each node type.
The factor has mean `X beta` and independent unit Normal residuals. Continuous
responses have Normal likelihoods after sample standardization; binary responses
have Bernoulli-logit likelihoods. Shared loadings give a covariate coefficient
matrix of rank at most one. The PageRank loading is positive; other loadings are
free. Numeric predictors are standardized; the role dummy remains 0/1.

This is a static baseline: no node random effects, temporal process, activity
component or cross-node dependence. The Gaussian continuous likelihood is an
approximation, not a bounded-response model. It cannot reproduce the exact
HHI/dependence mass at one or betweenness mass at zero. Predictive checks report
the observed boundary masses and replicated values outside [0,1]. Review prior
checks before fitting and posterior checks before interpreting results. The
factor's interpretation as criticality must be justified by the loadings and fit;
terminal and chokepoint scores are not directly comparable across models.

## Outputs and validation

Each invocation creates a new timestamped directory under
`outputs/model/simple_criticality/`. Outputs include exclusion counts, input
hashes, scaling constants, prior/posterior NetCDF draws, parameter summaries,
sampling diagnostics, trace plots and predictive checks. `factor_scores.csv`
contains node/month keys, posterior factor scores, covariate-explained factor
means, and equal-tailed 95% intervals; it is a result file, not a new input panel.

Inspect divergences, R-hat and bulk/tail ESS for both parameters and latent
residuals. Failed basic checks generate a warning. These checks do not establish
model adequacy. Posterior predictive checks are in-sample checks, not a temporal
holdout evaluation. Short runs with `--draws 20 --tune 20 --chains 2` are only
execution tests, never thesis estimates. Guarded `main()` and one default core
support Windows execution.
