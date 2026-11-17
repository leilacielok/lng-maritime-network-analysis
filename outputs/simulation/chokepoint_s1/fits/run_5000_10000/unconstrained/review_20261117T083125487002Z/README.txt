Simulation review (no refitting).
parameter_summary.csv: posterior means, HDIs and MCMC diagnostics for global parameters.
parameter_recovery.csv: true values and equal-tailed 95% posterior intervals.
Raw beta, loadings and factors are rotation-sensitive, including in shrinkage.
gram_recovery.csv: rotation-invariant loading recovery; 95% equal-tailed intervals.
posterior_checks.csv / joint_predictive_checks.csv: conditional in-sample PPCs.
PPCs condition on latent factors inferred from the same observations, not out-of-sample predictions.
Geweke/HW use production diagnostics.py; not proof of convergence.
Interpret posterior recovery cautiously if divergences or poor mixing are present.
