Python implementation; not R coda. Global parameters only by default.
Tests are per chain, using retained posterior draws without concatenation or thinning.
Geweke: first 10% vs final 50%; thresholds are unadjusted for multiple comparisons.
Heidelberger-Welch: stationarity may discard up to 50%; this does not modify the posterior.
Half-width results apply to the retained segment and are NA if stationarity fails.
A relative half-width failure for a mean near zero need not imply poor mixing.
Pass is a diagnostic outcome, not proof of convergence; inspect R-hat, ESS and divergences too.
