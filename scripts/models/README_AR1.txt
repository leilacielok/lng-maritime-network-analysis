AR(1) extension — installation and testing
===========================================
Copy these .py files into scripts/models/ (back up existing files first).
Do not rename them with numeric suffixes.

Model: rho ~ Uniform(-1,1), stationary marginal Var(z)=1.
Non-centred innovations u ~ Normal(0,1). For the first selected month
of each node z=u; for later selected observations separated by d calendar
months, z_new=rho**d*z_previous+sqrt(1-rho**(2*d))*u_new.
Only the current estimation observations are included; no missing-month
latent variables or response likelihoods are introduced.

Validate panel and AR1 indices:
python scripts/models/estimation.py --node-type terminal --rank 1 --residual-process ar1 --check-only
python scripts/models/estimation.py --node-type chokepoint --rank 1 --residual-process ar1 --check-only

SHORT smoke test, baseline terminals:
python scripts/models/estimation.py --node-type terminal --rank 1 --residual-process ar1 --model baseline --low-ram --cores 1 --chains 2 --tune 50 --draws 50 --prior-draws 5 --run-name ar1_smoke_terminal

SHORT smoke test, node effects chokepoints:
python scripts/models/estimation.py --node-type chokepoint --rank 1 --residual-process ar1 --model node_effects --low-ram --cores 1 --chains 2 --tune 50 --draws 50 --prior-draws 5 --run-name ar1_smoke_cp_node

Run remaining combinations likewise, with a unique --run-name.
For real estimates use >=4 chains and adequate tuning/draws; smoke-test
numbers are NOT adequate for inference.

Review (example, substitute the actual run directory):
python scripts/models/review.py --run-dir outputs/model/baseline/ar1/terminal/ar1_smoke_terminal --node-type terminal

CAVEATS:
- Current container has incompatible installed PyMC/ArviZ versions; PyMC
  end-to-end sampling could not be tested here. Static compilation and
  PyTensor scan/gradient tests succeeded. Test in your lng-model environment.
- AR1 low-RAM stores the observation-level deterministic factor, unlike
  other deterministics. This supports existing review.py but adds disk usage.
- The original scripts' model specifications and IID paths were preserved.
- Prior predictive is conditional on the selected observation panel only.
