# Model scripts

- `common.py`: shared panel preparation, response definitions and bounded likelihood functions.
- `baseline.py`: rank-one model, prior profiles, initialization.
- `estimation.py`: CLI, sampling, provenance and output routing.
- `diagnostics.py`: sampling and predictive diagnostics.
- `findings.py`: parameter summaries and latent scores.
- `review.py`: retrospective diagnostics and response curves from existing posterior files.

From repository root:

```bash
python scripts/models/estimation.py --model-group baseline --node-type terminal --check-only
python scripts/models/estimation.py --model-group baseline --node-type terminal --prior-only
python scripts/models/estimation.py --model-group CON --node-type terminal
python scripts/models/estimation.py --model-group RS --node-type chokepoint
python scripts/models/estimation.py --model-group baseline --node-type terminal --tune 5000 --draws 10000
```

The last command is a longer-chain **baseline fit**, not a distinct statistical model.
Rank-2 and node-intercept extensions are not implemented: they require new model definitions, identification choices and compatible diagnostics.

Existing runs are not moved automatically. New runs archive all model source modules.
