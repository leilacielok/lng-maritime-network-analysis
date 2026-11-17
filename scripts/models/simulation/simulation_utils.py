"""Shared helpers for the rank-two simulation experiment.

Place this directory at scripts/models/simulation/; imports from the existing
scripts/models/baseline.py and common.py are resolved automatically.
"""
from pathlib import Path
import sys
import json
import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[1]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))


def sigmoid(x):
    x = np.asarray(x, dtype=float)
    return np.exp(-np.logaddexp(0, -x))


def save_json(path, value):
    def convert(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(value, indent=2, default=convert), encoding='utf-8')


def load_dataset(directory):
    import pandas as pd
    directory = Path(directory)
    meta = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
    frame = pd.read_csv(directory / 'observations.csv')
    x = np.load(directory / 'x.npy')
    yb = frame[meta['binary_responses']].to_numpy(dtype=int)
    return {
        'frame': frame, 'x': x, 'yb': yb,
        'yc': frame[meta['continuous_responses']].to_numpy(dtype=float),
        'audit': {'predictors': meta['predictors'],
                  'continuous_responses': meta['continuous_responses'],
                  'binary_responses': meta['binary_responses']},
        'prior_profile': meta['prior_profile'],
        'precision_prior_sd': meta['precision_prior_sd'],
    }, meta


def load_truth(directory):
    return dict(np.load(Path(directory) / 'truth.npz'))
