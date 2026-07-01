"""Load the persisted failure predictor and score a live trajectory.

The predictor is the feature logistic regression trained by
predictor/train_predictor.py (it beat every LoRA variant at this data scale).
score() returns P(doomed): probability the trajectory will NOT eventually pass,
reconstructing the exact 14-dim vector the model was trained on (same order as
dataset.encode: numeric feature_cols + one-hot failure_type).
"""

from __future__ import annotations

import joblib
import numpy as np

_ARTIFACT = "predictor/predictor.joblib"
_cache: dict = {}


class FailurePredictor:
    def __init__(self, bundle: dict):
        self.model = bundle["model"]
        self.feature_cols = bundle["feature_cols"]
        self.failure_types = bundle["failure_types"]
        self._ft_index = {t: i for i, t in enumerate(self.failure_types)}

    def _vector(self, feats: dict) -> np.ndarray:
        row = [float(feats.get(c, 0)) for c in self.feature_cols]
        onehot = [0.0] * len(self.failure_types)
        onehot[self._ft_index.get(feats.get("failure_type", ""), 0)] = 1.0
        return np.array([row + onehot])

    def score(self, feats: dict) -> float:
        """P(doomed): probability the trajectory will NOT eventually pass."""
        return float(self.model.predict_proba(self._vector(feats))[0, 1])


def get_predictor(artifact: str = _ARTIFACT) -> FailurePredictor:
    if artifact not in _cache:
        _cache[artifact] = FailurePredictor(joblib.load(artifact))
    return _cache[artifact]