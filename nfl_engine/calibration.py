"""Win-probability calibration.

Platt scaling (a logistic fit on the raw probability) rather than isotonic
regression. Isotonic looked better in-sample but overfits: on a 2020+ holdout
it scored worse than no calibration at all, while Platt improved on both. It
also returns a smooth continuous probability instead of ~30 plateaus, which
matters now that probabilities are ranked against market prices.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression


class PlattCalibrator:
    """Uniform .predict() interface over a logistic calibration fit."""

    def __init__(self, C: float = 1e6):
        self.model = LogisticRegression(C=C)

    def fit(self, raw_prob, outcome) -> "PlattCalibrator":
        x = np.asarray(raw_prob, dtype=float).reshape(-1, 1)
        self.model.fit(x, np.asarray(outcome, dtype=int))
        return self

    def predict(self, raw_prob):
        x = np.asarray(raw_prob, dtype=float).reshape(-1, 1)
        return self.model.predict_proba(x)[:, 1]
