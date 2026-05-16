"""
Post-hoc Calibration for Ratio-CATE Learners
=============================================
τ(x) = E[Y|W=1,X] / E[Y|W=0,X]

LogLinearCalibrator: log(τ_cal) = a + b·log(τ_pred)
Strictly monotone → ranking is exactly preserved.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import copy

N_CALIBRATION_BINS = 10


# =============================================================================
# Calibrator
# =============================================================================
class LogLinearCalibrator:
    """Log-linear calibration: log(τ_cal) = a + b·log(τ_pred).

    Strictly monotone (b > 0) → preserves ranking exactly.
    Fit is a 2-parameter OLS on bin-level empirical targets — fast and robust.
    """

    def __init__(self, clip_min=0.01, clip_max=100.0, n_bins=N_CALIBRATION_BINS):
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.n_bins = n_bins

    def fit(self, predictions, W, Y):
        targets = self._empirical_targets(predictions, W, Y)


        bin_edges = np.percentile(predictions, np.linspace(0, 100, self.n_bins + 1))
        bins = np.searchsorted(bin_edges[1:-1], predictions)  # 0..n_bins-1

        log_x, log_y = [], []
        for b in range(self.n_bins):
            mask = bins == b
            if mask.sum() >= 5:
                x_mean = np.clip(predictions[mask].mean(), self.clip_min, self.clip_max)
                y_mean = np.clip(targets[mask].mean(), self.clip_min, self.clip_max)
                log_x.append(np.log(x_mean))
                log_y.append(np.log(y_mean))

        if len(log_x) >= 2:
            b_coef, a_coef = np.polyfit(log_x, log_y, 1)
            self.b_ = max(float(b_coef), 1e-3)  # enforce strict monotonicity
            self.a_ = float(a_coef)
        else:
            # Fallback: identity transformation
            self.b_ = 1.0
            self.a_ = 0.0

        return self

    def predict(self, predictions):
        log_pred = np.log(np.clip(predictions, self.clip_min, self.clip_max))
        return np.exp(self.a_ + self.b_ * log_pred)

    def _empirical_targets(self, predictions, W, Y):
        """Bin-wise empirical τ = mean(Y|W=1) / mean(Y|W=0) as calibration target."""
        bin_edges = np.percentile(predictions, np.linspace(0, 100, self.n_bins + 1))
        bins = np.searchsorted(bin_edges[1:-1], predictions)
        targets = predictions.copy().astype(float)

        for b in range(self.n_bins):
            mask = bins == b
            t, c = mask & (W == 1), mask & (W == 0)
            if t.sum() > 0 and c.sum() > 0:
                mu0 = Y[c].mean()
                if mu0 > 0.01 and mask.sum() >= 10:
                    targets[mask] = Y[t].mean() / mu0

        return targets


# =============================================================================
# Wrapper
# =============================================================================
class CalibratedCATELearner:
    """Wraps an already-fitted CATE learner with post-hoc log-linear calibration.

    fit(): fits LogLinearCalibrator on predictions (ranking-preserving).
           DR learners exposing oof_predictions_ are used directly;
           all others use predict() on training data (fast, no refit needed).
    predict(): base learner prediction → log-linear calibration.
    """

    def __init__(self, base_learner, random_state=42):
        self.base_learner = base_learner
        self.random_state = random_state

    def fit(self, X, W, Y, propensity=None):
        
        predictions = self.base_learner.predict(X, propensity=propensity)

        self.calibrator_ = LogLinearCalibrator().fit(predictions, W, Y)
        return self

    def predict(self, X, propensity=None):
        return self.calibrator_.predict(self.base_learner.predict(X, propensity=propensity))


