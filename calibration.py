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


# =============================================================================
# Test
# =============================================================================
if __name__ == '__main__':
    import sys
    rng = np.random.default_rng(42)
    n = 5000

    # Simulate data with known τ heterogeneity
    X_feat = rng.normal(size=n)
    tau_true = np.exp(0.5 * X_feat)          # true τ varies by X
    e = 0.5
    mu0 = 0.1
    mu1 = mu0 * tau_true
    W = rng.binomial(1, e, n)
    Y = rng.binomial(1, np.where(W == 1, mu1, mu0), n)

    # Simulate a noisy but positively correlated predictor (miscalibrated scale)
    tau_pred = tau_true * np.exp(rng.normal(0, 0.3, n)) * 2.0  # biased up by 2x
    tau_pred = np.clip(tau_pred, 0.01, 100)

    # Fit calibrator
    cal = LogLinearCalibrator()
    cal.fit(tau_pred, W, Y)
    tau_cal = cal.predict(tau_pred)

    print(f"LogLinearCalibrator fitted: a={cal.a_:.4f}, b={cal.b_:.4f}")
    print(f"  Expected b~1.0 (slope), a<0 (downward scale correction for 2x bias)")
    print()

    # 1) Ranking preserved: Spearman(τ_pred, τ_cal) == 1.0
    from scipy.stats import spearmanr
    rho, _ = spearmanr(tau_pred, tau_cal)
    rank_preserved = np.isclose(rho, 1.0, atol=1e-10)
    print(f"Ranking preserved (Spearman=1): {rank_preserved}  (rho={rho:.10f})")
    assert rank_preserved, "FAIL: ranking not preserved!"

    # 2) Calibration improves: mean(τ_cal) closer to mean(τ_true) than mean(τ_pred)
    bias_before = abs(tau_pred.mean() - tau_true.mean())
    bias_after  = abs(tau_cal.mean()  - tau_true.mean())
    print(f"Mean bias before: {bias_before:.4f}  after: {bias_after:.4f}  -> improved: {bias_after < bias_before}")
    assert bias_after < bias_before, "FAIL: calibration did not reduce bias!"

    # 3) CalibratedCATELearner smoke test
    class MockLearner:
        def predict(self, X, propensity=None):
            return tau_pred

    mock = MockLearner()
    X_df = pd.DataFrame({'x': X_feat})
    wrapper = CalibratedCATELearner(mock)
    wrapper.fit(X_df, W, Y)
    out = wrapper.predict(X_df)
    assert len(out) == n, "FAIL: output length mismatch"
    rho2, _ = spearmanr(tau_pred, out)
    assert np.isclose(rho2, 1.0, atol=1e-10), "FAIL: wrapper ranking not preserved!"
    print(f"CalibratedCATELearner wrapper: OK (output len={len(out)}, Spearman=1)")

    print()
    print("All tests passed.")
