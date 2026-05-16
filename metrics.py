"""
Evaluation Metrics for Ratio-CATE Estimation
============================================
τ(x) = E[Y|W=1,X] / E[Y|W=0,X]
"""

import warnings
import numpy as np
from typing import Optional
from scipy import stats


# =============================================================================
# Config
# =============================================================================
DEFAULT_K_VALUE = 0.2
DEFAULT_N_BINS = 10
EPS = 1e-10  # Numerical stability


# =============================================================================
# Helper
# =============================================================================
def _safe_divide(a: np.ndarray, b: np.ndarray, default: float = 1.0) -> np.ndarray:
    """Safe division with default for zero/invalid denominators."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(b > EPS, a / b, default)
    return np.where(np.isfinite(result), result, default)


# =============================================================================
# Ranking Metrics
# =============================================================================
def qini_coefficient_ratio(tau_pred: np.ndarray, W: np.ndarray, Y: np.ndarray) -> float:
    """
    Ratio-Qini Coefficient: Area between cumulative ratio curve and baseline.

    Higher = better ranking of treatment effects.
    """
    n = len(tau_pred)
    order = np.argsort(-tau_pred)
    W_sorted, Y_sorted = W[order], Y[order]

    # Cumulative rates
    cum_n1 = np.cumsum(W_sorted)
    cum_n0 = np.cumsum(1 - W_sorted)
    cum_y1 = np.cumsum(Y_sorted * W_sorted)
    cum_y0 = np.cumsum(Y_sorted * (1 - W_sorted))

    rate1 = _safe_divide(cum_y1, cum_n1, default=0)
    rate0 = _safe_divide(cum_y0, cum_n0, default=1)
    ratio_curve = _safe_divide(rate1, rate0, default=1)

    # AUC minus baseline
    auc = np.trapezoid(ratio_curve, np.linspace(0, 1, n))
    baseline = _safe_divide(Y[W == 1].mean(), Y[W == 0].mean(), default=1)

    return auc - baseline

def qini_coefficient_difference(tau_pred: np.ndarray, W: np.ndarray, Y: np.ndarray) -> float:
    """
    Difference-Qini Coefficient (standard cumulative-gain formulation).

    For units sorted by descending tau_pred, computes the cumulative
    treated-vs-control gain curve

        g(s) = (cum Y_t / n_t) - (cum Y_c / n_c)

    and returns its AUC minus the random-targeting baseline (the
    triangle from (0,0) to (1, overall_gain)).

    Higher = better ranking of treatment effects.
    """
    n = len(tau_pred)
    order = np.argsort(-tau_pred)
    W_sorted, Y_sorted = W[order], Y[order]

    n_t = max(int(W.sum()), 1)
    n_c = max(int((1 - W).sum()), 1)

    cum_y1 = np.cumsum(Y_sorted * W_sorted) / n_t
    cum_y0 = np.cumsum(Y_sorted * (1 - W_sorted)) / n_c
    gain_curve = cum_y1 - cum_y0

    s = np.linspace(0, 1, n)
    auc = np.trapezoid(gain_curve, s)

    overall_gain = Y[W == 1].mean() - Y[W == 0].mean()
    baseline_auc = 0.5 * overall_gain  # triangle (0,0) -> (1, overall_gain)

    return auc - baseline_auc


# =============================================================================
# Calibration Metrics
# =============================================================================
def calibration_error_ratio(tau_pred: np.ndarray, W: np.ndarray, Y: np.ndarray,
                      n_bins: int = DEFAULT_N_BINS) -> float:
    """
    Calibration Error: Weighted mean absolute log-error across bins.

    Lower = better calibrated predictions. If tau_pred is constant (e.g.,
    all probabilities clipped to the same bound), we fall back to a single
    bucket comparing the constant prediction to the overall empirical ratio.
    """
    bin_edges = np.unique(np.percentile(tau_pred, np.linspace(0, 100, n_bins + 1)))

    # Constant tau_pred → single-bucket fallback.
    if len(bin_edges) < 2:
        mask_t, mask_c = W == 1, W == 0
        if mask_t.sum() == 0 or mask_c.sum() == 0:
            return np.nan
        mu_t, mu_c = Y[mask_t].mean(), Y[mask_c].mean()
        tau_hat = tau_pred.mean()
        if mu_c <= EPS or mu_t <= EPS or tau_hat <= EPS:
            return np.nan
        return np.exp(np.abs(np.log(tau_hat) - np.log(mu_t / mu_c)))

    bin_idx = np.digitize(tau_pred, bin_edges[1:-1])

    errors, weights = [], []

    for b in range(len(bin_edges) - 1):
        mask = bin_idx == b
        mask_t, mask_c = mask & (W == 1), mask & (W == 0)

        if mask_t.sum() == 0 or mask_c.sum() == 0:
            continue

        mu_t, mu_c = Y[mask_t].mean(), Y[mask_c].mean()
        tau_hat = tau_pred[mask].mean()

        if mu_c <= EPS or mu_t <= EPS or tau_hat <= EPS:
            continue

        tau_emp = mu_t / mu_c
        errors.append(np.abs(np.log(tau_hat) - np.log(tau_emp)))
        weights.append(mask.sum())

    return np.exp(np.average(errors, weights=weights)) if errors else np.nan

def calibration_error_difference(tau_pred: np.ndarray, W: np.ndarray, Y: np.ndarray,
                      n_bins: int = DEFAULT_N_BINS) -> float:
    """
    Calibration Error (difference scale): weighted mean absolute error
    between predicted difference CATE and empirical mean difference per bin.

    Lower = better calibrated predictions. If tau_pred is constant (e.g.,
    all probabilities clipped to the same bound), we fall back to a single
    bucket comparing the constant prediction to the overall empirical
    difference.
    """
    bin_edges = np.unique(np.percentile(tau_pred, np.linspace(0, 100, n_bins + 1)))

    # Constant tau_pred → single-bucket fallback.
    if len(bin_edges) < 2:
        mask_t, mask_c = W == 1, W == 0
        if mask_t.sum() == 0 or mask_c.sum() == 0:
            return np.nan
        mu_t, mu_c = Y[mask_t].mean(), Y[mask_c].mean()
        tau_hat = tau_pred.mean()
        return np.abs(tau_hat - (mu_t - mu_c))

    bin_idx = np.digitize(tau_pred, bin_edges[1:-1])

    errors, weights = [], []

    for b in range(len(bin_edges) - 1):
        mask = bin_idx == b
        mask_t, mask_c = mask & (W == 1), mask & (W == 0)

        if mask_t.sum() == 0 or mask_c.sum() == 0:
            continue

        mu_t, mu_c = Y[mask_t].mean(), Y[mask_c].mean()
        tau_hat = tau_pred[mask].mean()

        if mu_c <= EPS or mu_t <= EPS or tau_hat <= EPS:
            continue

        tau_emp = mu_t - mu_c
        errors.append(np.abs(tau_hat - tau_emp))
        weights.append(mask.sum())

    return (np.average(errors, weights=weights)) if errors else np.nan



# =============================================================================
# Main Evaluation Function
# =============================================================================
def evaluate_predictions(
    tau_ratio_pred: np.ndarray,
    tau_difference_pred: np.ndarray,
    W: np.ndarray,
    Y: np.ndarray,
    tau_true: Optional[np.ndarray] = None,
    k: float = DEFAULT_K_VALUE,
    n_bins: int = DEFAULT_N_BINS
) -> dict[str, float]:
    """
    Evaluate CATE predictions with all relevant metrics.

    Returns
    -------
    dict[str, float]
        All computed metrics
    """
    results = {
        # Ranking
        'qini_ratio':       qini_coefficient_ratio(tau_ratio_pred, W, Y),
        'qini_difference':  qini_coefficient_difference(tau_difference_pred, W, Y),
        # Calibration
        'cal_error_ratio':       calibration_error_ratio(tau_ratio_pred, W, Y, n_bins),
        'cal_error_difference':  calibration_error_difference(tau_difference_pred, W, Y, n_bins),
    }

    return results


