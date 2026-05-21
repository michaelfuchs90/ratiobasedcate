"""
Evaluation Metrics for Ratio-CATE Estimation
============================================
τ(x) = E[Y|W=1,X] / E[Y|W=0,X]
"""

import warnings
import numpy as np


DEFAULT_N_BINS = 10
EPS = 1e-10  # divide-by-zero guard for empty bins / zero outcomes


def _safe_divide(a: np.ndarray, b: np.ndarray, default: float = 1.0) -> np.ndarray:
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(b > EPS, a / b, default)
    return np.where(np.isfinite(result), result, default)


# Ranking metrics ============================================================
def qini_coefficient_ratio(tau_pred: np.ndarray, W: np.ndarray, Y: np.ndarray) -> float:
    """Ratio-Qini coefficient (cumulative ratio-curve AUC minus baseline). Higher = better."""
    n = len(tau_pred)
    order = np.argsort(-tau_pred)
    W_sorted, Y_sorted = W[order], Y[order]

    cum_n1 = np.cumsum(W_sorted)
    cum_n0 = np.cumsum(1 - W_sorted)
    cum_y1 = np.cumsum(Y_sorted * W_sorted)
    cum_y0 = np.cumsum(Y_sorted * (1 - W_sorted))

    rate1 = _safe_divide(cum_y1, cum_n1, default=0)
    rate0 = _safe_divide(cum_y0, cum_n0, default=1)
    ratio_curve = _safe_divide(rate1, rate0, default=1)

    auc = np.trapezoid(ratio_curve, np.linspace(0, 1, n))
    baseline = _safe_divide(Y[W == 1].mean(), Y[W == 0].mean(), default=1)

    return auc - baseline


def qini_coefficient_difference(tau_pred: np.ndarray, W: np.ndarray, Y: np.ndarray) -> float:
    """Difference-Qini (cumulative-gain AUC minus random-targeting baseline)."""
    n = len(tau_pred)
    order = np.argsort(-tau_pred)
    W_sorted, Y_sorted = W[order], Y[order]

    n_t = max(int(W.sum()), 1)
    n_c = max(int((1 - W).sum()), 1)

    cum_y1 = np.cumsum(Y_sorted * W_sorted) / n_t
    cum_y0 = np.cumsum(Y_sorted * (1 - W_sorted)) / n_c
    gain_curve = cum_y1 - cum_y0

    auc = np.trapezoid(gain_curve, np.linspace(0, 1, n))
    overall_gain = Y[W == 1].mean() - Y[W == 0].mean()
    baseline_auc = 0.5 * overall_gain  # area under random-targeting line (0,0) -> (1, overall_gain)

    return auc - baseline_auc


# Calibration metrics ========================================================
def calibration_error_ratio(tau_pred: np.ndarray, W: np.ndarray, Y: np.ndarray,
                            n_bins: int = DEFAULT_N_BINS) -> float:
    """Bin-weighted mean absolute log-error vs. empirical τ per bin. Lower = better."""
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

    if not errors:
        warnings.warn(
            'calibration_error_ratio: every bin lacked treated/control overlap '
            'or had near-zero outcomes; returning NaN.',
            RuntimeWarning, stacklevel=2,
        )
        return np.nan
    return np.exp(np.average(errors, weights=weights))


def calibration_error_difference(tau_pred: np.ndarray, W: np.ndarray, Y: np.ndarray,
                                 n_bins: int = DEFAULT_N_BINS) -> float:
    """Weighted mean absolute error vs. empirical mean difference per bin. Lower = better."""
    bin_edges = np.unique(np.percentile(tau_pred, np.linspace(0, 100, n_bins + 1)))

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

    if not errors:
        warnings.warn(
            'calibration_error_difference: every bin lacked treated/control overlap '
            'or had near-zero outcomes; returning NaN.',
            RuntimeWarning, stacklevel=2,
        )
        return np.nan
    return np.average(errors, weights=weights)


def evaluate_predictions(
    tau_ratio_pred: np.ndarray,
    tau_difference_pred: np.ndarray,
    W: np.ndarray,
    Y: np.ndarray,
    n_bins: int = DEFAULT_N_BINS,
) -> dict[str, float]:
    """Compute Qini and calibration error on both ratio and difference scales."""
    return {
        'qini_ratio':           qini_coefficient_ratio(tau_ratio_pred, W, Y),
        'qini_difference':      qini_coefficient_difference(tau_difference_pred, W, Y),
        'cal_error_ratio':      calibration_error_ratio(tau_ratio_pred, W, Y, n_bins),
        'cal_error_difference': calibration_error_difference(tau_difference_pred, W, Y, n_bins),
    }
