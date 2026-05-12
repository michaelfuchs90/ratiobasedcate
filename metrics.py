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
def qini_coefficient(tau_pred: np.ndarray, W: np.ndarray, Y: np.ndarray) -> float:
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


def uplift_at_k(tau_pred: np.ndarray, W: np.ndarray, Y: np.ndarray, k: float) -> float:
    """
    Uplift@k: Empirical ratio in top-k% relative to overall.

    Values > 1 indicate good ranking (top-k outperforms average).
    """
    n_top = max(1, int(len(tau_pred) * k))
    top_idx = np.argsort(-tau_pred)[:n_top]

    # Top-k empirical ratio
    mask_t, mask_c = W[top_idx] == 1, W[top_idx] == 0
    if mask_t.sum() == 0 or mask_c.sum() == 0:
        return np.nan

    tau_top = _safe_divide(
        np.array([Y[top_idx][mask_t].mean()]),
        np.array([Y[top_idx][mask_c].mean()]),
        default=np.nan
    )[0]
    if not np.isfinite(tau_top):
        return np.nan

    # Overall empirical ratio
    tau_all = _safe_divide(Y[W == 1].mean(), Y[W == 0].mean(), default=1)

    return _safe_divide(np.array([tau_top]), np.array([tau_all]), default=1)[0]


# =============================================================================
# Calibration Metrics
# =============================================================================
def calibration_error(tau_pred: np.ndarray, W: np.ndarray, Y: np.ndarray,
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




def calibration_slope(tau_pred: np.ndarray, W: np.ndarray, Y: np.ndarray,
                      n_bins: int = DEFAULT_N_BINS) -> float:
    """
    Calibration Slope: Regression of log(τ_empirical) on log(τ_predicted).

    Ideal = 1.0. Values < 1 indicate overconfidence.
    """
    bin_edges = np.unique(np.percentile(tau_pred, np.linspace(0, 100, n_bins + 1)))
    bin_idx = np.digitize(tau_pred, bin_edges[1:-1])

    log_pred, log_emp = [], []

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
        log_pred.append(np.log(tau_hat))
        log_emp.append(np.log(tau_emp))

    if len(log_pred) < 2:
        return np.nan
    try:
        result = stats.linregress(log_pred, log_emp).slope
    except Exception:
        result = np.nan
 
    return result


# =============================================================================
# Ground Truth Metrics
# =============================================================================
def rmse_tau(tau_pred: np.ndarray, tau_true: np.ndarray) -> float:
    """RMSE on original scale."""
    return np.sqrt(np.mean((tau_pred - tau_true) ** 2))


def rmse_log_tau(tau_pred: np.ndarray, tau_true: np.ndarray) -> float:
    """RMSE on log scale (appropriate for ratios)."""
    log_pred = np.log(np.clip(tau_pred, EPS, None))
    log_true = np.log(np.clip(tau_true, EPS, None))
    return np.sqrt(np.mean((log_pred - log_true) ** 2))


def mae_tau(tau_pred: np.ndarray, tau_true: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(tau_pred - tau_true))


def spearman_tau(tau_pred: np.ndarray, tau_true: np.ndarray) -> float:
    """Spearman rank correlation."""
    return stats.spearmanr(tau_pred, tau_true)[0]


def pearson_tau(tau_pred: np.ndarray, tau_true: np.ndarray) -> float:
    """Pearson correlation."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", stats.NearConstantInputWarning)
        return stats.pearsonr(tau_pred, tau_true)[0]


# =============================================================================
# Main Evaluation Function
# =============================================================================
def evaluate_predictions(
    tau_pred: np.ndarray,
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
        'qini': qini_coefficient(tau_pred, W, Y),
        f'uplift@{k}': uplift_at_k(tau_pred, W, Y, k),
        # Calibration
        'cal_error': calibration_error(tau_pred, W, Y, n_bins),
        'cal_slope': calibration_slope(tau_pred, W, Y, n_bins),
    }

    # Ground truth (synthetic only)
    if tau_true is not None:
        results.update({
            'rmse': rmse_tau(tau_pred, tau_true),
            'rmse_log': rmse_log_tau(tau_pred, tau_true),
            'mae': mae_tau(tau_pred, tau_true),
            'spearman': spearman_tau(tau_pred, tau_true),
            'pearson': pearson_tau(tau_pred, tau_true),
        })

    return results


# =============================================================================
# Test
# =============================================================================
if __name__ == '__main__':
    np.random.seed(42)
    n = 5000

    # Synthetic data
    tau_true = 1 + np.random.exponential(0.5, n)
    tau_pred = tau_true + np.random.normal(0, 0.3, n)
    tau_pred = np.clip(tau_pred, 0.1, None)

    W = np.random.binomial(1, 0.5, n)
    mu0 = 0.1
    mu1 = mu0 * tau_true
    Y = np.where(W == 1, np.random.binomial(1, np.clip(mu1, 0, 1)), 
                         np.random.binomial(1, mu0))

    results = evaluate_predictions(tau_pred, W, Y, tau_true)

    print("Evaluation Results")
    print("=" * 40)
    for metric, value in results.items():
        print(f"{metric:15} {value:.4f}")
