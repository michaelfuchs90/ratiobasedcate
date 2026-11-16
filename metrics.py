"""
Evaluation Metrics for Ratio-CATE Estimation
============================================
τ(x) = E[Y|W=1,X] / E[Y|W=0,X]
"""

import numpy as np


EPS = 1e-10  # divide-by-zero guard for empty bins / zero outcomes


def _safe_divide(a: np.ndarray, b: np.ndarray, default: float = 1.0) -> np.ndarray:
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(b > EPS, a / b, default)
    return np.where(np.isfinite(result), result, default)


# --- Ranking -----------------------------------------------------------------

def qini_coefficient_ratio(
    tau_pred: np.ndarray,
    W: np.ndarray,
    Y: np.ndarray,
    mu0_true: np.ndarray = None,
    tau_true: np.ndarray = None,
    weights: np.ndarray = None,
) -> float:
    """Ratio Qini: oracle path if mu0_true/tau_true given, else count-based.
    ``weights`` (count-based only, e.g. IPW) make every arm rate a weighted rate.
    Ties in tau_pred are broken at random (seeded)."""
    n = len(tau_pred)
    w = np.ones(n) if weights is None else weights

    rng = np.random.default_rng(0)
    order = np.lexsort((rng.random(n), -tau_pred))

    if mu0_true is not None and tau_true is not None:
        mu0_sorted = mu0_true[order]
        mu1_sorted = mu0_sorted * tau_true[order]
        ratio_curve = np.cumsum(mu1_sorted) / np.cumsum(mu0_sorted)
        baseline = mu1_sorted.mean() / mu0_sorted.mean()
    else:
        W_sorted, Y_sorted, w_sorted = W[order], Y[order], w[order]
        cum_n1 = np.cumsum(w_sorted * W_sorted)
        cum_n0 = np.cumsum(w_sorted * (1 - W_sorted))
        cum_y1 = np.cumsum(w_sorted * Y_sorted * W_sorted)
        cum_y0 = np.cumsum(w_sorted * Y_sorted * (1 - W_sorted))
        rate1 = _safe_divide(cum_y1, cum_n1, default=0)
        rate0 = _safe_divide(cum_y0, cum_n0, default=1)
        ratio_curve = _safe_divide(rate1, rate0, default=1)
        baseline = _safe_divide(cum_y1[-1] / cum_n1[-1], cum_y0[-1] / cum_n0[-1], default=1)

    auc = np.trapezoid(ratio_curve, np.linspace(0, 1, n))
    return auc - baseline


# --- Calibration -------------------------------------------------------------

MAX_BINS = 10              # upper bound on the number of tau_pred buckets
MIN_CTRL_CONV_PER_BIN = 30  # buckets are sized so each holds ~this many control converters
MIN_BINS_FOR_DECOMP = 3     # fewer valid buckets: slope/resid undefined or trivially 0


def _calibration_summary(log_pred: np.ndarray, log_target: np.ndarray,
                         weights: np.ndarray, min_points: int) -> dict:
    """cal = exp(mean |log_pred - log_target|) (1 = perfect), bias = mean(log_pred - log_target)
    (> 0 = level over-predicted), slope/resid = weighted OLS of log_target on log_pred
    (slope < 1: tau_pred too extreme; resid: what no affine recalibration fixes).
    Constant tau_pred: slope/resid NaN."""
    d = log_pred - log_target
    out = {'cal': float(np.exp(np.average(np.abs(d), weights=weights))),
           'bias': float(np.average(d, weights=weights)),
           'slope': np.nan, 'resid': np.nan}
    if len(log_pred) >= min_points and np.isfinite(log_pred).all() and np.ptp(log_pred) > 0:
        # np.polyfit weights multiply residuals, hence sqrt for weighted OLS.
        slope, icpt = np.polyfit(log_pred, log_target, 1, w=np.sqrt(weights))
        r = log_target - (icpt + slope * log_pred)
        out.update(slope=float(slope), resid=float(np.sqrt(np.average(r**2, weights=weights))))
    return out


def calibration_ratio(tau_pred: np.ndarray, W: np.ndarray, Y: np.ndarray,
                      tau_true: np.ndarray = None, weights: np.ndarray = None) -> dict:
    """Pointwise (``*_point``, needs tau_true) and binned (``*_binned``) ratio calibration;
    ``weights`` (binned only) weight the bucket arm rates, bucket sizes stay unweighted."""
    n = len(tau_pred)
    w = np.ones(n) if weights is None else weights
    log_pred = np.log(np.clip(tau_pred, EPS, None))
    nan4 = dict.fromkeys(['cal', 'bias', 'slope', 'resid'], np.nan)

    point = nan4
    if tau_true is not None:
        point = _calibration_summary(log_pred, np.log(np.clip(tau_true, EPS, None)),
                                     np.ones(n), min_points=2)

    # Equal-count rank buckets, ties broken at random (seeded, as in the Qini).
    n_bins = int(np.clip(Y[W == 0].sum() // MIN_CTRL_CONV_PER_BIN, 1, MAX_BINS))
    order = np.lexsort((np.random.default_rng(0).random(n), tau_pred))
    bin_idx = np.empty(n, dtype=int)
    bin_idx[order] = np.arange(n) * n_bins // n
    lp, lt, wt = [], [], []
    for b in range(n_bins):
        mask = bin_idx == b
        mask_t, mask_c = mask & (W == 1), mask & (W == 0)
        if mask_t.sum() == 0 or mask_c.sum() == 0:
            continue
        mu_t = np.average(Y[mask_t], weights=w[mask_t])
        mu_c = np.average(Y[mask_c], weights=w[mask_c])
        tau_hat = tau_pred[mask].mean()
        if mu_c <= EPS or mu_t <= EPS:
            continue
        # non-positive bucket mean: floored as in the pointwise error, not skipped
        lp.append(np.log(max(tau_hat, EPS)))
        lt.append(np.log(mu_t / mu_c))
        wt.append(mask.sum())
    binned = nan4
    if lp:
        binned = _calibration_summary(np.array(lp), np.array(lt), np.array(wt, float),
                                      min_points=MIN_BINS_FOR_DECOMP)

    return {**{f'{k}_point': v for k, v in point.items()},
            **{f'{k}_binned': v for k, v in binned.items()}}
