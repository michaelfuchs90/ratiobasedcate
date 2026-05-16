"""
Meta-Learners for Ratio-Based CATE Estimation
=============================================
τ(x) = E[Y|W=1,X] / E[Y|W=0,X]
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import lightgbm as lgb
from typing import Optional, Union, Callable
from abc import ABC, abstractmethod
from calibration import CalibratedCATELearner

# =============================================================================
# Config
# =============================================================================
LGBM_PARAMS = dict(verbose=-1)

# Clipping bounds (following Section 5.6 of the paper)
CLIP_PROPENSITY = (0.01, 0.99)          # e(x) ∈ [ε_e, 1-ε_e] with ε_e = 0.01
CLIP_OUTCOME_PROB = (0.001, 0.999)      # μ_w(x) ∈ [ε_μ, 1-ε_μ] with ε_μ = 0.001
CLIP_MARGINAL_CONV = (0.001, 1.0)       # m(x) ∈ [ε_m, 1] with ε_m = 0.001
CLIP_CONVERTER_PROP = (0.01, 0.99)      # p(x) ∈ [ε_p, 1-ε_p] with ε_p = 0.01
CLIP_LOG_PSEUDO = (-10.0, 10.0)         # Log-scale pseudo-outcomes ∈ [-C, C] with C = 10
CLIP_DIRECT_PSEUDO = (0.001, 1000.0)    # Direct-scale pseudo-outcomes ∈ [ε_τ, 1/ε_τ]
CLIP_TAU = (0.01, 100.0)                # Final τ predictions


def clip(x: np.ndarray, bounds: tuple) -> np.ndarray:
    """Clip array to specified bounds."""
    return np.clip(x, bounds[0], bounds[1])


def winsorise(x: np.ndarray, lower: float = 1.0, upper: float = 99.0) -> np.ndarray:
    """Winsorise array at given percentiles (default 1st/99th)."""
    lo, hi = np.percentile(x, [lower, upper])
    return np.clip(x, lo, hi)


# =============================================================================
# Base Class
# =============================================================================


class BaseLearner(ABC):
    """Abstract base for CATE learners.

    All methods expect:
        X: pd.DataFrame with features, non-numeric covariables must be of type category
        W: np.ndarray with binary treatment (0/1)
        Y: np.ndarray with binary outcome (0/1)
        propensity: np.ndarray with P(W=1|X), or None to estimate
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame, W: np.ndarray, Y: np.ndarray, 
            propensity: np.ndarray = None) -> 'BaseLearner':
        pass

    
    def predict_ratio_cate(self, X: pd.DataFrame, propensity: np.ndarray = None) -> np.ndarray:
        return None
    def predict_difference_cate(self, X: pd.DataFrame, propensity: np.ndarray = None) -> np.ndarray:
        return None



# =============================================================================
# Plug-in Learners
# =============================================================================
class SLearner(BaseLearner):
    """Single model with treatment as feature."""

    def __init__(self, random_state=42):
        self.random_state = random_state

    def fit(self, X, W, Y, propensity=None):
        X = X.copy()
        X['_W'] = W
        self._model = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=self.random_state).fit(X, Y)
        return self

    def predict_ratio_cate(self, X, propensity=None):
        X1, X0 = X.copy(), X.copy()
        X1['_W'], X0['_W'] = 1, 0
        mu1 = clip(self._model.predict_proba(X1)[:, 1], CLIP_OUTCOME_PROB)
        mu0 = clip(self._model.predict_proba(X0)[:, 1], CLIP_OUTCOME_PROB)
        tau = mu1 / mu0
        return clip(tau, CLIP_TAU)

    def predict_difference_cate(self, X, propensity=None):
        X1, X0 = X.copy(), X.copy()
        X1['_W'], X0['_W'] = 1, 0
        mu1 = clip(self._model.predict_proba(X1)[:, 1], CLIP_OUTCOME_PROB)
        mu0 = clip(self._model.predict_proba(X0)[:, 1], CLIP_OUTCOME_PROB)
        return mu1 - mu0

class TLearner(BaseLearner):
    """Separate models for treated/control."""

    def __init__(self, random_state=42):
        self.random_state = random_state

    def fit(self, X, W, Y, propensity=None):
        self._m1 = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=self.random_state).fit(X[W==1], Y[W==1])
        self._m0 = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=self.random_state).fit(X[W==0], Y[W==0])
        return self

    def predict_ratio_cate(self, X, propensity=None):
        mu1 = clip(self._m1.predict_proba(X)[:, 1], CLIP_OUTCOME_PROB)
        mu0 = clip(self._m0.predict_proba(X)[:, 1], CLIP_OUTCOME_PROB)
        tau = mu1 / mu0
        return clip(tau, CLIP_TAU)

    def predict_difference_cate(self, X, propensity=None):
        mu1 = clip(self._m1.predict_proba(X)[:, 1], CLIP_OUTCOME_PROB)
        mu0 = clip(self._m0.predict_proba(X)[:, 1], CLIP_OUTCOME_PROB)
        return mu1-mu0

class QLearner(BaseLearner):
    """Q-Learner for ratio-based CATE.

    Implements the Q-Learner identity (paper eq:q-identity):

        τ(x) = [p(x) / (1 - p(x))] · [(1 - e(x)) / e(x)],

    where p(x) = P(W=1 | Y=1, X=x) is the converter propensity and
    e(x) = P(W=1 | X=x) is the treatment propensity. Both models are
    trained on the same data so that their errors are correlated; this
    correlation is what drives the variance-equivalence argument in
    Proposition~\\ref{prop:variance}.

    Parameters
    ----------
    random_state : int, default 42
        Seed passed to both LightGBM classifiers.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state

    def fit(self, X, W, Y, propensity=None):
        # Propensity and Treatment probability among converters
        self._e_model = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=self.random_state).fit(X, W)
        self._p_model = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=self.random_state).fit(X[Y == 1], W[Y == 1])
        return self

    def predict_ratio_cate(self, X, propensity=None):
        # Always use the estimated propensity model — see Remark (Variance Equivalence):
        # the variance cancellation requires p̂ and ê to be correlated through the same data.
        e = clip(self._e_model.predict_proba(X)[:, 1], CLIP_PROPENSITY)
        p = clip(self._p_model.predict_proba(X)[:, 1], CLIP_CONVERTER_PROP)
        tau = (p / (1 - p)) * ((1 - e) / e)
        return clip(tau, CLIP_TAU)





# =============================================================================
# Doubly Robust Learners
# =============================================================================
class DRBaseLearner(BaseLearner):
    """Base class for outcome-regression-based doubly robust ratio-CATE learners.

    Shared machinery for DR-T and DR-S: cross-fits outcome models μ̂_1(x),
    μ̂_0(x) and the propensity ê(x), forms a DR pseudo-outcome, and
    regresses it on X in a second stage. Subclasses only implement
    ``_fit_outcome_models``.

    Parameters
    ----------
    scale : {'log', 'direct'}, default 'log'
        Pseudo-outcome scale. ``'log'`` regresses log τ̂ using
        eq:dr-t-log; ``'direct'`` regresses τ̂ using eq:dr-t with a
        Poisson second-stage objective.
    n_splits : int, default 5
        Number of cross-fitting folds.
    random_state : int, default 42
        Seed for KFold and all LightGBM models.
    """

    def __init__(self, scale='log', n_splits=5, random_state=42):
        self.scale = scale
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(self, X, W, Y, propensity=None):
        n = len(Y)

        # Cross-fitted nuisances
        mu1_cf, mu0_cf, e_cf = np.zeros(n), np.zeros(n), np.zeros(n)

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for tr, te in kf.split(X):
            Xtr, Xte, Wtr, Ytr = X.iloc[tr], X.iloc[te], W[tr], Y[tr]

            mu1_cf[te], mu0_cf[te] = self._fit_outcome_models(Xtr, Xte, Wtr, Ytr)

            # Always estimate propensity from data — same reasoning as Q-Learner.
            e_cf[te] = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=self.random_state).fit(Xtr, Wtr).predict_proba(Xte)[:, 1]

        # Apply clipping to cross-fitted nuisances
        mu1_cf = clip(mu1_cf, CLIP_OUTCOME_PROB)
        mu0_cf = clip(mu0_cf, CLIP_OUTCOME_PROB)
        e_cf = clip(e_cf, CLIP_PROPENSITY)

        # Pseudo-outcomes
        psi = self._pseudo(W, Y, mu1_cf, mu0_cf, e_cf)
        if self.scale == 'direct':
            self._final = lgb.LGBMRegressor(**LGBM_PARAMS, objective='poisson', random_state=self.random_state).fit(X, clip(psi, (CLIP_DIRECT_PSEUDO[0], None)))

        else:
            self._final = lgb.LGBMRegressor(**LGBM_PARAMS, random_state=self.random_state).fit(X, psi)

        return self

    def _pseudo(self, W, Y, mu1, mu0, e):
        """Compute DR-S/T pseudo-outcomes (paper eq:dr-t and eq:dr-t-log)."""

        if self.scale == 'log':
            # Log-scale pseudo-outcome (paper eq:dr-t-log):
            # Γ = log(μ₁) - log(μ₀) + W(Y-μ₁)/(e·μ₁) - (1-W)(Y-μ₀)/((1-e)·μ₀)
            r1 = W * (Y - mu1) / (e * mu1)
            r0 = (1 - W) * (Y - mu0) / ((1 - e) * mu0)

            psi = np.log(mu1) - np.log(mu0) + r1 - r0

            psi = clip(psi, CLIP_LOG_PSEUDO)

        else:  # direct scale
            # Direct-scale pseudo-outcome (paper eq:dr-t):
            # Γ = τ̂ + W(Y-μ₁)/(e·μ₀) - τ̂·(1-W)(Y-μ₀)/((1-e)·μ₀)
            tau = mu1 / mu0
            a1 = W * (Y - mu1) / (e * mu0)
            a0 = tau * (1 - W) * (Y - mu0) / ((1 - e) * mu0)

            psi = tau + a1 - a0

            psi = clip(psi, CLIP_DIRECT_PSEUDO)

        return psi

    def predict_ratio_cate(self, X, propensity=None):
        prediction = self._final.predict(X)
        if self.scale=='log':
            prediction = clip(prediction, CLIP_LOG_PSEUDO)
            tau = np.exp(prediction)
        else: 
            tau = prediction
        return clip(tau, CLIP_TAU)

    def _fit_outcome_models(self, Xtr, Xte, Wtr, Ytr):
        raise NotImplementedError("Subclasses must implement _fit_outcome_models")





# =============================================================================
# DR-T Learner (separate outcome models per treatment arm)
# =============================================================================
class DRTLearner(DRBaseLearner):
    """Doubly robust T-Learner (paper Section~\\ref{sec:dr-t}).

    Fits separate outcome models μ̂_1(x), μ̂_0(x) on the treated and
    control subsamples respectively. Inherits the cross-fitted fit/predict
    loop and DR pseudo-outcome construction (eq:dr-t / eq:dr-t-log) from
    ``DRBaseLearner``. Classical double robustness: consistent if either
    the propensity or the outcome models are correctly specified.

    Parameters are documented in ``DRBaseLearner``.
    """

    def _fit_outcome_models(self, Xtr, Xte, Wtr, Ytr):
        m1 = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=self.random_state)
        m0 = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=self.random_state)

        mu1 = m1.fit(Xtr[Wtr == 1], Ytr[Wtr == 1]).predict_proba(Xte)[:, 1]
        mu0 = m0.fit(Xtr[Wtr == 0], Ytr[Wtr == 0]).predict_proba(Xte)[:, 1]

        return mu1, mu0

class DRSLearner(DRBaseLearner):
    """Doubly robust S-Learner (paper Section~\\ref{sec:dr-t}).

    Fits a single joint outcome model μ̂(x, w) with the treatment as an
    additional feature and evaluates it at w=1 and w=0 to obtain μ̂_1(x)
    and μ̂_0(x). Uses the same DR pseudo-outcome (eq:dr-t / eq:dr-t-log)
    and cross-fitting machinery as ``DRTLearner`` via ``DRBaseLearner``.
    Classical double robustness.

    Parameters are documented in ``DRBaseLearner``.
    """

    def _fit_outcome_models(self, Xtr, Xte, Wtr, Ytr):
        # Augment training features with treatment indicator
        Xtr_aug = Xtr.copy()
        Xtr_aug['_W'] = Wtr

        # Fit single joint outcome model on all observations
        model = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=self.random_state)
        model.fit(Xtr_aug, Ytr)

        Xte_1, Xte_0 = Xte.copy(), Xte.copy()
        Xte_1['_W'], Xte_0['_W'] = 1, 0
        mu1 = model.predict_proba(Xte_1)[:, 1]
        mu0 = model.predict_proba(Xte_0)[:, 1]

        return mu1, mu0








class DRQLearner(BaseLearner):
    """Doubly robust Q-Learner (paper Section~\\ref{sec:dr-q}).

    Augments the Q-Learner identity with influence-function corrections
    for the converter propensity p(x), the marginal conversion rate m(x),
    and the treatment propensity e(x). Unlike DR-S/T, DR-Q achieves only
    *conditional* double robustness: consistency requires exact e(x),
    which is available in RCTs but typically not in observational data.

    Parameters
    ----------
    scale : {'log', 'direct'}, default 'log'
        Pseudo-outcome scale. ``'log'`` uses eq:dr-q-log;
        ``'direct'`` uses eq:dr-q with a Poisson second-stage objective.
    n_splits : int, default 5
        Number of cross-fitting folds.
    random_state : int, default 42
        Seed for KFold and all LightGBM models.
    """

    def __init__(self, scale='log', n_splits=5, random_state=42):
        self.scale = scale
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(self, X, W, Y, propensity=None):
        n = len(Y)

        # Cross-fitted nuisances
        p_cf, m_cf, e_cf = np.zeros(n), np.zeros(n), np.zeros(n)

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for tr, te in kf.split(X):
            Xtr, Xte, Wtr, Ytr = X.iloc[tr], X.iloc[te], W[tr], Y[tr]
            conv_tr = Ytr == 1
            p_cf[te] = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=self.random_state).fit(Xtr[conv_tr], Wtr[conv_tr]).predict_proba(Xte)[:, 1] 
            m_cf[te] = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=self.random_state).fit(Xtr, Ytr).predict_proba(Xte)[:, 1]

            # Always estimate propensity from data — same reasoning as Q-Learner.
            e_cf[te] = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=self.random_state).fit(Xtr, Wtr).predict_proba(Xte)[:, 1]

        # Apply clipping to cross-fitted nuisances
        p_cf = clip(p_cf, CLIP_CONVERTER_PROP)
        m_cf = clip(m_cf, CLIP_MARGINAL_CONV)
        e_cf = clip(e_cf, CLIP_PROPENSITY)

        # Pseudo-outcomes
        psi = self._pseudo(W, Y, p_cf, e_cf, m_cf)
        if self.scale == 'direct':
            self._final = lgb.LGBMRegressor(**LGBM_PARAMS, objective='poisson', random_state=self.random_state).fit(X, clip(psi, (CLIP_DIRECT_PSEUDO[0], None)))
        else:
            self._final = lgb.LGBMRegressor(**LGBM_PARAMS, random_state=self.random_state).fit(X, psi)

        return self

    def _pseudo(self, W, Y, p, e, m):
        """Compute DR-Q pseudo-outcomes (paper eq:dr-q and eq:dr-q-log)."""

        if self.scale == 'log':
            # Log-scale pseudo-outcome (paper eq:dr-q-log):
            # log(τ) = logit(p) - logit(e)
            log_tau = np.log(p) - np.log(1 - p) + np.log(1 - e) - np.log(e)

            # First-order corrections
            # φ_logit(p) = Y(W-p) / (m·p·(1-p))
            # φ_logit(e) = (W-e) / (e·(1-e))
            ap = Y * (W - p) / (m * p * (1 - p))
            ae = (W - e) / (e * (1 - e))

            psi = log_tau + ap - ae

            psi = clip(psi, CLIP_LOG_PSEUDO)

        else:  # direct scale
            # Direct-scale pseudo-outcome (paper eq:dr-q):
            A = p / (1 - p)           # Converter odds
            B = (1 - e) / e           # Inverse propensity odds
            tau_plugin = A * B

            # First-order corrections via product rule: φ_τ = B·φ_A + A·φ_B
            # φ_A = Y(W-p) / (m·(1-p)²)
            # φ_B = -(W-e) / e²
            phiA = Y * (W - p) / (m * (1 - p)**2)
            phiB = -(W - e) / (e**2)

            correction = B * phiA + A * phiB
            psi = tau_plugin + correction

            psi = clip(psi, CLIP_DIRECT_PSEUDO)

        return psi

    def predict_ratio_cate(self, X, propensity=None):
        prediction = self._final.predict(X)
        if self.scale=="log":
            prediction = clip(prediction, CLIP_LOG_PSEUDO)
            tau = np.exp(prediction)
        else:
            tau = prediction
        return clip(tau, CLIP_TAU)




class QSimpleLearner(BaseLearner):
    """Q-Learner for RCTs using known propensity (no propensity model fitted on W).

    Instead of estimating e(x) from observed treatment assignments, this learner
    models the *known* propensity function by regressing the provided propensity
    values on X.  This generalises to both constant-propensity RCTs (e.g. e=0.5)
    and stratified designs where e(x) varies by covariate.

    Allows direct comparison with Q-Learner to empirically test the variance-
    equivalence claim: does correlated (p̂, ê) estimated from data beat using the
    exact known e(x)?

    Requires known propensity — designed for RCT settings.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state

    def fit(self, X, W, Y, propensity=None):
        if propensity is None:
            raise ValueError("QSimpleLearner requires known propensity (designed for RCTs)")
        # Model the known propensity function e(x) — fitted to provided values, not W.
        # For constant propensity this trivially predicts the constant; for stratified
        # RCTs it learns the design function.
        self._e_model = lgb.LGBMRegressor(**LGBM_PARAMS, random_state=self.random_state).fit(X, propensity)
        self._p_model = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=self.random_state).fit(X[Y == 1], W[Y == 1])
        return self

    def predict_ratio_cate(self, X, propensity=None):
        p = clip(self._p_model.predict_proba(X)[:, 1], CLIP_CONVERTER_PROP)
        e = clip(self._e_model.predict(X), CLIP_PROPENSITY)
        tau = (p / (1 - p)) * ((1 - e) / e)
        return clip(tau, CLIP_TAU)


class DRQSimpleLearner(BaseLearner):
    """Doubly robust Q-Simple Learner for RCTs (paper Section~\\ref{sec:dr-q-simple}).

    Operates only on converters (Y=1), which eliminates the marginal
    conversion rate m(x) from the correction term (paper eq:dr-q-simple
    and eq:dr-q-simple-log). Requires known propensity e(x); designed for
    RCT settings where e(x) is known by design.

    Parameters
    ----------
    scale : {'log', 'direct'}, default 'log'
        Pseudo-outcome scale.
    n_splits : int, default 5
        Number of cross-fitting folds for p(x).
    random_state : int, default 42
        Seed for KFold and all LightGBM models.
    """

    def __init__(self, scale='log', n_splits=5, random_state=42):
        self.scale, self.n_splits, self.random_state = scale, n_splits, random_state

    def fit(self, X, W, Y, propensity=None):
        if propensity is None:
            raise ValueError("DRQSimpleLearner requires known propensity (designed for RCTs)")

        n = len(Y)
        converters = Y == 1
        n_conv = converters.sum()

        # Cross-fitted p(x) on converters only
        p_cf = np.zeros(n)

        # We need to cross-fit on the converter subset
        X_conv = X[converters].reset_index(drop=True)
        W_conv = W[converters]
        conv_indices = np.where(converters)[0]

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for tr, te in kf.split(X_conv):
            Xtr, Xte = X_conv.iloc[tr], X_conv.iloc[te]
            Wtr = W_conv[tr]

            p_model = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=self.random_state)
            p_cf[conv_indices[te]] = p_model.fit(Xtr, Wtr).predict_proba(Xte)[:, 1]

        # Apply clipping
        p_cf_conv = clip(p_cf[converters], CLIP_CONVERTER_PROP)
        e_conv = clip(propensity[converters], CLIP_PROPENSITY)
        W_conv = W[converters]

        # Pseudo-outcomes (only for converters)
        psi_conv = self._pseudo(W_conv, p_cf_conv, e_conv)

        # Fit final model on converters
        if self.scale == 'direct':
            self._final = lgb.LGBMRegressor(**LGBM_PARAMS, objective='poisson', random_state=self.random_state)
            self._final.fit(X[converters], clip(psi_conv, (CLIP_DIRECT_PSEUDO[0], None)))
        else:
            self._final = lgb.LGBMRegressor(**LGBM_PARAMS, random_state=self.random_state)
            self._final.fit(X[converters], psi_conv)

        return self

    def _pseudo(self, W, p, e):
        """Compute DR-Q-Simple pseudo-outcomes (paper eq:dr-q-simple).

        Operates on converters only, which eliminates m(x) from the
        correction term compared to DR-Q.
        """

        if self.scale == 'log':
            # Log-scale pseudo-outcome (paper eq:dr-q-simple-log):
            # Γ = logit(p) - logit(e) + (W - p) / (p(1-p))
            log_tau = np.log(p) - np.log(1 - p) + np.log(1 - e) - np.log(e)
            correction = (W - p) / (p * (1 - p))
            psi = log_tau + correction
            psi = clip(psi, CLIP_LOG_PSEUDO)

        else:  # direct scale
            # Direct-scale pseudo-outcome (paper eq:dr-q-simple):
            # Γ = τ + (1-e)(W-p) / (e(1-p)²)
            tau_plugin = (p / (1 - p)) * ((1 - e) / e)
            correction = (1 - e) * (W - p) / (e * (1 - p)**2)
            psi = tau_plugin + correction
            psi = clip(psi, CLIP_DIRECT_PSEUDO)

        return psi

    def predict_ratio_cate(self, X, propensity=None):
        prediction = self._final.predict(X)
        if self.scale == 'log':
            prediction = clip(prediction, CLIP_LOG_PSEUDO)
            tau = np.exp(prediction)
        else:
            tau = prediction
        return clip(tau, CLIP_TAU)





# =============================================================================
# X-Learner (difference-scale, converted to ratio via mu0)
# =============================================================================

class XLearner(BaseLearner):
    """X-Learner (Künzel et al. 2019) with ratio-CATE output.

    Fits T-learner outcome models μ̂_1(x), μ̂_0(x), imputes pseudo-effects
    for each arm, fits arm-specific CATE models τ̂_1(x) / τ̂_0(x), and
    combines them with the propensity score:

        τ̂_diff(x) = e(x) · τ̂_0(x) + (1 − e(x)) · τ̂_1(x)

        

    Parameters
    ----------
    random_state : int, default 42
    """

    def __init__(self, random_state=42):
        self.random_state = random_state

    def fit(self, X, W, Y, propensity=None):
        # Stage 1: T-learner outcome models
        self._m1 = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=self.random_state)
        self._m0 = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=self.random_state)
        self._m1.fit(X[W == 1], Y[W == 1])
        self._m0.fit(X[W == 0], Y[W == 0])

        mu1 = clip(self._m1.predict_proba(X)[:, 1], CLIP_OUTCOME_PROB)
        mu0 = clip(self._m0.predict_proba(X)[:, 1], CLIP_OUTCOME_PROB)

        # Stage 2: Imputed pseudo-effects per arm
        # For treated units: D1 = Y - μ̂_0(x)  (observed minus counterfactual control)
        # For control units: D0 = μ̂_1(x) - Y  (counterfactual treated minus observed)
        D1 = Y[W == 1] - mu0[W == 1]
        D0 = mu1[W == 0] - Y[W == 0]

        # Stage 3: Fit arm-specific CATE models on imputed effects
        self._tau1 = lgb.LGBMRegressor(**LGBM_PARAMS, random_state=self.random_state)
        self._tau0 = lgb.LGBMRegressor(**LGBM_PARAMS, random_state=self.random_state)
        self._tau1.fit(X[W == 1], D1)
        self._tau0.fit(X[W == 0], D0)

        # Stage 4: Propensity model for blending
        self._e_model = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=self.random_state)
        self._e_model.fit(X, W)

        return self

    def predict_difference_cate(self, X, propensity=None):
        e = clip(self._e_model.predict_proba(X)[:, 1], CLIP_PROPENSITY)
        tau1_hat = self._tau1.predict(X)
        tau0_hat = self._tau0.predict(X)

        # Propensity-weighted blend (Künzel et al. eq. 9)
        tau_diff = e * tau0_hat + (1.0 - e) * tau1_hat

        return tau_diff


# =============================================================================
# X-Learner on the Log Scale (no mu0 transform needed)
# =============================================================================
class XLearnerLog(BaseLearner):
    """X-Learner on the log-ratio scale.

    Uses the T-learner log-ratio plug-in as the imputed pseudo-effect
    for each arm:

        D1_i = log μ̂_1(x_i) - log μ̂_0(x_i)   for treated units
        D0_i = log μ̂_1(x_i) - log μ̂_0(x_i)   for control units

    Arm-specific models τ̂_1(x), τ̂_0(x) are then fitted on these
    imputed log-effects and blended with the propensity score.
    predict() exponentiates — no mu0 transform needed.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state

    def fit(self, X, W, Y, propensity=None):
        self._m1 = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=self.random_state)
        self._m0 = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=self.random_state)
        self._m1.fit(X[W == 1], Y[W == 1])
        self._m0.fit(X[W == 0], Y[W == 0])

        mu1 = clip(self._m1.predict_proba(X)[:, 1], CLIP_OUTCOME_PROB)
        mu0 = clip(self._m0.predict_proba(X)[:, 1], CLIP_OUTCOME_PROB)
        log_tau_plugin = np.log(mu1) - np.log(mu0)
        log_tau_plugin = clip(log_tau_plugin, CLIP_LOG_PSEUDO)

        # Imputed log-ratio pseudo-effects: plug-in evaluated at each unit's X
        D1 = log_tau_plugin[W == 1]
        D0 = log_tau_plugin[W == 0]

        self._tau1 = lgb.LGBMRegressor(**LGBM_PARAMS, random_state=self.random_state)
        self._tau0 = lgb.LGBMRegressor(**LGBM_PARAMS, random_state=self.random_state)
        self._tau1.fit(X[W == 1], D1)
        self._tau0.fit(X[W == 0], D0)

        self._e_model = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=self.random_state)
        self._e_model.fit(X, W)
        return self

    def predict_ratio_cate(self, X, propensity=None):
        e = clip(self._e_model.predict_proba(X)[:, 1], CLIP_PROPENSITY)
        log_tau1 = clip(self._tau1.predict(X), CLIP_LOG_PSEUDO)
        log_tau0 = clip(self._tau0.predict(X), CLIP_LOG_PSEUDO)
        log_tau = e * log_tau0 + (1.0 - e) * log_tau1
        return clip(np.exp(clip(log_tau, CLIP_LOG_PSEUDO)), CLIP_TAU)


class RLearner(BaseLearner):
    """R-Learner with ratio-CATE output. Cross-fits m(x) and e(x) only."""

    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(self, X, W, Y, propensity=None):
        n = len(Y)
        m_cf  = np.zeros(n)
        e_cf  = np.zeros(n)

        kf = KFold(n_splits=self.n_splits, shuffle=True,
                   random_state=self.random_state)
        for tr, te in kf.split(X):
            Xtr, Xte = X.iloc[tr], X.iloc[te]
            Wtr, Ytr = W[tr], Y[tr]

            m_model = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=self.random_state)
            m_model.fit(Xtr, Ytr)
            m_cf[te] = m_model.predict_proba(Xte)[:, 1]

            e_model = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=self.random_state)
            e_model.fit(Xtr, Wtr)
            e_cf[te] = e_model.predict_proba(Xte)[:, 1]

        m_cf = clip(m_cf, CLIP_OUTCOME_PROB)
        e_cf = clip(e_cf, CLIP_PROPENSITY)

        Y_tilde = Y - m_cf
        W_tilde = W - e_cf
        weights = W_tilde ** 2

        # Pseudo-outcome: Ỹ / W̃, winsorised for stability
        W_tilde_safe = np.where(np.abs(W_tilde) < 1e-3, 1e-3, W_tilde)
        pseudo = winsorise(Y_tilde / W_tilde_safe)

        self._final = lgb.LGBMRegressor(**LGBM_PARAMS, random_state=self.random_state)
        self._final.fit(X, pseudo, sample_weight=weights)

        return self

    def predict_difference_cate(self, X, propensity=None):
        return self._final.predict(X)


# =============================================================================
# Log R-Learner
# =============================================================================

class RLearnerLog(BaseLearner):
    """R-Learner on the log-ratio scale for binary outcomes.

    Derives from the multiplicative partially linear model:

        log μ_w(x) = log μ_0(x) + W · log τ(x)

    which implies the residualised form:

        L̃_i = log τ(x_i) · W̃_i + ε_i

    where:
        L̃_i = log μ̂_{W_i}(x_i) - ℓ̂(x_i)       (log-outcome residual)
        W̃_i = W_i - ê(x_i)                        (treatment residual)
        ℓ̂(x) = ê(x)·log μ̂_1(x) + (1-ê(x))·log μ̂_0(x)   (marginal log-outcome)

    Because Y ∈ {0,1}, we cannot take log Y directly. Instead we
    residualise the *model-predicted* log-probability log μ̂_{W_i}(x_i),
    which is well-defined. This is the standard substitution for binary
    outcomes in log-linear partially linear models.

    The R-loss on the log scale:

        min_θ Σ_i (L̃_i - θ(x_i)·W̃_i)² · W̃_i²

    gives pseudo-outcome θ̂ = L̃ / W̃ with weights W̃².

    predict() returns τ̂(x) = exp(θ̂(x)), clipped to CLIP_TAU.

    Parameters
    ----------
    n_splits : int, default 5
        Cross-fitting folds for μ̂_1, μ̂_0, ê.
    random_state : int, default 42
    """

    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(self, X, W, Y, propensity=None):
        n = len(Y)
        mu1_cf = np.zeros(n)
        mu0_cf = np.zeros(n)
        e_cf   = np.zeros(n)

        kf = KFold(n_splits=self.n_splits, shuffle=True,
                   random_state=self.random_state)
        for tr, te in kf.split(X):
            Xtr, Xte = X.iloc[tr], X.iloc[te]
            Wtr, Ytr = W[tr], Y[tr]

            m1 = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=self.random_state)
            m0 = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=self.random_state)
            m1.fit(Xtr[Wtr == 1], Ytr[Wtr == 1])
            m0.fit(Xtr[Wtr == 0], Ytr[Wtr == 0])
            mu1_cf[te] = m1.predict_proba(Xte)[:, 1]
            mu0_cf[te] = m0.predict_proba(Xte)[:, 1]

            e_model = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=self.random_state)
            e_model.fit(Xtr, Wtr)
            e_cf[te] = e_model.predict_proba(Xte)[:, 1]

        mu1_cf = clip(mu1_cf, CLIP_OUTCOME_PROB)
        mu0_cf = clip(mu0_cf, CLIP_OUTCOME_PROB)
        e_cf   = clip(e_cf,   CLIP_PROPENSITY)

        # Log-outcome at the observed treatment arm for each unit
        # log μ̂_{W_i}(x_i): use mu1 where treated, mu0 where control
        log_mu_obs = np.where(W == 1, np.log(mu1_cf), np.log(mu0_cf))

        # Marginal log-outcome: ê(x)·log μ̂_1(x) + (1-ê(x))·log μ̂_0(x)
        log_mu_marginal = e_cf * np.log(mu1_cf) + (1.0 - e_cf) * np.log(mu0_cf)

        # Log-outcome residual L̃ and treatment residual W̃
        L_tilde = log_mu_obs - log_mu_marginal
        W_tilde = W - e_cf

        # R-loss pseudo-outcome: L̃ / W̃, weighted by W̃²
        W_tilde_safe = np.where(np.abs(W_tilde) < 1e-3, 1e-3, W_tilde)
        pseudo = L_tilde / W_tilde_safe
        pseudo = clip(pseudo, CLIP_LOG_PSEUDO)
        weights = W_tilde ** 2

        self._final = lgb.LGBMRegressor(**LGBM_PARAMS,
                                         random_state=self.random_state)
        self._final.fit(X, pseudo, sample_weight=weights)
        return self

    def predict_ratio_cate(self, X, propensity=None):
        log_tau = clip(self._final.predict(X), CLIP_LOG_PSEUDO)
        return clip(np.exp(log_tau), CLIP_TAU)


# =============================================================================
# DR-Learner (difference-scale DR, converted to ratio via mu0)
# =============================================================================

class DRLearner(BaseLearner):
    """Classical DR-Learner (Kennedy 2022) with ratio-CATE output.

    Constructs the standard AIPW pseudo-outcome:

        Γ_i = (μ̂_1(x) − μ̂_0(x))
              + W(Y − μ̂_1(x)) / ê(x)
              − (1−W)(Y − μ̂_0(x)) / (1 − ê(x))

    and regresses it on X. 
    This is the standard difference-scale DR-learner; it is doubly robust
    in the classical sense (consistent if either outcome models or
    propensity is correctly specified).

    Parameters
    ----------
    n_splits : int, default 5
    random_state : int, default 42
    """

    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(self, X, W, Y, propensity=None):
        n = len(Y)
        mu1_cf = np.zeros(n)
        mu0_cf = np.zeros(n)
        e_cf   = np.zeros(n)

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for tr, te in kf.split(X):
            Xtr, Xte = X.iloc[tr], X.iloc[te]
            Wtr, Ytr = W[tr], Y[tr]

            m1 = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=self.random_state)
            m0 = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=self.random_state)
            m1.fit(Xtr[Wtr == 1], Ytr[Wtr == 1])
            m0.fit(Xtr[Wtr == 0], Ytr[Wtr == 0])
            mu1_cf[te] = m1.predict_proba(Xte)[:, 1]
            mu0_cf[te] = m0.predict_proba(Xte)[:, 1]

            e_model = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=self.random_state)
            e_model.fit(Xtr, Wtr)
            e_cf[te] = e_model.predict_proba(Xte)[:, 1]

        mu1_cf = clip(mu1_cf, CLIP_OUTCOME_PROB)
        mu0_cf = clip(mu0_cf, CLIP_OUTCOME_PROB)
        e_cf   = clip(e_cf,   CLIP_PROPENSITY)

        # AIPW pseudo-outcome (difference scale)
        psi = (mu1_cf - mu0_cf
               + W * (Y - mu1_cf) / e_cf
               - (1 - W) * (Y - mu0_cf) / (1 - e_cf))
        psi = clip(psi, CLIP_DIRECT_PSEUDO)

        self._final = lgb.LGBMRegressor(**LGBM_PARAMS, random_state=self.random_state)
        self._final.fit(X, psi)


        return self

    def predict_difference_cate(self, X, propensity=None):
        return self._final.predict(X)









# =============================================================================
# Factory
# =============================================================================
ALL_LEARNER = {
    's':              lambda rs: SLearner(rs),
    't':              lambda rs: TLearner(rs),
    'q':              lambda rs: QLearner(rs),
    'q_simple':       lambda rs: QSimpleLearner(rs),
    'drt_log':            lambda rs: DRTLearner(random_state=rs),
    'drs_log':            lambda rs: DRSLearner(random_state=rs),
    'drq_log':            lambda rs: DRQLearner(random_state=rs),
    'drt_direct':     lambda rs: DRTLearner(scale='direct', random_state=rs),
    'drs_direct':     lambda rs: DRSLearner(scale='direct', random_state=rs),
    'drq_direct':     lambda rs: DRQLearner(scale='direct', random_state=rs),
    'drq_simple_log':     lambda rs: DRQSimpleLearner(scale='log', random_state=rs),
    'drq_simple_direct':  lambda rs: DRQSimpleLearner(scale='direct', random_state=rs),
    'x':              lambda rs: XLearner(rs),
    'x_log':          lambda rs: XLearnerLog(rs),
    'r':              lambda rs: RLearner(random_state=rs),
    'r_log':            lambda rs: RLearnerLog(random_state=rs),
    'dr_diff':        lambda rs: DRLearner(random_state=rs),
}



def get_learner(name: str, random_state: int = 42) -> BaseLearner:
    """Get learner by name. See ALL_LEARNER.keys() for all options."""
    if name not in ALL_LEARNER:
        raise ValueError(f"Unknown: '{name}'. Available: {list(ALL_LEARNER.keys())}")
    return ALL_LEARNER[name](random_state)



