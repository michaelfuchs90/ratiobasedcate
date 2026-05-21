"""
Meta-Learners for Ratio-Based CATE Estimation
=============================================
τ(x) = E[Y|W=1,X] / E[Y|W=0,X]
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import lightgbm as lgb
from abc import ABC, abstractmethod


LGBM_PARAMS = dict(verbose=-1)

# Clipping bounds (paper Section 5.6)
CLIP_PROPENSITY = (0.01, 0.99)          # e(x) ∈ [ε_e, 1-ε_e] with ε_e = 0.01
CLIP_OUTCOME_PROB = (0.001, 0.999)      # μ_w(x) ∈ [ε_μ, 1-ε_μ] with ε_μ = 0.001
CLIP_MARGINAL_CONV = (0.001, 1.0)       # m(x) ∈ [ε_m, 1] with ε_m = 0.001
CLIP_CONVERTER_PROP = (0.01, 0.99)      # p(x) ∈ [ε_p, 1-ε_p] with ε_p = 0.01
CLIP_LOG_PSEUDO = (-10.0, 10.0)         # Log-scale pseudo-outcomes ∈ [-C, C] with C = 10
CLIP_DIRECT_PSEUDO = (0.001, 1000.0)    # Direct-scale pseudo-outcomes ∈ [ε_τ, 1/ε_τ]
CLIP_TAU = (0.01, 100.0)                # Final τ predictions


def clip(x: np.ndarray, bounds: tuple) -> np.ndarray:
    return np.clip(x, bounds[0], bounds[1])


def winsorise(x: np.ndarray, lower: float = 1.0, upper: float = 99.0) -> np.ndarray:
    """Winsorise array at given percentiles (default 1st/99th)."""
    lo, hi = np.percentile(x, [lower, upper])
    return np.clip(x, lo, hi)


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
    """Q-Learner for ratio-CATE (paper eq:q-identity).

    τ(x) = [p(x) / (1 - p(x))] · [(1 - e(x)) / e(x)],
    where p(x) = P(W=1 | Y=1, X=x) and e(x) = P(W=1 | X=x). Both models are
    fit on the same data so their errors are correlated — this drives the
    variance-equivalence argument in Proposition~\\ref{prop:variance}.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state

    def fit(self, X, W, Y, propensity=None):
        self._e_model = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=self.random_state).fit(X, W)
        self._p_model = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=self.random_state).fit(X[Y == 1], W[Y == 1])
        return self

    def predict_ratio_cate(self, X, propensity=None):
        # Always use the estimated propensity, even on RCTs: variance cancellation
        # requires p̂ and ê to share data-driven correlation (see Prop. variance).
        e = clip(self._e_model.predict_proba(X)[:, 1], CLIP_PROPENSITY)
        p = clip(self._p_model.predict_proba(X)[:, 1], CLIP_CONVERTER_PROP)
        tau = (p / (1 - p)) * ((1 - e) / e)
        return clip(tau, CLIP_TAU)


class DRBaseLearner(BaseLearner):
    """Base for outcome-regression DR ratio-CATE learners (DR-T, DR-S).

    Cross-fits μ̂_1(x), μ̂_0(x), ê(x), forms a DR pseudo-outcome, and regresses
    it on X. Subclasses implement ``_fit_outcome_models``.

    ``scale='log'`` regresses log τ̂ via eq:dr-t-log; ``'direct'`` regresses
    τ̂ via eq:dr-t with a Poisson second-stage objective.
    """

    def __init__(self, scale='log', n_splits=5, random_state=42):
        self.scale = scale
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(self, X, W, Y, propensity=None):
        n = len(Y)
        mu1_cf, mu0_cf, e_cf = np.zeros(n), np.zeros(n), np.zeros(n)

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for tr, te in kf.split(X):
            Xtr, Xte, Wtr, Ytr = X.iloc[tr], X.iloc[te], W[tr], Y[tr]

            mu1_cf[te], mu0_cf[te] = self._fit_outcome_models(Xtr, Xte, Wtr, Ytr)

            # Always estimate propensity from data — same reasoning as Q-Learner.
            e_cf[te] = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=self.random_state).fit(Xtr, Wtr).predict_proba(Xte)[:, 1]

        mu1_cf = clip(mu1_cf, CLIP_OUTCOME_PROB)
        mu0_cf = clip(mu0_cf, CLIP_OUTCOME_PROB)
        e_cf = clip(e_cf, CLIP_PROPENSITY)

        psi = self._pseudo(W, Y, mu1_cf, mu0_cf, e_cf)
        if self.scale == 'direct':
            self._final = lgb.LGBMRegressor(**LGBM_PARAMS, objective='poisson', random_state=self.random_state).fit(X, clip(psi, (CLIP_DIRECT_PSEUDO[0], None)))

        else:
            self._final = lgb.LGBMRegressor(**LGBM_PARAMS, random_state=self.random_state).fit(X, psi)

        return self

    def _pseudo(self, W, Y, mu1, mu0, e):
        """DR-S/T pseudo-outcomes (paper eq:dr-t and eq:dr-t-log)."""
        if self.scale == 'log':
            # Γ = log(μ₁) - log(μ₀) + W(Y-μ₁)/(e·μ₁) - (1-W)(Y-μ₀)/((1-e)·μ₀)
            r1 = W * (Y - mu1) / (e * mu1)
            r0 = (1 - W) * (Y - mu0) / ((1 - e) * mu0)
            psi = np.log(mu1) - np.log(mu0) + r1 - r0
            psi = clip(psi, CLIP_LOG_PSEUDO)

        else:
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


class DRTLearner(DRBaseLearner):
    """Doubly robust T-Learner: separate μ̂_1, μ̂_0 fit on each treatment arm.

    Classically doubly robust: consistent if either propensity or outcome
    models are correctly specified. See ``DRBaseLearner`` for parameters.
    """

    def _fit_outcome_models(self, Xtr, Xte, Wtr, Ytr):
        m1 = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=self.random_state)
        m0 = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=self.random_state)

        mu1 = m1.fit(Xtr[Wtr == 1], Ytr[Wtr == 1]).predict_proba(Xte)[:, 1]
        mu0 = m0.fit(Xtr[Wtr == 0], Ytr[Wtr == 0]).predict_proba(Xte)[:, 1]

        return mu1, mu0


class DRSLearner(DRBaseLearner):
    """Doubly robust S-Learner: single joint μ̂(x, w) evaluated at w=0 and w=1.

    Classically doubly robust. See ``DRBaseLearner`` for parameters.
    """

    def _fit_outcome_models(self, Xtr, Xte, Wtr, Ytr):
        Xtr_aug = Xtr.copy()
        Xtr_aug['_W'] = Wtr

        model = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=self.random_state)
        model.fit(Xtr_aug, Ytr)

        Xte_1, Xte_0 = Xte.copy(), Xte.copy()
        Xte_1['_W'], Xte_0['_W'] = 1, 0
        mu1 = model.predict_proba(Xte_1)[:, 1]
        mu0 = model.predict_proba(Xte_0)[:, 1]

        return mu1, mu0


class DRQLearner(BaseLearner):
    """Doubly robust Q-Learner (paper Section~\\ref{sec:dr-q}).

    Augments the Q-Learner identity with influence-function corrections for
    p(x), m(x), and e(x). Only *conditionally* doubly robust: consistency
    requires exact e(x) (available in RCTs, generally not in observational
    data).

    ``scale='log'`` uses eq:dr-q-log; ``'direct'`` uses eq:dr-q with a Poisson
    second-stage objective.
    """

    def __init__(self, scale='log', n_splits=5, random_state=42):
        self.scale = scale
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(self, X, W, Y, propensity=None):
        n = len(Y)
        p_cf, m_cf, e_cf = np.zeros(n), np.zeros(n), np.zeros(n)

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for tr, te in kf.split(X):
            Xtr, Xte, Wtr, Ytr = X.iloc[tr], X.iloc[te], W[tr], Y[tr]
            conv_tr = Ytr == 1
            p_cf[te] = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=self.random_state).fit(Xtr[conv_tr], Wtr[conv_tr]).predict_proba(Xte)[:, 1]
            m_cf[te] = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=self.random_state).fit(Xtr, Ytr).predict_proba(Xte)[:, 1]

            # Always estimate propensity from data — same reasoning as Q-Learner.
            e_cf[te] = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=self.random_state).fit(Xtr, Wtr).predict_proba(Xte)[:, 1]

        p_cf = clip(p_cf, CLIP_CONVERTER_PROP)
        m_cf = clip(m_cf, CLIP_MARGINAL_CONV)
        e_cf = clip(e_cf, CLIP_PROPENSITY)

        psi = self._pseudo(W, Y, p_cf, e_cf, m_cf)
        if self.scale == 'direct':
            self._final = lgb.LGBMRegressor(**LGBM_PARAMS, objective='poisson', random_state=self.random_state).fit(X, clip(psi, (CLIP_DIRECT_PSEUDO[0], None)))
        else:
            self._final = lgb.LGBMRegressor(**LGBM_PARAMS, random_state=self.random_state).fit(X, psi)

        return self

    def _pseudo(self, W, Y, p, e, m):
        """DR-Q pseudo-outcomes (paper eq:dr-q and eq:dr-q-log)."""
        if self.scale == 'log':
            # log(τ) = logit(p) - logit(e); corrections φ_logit(p), φ_logit(e)
            log_tau = np.log(p) - np.log(1 - p) + np.log(1 - e) - np.log(e)
            ap = Y * (W - p) / (m * p * (1 - p))
            ae = (W - e) / (e * (1 - e))
            psi = log_tau + ap - ae
            psi = clip(psi, CLIP_LOG_PSEUDO)

        else:
            # τ = A·B with A = p/(1-p), B = (1-e)/e; product-rule corrections
            # φ_τ = B·φ_A + A·φ_B, with φ_A = Y(W-p)/(m(1-p)²), φ_B = -(W-e)/e²
            A = p / (1 - p)
            B = (1 - e) / e
            tau_plugin = A * B

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
    """Q-Learner using the known propensity instead of estimating it from W.

    Models the *known* e(x) by regressing the provided propensity values on
    X (works for constant-propensity RCTs and stratified designs alike).
    Used to empirically test variance equivalence against ``QLearner``: does
    correlated (p̂, ê) from data beat using the exact known e(x)?

    Requires known propensity → designed for RCT settings.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state

    def fit(self, X, W, Y, propensity=None):
        if propensity is None:
            raise ValueError("QSimpleLearner requires known propensity (designed for RCTs)")
        # Fit to provided propensity values, not W. For constant propensity this
        # trivially predicts the constant; for stratified designs it learns e(x).
        self._e_model = lgb.LGBMRegressor(**LGBM_PARAMS, random_state=self.random_state).fit(X, propensity)
        self._p_model = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=self.random_state).fit(X[Y == 1], W[Y == 1])
        return self

    def predict_ratio_cate(self, X, propensity=None):
        p = clip(self._p_model.predict_proba(X)[:, 1], CLIP_CONVERTER_PROP)
        e = clip(self._e_model.predict(X), CLIP_PROPENSITY)
        tau = (p / (1 - p)) * ((1 - e) / e)
        return clip(tau, CLIP_TAU)


class DRQSimpleLearner(BaseLearner):
    """Doubly robust Q-Simple Learner (paper Section~\\ref{sec:dr-q-simple}).

    Operates only on converters (Y=1), eliminating m(x) from the correction
    term (paper eq:dr-q-simple / eq:dr-q-simple-log). Requires known e(x);
    designed for RCT settings.
    """

    def __init__(self, scale='log', n_splits=5, random_state=42):
        self.scale, self.n_splits, self.random_state = scale, n_splits, random_state

    def fit(self, X, W, Y, propensity=None):
        if propensity is None:
            raise ValueError("DRQSimpleLearner requires known propensity (designed for RCTs)")

        n = len(Y)
        converters = Y == 1

        # Cross-fit p(x) on the converter subset only.
        p_cf = np.zeros(n)
        X_conv = X[converters].reset_index(drop=True)
        W_conv = W[converters]
        conv_indices = np.where(converters)[0]

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for tr, te in kf.split(X_conv):
            Xtr, Xte = X_conv.iloc[tr], X_conv.iloc[te]
            Wtr = W_conv[tr]

            p_model = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=self.random_state)
            p_cf[conv_indices[te]] = p_model.fit(Xtr, Wtr).predict_proba(Xte)[:, 1]

        p_cf_conv = clip(p_cf[converters], CLIP_CONVERTER_PROP)
        e_conv = clip(propensity[converters], CLIP_PROPENSITY)
        W_conv = W[converters]

        psi_conv = self._pseudo(W_conv, p_cf_conv, e_conv)

        if self.scale == 'direct':
            self._final = lgb.LGBMRegressor(**LGBM_PARAMS, objective='poisson', random_state=self.random_state)
            self._final.fit(X[converters], clip(psi_conv, (CLIP_DIRECT_PSEUDO[0], None)))
        else:
            self._final = lgb.LGBMRegressor(**LGBM_PARAMS, random_state=self.random_state)
            self._final.fit(X[converters], psi_conv)

        return self

    def _pseudo(self, W, p, e):
        """DR-Q-Simple pseudo-outcomes on converters only (paper eq:dr-q-simple)."""
        if self.scale == 'log':
            # Γ = logit(p) - logit(e) + (W - p) / (p(1-p))
            log_tau = np.log(p) - np.log(1 - p) + np.log(1 - e) - np.log(e)
            correction = (W - p) / (p * (1 - p))
            psi = log_tau + correction
            psi = clip(psi, CLIP_LOG_PSEUDO)

        else:
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


class XLearner(BaseLearner):
    """X-Learner (Künzel et al. 2019), difference-scale.

    Fits T-learner outcomes μ̂_1(x), μ̂_0(x), imputes pseudo-effects per arm,
    fits arm-specific CATE models τ̂_1(x), τ̂_0(x), and blends them with the
    propensity:  τ̂_diff(x) = e(x) · τ̂_0(x) + (1 − e(x)) · τ̂_1(x).
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

        # Stage 2: imputed pseudo-effects per arm.
        # Treated: D1 = Y - μ̂_0 (observed minus counterfactual control).
        # Control: D0 = μ̂_1 - Y (counterfactual treated minus observed).
        D1 = Y[W == 1] - mu0[W == 1]
        D0 = mu1[W == 0] - Y[W == 0]

        # Stage 3: arm-specific CATE models on imputed effects.
        self._tau1 = lgb.LGBMRegressor(**LGBM_PARAMS, random_state=self.random_state)
        self._tau0 = lgb.LGBMRegressor(**LGBM_PARAMS, random_state=self.random_state)
        self._tau1.fit(X[W == 1], D1)
        self._tau0.fit(X[W == 0], D0)

        # Stage 4: propensity model for blending.
        self._e_model = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=self.random_state)
        self._e_model.fit(X, W)

        return self

    def predict_difference_cate(self, X, propensity=None):
        e = clip(self._e_model.predict_proba(X)[:, 1], CLIP_PROPENSITY)
        tau1_hat = self._tau1.predict(X)
        tau0_hat = self._tau0.predict(X)

        # Künzel et al. eq. 9
        tau_diff = e * tau0_hat + (1.0 - e) * tau1_hat

        return tau_diff


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

        # Pseudo-outcome Ỹ / W̃, with W̃ floored away from zero and winsorised
        # to keep extreme ratios from dominating the second-stage fit.
        W_tilde_safe = np.where(np.abs(W_tilde) < 1e-3, 1e-3, W_tilde)
        pseudo = winsorise(Y_tilde / W_tilde_safe)

        self._final = lgb.LGBMRegressor(**LGBM_PARAMS, random_state=self.random_state)
        self._final.fit(X, pseudo, sample_weight=weights)

        return self

    def predict_difference_cate(self, X, propensity=None):
        return self._final.predict(X)


class DRLearner(BaseLearner):
    """Classical AIPW DR-Learner (Kennedy 2022), difference-scale.

    Pseudo-outcome:
        Γ_i = (μ̂_1 − μ̂_0) + W(Y − μ̂_1)/ê − (1−W)(Y − μ̂_0)/(1−ê).
    Doubly robust: consistent if either outcome models or propensity are
    correctly specified.
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

        psi = (mu1_cf - mu0_cf
               + W * (Y - mu1_cf) / e_cf
               - (1 - W) * (Y - mu0_cf) / (1 - e_cf))
        psi = clip(psi, CLIP_DIRECT_PSEUDO)

        self._final = lgb.LGBMRegressor(**LGBM_PARAMS, random_state=self.random_state)
        self._final.fit(X, psi)

        return self

    def predict_difference_cate(self, X, propensity=None):
        return self._final.predict(X)


# Display order: plug-ins (S baseline first), classical DR (S/T) direct then
# log, Q-family DR direct then log, then difference-targeted comparators
# (X, R, classical AIPW DR).
ALL_LEARNER = {
    'S':                lambda rs: SLearner(rs),
    'T':                lambda rs: TLearner(rs),
    'X':                lambda rs: XLearner(rs),
    'R':                lambda rs: RLearner(random_state=rs),
    'DR':               lambda rs: DRLearner(random_state=rs),
    'Q':                lambda rs: QLearner(rs),
    'Q-Simple':         lambda rs: QSimpleLearner(rs),
    'DR-S':             lambda rs: DRSLearner(scale='direct', random_state=rs),
    'DR-T':             lambda rs: DRTLearner(scale='direct', random_state=rs),
    'DR-Q':             lambda rs: DRQLearner(scale='direct', random_state=rs),
    'DR-Q-Simple':      lambda rs: DRQSimpleLearner(scale='direct', random_state=rs),
    'DR-S log':         lambda rs: DRSLearner(scale='log',    random_state=rs),
    'DR-T log':         lambda rs: DRTLearner(scale='log',    random_state=rs),
    'DR-Q log':         lambda rs: DRQLearner(scale='log',    random_state=rs),
    'DR-Q-Simple log':  lambda rs: DRQSimpleLearner(scale='log',    random_state=rs),
}


def get_learner(name: str, random_state: int = 42) -> BaseLearner:
    """Get learner by name. See ALL_LEARNER.keys() for all options."""
    if name not in ALL_LEARNER:
        raise ValueError(f"Unknown: '{name}'. Available: {list(ALL_LEARNER.keys())}")
    return ALL_LEARNER[name](random_state)
