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


# Single-threaded: the benchmark parallelises over seeds instead.
LGBM_PARAMS = dict(verbose=-1, n_jobs=1)

# Underflow guard
EPS = 1e-10
EPS_BOUNDS = (EPS, 1 - EPS)


def clip(x: np.ndarray, bounds: tuple) -> np.ndarray:
    return np.clip(x, bounds[0], bounds[1])


def _from_scale(pred: np.ndarray, scale: str) -> np.ndarray:
    return np.exp(pred) if scale == 'log' else pred


def _e_hat(propensity, idx, Xtr, Wtr, Xte, rs):
    """Design e on rows idx if given, else a LightGBM default ê."""
    if propensity is not None:
        return np.asarray(propensity)[idx]
    return lgb.LGBMClassifier(**LGBM_PARAMS, random_state=rs).fit(Xtr, Wtr).predict_proba(Xte)[:, 1]


class BaseLearner(ABC):
    """Ratio learners implement ``predict_ratio_cate``, difference learners
    ``predict_difference_cate``; the other one returns None."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, W: np.ndarray, Y: np.ndarray,
            propensity: np.ndarray = None) -> 'BaseLearner':
        pass

    def predict_ratio_cate(self, X: pd.DataFrame, propensity: np.ndarray = None) -> np.ndarray:
        return None

    def predict_difference_cate(self, X: pd.DataFrame, propensity: np.ndarray = None) -> np.ndarray:
        return None


class SLearner(BaseLearner):
    """One outcome model with W as a feature; τ = μ̂(x, 1) / μ̂(x, 0)."""

    def __init__(self, random_state=42):
        self.random_state = random_state

    def fit(self, X, W, Y, propensity=None, lgbm_params: dict = None):
        params = {**LGBM_PARAMS, **(lgbm_params or {})}
        X = X.copy()
        X['_W'] = W
        self._model = lgb.LGBMClassifier(**params, random_state=self.random_state).fit(X, Y)
        return self

    def predict_ratio_cate(self, X, propensity=None):
        X1, X0 = X.copy(), X.copy()
        X1['_W'], X0['_W'] = 1, 0
        mu1 = clip(self._model.predict_proba(X1)[:, 1], EPS_BOUNDS)
        mu0 = clip(self._model.predict_proba(X0)[:, 1], EPS_BOUNDS)
        return mu1 / mu0


class TLearner(BaseLearner):
    """One outcome model per arm; τ = μ̂₁(x) / μ̂₀(x)."""

    def __init__(self, random_state=42):
        self.random_state = random_state

    def fit(self, X, W, Y, propensity=None, m1_params: dict = None, m0_params: dict = None):
        p1, p0 = {**LGBM_PARAMS, **(m1_params or {})}, {**LGBM_PARAMS, **(m0_params or {})}
        self._m1 = lgb.LGBMClassifier(**p1, random_state=self.random_state).fit(X[W == 1], Y[W == 1])
        self._m0 = lgb.LGBMClassifier(**p0, random_state=self.random_state).fit(X[W == 0], Y[W == 0])
        return self

    def predict_ratio_cate(self, X, propensity=None):
        mu1 = clip(self._m1.predict_proba(X)[:, 1], EPS_BOUNDS)
        mu0 = clip(self._m0.predict_proba(X)[:, 1], EPS_BOUNDS)
        return mu1 / mu0


class QLearner(BaseLearner):
    """τ = p/(1-p) · (1-e)/e with p = P(W=1 | X, Y=1) fitted on the converters
    and e = P(W=1 | X) fitted on all units."""

    def __init__(self, random_state=42):
        self.random_state = random_state

    def fit(self, X, W, Y, propensity=None, lgbm_params: dict = None, e_lgbm_params: dict = None):
        p_params = {**LGBM_PARAMS, **(lgbm_params or {})}
        e_params = {**LGBM_PARAMS, **(e_lgbm_params or {})}
        self._e_model = lgb.LGBMClassifier(**e_params, random_state=self.random_state).fit(X, W)
        self._p_model = lgb.LGBMClassifier(**p_params, random_state=self.random_state).fit(X[Y == 1], W[Y == 1])
        return self

    def predict_ratio_cate(self, X, propensity=None):
        # Estimated e even on RCTs: it absorbs the realised imbalance that p̂ picks up.
        e = clip(self._e_model.predict_proba(X)[:, 1], EPS_BOUNDS)
        p = clip(self._p_model.predict_proba(X)[:, 1], EPS_BOUNDS)
        return (p / (1 - p)) * ((1 - e) / e)


class QSimpleLearner(BaseLearner):
    """Q-Learner with a known constant propensity e (randomized data)."""

    def __init__(self, random_state=42):
        self.random_state = random_state

    def fit(self, X, W, Y, propensity=None, lgbm_params: dict = None):
        if propensity is None:
            raise ValueError("QSimpleLearner requires known propensity (designed for RCTs)")
        distinct = np.unique(propensity)
        if len(distinct) != 1:
            raise ValueError(f"QSimpleLearner requires a constant propensity, got {len(distinct)} distinct values.")
        self._e = float(distinct[0])
        params = {**LGBM_PARAMS, **(lgbm_params or {})}
        self._p_model = lgb.LGBMClassifier(**params, random_state=self.random_state).fit(X[Y == 1], W[Y == 1])
        return self

    def predict_ratio_cate(self, X, propensity=None):
        p = clip(self._p_model.predict_proba(X)[:, 1], EPS_BOUNDS)
        return (p / (1 - p)) * ((1 - self._e) / self._e)


class DRBaseLearner(BaseLearner):
    """DR-S / DR-T: cross-fitted μ̂₁, μ̂₀, ê, then L2 regression of the ratio
    pseudo-outcome (``scale`` 'direct' or 'log')."""

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
            e_cf[te] = _e_hat(propensity, te, Xtr, Wtr, Xte, self.random_state)

        mu1_cf = clip(mu1_cf, EPS_BOUNDS)
        mu0_cf = clip(mu0_cf, EPS_BOUNDS)
        e_cf = clip(e_cf, EPS_BOUNDS)

        psi = self._pseudo(W, Y, mu1_cf, mu0_cf, e_cf)
        self._final = lgb.LGBMRegressor(**LGBM_PARAMS, random_state=self.random_state).fit(X, psi)
        return self

    def _pseudo(self, W, Y, mu1, mu0, e):
        if self.scale == 'log':
            # Γ = log(μ₁) - log(μ₀) + W(Y-μ₁)/(e·μ₁) - (1-W)(Y-μ₀)/((1-e)·μ₀)
            r1 = W * (Y - mu1) / (e * mu1)
            r0 = (1 - W) * (Y - mu0) / ((1 - e) * mu0)
            return np.log(mu1) - np.log(mu0) + r1 - r0
        # Γ = τ̂ + W(Y-μ₁)/(e·μ₀) - τ̂·(1-W)(Y-μ₀)/((1-e)·μ₀)
        tau = mu1 / mu0
        a1 = W * (Y - mu1) / (e * mu0)
        a0 = tau * (1 - W) * (Y - mu0) / ((1 - e) * mu0)
        return tau + a1 - a0

    def predict_ratio_cate(self, X, propensity=None):
        return _from_scale(self._final.predict(X), self.scale)

    def _fit_outcome_models(self, Xtr, Xte, Wtr, Ytr):
        raise NotImplementedError("Subclasses must implement _fit_outcome_models")


class DRTLearner(DRBaseLearner):
    """DR with T-Learner outcome models."""

    def _fit_outcome_models(self, Xtr, Xte, Wtr, Ytr):
        m1 = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=self.random_state)
        m0 = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=self.random_state)
        mu1 = m1.fit(Xtr[Wtr == 1], Ytr[Wtr == 1]).predict_proba(Xte)[:, 1]
        mu0 = m0.fit(Xtr[Wtr == 0], Ytr[Wtr == 0]).predict_proba(Xte)[:, 1]
        return mu1, mu0


class DRSLearner(DRBaseLearner):
    """DR with an S-Learner outcome model."""

    def _fit_outcome_models(self, Xtr, Xte, Wtr, Ytr):
        Xtr_aug = Xtr.copy()
        Xtr_aug['_W'] = Wtr
        model = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=self.random_state).fit(Xtr_aug, Ytr)
        Xte_1, Xte_0 = Xte.copy(), Xte.copy()
        Xte_1['_W'], Xte_0['_W'] = 1, 0
        return model.predict_proba(Xte_1)[:, 1], model.predict_proba(Xte_0)[:, 1]


class DRQLearner(BaseLearner):
    """DR-Q (paper eq:dr-q, eq:dr-q-log): cross-fitted p̂, m̂ = P(Y=1 | X), ê,
    then L2 regression of the pseudo-outcome."""

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
            e_cf[te] = _e_hat(propensity, te, Xtr, Wtr, Xte, self.random_state)

        p_cf = clip(p_cf, EPS_BOUNDS)
        m_cf = clip(m_cf, EPS_BOUNDS)
        e_cf = clip(e_cf, EPS_BOUNDS)

        psi = self._pseudo(W, Y, p_cf, e_cf, m_cf)
        self._final = lgb.LGBMRegressor(**LGBM_PARAMS, random_state=self.random_state).fit(X, psi)
        return self

    def _pseudo(self, W, Y, p, e, m):
        if self.scale == 'log':
            # log τ = logit(p) - logit(e), plus the corrections of both logits
            log_tau = np.log(p) - np.log(1 - p) + np.log(1 - e) - np.log(e)
            ap = Y * (W - p) / (m * p * (1 - p))
            ae = (W - e) / (e * (1 - e))
            return log_tau + ap - ae
        # τ = A·B, A = p/(1-p), B = (1-e)/e; correction B·φ_A + A·φ_B
        A = p / (1 - p)
        B = (1 - e) / e
        phiA = Y * (W - p) / (m * (1 - p)**2)
        phiB = -(W - e) / (e**2)
        return A * B + (B * phiA + A * phiB)

    def predict_ratio_cate(self, X, propensity=None):
        return _from_scale(self._final.predict(X), self.scale)


class DRQSimpleLearner(BaseLearner):
    """DR-Q-Simple (paper eq:dr-q-simple): cross-fitted p̂ and known e, fitted
    on the converters only."""

    def __init__(self, scale='log', n_splits=5, random_state=42):
        self.scale, self.n_splits, self.random_state = scale, n_splits, random_state

    def fit(self, X, W, Y, propensity=None):
        if propensity is None:
            raise ValueError("DRQSimpleLearner requires known propensity (designed for RCTs)")

        n = len(Y)
        converters = Y == 1
        p_cf = np.zeros(n)
        X_conv = X[converters].reset_index(drop=True)
        W_conv = W[converters]
        conv_indices = np.where(converters)[0]

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for tr, te in kf.split(X_conv):
            p_model = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=self.random_state)
            p_cf[conv_indices[te]] = p_model.fit(X_conv.iloc[tr], W_conv[tr]).predict_proba(X_conv.iloc[te])[:, 1]

        p_cf_conv = clip(p_cf[converters], EPS_BOUNDS)
        e_conv = clip(propensity[converters], EPS_BOUNDS)
        psi_conv = self._pseudo(W_conv, p_cf_conv, e_conv)
        self._final = lgb.LGBMRegressor(**LGBM_PARAMS, random_state=self.random_state).fit(X[converters], psi_conv)
        return self

    def _pseudo(self, W, p, e):
        if self.scale == 'log':
            # Γ = logit(p) - logit(e) + (W - p) / (p(1-p))
            log_tau = np.log(p) - np.log(1 - p) + np.log(1 - e) - np.log(e)
            return log_tau + (W - p) / (p * (1 - p))
        # Γ = τ + (1-e)(W-p) / (e(1-p)²)
        tau_plugin = (p / (1 - p)) * ((1 - e) / e)
        return tau_plugin + (1 - e) * (W - p) / (e * (1 - p)**2)

    def predict_ratio_cate(self, X, propensity=None):
        return _from_scale(self._final.predict(X), self.scale)


class XLearner(BaseLearner):
    """X-Learner (Künzel et al. 2019), difference scale:
    τ̂(x) = ê(x) · τ̂₀(x) + (1 − ê(x)) · τ̂₁(x)."""

    def __init__(self, random_state=42):
        self.random_state = random_state

    def fit(self, X, W, Y, propensity=None):
        # Stage 1: T-Learner outcome models
        self._m1 = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=self.random_state).fit(X[W == 1], Y[W == 1])
        self._m0 = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=self.random_state).fit(X[W == 0], Y[W == 0])
        mu1 = clip(self._m1.predict_proba(X)[:, 1], EPS_BOUNDS)
        mu0 = clip(self._m0.predict_proba(X)[:, 1], EPS_BOUNDS)

        # Stage 2: imputed effects per arm, one CATE model each
        D1 = Y[W == 1] - mu0[W == 1]
        D0 = mu1[W == 0] - Y[W == 0]
        self._tau1 = lgb.LGBMRegressor(**LGBM_PARAMS, random_state=self.random_state).fit(X[W == 1], D1)
        self._tau0 = lgb.LGBMRegressor(**LGBM_PARAMS, random_state=self.random_state).fit(X[W == 0], D0)

        # Stage 3: propensity for blending (fitted only without a design e)
        self._e_model = None
        if propensity is None:
            self._e_model = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=self.random_state).fit(X, W)
        return self

    def predict_difference_cate(self, X, propensity=None):
        e = propensity if propensity is not None else self._e_model.predict_proba(X)[:, 1]
        e = clip(np.asarray(e, dtype=float), EPS_BOUNDS)
        return e * self._tau0.predict(X) + (1.0 - e) * self._tau1.predict(X)


class RLearner(BaseLearner):
    """R-Learner (Nie & Wager 2021), difference scale; cross-fits m(x) and e(x)."""

    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(self, X, W, Y, propensity=None):
        n = len(Y)
        m_cf, e_cf = np.zeros(n), np.zeros(n)

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for tr, te in kf.split(X):
            Xtr, Xte, Wtr, Ytr = X.iloc[tr], X.iloc[te], W[tr], Y[tr]
            m_cf[te] = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=self.random_state).fit(Xtr, Ytr).predict_proba(Xte)[:, 1]
            e_cf[te] = _e_hat(propensity, te, Xtr, Wtr, Xte, self.random_state)

        m_cf = clip(m_cf, EPS_BOUNDS)
        e_cf = clip(e_cf, EPS_BOUNDS)
        Y_tilde, W_tilde = Y - m_cf, W - e_cf

        # Pseudo-outcome Ỹ/W̃ with |W̃| floored at 1e-5, as EconML's NonParamDML.
        W_tilde_safe = np.copysign(np.maximum(np.abs(W_tilde), 1e-5), W_tilde)
        self._final = lgb.LGBMRegressor(**LGBM_PARAMS, random_state=self.random_state)
        self._final.fit(X, Y_tilde / W_tilde_safe, sample_weight=W_tilde ** 2)
        return self

    def predict_difference_cate(self, X, propensity=None):
        return self._final.predict(X)


class DRLearner(BaseLearner):
    """AIPW DR-Learner (Kennedy 2023), difference scale:
    Γ = μ̂₁ − μ̂₀ + W(Y − μ̂₁)/ê − (1−W)(Y − μ̂₀)/(1−ê)."""

    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(self, X, W, Y, propensity=None):
        n = len(Y)
        mu1_cf, mu0_cf, e_cf = np.zeros(n), np.zeros(n), np.zeros(n)

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for tr, te in kf.split(X):
            Xtr, Xte, Wtr, Ytr = X.iloc[tr], X.iloc[te], W[tr], Y[tr]
            m1 = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=self.random_state).fit(Xtr[Wtr == 1], Ytr[Wtr == 1])
            m0 = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=self.random_state).fit(Xtr[Wtr == 0], Ytr[Wtr == 0])
            mu1_cf[te] = m1.predict_proba(Xte)[:, 1]
            mu0_cf[te] = m0.predict_proba(Xte)[:, 1]
            e_cf[te] = _e_hat(propensity, te, Xtr, Wtr, Xte, self.random_state)

        mu1_cf = clip(mu1_cf, EPS_BOUNDS)
        mu0_cf = clip(mu0_cf, EPS_BOUNDS)
        e_cf = clip(e_cf, EPS_BOUNDS)

        psi = mu1_cf - mu0_cf + W * (Y - mu1_cf) / e_cf - (1 - W) * (Y - mu0_cf) / (1 - e_cf)
        self._final = lgb.LGBMRegressor(**LGBM_PARAMS, random_state=self.random_state).fit(X, psi)
        return self

    def predict_difference_cate(self, X, propensity=None):
        return self._final.predict(X)


ALL_LEARNER = {
    'S':               lambda rs: SLearner(rs),
    'T':               lambda rs: TLearner(rs),
    'X':               lambda rs: XLearner(rs),
    'R':               lambda rs: RLearner(random_state=rs),
    'DR':              lambda rs: DRLearner(random_state=rs),
    'Q':               lambda rs: QLearner(rs),
    'Q-Simple':        lambda rs: QSimpleLearner(rs),
    'DR-S':            lambda rs: DRSLearner(scale='direct', random_state=rs),
    'DR-T':            lambda rs: DRTLearner(scale='direct', random_state=rs),
    'DR-Q':            lambda rs: DRQLearner(scale='direct', random_state=rs),
    'DR-Q-Simple':     lambda rs: DRQSimpleLearner(scale='direct', random_state=rs),
    'DR-S log':        lambda rs: DRSLearner(scale='log', random_state=rs),
    'DR-T log':        lambda rs: DRTLearner(scale='log', random_state=rs),
    'DR-Q log':        lambda rs: DRQLearner(scale='log', random_state=rs),
    'DR-Q-Simple log': lambda rs: DRQSimpleLearner(scale='log', random_state=rs),
}


def get_learner(name: str, random_state: int = 42) -> BaseLearner:
    """Get learner by name. See ALL_LEARNER.keys() for all options."""
    if name not in ALL_LEARNER:
        raise ValueError(f"Unknown: '{name}'. Available: {list(ALL_LEARNER.keys())}")
    return ALL_LEARNER[name](random_state)
