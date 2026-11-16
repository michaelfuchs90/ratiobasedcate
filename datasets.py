"""
Dataset loaders for the ratio-CATE benchmark.

- Synthetic: ``synthetic_<rate>_<n_conv_train>``, a 33-dataset grid of
  conversion rate x training converter count with oracle tau(x), mu_0(x).
  ``synthconf_<rate>_<n_conv_train>``: same DGP, confounded e(x), no design e.
- RCT: 7 bases, each at 12,800 training converters ``<base>_12800`` or the
  whole pool ``<base>_all`` if smaller (stratified-by-Y subsample, see
  ``load_rct_level``).
- Observational: Cattaneo, NHEFS, JTPA, RHC.

All are ``UpliftDataset`` containers with pre-computed train/test splits.
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from sklearn.model_selection import train_test_split

CACHE_DIR = Path(__file__).parent / "data" / "processed"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR = CACHE_DIR.parent / "raw"


@dataclass
class UpliftDataset:
    """Container for uplift modeling data with train/test splits."""
    X: pd.DataFrame
    W: np.ndarray
    Y: np.ndarray
    tau_true: Optional[np.ndarray] = None
    propensity_true: Optional[np.ndarray] = None
    name: str = ""
    test_size: float = 0.2
    random_state: int = 42
    # False: the last ``test_size`` rows are the test split (caller-set split).
    shuffle: bool = True
    # Exact ground truth, synthetic DGP only. ``oracle_truth`` gates the
    # pointwise/oracle metrics; an approximate tau_true must NOT set it.
    mu0_true: Optional[np.ndarray] = None
    oracle_truth: bool = False

    X_train: pd.DataFrame = field(default=None, repr=False)
    X_test: pd.DataFrame = field(default=None, repr=False)
    W_train: np.ndarray = field(default=None, repr=False)
    W_test: np.ndarray = field(default=None, repr=False)
    Y_train: np.ndarray = field(default=None, repr=False)
    Y_test: np.ndarray = field(default=None, repr=False)
    tau_true_train: np.ndarray = field(default=None, repr=False)
    tau_true_test: np.ndarray = field(default=None, repr=False)
    propensity_true_train: np.ndarray = field(default=None, repr=False)
    propensity_true_test: np.ndarray = field(default=None, repr=False)
    mu0_true_train: np.ndarray = field(default=None, repr=False)
    mu0_true_test: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        self.W = np.asarray(self.W, dtype=np.int32)
        self.Y = np.asarray(self.Y, dtype=np.int32)

        idx = np.arange(len(self.Y))
        idx_train, idx_test = train_test_split(idx, test_size=self.test_size, shuffle=self.shuffle,
                                               random_state=self.random_state if self.shuffle else None)

        self.X_train, self.X_test = self.X.iloc[idx_train], self.X.iloc[idx_test]
        self.W_train, self.W_test = self.W[idx_train], self.W[idx_test]
        self.Y_train, self.Y_test = self.Y[idx_train], self.Y[idx_test]

        if self.tau_true is not None:
            self.tau_true_train, self.tau_true_test = self.tau_true[idx_train], self.tau_true[idx_test]
        if self.propensity_true is not None:
            self.propensity_true_train, self.propensity_true_test = self.propensity_true[idx_train], self.propensity_true[idx_test]
        if self.mu0_true is not None:
            self.mu0_true_train, self.mu0_true_test = self.mu0_true[idx_train], self.mu0_true[idx_test]


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert object/string columns to category dtype for LightGBM and sanitise column names."""
    df = df.copy()
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
    df.columns = [re.sub(r'[{}\[\]"\\,:]', '_', c) for c in df.columns]
    for col in df.columns:
        if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype('category')
    return df


def _load_or_fetch(name: str, fetch_fn) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load from feather cache or fetch via ``fetch_fn`` and persist for next time."""
    path = CACHE_DIR / f"{name}.feather"

    if path.exists():
        df = pd.read_feather(path)
        W = df.pop('__W__').to_numpy(dtype=np.int32)
        Y = df.pop('__Y__').to_numpy(dtype=np.int32)
        df = _prepare_features(df)
        return df, W, Y

    X, W, Y = fetch_fn()
    X = _prepare_features(X)

    # Feather cannot store categoricals with mixed types — coerce to string.
    df_save = X.copy()
    for col in df_save.columns:
        if df_save[col].dtype.name == 'category':
            df_save[col] = df_save[col].astype(str)
    df_save['__W__'] = W
    df_save['__Y__'] = Y
    df_save.to_feather(path)

    return X, W, Y


def _raw(file: str, source: str) -> Path:
    """Path of a raw file in RAW_DIR; tells where to get it if missing."""
    path = RAW_DIR / file
    if not path.exists():
        raise FileNotFoundError(f"{file} not found. Place it in {RAW_DIR}/. Source: {source}")
    return path


def _read_lower(path: Path) -> pd.DataFrame:
    df = pd.read_stata(path) if path.suffix == '.dta' else pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    return df


def _features(df: pd.DataFrame, exclude: set) -> pd.DataFrame:
    return _prepare_features(df[[c for c in df.columns if c not in exclude]])


# --- RCT bases (full data; benchmark datasets are built by load_rct_level) ----

def load_hillstrom(outcome: str = 'visit', test_size: float = 0.2, random_state: int = 42) -> UpliftDataset:
    """Hillstrom Email Marketing Dataset."""

    def fetch():
        from sklift.datasets import fetch_hillstrom
        bunch = fetch_hillstrom(target_col=outcome)

        df = bunch['data'].copy()
        df['treatment'] = bunch['treatment']
        df['target'] = bunch['target']

        # Both e-mail arms count as treated.
        df = df[df['treatment'].isin(['Mens E-Mail', 'Womens E-Mail', 'No E-Mail'])]
        W = (df['treatment'] != 'No E-Mail').astype(int).values
        Y = df['target'].values

        feature_cols = ['recency', 'history', 'mens', 'womens', 'newbie', 'channel', 'zip_code']
        if 'history_segment' in df.columns:
            feature_cols.append('history_segment')

        X = df[[c for c in feature_cols if c in df.columns]]
        return X, W, Y.astype(int)

    X, W, Y = _load_or_fetch(f'hillstrom_{outcome}', fetch)
    e = np.full(len(W), W.mean())
    return UpliftDataset(X=X, W=W, Y=Y, propensity_true=e, name=f'hillstrom_{outcome}', test_size=test_size, random_state=random_state)


def load_criteo(test_size: float = 0.2, random_state: int = 42) -> UpliftDataset:
    """Criteo Uplift Dataset."""

    def fetch():
        from sklift.datasets import fetch_criteo
        data = fetch_criteo()
        X, W, Y = data['data'].copy(), data['treatment'].values, data['target'].values
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])
        return X, W, Y

    X, W, Y = _load_or_fetch('criteo_100pct', fetch)
    e = np.full(len(W), W.mean())
    return UpliftDataset(X=X, W=W, Y=Y, propensity_true=e, name='criteo', test_size=test_size, random_state=random_state)


def load_megafon(test_size: float = 0.2, random_state: int = 42) -> UpliftDataset:
    """Megafon Telecom Churn Dataset."""

    def fetch():
        from sklift.datasets import fetch_megafon
        data = fetch_megafon()
        X = data['data'].copy() if isinstance(data['data'], pd.DataFrame) else pd.DataFrame(data['data'])
        W = (data['treatment'] != 'control').astype(np.int32).values
        return X, W, data['target'].values

    X, W, Y = _load_or_fetch('megafon', fetch)
    e = np.full(len(W), W.mean())
    return UpliftDataset(X=X, W=W, Y=Y, propensity_true=e, name='megafon', test_size=test_size, random_state=random_state)


def load_x5_retail(test_size: float = 0.2, random_state: int = 42) -> UpliftDataset:
    """X5 RetailHero Dataset — SMS promotion campaign."""

    def fetch():
        from sklift.datasets import fetch_x5
        data = fetch_x5()

        clients = data['data']['clients'].copy()
        train = data['data']['train'].copy()
        purchases = data['data']['purchases'].copy()

        W = data['treatment'].values
        Y = data['target'].values

        purchase_agg = purchases.groupby('client_id').agg(
            n_purchases=('transaction_id', 'count'),
            total_spent=('purchase_sum', 'sum'),
            avg_purchase=('purchase_sum', 'mean'),
            n_stores=('store_id', 'nunique'),
            n_products=('product_id', 'nunique'),
        ).reset_index()

        df = train.merge(clients, on='client_id', how='left')\
                  .merge(purchase_agg, on='client_id', how='left')\
                  .fillna(0)

        exclude = {'client_id', 'treatment_flg', 'target', 'first_issue_date', 'first_redeem_date'}
        X = df[[c for c in df.columns if c not in exclude]]

        return X, W, Y

    X, W, Y = _load_or_fetch('x5_retail', fetch)
    e = np.full(len(W), W.mean())
    return UpliftDataset(X=X, W=W, Y=Y, propensity_true=e, name='x5_retail', test_size=test_size, random_state=random_state)


def load_lenta(test_size: float = 0.2, random_state: int = 42) -> UpliftDataset:
    """Lenta grocery retail RCT. The raw CSV (~140 MB) is not redistributed."""
    def fetch():
        df = _read_lower(_raw("lenta_dataset.csv.gz", "https://sklift.s3.eu-west-2.amazonaws.com/lenta_dataset.csv.gz"))
        W = df['group'].map({'test': 1, 'control': 0}).values
        Y = df['response_att'].astype(int).values
        # response_sms / response_viber are post-treatment (they predict W).
        return _features(df, {'group', 'response_att', 'response_sms', 'response_viber'}), W, Y

    X, W, Y = _load_or_fetch('lenta_100pct', fetch)
    e = np.full(len(W), W.mean())
    return UpliftDataset(X=X, W=W, Y=Y, propensity_true=e, name='lenta',
                         test_size=test_size, random_state=random_state)


def load_twins(test_size: float = 0.2, random_state: int = 42) -> UpliftDataset:
    """Twins (Louizos et al., 2017): both potential outcomes observed,
    W ~ Bernoulli(0.5) simulated per seed."""
    src = "https://github.com/AMLab-Amsterdam/CEVAE"
    X = pd.read_csv(_raw("twin_pairs_X_3years_samesex.csv", src), index_col=0)
    Y01 = pd.read_csv(_raw("twin_pairs_Y_3years_samesex.csv", src), index_col=0)
    # The T file holds birth weights (covariates), not the treatment.
    T = pd.read_csv(_raw("twin_pairs_T_3years_samesex.csv", src), index_col=0)
    X = _prepare_features(pd.concat([X, T], axis=1).reset_index(drop=True))

    W = np.random.RandomState(random_state).binomial(1, 0.5, size=len(Y01))
    Y = np.where(W == 1, Y01['mort_1'].values, Y01['mort_0'].values).astype(int)
    return UpliftDataset(X=X, W=W, Y=Y, propensity_true=np.full(len(W), 0.5),
                         name='twins', test_size=test_size, random_state=random_state)


# --- Observational ------------------------------------------------------------

def load_rhc(test_size: float = 0.2, random_state: int = 42) -> UpliftDataset:
    """Right Heart Catheterization (Connors et al., 1996). W: RHC, Y: death within 180 days."""
    def fetch():
        df = _read_lower(_raw("rhc.csv", "https://hbiostat.org/data/repo/rhc.csv"))
        W = (df['swang1'].str.strip().str.upper() == 'RHC').astype(int).values
        Y = (df['death'].str.strip().str.lower() == 'yes').astype(int).values
        # IDs and post-treatment columns; the dates leak Y (dthdte is NaN iff death == 0).
        exclude = {'swang1', 'dth30', 'ptid', 'surv2md1', 't3d30', 'death',
                   'sadmdte', 'dschdte', 'dthdte', 'lstctdte', 'unnamed: 0'}
        return _features(df, exclude), W, Y

    X, W, Y = _load_or_fetch('rhc', fetch)
    return UpliftDataset(X=X, W=W, Y=Y, name='rhc', test_size=test_size, random_state=random_state)


def load_cattaneo(test_size: float = 0.2, random_state: int = 42) -> UpliftDataset:
    """Cattaneo (2010) birthweight data. W: maternal smoking, Y: birth weight < 2500 g."""
    def fetch():
        df = _read_lower(_raw("cattaneo2.dta", "https://raw.githubusercontent.com/scunning1975/mixtape/master/cattaneo2.dta"))
        W = (df['mbsmoke'] == 'Smoker').astype(int).values
        Y = df['lbweight'].astype(int).values
        # msmoke (smoking intensity) determines W.
        return _features(df, {'mbsmoke', 'lbweight', 'bweight', 'msmoke'}), W, Y

    X, W, Y = _load_or_fetch('cattaneo', fetch)
    return UpliftDataset(X=X, W=W, Y=Y, name='cattaneo', test_size=test_size, random_state=random_state)


def load_nhefs(test_size: float = 0.2, random_state: int = 42) -> UpliftDataset:
    """NHEFS (Hernán & Robins). W: quit smoking, Y: death by 1992."""
    def fetch():
        df = _read_lower(_raw("nhefs.csv", "https://cdn1.sph.harvard.edu/wp-content/uploads/sites/1268/1268/20/nhefs.csv"))
        W = df['qsmk'].astype(int).values
        Y = df['death'].astype(int).values
        # IDs and post-treatment columns.
        exclude = {'qsmk', 'death', 'seqn', 'id', 'wt82', 'wt82_71', 'yrdth', 'modth', 'dadth',
                   'smkintensity82_71', 'censored'}
        return _features(df, exclude), W, Y

    X, W, Y = _load_or_fetch('nhefs', fetch)
    return UpliftDataset(X=X, W=W, Y=Y, name='nhefs', test_size=test_size, random_state=random_state)


def load_jtpa(test_size: float = 0.2, random_state: int = 42) -> UpliftDataset:
    """JTPA (Abadie, Angrist & Imbens, 2002). W: actual training participation
    (self-selected), Y: employed afterwards."""
    def fetch():
        df = _read_lower(_raw("jtpa.csv", "Abadie, Angrist & Imbens (2002)"))
        W = df['d2'].astype(int).values
        Y = df['y2'].astype(int).values
        # Random offer / instruments (z1, z2, d1) and post-treatment outcomes.
        return _features(df, {'z1', 'z2', 'd1', 'd2', 'y1', 'y2', 'earnings'}), W, Y

    X, W, Y = _load_or_fetch('jtpa', fetch)
    return UpliftDataset(X=X, W=W, Y=Y, name='jtpa', test_size=test_size, random_state=random_state)


# --- Semi-synthetic DGP --------------------------------------------------------
# Covariates bootstrapped from Criteo; mu_0(x) and tau(x) are smooth surfaces on
# different feature mixes. A shared scale s hits the target E[Y] and leaves
# tau = mu_1/mu_0 unchanged. Known e = 0.5; exact tau(x), mu_0(x) for the oracle.

_SYNTH_FEATURES = ['f0', 'f2', 'f3', 'f5', 'f6', 'f7', 'f9', 'f10']
_SYNTH_POOL_PATH = CACHE_DIR / 'synth_covariate_pool.feather'
_SYNTH_POOL_SIZE = 500_000
_synth_pool_cache: Optional[tuple] = None


def _synth_pool() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Cached (pool, mean, std): a fixed Criteo subsample defining the DGP scale."""
    global _synth_pool_cache
    if _synth_pool_cache is not None:
        return _synth_pool_cache

    if _SYNTH_POOL_PATH.exists():
        pool = pd.read_feather(_SYNTH_POOL_PATH)
    else:
        criteo_path = CACHE_DIR / 'criteo_100pct.feather'
        if not criteo_path.exists():
            load_criteo()  # populate the Criteo cache once
        criteo = pd.read_feather(criteo_path, columns=_SYNTH_FEATURES)
        rng = np.random.RandomState(0)  # fixed: the pool is part of the DGP
        idx = rng.choice(len(criteo), size=min(_SYNTH_POOL_SIZE, len(criteo)),
                         replace=False)
        pool = criteo.iloc[idx].reset_index(drop=True)
        pool.to_feather(_SYNTH_POOL_PATH)

    mean, std = pool.mean(), pool.std().clip(lower=1e-8)
    _synth_pool_cache = (pool, mean, std)
    return _synth_pool_cache


def _std_surface(v: np.ndarray, target_mean: float, target_std: float) -> np.ndarray:
    """Standardise a raw surface to a fixed mean/std (isolates form from scale)."""
    return target_mean + target_std * (v - v.mean()) / (v.std() + 1e-12)


def _baseline_logit(Z: np.ndarray) -> np.ndarray:
    """Logit of the baseline shape g_0(x); mu_0 = s * sigmoid(this)."""
    z0, z2, z6 = Z[:, 0], Z[:, 1], Z[:, 4]
    b = z0 - 0.8 * z2 + 0.6 * z6
    return _std_surface(b, 0.0, 1.2)


def _logtau_fixed(Z: np.ndarray) -> np.ndarray:
    """log tau(x), fixed across the grid; different feature mix than the baseline."""
    z0, z2, z6, z9 = Z[:, 0], Z[:, 1], Z[:, 4], Z[:, 6]
    lt = 0.5 * z2 - 0.4 * z6 + 0.4 * z2 * z9 + 0.3 * np.tanh(z0)
    return _std_surface(lt, 0.25, 0.5)


def _synth_scale(shape: np.ndarray, tau: np.ndarray, target_cvr: float,
                 e: float = 0.5, eps: float = 1e-5) -> float:
    """Bisection for the multiplicative scale s giving marginal E[Y] ~ target."""
    lo, hi = 0.0, 1e6
    for _ in range(100):
        s = 0.5 * (lo + hi)
        mu0 = np.clip(s * shape, eps, 1 - eps)
        mu1 = np.clip(tau * s * shape, eps, 1 - eps)
        ey = np.mean(e * mu1 + (1 - e) * mu0)  # e scalar or per unit
        lo, hi = (s, hi) if ey < target_cvr else (lo, s)
    return 0.5 * (lo + hi)


# Confounded variant: logit e(x) = CONF_SLOPE * z0, clipped for overlap.
CONF_SLOPE = 0.8
CONF_CLIP = (0.1, 0.9)


def load_synthetic_DGP(conversion_rate: float = 0.02, n_conv_train: int = 800,
                       n_test: int = 100_000, e: float = 0.5,
                       random_state: int = 42, eps: float = 1e-5,
                       confounded: bool = False) -> UpliftDataset:
    """Synthetic ratio-CATE dataset at a target conversion rate and training converter count.

    n_train = n_conv_train / conversion_rate; n_test is fixed (the oracle Qini
    needs no test converters). ``eps`` floors mu_0 and mu_1 and must stay well
    below the smallest conversion rate, else tau collapses toward 1.
    ``confounded``: e(x) = sigmoid(CONF_SLOPE * z0), z0 also drives the baseline;
    no design propensity is passed on (handled as observational).
    """
    n_train = max(1, round(n_conv_train / conversion_rate))
    n = n_train + n_test

    pool, mean, std = _synth_pool()
    rng = np.random.RandomState(random_state)
    X = pool.iloc[rng.randint(0, len(pool), size=n)].reset_index(drop=True)
    Z = ((X[_SYNTH_FEATURES] - mean) / std).to_numpy()

    shape = 1.0 / (1.0 + np.exp(-_baseline_logit(Z)))
    tau = np.exp(_logtau_fixed(Z))
    if confounded:
        e = np.clip(1.0 / (1.0 + np.exp(-CONF_SLOPE * Z[:, 0])), *CONF_CLIP)
    s = _synth_scale(shape, tau, conversion_rate, e=e, eps=eps)
    mu0 = np.clip(s * shape, eps, 1 - eps)
    mu1 = np.clip(tau * s * shape, eps, 1 - eps)

    W = rng.binomial(1, e, size=n).astype(np.int32)
    Y = rng.binomial(1, np.where(W == 1, mu1, mu0)).astype(np.int32)
    prefix = 'synthconf' if confounded else 'synthetic'
    return UpliftDataset(X=X, W=W, Y=Y, tau_true=mu1 / mu0, mu0_true=mu0,
                         propensity_true=None if confounded else np.full(n, e), oracle_truth=True,
                         name=f'{prefix}_{conversion_rate}_{n_conv_train}',
                         test_size=n_test, random_state=random_state)


def parse_synthetic_name(name: str) -> tuple[float, int]:
    """(conversion_rate, n_conv_train) from a ``synthetic_`` / ``synthconf_<rate>_<n>`` name."""
    if not name.startswith(('synthetic_', 'synthconf_')):
        raise ValueError(f"Not a synthetic dataset name: {name!r}")
    rate_str, conv_str = name.split('_', 1)[1].rsplit('_', 1)
    return float(rate_str), int(conv_str)


# --- RCT datasets: <base>_12800, or <base>_all (whole pool) if smaller ---------
# Per seed: random 20% test split (capped at RCT_TEST_CAP), train drawn from the
# other 80% stratified by Y at the natural CVR. Sampling within Y strata keeps
# P(W|X,Y), so the design propensity stays valid for Q-Simple.

# Ascending conversion rate.
RCT_BASES = ['H(Conv)', 'Twins', 'Criteo', 'Lenta', 'H(Vis)', 'MegaFon', 'X5']
RCT_MAIN_CONVERTERS = 12_800  # training converters: 12,800, or the whole pool if smaller
RCT_TEST_CAP = 500_000  # only Criteo is affected

# Converters in the 80% train pool (one split; Twins varies by seed).
_RCT_POOL_CONVERTERS = {'H(Conv)': 462, 'Twins': 2005, 'Criteo': 525_543, 'Lenta': 59_494,
                        'H(Vis)': 7515, 'MegaFon': 98_011, 'X5': 99_202}
_rct_base_cache: dict = {}


def _rct_level(base: str):
    return RCT_MAIN_CONVERTERS if _RCT_POOL_CONVERTERS[base] >= RCT_MAIN_CONVERTERS else 'all'


def load_rct_level(base: str, n_conv_train, test_size: float = 0.2,
                   random_state: int = 42) -> UpliftDataset:
    """One RCT base at a training converter count (int, or 'all' for the whole pool)."""
    if base == 'Twins':  # W is simulated per seed -> reload, don't cache
        full = get_dataset(base, random_state=random_state)
    else:
        if base not in _rct_base_cache:
            _rct_base_cache[base] = get_dataset(base)
        full = _rct_base_cache[base]
    X, W, Y = full.X.reset_index(drop=True), full.W, full.Y

    rng = np.random.RandomState(random_state)
    perm = rng.permutation(len(Y))
    n_split = int(round(test_size * len(Y)))
    test, pool = perm[:min(n_split, RCT_TEST_CAP)], perm[n_split:]

    if n_conv_train == 'all':
        train = pool
    else:
        pos, neg = pool[Y[pool] == 1], pool[Y[pool] == 0]
        if n_conv_train > len(pos):
            raise ValueError(f"{base}: pool has {len(pos)} converters < {n_conv_train}.")
        n_neg = int(round(n_conv_train * len(neg) / len(pos)))
        train = np.concatenate([rng.choice(pos, n_conv_train, replace=False),
                                rng.choice(neg, n_neg, replace=False)])
        rng.shuffle(train)

    idx = np.concatenate([train, test])
    return UpliftDataset(X=X.iloc[idx].reset_index(drop=True), W=W[idx], Y=Y[idx],
                         propensity_true=full.propensity_true[idx],
                         tau_true=None if full.tau_true is None else full.tau_true[idx],
                         name=f'{base}_{n_conv_train}', test_size=len(test), shuffle=False,
                         random_state=random_state)


# --- Registry -----------------------------------------------------------------

# Synthetic grid: conversion rate -> training converter counts.
SYNTHETIC_GRID = {
    0.002: [50, 100, 200, 400, 800, 1600],
    0.01:  [50, 100, 200, 400, 800, 1600, 3200, 6400, 12_800],
    0.05:  [50, 100, 200, 400, 800, 1600, 3200, 6400, 12_800],
    0.2:   [50, 100, 200, 400, 800, 1600, 3200, 6400, 12_800],
}

SYNTHETIC_DATASETS = [f'synthetic_{r}_{c}' for r, cs in SYNTHETIC_GRID.items() for c in cs]
SYNTHCONF_DATASETS = [f'synthconf_0.02_{c}' for c in (800, 3200, 12_800)]
RCT_DATASETS = [f'{b}_{_rct_level(b)}' for b in RCT_BASES]
OBS_DATASETS = ['Cattaneo', 'NHEFS', 'JTPA', 'RHC']

# Marginal conversion rate E[Y] over the full sample, ascending within group.
CONVERSION_RATES = {
    'H(Conv)':  0.009,
    'Twins':    0.036,
    'Criteo':   0.047,
    'Lenta':    0.108,
    'H(Vis)':   0.147,
    'MegaFon':  0.204,
    'X5':       0.620,
    'Cattaneo': 0.060,
    'NHEFS':    0.195,
    'JTPA':     0.500,
    'RHC':      0.649,
}
CONVERSION_RATES.update({name: parse_synthetic_name(name)[0] for name in SYNTHETIC_DATASETS + SYNTHCONF_DATASETS})
CONVERSION_RATES.update({name: CONVERSION_RATES[name.rsplit('_', 1)[0]] for name in RCT_DATASETS})


def get_dataset(name: str, **kwargs) -> UpliftDataset:
    """Load dataset by name."""
    loaders = {
        'H(Vis)':   lambda **kw: load_hillstrom('visit', **kw),
        'H(Conv)':  lambda **kw: load_hillstrom('conversion', **kw),
        'Twins':    load_twins,
        'Criteo':   load_criteo,
        'MegaFon':  load_megafon,
        'X5':       load_x5_retail,
        'Lenta':    load_lenta,
        'RHC':      load_rhc,
        'Cattaneo': load_cattaneo,
        'NHEFS':    load_nhefs,
        'JTPA':     load_jtpa,
    }

    if name.startswith('synthetic_'):
        cvr, n_conv_train = parse_synthetic_name(name)
        return load_synthetic_DGP(conversion_rate=cvr, n_conv_train=n_conv_train, **kwargs)
    if name.startswith('synthconf_'):
        cvr, n_conv_train = parse_synthetic_name(name)
        return load_synthetic_DGP(conversion_rate=cvr, n_conv_train=n_conv_train,
                                  confounded=True, **kwargs)

    base, _, level = name.rpartition('_')
    if base in RCT_BASES:
        return load_rct_level(base, level if level == 'all' else int(level), **kwargs)

    if name not in loaders:
        raise ValueError(f"Unknown: {name}. Available: {list(loaders.keys())}")
    return loaders[name](**kwargs)
