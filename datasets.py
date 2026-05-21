"""
Datasets for Q-Learner Benchmark
================================
Real-world and synthetic datasets for CATE estimation evaluation.
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from sklearn.model_selection import train_test_split

CACHE_DIR = Path("data/processed")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Data Container
# =============================================================================
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

    def __post_init__(self):
        self.W = np.asarray(self.W, dtype=np.int32)
        self.Y = np.asarray(self.Y, dtype=np.int32)

        idx = np.arange(len(self.Y))
        idx_train, idx_test = train_test_split(idx, test_size=self.test_size, random_state=self.random_state)

        self.X_train, self.X_test = self.X.iloc[idx_train], self.X.iloc[idx_test]
        self.W_train, self.W_test = self.W[idx_train], self.W[idx_test]
        self.Y_train, self.Y_test = self.Y[idx_train], self.Y[idx_test]

        if self.tau_true is not None:
            self.tau_true_train, self.tau_true_test = self.tau_true[idx_train], self.tau_true[idx_test]
        if self.propensity_true is not None:
            self.propensity_true_train, self.propensity_true_test = self.propensity_true[idx_train], self.propensity_true[idx_test]


# =============================================================================
# Utilities
# =============================================================================
def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert object/string columns to category dtype for LightGBM."""
    df = df.copy()
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
    df.columns = [re.sub(r'[{}\[\]"\\,:]', '_', c) for c in df.columns]
    for col in df.columns:
        if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype('category')
    return df


def _load_or_fetch(name: str, fetch_fn) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load from feather cache or fetch and save."""
    path = CACHE_DIR / f"{name}.feather"

    if path.exists():
        df = pd.read_feather(path)
        W = df.pop('__W__').to_numpy(dtype=np.int32)
        Y = df.pop('__Y__').to_numpy(dtype=np.int32)
        # Restore category dtypes
        df = _prepare_features(df)
        return df, W, Y

    X, W, Y = fetch_fn()
    X = _prepare_features(X)

    # Save to cache
    df_save = X.copy()
    # Convert categories to string for feather compatibility
    for col in df_save.columns:
        if df_save[col].dtype.name == 'category':
            df_save[col] = df_save[col].astype(str)
    df_save['__W__'] = W
    df_save['__Y__'] = Y
    df_save.to_feather(path)

    return X, W, Y


# =============================================================================
# Real-World Datasets
# =============================================================================
def load_hillstrom(outcome: str = 'visit', test_size: float = 0.2, random_state: int = 42) -> UpliftDataset:
    """Hillstrom Email Marketing Dataset."""

    def fetch():
        from sklift.datasets import fetch_hillstrom

        # IMPORTANT: pass target_col so the requested outcome is loaded.
        bunch = fetch_hillstrom(target_col=outcome)

        df = bunch['data'].copy()
        treatment = bunch['treatment']
        target = bunch['target']

        # Attach treatment to the dataframe for filtering.
        df['treatment'] = treatment
        df['target'] = target

        # Keep only the relevant treatment arms.
        mask = df['treatment'].isin(['Mens E-Mail', 'Womens E-Mail', 'No E-Mail'])
        df = df[mask]

        W = (df['treatment'] != 'No E-Mail').astype(int).values
        Y = df['target'].values

        feature_cols = ['recency', 'history', 'mens', 'womens', 'newbie', 'channel', 'zip_code']
        # Also include history_segment if available.
        if 'history_segment' in df.columns:
            feature_cols.append('history_segment')

        X = df[[c for c in feature_cols if c in df.columns]]
        return X, W, Y.astype(int)

    X, W, Y = _load_or_fetch(f'hillstrom_{outcome}', fetch)
    e = np.full(len(W), W.mean())
    return UpliftDataset(X=X, W=W, Y=Y, propensity_true=e, name=f'hillstrom_{outcome}', test_size=test_size, random_state=random_state)


def load_criteo(sample_frac: float = 1, test_size: float = 0.2, random_state: int = 42) -> UpliftDataset:
    """Criteo Uplift Dataset (sampled)."""

    def fetch():
        from sklift.datasets import fetch_criteo
        data = fetch_criteo()
        X, W, Y = data['data'].copy(), data['treatment'].values, data['target'].values

        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(Y), size=int(len(Y) * sample_frac), replace=False)
        X = X.iloc[idx].reset_index(drop=True)

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])
        return X, W[idx], Y[idx]

    X, W, Y = _load_or_fetch(f'criteo_{int(sample_frac*100)}pct', fetch)
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
    """X5 RetailHero Dataset - SMS promotion campaign."""

    def fetch():
        from sklift.datasets import fetch_x5
        data = fetch_x5()

        # data['data'] ist ein Bunch mit clients, train, purchases
        clients = data['data']['clients'].copy()
        train = data['data']['train'].copy()
        purchases = data['data']['purchases'].copy()

        # Treatment and target are directly available.
        W = data['treatment'].values
        Y = data['target'].values

        # Aggregate purchase features
        purchase_agg = purchases.groupby('client_id').agg(
            n_purchases=('transaction_id', 'count'),
            total_spent=('purchase_sum', 'sum'),
            avg_purchase=('purchase_sum', 'mean'),
            n_stores=('store_id', 'nunique'),
            n_products=('product_id', 'nunique'),
        ).reset_index()

        # Merge all data
        df = train.merge(clients, on='client_id', how='left')\
                  .merge(purchase_agg, on='client_id', how='left')\
                  .fillna(0)

        # Select features (exclude IDs and targets)
        exclude = {'client_id', 'treatment_flg', 'target', 'first_issue_date', 'first_redeem_date'}
        X = df[[c for c in df.columns if c not in exclude]]

        return X, W, Y

    X, W, Y = _load_or_fetch('x5_retail', fetch)
    e = np.full(len(W), W.mean())
    return UpliftDataset(X=X, W=W, Y=Y, propensity_true=e, name='x5_retail', test_size=test_size, random_state=random_state)


# =============================================================================
# Additional Real-World / Semi-Synthetic Datasets
# =============================================================================
RAW_DIR = CACHE_DIR.parent / "raw"


def load_twins(n_bins_tau: int = 50, test_size: float = 0.2, random_state: int = 42) -> UpliftDataset:
    """Twins semi-synthetic dataset (Louizos et al., 2017).

    Both potential outcomes observed → ground truth tau available.
    Treatment is simulated: W ~ Bernoulli(0.5).
    Source: https://github.com/AMLab-Amsterdam/CEVAE
    """
    rng = np.random.RandomState(random_state)

    # --- locate files (3-file CEVAE format or single combined CSV) ---
    x_path = RAW_DIR / "twin_pairs_X_3years_samesex.csv"
    y_path = RAW_DIR / "twin_pairs_Y_3years_samesex.csv"
    t_path = RAW_DIR / "twin_pairs_T_3years_samesex.csv"
    combined_path = RAW_DIR / "twins.csv"

    if x_path.exists() and y_path.exists():
        df_x = pd.read_csv(x_path, index_col=0)
        df_y = pd.read_csv(y_path, index_col=0)
        Y0 = df_y['mort_0'].values.astype(int)
        Y1 = df_y['mort_1'].values.astype(int)
        # T file contains birth weights (covariates), not treatment assignment
        if t_path.exists():
            df_t = pd.read_csv(t_path, index_col=0)
            df_x = pd.concat([df_x, df_t], axis=1)
        X = df_x
    elif combined_path.exists():
        df = pd.read_csv(combined_path, index_col=0)
        cols_lower = {c: c.lower().strip() for c in df.columns}
        df.rename(columns=cols_lower, inplace=True)
        y0_col = next((c for c in df.columns if c in ('mort_0', 'y0', 'y_0')), None)
        y1_col = next((c for c in df.columns if c in ('mort_1', 'y1', 'y_1')), None)
        if y0_col is None or y1_col is None:
            raise ValueError(f"Cannot detect outcome columns. Found: {list(df.columns[:10])}...")
        Y0 = df[y0_col].values.astype(int)
        Y1 = df[y1_col].values.astype(int)
        X = df.drop(columns=[y0_col, y1_col])
    else:
        raise FileNotFoundError(
            f"Twins data not found in {RAW_DIR}/.\n"
            "Expected: twin_pairs_X/Y/T_3years_samesex.csv (from CEVAE repo)\n"
            "   or: twins.csv (combined format).\n"
            "Source: https://github.com/AMLab-Amsterdam/CEVAE"
        )

    # Drop rows with missing outcomes
    valid = np.isfinite(Y0) & np.isfinite(Y1)
    X, Y0, Y1 = X.loc[valid].reset_index(drop=True), Y0[valid], Y1[valid]

    X = _prepare_features(X)

    # --- simulate treatment (RCT-like, e = 0.5) ---
    W = rng.binomial(1, 0.5, size=len(Y0))
    Y = np.where(W == 1, Y1, Y0)
    e_x = np.full(len(W), 0.5)

    # --- compute ground truth tau via binning ---
    X_num = X.select_dtypes(include=[np.number])
    score = ((X_num - X_num.mean()) / X_num.std().clip(lower=1e-8)).sum(axis=1).values
    bin_edges = np.percentile(score, np.linspace(0, 100, n_bins_tau + 1))
    bins = np.clip(np.searchsorted(bin_edges[1:-1], score), 0, n_bins_tau - 1)

    tau_true = np.ones(len(Y), dtype=float)
    for b in range(n_bins_tau):
        mask = bins == b
        if mask.sum() >= 5:
            mu0_bin = max(Y0[mask].mean(), 0.001)
            mu1_bin = Y1[mask].mean()
            tau_true[mask] = mu1_bin / mu0_bin

    return UpliftDataset(X=X, W=W, Y=Y, tau_true=tau_true, propensity_true=e_x,
                         name='twins', test_size=test_size, random_state=random_state)


def load_lenta(sample_frac: float = 1, test_size: float = 0.2, random_state: int = 42) -> UpliftDataset:
    """Lenta grocery retail uplift dataset (Kaggle RCT)."""
    def fetch():
        candidates = [RAW_DIR / "lenta_dataset.csv.gz", RAW_DIR / "lenta_dataset.csv",
                      RAW_DIR / "lenta.csv", Path("lenta_dataset.csv")]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            raise FileNotFoundError(
                f"Lenta data not found. Place lenta_dataset.csv in {RAW_DIR}/.\n"
                "Source: https://sklift.s3.eu-west-2.amazonaws.com/lenta_dataset.csv.gz"
            )

        df = pd.read_csv(path)
        cols_lower = {c: c.lower().strip() for c in df.columns}
        df.rename(columns=cols_lower, inplace=True)

        # Detect treatment column
        treat_col = next((c for c in df.columns if c in ('group', 'treatment_flg', 'treatment')), None)
        if treat_col is None:
            raise ValueError(f"Cannot detect treatment column. Found: {list(df.columns)}")

        # Detect outcome column
        outcome_col = next((c for c in df.columns if c in ('response_att', 'target', 'response')), None)
        if outcome_col is None:
            raise ValueError(f"Cannot detect outcome column. Found: {list(df.columns)}")

        # Treatment: map string labels to 0/1 if needed
        if df[treat_col].dtype == object or pd.api.types.is_string_dtype(df[treat_col]):
            W = df[treat_col].map({'test': 1, 'treatment': 1, 'control': 0}).values
        else:
            W = df[treat_col].astype(int).values

        Y = df[outcome_col].astype(int).values

        # Features: everything except treatment and outcome
        exclude = {treat_col, outcome_col}
        X = df[[c for c in df.columns if c not in exclude]].copy()

        # Sample down (dataset is very large)
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(Y), size=int(len(Y) * sample_frac), replace=False)
        return _prepare_features(X.iloc[idx].reset_index(drop=True)), W[idx], Y[idx]

    X, W, Y = _load_or_fetch(f'lenta_{int(sample_frac * 100)}pct', fetch)
    e = np.full(len(W), W.mean())
    return UpliftDataset(X=X, W=W, Y=Y, propensity_true=e, name='lenta',
                         test_size=test_size, random_state=random_state)


def load_rhc(test_size: float = 0.2, random_state: int = 42) -> UpliftDataset:
    """Right Heart Catheterization dataset (Connors et al., 1996). Observational."""
    def fetch():
        candidates = [RAW_DIR / "rhc.csv", Path("rhc.csv")]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            raise FileNotFoundError(
                f"RHC data not found. Place rhc.csv in {RAW_DIR}/.\n"
                "Source: https://hbiostat.org/data/repo/rhc.csv"
            )

        df = pd.read_csv(path)
        cols_lower = {c: c.lower().strip() for c in df.columns}
        df.rename(columns=cols_lower, inplace=True)

        # Treatment: swang1 = 'RHC' vs 'No RHC'
        treat_col = next((c for c in df.columns if c in ('swang1', 'rhc', 'treatment')), None)
        if treat_col is None:
            raise ValueError(f"Cannot detect treatment column. Found: {list(df.columns)}")

        if df[treat_col].dtype == object or pd.api.types.is_string_dtype(df[treat_col]):
            W = (df[treat_col].str.strip().str.upper() == 'RHC').astype(int).values
        else:
            W = df[treat_col].astype(int).values

        # Outcome: prefer dth30 (30-day mortality); fallback to death
        outcome_col = next((c for c in df.columns if c in ('dth30', 'death', 'dth')), None)
        if outcome_col is None:
            raise ValueError(f"Cannot detect outcome column. Found: {list(df.columns)}")

        if df[outcome_col].dtype == object or pd.api.types.is_string_dtype(df[outcome_col]):
            Y = df[outcome_col].str.strip().str.lower().map(
                {'yes': 1, 'no': 0, 'true': 1, 'false': 0, '1': 1, '0': 0}
            ).fillna(0).astype(int).values
        else:
            Y = df[outcome_col].astype(int).values

        # Features: exclude IDs, treatment, outcome, survival time
        exclude = {treat_col, outcome_col, 'ptid', 'surv2md1', 't3d30', 'dth30', 'death',
                   'unnamed: 0', ''}
        feature_cols = [c for c in df.columns if c not in exclude and c.strip() != '']
        X = df[feature_cols].copy()
        return _prepare_features(X), W, Y

    X, W, Y = _load_or_fetch('rhc', fetch)
    # Observational → propensity unknown, will be estimated by learners
    return UpliftDataset(X=X, W=W, Y=Y, propensity_true=None, name='rhc',
                         test_size=test_size, random_state=random_state)


def load_cattaneo(test_size: float = 0.2, random_state: int = 42) -> UpliftDataset:
    """Cattaneo (2010) birthweight dataset. Observational.

    Treatment: maternal smoking during pregnancy.
    Outcome: low birth weight (< 2500g).
    Source: https://raw.githubusercontent.com/scunning1975/mixtape/master/cattaneo2.dta
    """
    def fetch():
        candidates = [RAW_DIR / "cattaneo2.dta", RAW_DIR / "cattaneo2.csv"]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            raise FileNotFoundError(
                f"Cattaneo data not found. Place cattaneo2.dta or cattaneo2.csv in {RAW_DIR}/.\n"
                "Source: https://raw.githubusercontent.com/scunning1975/mixtape/master/cattaneo2.dta"
            )

        if path.suffix == '.dta':
            df = pd.read_stata(path)
        else:
            df = pd.read_csv(path)

        cols_lower = {c: c.lower().strip() for c in df.columns}
        df.rename(columns=cols_lower, inplace=True)

        # Treatment: mbsmoke (maternal smoking)
        treat_col = next((c for c in df.columns if c in ('mbsmoke', 'smoke', 'treatment')), None)
        if treat_col is None:
            raise ValueError(f"Cannot detect treatment column. Found: {list(df.columns)}")

        if df[treat_col].dtype == object or isinstance(df[treat_col].dtype, pd.CategoricalDtype):
            mapping = {v: (1 if v.lower() in ('smoker', 'yes', '1') else 0) for v in df[treat_col].unique()}
            W = df[treat_col].map(mapping).values
        else:
            W = df[treat_col].astype(int).values

        # Outcome: low birth weight (lbweight or bweight < 2500)
        if 'lbweight' in df.columns:
            Y = df['lbweight'].astype(int).values
        elif 'bweight' in df.columns:
            Y = (df['bweight'] < 2500).astype(int).values
        else:
            raise ValueError(f"Cannot detect outcome column. Found: {list(df.columns)}")

        # Features: exclude treatment, outcome, and smoking intensity (perfectly predicts treatment)
        exclude = {treat_col, 'lbweight', 'bweight', 'msmoke'}
        feature_cols = [c for c in df.columns if c not in exclude and c.strip() != '']
        X = df[feature_cols].copy()
        return _prepare_features(X), W, Y

    X, W, Y = _load_or_fetch('cattaneo', fetch)
    return UpliftDataset(X=X, W=W, Y=Y, propensity_true=None, name='cattaneo',
                         test_size=test_size, random_state=random_state)


def load_nhefs(test_size: float = 0.2, random_state: int = 42) -> UpliftDataset:
    """NHEFS dataset (Hernan & Robins, Causal Inference textbook). Observational.

    Treatment: quit smoking (1 = quit, 0 = did not quit).
    Outcome: death by 1992.
    Source: https://cdn1.sph.harvard.edu/wp-content/uploads/sites/1268/1268/20/nhefs.csv
    """
    def fetch():
        candidates = [RAW_DIR / "nhefs.csv", RAW_DIR / "nhefs.dta"]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            raise FileNotFoundError(
                f"NHEFS data not found. Place nhefs.csv in {RAW_DIR}/.\n"
                "Source: https://cdn1.sph.harvard.edu/wp-content/uploads/sites/1268/1268/20/nhefs.csv"
            )

        if path.suffix == '.dta':
            df = pd.read_stata(path)
        else:
            df = pd.read_csv(path)

        cols_lower = {c: c.lower().strip() for c in df.columns}
        df.rename(columns=cols_lower, inplace=True)

        # Treatment: qsmk (quit smoking)
        treat_col = next((c for c in df.columns if c in ('qsmk', 'quit_smoking', 'treatment')), None)
        if treat_col is None:
            raise ValueError(f"Cannot detect treatment column. Found: {list(df.columns)}")
        W = df[treat_col].astype(int).values

        # Outcome: death
        outcome_col = next((c for c in df.columns if c in ('death', 'died', 'y')), None)
        if outcome_col is None:
            raise ValueError(f"Cannot detect outcome column. Found: {list(df.columns)}")
        Y = df[outcome_col].astype(int).values

        # Features: exclude treatment, outcome, IDs, post-treatment vars
        exclude = {treat_col, outcome_col, 'seqn', 'id', 'wt82', 'wt82_71',
                   'yrdth', 'modth', 'dadth', 'smkintensity82_71', 'censored'}
        feature_cols = [c for c in df.columns if c not in exclude and c.strip() != '']
        X = df[feature_cols].copy()

        # Drop rows with missing outcome or treatment
        valid = np.isfinite(W) & np.isfinite(Y)
        X, W, Y = X.loc[valid].reset_index(drop=True), W[valid], Y[valid]
        return _prepare_features(X), W, Y

    X, W, Y = _load_or_fetch('nhefs', fetch)
    return UpliftDataset(X=X, W=W, Y=Y, propensity_true=None, name='nhefs',
                         test_size=test_size, random_state=random_state)


def load_jtpa(test_size: float = 0.2, random_state: int = 42) -> UpliftDataset:
    """JTPA (Job Training Partnership Act) dataset. Observational.

    Treatment: D2 — actual job-training participation (confounded by self-selection).
    Outcome: Y2 — employed post-training (binary).
    Features: sex, n_hs2 (months since HS), edu, prevearn.
    Source: Abadie, Angrist & Imbens (2002).
    """
    def fetch():
        candidates = [RAW_DIR / "jtpa.csv", RAW_DIR / "jtpa.dta"]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            raise FileNotFoundError(
                f"JTPA data not found. Place jtpa.csv in {RAW_DIR}/."
            )

        if path.suffix == '.dta':
            df = pd.read_stata(path)
        else:
            df = pd.read_csv(path)

        cols_lower = {c: c.lower().strip() for c in df.columns}
        df.rename(columns=cols_lower, inplace=True)

        # Treatment: actual participation D2 (observational, confounded)
        W = df['d2'].astype(int).values
        # Outcome: employed Y2
        Y = df['y2'].astype(int).values

        # Features: pre-treatment covariates only
        # Exclude instruments (z1, z2), treatments (d1, d2), outcomes (y1, y2), post-treatment earnings
        exclude = {'z1', 'z2', 'd1', 'd2', 'y1', 'y2', 'earnings'}
        feature_cols = [c for c in df.columns if c not in exclude]
        X = df[feature_cols].copy()

        return _prepare_features(X), W, Y

    X, W, Y = _load_or_fetch('jtpa', fetch)
    return UpliftDataset(X=X, W=W, Y=Y, propensity_true=None, name='jtpa',
                         test_size=test_size, random_state=random_state)


# =============================================================================
# Factory
# =============================================================================

# Datasets are listed in ascending order of conversion rate within each group.
RCT_DATASETS = ['H(Conv)', 'Twins', 'Criteo', 'Lenta', 'H(Vis)', 'MegaFon', 'X5']
OBS_DATASETS = ['Cattaneo', 'NHEFS', 'JTPA', 'RHC']
ALL_DATASETS = RCT_DATASETS + OBS_DATASETS

# Marginal conversion rate per dataset, measured on the loaded data
# (E[Y] over the full sample). Used by analysis code that splits results
# by conversion-rate regime. Values may differ slightly from the paper's
# Tables 1 and 2 when the loader binarizes/filters the raw data.
CONVERSION_RATES = {
    # RCT, ascending
    'H(Conv)':  0.009,
    'Twins':    0.036,
    'Criteo':   0.047,
    'Lenta':    0.108,
    'H(Vis)':   0.147,
    'MegaFon':  0.204,
    'X5':       0.620,
    # Observational, ascending
    'Cattaneo': 0.060,
    'NHEFS':    0.195,
    'JTPA':     0.500,
    'RHC':      0.649,
}

def get_dataset(name: str, **kwargs) -> UpliftDataset:
    """Load dataset by name."""
    loaders = {
        'H(Vis)':   lambda **kw: load_hillstrom('visit', **kw),
        'H(Conv)':  lambda **kw: load_hillstrom('conversion', **kw),
        'Criteo':   load_criteo,
        'MegaFon':  load_megafon,
        'X5':       load_x5_retail,
        'Lenta':    load_lenta,
        'RHC':      load_rhc,
        'Twins':    load_twins,
        'Cattaneo': load_cattaneo,
        'NHEFS':    load_nhefs,
        'JTPA':     load_jtpa,
    }

    if name not in loaders:
        raise ValueError(f"Unknown: {name}. Available: {list(loaders.keys())}")
    return loaders[name](**kwargs)


