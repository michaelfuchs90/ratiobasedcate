"""
s_test diagnostic for the corrected resampled t-test on the subsampled RCT
datasets (paper, Appendix app:stest): rho = o_test * s_test + o_train * (1 - s_test),
variance factor rho / (1 - rho).
``run`` refits the learners as the benchmark does and bootstraps their test
split into STEST_CSV (b = -1: unresampled, must reproduce results_rct.csv;
see ``check_unresampled``); ``summarise`` turns that into s_test, rho and the factor.
Pool-sized ``*_all`` datasets keep the classic factor 0.25.
"""

import os
from itertools import combinations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from benchmark import _fit_predict, _mu0_test, load_tuned_params
from datasets import RCT_DATASETS, RCT_TEST_CAP, _RCT_POOL_CONVERTERS, get_dataset
from metrics import qini_coefficient_ratio, calibration_ratio

STEST_LEARNERS = ['S', 'S_tuned', 'S-Qini_tuned', 'Q-Simple', 'Q-Simple_tuned', 'Q-Simple-Qini_tuned']
STEST_CELLS = [d for d in RCT_DATASETS if not d.endswith('_all')]
STEST_SEEDS = [42, 43, 44, 45, 46]  # first five benchmark seeds
STEST_CSV = 'results/stest_boot.csv'


def _scores(tau, W, Y) -> dict:
    cal = calibration_ratio(tau, W, Y)
    return dict(qini_ratio=qini_coefficient_ratio(tau, W, Y), cal_binned=cal['cal_binned'],
                slope_binned=cal['slope_binned'])


def _boot_one(dataset: str, seed: int, learners: list, n_boot: int, tuned_params: dict) -> list:
    """Fit ``learners`` on one (dataset, seed); score them on the test split
    (b = -1) and on ``n_boot`` bootstrap resamples of it (same rows for all learners)."""
    data = get_dataset(dataset, random_state=seed)
    mu0_test = _mu0_test(data, seed)
    taus = {l: _fit_predict(data, l, dataset, seed, tuned_params, mu0_test) for l in learners}
    W, Y = np.asarray(data.W_test), np.asarray(data.Y_test)
    rng = np.random.RandomState(seed)
    rows = []
    for b in range(-1, n_boot):
        idx = np.arange(len(Y)) if b < 0 else rng.randint(0, len(Y), len(Y))
        for l, tau in taus.items():
            rows.append(dict(dataset=dataset, seed=seed, b=b, learner=l, **_scores(tau[idx], W[idx], Y[idx])))
    return rows


def run(cells: list = STEST_CELLS, seeds: list = STEST_SEEDS, learners: list = STEST_LEARNERS,
        n_boot: int = 200, n_jobs: int = 1, out_csv: str = STEST_CSV) -> pd.DataFrame:
    """Bootstrap every pending (dataset, seed) of ``cells`` x ``seeds``; one
    job per pair, the main process appends each job's rows to ``out_csv``."""
    done = set()
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv, usecols=['dataset', 'seed'])
        done = set(zip(prev['dataset'], prev['seed']))
    jobs = [(d, s) for d in cells for s in seeds if (d, s) not in done]
    tuned_params = load_tuned_params()
    print(f'{len(jobs)} jobs pending ({len(done)} done)')
    out = Parallel(n_jobs=n_jobs, return_as='generator_unordered')(
        delayed(_boot_one)(d, s, learners, n_boot, tuned_params) for d, s in jobs)
    for rows in tqdm(out, total=len(jobs)):
        pd.DataFrame(rows).to_csv(out_csv, mode='a', header=not os.path.exists(out_csv), index=False)
    return pd.read_csv(out_csv)


def overlap(dataset: str) -> tuple:
    """(o_test, o_train) of a subsampled RCT cell ``<base>_<n_conv>``."""
    base, n_conv = dataset.rsplit('_', 1)
    n = len(get_dataset(base).Y)
    return min(round(0.2 * n), RCT_TEST_CAP) / n, int(n_conv) / _RCT_POOL_CONVERTERS[base]


def _metric(df: pd.DataFrame, metric: str) -> pd.Series:
    return np.log(df['cal_binned']) if metric == 'log_cal_binned' else df[metric]


def _diffs(df: pd.DataFrame, a: str, b: str, metric: str, index: list) -> pd.Series:
    v = df.assign(v=_metric(df, metric)).pivot_table(index=index, columns='learner', values='v')
    return (v[a] - v[b]).replace([np.inf, -np.inf], np.nan).dropna()


def summarise(boot: pd.DataFrame, results: pd.DataFrame, learners: list = STEST_LEARNERS,
              metrics: tuple = ('qini_ratio', 'log_cal_binned')) -> pd.DataFrame:
    """Per (dataset, pair, metric): s_test = mean bootstrap variance of the paired
    difference / its variance over all seeds in ``results``, rho and rho / (1 - rho)."""
    bs = boot[boot['b'] >= 0]
    rows = []
    for dataset in sorted(set(bs['dataset'])):
        o_test, o_train = overlap(dataset)
        b_d, r_d = bs[bs['dataset'] == dataset], results[results['dataset'] == dataset]
        for a, b in combinations(learners, 2):
            for metric in metrics:
                v_test = _diffs(b_d, a, b, metric, ['seed', 'b']).groupby(level='seed').var().mean()
                v_total = _diffs(r_d, a, b, metric, ['seed']).var()
                s_test = float(np.clip(v_test / v_total, 0, 1)) if v_total > 0 else np.nan
                rho = o_test * s_test + o_train * (1 - s_test)
                rows.append(dict(dataset=dataset, a=a, b=b, metric=metric, v_test=v_test, v_total=v_total,
                                 s_test=s_test, o_test=o_test, o_train=o_train, rho=rho, nb_factor=rho / (1 - rho)))
    return pd.DataFrame(rows)


def check_unresampled(boot: pd.DataFrame, results: pd.DataFrame, tol: float = 1e-9) -> pd.DataFrame:
    """b = -1 rows against results_rct.csv: the refits must reproduce the benchmark."""
    key = ['dataset', 'seed', 'learner']
    m = boot[boot['b'] == -1].merge(results, on=key, suffixes=('', '_bench'))
    m['ok'] = (m['qini_ratio'] - m['qini_ratio_bench']).abs() < tol
    return m[key + ['qini_ratio', 'qini_ratio_bench', 'ok']]
