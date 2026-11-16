"""
Resume-aware, seed-parallel benchmark loop: one job = one (dataset, seed), fitting
every learner without a row in the results CSV yet.

Failed or inapplicable learners get a row with the reason in ``error``; if only the
calibration fails, Qini is kept and the reason goes to ``cal_error``. Difference
learners are converted via tau_r = 1 + tau_d / mu_0(x). Without a design propensity
(observational data) the metrics are also reported IPW-weighted (``*_ipw``).
"""

import os
from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd
import lightgbm as lgb
from joblib import Parallel, delayed
from tqdm import tqdm

from datasets import get_dataset
from learner import get_learner, clip, EPS, EPS_BOUNDS, LGBM_PARAMS
from metrics import qini_coefficient_ratio, calibration_ratio


# '<name>_tuned' uses the params tune.py stored for <name>; *-Qini: tuned on CV Qini.
TUNED_SUFFIX = '_tuned'
TUNED_MODEL = {'S-Qini': 'S', 'Q-Simple-Qini': 'Q-Simple', 'Q-Qini': 'Q'}  # tuned-params row -> learner class
TUNED_PARAMS_CSV = 'results/tuned_params.csv'


def load_tuned_params(path: str = TUNED_PARAMS_CSV) -> dict:
    """{(base_learner, dataset_name): lgbm_params}, or {} if nothing is tuned yet."""
    from tune import SEARCH_SPACE  # lazy: tune imports benchmark
    if not os.path.exists(path):
        return {}
    out = {}
    for _, row in pd.read_csv(path).iterrows():
        params = {c: kind(row[c]) for c, (kind, *_) in SEARCH_SPACE.items()}
        out[(row['learner'], row['dataset'])] = dict(params, subsample_freq=1)
    return out


def _fit_kwargs(learner_name: str, dataset_name: str, propensity, tuned_params: dict):
    """(base learner name, fit kwargs); a Q row holds e-model params, its
    p-model reuses the Q-Simple row (same task); Q-Qini = the Q row's e-model
    plus its own Qini-tuned p-model."""
    kwargs = dict(propensity=propensity)
    if not learner_name.endswith(TUNED_SUFFIX):
        return learner_name, kwargs
    base = learner_name[:-len(TUNED_SUFFIX)]
    if base == 'T':  # two rows: one per arm
        kwargs['m1_params'] = tuned_params[('T1', dataset_name)]
        kwargs['m0_params'] = tuned_params[('T0', dataset_name)]
    elif base in ('Q', 'Q-Qini'):
        kwargs['e_lgbm_params'] = tuned_params[('Q', dataset_name)]
        kwargs['lgbm_params'] = tuned_params[('Q-Simple' if base == 'Q' else base, dataset_name)]
    else:
        kwargs['lgbm_params'] = tuned_params[(base, dataset_name)]
    return TUNED_MODEL.get(base, base), kwargs


def ipw_weights(X_train, W_train, X_test, W_test, seed: int) -> np.ndarray:
    """1/ê for treated, 1/(1-ê) for control test rows; ê = LightGBM default
    nuisance fitted on the training rows."""
    e_model = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=seed).fit(X_train, W_train)
    e_test = clip(e_model.predict_proba(X_test)[:, 1], EPS_BOUNDS)
    return np.where(W_test == 1, 1 / e_test, 1 / (1 - e_test))


def _mu0_test(data, seed: int) -> np.ndarray:
    """mu_0 on the test rows (control-only LightGBM default), for difference learners."""
    mu0_model = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=seed)
    mu0_model.fit(data.X_train[data.W_train == 0], data.Y_train[data.W_train == 0])
    return mu0_model.predict_proba(data.X_test)[:, 1].clip(EPS, None)


def _fit_predict(data, learner_name: str, dataset_name: str, seed: int, tuned_params: dict,
                 mu0_test: np.ndarray) -> np.ndarray:
    """Fit one learner on the training split; its ratio CATE on the test split."""
    base, kwargs = _fit_kwargs(learner_name, dataset_name, data.propensity_true_train, tuned_params)
    learner = get_learner(base, random_state=seed)
    learner.fit(data.X_train, data.W_train, data.Y_train, **kwargs)
    tau_r = learner.predict_ratio_cate(data.X_test, propensity=data.propensity_true_test)
    if tau_r is None:
        tau_r = 1.0 + learner.predict_difference_cate(data.X_test, propensity=data.propensity_true_test) / mu0_test
    return tau_r


def _run_seed(dataset_name: str, run_idx: int, seed: int, learners: list, tuned_params: dict):
    """Fit and score ``learners`` on one (dataset, seed); one row per learner."""
    data = get_dataset(dataset_name, random_state=seed)
    tau_true = data.tau_true_test if data.oracle_truth else None
    mu0_true = data.mu0_true_test if data.oracle_truth else None
    mu0_test = _mu0_test(data, seed)

    ipw = None
    if data.propensity_true_train is None:
        ipw = ipw_weights(data.X_train, data.W_train, data.X_test, data.W_test, seed)

    rows = []
    for learner_name in learners:
        row = dict(dataset=dataset_name, learner=learner_name, run=run_idx, seed=seed, error=None, cal_error=None)
        rows.append(row)
        if 'Simple' in learner_name and data.propensity_true_train is None:
            row['error'] = 'needs a known constant propensity'
            continue
        try:
            tau_r = _fit_predict(data, learner_name, dataset_name, seed, tuned_params, mu0_test)
            row['frac_nonpos'] = float(np.mean(tau_r <= 0))  # share hitting the cal floor
            row['qini_ratio'] = qini_coefficient_ratio(tau_r, data.W_test, data.Y_test, mu0_true, tau_true)
            if ipw is not None:
                row['qini_ratio_ipw'] = qini_coefficient_ratio(tau_r, data.W_test, data.Y_test, weights=ipw)
            try:
                row.update(calibration_ratio(tau_r, data.W_test, data.Y_test, tau_true))
                if ipw is not None:
                    cal = calibration_ratio(tau_r, data.W_test, data.Y_test, weights=ipw)
                    row.update({f'{k}_ipw': v for k, v in cal.items() if k.endswith('_binned')})
            except Exception as e:
                row['cal_error'] = f"{type(e).__name__}: {e}"
        except Exception as e:
            row['error'] = f"{type(e).__name__}: {e}"
    return rows


def run_benchmark(
    datasets: Iterable[str],
    learners: Iterable[str],
    n_runs: int,
    base_seed: int,
    results_csv: str,
    *,
    n_jobs: int = 1,
    checkpoint_every: int = 10,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run every (dataset, seed = base_seed + run) job with pending learners,
    resuming from ``results_csv``. Each worker holds its own data cache (Criteo
    ~1.5 GB) -- size ``n_jobs`` to the available memory."""
    results = pd.read_csv(results_csv).to_dict('records') if os.path.exists(results_csv) else []
    done = {(r['dataset'], r['learner'], r['seed']) for r in results}

    learners = list(learners)
    jobs = []
    for dataset_name in datasets:
        for run_idx in range(n_runs):
            seed = base_seed + run_idx
            pending = [l for l in learners if (dataset_name, l, seed) not in done]
            if pending:
                jobs.append((dataset_name, run_idx, seed, pending))
    if verbose:
        print(f"{len(results)} existing rows, {len(jobs)} (dataset, seed) jobs to run")
    if not jobs:
        return pd.DataFrame(results)

    tuned_params = load_tuned_params()
    needed = {(l[:-len(TUNED_SUFFIX)], ds) for ds, _, _, pending in jobs for l in pending
              if l.endswith(TUNED_SUFFIX)}
    needed |= {(arm, ds) for l, ds in needed if l == 'T' for arm in ('T1', 'T0')}  # T_tuned
    needed -= {(l, ds) for l, ds in needed if l == 'T'}
    needed |= {('Q-Simple', ds) for l, ds in needed if l == 'Q'}  # Q_tuned's p-model
    needed |= {('Q', ds) for l, ds in needed if l == 'Q-Qini'}    # Q-Qini_tuned's e-model
    missing = needed - tuned_params.keys()
    if missing:
        raise ValueError(f"No tuned params for {sorted(missing)}; run tune.py first.")

    outputs = Parallel(n_jobs=n_jobs, return_as='generator_unordered')(
        delayed(_run_seed)(*job, tuned_params) for job in jobs)
    cal_errors = Counter()
    for i, rows in enumerate(tqdm(outputs, total=len(jobs), disable=not verbose), 1):
        results.extend(rows)
        if verbose:
            for r in rows:
                if r['error']:
                    print(f"\n{r['learner']} on {r['dataset']} (seed {r['seed']}): {r['error']}")
                elif r['cal_error']:
                    cal_errors[r['learner'], r['cal_error']] += 1
        if i % checkpoint_every == 0 or i == len(jobs):
            pd.DataFrame(results).to_csv(results_csv, index=False)
    for (learner, msg), k in cal_errors.items():
        print(f"cal_error {learner} ({k} rows): {msg}")

    return pd.DataFrame(results)
