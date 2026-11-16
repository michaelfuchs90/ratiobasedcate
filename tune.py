"""Hyperparameter tuning (Optuna TPE, seed 0) per (learner, dataset): maximise a
5-fold CV score on TUNING_SEEDS (disjoint from the benchmark seeds), resumable
via TUNED_PARAMS_CSV. On real data the tuning seeds are other splits of the same
units, so tuning and benchmark test sets share units.
  S / Q-Simple / Q: neg log-loss of the model's own task (Y on X,W / W on X among
                    converters / W on X; Q's p-model reuses the Q-Simple row)
  T1 / T0:          neg log-loss of T's outcome model in the treated / control arm
  *-Qini:           held-out count-based ratio Qini of the learner's tau,
                    IPW-weighted without a design propensity; Q-Qini reuses
                    the Q row's e-model
Q and Q-Qini only on observational data: with a constant true e, tuning drives
ê to a constant, i.e. Q to Q-Simple.
"""
import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from joblib import Parallel, delayed
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedKFold

from datasets import get_dataset
from benchmark import ipw_weights, load_tuned_params
from learner import LGBM_PARAMS, SLearner, QLearner, QSimpleLearner
from metrics import qini_coefficient_ratio

optuna.logging.set_verbosity(optuna.logging.WARNING)

TUNING_SEEDS = range(9000, 9002)
N_FOLDS = 5
N_TRIALS = 100
TUNED_PARAMS_CSV = 'tuned_params.csv'
NEG_LOG_LOSS = get_scorer('neg_log_loss')

# Enqueued as the first trial so TPE always evaluates a known-good region.
_START = dict(min_child_samples=20, min_child_weight=1e-3, learning_rate=0.1, n_estimators=100,
              reg_lambda=1e-3, subsample=1.0, colsample_bytree=1.0, subsample_freq=1)
KNOWN_GOOD = {'S': dict(_START, num_leaves=7), 'Q-Simple': dict(_START, num_leaves=2)}
KNOWN_GOOD['Q'] = KNOWN_GOOD['Q-Simple']
KNOWN_GOOD['T1'] = KNOWN_GOOD['T0'] = KNOWN_GOOD['S']
KNOWN_GOOD['S-Qini'] = KNOWN_GOOD['S']
KNOWN_GOOD['Q-Simple-Qini'] = KNOWN_GOOD['Q-Qini'] = KNOWN_GOOD['Q-Simple']
# Learners tuned on held-out Qini instead of log-loss -> learner class.
QINI_TUNED = {'S-Qini': SLearner, 'Q-Simple-Qini': QSimpleLearner, 'Q-Qini': QLearner}


# LightGBM search space: name -> (type, low, high, log scale). Also printed as
# the paper's search-space table (visualization.table_search_space).
SEARCH_SPACE = {
    'num_leaves':        (int,   2,    128,  True),
    'min_child_samples': (int,   2,    50,   False),
    'min_child_weight':  (float, 1e-6, 1e-1, True),
    'learning_rate':     (float, 0.005, 0.2, True),
    'n_estimators':      (int,   50,   1000, True),
    'reg_lambda':        (float, 1e-3, 20.0, True),
    'subsample':         (float, 0.5,  1.0,  False),
    'colsample_bytree':  (float, 0.5,  1.0,  False),
}


def sample_lgbm_params(trial: optuna.Trial) -> dict:
    p = {name: (trial.suggest_int if kind is int else trial.suggest_float)(name, lo, hi, log=log)
         for name, (kind, lo, hi, log) in SEARCH_SPACE.items()}
    return dict(p, subsample_freq=1)  # required for subsample < 1 to take effect; not tuned


def native_task(learner_name: str, X, W, Y, propensity=None, fit_kw=None):
    """(features, target) of the model being tuned; Qini-tuned learners:
    (X, (learner class, W, Y, propensity, fixed fit kwargs))."""
    if learner_name in QINI_TUNED:
        return X, (QINI_TUNED[learner_name], W, Y, propensity, fit_kw or {})
    if learner_name == 'S':
        return X.assign(_W=W), Y
    if learner_name == 'Q':
        return X, W
    if learner_name in ('T1', 'T0'):
        arm = W == (learner_name == 'T1')
        return X[arm], Y[arm]
    conv = Y == 1  # Q-Simple
    return X[conv], W[conv]


def cv_neg_logloss(X, y, params: dict, seed: int) -> list:
    """Per-fold log-loss jobs, folds capped at the minority-class size (none below 2)."""
    n_folds = min(N_FOLDS, int(np.bincount(y).min()))
    if n_folds < 2:
        return []

    def fold(tr, va):
        model = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=seed, **params).fit(X.iloc[tr], y[tr])
        return float(NEG_LOG_LOSS(model, X.iloc[va], y[va]))
    return [delayed(fold)(tr, va) for tr, va in StratifiedKFold(n_folds).split(X, y)]


def cv_qini(X, task, params: dict, seed: int) -> list:
    """Per-fold jobs of the held-out count-based ratio Qini of the learner in
    ``task``, folds stratified by (W, Y), IPW-weighted if there is no design
    propensity; no jobs if any (W, Y) cell has fewer than 2 rows."""
    model_cls, W, Y, e, fit_kw = task
    strata = 2 * W + Y
    n_folds = min(N_FOLDS, int(np.bincount(strata, minlength=4).min()))
    if n_folds < 2:
        return []

    def fold(tr, va):
        model = model_cls(seed).fit(X.iloc[tr], W[tr], Y[tr], lgbm_params=params,
                                    propensity=None if e is None else e[tr], **fit_kw)
        w = None if e is not None else ipw_weights(X.iloc[tr], W[tr], X.iloc[va], W[va], seed)
        return qini_coefficient_ratio(model.predict_ratio_cate(X.iloc[va]), W[va], Y[va], weights=w)
    return [delayed(fold)(tr, va)
            for tr, va in StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(X, strata)]


def tune_cell(learner_name: str, dataset_name: str, n_trials: int = N_TRIALS,
              inner_jobs: int = 1) -> dict:
    """Best params (+ objective value) for one (learner, dataset); the
    seeds x folds fits of each trial run on ``inner_jobs`` threads."""
    t0 = time.time()
    fit_kw = None
    if learner_name == 'Q-Qini':
        fit_kw = dict(e_lgbm_params=load_tuned_params()[('Q', dataset_name)])
    tasks = []
    for seed in TUNING_SEEDS:
        ds = get_dataset(dataset_name, random_state=seed)
        tasks.append((seed, *native_task(learner_name, ds.X_train, ds.W_train, ds.Y_train,
                                         ds.propensity_true_train, fit_kw)))

    def objective(trial):
        params = sample_lgbm_params(trial)
        score = cv_qini if learner_name in QINI_TUNED else cv_neg_logloss
        jobs = [score(X, y, params, seed) for seed, X, y in tasks]
        flat = iter(Parallel(n_jobs=inner_jobs, backend='threading')(j for js in jobs for j in js))
        losses = [float(np.mean([next(flat) for _ in js])) for js in jobs if js]
        losses = [l for l in losses if not np.isnan(l)]
        if not losses:
            raise optuna.TrialPruned()  # too few minority-class rows for any CV split
        return float(np.mean(losses))

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=0))
    study.enqueue_trial(KNOWN_GOOD[learner_name])
    study.optimize(objective, n_trials=n_trials)

    if any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        params, value = study.best_params, study.best_value
    else:
        params, value = KNOWN_GOOD[learner_name], float('nan')
    params = {k: v for k, v in params.items() if k != 'subsample_freq'}
    return dict(learner=learner_name, dataset=dataset_name, **params, cv_score=value,
                n_trials=n_trials, n_tuning_seeds=len(TUNING_SEEDS), seconds=time.time() - t0)


def run(cells: list[tuple[str, str]], n_trials: int = N_TRIALS, n_jobs: int = 1,
        inner_jobs: int = 1):
    """Tune the (learner, dataset) pairs not yet in TUNED_PARAMS_CSV: one process per
    pair with ``inner_jobs`` threads (n_jobs x inner_jobs ~ cores; each worker loads
    its own data, Criteo ~1.5 GB). Order: Q-Simple before Q, Q before Q-Qini."""
    done = set()
    if os.path.exists(TUNED_PARAMS_CSV):
        done = set(pd.read_csv(TUNED_PARAMS_CSV)[['learner', 'dataset']].itertuples(index=False, name=None))

    pending = list(dict.fromkeys(c for c in cells if c not in done))
    for learner_name, dataset_name in pending:
        if learner_name == 'Q' and ('Q-Simple', dataset_name) not in done | set(pending):
            raise ValueError(f"('Q', '{dataset_name}') needs ('Q-Simple', '{dataset_name}'): "
                             "it supplies Q's p-model params.")
        if learner_name == 'Q-Qini' and ('Q', dataset_name) not in done:
            raise ValueError(f"('Q-Qini', '{dataset_name}') needs ('Q', '{dataset_name}') "
                             "tuned first: it supplies Q-Qini's e-model params.")

    rows = Parallel(n_jobs=n_jobs, return_as='generator_unordered')(
        delayed(tune_cell)(l, ds, n_trials, inner_jobs) for l, ds in pending)
    for row in rows:
        pd.DataFrame([row]).to_csv(TUNED_PARAMS_CSV, mode='a', index=False,
                                   header=not os.path.exists(TUNED_PARAMS_CSV))
        print(f"{row['learner']:<9} {row['dataset']:<20} cv_score={row['cv_score']:+.4f} "
              f"({row['seconds']:.0f}s)", flush=True)
