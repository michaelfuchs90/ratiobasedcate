"""
Resume-aware benchmark sweep.

Iterates over (dataset, run, learner) combinations, fits each learner,
and writes per-seed results to a CSV. Re-running resumes from the CSV:
any (dataset, learner, seed) row already present is skipped.

For each (dataset, seed) we additionally fit a single mu_0 model and use
it to recover the missing CATE type (ratio <-> difference) for learners
that natively output only one of them.
"""

import os
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
from tqdm import tqdm

from datasets import get_dataset
from learner import get_learner
from metrics import evaluate_predictions


LGBM_PARAMS = dict(verbose=-1)
MU0_FLOOR = 1e-3   # avoids division by ~0 in tau_r = 1 + tau_d / mu_0


def fit_mu0_predictor(X_train, W_train, Y_train, random_state):
    """Fit mu_0(x) = E[Y | W=0, X=x] on the control arm of the training data."""
    model = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=random_state)
    model.fit(X_train[W_train == 0], Y_train[W_train == 0])
    return model


def predict_mu0(mu0_model, X):
    return np.clip(mu0_model.predict_proba(X)[:, 1], MU0_FLOOR, None)


def recover_cates(tau_r, tau_d, mu0):
    """If one CATE type is None, recover it from the other via mu_0."""
    if tau_r is None and tau_d is not None:
        tau_r = 1.0 + tau_d / mu0
    elif tau_d is None and tau_r is not None:
        tau_d = (tau_r - 1.0) * mu0
    return tau_r, tau_d


def run_benchmark(
    datasets: Iterable[str],
    learners: Iterable[str],
    n_runs: int,
    base_seed: int,
    results_csv: str = 'benchmark_results.csv',
    *,
    existing_results: Optional[List[dict]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the benchmark sweep, resuming from ``results_csv`` if it exists.

    Seeds are ``base_seed + run_idx``. ``existing_results`` overrides the
    CSV when given. Returns one row per (dataset, learner, seed).
    """
    if existing_results is not None:
        results = list(existing_results)
    elif os.path.exists(results_csv):
        df_existing = pd.read_csv(results_csv)
        results = df_existing.to_dict('records')
        if verbose:
            print(f"Loaded {len(results)} existing results from {results_csv}")
    else:
        results = []

    done = {(r['dataset'], r['learner'], r['seed']) for r in results}
    if verbose:
        print(f"Skipping {len(done)} already-completed combinations")

    learners = list(learners)
    datasets = list(datasets)

    for dataset_name in tqdm(datasets, desc='Datasets'):
        for run_idx in range(n_runs):
            seed = base_seed + run_idx

            if all((dataset_name, ln, seed) in done for ln in learners):
                continue

            data = get_dataset(dataset_name, random_state=seed)
            prop_train = data.propensity_true_train
            prop_test = data.propensity_true_test

            # mu_0 model — shared across all learners on this (dataset, seed).
            mu0_model = fit_mu0_predictor(
                data.X_train, data.W_train, data.Y_train, random_state=seed
            )
            mu0_test = predict_mu0(mu0_model, data.X_test)

            for learner_name in learners:
                if (dataset_name, learner_name, seed) in done:
                    continue

                # Q-Simple / DR-Q-Simple require known propensity (RCT only).
                if 'Simple' in learner_name and prop_train is None:
                    continue

                try:
                    learner = get_learner(learner_name, random_state=seed)
                    learner.fit(
                        data.X_train, data.W_train, data.Y_train,
                        propensity=prop_train,
                    )

                    tau_r = learner.predict_ratio_cate(
                        data.X_test, propensity=prop_test
                    )
                    tau_d = learner.predict_difference_cate(
                        data.X_test, propensity=prop_test
                    )
                    tau_r, tau_d = recover_cates(tau_r, tau_d, mu0_test)

                    metrics = evaluate_predictions(
                        tau_ratio_pred=tau_r,
                        tau_difference_pred=tau_d,
                        W=data.W_test,
                        Y=data.Y_test,
                    )

                    results.append({
                        'dataset': dataset_name,
                        'learner': learner_name,
                        'run':     run_idx,
                        'seed':    seed,
                        **metrics,
                    })
                    done.add((dataset_name, learner_name, seed))

                except Exception as e:
                    if verbose:
                        print(f"\nError with {learner_name} on "
                              f"{dataset_name} (seed {seed}): {e}")
                    continue

        # Checkpoint after each dataset so a crash doesn't lose all progress.
        pd.DataFrame(results).to_csv(results_csv, index=False)

    return pd.DataFrame(results)
