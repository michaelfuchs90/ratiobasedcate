"""Targeted: add more seeds for a few datasets to flip currently-grey cells.

Per-dataset recommendation (from the heatmap-significance analysis on
the qini_ratio metric, paired test):

    Dataset    Current   Target   Additional   Notes
    --------   -------   ------   ----------   -----------------------------
    Lenta         50       100         50      Headline low-conv, 3/5 grey flip
    H(Conv)       50       200        150      Low-conv headline; 5/7 flip
    NHEFS         50       300        250      Obs (DR story); all 5 flip
    JTPA          50       150        100      Obs; 2/4 flip
    H(Vis)        50       100         50      RCT; 1/2 flip
    -- skipped: X5, Cattaneo, Twins, RHC (poor ROI),
    --          Criteo, MegaFon (already fully significant).

Run from the project root with:

    /c/Python314/python add_seeds.py
"""

from learner import ALL_LEARNER
from benchmark import run_benchmark


PER_DATASET_TARGET = {
    'Lenta':   100,
    'H(Vis)':  100,
    'JTPA':    150,
    'H(Conv)': 200,
    'NHEFS':   300,
}

LEARNERS = list(ALL_LEARNER.keys())
BASE_SEED = 42
RESULTS_CSV = 'benchmark_results.csv'


def main() -> None:
    # Order: smallest target first → cheapest wins arrive sooner
    for dataset, target_seeds in sorted(PER_DATASET_TARGET.items(),
                                        key=lambda kv: kv[1]):
        print()
        print('=' * 72)
        print(f'Dataset: {dataset}  →  target {target_seeds} seeds')
        print('=' * 72)
        run_benchmark(
            datasets=[dataset],
            learners=LEARNERS,
            n_runs=target_seeds,
            base_seed=BASE_SEED,
            results_csv=RESULTS_CSV,
            verbose=True,
        )


if __name__ == '__main__':
    main()
