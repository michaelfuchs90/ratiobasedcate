# Ratio-CATE: Doubly Robust Meta-Learners for Ratio-Based Treatment Effects

Code and data accompanying the paper
**"Beyond Differences: Doubly Robust Meta-Learners for Ratio-Based Treatment Effects"**
(Fuchs, 2026).
<!-- AUTHOR-TODO: Replace XXXX.XXXXX with arXiv ID once assigned, then fill in the BibTeX in the Citation section below. -->

## Overview

Many decisions depend on the *relative* lift produced by a treatment rather
than the absolute difference — think "how many times more likely is this
customer to buy if we email them?" or "what is the relative mortality
reduction under this therapy?" When this is the question, the right
estimand is the **ratio-based Conditional Average Treatment Effect**:

```
τ(x) = E[Y | W = 1, X = x] / E[Y | W = 0, X = x].
```

This repository implements the methods proposed in the paper:

- The **Q-Learner**, which expresses `τ(x)` through an algebraic identity
  in terms of the propensity `e(x)` and the converter-propensity `p(x)`,
  reducing ratio-CATE estimation to two binary classification tasks.
- **Doubly robust extensions** (DR-T, DR-S, DR-Q, DR-Q-Simple and their
  log-scale variants) with differing robustness guarantees: classical
  double robustness for DR-T/DR-S, conditional robustness for DR-Q.
- A **rank-preserving log-linear calibration** procedure.
- Qini-coefficient ranking metric adapted to ratio effects, plus a
  bin-level log-absolute calibration error on the multiplicative scale.

Empirical highlights from the paper:

- On **seven RCT datasets**, no single learner dominates for ranking,
  but the **Q-Learner** is the most consistently competitive across
  datasets, with the S-Learner a close second on the mean but less
  stable. The S-Learner is the best-calibrated method.
- On **four observational datasets**, **DR-S log** emerges as the
  clear ranking winner, with DR-T log as a close second, confirming
  the importance of classical double robustness combined with
  log-scale variance stabilisation. DR-Q's conditional robustness
  proves insufficient (negative Qini on RHC and NHEFS).

## Installation

Tested with **Python 3.11**.

```bash
git clone https://github.com/michaelfuchs90/ratiobasedcate.git
cd ratiobasedcate
pip install -r requirements.txt
```

The dependencies are the standard PyData stack plus
[`scikit-uplift`](https://www.uplift-modeling.com/) (for the uplift-modeling
dataset loaders), [`lightgbm`](https://lightgbm.readthedocs.io/), and
`jupyter`. See `requirements.txt`.

## Repository structure

```
learner.py           17 learner implementations (S/T/Q/X/R and DR-* variants)
calibration.py       Rank-preserving log-linear calibration
metrics.py           Qini, uplift@k, calibration error
datasets.py          Loaders for 12 benchmark datasets
Benchmark.ipynb      End-to-end benchmark + figures + LaTeX table generation
Benchmark.html       Rendered snapshot of the notebook
benchmark_results.csv  Per-seed result table (11 datasets × 17 learners × 25 seeds)
benchmark_summary.csv  Aggregated summary metrics
data/raw/            Locally-stored CSVs for datasets not covered by sklift
data/processed/      Feather-cached per-seed outputs (regenerable; .gitignored)
paper.tex            Paper source
fig_qini_gap.png     Figure 1 of the paper
fig_cal_gap.png      Figure 2 of the paper
```

## Quick start

```python
from datasets import get_dataset
from learner import get_learner
from metrics import qini, calibration_error

# Load a dataset (returns an UpliftDataset with train/test splits)
data = get_dataset('hillstrom_visit', random_state=42)

# Instantiate and fit a Q-Learner
learner = get_learner('q', random_state=42)
learner.fit(data.X_train, data.W_train, data.Y_train)

# Predict ratio CATEs on the test set
tau = learner.predict(data.X_test)

# Evaluate
print(f"Qini  = {qini(data.Y_test, data.W_test, tau):.3f}")
print(f"CalEr = {calibration_error(data.Y_test, data.W_test, tau, n_bins=10):.3f}")
```

Available learners (see `learner.ALL_LEARNER`):
`s`, `t`, `q`, `q_simple`, `drs_log`, `drt_log`, `drq_log`,
`drs_direct`, `drt_direct`, `drq_direct`, `drq_simple_log`,
`drq_simple_direct`.

## Reproducing the paper

Run `Benchmark.ipynb` top-to-bottom. On a workstation the full sweep
(11 datasets × 17 learners × 25 seeds) takes several hours. The
notebook is idempotent: on restart it resumes from
`benchmark_results.csv` and only runs missing (dataset, learner, seed)
combinations.

After the sweep, the plotting cells regenerate

- `fig_qini_gap.png` — mean Qini ratio to best learner vs. worst-case
  Qini ratio across datasets (Figure 1 of the paper).
- `fig_cal_gap.png` — mean calibration-error ratio vs. worst-case
  calibration-error ratio across datasets (Figure 2 of the paper).

The final notebook cell emits the three LaTeX tables included in the
paper's appendix (Qini on RCT, Qini on observational, calibration).

## Datasets

Eleven datasets are included. Seven RCT, four observational. Twins is
semi-synthetic.

### Automatically downloaded via `scikit-uplift`

| Dataset            | Source                                        | Notes |
|--------------------|-----------------------------------------------|-------|
| Hillstrom (Visit)  | [Hillstrom 2008][hillstrom]                   | email marketing |
| Hillstrom (Conv.)  | [Hillstrom 2008][hillstrom]                   | same dataset, conversion outcome |
| Criteo             | [Diemert et al. 2018][diemert]                | display advertising |
| MegaFon            | [MegaFon 2019][megafon]                       | telecom uplift challenge |
| X5 Retail          | [X5 Retail 2020][x5]                          | retail personalisation challenge |
| Lenta              | [Lenta 2021][lenta]                           | retail uplift |

[hillstrom]: https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html
[diemert]: https://ailab.criteo.com/criteo-uplift-prediction-dataset/
[megafon]: https://ods.ai/competitions/megafon-df21-chegg
[x5]: https://ods.ai/competitions/x5-retailhero-uplift-modeling
[lenta]: https://www.uplift-modeling.com/en/latest/api/datasets/fetch_lenta.html

### Redistributed in `data/raw/`

| File                                     | Source                                 | Paper citation |
|------------------------------------------|----------------------------------------|----------------|
| `rhc.csv`                                | Right Heart Catheterization study      | [Connors et al. 1996][rhc] |
| `nhefs.csv`                              | NHANES-I Epidemiologic Follow-up Study | [Hernán & Robins 2020][hernan] |
| `jtpa.csv`                               | Job Training Partnership Act study     | [Abadie et al. 2002][abadie] |
| `cattaneo2.dta`                          | Birthweight study                      | [Cattaneo 2010][cattaneo] |
| `twin_pairs_*.csv`                       | Twins semi-synthetic benchmark         | [Louizos et al. 2017][louizos] |
| `lenta_dataset.csv.gz`, `clients.csv.gz`, `criteo.csv.gz`, `hillstorm_no_indices.csv.gz`, `megafon_dataset.csv.gz`, `purchases.csv.gz`, `uplift_train.csv.gz` | `scikit-uplift` caches (fetched automatically) | various (see sklift documentation) |

[rhc]: https://hbiostat.org/data/
[hernan]: https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/
[abadie]: https://economics.mit.edu/faculty/angrist/data1/data/abangim02
[cattaneo]: http://www.stata-press.com/data/r13/cattaneo2.dta
[louizos]: https://github.com/AMLab-Amsterdam/CEVAE

If you use any of these datasets, please cite the original source as
well as this paper.

## Citation

<!-- AUTHOR-TODO: Replace arXiv ID once assigned. -->

```bibtex
@article{fuchs2026ratiocate,
  title   = {Beyond Differences: Doubly Robust Meta-Learners for
             Ratio-Based Treatment Effects},
  author  = {Fuchs, Michael},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

## License

- **Code**: MIT License — see `LICENSE`.
- **Paper text and figures**: distributed on arXiv under arXiv's default
  non-exclusive licence for distribution.
- **Dataset files** in `data/raw/` retain the licences of their original
  sources. See the *Datasets* section above.

## Contact

Michael Fuchs — michael.fuchs@outlook.de
