# Ratio-CATE: Doubly Robust Meta-Learners for Ratio-Based Treatment Effects

Code and data accompanying the paper
**"Beyond Differences: Doubly Robust Meta-Learners for Ratio-Based Treatment Effects"**
(Fuchs & Kreiß, 2026).
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

- The **Q-Learner**, which expresses `τ(x)` through a Bayes-rule
  decomposition into a product of two odds ratios, reducing ratio-CATE
  estimation for binary outcomes to two binary classification tasks.
  As a special case, the **Q-Simple Learner** uses only the converter
  subset and is applicable when the propensity `e(x)` is known.
- **Doubly robust extensions** of the S-, T-, Q-, and Q-Simple
  Learners, in both direct- and log-scale variants, with differing
  robustness guarantees: classical double robustness for DR-S/DR-T,
  conditional robustness for DR-Q.
- The Qini coefficient adapted to ratio effects, and a bin-level
  log-absolute calibration error on the multiplicative scale.

Empirical highlights from the paper:

- **RCTs split by conversion rate.** On the three high-conversion
  RCTs (X5, MegaFon, Criteo), the plain S-Learner sits within a hair
  of the dataset winner — no causal meta-learner is needed. On the
  four low-conversion RCTs (H(Conv), Twins, Criteo, Lenta), the
  **Q-Learner** is the most consistent choice, significantly beating
  the S-Learner on three of four datasets. No DR variant is
  consistently ahead of the leading plug-in on RCT data.
- **Observational data: DR earns its keep.** On the four
  observational datasets (Cattaneo, NHEFS, JTPA, RHC), a DR variant
  introduced here takes the top spot on three of four; on RHC the
  R-Learner edges them out. Among DR variants, log-scale dominates at
  low conversion rates (Cattaneo, NHEFS) and direct-scale becomes
  competitive at higher conversion rates (JTPA, RHC).
- **Calibration.** The S-Learner is the best- or near-best-calibrated
  method on every dataset; direct-scale DR variants are persistently
  miscalibrated by factors of 2–10×.

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
learner.py             15 learner implementations (S/T/Q/X/R/DR and DR-S/T/Q/Q-Simple in direct + log variants)
calibration.py         Rank-preserving log-linear calibration
metrics.py             Qini and multiplicative calibration error (ratio + difference scales)
datasets.py            Loaders for 11 benchmark datasets
benchmark.py           Per-seed benchmark loop with idempotent CSV checkpointing
visualization.py       Heatmaps and the regime-specific scatter / dot-cloud figures
tables.py              LaTeX table generation
Benchmark.ipynb        End-to-end driver: benchmark + figures + LaTeX tables
Benchmark.html         Rendered snapshot of the notebook
benchmark_results.csv  Per-seed result table (11 datasets × 15 learners × ≥50 seeds)
data/raw/              Locally-stored CSVs for datasets not covered by sklift
data/processed/        Feather-cached per-seed outputs (regenerable; .gitignored)
paper.tex              Paper source
fig_rct_dots_with_lines.png  RCT figure (Q vs. Q-Simple by conversion rate)
fig_obs_new_vs_rest.png      Observational figure (best DR vs. best non-DR)
fig_heatmap_qini_*.png       Per-(learner, dataset) Qini heatmaps (appendix)
fig_heatmap_cal_*.png        Per-(learner, dataset) calibration heatmaps (appendix)
```

## Quick start

```python
from datasets import get_dataset
from learner import get_learner
from metrics import qini_coefficient_ratio, calibration_error_ratio

# Load a dataset (returns an UpliftDataset with train/test splits)
data = get_dataset('H(Conv)', random_state=42)

# Instantiate and fit a Q-Learner
learner = get_learner('Q', random_state=42)
learner.fit(data.X_train, data.W_train, data.Y_train)

# Predict ratio CATEs on the test set
tau = learner.predict(data.X_test)

# Evaluate
print(f"Qini    = {qini_coefficient_ratio(tau, data.W_test, data.Y_test):.3f}")
print(f"CalEr   = {calibration_error_ratio(tau, data.W_test, data.Y_test):.3f}")
```

Available learners (keys of `learner.ALL_LEARNER`):
`S`, `T`, `X`, `R`, `DR`, `Q`, `Q-Simple`,
`DR-S`, `DR-T`, `DR-Q`, `DR-Q-Simple`,
`DR-S log`, `DR-T log`, `DR-Q log`, `DR-Q-Simple log`.

Available datasets (keys of `datasets.ALL_DATASETS`): seven RCT
(`H(Conv)`, `Twins`, `Criteo`, `Lenta`, `H(Vis)`, `MegaFon`, `X5`)
and four observational (`Cattaneo`, `NHEFS`, `JTPA`, `RHC`).

## Reproducing the paper

Run `Benchmark.ipynb` top-to-bottom. The full sweep (11 datasets × 15
learners × 50–500 seeds, with extra seeds added on noisier datasets
to reach paired-test significance) takes many hours on a workstation.
The notebook is idempotent: on restart it resumes from
`benchmark_results.csv` and only runs missing (dataset, learner,
seed) combinations.

After the sweep, the plotting cells regenerate

- `fig_rct_dots_with_lines.png` — Q-Learner / Q-Simple / S-baseline
  ratios across the seven RCT datasets, sorted by conversion rate
  (Figure 1 of the paper).
- `fig_obs_new_vs_rest.png` — best DR variant vs. best non-DR
  competitor on the four observational datasets, with R-Learner
  overlay (Figure 2 of the paper).
- Per-(learner, dataset) heatmaps for Qini and calibration error
  (appendix figures).

The final notebook cell emits the LaTeX tables included in the
paper's appendix (per-dataset Qini and calibration error on RCT and
observational data).

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
  author  = {Fuchs, Michael and Krei{\ss}, Dominik},
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
