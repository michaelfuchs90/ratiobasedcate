# Ratio-CATE: The Q-Learner

Code, data and results for the paper
**"Beyond Differences: Estimating and Tuning Ratio-Based Treatment Effects"** (Fuchs & Kreiss, 2026), [arXiv:2605.26288](https://arxiv.org/abs/2605.26288).

When a treatment effect is a ratio, the estimand is the ratio-based CATE

```
τ(x) = E[Y | W = 1, X = x] / E[Y | W = 0, X = x].
```

The **Q-Learner** writes τ(x) as a ratio of two odds and so reduces ratio-CATE
estimation for binary outcomes to two classification tasks: the treatment propensity among
converters, p(x) = P(W = 1 | X = x, Y = 1), and the propensity e(x). Tuning these classifiers
on log-loss tunes the effect estimate itself. The **Q-Simple-Learner** uses the known constant
propensity of randomized data. The repository also contains the S-, T-, X-, R- and AIPW-DR
learners, doubly robust DR-S/T/Q/Q-Simple learners in direct and log scale, the ratio Qini and
the calibration metrics, and the full benchmark (synthetic, randomized and observational data).

## Installation

Tested with Python 3.11.

```bash
git clone https://github.com/michaelfuchs90/ratiobasedcate.git
cd ratiobasedcate
pip install -r requirements.txt
```

## Quick start

```python
from datasets import get_dataset
from learner import get_learner
from metrics import qini_coefficient_ratio, calibration_ratio

data = get_dataset('H(Vis)_all', random_state=42)  # Hillstrom e-mail RCT, visit outcome

learner = get_learner('Q-Simple', random_state=42)  # known constant propensity (RCT)
learner.fit(data.X_train, data.W_train, data.Y_train, propensity=data.propensity_true_train)
tau = learner.predict_ratio_cate(data.X_test)       # ratio CATE per test unit

print(qini_coefficient_ratio(tau, data.W_test, data.Y_test))
print(calibration_ratio(tau, data.W_test, data.Y_test)['cal_binned'])
```

Learners (`learner.ALL_LEARNER`): `S`, `T`, `X`, `R`, `DR`, `Q`, `Q-Simple`, `DR-S`, `DR-T`,
`DR-Q`, `DR-Q-Simple` and the log-scale variants `DR-S log`, `DR-T log`, `DR-Q log`,
`DR-Q-Simple log`. `Q-Simple` needs a known constant propensity, `DR-Q-Simple` a known propensity. Difference
learners (`X`, `R`, `DR`) implement `predict_difference_cate` instead of `predict_ratio_cate`.

## Repository structure

```
learner.py          all learners
metrics.py          ratio Qini (oracle and count-based, optionally IPW-weighted), calibration
datasets.py         loaders: synthetic DGP, randomized and observational datasets
tune.py             Optuna tuning (CV log-loss or CV Qini) -> tuned_params.csv
benchmark.py        resume-aware benchmark loop -> results_{synthetic,synthconf,rct,obs}.csv
stest.py            variance factors of the corrected resampled t-test (stest_*.csv)
visualization.py    figures and tables of the paper
Benchmark.ipynb     runs everything above and writes every figure and table of paper.tex
paper.tex           paper source (fig_*.png, arxiv.sty)
data/raw/           redistributed raw data (see below)
```

## Reproducing the paper

Run `Benchmark.ipynb` top to bottom. With the shipped CSVs, tuning (section 3), benchmark and
significance diagnostic (section 4) find nothing left to do (they resume from the CSVs), and
sections 6-7 regenerate the six figures and write all tables into `paper.tex`. To recompute
from scratch, delete `tuned_params.csv`, `results_*.csv`, `stest_*.csv` and
`oracle_synthetic.csv`; tuning the synthetic datasets alone took about 90 hours.

## Datasets

Synthetic datasets (`synthetic_<rate>_<converters>`) bootstrap their covariates from Criteo;
`synthconf_<rate>_<converters>` is the confounded variant, with e(x) depending on the covariates
and no design propensity.
Randomized datasets: `<base>_12800` (12,800 training converters) or `<base>_all` (whole
training pool).

| Dataset | Type | Source | Obtained |
|---|---|---|---|
| Hillstrom (H(Vis), H(Conv)) | RCT | [Hillstrom 2008](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html) | `scikit-uplift` download |
| Criteo | RCT | [Diemert et al. 2018](https://ailab.criteo.com/criteo-uplift-prediction-dataset/) | `scikit-uplift` download |
| MegaFon | RCT | [MegaFon 2021](https://ods.ai/competitions/megafon-df21-chegg) | `scikit-uplift` download |
| X5 Retail | RCT | [X5 Retail 2020](https://ods.ai/competitions/x5-retailhero-uplift-modeling) | `scikit-uplift` download |
| Lenta | RCT | [Lenta 2020](https://www.uplift-modeling.com/en/latest/api/datasets/fetch_lenta.html) | manual: place [`lenta_dataset.csv.gz`](https://sklift.s3.eu-west-2.amazonaws.com/lenta_dataset.csv.gz) in `data/raw/` |
| Twins | semi-synthetic | [Louizos et al. 2017](https://github.com/AMLab-Amsterdam/CEVAE) | `data/raw/twin_pairs_*.csv` |
| Cattaneo | observational | [Cattaneo 2010](https://raw.githubusercontent.com/scunning1975/mixtape/master/cattaneo2.dta) | `data/raw/cattaneo2.dta` |
| NHEFS | observational | [Hernán & Robins 2020](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/) | `data/raw/nhefs.csv` |
| JTPA | observational | [Abadie et al. 2002](https://economics.mit.edu/faculty/angrist/data1/data/abangim02) | `data/raw/jtpa.csv` |
| RHC | observational | [Connors et al. 1996](https://hbiostat.org/data/) | `data/raw/rhc.csv` |

Loaded datasets are cached as feather files in `data/processed/`. If you use a dataset, please
cite its original source as well as this paper.

## Citation

```bibtex
@article{fuchs2026ratiocate,
  title   = {Beyond Differences: Estimating and Tuning Ratio-Based Treatment Effects},
  author  = {Fuchs, Michael and Krei{\ss}, Dominik},
  journal = {arXiv preprint arXiv:2605.26288},
  year    = {2026}
}
```

## License

Code: MIT (see `LICENSE`). Paper text and figures: arXiv's non-exclusive distribution licence.
Files in `data/raw/` keep the licences of their original sources.

## Contact

Michael Fuchs — michael.fuchs@outlook.de
