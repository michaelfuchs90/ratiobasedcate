"""
LaTeX table generation for benchmark metrics.

A single function `make_metric_table` produces a per-dataset LaTeX table
with mean column and worst-case ratio column for any metric (Qini ratio,
Qini difference, CalError ratio, CalError difference).
"""

from typing import Optional

import numpy as np
import pandas as pd

from visualization import MetricSpec, METRIC_SPECS


# =============================================================================
# Default display orderings (mirror the notebook's existing order)
# =============================================================================
LEARNER_ORDER_DEFAULT = [
    'S', 'T', 'Q', 'Q-Simple',
    'DR-Q', 'DR-Q log', 'DR-Q-Simple', 'DR-Q-Simple log',
    'DR-T', 'DR-T log', 'DR-S', 'DR-S log',
    'X', 'X log', 'R', 'R log', 'DR-Diff',
]

# After how many learners to insert a horizontal rule
HLINE_AFTER = {
    'plug-vs-dr': 4,   # after the four plug-ins (S, T, Q, Q-Simple)
}

DATASET_SHORT_DEFAULT = {
    'hillstrom_visit': 'H(Vis)', 'hillstrom_conversion': 'H(Conv)',
    'criteo': 'Criteo', 'megafon': 'MegaFon', 'x5_retail': 'X5',
    'lenta': 'Lenta', 'twins': 'Twins',
    'rhc': 'RHC', 'cattaneo': 'Cattaneo', 'nhefs': 'NHEFS', 'jtpa': 'JTPA',
}


# =============================================================================
# Helpers
# =============================================================================
def _fmt_value(v: float, bold: bool, allow_negative: bool = True) -> str:
    """Format a numeric cell, optionally bold-faced for the per-column best."""
    if pd.isna(v):
        return '--'
    if v < 0 and allow_negative:
        s = f'$-${abs(v):.3f}'
    else:
        s = f'{v:.3f}'
    return f'\\textbf{{{s}}}' if bold else s


# =============================================================================
# Public API
# =============================================================================
def make_metric_table(
    df_results: pd.DataFrame,
    metric: str,
    datasets: list,
    caption: str,
    label: str,
    plot_learners: dict,
    *,
    learner_order: Optional[list] = None,
    dataset_short: Optional[dict] = None,
    hline_after: int = 4,
) -> str:
    """Build a per-dataset LaTeX table with summary columns.

    Layout:
        Learner | ds_1 | ds_2 | ... | ds_n | Mean | Worst
        --------+------+------+-----+------+------+-------
        S       | val  | val  | ... | val  | mean | worst
                | (se) | (se) | ... | (se) |      |
        T       | ...

    Parameters
    ----------
    df_results : pd.DataFrame
        Per-seed benchmark results.
    metric : str | MetricSpec
        Metric key (e.g. ``'qini_ratio'``) or a ``MetricSpec``.
    datasets : list of str
        Dataset keys to include as columns (in the desired order).
    caption, label : str
        LaTeX caption and label.
    plot_learners : dict[str, str]
        Mapping ``learner_key`` → display label.
    learner_order : list of str, optional
        Display labels in the order they should appear in the table.
        Labels not present in the data are skipped silently.
    dataset_short : dict, optional
        Mapping dataset key → short header label.
    hline_after : int
        Insert a horizontal rule after this many learners (default 4 to
        separate plug-ins from DR variants).

    Returns
    -------
    str : LaTeX source for the table.
    """
    spec = METRIC_SPECS[metric] if isinstance(metric, str) else metric
    learner_order = learner_order or LEARNER_ORDER_DEFAULT
    dataset_short = dataset_short or DATASET_SHORT_DEFAULT

    df = df_results[df_results['learner'].isin(plot_learners)].copy()
    df['label'] = df['learner'].map(plot_learners)
    sub = df[df['dataset'].isin(datasets)]

    means = sub.groupby(['label', 'dataset'])[spec.column].mean().unstack('dataset')
    sems = sub.groupby(['label', 'dataset'])[spec.column].sem().unstack('dataset')

    ds_order = [d for d in datasets if d in means.columns]
    ds_headers = ' & '.join(dataset_short.get(d, d) for d in ds_order)

    # Per-dataset best learner & per-dataset best value (for ratio summary)
    if spec.direction == 'max':
        best_per_ds = {d: means[d].idxmax() for d in ds_order}
        row_best = means.reindex(columns=ds_order).max(axis=0)
        best_mean_lname = means.reindex(columns=ds_order).mean(axis=1).idxmax()
        summary_header = r'$\bar{Q}$ & $R_{\min}$'
    else:
        best_per_ds = {d: means[d].idxmin() for d in ds_order}
        row_best = means.reindex(columns=ds_order).min(axis=0)
        best_mean_lname = means.reindex(columns=ds_order).mean(axis=1).idxmin()
        summary_header = r'$\overline{\mathrm{CE}}$ & $R_{\max}$'

    col_spec = 'l | ' + 'r' * len(ds_order) + ' | r r'

    lines = [
        r'\begin{table}[H]',
        r'\centering',
        r'\footnotesize',
        f'\\caption{{{caption}}}',
        f'\\label{{{label}}}',
        r'\setlength{\tabcolsep}{3pt}',
        f'\\begin{{tabular}}{{{col_spec}}}',
        r'\hline',
        f'Learner & {ds_headers} & {summary_header} \\\\',
        r'\hline',
    ]

    learners_present = [l for l in learner_order if l in means.index]

    for i, lname in enumerate(learners_present):
        if i == hline_after:
            lines.append(r'\hline')

        row_m = means.loc[lname].reindex(ds_order)
        row_s = sems.loc[lname].reindex(ds_order)
        mean_v = row_m.mean()

        # Ratio summary: per-dataset value / per-dataset best, then aggregate
        row_r = row_m / row_best
        if spec.direction == 'max':
            worst = row_r.min()
        else:
            worst = row_r.max()

        cell_vals = [_fmt_value(row_m[d], best_per_ds.get(d) == lname)
                     for d in ds_order]
        mean_s = _fmt_value(mean_v, lname == best_mean_lname)

        se_vals = [
            f'\\scriptsize({row_s[d]:.3f})' if not pd.isna(row_s[d])
            else '\\scriptsize(--)'
            for d in ds_order
        ]
        sp = '[2pt]' if i < len(learners_present) - 1 else ''

        worst_s = (
            f'{worst:.3f}' if not pd.isna(worst) and worst >= 0
            else f'$-${abs(worst):.3f}' if not pd.isna(worst)
            else '--'
        )

        lines.append(f'{lname}')
        lines.append(f' & {" & ".join(cell_vals)} & {mean_s} & {worst_s} \\\\')
        lines.append(f' & {" & ".join(se_vals)} & & \\\\{sp}')

    # Bottom row: dataset means across learners
    ds_means = means.reindex(columns=ds_order).mean(axis=0)
    ds_mean_vals = ' & '.join(
        f'{ds_means[d]:.3f}' if not pd.isna(ds_means[d]) else '--'
        for d in ds_order
    )
    lines += [
        r'\hline',
        f'\\textit{{Dataset mean}} & {ds_mean_vals} & & \\\\',
        r'\hline',
        r'\end{tabular}',
        r'\end{table}',
    ]

    return '\n'.join(lines)
