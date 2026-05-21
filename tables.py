"""
LaTeX table generation for benchmark metrics.

`make_metric_table` produces a per-dataset LaTeX table for any metric
(Qini ratio, Qini difference, CalError ratio, CalError difference).

The body of the paper measures learner performance via the per-dataset
ratio R_l(d) = Q_l(d) / Q_S(d) (S-Learner as baseline) and refrains
from cross-dataset aggregation. These tables therefore report raw
per-dataset values without aggregate summary columns; aggregation is
deliberately deferred to the heatmaps in the body.

Display labels for learners and datasets are stored directly in
``df_results``; this module no longer translates internal keys to
display names.
"""

from typing import Optional

import pandas as pd

from learner import ALL_LEARNER
from visualization import MetricSpec, METRIC_SPECS


# =============================================================================
# Default display ordering for the learner column
# =============================================================================
# The factory in learner.py is already in display order; mirror it here.
LEARNER_ORDER_DEFAULT = list(ALL_LEARNER.keys())

# After how many learners to insert a horizontal rule
# (after the four plug-ins: S, T, Q, Q-Simple).
HLINE_AFTER_DEFAULT = 4


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
    metric,
    datasets: list,
    caption: str,
    label: str,
    *,
    learner_order: Optional[list] = None,
    hline_after: int = HLINE_AFTER_DEFAULT,
) -> str:
    """Build a per-dataset LaTeX table.

    Layout:
        Learner | ds_1 | ds_2 | ... | ds_n
        --------+------+------+-----+------
        S       | val  | val  | ... | val
                | (se) | (se) | ... | (se)
        T       | ...

    Parameters
    ----------
    df_results : pd.DataFrame
        Per-seed benchmark results, with ``learner`` and ``dataset``
        columns already containing display labels.
    metric : str | MetricSpec
        Metric key (e.g. ``'qini_ratio'``) or a ``MetricSpec``.
    datasets : list of str
        Display labels of datasets to include as columns, in display order.
    caption, label : str
        LaTeX caption and label.
    learner_order : list of str, optional
        Display labels in the order they should appear. Labels not present
        in the data are skipped silently. Defaults to the order in
        ``learner.ALL_LEARNER``.
    hline_after : int
        Insert a horizontal rule after this many learners.

    Returns
    -------
    str : LaTeX source for the table.
    """
    spec = METRIC_SPECS[metric] if isinstance(metric, str) else metric
    learner_order = learner_order or LEARNER_ORDER_DEFAULT

    sub = df_results[df_results['dataset'].isin(datasets)]

    means = sub.groupby(['learner', 'dataset'])[spec.column].mean().unstack('dataset')
    sems = sub.groupby(['learner', 'dataset'])[spec.column].sem().unstack('dataset')

    ds_order = [d for d in datasets if d in means.columns]
    ds_headers = ' & '.join(ds_order)

    # Per-dataset best learner (for column-wise bolding)
    if spec.direction == 'max':
        best_per_ds = {d: means[d].idxmax() for d in ds_order}
    else:
        best_per_ds = {d: means[d].idxmin() for d in ds_order}

    col_spec = 'l | ' + 'r' * len(ds_order)

    lines = [
        r'\begin{table}[H]',
        r'\centering',
        r'\footnotesize',
        f'\\caption{{{caption}}}',
        f'\\label{{{label}}}',
        r'\setlength{\tabcolsep}{3pt}',
        f'\\begin{{tabular}}{{{col_spec}}}',
        r'\hline',
        f'Learner & {ds_headers} \\\\',
        r'\hline',
    ]

    learners_present = [l for l in learner_order if l in means.index]

    for i, lname in enumerate(learners_present):
        if i == hline_after:
            lines.append(r'\hline')

        row_m = means.loc[lname].reindex(ds_order)
        row_s = sems.loc[lname].reindex(ds_order)

        cell_vals = [_fmt_value(row_m[d], best_per_ds.get(d) == lname)
                     for d in ds_order]
        se_vals = [
            f'\\scriptsize({row_s[d]:.3f})' if not pd.isna(row_s[d])
            else '\\scriptsize(--)'
            for d in ds_order
        ]
        sp = '[2pt]' if i < len(learners_present) - 1 else ''

        lines.append(f'{lname}')
        lines.append(f' & {" & ".join(cell_vals)} \\\\')
        lines.append(f' & {" & ".join(se_vals)} \\\\{sp}')

    # Bottom row: per-dataset mean across learners (not a cross-dataset aggregate).
    ds_means = means.reindex(columns=ds_order).mean(axis=0)
    ds_mean_vals = ' & '.join(
        f'{ds_means[d]:.3f}' if not pd.isna(ds_means[d]) else '--'
        for d in ds_order
    )
    lines += [
        r'\hline',
        f'\\textit{{Dataset mean}} & {ds_mean_vals} \\\\',
        r'\hline',
        r'\end{tabular}',
        r'\end{table}',
    ]

    return '\n'.join(lines)
