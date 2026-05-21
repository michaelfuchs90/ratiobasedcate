"""
LaTeX table generation for benchmark metrics.

Tables report raw per-dataset values without aggregate summary columns;
the paper deliberately defers cross-dataset aggregation to the heatmaps
in the body.
"""

from typing import List, Optional

import pandas as pd

from learner import ALL_LEARNER
from visualization import MetricLike, MetricSpec, METRIC_SPECS  # noqa: F401  (MetricSpec re-exported)


# Mirrors the display order of the learner factory.
LEARNER_ORDER_DEFAULT = list(ALL_LEARNER.keys())

# Horizontal rule after the four plug-ins (S, T, Q, Q-Simple).
HLINE_AFTER_DEFAULT = 4


def _fmt_value(v: float, bold: bool, allow_negative: bool = True) -> str:
    """Format a numeric cell, optionally bold-faced for the per-column best."""
    if pd.isna(v):
        return '--'
    if v < 0 and allow_negative:
        s = f'$-${abs(v):.3f}'
    else:
        s = f'{v:.3f}'
    return f'\\textbf{{{s}}}' if bold else s


def make_metric_table(
    df_results: pd.DataFrame,
    metric: MetricLike,
    datasets: List[str],
    caption: str,
    label: str,
    *,
    learner_order: Optional[List[str]] = None,
    hline_after: int = HLINE_AFTER_DEFAULT,
) -> str:
    """Build a per-dataset LaTeX table.

    Each cell shows mean ± SE per (learner, dataset). The per-column best
    is bold-faced; ``learner_order`` defaults to ``ALL_LEARNER`` order.
    ``metric`` may be a key (e.g. ``'qini_ratio'``) or a ``MetricSpec``.
    """
    spec = METRIC_SPECS[metric] if isinstance(metric, str) else metric
    learner_order = learner_order or LEARNER_ORDER_DEFAULT

    sub = df_results[df_results['dataset'].isin(datasets)]

    means = sub.groupby(['learner', 'dataset'])[spec.column].mean().unstack('dataset')
    sems = sub.groupby(['learner', 'dataset'])[spec.column].sem().unstack('dataset')

    ds_order = [d for d in datasets if d in means.columns]
    ds_headers = ' & '.join(ds_order)

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
