"""
Mean-vs-worst-case scatter plots for benchmark metrics.

A single function `plot_metric_panels` produces one figure with N side-by-side
panels, one per dataset group, for any metric column in the benchmark
results dataframe (Qini ratio, Qini difference, CalError ratio,
CalError difference, or any future metric).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    from adjustText import adjust_text  # noqa: F401  (kept for compatibility)
except ImportError:
    pass


# =============================================================================
# Plotting palette (mirrors the notebook's existing definitions)
# =============================================================================
LEARNER_FAMILY_DEFAULT = {
    'S': 'plug-in', 'T': 'plug-in',
    'Q': 'Q-family', 'Q-Simple': 'Q-family',
    'DR-Q': 'Q-family', 'DR-Q log': 'Q-family',
    'DR-Q-Simple': 'Q-family', 'DR-Q-Simple log': 'Q-family',
    'DR-T': 'DR-S/T', 'DR-T log': 'DR-S/T',
    'DR-S': 'DR-S/T', 'DR-S log': 'DR-S/T',
    'X': 'Difference methods', 'X log': 'Difference methods',
    'R': 'Difference methods', 'R log': 'Difference methods',
    'DR': 'Difference methods',
}

FAMILY_COLORS_DEFAULT = {
    'plug-in': '#4C72B0',
    'Q-family': '#DD8452',
    'DR-S/T': '#55A868',
    'Difference methods': '#C2E42B',
}

FAMILY_MARKERS_DEFAULT = {
    'plug-in': 'o',
    'Q-family': 's',
    'DR-S/T': 'D',
    'Difference methods': 'o',
}


# =============================================================================
# Metric specifications
# =============================================================================
@dataclass
class MetricSpec:
    """How to aggregate and present a metric.

    Attributes
    ----------
    column : str
        Column name in df_results.
    direction : {'max', 'min'}
        'max' → higher is better (Qini); ratio = value / per-dataset max.
        'min' → lower is better (CalError); ratio = value / per-dataset min.
    mean_label : str
        Y-axis label, e.g. "Mean Qini ratio to best (1 = best)".
    worst_label : str
        X-axis label, e.g. "Worst-case Qini ratio across datasets (1 = best)".
    crowd_threshold : float
        Distance below which scatter labels collapse into a numbered legend.
    """
    column: str
    direction: str  # 'max' or 'min'
    mean_label: str
    worst_label: str
    crowd_threshold: float = 0.19


# Preregistered specs for the four standard metrics
METRIC_SPECS = {
    'qini_ratio': MetricSpec(
        column='qini_ratio',
        direction='max',
        mean_label='Mean Qini ratio to best (1 = best)',
        worst_label='Worst-case Qini ratio across datasets (1 = best)',
        crowd_threshold=0.19,
    ),
    'qini_difference': MetricSpec(
        column='qini_difference',
        direction='max',
        mean_label='Mean Qini-difference ratio to best (1 = best)',
        worst_label='Worst-case Qini-difference ratio across datasets (1 = best)',
        crowd_threshold=0.19,
    ),
    'cal_error_ratio': MetricSpec(
        column='cal_error_ratio',
        direction='min',
        mean_label='Mean CalError ratio to best (1 = best)',
        worst_label='Worst-case CalError ratio across datasets (1 = best)',
        crowd_threshold=0.05,
    ),
    'cal_error_difference': MetricSpec(
        column='cal_error_difference',
        direction='min',
        mean_label='Mean CalError-diff ratio to best (1 = best)',
        worst_label='Worst-case CalError-diff ratio across datasets (1 = best)',
        crowd_threshold=0.05,
    ),
}


# =============================================================================
# Aggregation
# =============================================================================
def metric_ratio_matrix(df_results: pd.DataFrame, spec: MetricSpec,
                        plot_learners: dict) -> pd.DataFrame:
    """Mean metric per (label, dataset), then convert to per-dataset ratio.

    Returns a DataFrame indexed by display label with one column per dataset.
    Values are ratio-to-best on each dataset:
      direction='max' → value / per-dataset max  (≤ 1, 1 = best)
      direction='min' → value / per-dataset min  (≥ 1, 1 = best)
    """
    df = df_results[df_results['learner'].isin(plot_learners)].copy()
    df['label'] = df['learner'].map(plot_learners)

    matrix = df.groupby(['label', 'dataset'])[spec.column].mean().unstack('dataset')
    
    if spec.direction == 'max':
        #per_ds_best = matrix.max(axis=0)
        
        #S is the benchmark, since it its the classical 
        #ML model, without further effort. Any metalearner
        #must justify its existense by being better than
        #the no effort variant at least sometimes
        per_ds_best = matrix.loc["S"]
        
    else:
        #per_ds_best = matrix.min(axis=0)
        per_ds_best = matrix.loc["S"]

    return matrix.divide(per_ds_best, axis=1)


def learner_summary(ratio_matrix: pd.DataFrame, datasets: list,
                    direction: str) -> pd.DataFrame:
    """Mean across the requested datasets and worst-case ratio.

    For direction='max' (Qini-style), worst = min across datasets.
    For direction='min' (CalError-style), worst = max across datasets.
    """
    sub = ratio_matrix.reindex(columns=datasets).dropna(axis=1, how='all')
    if direction == 'max':
        worst_vals = sub.min(axis=1)
        best_vals = sub.max(axis=0)
    else:
        worst_vals = sub.max(axis=1)
        best_vals = sub.min(axis=0)
        

    # Every learner should count roughly the same, therefore use 1/best value
    # as weight 
    weights = 1/best_vals.reindex(sub.columns)
    weighted_mean = sub.mul(weights, axis=1).sum(axis=1) / weights.sum()


    #return pd.DataFrame({'mean': sub.mean(axis=1), 'worst': worst_vals})
    return pd.DataFrame({'mean': weighted_mean, 'worst': worst_vals})


# =============================================================================
# Scatter panel
# =============================================================================
def _scatter_panel(ax, summary, title, xlabel, ylabel,
                   learner_family, family_colors, family_markers,
                   crowd_threshold):
    """One scatter panel: worst-case (x) vs mean (y), with crowded labels
    collapsed into a numbered legend on the right.
    """
    if summary.empty:
        ax.set_title(title)
        ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                ha='center', va='center', fontsize=11, color='gray')
        return

    coords = summary[['worst', 'mean']].to_numpy()
    xrng = max(coords[:, 0].max() - coords[:, 0].min(), 1e-9)
    yrng = max(coords[:, 1].max() - coords[:, 1].min(), 1e-9)
    norm = coords / np.array([xrng, yrng])
    d = np.linalg.norm(norm[:, None, :] - norm[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    crowded = d.min(axis=1) < crowd_threshold

    legend_lines, n = [], 0
    for i, (label, row) in enumerate(summary.iterrows()):
        fam = learner_family.get(label, 'plug-in')
        ax.scatter(row['worst'], row['mean'],
                   c=family_colors[fam], marker=family_markers[fam],
                   s=140, zorder=5, edgecolors='black', linewidth=0.5)
        if crowded[i]:
            n += 1
            ax.annotate(str(n), (row['worst'], row['mean']),
                        color='white', fontsize=7, fontweight='bold',
                        ha='center', va='center', zorder=6)
            legend_lines.append(f"{n}: {label}")
        else:
            ax.annotate(label, (row['worst'], row['mean']),
                        textcoords='offset points', xytext=(7, 0),
                        fontsize=9, ha='left', va='center', zorder=6)

    if legend_lines:
        ax.text(1.02, 0.5, '\n'.join(legend_lines),
                transform=ax.transAxes, fontsize=8, va='center',
                family='monospace',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', lw=0.5))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    # Ensure x-axis left limit is at most -1
    x_left, x_right = ax.get_xlim()
    ax.set_xlim(left=max(x_left, -0.5), right=x_right)


# =============================================================================
# Public API
# =============================================================================
def plot_metric_panels(
    df_results: pd.DataFrame,
    metric: str,
    panels: list,
    plot_learners: dict,
    *,
    learner_family: Optional[dict] = None,
    family_colors: Optional[dict] = None,
    family_markers: Optional[dict] = None,
    figsize: tuple = (12, 5),
    savepath: Optional[str] = None,
    dpi: int = 300,
    show: bool = True,
):
    """One figure with N side-by-side scatter panels for the given metric.

    Parameters
    ----------
    df_results : pd.DataFrame
        Per-seed benchmark results. Must contain columns:
        ``learner``, ``dataset``, plus the metric column.
    metric : str | MetricSpec
        Either a key into ``METRIC_SPECS`` (e.g. ``'qini_ratio'``) or a
        ``MetricSpec`` instance.
    panels : list of (title, list_of_datasets)
        One panel per entry. Use 2 entries for the standard RCT/Obs split,
        or any other grouping (e.g. low-conv vs high-conv RCTs).
    plot_learners : dict[str, str]
        Mapping ``learner_key`` → display label.
    learner_family, family_colors, family_markers : dict, optional
        Override the default palette.
    figsize, savepath, dpi, show : usual matplotlib options.

    Returns
    -------
    fig : matplotlib.figure.Figure
    summaries : list of pd.DataFrame
        Per-panel summary (mean, worst) tables, in the same order as
        ``panels``.
    """
    spec = METRIC_SPECS[metric] if isinstance(metric, str) else metric
    learner_family = learner_family or LEARNER_FAMILY_DEFAULT
    family_colors = family_colors or FAMILY_COLORS_DEFAULT
    family_markers = family_markers or FAMILY_MARKERS_DEFAULT

    ratio_matrix = metric_ratio_matrix(df_results, spec, plot_learners)

    fig, axes = plt.subplots(1, len(panels), figsize=figsize, squeeze=False)
    axes = axes[0]

    summaries = []
    for ax, (title, ds_list) in zip(axes, panels):
        summary = learner_summary(ratio_matrix, ds_list, spec.direction)
        summaries.append(summary)
        _scatter_panel(
            ax, summary, title,
            xlabel=spec.worst_label, ylabel=spec.mean_label,
            learner_family=learner_family,
            family_colors=family_colors,
            family_markers=family_markers,
            crowd_threshold=spec.crowd_threshold,
        )
        ax.axhline(1, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
        ax.axvline(1, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)

    # Family legend at the bottom
    legend_elements = [
        Line2D([0], [0], marker=family_markers[f], color='w',
               markerfacecolor=family_colors[f], markersize=10,
               markeredgecolor='black', markeredgewidth=0.5, label=f)
        for f in family_colors
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=len(family_colors), fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.05, 1, 1])

    if savepath:
        plt.savefig(savepath, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()

    return fig, summaries
