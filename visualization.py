"""
Visualizations for benchmark results.

Per-(learner, dataset) statistics are computed against the S-Learner as
the natural ML baseline:

    R_l(d) = Q_l(d) / Q_S(d)         for direction='max' (Qini)
    R_l(d) = E_l(d) / E_S(d)         for direction='min' (CalError)

Reading: R_l > 1 means better than the standard ML baseline (for Qini)
or worse (for CalError); R_l < 1 means the opposite.

Public API
----------
- ``MetricSpec`` / ``METRIC_SPECS``: how each benchmark column is presented.
- ``plot_heatmap``: per-(learner, dataset) heatmap colored by R_l, with
  cells flagged as "equal to S" (within statistical tolerance) shown
  neutrally.
"""

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


# =============================================================================
# Metric specifications
# =============================================================================
@dataclass
class MetricSpec:
    """How a metric is aggregated and presented.

    Attributes
    ----------
    column : str
        Column name in df_results.
    direction : {'max', 'min'}
        'max' → higher is better (Qini). R_l > 1 means better than S.
        'min' → lower is better (CalError). R_l < 1 means better than S.
    pretty_name : str
        Short, paper-ready name for figure titles and table headers.
    """
    column: str
    direction: str
    pretty_name: str


METRIC_SPECS = {
    'qini_ratio': MetricSpec(
        column='qini_ratio',
        direction='max',
        pretty_name='Qini (ratio CATE)',
    ),
    'qini_difference': MetricSpec(
        column='qini_difference',
        direction='max',
        pretty_name='Qini (difference CATE)',
    ),
    'cal_error_ratio': MetricSpec(
        column='cal_error_ratio',
        direction='min',
        pretty_name='CalError (ratio CATE)',
    ),
    'cal_error_difference': MetricSpec(
        column='cal_error_difference',
        direction='min',
        pretty_name='CalError (difference CATE)',
    ),
}


# =============================================================================
# Paired equality test against the baseline (shared with benchmark.py)
# =============================================================================
def paired_equality_matrix(
    df_results: pd.DataFrame,
    metric,
    *,
    baseline: str,
    learners: Iterable[str],
    datasets: Iterable[str],
    z: float = 1.96,
) -> pd.DataFrame:
    """For each (learner, dataset), test whether the per-seed paired
    difference ``delta(seed) = metric_learner(seed) - metric_baseline(seed)``
    is statistically distinguishable from zero.

    Returns a boolean DataFrame indexed by ``learners`` with columns
    ``datasets``: True where the learner is statistically equivalent to
    the baseline (paired-mean difference within ``z * SE``), False where
    it differs significantly.

    The pairing removes seed-level common variance shared by all
    learners on the same train/test split.
    """
    spec = METRIC_SPECS[metric] if isinstance(metric, str) else metric
    learners = list(learners)
    datasets = list(datasets)

    per_seed = df_results.pivot_table(
        index=['dataset', 'seed'], columns='learner', values=spec.column
    )
    if baseline not in per_seed.columns:
        raise KeyError(f"Baseline learner {baseline!r} missing from per-seed pivot.")
    deltas = per_seed.subtract(per_seed[baseline], axis=0)

    mean_delta = (
        deltas.groupby(level='dataset').mean()
        .T.reindex(index=learners, columns=datasets)
    )
    sd_delta = (
        deltas.groupby(level='dataset').std()
        .T.reindex(index=learners, columns=datasets)
    )
    n_paired = (
        deltas.groupby(level='dataset').count()
        .T.reindex(index=learners, columns=datasets)
    )
    se_delta = sd_delta / np.sqrt(n_paired)

    equal = mean_delta.abs() <= z * se_delta
    if baseline in equal.index:
        equal.loc[baseline] = True
    return equal


def insignificant_datasets(
    df_results: pd.DataFrame,
    metric,
    *,
    baseline: str,
    learners: Iterable[str],
    datasets: Iterable[str],
    z: float = 1.96,
    min_grey_cells: int = 1,
) -> list:
    """Datasets that still have at least ``min_grey_cells`` learners
    statistically indistinguishable from the baseline (paired test).

    Useful as a "do we need more seeds?" check.
    """
    eq = paired_equality_matrix(
        df_results, metric, baseline=baseline,
        learners=learners, datasets=datasets, z=z,
    )
    # Drop the baseline row (always equal by construction).
    eq_no_base = eq.drop(index=baseline, errors='ignore')
    grey_per_ds = eq_no_base.sum(axis=0)
    return [d for d in datasets if grey_per_ds.get(d, 0) >= min_grey_cells]


# =============================================================================
# Heatmap
# =============================================================================
def plot_heatmap(
    df_results: pd.DataFrame,
    metric,
    datasets: Iterable[str],
    *,
    learner_order: Iterable[str],
    conv_rates: dict,
    baseline: str = 'S',
    z: float = 1.96,
    color_clip: float = 2.0,
    figsize: Optional[tuple] = None,
    title: Optional[str] = None,
    savepath: Optional[str] = None,
    dpi: int = 300,
    show: bool = True,
):
    """Per-(learner, dataset) heatmap of metric ratios against ``baseline``.

    Cells colored on a log-ratio diverging scale:
      - **green** if learner outperforms the baseline,
      - **red** if it underperforms,
      - **neutral grey** if the difference is statistically inconclusive
        (means within ``z`` combined SEs).
    Cells where the ratio breaks down (either side ≤ 0) are marked grey
    with the raw value labelled "n/a".

    Parameters
    ----------
    df_results : pd.DataFrame
        Per-seed benchmark results. Must contain ``learner``, ``dataset``,
        ``seed`` columns and the metric column.
    metric : str | MetricSpec
        Metric key or instance.
    datasets : iterable of str
        Display labels of datasets to include.
    learner_order : iterable of str
        Display labels of learners, in the desired y-axis order
        (top-to-bottom).
    conv_rates : dict[str, float]
        Conversion rate per dataset; used to sort the x-axis ascending.
    baseline : str
        Display label of the baseline learner. Defaults to ``'S'``.
    z : float
        Z-score for the "equal to S" tolerance band (default 1.96 ≈ 95%).
    color_clip : float
        Color saturation point on the log-ratio scale: cells with
        ``|log2(R)| >= color_clip`` saturate. Default 2.0 means
        ratios ≥ 4× / ≤ 0.25× saturate.
    figsize : (w, h), optional
        Matplotlib figure size. Auto-sized when omitted.
    title : str, optional
        Figure title.
    savepath : str, optional
        If given, save to this path.
    dpi : int
        DPI for the saved figure.
    show : bool
        Whether to call ``plt.show()``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ratio_matrix : pd.DataFrame
        Per-(learner, dataset) ratio R_l(d).
    """
    spec = METRIC_SPECS[metric] if isinstance(metric, str) else metric
    datasets = list(datasets)
    learner_order = [l for l in learner_order if l != baseline]
    learner_order = [baseline] + learner_order  # baseline always first row

    # x-axis: datasets sorted by ascending conversion rate
    ds_sorted = sorted(datasets, key=lambda d: conv_rates[d])

    sub = df_results[df_results['dataset'].isin(ds_sorted)
                     & df_results['learner'].isin(learner_order)].copy()

    means = sub.groupby(['learner', 'dataset'])[spec.column].mean().unstack('dataset')

    learners_present = [l for l in learner_order if l in means.index]
    means = means.reindex(index=learners_present, columns=ds_sorted)

    if baseline not in means.index:
        raise KeyError(f"Baseline learner {baseline!r} missing from results.")

    base_mean = means.loc[baseline]

    # Ratios (defined when both means same sign and baseline non-zero)
    ratio = means.divide(base_mean, axis=1)

    # Paired equality test against baseline (per-seed differences).
    equal_to_baseline = paired_equality_matrix(
        sub, metric=spec, baseline=baseline,
        learners=learners_present, datasets=ds_sorted, z=z,
    )

    # Cell-state classification:
    #   - "learner_negative": Q_l < 0 (worse than random) -- paint deep red,
    #     show raw negative Qini value.
    #   - "ratio_undef": baseline non-positive or either side NaN -- the
    #     ratio comparison itself is meaningless; paint neutral grey "n/a".
    learner_negative = means.lt(0) & base_mean.gt(0) & means.notna()
    ratio_undef = (
        base_mean.le(0)
        | means.isna()
        | base_mean.isna()
    ) & ~learner_negative

    # Color matrix: log2(ratio), with "equal", "negative" and "undefined"
    # cells excluded; they get explicit overlays below.
    safe_ratio = ratio.where(~ratio_undef & ~learner_negative)
    log_ratio = np.log2(safe_ratio)
    if spec.direction == 'min':
        log_ratio = -log_ratio  # invert so green = better in both cases

    color_data = log_ratio.where(~equal_to_baseline)

    # ---- Render -------------------------------------------------------------
    n_rows, n_cols = means.shape
    if figsize is None:
        figsize = (1.0 + 0.85 * n_cols, 0.6 + 0.42 * n_rows)

    fig, ax = plt.subplots(figsize=figsize)

    cmap = plt.get_cmap('RdYlGn')
    norm = TwoSlopeNorm(vmin=-color_clip, vcenter=0.0, vmax=color_clip)

    # Background colors
    bg = color_data.to_numpy(dtype=float)
    img = ax.imshow(bg, aspect='auto', cmap=cmap, norm=norm,
                    interpolation='nearest')

    # Overlay colors for special cells:
    #   neutral grey  → "equal to baseline"
    #   medium grey   → ratio undefined (baseline broken)
    #   deep red      → learner went negative (worse than random)
    NEUTRAL_RGB = (0.92, 0.92, 0.92)
    UNDEF_RGB = (0.78, 0.78, 0.78)
    DEEP_RED_RGB = (0.42, 0.0, 0.10)  # darker than RdYlGn's red endpoint
    for i in range(n_rows):
        for j in range(n_cols):
            if learner_negative.iat[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           facecolor=DEEP_RED_RGB,
                                           edgecolor='white', linewidth=0.5,
                                           zorder=1))
            elif ratio_undef.iat[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           facecolor=UNDEF_RGB,
                                           edgecolor='white', linewidth=0.5,
                                           zorder=1))
            elif equal_to_baseline.iat[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           facecolor=NEUTRAL_RGB,
                                           edgecolor='white', linewidth=0.5,
                                           zorder=1))

    # Cell text:
    #   - learner_negative: raw negative Qini value (white text on deep red)
    #   - ratio_undef:      "n/a" (black text on grey)
    #   - otherwise:        Q_l / Q_S, color picked for contrast
    for i, lname in enumerate(learners_present):
        for j, ds in enumerate(ds_sorted):
            if learner_negative.iat[i, j]:
                #txt = f'{means.iat[i, j]:+.3f}'
                txt = 'Qini < 0'
                color = 'white'
            elif ratio_undef.iat[i, j]:
                txt = 'n/a'
                color = 'black'
            else:
                txt = f'{ratio.iat[i, j]:.2f}'
                color = _text_color_for(color_data.iat[i, j], color_clip,
                                        equal_to_baseline.iat[i, j])
            ax.text(j, i, txt, ha='center', va='center',
                    fontsize=8, color=color, zorder=3)

    # Axis ticks: bottom = dataset (conv rate), left = learner
    xticklabels = [f'{d}\n({conv_rates[d] * 100:.1f}%)' for d in ds_sorted]
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(xticklabels, fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(learners_present, fontsize=9)

    # Highlight the baseline row with a thicker frame
    base_idx = learners_present.index(baseline)
    ax.add_patch(plt.Rectangle((-0.5, base_idx - 0.5), n_cols, 1,
                               fill=False, edgecolor='black', linewidth=1.5,
                               zorder=4))

    ax.set_xlabel(f'Dataset (sorted by conversion rate)', fontsize=10)
    ax.set_ylabel('Learner', fontsize=10)
    if title is None:
        title = f'{spec.pretty_name}: ratio to {baseline}-Learner'
    ax.set_title(title, fontsize=11)

    # Colorbar
    cbar = fig.colorbar(img, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(
        ('better than S  ←  log₂(R)  →  worse than S'
         if spec.direction == 'min'
         else 'worse than S  ←  log₂(R)  →  better than S'),
        fontsize=9,
    )
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()

    return fig, ratio


# =============================================================================
# Paired direct-vs-log slope plot
# =============================================================================
def plot_log_vs_direct(
    df_results: pd.DataFrame,
    metric,
    datasets: Iterable[str],
    *,
    conv_rates: dict,
    baseline: str = 'S',
    families: Optional[list] = None,
    extras: Optional[list] = None,
    yscale: str = 'log',
    figsize: Optional[tuple] = None,
    title: Optional[str] = None,
    savepath: Optional[str] = None,
    dpi: int = 300,
    show: bool = True,
):
    """Paired direct-vs-log slope plot of R_l(d) across datasets.

    For each DR family in ``families`` (default ['DR-S', 'DR-T', 'DR-Q']),
    draws two lines connecting the family's mean R_l(d) value across
    ``datasets`` sorted by ascending conversion rate:

        solid line  : direct-scale variant       (e.g. 'DR-S')
        dashed line : log-scale variant          (e.g. 'DR-S log')

    Crossings between the solid and dashed line of a family visualise
    the regime switch: at low conversion rates the log line typically
    sits above the direct line; at high conversion rates the order
    flips.

    Learners listed in ``extras`` (default ['R']) are drawn as dotted
    lines in distinguishable colors.

    Non-positive R values (learner Qini ≤ 0 or baseline ≤ 0) are masked
    to NaN so the line breaks cleanly; consult the heatmap for those
    cells.

    Parameters
    ----------
    df_results : pd.DataFrame
        Per-seed benchmark results.
    metric : str | MetricSpec
        Metric key (typically ``'qini_ratio'``) or a ``MetricSpec``.
    datasets : iterable of str
        Display labels of datasets to include.
    conv_rates : dict[str, float]
        Conversion rate per dataset; used to sort the x-axis.
    baseline : str
        Display label of the baseline learner (default ``'S'``).
    families : list of str, optional
        DR family stems; each must have a direct variant ``stem`` and
        a log variant ``stem + ' log'`` in the data.
    extras : list of str, optional
        Additional learners to overlay as dotted lines.
    yscale : {'log', 'linear'}
        Y-axis scale; ``'log'`` (default) keeps both the low-CVR
        regime switch (R ≈ 1–3) and outliers (R ≈ 100+) legible.
    figsize, title, savepath, dpi, show
        Standard plotting controls.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ratio : pd.DataFrame
        Per-(learner, dataset) ratio used for plotting.
    """
    spec = METRIC_SPECS[metric] if isinstance(metric, str) else metric
    if families is None:
        families = ['DR-S', 'DR-T', 'DR-Q']
    if extras is None:
        extras = ['R']

    datasets = list(datasets)
    ds_sorted = sorted(datasets, key=lambda d: conv_rates[d])

    relevant = {baseline}
    for fam in families:
        relevant.update([fam, f'{fam} log'])
    relevant.update(extras)

    sub = df_results[df_results['dataset'].isin(ds_sorted)
                     & df_results['learner'].isin(relevant)]
    means = (sub.groupby(['learner', 'dataset'])[spec.column]
             .mean().unstack('dataset'))

    if baseline not in means.index:
        raise KeyError(f"Baseline learner {baseline!r} missing from results.")

    base_mean = means.loc[baseline].reindex(ds_sorted)
    ratio = means.divide(base_mean, axis=1).reindex(columns=ds_sorted)

    # Mask non-positive ratios for log scale (negative learner Qini etc.).
    plot_ratio = ratio.where(ratio > 0)

    # ---- Render -------------------------------------------------------------
    if figsize is None:
        figsize = (1.5 + 1.1 * len(ds_sorted), 4.6)
    fig, ax = plt.subplots(figsize=figsize)

    family_colors = plt.get_cmap('tab10').colors
    x = np.arange(len(ds_sorted))

    for fi, fam in enumerate(families):
        c = family_colors[fi]
        for variant, ls, marker in [(fam, '-', 'o'),
                                    (f'{fam} log', '--', 's')]:
            if variant not in plot_ratio.index:
                continue
            y = plot_ratio.loc[variant].to_numpy()
            ax.plot(x, y, color=c, linestyle=ls, marker=marker,
                    markersize=6, linewidth=1.6, label=variant)

    for ei, name in enumerate(extras):
        if name not in plot_ratio.index:
            continue
        c = family_colors[len(families) + ei]
        y = plot_ratio.loc[name].to_numpy()
        ax.plot(x, y, color=c, linestyle=':', marker='^',
                markersize=7, linewidth=1.8, label=name)

    # Baseline reference line: R = 1 means equal to S-Learner.
    ax.axhline(1.0, color='black', linewidth=1.0, alpha=0.6, zorder=0)

    if yscale == 'log':
        ax.set_yscale('log')

    # Mark cells where the ratio was masked (R ≤ 0) with a small "x" at
    # the bottom of the panel so the reader sees the line break is real.
    ymin, _ = ax.get_ylim()
    for variant in plot_ratio.index:
        for j, ds in enumerate(ds_sorted):
            r = ratio.loc[variant, ds]
            if pd.notna(r) and r <= 0:
                ax.plot(j, ymin, marker='x', markersize=7,
                        color='dimgray', clip_on=False, zorder=1)

    xticklabels = [f'{d}\n({conv_rates[d] * 100:.1f}%)' for d in ds_sorted]
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, fontsize=9)
    ax.set_xlabel('Dataset (sorted by baseline conversion rate)', fontsize=10)
    ylabel = r'$R_l(d) = Q_l / Q_S$'
    if yscale == 'log':
        ylabel += '   (log scale)'
    ax.set_ylabel(ylabel, fontsize=10)

    if title is None:
        title = f'{spec.pretty_name}: direct (solid) vs. log (dashed) DR families'
    ax.set_title(title, fontsize=11)

    ax.legend(loc='best', fontsize=8, framealpha=0.9, ncol=2)
    ax.grid(True, which='both', alpha=0.25)

    fig.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()

    return fig, ratio


# =============================================================================
# Best-per-group envelope plot
# =============================================================================
def plot_best_per_group(
    df_results: pd.DataFrame,
    metric,
    datasets: Iterable[str],
    *,
    conv_rates: dict,
    groups: dict,
    baseline: str = 'S',
    extras: Optional[list] = None,
    yscale: str = 'log',
    annotate_winner: bool = True,
    figsize: Optional[tuple] = None,
    title: Optional[str] = None,
    savepath: Optional[str] = None,
    dpi: int = 300,
    show: bool = True,
):
    """For each ``group`` of learners, plot the per-dataset best ratio.

    For every dataset (sorted by ascending conversion rate) and every
    group, the function picks the learner inside the group with the
    highest mean R_l(d) (for ``direction='max'`` metrics) or lowest
    (for ``direction='min'``), and connects those points across
    datasets with one line per group.

    Each marker is optionally annotated with the winning learner's
    name, so the reader can see where the per-group winner changes
    (e.g. R-Learner taking over the "other" group on RHC).

    Parameters
    ----------
    df_results : pd.DataFrame
        Per-seed benchmark results.
    metric : str | MetricSpec
        Metric key (typically ``'qini_ratio'``).
    datasets : iterable of str
        Display labels of datasets to include.
    conv_rates : dict[str, float]
        Conversion rate per dataset; sets the x-axis order.
    groups : dict[str, list of str]
        Mapping group label -> list of learner display labels in that
        group. Insertion order controls plotting order. Example::

            {
                'best direct DR': ['DR-S', 'DR-T', 'DR-Q'],
                'best log DR':    ['DR-S log', 'DR-T log', 'DR-Q log'],
                'best other':     ['T', 'Q', 'Q-Simple', 'X', 'R', 'DR'],
            }

    baseline : str
        Display label of the learner used to normalise R_l(d).
    extras : list of str, optional
        Individual learners to overlay as dotted lines (one per
        learner). Useful for highlighting outliers like the R-Learner
        on RHC without folding them into either group's "best of"
        envelope.
    yscale : {'log', 'linear'}
        Y-axis scale (``'log'`` keeps low- and high-CVR regimes legible).
    annotate_winner : bool
        Annotate each marker with the winning learner's name.
    figsize, title, savepath, dpi, show
        Standard plotting controls.

    Returns
    -------
    fig : matplotlib.figure.Figure
    best_ratio : pd.DataFrame
        Group x dataset matrix of per-group best R_l(d).
    best_winner : pd.DataFrame
        Group x dataset matrix of the winning learner's display label.
    """
    spec = METRIC_SPECS[metric] if isinstance(metric, str) else metric
    datasets = list(datasets)
    ds_sorted = sorted(datasets, key=lambda d: conv_rates[d])

    extras = list(extras) if extras else []

    relevant = {baseline}
    for members in groups.values():
        relevant.update(members)
    relevant.update(extras)

    sub = df_results[df_results['dataset'].isin(ds_sorted)
                     & df_results['learner'].isin(relevant)]
    means = (sub.groupby(['learner', 'dataset'])[spec.column]
             .mean().unstack('dataset'))

    if baseline not in means.index:
        raise KeyError(f"Baseline learner {baseline!r} missing from results.")
    base_mean = means.loc[baseline].reindex(ds_sorted)
    ratio = means.divide(base_mean, axis=1).reindex(columns=ds_sorted)

    pick = (lambda s: s.idxmax()) if spec.direction == 'max' else (lambda s: s.idxmin())

    best_ratio = pd.DataFrame(index=list(groups.keys()), columns=ds_sorted, dtype=float)
    best_winner = pd.DataFrame(index=list(groups.keys()), columns=ds_sorted, dtype=object)
    for gname, members in groups.items():
        present = [m for m in members if m in ratio.index]
        if not present:
            continue
        gmat = ratio.loc[present]
        for ds in ds_sorted:
            col = gmat[ds].dropna()
            if col.empty:
                continue
            winner = pick(col)
            best_ratio.at[gname, ds] = col.loc[winner]
            best_winner.at[gname, ds] = winner

    plot_ratio = best_ratio.where(best_ratio > 0) if yscale == 'log' else best_ratio

    # ---- Render -------------------------------------------------------------
    if figsize is None:
        figsize = (1.5 + 1.1 * len(ds_sorted), 4.6)
    fig, ax = plt.subplots(figsize=figsize)

    palette = plt.get_cmap('tab10').colors
    markers = ['o', 's', '^', 'D', 'v']
    x = np.arange(len(ds_sorted))

    for gi, gname in enumerate(groups.keys()):
        c = palette[gi]
        m = markers[gi % len(markers)]
        y = plot_ratio.loc[gname].to_numpy(dtype=float)
        ax.plot(x, y, color=c, marker=m, markersize=7, linewidth=1.8,
                label=gname)

        if annotate_winner:
            for j, ds in enumerate(ds_sorted):
                w = best_winner.at[gname, ds]
                yv = y[j]
                if not isinstance(w, str) or pd.isna(yv):
                    continue
                ax.annotate(
                    w, xy=(j, yv),
                    xytext=(0, 4), textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=8, fontweight='bold', color='black',
                )

    # Overlay individual extra learners as dotted lines.
    extra_offset = len(groups)
    for ei, name in enumerate(extras):
        if name not in ratio.index:
            continue
        c = palette[(extra_offset + ei) % len(palette)]
        m = markers[(extra_offset + ei) % len(markers)]
        y_raw = ratio.loc[name].reindex(ds_sorted).to_numpy(dtype=float)
        y = np.where(y_raw > 0, y_raw, np.nan) if yscale == 'log' else y_raw
        ax.plot(x, y, color=c, marker=m, markersize=7, linewidth=1.4,
                linestyle=':', label=name)

    # Highlight the S-Learner baseline (R = 1).
    baseline_color = '#d62728'  # tab:red — distinct from group palette
    ax.axhline(1.0, color=baseline_color, linewidth=1.6, alpha=0.9,
               zorder=1.5)
    ax.text(len(ds_sorted) - 1.1, 1.0, baseline,
            color=baseline_color, fontsize=10, fontweight='bold',
            va='center', ha='left', zorder=3,
            bbox=dict(facecolor='white', edgecolor='none', pad=1.0))

    if yscale == 'log':
        ax.set_yscale('log')
        # Render y-axis ticks as plain decimals (0.1, 1, 10, 100, ...).
        from matplotlib.ticker import FuncFormatter
        def _ylab(v, _pos):
            if v <= 0:
                return ''
            exp = np.log10(v)
            if abs(exp - round(exp)) > 1e-6:
                return ''
            n = int(round(exp))
            if n >= 0:
                return f'{10 ** n:d}'
            return f'{10.0 ** n:g}'
        ax.yaxis.set_major_formatter(FuncFormatter(_ylab))

    xticklabels = [f'{d}\n({conv_rates[d] * 100:.1f}%)' for d in ds_sorted]
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, fontsize=9)
    ax.set_xlabel('Dataset (sorted by baseline conversion rate)', fontsize=10)
    ylabel = rf'$R_l(d) = {spec.column.split("_")[0].capitalize()}_l / {spec.column.split("_")[0].capitalize()}_{{\,{baseline}}}$'
    if yscale == 'log':
        ylabel += '   (log scale)'
    ax.set_ylabel(ylabel, fontsize=10)

    if title is None:
        title = f'{spec.pretty_name}: per-group winner (vs. {baseline}-Learner)'
    ax.set_title(title, fontsize=11)

    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.grid(True, which='both', alpha=0.25)

    fig.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()

    return fig, best_ratio, best_winner


# =============================================================================
# Per-dataset dot scatter
# =============================================================================
def plot_dot_per_learner(
    df_results: pd.DataFrame,
    metric,
    datasets: Iterable[str],
    *,
    conv_rates: dict,
    baseline: str = 'S',
    learners: Optional[list] = None,
    yscale: str = 'log',
    annotate_best: int = 3,
    figsize: Optional[tuple] = None,
    title: Optional[str] = None,
    savepath: Optional[str] = None,
    dpi: int = 300,
    show: bool = True,
):
    """One dot per (learner, dataset): R_l(d) = Qini_l / Qini_baseline.

    Each dataset becomes one column on the x-axis (sorted by ascending
    conversion rate). Within each column, every non-baseline learner is
    drawn as a dot at its mean R_l(d). Dots are jittered horizontally
    so overlapping values stay readable.

    The baseline is shown as a horizontal red line at R = 1; non-positive
    R values are masked when ``yscale='log'`` (the corresponding learner
    is omitted from the column with a small "x" mark at the bottom).

    Parameters
    ----------
    df_results : pd.DataFrame
        Per-seed benchmark results.
    metric : str | MetricSpec
        Metric key (typically ``'qini_ratio'``).
    datasets : iterable of str
        Display labels of datasets to include.
    conv_rates : dict[str, float]
        Conversion rate per dataset; sets the x-axis order.
    baseline : str
        Display label of the baseline learner (default ``'S'``).
    learners : list of str, optional
        Display labels of learners to plot (one dot per learner per
        dataset). The baseline is excluded. Defaults to all learners
        present in ``df_results`` other than the baseline.
    yscale : {'log', 'linear'}
        Y-axis scale (``'log'`` recommended when ratios span an order
        of magnitude).
    annotate_best : int
        Annotate the top-``k`` learners per dataset with their display
        label, so the reader can identify which dot is which without a
        legend.
    figsize, title, savepath, dpi, show
        Standard plotting controls.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ratio : pd.DataFrame
        Per-(learner, dataset) ratio used for plotting.
    """
    spec = METRIC_SPECS[metric] if isinstance(metric, str) else metric
    datasets = list(datasets)
    ds_sorted = sorted(datasets, key=lambda d: conv_rates[d])

    sub = df_results[df_results['dataset'].isin(ds_sorted)]
    means = (sub.groupby(['learner', 'dataset'])[spec.column]
             .mean().unstack('dataset'))

    if baseline not in means.index:
        raise KeyError(f"Baseline learner {baseline!r} missing from results.")
    base_mean = means.loc[baseline].reindex(ds_sorted)

    if learners is None:
        learners = [l for l in means.index if l != baseline]
    else:
        learners = [l for l in learners if l != baseline and l in means.index]

    ratio = (means.loc[learners].divide(base_mean, axis=1)
             .reindex(columns=ds_sorted))

    # ---- Render -------------------------------------------------------------
    n_cols = len(ds_sorted)
    if figsize is None:
        figsize = (1.2 + 1.2 * n_cols, 4.6)
    fig, ax = plt.subplots(figsize=figsize)

    rng = np.random.default_rng(0)
    palette = plt.get_cmap('tab20').colors

    pick = (lambda s, k: s.nlargest(k)) if spec.direction == 'max' \
        else (lambda s, k: s.nsmallest(k))

    for j, ds in enumerate(ds_sorted):
        col = ratio[ds].dropna()
        if yscale == 'log':
            col = col[col > 0]
        if col.empty:
            continue

        jitter = rng.uniform(-0.18, 0.18, size=len(col))
        xs = j + jitter
        ys = col.to_numpy()
        for k, lname in enumerate(col.index):
            ax.scatter(xs[k], ys[k], s=42, alpha=0.85,
                       color=palette[hash(lname) % len(palette)],
                       edgecolors='black', linewidths=0.4, zorder=2)

        if annotate_best > 0:
            top = pick(col, min(annotate_best, len(col)))
            for lname, val in top.items():
                xpos = j + jitter[list(col.index).index(lname)]
                ax.annotate(
                    lname, xy=(xpos, val),
                    xytext=(5, 0), textcoords='offset points',
                    ha='left', va='center', fontsize=7.5,
                )

        # Mark non-positive ratios as small "x" at the panel bottom.
        full_col = ratio[ds]
        bad = full_col[(full_col.notna()) & (full_col <= 0)]
        if not bad.empty and yscale == 'log':
            ymin, _ = ax.get_ylim()
            for lname in bad.index:
                ax.plot(j, ymin, marker='x', markersize=7,
                        color='dimgray', clip_on=False, zorder=1)

    # S-Learner baseline.
    baseline_color = '#d62728'
    ax.axhline(1.0, color=baseline_color, linewidth=1.6, alpha=0.9,
               zorder=1.5)
    ax.text(n_cols - 0.55, 1.0, baseline,
            color=baseline_color, fontsize=10, fontweight='bold',
            va='center', ha='left', zorder=3,
            bbox=dict(facecolor='white', edgecolor='none', pad=1.0))

    if yscale == 'log':
        ax.set_yscale('log')
        from matplotlib.ticker import FuncFormatter
        def _ylab(v, _pos):
            if v <= 0:
                return ''
            exp = np.log10(v)
            if abs(exp - round(exp)) > 1e-6:
                return ''
            n = int(round(exp))
            if n >= 0:
                return f'{10 ** n:d}'
            return f'{10.0 ** n:g}'
        ax.yaxis.set_major_formatter(FuncFormatter(_ylab))

    xticklabels = [f'{d}\n({conv_rates[d] * 100:.1f}%)' for d in ds_sorted]
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(xticklabels, fontsize=9)
    ax.set_xlim(-0.6, n_cols - 0.4)
    ax.set_xlabel('Dataset (sorted by baseline conversion rate)', fontsize=10)

    qname = spec.column.split('_')[0].capitalize()
    ylabel = rf'$R_l(d) = \mathrm{{{qname}}}_l / \mathrm{{{qname}}}_{{\,{baseline}}}$'
    if yscale == 'log':
        ylabel += '   (log scale)'
    ax.set_ylabel(ylabel, fontsize=10)

    if title is None:
        title = f'{spec.pretty_name}: every learner relative to {baseline}-Learner'
    ax.set_title(title, fontsize=11)

    ax.grid(True, axis='y', which='both', alpha=0.25)

    fig.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()

    return fig, ratio


# =============================================================================
# Group-coloured dot cloud with extra-learner lines
# =============================================================================
def plot_dots_with_lines(
    df_results: pd.DataFrame,
    metric,
    datasets: Iterable[str],
    *,
    conv_rates: dict,
    groups: dict,
    extras: Optional[list] = None,
    secondary: Optional[list] = None,
    baseline: str = 'S',
    regime_split_after: Optional[int] = None,
    yscale: str = 'log',
    annotate_top: int = 1,
    figsize: Optional[tuple] = None,
    title: Optional[str] = None,
    savepath: Optional[str] = None,
    dpi: int = 300,
    show: bool = True,
):
    """Group-coloured dot cloud with selected learners drawn as lines.

    For each dataset (sorted by ascending conversion rate), every
    learner that belongs to a group in ``groups`` is drawn as a single
    dot at its mean R_l(d), coloured by group. Learners in ``extras``
    are pulled out and drawn as lines connecting their R_l(d) values
    across datasets, so individual featured learners can be tracked.

    A vertical reference line can split the panel between low- and
    high-conversion-rate regimes (e.g. ``regime_split_after=3`` draws
    the line between the 4th and 5th dataset on the x-axis).

    Parameters
    ----------
    df_results : pd.DataFrame
        Per-seed benchmark results.
    metric : str | MetricSpec
        Metric key (typically ``'qini_ratio'``).
    datasets : iterable of str
        Display labels of datasets to include.
    conv_rates : dict[str, float]
        Conversion rate per dataset; sets the x-axis order.
    groups : dict[str, list of str]
        Mapping group label -> list of learner display labels in that
        group. Each learner becomes one dot per dataset, coloured by
        its group. Insertion order controls legend order.
    extras : list of str, optional
        Learners to draw as lines on top of the dot cloud rather than
        as dots.
    baseline : str
        Display label of the baseline learner. Excluded from both dots
        and lines; shown as a red horizontal line at $R = 1$.
    regime_split_after : int, optional
        If given, draw a dashed vertical line between dataset index
        ``regime_split_after`` and ``regime_split_after + 1``. Use
        ``3`` for "after the 4th dataset" (zero-indexed).
    yscale : {'log', 'linear'}
        Y-axis scale.
    annotate_top : int
        Annotate the top-``k`` learner names per dataset (across all
        plotted learners, dots and lines combined). Set to ``0`` to
        suppress.
    figsize, title, savepath, dpi, show
        Standard plotting controls.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ratio : pd.DataFrame
        Per-(learner, dataset) ratio used for plotting.
    """
    spec = METRIC_SPECS[metric] if isinstance(metric, str) else metric
    extras = list(extras) if extras else []
    secondary = list(secondary) if secondary else []
    datasets = list(datasets)
    ds_sorted = sorted(datasets, key=lambda d: conv_rates[d])

    relevant = {baseline}
    for members in groups.values():
        relevant.update(members)
    relevant.update(extras)
    relevant.update(secondary)

    sub = df_results[df_results['dataset'].isin(ds_sorted)
                     & df_results['learner'].isin(relevant)]
    means = (sub.groupby(['learner', 'dataset'])[spec.column]
             .mean().unstack('dataset'))

    if baseline not in means.index:
        raise KeyError(f"Baseline learner {baseline!r} missing from results.")
    base_mean = means.loc[baseline].reindex(ds_sorted)
    ratio = means.divide(base_mean, axis=1).reindex(columns=ds_sorted)

    # ---- Render -------------------------------------------------------------
    n_cols = len(ds_sorted)
    if figsize is None:
        figsize = (1.6 + 1.05 * n_cols, 4.8)
    fig, ax = plt.subplots(figsize=figsize)

    palette = plt.get_cmap('tab10').colors

    # Curated colour assignments. Red is reserved for the baseline,
    # so it does not appear in either group_colors or extra_colors.
    # The two lists do not overlap, so groups and extras are always
    # visually distinct from each other and from the baseline.
    # Second slot is a warm goldenrod yellow — strongly distinct from
    # the blue first slot, while dark enough to stay legible on white.
    group_colors = [palette[0], '#e6b800', palette[5], palette[7], palette[4]]
    # blue, gold, brown, gray, purple
    extra_colors = [palette[i] for i in (2, 1, 9, 6)]      # green, orange, cyan, pink

    # Legend handles: baseline first, then extras (lines), then groups
    # (dot clouds). Built up in this order so the legend reads
    # top-down "baseline / featured / context".
    from matplotlib.lines import Line2D
    group_handles = []
    extra_handles = []

    # Dots for grouped learners (zorder=2, behind extras). Secondary
    # learners stay in the dot cloud (so they keep their group colour);
    # only their connecting line is drawn separately in grey.
    for gi, (gname, members) in enumerate(groups.items()):
        c = group_colors[gi % len(group_colors)]
        members_present = [m for m in members
                           if m in ratio.index and m not in extras
                           and m != baseline]
        if not members_present:
            continue
        for j, ds in enumerate(ds_sorted):
            col = ratio.loc[members_present, ds].dropna()
            if yscale == 'log':
                col = col[col > 0]
            if col.empty:
                continue
            ax.scatter(np.full(len(col), j), col.to_numpy(),
                       s=42, alpha=0.75, color=c,
                       edgecolors='black', linewidths=0.4, zorder=2)
        group_handles.append(
            Line2D([], [], marker='o', linestyle='', color=c,
                   markeredgecolor='black', markeredgewidth=0.4,
                   markersize=8, label=gname)
        )

    # Extras are drawn as lines (zorder=5, in front of dots and grid).
    # Markers use distinct shapes. On log scale, non-positive values
    # cannot be drawn at their true coordinate, so we drop the line to
    # the panel floor and mark the failing column with an "x" — the line
    # passes through the "x" so the reader sees the dip as part of the
    # learner's trajectory rather than as a broken line.
    line_markers = ['s', '^', 'D', 'v', 'P', 'X']
    x = np.arange(n_cols)

    # First pass: plot extras only at their valid (positive) positions so
    # matplotlib autoscales the y-axis based on positive data alone. We
    # plot nothing connecting yet — connections are drawn in a second
    # pass once ymin is known.
    extras_state = []  # list of (name, color, marker, y_raw, clipped_mask)
    for ei, name in enumerate(extras):
        if name not in ratio.index:
            continue
        c = extra_colors[ei % len(extra_colors)]
        m = line_markers[ei % len(line_markers)]
        y_raw = ratio.loc[name].reindex(ds_sorted).to_numpy(dtype=float)

        if yscale == 'log':
            clipped_mask = (y_raw <= 0) | np.isnan(y_raw)
            # Markers at valid points only; keeps autoscale tight.
            ax.scatter(x[~clipped_mask], y_raw[~clipped_mask],
                       color=c, marker=m, s=64,
                       edgecolors='black', linewidths=0.5, zorder=6)
            extras_state.append((name, c, m, y_raw, clipped_mask))
        else:
            ax.plot(x, y_raw, color=c, marker=m, markersize=8, linewidth=2.2,
                    linestyle='-', markeredgecolor='black',
                    markeredgewidth=0.5, label=name, zorder=5)

        extra_handles.append(
            Line2D([], [], color=c, marker=m, markersize=8, linewidth=2.2,
                   markeredgecolor='black', markeredgewidth=0.5,
                   label=name)
        )

    # Secondary lines: thin gray context lines connecting an
    # additional learner's trajectory across datasets, without
    # changing how that learner is drawn in the dot cloud (the
    # learner's coloured group dot still appears at each dataset).
    # Different secondaries are distinguished by linestyle.
    secondary_styles = ['-', '--', ':', '-.']
    secondary_color = '#777777'
    secondary_handles = []
    secondary_state = []  # for log-scale clipping handling
    for si, name in enumerate(secondary):
        if name not in ratio.index:
            continue
        ls = secondary_styles[si % len(secondary_styles)]
        y_raw = ratio.loc[name].reindex(ds_sorted).to_numpy(dtype=float)

        if yscale == 'log':
            clipped_mask = (y_raw <= 0) | np.isnan(y_raw)
            secondary_state.append((name, ls, y_raw, clipped_mask))
        else:
            ax.plot(x, y_raw, color=secondary_color, linestyle=ls,
                    linewidth=1.0, label=name, zorder=3)

        secondary_handles.append(
            Line2D([], [], color=secondary_color, linestyle=ls,
                   linewidth=1.0, label=name)
        )

    legend_handles = extra_handles + secondary_handles + group_handles

    # Annotate top-k learner per column (across dots and lines).
    if annotate_top > 0:
        plotted = set()
        for members in groups.values():
            plotted.update(m for m in members
                           if m in ratio.index and m not in extras
                           and m not in secondary and m != baseline)
        plotted.update(e for e in extras if e in ratio.index)
        plotted.update(s for s in secondary if s in ratio.index)

        pick = ((lambda s, k: s.nlargest(k)) if spec.direction == 'max'
                else (lambda s, k: s.nsmallest(k)))

        # Determine the panel-wide max so labels near the top can be
        # placed below the marker instead of above it (which would
        # otherwise overflow the axes).
        all_pos = ratio.where(ratio > 0).stack()
        panel_max = float(all_pos.max()) if not all_pos.empty else None

        for j, ds in enumerate(ds_sorted):
            col = ratio.loc[list(plotted), ds].dropna()
            if yscale == 'log':
                col = col[col > 0]
            if col.empty:
                continue
            top = pick(col, min(annotate_top, len(col)))
            for lname, val in top.items():
                # Place below the marker if the value sits within the
                # top 25% of the panel range, so the label doesn't
                # overflow upward.
                place_below = False
                if panel_max is not None and val > 0:
                    if yscale == 'log':
                        ratio_to_max = np.log10(val) / np.log10(panel_max) \
                            if panel_max > 1 else 0.0
                        place_below = ratio_to_max > 0.75
                    else:
                        place_below = val > 0.75 * panel_max
                dy, va = (-9, 'top') if place_below else (8, 'bottom')

                ax.annotate(
                    lname, xy=(j, val),
                    xytext=(0, dy), textcoords='offset points',
                    ha='center', va=va,
                    fontsize=8, fontweight='bold',
                )

    # Regime split.
    if regime_split_after is not None:
        ax.axvline(regime_split_after + 0.5, color='black',
                   linestyle='--', linewidth=1.0, alpha=0.6, zorder=1)

    # S-Learner baseline.
    baseline_color = '#d62728'
    ax.axhline(1.0, color=baseline_color, linewidth=1.6, alpha=0.9,
               zorder=1.5)
    ax.text(n_cols - 0.55, 1.0, baseline,
            color=baseline_color, fontsize=10, fontweight='bold',
            va='center', ha='left', zorder=4,
            bbox=dict(facecolor='white', edgecolor='none', pad=1.0))
    baseline_handle = Line2D([], [], color=baseline_color, linewidth=1.6,
                             label=f'{baseline} (baseline)')
    legend_handles = [baseline_handle] + legend_handles

    if yscale == 'log':
        ax.set_yscale('log')
        from matplotlib.ticker import FuncFormatter
        def _ylab(v, _pos):
            if v <= 0:
                return ''
            exp = np.log10(v)
            if abs(exp - round(exp)) > 1e-6:
                return ''
            n = int(round(exp))
            if n >= 0:
                return f'{10 ** n:d}'
            return f'{10.0 ** n:g}'
        ax.yaxis.set_major_formatter(FuncFormatter(_ylab))

    # Draw extras lines now that the y-axis has been scaled — for
    # log-scale plots we want the line to dip down to the panel floor
    # at columns where the value is non-positive, with an "x" marker
    # indicating the failure. This way the line stays continuous across
    # all datasets instead of breaking at the bad column.
    if extras_state:
        ymin, _ = ax.get_ylim()
        for name, c, marker, y_raw, clipped_mask in extras_state:
            y_line = np.where(clipped_mask, ymin, y_raw)
            ax.plot(x, y_line, color=c, linewidth=2.2,
                    linestyle='-', label=name, zorder=5)
            for j_neg in np.where(clipped_mask)[0]:
                ax.scatter(j_neg, ymin, marker='x', s=70, color=c,
                           linewidths=2.0, clip_on=False, zorder=6)
                ax.annotate(
                    f'{name} < 0', xy=(j_neg, ymin),
                    xytext=(0, 8), textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=7.5, color='black', fontweight='bold',
                    zorder=10,
                    bbox=dict(facecolor='white', edgecolor='none',
                              alpha=0.85, pad=1.0),
                )

    # Same idea for secondary (thin gray) lines: connect through the
    # panel floor at clipped columns, with a small "x" marker so the
    # failure stays visible.
    if secondary_state:
        ymin, _ = ax.get_ylim()
        for name, ls, y_raw, clipped_mask in secondary_state:
            y_line = np.where(clipped_mask, ymin, y_raw)
            ax.plot(x, y_line, color=secondary_color, linestyle=ls,
                    linewidth=1.0, label=name, zorder=3)
            for j_neg in np.where(clipped_mask)[0]:
                ax.scatter(j_neg, ymin, marker='x', s=36,
                           color=secondary_color, linewidths=1.4,
                           clip_on=False, zorder=4)
                ax.annotate(
                    f'{name} < 0', xy=(j_neg, ymin),
                    xytext=(0, 8), textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=7, color='black',
                    zorder=10,
                    bbox=dict(facecolor='white', edgecolor='none',
                              alpha=0.85, pad=1.0),
                )

    xticklabels = [f'{d}\n({conv_rates[d] * 100:.1f}%)' for d in ds_sorted]
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(xticklabels, fontsize=9)
    ax.set_xlim(-0.6, n_cols - 0.4)
    ax.set_xlabel('Dataset (sorted by baseline conversion rate)', fontsize=10)

    qname = spec.column.split('_')[0].capitalize()
    ylabel = rf'$R_l(d) = \mathrm{{{qname}}}_l / \mathrm{{{qname}}}_{{\,{baseline}}}$'
    if yscale == 'log':
        ylabel += '   (log scale)'
    ax.set_ylabel(ylabel, fontsize=10)

    if title is None:
        title = f'{spec.pretty_name}: per-learner ratios vs. {baseline}-Learner'
    ax.set_title(title, fontsize=11)

    ax.legend(handles=legend_handles, loc='best', fontsize=8.5,
              framealpha=0.9)
    ax.grid(True, axis='y', which='both', alpha=0.25)

    fig.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()

    return fig, ratio


# =============================================================================
# Internal helpers
# =============================================================================
def _text_color_for(log_ratio_val: float, color_clip: float,
                    is_neutral: bool) -> str:
    """Pick a text color that contrasts with the background cell color."""
    if is_neutral or pd.isna(log_ratio_val):
        return 'black'
    # RdYlGn at the extremes is dark; mid is yellow. Black reads well on
    # most of the colormap; switch to white only at strong saturation.
    if abs(log_ratio_val) > color_clip * 0.7:
        return 'white'
    return 'black'
