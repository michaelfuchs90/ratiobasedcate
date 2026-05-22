"""
Visualizations for benchmark results.

Per-(learner, dataset) statistics are computed against the S-Learner as
the natural ML baseline:

    R_l(d) = Q_l(d) / Q_S(d)         for direction='max' (Qini)
    R_l(d) = E_l(d) / E_S(d)         for direction='min' (CalError)

R_l > 1 means better than the S baseline (for Qini) or worse (for
CalError); R_l < 1 means the opposite.
"""

import colorsys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure


# Metric can be passed as the dict key or as a fully-resolved MetricSpec.
MetricLike = Union[str, 'MetricSpec']


__all__ = [
    'MetricSpec',
    'MetricLike',
    'METRIC_SPECS',
    'insignificant_datasets',
    'plot_heatmap',
    'plot_best_per_group',
    'plot_dots_with_lines',
]


@dataclass
class MetricSpec:
    """How a metric is aggregated and presented.

    ``direction='max'`` means higher is better (Qini, R>1 beats S);
    ``'min'`` means lower is better (CalError, R<1 beats S).
    ``pretty_name`` is used in figure titles and table headers.
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


def _paired_equality_matrix(
    df_results: pd.DataFrame,
    metric,
    *,
    baseline: str,
    learners: Iterable[str],
    datasets: Iterable[str],
    z: float = 1.96,
) -> pd.DataFrame:
    """Per-(learner, dataset) paired test against ``baseline``.

    For each (learner, dataset), tests whether the per-seed paired difference
    delta(seed) = metric_learner(seed) - metric_baseline(seed) is within
    ``z * SE`` of zero. Pairing removes the seed-level common variance shared
    by all learners on the same train/test split. Returns a boolean matrix:
    True ↔ learner statistically equivalent to baseline.
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
    metric: MetricLike,
    *,
    baseline: str,
    learners: Iterable[str],
    datasets: Iterable[str],
    z: float = 1.96,
    min_grey_cells: int = 1,
) -> List[str]:
    """Datasets where ≥ ``min_grey_cells`` learners are statistically
    indistinguishable from the baseline. Useful as a "do we need more seeds?" check.
    """
    eq = _paired_equality_matrix(
        df_results, metric, baseline=baseline,
        learners=learners, datasets=datasets, z=z,
    )
    eq_no_base = eq.drop(index=baseline, errors='ignore')
    grey_per_ds = eq_no_base.sum(axis=0)
    return [d for d in datasets if grey_per_ds.get(d, 0) >= min_grey_cells]


def plot_heatmap(
    df_results: pd.DataFrame,
    metric: MetricLike,
    datasets: Iterable[str],
    *,
    learner_order: Iterable[str],
    conv_rates: Dict[str, float],
    baseline: str = 'S',
    z: float = 1.96,
    color_clip: float = 2.0,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    savepath: Optional[str] = None,
    dpi: int = 300,
    show: bool = True,
) -> Tuple[Figure, pd.DataFrame]:
    """Per-(learner, dataset) heatmap of metric ratios against ``baseline``.

    Diverging log-ratio scale: green = beats baseline, red = worse, neutral
    grey = paired-test inconclusive (within ``z`` SEs). ``color_clip`` is the
    log2 saturation point (default 2.0 ⇒ ≥4× / ≤0.25× saturate). x-axis is
    sorted by ``conv_rates`` ascending.

    Returns ``(fig, ratio_matrix)``.
    """
    spec = METRIC_SPECS[metric] if isinstance(metric, str) else metric
    datasets = list(datasets)
    learner_order = [l for l in learner_order if l != baseline]
    learner_order = [baseline] + learner_order  # baseline always first row

    ds_sorted = sorted(datasets, key=lambda d: conv_rates[d])

    sub = df_results[df_results['dataset'].isin(ds_sorted)
                     & df_results['learner'].isin(learner_order)].copy()

    means = sub.groupby(['learner', 'dataset'])[spec.column].mean().unstack('dataset')

    learners_present = [l for l in learner_order if l in means.index]
    means = means.reindex(index=learners_present, columns=ds_sorted)

    if baseline not in means.index:
        raise KeyError(f"Baseline learner {baseline!r} missing from results.")

    base_mean = means.loc[baseline]
    ratio = means.divide(base_mean, axis=1)

    equal_to_baseline = _paired_equality_matrix(
        sub, metric=spec, baseline=baseline,
        learners=learners_present, datasets=ds_sorted, z=z,
    )

    # Cell-state classification:
    #   "learner_negative": Q_l < 0 (worse than random) → deep red, raw value shown.
    #   "ratio_undef":      baseline non-positive or NaN → ratio meaningless, show "n/a".
    learner_negative = means.lt(0) & base_mean.gt(0) & means.notna()
    ratio_undef = (
        base_mean.le(0)
        | means.isna()
        | base_mean.isna()
    ) & ~learner_negative

    safe_ratio = ratio.where(~ratio_undef & ~learner_negative)
    log_ratio = np.log2(safe_ratio)
    if spec.direction == 'min':
        log_ratio = -log_ratio  # invert so green = better in both directions

    color_data = log_ratio.where(~equal_to_baseline)

    n_rows, n_cols = means.shape
    if figsize is None:
        figsize = (1.0 + 0.85 * n_cols, 0.6 + 0.42 * n_rows)

    fig, ax = plt.subplots(figsize=figsize)

    cmap = plt.get_cmap('RdYlGn')
    norm = TwoSlopeNorm(vmin=-color_clip, vcenter=0.0, vmax=color_clip)

    bg = color_data.to_numpy(dtype=float)
    img = ax.imshow(bg, aspect='auto', cmap=cmap, norm=norm,
                    interpolation='nearest')

    # Overlays for special cells: equal-to-baseline (neutral grey),
    # ratio-undefined (medium grey), learner-negative (deep red, darker
    # than the RdYlGn red endpoint so it reads as "broken" not "bad").
    NEUTRAL_RGB = (0.92, 0.92, 0.92)
    UNDEF_RGB = (0.78, 0.78, 0.78)
    DEEP_RED_RGB = (0.42, 0.0, 0.10)
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

    for i, lname in enumerate(learners_present):
        for j, ds in enumerate(ds_sorted):
            if learner_negative.iat[i, j]:
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

    xticklabels = [f'{d}\n({conv_rates[d] * 100:.1f}%)' for d in ds_sorted]
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(xticklabels, fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(learners_present, fontsize=9)

    # Highlight the baseline row.
    base_idx = learners_present.index(baseline)
    ax.add_patch(plt.Rectangle((-0.5, base_idx - 0.5), n_cols, 1,
                               fill=False, edgecolor='black', linewidth=1.5,
                               zorder=4))

    ax.set_xlabel(f'Dataset (sorted by conversion rate)', fontsize=10)
    ax.set_ylabel('Learner', fontsize=10)
    if title is None:
        title = f'{spec.pretty_name}: ratio to {baseline}-Learner'
    ax.set_title(title, fontsize=11)

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


def _desaturate(color, saturation: float = 0.45, lightness_boost: float = 0.10):
    """Mute a colour by lowering HLS saturation and brightening slightly."""
    r, g, b = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    s = max(0.0, min(1.0, s * saturation))
    l = max(0.0, min(1.0, l + lightness_boost))
    return colorsys.hls_to_rgb(h, l, s)


def _shade_good_bad(ax, *, direction: str, baseline_y: float = 1.0,
                    good_color: str = '#2ca02c', bad_color: str = '#d62728',
                    alpha: float = 0.06) -> None:
    """Tint the background green on the "better than baseline" side and red on
    the "worse" side. Direction='max' ⇒ y > baseline is good (green above);
    direction='min' ⇒ y < baseline is good (green below).

    Call after data is plotted so ``ax.get_ylim()`` reflects autoscale.
    """
    ymin, ymax = ax.get_ylim()
    good_top, bad_bot = (ymax, ymin) if direction == 'max' else (ymin, ymax)
    good_bot, bad_top = (baseline_y, baseline_y)
    ax.axhspan(min(good_bot, good_top), max(good_bot, good_top),
               facecolor=good_color, alpha=alpha, zorder=0)
    ax.axhspan(min(bad_bot, bad_top), max(bad_bot, bad_top),
               facecolor=bad_color, alpha=alpha, zorder=0)
    ax.set_ylim(ymin, ymax)


# Learners that keep their full saturation. S is the baseline (drawn as the
# red reference line, not as a coloured series), so only Q stays vivid.
_VIVID_LEARNERS = frozenset({'Q'})

# Lower bound for the log-scale y-axis in plot_dots_with_lines. A handful of
# learners (e.g. DR-Q on Twins) dip an order of magnitude below the next
# worst point and stretch the panel asymmetrically; we clip them to this
# floor and flag them with a "< 0.1" annotation instead.
_Y_FLOOR_LOG = 0.1


def plot_best_per_group(
    df_results: pd.DataFrame,
    metric: MetricLike,
    datasets: Iterable[str],
    *,
    conv_rates: Dict[str, float],
    groups: Dict[str, List[str]],
    baseline: str = 'S',
    extras: Optional[List[str]] = None,
    yscale: str = 'log',
    annotate_winner: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    savepath: Optional[str] = None,
    dpi: int = 300,
    show: bool = True,
) -> Tuple[Figure, pd.DataFrame, pd.DataFrame]:
    """For each ``group`` of learners, plot the per-dataset best R_l(d).

    For every dataset (sorted by ``conv_rates``) and every group, picks the
    learner inside the group with the best mean R_l(d) and connects those
    points across datasets. Each marker can be annotated with the winning
    learner's name. ``extras`` are individual learners drawn as dotted
    overlays on top of the group envelopes.

    Returns ``(fig, best_ratio, best_winner)``.
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

    plot_ratio = (best_ratio.where(best_ratio > _Y_FLOOR_LOG)
                  if yscale == 'log' else best_ratio)

    if figsize is None:
        figsize = (1.5 + 1.1 * len(ds_sorted), 4.6)
    fig, ax = plt.subplots(figsize=figsize)

    palette = plt.get_cmap('tab10').colors
    markers = ['o', 's', '^', 'D', 'v']
    x = np.arange(len(ds_sorted))

    for gi, gname in enumerate(groups.keys()):
        c = _desaturate(palette[gi])
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

    extra_offset = len(groups)
    for ei, name in enumerate(extras):
        if name not in ratio.index:
            continue
        raw_c = palette[(extra_offset + ei) % len(palette)]
        c = raw_c if name in _VIVID_LEARNERS else _desaturate(raw_c)
        m = markers[(extra_offset + ei) % len(markers)]
        y_raw = ratio.loc[name].reindex(ds_sorted).to_numpy(dtype=float)
        y = (np.where(y_raw > _Y_FLOOR_LOG, y_raw, np.nan)
             if yscale == 'log' else y_raw)
        ax.plot(x, y, color=c, marker=m, markersize=7, linewidth=1.4,
                linestyle=':', label=name)

    # S-Learner reference line at R = 1.
    baseline_color = '#d62728'  # tab:red — distinct from group palette
    ax.axhline(1.0, color=baseline_color, linewidth=1.6, alpha=0.9,
               zorder=1.5)
    ax.text(len(ds_sorted) - 1.1, 1.0, baseline,
            color=baseline_color, fontsize=10, fontweight='bold',
            va='center', ha='left', zorder=3,
            bbox=dict(facecolor='white', edgecolor='none', pad=1.0))

    if yscale == 'log':
        ax.set_yscale('log')
        # Render y-ticks as plain decimals (0.1, 1, 10, ...) instead of 10^k.
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

    _shade_good_bad(ax, direction=spec.direction)

    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.grid(True, which='both', alpha=0.25)

    fig.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()

    return fig, best_ratio, best_winner


def plot_dots_with_lines(
    df_results: pd.DataFrame,
    metric: MetricLike,
    datasets: Iterable[str],
    *,
    conv_rates: Dict[str, float],
    groups: Dict[str, List[str]],
    extras: Optional[List[str]] = None,
    secondary: Optional[List[str]] = None,
    baseline: str = 'S',
    regime_split_after: Optional[int] = None,
    yscale: str = 'log',
    annotate_top: int = 1,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    savepath: Optional[str] = None,
    dpi: int = 300,
    show: bool = True,
) -> Tuple[Figure, pd.DataFrame]:
    """Group-coloured dot cloud with selected learners drawn as lines.

    Each grouped learner becomes one dot per dataset coloured by its group.
    ``extras`` learners are pulled out and drawn as bold lines connecting
    R_l(d) across datasets. ``secondary`` learners stay in the dot cloud
    but get an extra thin grey context line. ``regime_split_after=k`` draws
    a dashed vertical line between dataset k and k+1 (zero-indexed).

    Returns ``(fig, ratio)``.
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

    n_cols = len(ds_sorted)
    if figsize is None:
        figsize = (1.6 + 1.05 * n_cols, 4.8)
    fig, ax = plt.subplots(figsize=figsize)

    palette = plt.get_cmap('tab10').colors

    # Curated palette: red is reserved for the baseline; group_colors and
    # extra_colors are disjoint so groups, extras and baseline are always
    # visually distinct. Slot 1 is goldenrod (warm but legible on white).
    raw_group_colors = [palette[0], '#e6b800', palette[5], palette[7], palette[4]]
    # Group dots are always context: muted regardless of which learners they
    # contain (no group is a single privileged learner).
    group_colors = [_desaturate(c) for c in raw_group_colors]
    # blue, gold, brown, gray, purple
    raw_extra_colors = [palette[i] for i in (2, 1, 9, 6)]      # green, orange, cyan, pink

    # Legend order: baseline, then extras (lines), then groups (dot clouds) —
    # reads top-down "baseline / featured / context".
    from matplotlib.lines import Line2D
    group_handles = []
    extra_handles = []

    # Dots for grouped learners. Secondary learners stay in the dot cloud
    # (keep their group colour); only their connecting line is drawn separately.
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
                col = col[col > _Y_FLOOR_LOG]
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

    # Extras: bold lines on top of the dot cloud. On log-scale, non-positive
    # values cannot be drawn at their true coordinate — we drop the line to
    # the panel floor and mark the failing column with an "x" so the dip
    # reads as part of the trajectory, not a broken line.
    line_markers = ['s', '^', 'D', 'v', 'P', 'X']
    x = np.arange(n_cols)

    # First pass: plot extras only at valid (positive) positions so matplotlib
    # autoscales y on positive data alone. Connecting segments are drawn
    # later, after ymin is known.
    extras_state = []  # (name, color, marker, y_raw, clipped_mask)
    for ei, name in enumerate(extras):
        if name not in ratio.index:
            continue
        raw_c = raw_extra_colors[ei % len(raw_extra_colors)]
        c = raw_c if name in _VIVID_LEARNERS else _desaturate(raw_c)
        m = line_markers[ei % len(line_markers)]
        y_raw = ratio.loc[name].reindex(ds_sorted).to_numpy(dtype=float)

        if yscale == 'log':
            clipped_mask = (y_raw <= _Y_FLOOR_LOG) | np.isnan(y_raw)
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

    # Secondary lines: thin grey context lines (different linestyles to
    # distinguish multiple secondaries) without changing how the learner
    # appears in the dot cloud.
    secondary_styles = ['-', '--', ':', '-.']
    secondary_color = '#777777'
    secondary_handles = []
    secondary_state = []
    for si, name in enumerate(secondary):
        if name not in ratio.index:
            continue
        ls = secondary_styles[si % len(secondary_styles)]
        y_raw = ratio.loc[name].reindex(ds_sorted).to_numpy(dtype=float)

        if yscale == 'log':
            clipped_mask = (y_raw <= _Y_FLOOR_LOG) | np.isnan(y_raw)
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

        # Panel-wide max so labels near the top can be placed below the marker
        # instead of overflowing the axes.
        all_pos = ratio.where(ratio > 0).stack()
        panel_max = float(all_pos.max()) if not all_pos.empty else None

        for j, ds in enumerate(ds_sorted):
            col = ratio.loc[list(plotted), ds].dropna()
            if yscale == 'log':
                col = col[col > _Y_FLOOR_LOG]
            if col.empty:
                continue
            top = pick(col, min(annotate_top, len(col)))
            for lname, val in top.items():
                # Place below if val sits in the top 25% of the panel range.
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

    if regime_split_after is not None:
        ax.axvline(regime_split_after + 0.5, color='black',
                   linestyle='--', linewidth=1.0, alpha=0.6, zorder=1)

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

    # Extras and secondary lines: gap the line over clipped columns instead
    # of dragging it to a floor marker. Values below _Y_FLOOR_LOG are
    # excluded entirely so matplotlib autoscale stays tight.
    for name, c, marker, y_raw, clipped_mask in extras_state:
        y_line = np.where(clipped_mask, np.nan, y_raw)
        ax.plot(x, y_line, color=c, linewidth=2.2,
                linestyle='-', label=name, zorder=5)

    for name, ls, y_raw, clipped_mask in secondary_state:
        y_line = np.where(clipped_mask, np.nan, y_raw)
        ax.plot(x, y_line, color=secondary_color, linestyle=ls,
                linewidth=1.0, label=name, zorder=3)

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

    _shade_good_bad(ax, direction=spec.direction)

    ax.legend(handles=legend_handles, loc='best', fontsize=8.5,
              framealpha=0.9)
    ax.grid(True, axis='y', which='both', alpha=0.25)

    fig.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()

    return fig, ratio


def _text_color_for(log_ratio_val: float, color_clip: float,
                    is_neutral: bool) -> str:
    """Pick a text colour that contrasts with the background cell colour."""
    if is_neutral or pd.isna(log_ratio_val):
        return 'black'
    # RdYlGn: dark at the extremes, yellow in the middle. Black reads on most
    # of the colormap; switch to white only at strong saturation.
    if abs(log_ratio_val) > color_clip * 0.7:
        return 'white'
    return 'black'
