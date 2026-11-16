"""
Figures and tables of paper.tex, called from sections 6-7 of ``Benchmark.ipynb``;
``put_table`` writes tables to tables/ and, if present, into paper.tex. ``oracle_qini`` recomputes the
oracle Qini (no fitting), which the results CSV does not store.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.image import AxesImage
from matplotlib.ticker import FixedLocator, NullLocator


# cal_point of a constant predictor on the synthetic DGP.
CONSTANT_CAL_POINT = 1.42

# Background tint of good / bad regions.
GOOD_COLOR, BAD_COLOR = '#2ca02c', '#d62728'


def _goodness(ax, good: str, ref: float, axis: str, n: int) -> np.ndarray:
    """Goodness in -1 (bad) .. 1 (good) along one axis, in axes coordinates
    (index 0 = bottom / left); 0 at ``ref``, +-1 at the axis edges."""
    i = 1 if axis == 'y' else 0
    r = 0.5
    if ref is not None:
        pt = (ax.get_xlim()[0], ref) if axis == 'y' else (ref, ax.get_ylim()[0])
        r = float(np.clip((ax.transData + ax.transAxes.inverted()).transform(pt)[i], 0, 1))
    t = np.linspace(0, 1, n)
    v = np.where(t >= r, (t - r) / max(1 - r, 1e-9), (t - r) / max(r, 1e-9))  # signed, -1 .. 1
    return {'up': v, 'down': -v, 'center': 1 - 2 * np.abs(v)}[good]


def _paint(ax, s: np.ndarray, alpha) -> None:
    """Draw a goodness grid ``s`` (rows top -> bottom) as the axes background.
    ``alpha``: one value, or a (good, bad) pair."""
    a_good, a_bad = alpha if isinstance(alpha, tuple) else (alpha, alpha)
    rgba = np.zeros(s.shape + (4,))
    rgba[..., :3] = np.where(s[..., None] > 0, to_rgb(GOOD_COLOR), to_rgb(BAD_COLOR))
    rgba[..., 3] = np.where(s > 0, a_good, a_bad) * np.abs(s) ** 0.8
    img = AxesImage(ax, extent=(0, 1, 0, 1), transform=ax.transAxes, interpolation='bilinear', zorder=0)
    img.set_data(rgba)
    ax.add_image(img)


def _shade(ax, good: str, ref: float = None, axis: str = 'y', alpha: float = 0.08, n: int = 256) -> None:
    """Background gradient, neutral at ``ref`` (None: axis middle), green towards the
    ``good`` side ('up' / 'down' / 'center') and red towards the bad one. Call after
    the limits are final."""
    v = _goodness(ax, good, ref, axis, n)
    _paint(ax, v[::-1, None] if axis == 'y' else v[None], alpha)


def _shade_2d(ax, x_good: str, x_ref: float, y_good: str, y_ref: float,
              alpha=0.08, n: int = 256) -> None:
    """``_shade`` on both axes at once: a point is only as good as its worse axis,
    so green appears only where both metrics are good."""
    gx, gy = _goodness(ax, x_good, x_ref, 'x', n), _goodness(ax, y_good, y_ref, 'y', n)
    _paint(ax, np.minimum(gx[None, :], gy[::-1, None]), alpha)


# --- Synthetic part (results_synthetic.csv) ----------------------------------

def _add_cell(df: pd.DataFrame) -> pd.DataFrame:
    parts = df['dataset'].str.rsplit('_', n=2, expand=True)
    return df.assign(cvr=parts[1].astype(float), conv=parts[2].astype(int))


def _cvr_label(cvr: float) -> str:
    return f'{cvr:.1%}'.replace('.0%', '%')


def load_synthetic(path: str = 'results_synthetic.csv') -> pd.DataFrame:
    """Valid result rows (``error`` empty) with ``cvr`` and ``conv`` columns."""
    df = pd.read_csv(path)
    return _add_cell(df[df['error'].isna()])


def _style_conv_axis(ax, convs):
    ax.set_xscale('log')
    ax.set_xticks(convs)
    ax.set_xticklabels([f'{c}' if c < 1000 else f'{c / 1000:g}k' for c in convs],
                       rotation=45, fontsize=8)
    ax.minorticks_off()
    ax.grid(True, color='0.92', lw=0.6)
    ax.spines[['top', 'right']].set_visible(False)


def _style_cal_axis(ax, ylim, ref: float = CONSTANT_CAL_POINT):
    """Log axis for cal_*; ``ref`` = dotted reference line (default: constant
    predictor on the synthetic DGP; None on real data, where no such value exists)."""
    ax.set_yscale('log')
    if ylim is not None:
        ax.set_ylim(*ylim)
    fine = ax.get_ylim()[1] < 10  # call after plotting: fine ticks only on a narrow range
    ticks = [1.2, ref, 2, 3, 4] if ref is not None else [1, 1.1, 1.2, 1.5, 2, 3, 4, 6, 9]
    ax.yaxis.set_major_locator(FixedLocator(ticks if fine else 10.0 ** np.arange(8)))
    ax.yaxis.set_major_formatter(lambda v, _: f'{v:g}')
    ax.yaxis.set_minor_locator(NullLocator())
    if ref is not None:
        ax.axhline(ref, color='0.45', lw=1, ls=':')


# Learner families for the ranking-vs-calibration overview; marker = second
# encoding because five hues cannot all be told apart in a scatter.
LEARNER_FAMILY = {
    'S': 'plug-in', 'S_tuned': 'plug-in', 'S-Qini_tuned': 'plug-in', 'T': 'plug-in', 'T_tuned': 'plug-in',
    'Q': 'Q family', 'Q_tuned': 'Q family', 'Q-Simple': 'Q family', 'Q-Simple_tuned': 'Q family',
    'Q-Simple-Qini_tuned': 'Q family', 'Q-Qini_tuned': 'Q family',
    'X': 'difference', 'R': 'difference', 'DR': 'difference',
    'DR-S': 'DR direct', 'DR-T': 'DR direct', 'DR-Q': 'DR direct', 'DR-Q-Simple': 'DR direct',
    'DR-S log': 'DR log', 'DR-T log': 'DR log', 'DR-Q log': 'DR log', 'DR-Q-Simple log': 'DR log',
}

FAMILY_STYLE = {'plug-in': ('#2a78d6', 'o'), 'Q family': ('#eb6834', 's'), 'DR direct': ('#1baf7a', 'D'),
                'difference': ('#e34948', '^'), 'DR log': ('#6250d6', 'v')}


def plot_bias_vs_error(df: pd.DataFrame, convs: tuple = (200, 3200), cvrs: tuple = (0.002, 0.01, 0.05, 0.2),
                       label: tuple = ('S', 'S_tuned', 'Q-Simple_tuned', 'T', 'X', 'R', 'DR',
                                       'DR-T', 'DR-S', 'DR-Q', 'DR-T log'),
                       err_max: float = 20, x_range: tuple = (-20, 4), figsize: tuple = (11, 4.4)) -> plt.Figure:
    """Two ways to lose calibration: per learner and converter level, median level bias
    (x) against median mean |log error| (y, log). y = |x| is the floor (level error only).
    Points above ``err_max`` are pinned to it as open markers; only ``label`` gets text."""
    d = df[df['cvr'].isin(cvrs) & df['conv'].isin(convs)].assign(e=lambda x: np.log(x['cal_point']))
    g = d.groupby(['learner', 'conv']).agg(b=('bias_point', 'median'), e=('e', 'median')).reset_index()
    fig, axes = plt.subplots(1, len(convs), figsize=figsize, sharex=True, sharey=True, squeeze=False)
    for ax, conv in zip(axes[0], convs):
        xs = np.linspace(*x_range, 401)
        ax.plot(xs, np.abs(xs), color='0.45', lw=1, ls=':')
        ax.axhline(np.log(CONSTANT_CAL_POINT), color='0.75', lw=0.8, ls='--')
        ax.text(x_range[0], np.log(CONSTANT_CAL_POINT) * 1.08, 'constant predictor', fontsize=7, color='0.45')
        for _, r in g[g['conv'] == conv].iterrows():
            color, marker = FAMILY_STYLE[LEARNER_FAMILY[r['learner']]]
            off = not r['e'] <= err_max
            x = np.clip(r['b'], *x_range) if np.isfinite(r['b']) else x_range[1]
            y = err_max if off else r['e']
            ax.scatter(x, y, s=36, marker=marker, zorder=3, linewidths=1.2,
                       facecolors='none' if off else color, edgecolors=color)
            if r['learner'] in label:
                left = r['learner'] in ('S_tuned', 'DR-S')  # S_tuned sits on Q-Simple_tuned, DR-S on DR/R
                dy = {'DR': 6, 'R': -5}.get(r['learner'], 2)
                ax.annotate(r['learner'], (x, y), xytext=(-4 if left else 4, dy), textcoords='offset points',
                            fontsize=7, color='0.25', ha='right' if left else 'left')
        ax.set_yscale('log')
        ax.set_ylim(0.1, err_max * 1.6)
        ax.set_xlim(x_range[0] - 0.3, x_range[1] + 0.8)
        ax.set_title(f'{conv:,} training converters', fontsize=10)
        ax.set_xlabel(r'level bias: median of mean $\Delta$')
        ax.grid(True, color='0.92', lw=0.6)
        ax.spines[['top', 'right']].set_visible(False)
        _shade(ax, 'down', np.log(CONSTANT_CAL_POINT))
    axes[0, 0].set_ylabel(r'error: median of mean $|\Delta|$ (log)')
    for fam, (color, marker) in FAMILY_STYLE.items():
        axes[0, 0].scatter([], [], color=color, marker=marker, label=fam)
    axes[0, 0].scatter([], [], facecolors='none', edgecolors='0.4', marker='o', label=f'> {err_max:g} or inf')
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center', ncol=len(FAMILY_STYLE) + 1,
               frameon=False, bbox_to_anchor=(0.5, 1.03), fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


# Plug-in and Q-family learners, grey in plot_syn_untuned (no difference-to-ratio recovery).
GREY_UNTUNED = ('S', 'T', 'Q', 'Q-Simple', 'DR-Q-Simple', 'DR-Q-Simple log')


# Per-learner (colour, line style) of plot_syn_untuned: family hue, shade + style tell the members apart.
UNTUNED_STYLE = {
    'S': ('0.2', '-'), 'T': ('0.45', '--'), 'Q': ('0.6', ':'), 'Q-Simple': ('0.6', '-'),
    'DR-Q-Simple': ('0.35', '-.'), 'DR-Q-Simple log': ('0.5', (0, (1, 1))),
    'X': ('#e34948', '-'), 'R': ('#a32524', '--'), 'DR': ('#f29a99', ':'),
    'DR-S': ('#1baf7a', '-'), 'DR-T': ('#0d6e4b', '--'), 'DR-Q': ('#6fd8af', ':'),
    'DR-S log': ('#6250d6', '-'), 'DR-T log': ('#36289a', '--'), 'DR-Q log': ('#a59bec', ':'),
}


# Hand-placed label offsets (points[, horizontal alignment]) where default labels collide.
LABEL_OFFSET = {(200, 'X'): (-4, 14), (200, 'R'): (-6, -8, 'right'), (200, 'DR'): (-6, 10, 'right'),
                (200, 'DR-S'): (-2, -8, 'right'), (200, 'DR-Q'): (0, 14),
                (3200, 'X'): (6, 8), (3200, 'R'): (-8, 2, 'right'), (3200, 'DR'): (8, -6),
                (3200, 'DR-Q-Simple'): (-6, 14, 'right'), (3200, 'DR-S'): (24, -1),
                (3200, 'DR-T'): (8, 12), (3200, 'DR-Q log'): (5, 10), (3200, 'DR-T log'): (5, -8)}


# Main-text subset of plot_syn_untuned: S, T, Q-Simple and one member per other family.
UNTUNED_MAIN = ('S', 'T', 'Q-Simple', 'X', 'DR-S', 'DR-S log')

# Qini at which the background of plot_syn_untuned turns from red to green: half the oracle's 0.83.
UNTUNED_QINI_REF = 0.83 / 2

# x ticks of plot_syn_untuned, in cal_point units.
UNTUNED_TICKS = (1.2, 1.5, 2, 3, 5, 10, 100, 1e4)


def plot_syn_untuned(df: pd.DataFrame, convs: tuple = (200, 3200),
                     label: dict = None, learners: tuple = None,
                     figsize: tuple = (11, 4.4)) -> plt.Figure:
    """Untuned learners, one panel per converter count: mean Qini (y) against the
    geometric-mean cal_point (x, double log), one line per learner through its CVRs,
    open marker at the lowest. Infinite errors in a column on the right ('>').
    ``label``: conv -> learners named at their lowest-CVR marker; ``learners``: subset
    (e.g. ``UNTUNED_MAIN``, all labelled), default all."""
    if learners is not None:
        df = df[df['learner'].isin(learners)]
        label = label or {c: learners for c in convs}
    if label is None:
        grey = ('S', 'Q-Simple', 'T', 'DR-Q-Simple', 'DR-Q-Simple log')
        coloured = ('X', 'R', 'DR', 'DR-S', 'DR-T', 'DR-Q')
        label = {200: grey + coloured, 3200: grey + coloured + ('DR-S log', 'DR-T log', 'DR-Q log')}
    d = df[~df['learner'].str.contains('tuned') & df['conv'].isin(convs)]
    g = (d.assign(lc=np.log(d['cal_point'])).groupby(['learner', 'conv', 'cvr'])
         .agg(q=('qini_ratio', 'mean'), c=('lc', 'mean')).reset_index())  # c = log of the geometric mean
    edge = 2 * g['c'][np.isfinite(g['c'])].max()  # infinity column, shared by both panels
    fig, axes = plt.subplots(1, len(convs), figsize=figsize, sharex=True, sharey=True, squeeze=False)
    for ax, conv in zip(axes[0], convs):
        part = g[g['conv'] == conv]
        for learner, p in part.groupby('learner'):
            p = p.sort_values('cvr')
            color, ls = UNTUNED_STYLE[learner]
            z = 2 if learner in GREY_UNTUNED else 3
            x = p['c'].where(np.isfinite(p['c']), edge)
            ax.plot(x, p['q'], color=color, ls=ls, lw=1.8, zorder=z)
            ax.plot(x.iloc[0], p['q'].iloc[0], 'o', color=color, ms=7, mfc='white', mew=1.4, zorder=z + 2)
            inf = ~np.isfinite(p['c'])
            ax.plot(x[inf], p['q'][inf], '>', color=color, ms=5, zorder=z + 1)
            if learner in label.get(conv, ()):
                name = 'Q, Q-Simple' if learner == 'Q-Simple' and learners is None else learner
                off = LABEL_OFFSET.get((conv, learner), (5, -3))
                ax.annotate(name, (x.iloc[0], p['q'].iloc[0]), xytext=off[:2], textcoords='offset points',
                            fontsize=7, color=color if z == 3 else '0.25', va='top',
                            ha=off[2] if len(off) > 2 else 'left')
        ref = np.log(CONSTANT_CAL_POINT)
        ax.axvline(ref, color='0.45', lw=1, ls=':')
        ax.text(ref, 0.02, ' constant', fontsize=7.5, color='0.45', transform=ax.get_xaxis_transform())
        ax.axvline(edge, color='0.8', lw=0.8)
        ax.text(edge, 0.97, r'$\infty$ ', fontsize=9, color='0.45', ha='right', va='top',
                transform=ax.get_xaxis_transform())
        ax.set_xscale('log')
        ax.set_xticks([np.log(t) for t in UNTUNED_TICKS],
                      [f'{t:g}' if t < 1e3 else f'$10^{{{np.log10(t):.0f}}}$' for t in UNTUNED_TICKS])
        ax.minorticks_off()
        cvrs = sorted(part['cvr'].unique())
        ax.set_title(f'{conv:,} training converters (CVR {_cvr_label(cvrs[0])} to {_cvr_label(cvrs[-1])})',
                     fontsize=10)
        ax.set_xlabel(r'calibration error cal$_\mathrm{point}$ (geometric mean, double log)')
        ax.grid(True, color='0.92', lw=0.6)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0, 0].set_xlim(np.log(1.15), edge * 1.3)
    for ax in axes[0]:
        _shade_2d(ax, 'down', np.log(CONSTANT_CAL_POINT), 'up', UNTUNED_QINI_REF, alpha=(0.25, 0.08))
    axes[0, 0].set_ylabel('Qini (mean)')
    handles = [plt.Line2D([], [], color=c, lw=2, label=t)
               for c, t in zip(('0.45', '#e34948', '#1baf7a', '#6250d6'),
                               ('S, T, Q family', 'X, R, AIPW-DR', 'DR-S/T/Q direct', 'DR-S/T/Q log')
                               if learners is None else
                               ('S, T, Q-Simple', 'X (difference-based)', 'DR-S (augmentation)',
                                'DR-S log (augmentation)'))]
    handles += [plt.Line2D([], [], color='0.3', marker='o', mfc='white', ls='', ms=7, label='lowest CVR'),
                plt.Line2D([], [], color='0.3', marker='>', ls='', ms=5, label='infinite error')]
    fig.legend(handles=handles, loc='upper center', ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, 1.04), fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


# (learner, colour, marker, legend label) of the tuning figures: all solid; untuned S as grey
# reference; S variants told apart by lightness and marker (palette checked for CVD).
TUNING_MAIN = (('S', '0.55', 'o', 'S untuned'), ('S_tuned', '#6aa3e8', 'o', 'S log-loss'),
               ('S-Qini_tuned', '#1d4f91', '^', 'S Qini'), ('T_tuned', '#1baf7a', 's', 'T log-loss'),
               ('Q-Simple_tuned', '#eb6834', 'D', 'Q-Simple log-loss'))
# Same with Q in place of Q-Simple (unknown propensity, synthconf).
TUNING_CONF = tuple((l.replace('Q-Simple', 'Q'), c, m, n.replace('Q-Simple', 'Q')) for l, c, m, n in TUNING_MAIN)


def _tuning_lines(axes, df: pd.DataFrame, cols: tuple, corridor: tuple, lines: tuple = TUNING_MAIN):
    """The ``TUNING_MAIN`` lines in three axes against ``conv``: mean Qini, geometric-mean
    calibration error and share of runs whose slope lies in ``corridor`` (``cols`` = the
    three columns). Learners without rows are skipped; untuned lines thinner and lighter."""
    q, c, s = cols
    for learner, color, marker, name in lines:
        p = df[df['learner'] == learner]
        if p.empty:
            continue
        series = (p.groupby('conv')[q].mean(), np.exp(np.log(p[c]).groupby(p['conv']).mean()),
                  p[s].between(*corridor).groupby(p['conv']).mean())
        faint = dict(lw=1.2) if 'untuned' in name else dict(lw=1.8)
        for ax, v in zip(axes, series):
            ax.plot(v.index, v.values, color=color, marker=marker, ms=5, label=name, **faint)


def _tuning_legend(fig, ax, top: float, rename: tuple = None, lines: tuple = TUNING_MAIN):
    """Legend of the ``lines`` in ``ax``, one row in their order. ``rename`` = (old, new)
    substring of the labels."""
    handles, labels = ax.get_legend_handles_labels()
    names = [t[3] for t in lines]
    pairs = sorted((names.index(l), h, l) for h, l in zip(handles, labels) if l in names)
    labels = [l.replace(*rename) if rename else l for _, _, l in pairs]
    fig.legend([h for _, h, _ in pairs], labels, loc='upper center', ncol=len(pairs),
               frameon=False, bbox_to_anchor=(0.5, top), handlelength=2.5, fontsize=8.5)


def plot_syn_tuning_main(df: pd.DataFrame, oracle: pd.DataFrame = None, corridor: tuple = (0.8, 1.25),
                         figsize: tuple = (11.5, 4), lines: tuple = TUNING_MAIN,
                         cal_ylim: tuple = (1.05, 3.2)) -> plt.Figure:
    """S, T and Q-Simple (``lines``) x {untuned, log-loss, Qini}, pooled over CVRs: mean Qini,
    geometric-mean cal_point and share of runs with slope_point in ``corridor``
    (no SE bands: learners share every split, differences need paired tests)."""
    convs = sorted(df['conv'].unique())
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    _tuning_lines(axes, df, ('qini_ratio', 'cal_point', 'slope_point'), corridor, lines)
    if oracle is not None:
        o = oracle.groupby('conv')['qini_ratio'].mean()
        axes[0].plot(o.index, o.values, color='0.45', lw=1, ls='--')
        axes[0].text(convs[0], o.iloc[0] + 0.02, 'oracle', fontsize=7.5, color='0.45')
    axes[0].axhline(0, color='0.45', lw=0.8)
    _style_cal_axis(axes[1], cal_ylim)
    axes[1].text(convs[-1], CONSTANT_CAL_POINT * 0.985, 'constant', fontsize=7.5, color='0.45', ha='right', va='top')
    axes[2].set_ylim(-0.03, 1.03)
    titles = ('Ranking', 'Calibration error', f'Right spread: slope in [{corridor[0]:g}, {corridor[1]:g}]')
    ylabels = ('Qini (mean)', r'cal$_\mathrm{point}$ (geometric mean, log)', 'share of runs')
    for ax, title, ylabel in zip(axes, titles, ylabels):
        _style_conv_axis(ax, convs)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
    _shade(axes[0], 'up', UNTUNED_QINI_REF)
    _shade(axes[1], 'down', CONSTANT_CAL_POINT)
    _shade(axes[2], 'up', 0.5)
    _tuning_legend(fig, axes[0], 1.0, lines=lines)
    fig.supxlabel('training converters', fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


# --- Real data: shared helpers ----------------------------------------------

def _tuning_dataset_lines(row, df: pd.DataFrame, by: str, names: list, cols: tuple, corridor: tuple,
                          cal_max: float = None, starred: set = (), qini_ref: float = 0):
    """``TUNING_MAIN`` lines over the datasets ``names`` (equidistant, CVR in the tick
    label, '*' for ``starred``): mean ``cols[0]``, geometric mean ``cols[1]`` (pinned to
    ``cal_max`` as open marker) and share of runs with ``cols[2]`` in ``corridor``."""
    from datasets import CONVERSION_RATES
    q, c, s = cols
    for learner, color, marker, name in TUNING_MAIN:
        p = df[df['learner'] == learner]
        if p.empty:
            continue
        g = p[by].astype(str)
        share = p[s].between(*corridor).groupby(g).mean().where(p[s].notna().groupby(g).any())
        series = (p.groupby(g)[q].mean(), np.exp(np.log(p[c]).groupby(g).mean()), share)
        x = np.arange(len(names))
        lw = 1.2 if 'untuned' in name else 1.8
        for k, (ax, v) in enumerate(zip(row, series)):
            y = v.reindex(names).to_numpy()
            over = np.zeros(len(y), bool)
            if k == 1 and cal_max is not None:
                over = y > cal_max
                y = np.minimum(y, cal_max)
            ax.plot(x, y, color=color, lw=lw, marker=marker, ms=5, markevery=list(np.flatnonzero(~over)),
                    label=name)
            ax.plot(x[over], y[over], ls='none', marker=marker, ms=6, mfc='white', mec=color)
    for ax in row:
        ax.set_xticks(range(len(names)), [f'{n}{"*" if n in starred else ""}\n({_cvr_label(CONVERSION_RATES[n])})'
                                          for n in names], fontsize=7.5)
        ax.set_xlim(-0.4, len(names) - 0.6)
        ax.grid(True, axis='y', color='0.92', lw=0.6)
        ax.spines[['top', 'right']].set_visible(False)
    row[0].axhline(0, color='0.45', lw=0.8)
    _style_cal_axis(row[1], None if cal_max is None else (None, cal_max * 1.04), ref=None)
    row[2].set_ylim(-0.03, 1.03)
    _shade(row[0], 'up', qini_ref)
    _shade(row[1], 'down')
    _shade(row[2], 'up', 0.5)


def _tuning_titles(axes, corridor: tuple):
    for ax, title in zip(axes, ('Ranking', 'Calibration error',
                                f'Right spread: slope in [{corridor[0]:g}, {corridor[1]:g}]')):
        ax.set_title(title, fontsize=10)


# --- Randomized part (results_rct.csv) ---------------------------------------
# Seeds share test units (and, on pool-sized cells, training rows) -> corrected
# resampled t-test, factor rho / (1 - rho) from the s_test diagnostic (stest.py),
# else the classic Nadeau-Bengio factor n_test / n_train = 0.25.

RCT_NB = 0.25
STEST_SUMMARY = 'stest_summary.csv'
_STEST_METRIC = {('qini_ratio', False): 'qini_ratio', ('cal_binned', True): 'log_cal_binned'}


def nb_factors(metric: str, log: bool, path: str = STEST_SUMMARY) -> dict:
    """{dataset: rho / (1 - rho)} from the s_test diagnostic (stest.py), the
    largest over the learner pairs; uncovered cells (pool-sized ``*_all``) are
    missing (``table_full`` then uses ``RCT_NB``)."""
    if not os.path.exists(path) or (metric, log) not in _STEST_METRIC:
        return {}
    s = pd.read_csv(path)
    return s[s['metric'] == _STEST_METRIC[(metric, log)]].groupby('dataset')['nb_factor'].max().to_dict()


def _add_rct_cell(df: pd.DataFrame) -> pd.DataFrame:
    from datasets import RCT_BASES, _RCT_POOL_CONVERTERS
    base, lvl = df['dataset'].str.rsplit('_', n=1, expand=True).T.to_numpy()
    conv = [_RCT_POOL_CONVERTERS[b] if l == 'all' else int(l) for b, l in zip(base, lvl)]
    return df.assign(base=pd.Categorical(base, RCT_BASES, ordered=True), lvl=lvl, conv=conv)


def load_rct(path: str = 'results_rct.csv') -> pd.DataFrame:
    """Valid result rows (``error`` empty) with ``base``, ``lvl`` ('12800' or
    'all' = whole pool) and ``conv`` columns."""
    df = pd.read_csv(path)
    return _add_rct_cell(df[df['error'].isna()])


def _bases(df: pd.DataFrame) -> list:
    return [b for b in df['base'].cat.categories if b in set(df['base'])]


def rct_factors(df: pd.DataFrame, metric: str, log: bool) -> dict:
    """{base: rho / (1 - rho)} from ``nb_factors``; uncovered bases are missing (``table_full`` then uses ``RCT_NB``)."""
    f = nb_factors(metric, log)
    cell = df.groupby('base', observed=True)['dataset'].first()
    return {str(b): f[ds] for b, ds in cell.items() if ds in f}


def plot_rct_tuning(df: pd.DataFrame, corridor: tuple = (0.8, 1.25), cal_max: float = 2.0,
                    figsize: tuple = (11.5, 3.8)) -> plt.Figure:
    """``plot_syn_tuning_main`` on the randomized datasets (whole pool starred), sorted by
    CVR; Qini as a share of the best variant's mean Qini on that dataset."""
    df = df[df['learner'].isin([t[0] for t in TUNING_MAIN])]
    names = [str(b) for b in _bases(df)]
    best = df.groupby(['base', 'learner'], observed=True)['qini_ratio'].mean().groupby('base', observed=True).max()
    df = df.assign(qini_share=df['qini_ratio'] / df['base'].map(best).astype(float))
    starred = set(df.loc[df['lvl'] == 'all', 'base'].astype(str))
    fig, row = plt.subplots(1, 3, figsize=figsize)
    _tuning_dataset_lines(row, df, 'base', names, ('qini_share', 'cal_binned', 'slope_binned'),
                          corridor, cal_max, starred, qini_ref=0.5)
    _tuning_titles(row, corridor)
    for ax, ylabel in zip(row, ('Qini / best variant shown', r'cal$_\mathrm{binned}$ (geometric mean, log)',
                                'share of seeds')):
        ax.set_ylabel(ylabel)
    _tuning_legend(fig, row[0], 1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


# --- Observational part (results_obs.csv) -----------------------------------
# The *_ipw columns reweight with 1/e_hat (same LightGBM nuisance as the
# learners); train overlap across seeds -> factor 0.25.

def load_obs(path: str = 'results_obs.csv') -> pd.DataFrame:
    """Valid result rows (``error`` empty)."""
    df = pd.read_csv(path)
    return df[df['error'].isna()]


# --- Appendix tables ----------------------------------------------------------
# Learner groups, split by a rule: plug-in, Q family, difference-based,
# direct-scale and log-scale augmentations, tuned variants.
TABLE_GROUPS = (('S', 'T'), ('Q', 'Q-Simple'), ('X', 'R', 'DR'),
                ('DR-S', 'DR-T', 'DR-Q', 'DR-Q-Simple'), ('DR-S log', 'DR-T log', 'DR-Q log', 'DR-Q-Simple log'),
                ('S_tuned', 'S-Qini_tuned', 'T_tuned', 'Q-Simple_tuned', 'Q-Simple-Qini_tuned', 'Q_tuned', 'Q-Qini_tuned'))
TABLE_NAMES = {'DR': 'AIPW-DR', 'S_tuned': 'S (log-loss)', 'S-Qini_tuned': 'S (Qini)', 'T_tuned': 'T (log-loss)',
               'Q-Simple_tuned': 'Q-Simple (log-loss)', 'Q-Simple-Qini_tuned': 'Q-Simple (Qini)',
               'Q_tuned': 'Q (log-loss)', 'Q-Qini_tuned': 'Q (Qini)'}
UNTUNED_OF = {'S_tuned': 'S', 'S-Qini_tuned': 'S', 'T_tuned': 'T', 'Q-Simple_tuned': 'Q-Simple',
              'Q-Simple-Qini_tuned': 'Q-Simple', 'Q_tuned': 'Q', 'Q-Qini_tuned': 'Q'}


def _fmt_cal(v: float) -> str:
    if not np.isfinite(v):
        return r'$\infty$'
    if v >= 1000:
        return r'$10^{%d}$' % round(np.log10(v))
    return f'{v:.2f}'


def table_full(df: pd.DataFrame, by: str, cols: list, metric: str, log: bool = False,
               slope: str = None, factors: dict = None, corridor: tuple = (0.8, 1.25)) -> str:
    """LaTeX tabular: one row per learner (TABLE_GROUPS), one column per ``cols`` value of ``by``.
    Mean ``metric`` (geometric if ``log``, best in bold); below it the corrected SE, or with
    ``slope`` the share of seeds in ``corridor``. Dagger: tuned differs from untuned,
    |t| > 2 with Var = s^2 (1/J + f), f from ``factors`` or RCT_NB."""
    key = df[by].astype(str)
    v = (np.log(df[metric]) if log else df[metric]).replace(-np.inf, np.nan)
    g = v.groupby([df['learner'], key])
    mean = g.mean().unstack()
    f = pd.Series({c: (factors or {}).get(c, RCT_NB) for c in cols})
    if slope:
        s = df[slope]
        second = s.between(*corridor).groupby([df['learner'], key]).mean().where(
            s.notna().groupby([df['learner'], key]).any()).unstack()
    else:
        second = g.std().unstack() * np.sqrt(1 / g.count().unstack() + f)
    wide = v.groupby([df['learner'], key, df['dataset'], df['seed']]).first().unstack('learner')
    finite = mean.where(np.isfinite(mean))
    best = finite.idxmin() if log else finite.idxmax()
    present = set(df['learner'])
    lines = [r'\begin{tabular}{l|' + 'r' * len(cols) + '}', r'\hline',
             'Learner & ' + ' & '.join(cols) + r' \\']
    for group in TABLE_GROUPS:
        group = [l for l in group if l in present]
        if group:
            lines.append(r'\hline')
        for l in group:
            main, small = [], []
            for c in cols:
                m = mean.loc[l].get(c, np.nan)
                if np.isnan(m):
                    main.append('--')
                    small.append('')
                    continue
                txt = _fmt_cal(np.exp(m)) if log else f'{m:.3f}'.replace('-', '$-$')
                if best.get(c) == l:
                    txt = r'\textbf{' + txt + '}'
                if l in UNTUNED_OF and UNTUNED_OF[l] in wide.columns:
                    d = (wide[l] - wide[UNTUNED_OF[l]]).xs(c, level=0)
                    d = d.replace([np.inf, -np.inf], np.nan).dropna()
                    if len(d) > 1 and abs(d.mean()) > 2 * d.std() * np.sqrt(1 / len(d) + f[c]):
                        txt += r'$^\dagger$'
                main.append(txt)
                x = second.loc[l].get(c, np.nan)
                small.append('--' if np.isnan(x) else (f'{x:.0%}'.replace('%', r'\%') if slope else f'({x:.3f})'))
            lines.append(TABLE_NAMES.get(l, l) + ' & ' + ' & '.join(main) + r' \\')
            lines.append(' & ' + ' & '.join(r'\scriptsize ' + t for t in small) + r' \\[2pt]')
    return '\n'.join(lines + [r'\hline', r'\end{tabular}'])


def put_table(tabular: str, label: str, tex_path: str = 'paper.tex', out_dir: str = 'tables') -> None:
    r"""Write ``tabular`` to ``out_dir/<label>.tex`` (':' -> '_'); if ``tex_path``
    exists and has ``\label{label}``, also replace the ``tabular`` after it
    (caption, label and size commands stay as written; line endings kept)."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, label.replace(':', '_') + '.tex'), 'w', encoding='utf-8') as f:
        f.write(tabular + '\n')
    if not os.path.exists(tex_path):
        return
    with open(tex_path, encoding='utf-8', newline='') as f:
        s = f.read()
    if r'\label{%s}' % label not in s:
        return
    start = s.index(r'\begin{tabular}', s.index(r'\label{%s}' % label))
    end = s.index(r'\end{tabular}', start) + len(r'\end{tabular}')
    nl = '\r\n' if '\r\n' in s else '\n'
    with open(tex_path, 'w', encoding='utf-8', newline='') as f:
        f.write(s[:start] + tabular.replace('\n', nl) + s[end:])


# --- Dataset and search-space tables (Sec. 5.1, Appendix B) -------------------

DATASET_INFO = {  # table label, domain
    'H(Conv)': (r'H(Conv) \citep{hillstrom2008}', 'Email'),
    'Twins': (r'Twins$^\dagger$ \citep{louizos2017}', 'Neonatal'),
    'Criteo': (r'Criteo \citep{diemert2018}', 'E-commerce'),
    'Lenta': (r'Lenta \citeyearpar{lenta2021}', 'Grocery'),
    'H(Vis)': (r'H(Vis) \citep{hillstrom2008}', 'Email'),
    'MegaFon': (r'MegaFon \citeyearpar{megafon2019}', 'Telecom'),
    'X5': (r'X5 Retail \citeyearpar{x5retail2020}', 'Retail'),
    'Cattaneo': (r'Cattaneo \citep{cattaneo2010}', 'Birthweight'),
    'NHEFS': (r'NHEFS \citep{hernan2020}', 'Smoking'),
    'JTPA': (r'JTPA$^\dagger$ \citep{abadie2002}', 'Job training'),
    'RHC': (r'RHC \citep{connors1996rhc}', 'Medical'),
}


def dataset_stats(names: list, seeds: list) -> pd.DataFrame:
    """Size, conversion rate and treated share of the full data. Twins
    simulates W per seed -> mean over ``seeds``; the other datasets do not
    depend on the seed."""
    from datasets import get_dataset
    rows = []
    for name in names:
        ds = [get_dataset(name, random_state=s) for s in (seeds if name == 'Twins' else seeds[:1])]
        rows.append(dict(dataset=name, n=len(ds[0].Y), cvr=np.mean([d.Y.mean() for d in ds]),
                         treat=np.mean([d.W.mean() for d in ds])))
    return pd.DataFrame(rows).set_index('dataset')


def _fmt_n(n: int) -> str:
    return f'{n / 1e6:.1f}M' if n >= 1e6 else f'{n / 1e3:.0f}k' if n >= 1e4 else f'{n / 1e3:.1f}k'


def table_datasets(stats: pd.DataFrame, converters: dict = None) -> str:
    """Dataset table: n, conversion rate, treated share, training-pool
    converters (if given), domain; rows in the order of ``stats``."""
    head = ['Dataset', '$n$', 'CVR', 'Treat.'] + (['Converters'] if converters else []) + ['Domain']
    lines = [r'\begin{tabular}{lrrr%sl}' % ('r' if converters else ''), r'\toprule',
             ' & '.join(head) + r' \\', r'\midrule']
    for name, r in stats.iterrows():
        label, domain = DATASET_INFO[name]
        cells = [label, _fmt_n(r['n']), f"{100 * r['cvr']:.1f}\\%", f"{100 * r['treat']:.1f}\\%"]
        if converters:
            cells.append(f'{converters[name]:,}'.replace(',', '{,}'))
        lines.append(' & '.join(cells + [domain]) + r' \\')
    return '\n'.join(lines + [r'\bottomrule', r'\end{tabular}'])


def _fmt_bound(v: float) -> str:
    e = np.log10(v)
    return '$10^{%d}$' % round(e) if v < 1 and abs(e - round(e)) < 1e-9 else f'${v:g}$'


def table_search_space() -> str:
    """Search-space table from ``tune.SEARCH_SPACE``."""
    from tune import SEARCH_SPACE
    lines = [r'\begin{tabular}{ll}', r'\toprule', r'Hyperparameter & Range \\', r'\midrule']
    for name, (_, lo, hi, log) in SEARCH_SPACE.items():
        rng = f'{_fmt_bound(lo)}--{_fmt_bound(hi)}' + (r' ($\log$)' if log else '')
        lines.append(r'\texttt{%s} & %s \\' % (name.replace('_', r'\_'), rng))
    return '\n'.join(lines + [r'\bottomrule', r'\end{tabular}'])


# --- Recomputation helper (not stored in results_synthetic.csv) --------------

def _oracle_one(dataset: str, seed: int) -> dict:
    from datasets import get_dataset
    from metrics import qini_coefficient_ratio
    data = get_dataset(dataset, random_state=seed)
    q = qini_coefficient_ratio(data.tau_true_test, data.W_test, data.Y_test,
                               data.mu0_true_test, data.tau_true_test)
    return dict(dataset=dataset, learner='oracle', seed=seed, qini_ratio=q)


def oracle_qini(datasets: list, seeds: list, n_jobs: int = 1) -> pd.DataFrame:
    """Oracle Qini (true tau as prediction) per (dataset, seed), same seeds and
    test splits as the benchmark. Loads data only, fits nothing."""
    from joblib import Parallel, delayed
    rows = Parallel(n_jobs=n_jobs)(delayed(_oracle_one)(d, s) for d in datasets for s in seeds)
    return _add_cell(pd.DataFrame(rows))
