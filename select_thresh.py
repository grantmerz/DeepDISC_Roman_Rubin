"""
select_thresh.py — pick and defend an operating point from grid_search.py's CSVs.

Answers one reviewer question: why select on F1 rather than UBF, when UBF is the
headline metric. Shows the F1-optimal combo also lands at low UBF / high purity.

Usage:
    python select_thresh.py \
        --metrics-dir ~/lsst_runs/run_name/metrics \
        --csv-files gold_buf2/gs_metrics.csv \
        --output-dir gold_buf2/select \
        --mag-limits gold --buffers 2
"""
import argparse
import operator
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from time import perf_counter

from plot_grid_search import (
    _pareto_front, LOWER_IS_BETTER, METRIC_LABELS, LSST_COLOR, LSST_LABEL, load_csv_infos, plot_heatmap
)
from blend_metrics import METRIC_ABBREV

# {label} is "{mag_limit} buf{buffer}" (e.g. "gold buf2"); {pipeline}, {x_metric}, {y_metric},
# {ll} always available; {mag_limit}/{buffer} mirror plot_grid_search's scatter template.
DEFAULT_TRADEOFF_TITLE_TEMPLATE = '{label} — {pipeline} {y_metric} vs {x_metric} (LL={ll}")'
# {label}, {pipeline}, {ll}, {mag_limit}, {buffer} -- same placeholders as the trade-off title.
DEFAULT_PARETO_MATRIX_TITLE_TEMPLATE = '{label} — {pipeline} Pareto matrix (LL={ll}")'
DEFAULT_PARCOORDS_TITLE_TEMPLATE = '{label} — {pipeline} parallel coordinates (LL={ll}")'


class Timer:
    """Simple context manager for printing elapsed wall-clock time."""
    def __init__(self, label: str):
        self.label = label
    def __enter__(self):
        self.t0 = perf_counter()
        return self
    def __exit__(self, exc_type, exc, tb):
        dt = perf_counter() - self.t0
        print(f'====== [TIMING] {self.label}: {dt:.3f}s ========')


def col(pipeline, metric, ll):
    return f'{pipeline}_{metric}_{ll}'


def _tag(row):
    return f's{row["score_thresh"]}_n{row["nms_thresh"]}'


def _fmt_with_delta(value, baseline):
    """'0.9790 (+0.0645)' -- value, then signed delta (value - baseline)."""
    return f'{value:.4f} ({value - baseline:+.4f})'


_CONSTRAINT_RE = re.compile(r'^\s*(\w+)\s*(>=|<=|==|!=|>|<)\s*([-+]?\d*\.?\d+)\s*$')
_CONSTRAINT_OPS = {'>=': operator.ge, '<=': operator.le, '==': operator.eq,
                   '!=': operator.ne, '>': operator.gt, '<': operator.lt}


def parse_constraints(specs):
    """Parse ['purity >= 0.85', ...] into [(metric, op_func, value), ...]."""
    parsed = []
    for spec in specs:
        m = _CONSTRAINT_RE.match(spec)
        if not m:
            raise ValueError(f'Bad --constraint {spec!r}; expected "METRIC OP VALUE", '
                             f'e.g. "purity >= 0.85"')
        metric, op, value = m.group(1), m.group(2), float(m.group(3))
        parsed.append((metric, _CONSTRAINT_OPS[op], value))
    return parsed


def apply_constraints(df, pipeline, constraints, ll):
    """Filter df to rows satisfying every (metric, op, value) constraint at this ll."""
    for metric, op, value in constraints:
        key = col(pipeline, metric, ll)
        if key not in df.columns:
            raise KeyError(f'--constraint metric {metric!r} not found '
                           f'(looked for column {key!r})')
        df = df[op(df[key], value)]
    return df


# score_thresh / nms_thresh are raw grid-parameter columns in df, not the usual
# pipeline/ll-prefixed per-metric columns -- give them their own column + LaTeX-label
# lookup so plot_tradeoff can use them (e.g. color_metric='score_thresh') too.
_THRESH_LABELS = {'score_thresh': r'$\theta_{\mathrm{ROI}}$', 'nms_thresh': r'$\theta_{\mathrm{NMS}}$'}


def _metric_col(pipeline, metric, ll):
    return metric if metric in _THRESH_LABELS else col(pipeline, metric, ll)


def _metric_label(metric):
    return _THRESH_LABELS.get(metric) or METRIC_LABELS.get(metric, metric)


def _threshold_label(prefix, row):
    """LaTeX threshold label for legend entries, matching plot_grid_search.py's
    'Best F1: $\\theta_{ROI}$=.., $\\theta_{NMS}$=..' convention.
    Puts the combo on a new line so legend entries read e.g.
    'Best Unrecognized Blend Fraction:\nθ_ROI=0.50, θ_NMS=0.50'."""
    return (rf'{prefix}:' '\n'
           rf'$\theta_{{\mathrm{{ROI}}}}$={row["score_thresh"]:.2f}, '
           rf'$\theta_{{\mathrm{{NMS}}}}$={row["nms_thresh"]:.2f}')


# ============================================================================
# Ranking
# ============================================================================
def percentile(df, pipeline, metric, ll):
    """Per-config percentile 0-100, better = higher for every metric.
    Higher-is-better ranks ascending (largest raw -> 100); lower-is-better ranks
    descending (smallest raw -> 100)."""
    asc = metric not in LOWER_IS_BETTER
    return df[col(pipeline, metric, ll)].rank(pct=True, ascending=asc) * 100
def mean_percentile_across_lls(df, pipeline, metric, lls):
    """Rank combos by mean percentile across linking lengths.

    `df` must have the same (score_thresh, nms_thresh) combos valid for every LL
    (i.e. already aligned). For each LL, computes percentile 0-100 (higher=better);
    per combo, takes the mean across LLs.
    Returns a DataFrame indexed by (score, nms) sorted best-first."""
    idx_cols = ['score_thresh', 'nms_thresh']
    pcts = []
    for ll in lls:
        p = percentile(df, pipeline, metric, str(ll))
        p.index = pd.MultiIndex.from_frame(df[idx_cols])
        pcts.append(p)
    result = pd.concat(pcts, axis=1).mean(axis=1).sort_values(ascending=False)
    return result.rename('score').to_frame()




# ============================================================================
# Multi-metric operating point (default set)
# ============================================================================
def select_operating_point(df, pipeline, metric_set, ll):
    """Rank combos by percentile aggregated across a metric SET, on one config.
    Each column is direction-corrected by `percentile` (higher = better on every axis),
    then scored two ways: mean_pct (best average axis) and min_pct (no axis sacrificed).
    Returns a DataFrame indexed by (score, nms) with one column per set metric's
    percentile plus mean_pct and min_pct, sorted by mean_pct best-first."""
    pct = pd.DataFrame({m: percentile(df, pipeline, m, ll) for m in metric_set})
    pct.index = pd.MultiIndex.from_frame(df[['score_thresh', 'nms_thresh']])
    pct['mean_pct'] = pct[metric_set].mean(axis=1)
    pct['min_pct'] = pct[metric_set].min(axis=1)
    return pct.sort_values('mean_pct', ascending=False)


# ============================================================================
# Pareto set over a metric set
# ============================================================================
def pareto_set(df, pipeline, metrics, ll):
    """Boolean mask of combos non-dominated across `metrics` (N-D, default the 3-set).
    Combo i is dominated if some j is better-or-equal on every metric and strictly better
    on at least one, with 'better' per metric from LOWER_IS_BETTER. Vectorized numpy
    domination check (n x n broadcast) over ~165 combos -- no scipy, no k-d tree (see
    Ponytail notes for why neither applies)."""
    cols = [col(pipeline, m, ll) for m in metrics]
    vals = df[cols].values.astype(float)
    sign = np.array([-1.0 if m in LOWER_IS_BETTER else 1.0 for m in metrics])
    signed = vals * sign  # higher = better on every axis now
    ge = (signed[None, :, :] >= signed[:, None, :]).all(axis=2)
    gt = (signed[None, :, :] > signed[:, None, :]).any(axis=2)
    dominated_by = ge & gt
    np.fill_diagonal(dominated_by, False)
    return ~dominated_by.any(axis=1)


# marker -> (size, facecolor) for plot_pareto_matrix's mark_combos, matching plot_tradeoff's
# gold-star/white-diamond convention (with a third, silver, for the 3-set's 'min' pick).
_MARK_COMBO_STYLE = {'*': dict(s=220, facecolors='gold'), 'D': dict(s=130, facecolors='none'),
                     'P': dict(s=150, facecolors='silver'), '^': dict(s=130, facecolors='none')}


def plot_pareto_matrix(df, pipeline, metrics, ll, mark_combos=None, show_lsst=True, title=None):
    """One 2-D panel per metric pair (3 panels for the 3-set), laid out 1 x C(len,2).
    Styled like plot_tradeoff: called-out combos get gold-star/white-diamond legend
    entries with LaTeX threshold labels (no free-floating annotations), LSST reference
    dot + dotted crosshairs. Each panel: scatter all combos on the raw pair, axes inverted
    where metric is lower-is-better so up-and-right is better; overplot the N-D Pareto set
    (from `pareto_set` over ALL `metrics`, not the pair)."""
    mask = pareto_set(df, pipeline, metrics, ll)
    pairs = list(combinations(metrics, 2))
    fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 5))
    axes = np.atleast_1d(axes)

    legend_handles = {}
    star_diamond_labels = set()
    for ax, (mx, my) in zip(axes, pairs):
        x_col, y_col = col(pipeline, mx, ll), col(pipeline, my, ll)
        ax.scatter(df.loc[~mask, x_col], df.loc[~mask, y_col], c='lightgray', s=25,
                  edgecolors='k', linewidth=0.3, alpha=0.6, label='dominated')
        ax.scatter(df.loc[mask, x_col], df.loc[mask, y_col], c='crimson', s=40,
                  edgecolors='k', linewidth=0.5, zorder=4, label='Pareto set')
        if mx in LOWER_IS_BETTER:
            ax.invert_xaxis()
        if my in LOWER_IS_BETTER:
            ax.invert_yaxis()

        for combo, marker, label in (mark_combos or []):
            row = df[(df['score_thresh'] == combo[0]) & (df['nms_thresh'] == combo[1])].iloc[0]
            style = _MARK_COMBO_STYLE.get(marker, dict(s=150, facecolors='white'))
            full_label = _threshold_label(label, row)
            ax.scatter([row[x_col]], [row[y_col]], marker=marker, edgecolors='k',
                      linewidth=1.2, zorder=6, label=full_label, **style)
            if marker in ('*', 'D', 'P'):
                star_diamond_labels.add(full_label)

        if show_lsst:
            lx, ly = f'lsst_{mx}_{ll}', f'lsst_{my}_{ll}'
            if lx in df.columns and ly in df.columns and df[lx].notna().any():
                lsst_x, lsst_y = df[lx].dropna().iloc[0], df[ly].dropna().iloc[0]
                ax.scatter([lsst_x], [lsst_y], color=LSST_COLOR, marker='o', s=80,
                          edgecolors='k', linewidth=0.6, zorder=5, label=LSST_LABEL)
                ax.axvline(lsst_x, color=LSST_COLOR, ls=':', alpha=0.4)
                ax.axhline(lsst_y, color=LSST_COLOR, ls=':', alpha=0.4)

        ax.set_xlabel(METRIC_LABELS.get(mx, mx), fontsize=10)
        ax.set_ylabel(METRIC_LABELS.get(my, my), fontsize=10)
        ax.grid(True, alpha=0.3)

        for handle, label in zip(*ax.get_legend_handles_labels()):
            legend_handles.setdefault(label, handle)

    leg = axes[0].legend(legend_handles.values(), legend_handles.keys(), fontsize=9)
    for handle, text in zip(leg.legend_handles, leg.get_texts()):
        if text.get_text() in star_diamond_labels:
            if 'mean' in text.get_text() or 'min' in text.get_text():
                handle.set_sizes([75])
            else:
                handle.set_sizes([120])
    fig.suptitle(title or f'{pipeline.upper()} Pareto matrix (LL={ll}")', fontsize=13)
    fig.tight_layout()
    return fig


# ============================================================================
# Stability
# ============================================================================
def stability(df, pipeline, metric, ll, combo):
    """3x3 neighborhood spread around one (score, nms) combo, in raw metric units.
    Small span = plateau (robust); large span = sharp peak (won't generalize)."""
    key = col(pipeline, metric, ll)
    pivot = df.pivot_table(values=key, index='nms_thresh', columns='score_thresh',
                           aggfunc='mean')
    scores, nmss = sorted(pivot.columns), sorted(pivot.index)
    si, ni = scores.index(combo[0]), nmss.index(combo[1])
    # ponytail: 8-neighbor block via a slice, no adjacency graph
    block = pivot.loc[nmss[max(0, ni - 1):ni + 2], scores[max(0, si - 1):si + 2]]
    vals = block.values[~np.isnan(block.values)]
    span = vals.max() - vals.min() if len(vals) else np.nan
    return block, span, vals.std() if len(vals) else np.nan


def print_stability(df, pipeline, metric, ll, combo, label=None):
    block, span, std = stability(df, pipeline, metric, ll, combo)
    suffix = f" ({label})" if label else ""
    print(f'\nStability around s{combo[0]}_n{combo[1]} for '
         f'{METRIC_LABELS.get(metric, metric)}{suffix}:')
    print(block.round(4).to_string())
    print(f'span={span:.4f}  std={std:.4f}')


# ============================================================================
# Trade-off plot (the deliverable)
# ============================================================================
def plot_tradeoff(df, pipeline, x_metric='f1', y_metric='unrec_blend_frac',
                  color_metric='purity', ll='1.0', pareto=False, show_lsst=True,
                  mark_best_xy=True, mark_combos=None, highlight_mask=None,
                  highlight_label='Highlighted', title=None, legend_ncol=1):
    """Styled to match plot_grid_search.py's purity-vs-completeness scatter: same
    marker/alpha/edge conventions, gold-star 'best' legend entries (LaTeX threshold
    labels) instead of free-floating annotations, LSST reference dot + dotted
    axvline/axhline, and a step-function Pareto front.

    `mark_best_xy`, if False, skips the built-in best-x-metric star / best-y-metric
    diamond -- use when `mark_combos` alone should decide what gets starred (e.g.
    star = best F1 on a completeness-vs-purity plot, where F1 is neither axis).

    `mark_combos`, if given, is a list of (combo, marker, label) tuples -- same
    convention and `_MARK_COMBO_STYLE` as plot_pareto_matrix's mark_combos -- for
    calling out extra combos (e.g. best F1) beyond the built-in best-x/best-y
    star/diamond.

    `highlight_mask`, if given, is a boolean Series/array aligned to df.index --
    every matching combo gets an open ring (no fill, so the color-mapped dot
    underneath still shows) under one shared `highlight_label` legend entry, e.g.
    calling out the combos that dominate LSST without one legend entry per combo."""
    x_col, y_col, c_col = (_metric_col(pipeline, m, ll) for m in (x_metric, y_metric, color_metric))
    x_label, y_label = _metric_label(x_metric), _metric_label(y_metric)
    x_low, y_low = x_metric in LOWER_IS_BETTER, y_metric in LOWER_IS_BETTER

    fig, ax = plt.subplots(figsize=(8, 6))
    # no legend label here -- the base cloud's meaning is carried by the colorbar and
    # the called-out markers below, not by a generic "DD"/"LSST" entry.
    sc = ax.scatter(df[x_col], df[y_col], c=df[c_col], cmap='viridis', s=30, alpha=0.7,
                    edgecolors='k', linewidth=0.3)
    if x_low:
        ax.invert_xaxis()
    if y_low:
        ax.invert_yaxis()

    if highlight_mask is not None:
        ax.scatter(df.loc[highlight_mask, x_col], df.loc[highlight_mask, y_col],
                  s=110, facecolors='none', edgecolors='crimson', linewidth=1.8,
                  zorder=4, label=highlight_label)

    # label -> legend marker size, keyed by marker shape (star=120, everything else=75)
    # rather than raw scatter `s` -- keeps the legend readable regardless of how big
    # the on-plot marker is.
    legend_sizes = {}
    if mark_best_xy:
        x_idx = df[x_col].idxmin() if x_low else df[x_col].idxmax()
        y_idx = df[y_col].idxmin() if y_low else df[y_col].idxmax()
        if x_idx == y_idx:
            # the reviewer-defense case: same combo optimal for both -> one star, one label
            row = df.loc[x_idx]
            label = _threshold_label(f'Best {x_label} & {y_label}', row)
            ax.scatter([row[x_col]], [row[y_col]], marker='*', s=220, facecolors='gold',
                      edgecolors='k', linewidth=1.2, zorder=6, label=label)
            legend_sizes[label] = 120
        else:
            x_row, y_row = df.loc[x_idx], df.loc[y_idx]
            x_label_full = _threshold_label(f'Best {x_label}', x_row)
            y_label_full = _threshold_label(f'Best {y_label}', y_row)
            ax.scatter([x_row[x_col]], [x_row[y_col]], marker='*', s=220, facecolors='gold',
                      edgecolors='k', linewidth=1.2, zorder=6, label=x_label_full)
            ax.scatter([y_row[x_col]], [y_row[y_col]], marker='D', s=130, facecolors='none',
                       edgecolors='k', linewidth=1.2, zorder=6, label=y_label_full)
            legend_sizes[x_label_full] = 120
            legend_sizes[y_label_full] = 75

    for combo, marker, label in (mark_combos or []):
        row = df[(df['score_thresh'] == combo[0]) & (df['nms_thresh'] == combo[1])].iloc[0]
        style = _MARK_COMBO_STYLE.get(marker, dict(s=150, facecolors='white'))
        full_label = _threshold_label(label, row)
        ax.scatter([row[x_col]], [row[y_col]], marker=marker, edgecolors='k',
                  linewidth=1.2, zorder=6, label=full_label, **style)
        legend_sizes[full_label] = 120 if marker == '*' else 90

    if show_lsst:
        lx, ly = f'lsst_{x_metric}_{ll}', f'lsst_{y_metric}_{ll}'
        if lx in df.columns and ly in df.columns and df[lx].notna().any():
            lsst_x, lsst_y = df[lx].dropna().iloc[0], df[ly].dropna().iloc[0]
            # e.g. "LSST (C=0.9144, P=0.9876)" -- the raw operating point, not just a dot,
            # so the legend alone backs up a domination claim without reading off axes.
            x_abbr = METRIC_ABBREV.get(x_metric, x_metric).upper()
            y_abbr = METRIC_ABBREV.get(y_metric, y_metric).upper()
            # lsst_label = rf'{LSST_LABEL} (${x_abbr}$={lsst_x:.2f}, ${y_abbr}$={lsst_y:.2f})'
            lsst_label = f'{LSST_LABEL}'
            ax.scatter([lsst_x], [lsst_y], color=LSST_COLOR, marker='o', s=80,
                      edgecolors='k', linewidth=0.6, zorder=5, label=lsst_label)
            ax.axvline(lsst_x, color=LSST_COLOR, ls=':', alpha=0.4)
            ax.axhline(lsst_y, color=LSST_COLOR, ls=':', alpha=0.4)

    if pareto:
        pts = df[[x_col, y_col]].dropna().values.astype(float)
        signed = pts * [-1 if x_low else 1, -1 if y_low else 1]
        front = _pareto_front(signed) * [-1 if x_low else 1, -1 if y_low else 1]
        front = front[np.argsort(front[:, 0])]
        ax.step(front[:, 0], front[:, 1], where='post', color='k', ls='-',
               linewidth=1.2, alpha=0.7, zorder=4)

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title or f'{pipeline.upper()} {y_label} vs {x_label} (LL={ll}")', fontsize=13)
    leg = ax.legend(fontsize=9, framealpha=0.8, ncol=legend_ncol)
    for handle, text in zip(leg.legend_handles, leg.get_texts()):
        if text.get_text() in legend_sizes:
            handle.set_sizes([legend_sizes[text.get_text()]])
    ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax, label=_metric_label(color_metric))
    fig.tight_layout()
    return fig, ax


# ============================================================================
# Selection table
# ============================================================================
def print_selection_table(df, pipeline, ll, optimize, cols, compare_pipeline=None,
                         mean_combo=None, min_combo=None):
    """One row per metric in `optimize` = that metric's direction-aware optimal combo;
    columns = each metric in `cols` (raw value) plus the combo tag. If `compare_pipeline`
    is given, each cell is annotated 'value (+delta)' vs that pipeline's value on the
    same row, and a baseline row with that pipeline's own values is appended.

    If `mean_combo` and/or `min_combo` are given (each a (score_thresh, nms_thresh)
    tuple from the multi-metric operating point), additional rows 'Best mean pct' and
    'Best min pct' are inserted between the per-metric rows and the baseline row."""
    rows = []
    for metric in optimize:
        key = col(pipeline, metric, ll)
        idx = df[key].idxmin() if metric in LOWER_IS_BETTER else df[key].idxmax()
        row = df.loc[idx]
        entry = {'optimized_for': metric, 'combo': _tag(row)}
        _populate_entry(entry, row, pipeline, compare_pipeline, cols, ll)
        rows.append(entry)

    for combo, label in [(mean_combo, 'Best mean pct'), (min_combo, 'Best min pct')]:
        if combo is None:
            continue
        row = df[(df['score_thresh'] == combo[0]) & (df['nms_thresh'] == combo[1])]
        if not row.empty:
            row = row.iloc[0]
            entry = {'optimized_for': label, 'combo': _tag(row)}
            _populate_entry(entry, row, pipeline, compare_pipeline, cols, ll)
            rows.append(entry)

    table = pd.DataFrame(rows).set_index('optimized_for')

    if compare_pipeline:
        baseline = {'combo': 'n/a'}
        baseline.update({c: f'{df[col(compare_pipeline, c, ll)].dropna().iloc[0]:.4f}'
                        for c in cols})
        table.loc[f'{compare_pipeline} (baseline)'] = baseline

    print(f'\nSelection table ({pipeline.upper()}, LL={ll}"):')
    if compare_pipeline:
        print(table.to_string())
    else:
        print(table.to_string(float_format=lambda v: f'{v:.4f}'))
    return table


def _populate_entry(entry, row, pipeline, compare_pipeline, cols, ll):
    """Fill column values (raw or delta) into `entry` dict in-place."""
    if compare_pipeline:
        entry.update({c: _fmt_with_delta(row[col(pipeline, c, ll)],
                                         row[col(compare_pipeline, c, ll)])
                     for c in cols})
    else:
        entry.update({c: row[col(pipeline, c, ll)] for c in cols})


# ============================================================================
# Parallel coordinates (optional)
# ============================================================================
def plot_parcoords(df, pipeline, ll, cols, highlight='f1', output_path=None, title=None):
    """Styled like plot_tradeoff: the highlighted line's legend label uses the same
    LaTeX threshold format as the trade-off/Pareto-matrix 'Best' markers."""
    norm = pd.DataFrame(index=df.index)
    for c in cols:
        v = df[col(pipeline, c, ll)]
        lo, hi = v.min(), v.max()
        n = (v - lo) / (hi - lo) if hi > lo else 0.5
        norm[c] = (1 - n) if c in LOWER_IS_BETTER else n

    fig, ax = plt.subplots(figsize=(1.5 * len(cols) + 2, 5))
    x = range(len(cols))
    for _, row in norm.iterrows():
        ax.plot(x, row[cols].values, color='gray', alpha=0.15, linewidth=0.8)

    hi_key = col(pipeline, highlight, ll)
    best_idx = (df[hi_key].idxmin() if highlight in LOWER_IS_BETTER
               else df[hi_key].idxmax())
    highlight_label = METRIC_LABELS.get(highlight, highlight)
    ax.plot(x, norm.loc[best_idx, cols].values, color='crimson', linewidth=2.5,
           label=_threshold_label(f'Best {highlight_label}', df.loc[best_idx]))

    ax.set_xticks(list(x))
    ax.set_xticklabels([METRIC_LABELS.get(c, c) for c in cols], rotation=20, ha='right')
    ax.set_ylabel('Normalized (higher = better)')
    ax.set_title(title or f'{pipeline.upper()} parallel coordinates (LL={ll}")')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
    return fig


# ============================================================================
# CLI
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description='Select and defend an operating point from grid search results')
    p.add_argument('--metrics-dir', required=True)
    p.add_argument('--csv-files', nargs='+', required=True)
    p.add_argument('--output-dir', nargs='+', required=True)
    p.add_argument('--mag-limits', nargs='+', required=True)
    p.add_argument('--buffers', nargs='+', type=int, required=True)
    p.add_argument('--absolute-paths', action='store_true')
    p.add_argument('--linking-lengths', nargs='+', type=float, default=[0.5, 1.0])
    p.add_argument('--pipeline', default='dd', choices=['dd', 'lsst'])
    p.add_argument('--compare-pipeline', default='lsst', choices=['dd', 'lsst'],
                  help='Baseline pipeline to compare against in printed tables '
                       '(default: lsst). Annotates table cells with '
                       '"value (+delta)" vs this pipeline and appends a baseline '
                       'row/line with its own values.')

    p.add_argument('--metric', default='f1',
                  help='Metric for percentile + stability (default: f1)')
    p.add_argument('--constraint', action='append', default=[], dest='constraints',
                  metavar='"METRIC OP VALUE"',
                  help='Constraint like "purity >= 0.85" (repeatable). Filters combos '
                       'before ranking/selection/plotting for the relevant linking '
                       'length; no constraints applied by default.')
    p.add_argument('--top', type=int, default=10, help='Rows printed per ranking')
    p.add_argument('--save-csv', action=argparse.BooleanOptionalAction, default=True,
                  help='Write CSV outputs; use --no-save-csv to disable')

    p.add_argument('--headline-mag', default='gold')
    p.add_argument('--headline-buffer', type=int, default=2)

    p.add_argument('--optimize-set', nargs='+',
                  default=['f1', 'unrec_blend_frac'],
                  help='Metric set for the multi-metric operating point')
    p.add_argument('--table-optimize', nargs='+', default=['f1', 'unrec_blend_frac'])
    p.add_argument('--table-cols', nargs='+',
                  default=['f1', 'unrec_blend_frac', 'purity', 'completeness', 'spurious_frac',
                          'shred_frac', 'missed_frac', 'resolved_frac', 'absorbed_frac', 'survivor_frac', 'sig_blend_frac'])

    p.add_argument('--tradeoff-x', default='f1')
    p.add_argument('--tradeoff-y', default='unrec_blend_frac')
    p.add_argument('--tradeoff-color', default='purity')
    p.add_argument('--pareto', action='store_true')
    p.add_argument('--tradeoff-title-template', default=None,
                  help='Trade-off plot title format string. Placeholders: {label} '
                       '(e.g. "gold buf2"), {pipeline}, {x_metric}, {y_metric}, {ll}, '
                       '{mag_limit}, {buffer}. (default: "%s")' % DEFAULT_TRADEOFF_TITLE_TEMPLATE)

    p.add_argument('--pareto-matrix', action=argparse.BooleanOptionalAction, default=True,
                  help='Write the Pareto matrix plot; use --no-pareto-matrix to disable')
    p.add_argument('--pareto-set', nargs='+',
                  default=['f1', 'unrec_blend_frac'],
                  help='Metric set for the Pareto matrix (default: f1 unrec_blend_frac'
                       ' purity completeness resolved_frac spurious_frac shred_frac)')
    p.add_argument('--pareto-marks', nargs='+',
                  choices=['f1', 'mean', 'min'], default=['f1', 'mean'],
                  help='Combo markers to highlight on the Pareto matrix'
                       ' (default: f1 mean min)')
    p.add_argument('--pareto-matrix-title-template', default=None,
                  help='Pareto matrix title format string. Placeholders: {label} '
                       '(e.g. "gold buf2"), {pipeline}, {ll}, {mag_limit}, {buffer}. '
                       '(default: "%s")' % DEFAULT_PARETO_MATRIX_TITLE_TEMPLATE)

    p.add_argument('--parcoords', action='store_true')
    p.add_argument('--parcoords-highlight', default='f1')
    p.add_argument('--parcoords-cols', nargs='+',
                  default=['f1', 'unrec_blend_frac', 'purity', 'completeness',
                           'resolved_frac', 'spurious_frac', 'shred_frac'],
                  help='Metrics to include in the parallel coordinates plot'
                       ' (default: f1 unrec_blend_frac purity completeness'
                       ' resolved_frac spurious_frac shred_frac)')
    p.add_argument('--parcoords-title-template', default=None,
                  help='Parallel coordinates title format string. Placeholders: {label} '
                       '(e.g. "gold buf2"), {pipeline}, {ll}, {mag_limit}, {buffer}. '
                       '(default: "%s")' % DEFAULT_PARCOORDS_TITLE_TEMPLATE)

    p.add_argument('--stability-combo', nargs=2, type=float, default=None,
                  metavar=('SCORE', 'NMS'))
    return p.parse_args()


def main():
    start_all = perf_counter()
    args = parse_args()
    constraints = parse_constraints(args.constraints)
    with Timer('load_csv_infos'):
        csv_infos = load_csv_infos(args)
    out_dir = Path(csv_infos[0]['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    for ll in args.linking_lengths:
        ll_str = str(ll)
        ll_dir = out_dir / ll_str
        ll_dir.mkdir(parents=True, exist_ok=True)
        print(f'\n{"=" * 70}\nLL={ll_str}"  metric={args.metric}\n{"=" * 70}')
        metric_col = col(args.pipeline, args.metric, ll_str)
        df = csv_infos[0]['df']
        df = apply_constraints(df, args.pipeline, constraints, ll_str)
        if df.empty:
            print(f'Warning: constraints leave no combos at LL={ll_str}"; skipping')
            continue
        pct = percentile(df, args.pipeline, args.metric, ll_str)
        ranked = df.assign(percentile=pct).sort_values('percentile', ascending=False)
        display = ranked[['score_thresh', 'nms_thresh', metric_col, 'percentile']].head(args.top).copy()
        if args.compare_pipeline:
            baseline_val = df[col(args.compare_pipeline, args.metric, ll_str)].dropna().iloc[0]
            display[metric_col] = display[metric_col].apply(
                lambda v: _fmt_with_delta(v, baseline_val))
            print(display.to_string(index=False))
            print(f'  ({args.compare_pipeline} baseline: {baseline_val:.4f})')
        else:
            print(display.to_string(index=False))

        # ─── per-LL headline outputs ───
        with Timer('select_operating_point'):
            op_point = select_operating_point(df, args.pipeline, args.optimize_set, ll_str)
        print(f'\n-- multi-metric operating point ({", ".join(args.optimize_set)}) --')
        print(op_point.head(args.top).round(2).to_string())
        if args.save_csv:
            op_point.to_csv(ll_dir / f'optimize_set_{ll_str}.csv')

        mean_combo, min_combo = op_point['mean_pct'].idxmax(), op_point['min_pct'].idxmax()

        print_selection_table(df, args.pipeline, ll_str, args.table_optimize, args.table_cols,
                             compare_pipeline=args.compare_pipeline,
                             mean_combo=mean_combo, min_combo=min_combo)

        if args.stability_combo:
            combos = [(tuple(args.stability_combo), 'Manual')]
        else:
            combos = []
            for m in sorted({args.metric, 'f1'}):
                key = col(args.pipeline, m, ll_str)
                idx = df[key].idxmin() if m in LOWER_IS_BETTER else df[key].idxmax()
                row = df.loc[idx]
                m_label = METRIC_LABELS.get(m, m)
                combos.append(((row['score_thresh'], row['nms_thresh']), f'Best {m_label}'))
            combos.append((mean_combo, 'Best mean pct'))

        for combo, reason in combos:
            print_stability(df, args.pipeline, args.metric, ll_str, combo, label=reason)

        label = f'{args.headline_mag} buf{args.headline_buffer}'

        tradeoff_template = args.tradeoff_title_template or DEFAULT_TRADEOFF_TITLE_TEMPLATE
        tradeoff_title = tradeoff_template.format(
            label=label, pipeline=args.pipeline.upper(),
            x_metric=METRIC_LABELS.get(args.tradeoff_x, args.tradeoff_x),
            y_metric=METRIC_LABELS.get(args.tradeoff_y, args.tradeoff_y), ll=ll_str,
            mag_limit=args.headline_mag, buffer=args.headline_buffer)
        with Timer('plot_tradeoff'):
            fig, _ = plot_tradeoff(df, args.pipeline, x_metric=args.tradeoff_x,
                                  y_metric=args.tradeoff_y, color_metric=args.tradeoff_color,
                                  ll=ll_str, pareto=args.pareto, title=tradeoff_title)
            fig.savefig(ll_dir / f'tradeoff_{args.tradeoff_y}_vs_{args.tradeoff_x}.png',
                       dpi=200, bbox_inches='tight')
            plt.close(fig)

        if args.pareto_matrix and len(args.pareto_set) >= 2:
            mark_combos = []
            if 'f1' in args.pareto_marks:
                f1_row = df.loc[df[col(args.pipeline, 'f1', ll_str)].idxmax()]
                f1_combo = (f1_row['score_thresh'], f1_row['nms_thresh'])
                mark_combos.append((f1_combo, '*', 'Best F1'))
            if 'mean' in args.pareto_marks:
                mark_combos.append((mean_combo, 'D', 'Best mean pct'))
            if 'min' in args.pareto_marks:
                mark_combos.append((min_combo, 'P', 'Best min pct'))
            pareto_matrix_template = (args.pareto_matrix_title_template
                                      or DEFAULT_PARETO_MATRIX_TITLE_TEMPLATE)
            pareto_matrix_title = pareto_matrix_template.format(
                label=label, pipeline=args.pipeline.upper(), ll=ll_str,
                mag_limit=args.headline_mag, buffer=args.headline_buffer)
            with Timer('plot_pareto_matrix'):
                fig = plot_pareto_matrix(df, args.pipeline, args.pareto_set, ll_str,
                                         mark_combos=mark_combos, title=pareto_matrix_title)
                pm_name = 'pm_' + '_'.join(METRIC_ABBREV.get(m, m) for m in args.pareto_set)
                fig.savefig(ll_dir / f'{pm_name}.png', dpi=200, bbox_inches='tight')
                plt.close(fig)

        if args.parcoords:
            parcoords_template = args.parcoords_title_template or DEFAULT_PARCOORDS_TITLE_TEMPLATE
            parcoords_title = parcoords_template.format(
                label=label, pipeline=args.pipeline.upper(), ll=ll_str,
                mag_limit=args.headline_mag, buffer=args.headline_buffer)
            with Timer('plot_parcoords'):
                plot_parcoords(df, args.pipeline, ll_str, args.parcoords_cols,
                               highlight=args.parcoords_highlight,
                               output_path=ll_dir / 'parcoords.png', title=parcoords_title)

    # Cross-LL mean percentile ranking
    if len(args.linking_lengths) > 1:
        ll_strs = [str(ll) for ll in args.linking_lengths]
        # Align combos that pass constraints at every LL
        aligned = []
        for ll_str in ll_strs:
            d = apply_constraints(csv_infos[0]['df'], args.pipeline, constraints, ll_str)
            if not d.empty:
                aligned.append(d)
        if len(aligned) > 1:
            keysets = [set(map(tuple, d[['score_thresh', 'nms_thresh']].values)) for d in aligned]
            common = set.intersection(*keysets)
            aligned = [d[d[['score_thresh', 'nms_thresh']].apply(tuple, axis=1).isin(common)]
                      .reset_index(drop=True) for d in aligned]
            df_common = aligned[0]
            mean_ll = mean_percentile_across_lls(df_common, args.pipeline, args.metric, ll_strs)
            print(f'\n-- mean percentile across LLs ({", ".join(ll_strs)}) --')
            print(mean_ll.head(args.top).to_string())
            (out_dir / 'rankings_mean_ll.csv').write_text(mean_ll.to_csv())

            flat = mean_ll.reset_index()  # columns: score_thresh, nms_thresh, score
            metric_label = METRIC_LABELS.get(args.metric, args.metric)
            title = (f'Mean {metric_label} Percentile Across LLs'
                     f' ({", ".join(ll_strs)}")\n{args.pipeline.upper()}')
            fig, _, _ = plot_heatmap(flat, 'score', title=title, fmt='.0f')
            # plot_heatmap auto-derives the metric name from the column, which yields
            # '' for 'score' -- set the colorbar label ourselves.
            fig.axes[-1].set_ylabel('Mean Percentile')
            fig.savefig(out_dir / 'rankings_mean_ll.png', dpi=200, bbox_inches='tight')
            plt.close(fig)


    total_dt = perf_counter() - start_all
    print(f'[timing] total runtime: {total_dt:.3f}s')


# ============================================================================
# Self-check: direction handling is the one thing that breaks silently
# ============================================================================
def _self_check():
    toy = pd.DataFrame({
        'score_thresh': [0.1, 0.2, 0.3], 'nms_thresh': [0.5, 0.5, 0.5],
        'dd_f1_1.0': [0.5, 0.9, 0.7],                 # higher-is-better, row 1 best
        'dd_missed_frac_1.0': [0.5, 0.1, 0.7],        # lower-is-better, row 1 best
    })
    assert percentile(toy, 'dd', 'f1', '1.0').iloc[1] == 100
    assert percentile(toy, 'dd', 'missed_frac', '1.0').iloc[1] == 100

    dom = pd.DataFrame({
        'score_thresh': [0.1, 0.2], 'nms_thresh': [0.5, 0.5],
        'dd_f1_1.0': [0.9, 0.5], 'dd_unrec_blend_frac_1.0': [0.1, 0.5],
        'dd_resolved_frac_1.0': [0.9, 0.5],
    })
    mask = pareto_set(dom, 'dd', ['f1', 'unrec_blend_frac', 'resolved_frac'], '1.0')
    assert list(mask) == [True, False]

    # Cross-LL mean percentile
    ll_toy = pd.DataFrame({
        'score_thresh': [0.1, 0.2, 0.3], 'nms_thresh': [0.5, 0.5, 0.5],
        'dd_f1_0.5': [0.9, 0.6, 0.4], 'dd_f1_1.0': [0.8, 0.6, 0.4],
    })
    mll = mean_percentile_across_lls(ll_toy, 'dd', 'f1', [0.5, 1.0])
    # Combo (0.1, 0.5): f1_0.5=0.9 (pct=100), f1_1.0=0.8 (pct=100), mean=100  <-- best
    # Combo (0.2, 0.5): f1_0.5=0.6 (pct=50),  f1_1.0=0.6 (pct=50),  mean=50
    # Combo (0.3, 0.5): f1_0.5=0.4 (pct=0),   f1_1.0=0.4 (pct=0),   mean=0
    assert isinstance(mll.index, pd.MultiIndex)
    assert list(mll.index.names) == ['score_thresh', 'nms_thresh']
    assert mll.index[0] == (0.1, 0.5)


if __name__ == '__main__':
    _self_check()
    main()
