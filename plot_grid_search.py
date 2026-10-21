"""
plot_grid_search.py — heatmaps and purity-vs-completeness scatter for grid_search.py's
wide per-buffer CSVs.

One row per (score, nms) threshold combo; columns `{prefix}_{metric}_{ll}`, with
`lsst_*` columns constant across the grid unless grid_search.py was run with
--skip-lsst.

Per --output-dir, individual-mode files land under:
    {output-dir}/{ll}/individual/{pipeline}_{metric}.png
    {output-dir}/{ll}/delta/{pipeline}_delta_{metric}.png        (--delta)
    {output-dir}/{ll}/{pipeline}_purity_vs_completeness.png

Usage:
    python plot_grid_search.py \
        --metrics-dir ~/lsst_runs/lsst5_30k_4h200_bs192_ep50/metrics \
        --csv-files gold_buf2/gs_metrics.csv \
        --output-dir gold_buf2/plots \
        --mag-limits gold --buffers 2

    ===== LSST Baseline 30k ======
    Gold Mag Limit
    --------------
    python plot_grid_search.py --metrics-dir ~/lsst_runs/lsst5_30k_4h200_bs192_ep50/metrics --csv-files gold_buf2/gs_metrics.csv --output-dir gold_buf2/gs_hmaps --mag-limits gold --buffers 2 --no-comparison

    python plot_grid_search.py --metrics-dir ~/lsst_runs/lsst5_30k_4h200_bs192_ep50/metrics --csv-files gold_buf0/gs_metrics.csv gold_buf1/gs_metrics.csv gold_buf2/gs_metrics.csv --output-dir gold_buf0/gs_hmaps gold_buf1/gs_hmaps gold_buf2/gs_hmaps --mag-limits gold gold gold --buffers 0 1 2 --no-comparison
    python plot_grid_search.py --metrics-dir ~/lsst_runs/lsst5_30k_4h200_bs192_ep50/metrics --csv-files gold_buf0/gs_metrics.csv gold_buf1/gs_metrics.csv gold_buf2/gs_metrics.csv --output-dir gold_cmp --mag-limits gold gold gold --buffers 0 1 2 --metrics completeness purity f1 unrec_blend_frac shredded_frac shred_frac --no-individual --comparison

    Power Law Mag Limit
    -------------------
    python plot_grid_search.py --metrics-dir ~/lsst_runs/lsst5_30k_4h200_bs192_ep50/metrics --csv-files pl_buf0/gs_metrics.csv pl_buf1/gs_metrics.csv pl_buf2/gs_metrics.csv --output-dir pl_buf0/gs_hmaps pl_buf1/gs_hmaps pl_buf2/gs_hmaps --mag-limits power_law power_law power_law --buffers 0 1 2 --no-comparison
    python plot_grid_search.py --metrics-dir ~/lsst_runs/lsst5_30k_4h200_bs192_ep50/metrics --csv-files pl_buf0/gs_metrics.csv pl_buf1/gs_metrics.csv pl_buf2/gs_metrics.csv --output-dir pl_cmp --mag-limits power_law power_law power_law --buffers 0 1 2 --metrics completeness purity f1 unrec_blend_frac shredded_frac shred_frac --no-individual --comparison

    ===== CLIP 30k ======
    Gold Mag Limit
    --------------
    python plot_grid_search.py --metrics-dir ~/lsst_runs/clip5_30k_4h200_bs64_ep50/metrics --csv-files gold_buf0/gs_metrics.csv gold_buf1/gs_metrics.csv gold_buf2/gs_metrics.csv --output-dir gold_buf0/gs_hmaps gold_buf1/gs_hmaps gold_buf2/gs_hmaps --mag-limits gold gold gold --buffers 0 1 2 --no-comparison
    python plot_grid_search.py --metrics-dir ~/lsst_runs/clip5_30k_4h200_bs64_ep50/metrics --csv-files gold_buf0/gs_metrics.csv gold_buf1/gs_metrics.csv gold_buf2/gs_metrics.csv --output-dir gold_cmp --mag-limits gold gold gold --buffers 0 1 2 --metrics completeness purity f1 unrec_blend_frac shredded_frac shred_frac --no-individual --comparison

    Power Law Mag Limit
    -------------------
    python plot_grid_search.py --metrics-dir ~/lsst_runs/clip5_30k_4h200_bs64_ep50/metrics --csv-files pl_buf0/gs_metrics.csv pl_buf1/gs_metrics.csv pl_buf2/gs_metrics.csv --output-dir pl_buf0/gs_hmaps pl_buf1/gs_hmaps pl_buf2/gs_hmaps --mag-limits power_law power_law power_law --buffers 0 1 2 --no-comparison
    python plot_grid_search.py --metrics-dir ~/lsst_runs/clip5_30k_4h200_bs64_ep50/metrics --csv-files pl_buf0/gs_metrics.csv pl_buf1/gs_metrics.csv pl_buf2/gs_metrics.csv --output-dir pl_cmp --mag-limits power_law power_law power_law --buffers 0 1 2 --metrics completeness purity f1 unrec_blend_frac shredded_frac shred_frac --no-individual --comparison

    ===== LSST Baseline All ======
    Gold Mag Limit
    --------------
    python plot_grid_search.py --metrics-dir ~/lsst_runs/lsst5_all_4h200_bs192_ep20/metrics --csv-files gold_buf0/gs_metrics.csv gold_buf1/gs_metrics.csv gold_buf2/gs_metrics.csv --output-dir gold_buf0/gs_hmaps gold_buf1/gs_hmaps gold_buf2/gs_hmaps --mag-limits gold gold gold --buffers 0 1 2 --no-comparison
    python plot_grid_search.py --metrics-dir ~/lsst_runs/lsst5_all_4h200_bs192_ep20/metrics --csv-files gold_buf0/gs_metrics.csv gold_buf1/gs_metrics.csv gold_buf2/gs_metrics.csv --output-dir gold_cmp --mag-limits gold gold gold --buffers 0 1 2 --metrics completeness purity f1 unrec_blend_frac shredded_frac shred_frac --no-individual --comparison


    # With absolute paths (use --absolute-paths flag)
    python plot_grid_search.py --csv-files /path/to/file1.csv /path/to/file2.csv \
                                --output-dir /path/to/output --absolute-paths \
                                --mag-limits gold nominal \
                                --buffers 0 1

    # DD-LSST delta heatmaps + Pareto front outline, only LL=1.0"
    python plot_grid_search.py --metrics-dir ~/lsst_runs/lsst5_30k_4h200_bs192_ep50/metrics \
                                --csv-files gold_buf2/gs_metrics.csv \
                                --output-dir gold_buf2/plots \
                                --mag-limits gold --buffers 2 \
                                --delta --pareto --linking-lengths 1.0

    # Plot the LSST pipeline's own metrics instead of DD's (--pipeline dd is the default)
    python plot_grid_search.py --metrics-dir ~/lsst_runs/lsst5_30k_4h200_bs192_ep50/metrics \
                                --csv-files gold_buf2/gs_metrics.csv \
                                --output-dir gold_buf2/plots_lsst \
                                --mag-limits gold --buffers 2 --pipeline lsst

    # Custom title, no cell numbers
    python plot_grid_search.py --metrics-dir ~/lsst_runs/lsst5_30k_4h200_bs192_ep50/metrics \
                                --csv-files gold_buf2/gs_metrics.csv \
                                --output-dir gold_buf2/plots \
                                --mag-limits gold --buffers 2 --no-annotate \
                                --title-template '{metric} as a Function of $\\theta_{{\\mathrm{{ROI}}}}$ and $\\theta_{{\\mathrm{{NMS}}}}$ (LL = {ll}", $\\Delta m$ = {buffer})'
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter

import blend_metrics as bm             # single source for the metric set
import plot_binned_metrics as bpm      # single source for labels + palette

METRICS = list(bm.METRIC_NUMDEN) + ['f1']         # 12, matches grid_search.py
METRIC_LABELS = {**bpm.METRIC_LABELS, 'f1': 'F1'}  # bpm lacks f1
LSST_COLOR = bpm.LSST_COLOR                         # '#C85200'
LSST_LABEL = bpm.LSST_LABEL                         # 'LSST'

# confirmed direction grouping
LOWER_IS_BETTER = {'missed_frac', 'spurious_frac', 'unrec_blend_frac', 'shred_frac',
                   'shredded_frac', 'sig_blend_frac', 'survivor_frac', 'absorbed_frac'}
# (higher-is-better = completeness, purity, f1, resolved_frac)

# Heatmap (plot_heatmap / plot_delta_heatmap only -- not the purity-vs-completeness scatter).
HEATMAP_TICK_FONTSIZE = 14
HEATMAP_AXIS_LABEL_FONTSIZE = 15
HEATMAP_CBAR_FONTSIZE = 15
HEATMAP_TITLE_FONTSIZE = 16

# {metric} is the display name from METRIC_LABELS; {pipeline} is upper-cased.
# {mag_limit}/{buffer} are only available in individual mode (comparison mode spans
# multiple configs, so those two are unset there).
DEFAULT_TITLE_TEMPLATE = '{metric} ({pipeline}, LL={ll}")'
# {lsst} is the LSST reference value, pre-formatted as a percentage string. {mag_limit}/
# {buffer} are individual-mode-only, same as DEFAULT_TITLE_TEMPLATE.
DEFAULT_DELTA_TITLE_TEMPLATE = '{metric} DD-LSST (LSST={lsst}, LL={ll}")'
# {label} is csv_info['label'] (e.g. "gold buf2"); {pipeline}, {ll} always available;
# {mag_limit}/{buffer} mirror the individual-heatmap template's availability.
DEFAULT_SCATTER_TITLE_TEMPLATE = '{label} — {pipeline} Purity vs Completeness (LL={ll}")'


# ============================================================================
# Shared helpers
# ============================================================================
def _metric_key(col):
    """'dd_completeness_1.0' -> 'completeness'. Strips the trailing _{ll} then the
    leading {prefix}_. Used by plot_heatmap, plot_heatmap_comparison, and
    plot_delta_heatmap."""
    return '_'.join(col.rsplit('_', 1)[0].split('_')[1:])

def _cmap_for(metric):
    """Blue = better, Red = worse, using a colorblind-safe diverging colormap
    for both a normal metric heatmap and a DD-LSST delta heatmap alike
    """
    return 'coolwarm' if metric in LOWER_IS_BETTER else 'coolwarm_r'
    # return 'RdYlGn_r' if metric in LOWER_IS_BETTER else 'RdYlGn' Green = better, red = worse

def _annotate_cell(ax, j, i, text):
    ax.text(j, i, text, ha='center', va='center', fontsize=8, color='black',
           bbox=dict(facecolor='white', edgecolor='none', alpha=0.82, pad=0.18,
                     boxstyle='round,pad=0.18'), zorder=3)


def _mark_best_cell(ax, values, metric):
    """Outline + star the best cell in a 2D grid (rows=nms_thresh, cols=score_thresh)
    by the metric's direction (LOWER_IS_BETTER vs higher-is-better). Drawn regardless
    of the annotate flag, so the best combo is always visible on a heatmap."""
    if np.all(np.isnan(values)):
        return
    finder = np.nanargmin if metric in LOWER_IS_BETTER else np.nanargmax
    best_i, best_j = np.unravel_index(finder(values), values.shape)

    ax.add_patch(Rectangle((best_j - 0.5, best_i - 0.5), 1, 1, fill=False,
                           edgecolor='black', linewidth=2.0, zorder=10))
    ax.plot(best_j + 0.33, best_i - 0.33, marker='*', markersize=15,
           markerfacecolor='gold', markeredgecolor='black', markeredgewidth=0.7,
           zorder=11)


def _pareto_front(pts):
    """Non-dominated subset of 2D points where higher is better in both dims."""
    front = []
    for i, p in enumerate(pts):
        dominated = any(
            j != i and q[0] >= p[0] and q[1] >= p[1] and (q[0] > p[0] or q[1] > p[1])
            for j, q in enumerate(pts))
        if not dominated:
            front.append(p)
    return np.array(front)


# ============================================================================
# Heatmaps
# ============================================================================
def plot_heatmap(df, col, ax=None, title=None, cmap=None,
                 fmt='.3f', vmin=None, vmax=None, annotate=True):
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(
            max(6, df['score_thresh'].nunique() * 0.9 + 2),
            max(4, df['nms_thresh'].nunique() * 0.7 + 1.5)))
    else:
        fig = ax.get_figure()

    pivot = df.pivot_table(
        values=col, index='nms_thresh', columns='score_thresh', aggfunc='mean')

    metric = _metric_key(col)
    if cmap is None:
        cmap = _cmap_for(metric)

    im = ax.imshow(pivot.values, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f'{v:.2f}' for v in pivot.columns],
                       fontsize=HEATMAP_TICK_FONTSIZE, rotation=45, ha='right')
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f'{v:.2f}' for v in pivot.index], fontsize=HEATMAP_TICK_FONTSIZE)
    ax.set_xlabel('Score Threshold', fontsize=HEATMAP_AXIS_LABEL_FONTSIZE)
    ax.set_ylabel('NMS Threshold', fontsize=HEATMAP_AXIS_LABEL_FONTSIZE)

    if annotate:
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.iloc[i, j]
                if np.isnan(val):
                    continue
                _annotate_cell(ax, j, i, f'{val:{fmt}}')

    _mark_best_cell(ax, pivot.values, metric)

    if title:
        ax.set_title(title, fontsize=HEATMAP_TITLE_FONTSIZE)

    if own_fig:
        cb_label = METRIC_LABELS.get(metric, metric)
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label=cb_label)
        cbar.set_label(cb_label, fontsize=HEATMAP_CBAR_FONTSIZE)
        cbar.ax.tick_params(labelsize=HEATMAP_TICK_FONTSIZE)
        fig.tight_layout()

    return fig, ax, im


def plot_heatmap_comparison(dfs, labels, col, output_path=None, title=None, cmap=None):
    metric = _metric_key(col)
    cbar_label = METRIC_LABELS.get(metric, metric)
    n = len(dfs)
    if n == 0:
        return None

    all_vals = np.concatenate([df[col].dropna().values for df in dfs if col in df.columns])
    if len(all_vals) == 0:
        return None
    vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)
    pad = (vmax - vmin) * 0.02
    vmin -= pad
    vmax += pad

    panel_w = max(5, dfs[0]['score_thresh'].nunique() * 0.7 + 1.5)
    panel_h = max(3.5, dfs[0]['nms_thresh'].nunique() * 0.55 + 1)
    fig, axes = plt.subplots(1, n, figsize=(panel_w * n + 1.5, panel_h + 1.2),
                             squeeze=False, sharex=True, sharey=True)

    last_im = None
    for idx, (df, label) in enumerate(zip(dfs, labels)):
        ax = axes[0, idx]
        if col not in df.columns:
            ax.set_visible(False)
            continue
        _, _, im = plot_heatmap(df, col, ax=ax, title=label, cmap=cmap,
                                vmin=vmin, vmax=vmax, annotate=(n <= 4))
        last_im = im
        ax.tick_params(labelbottom=True, labelleft=True)
        if idx > 0:
            ax.set_ylabel('')

    if last_im is not None:
        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(last_im, cax=cbar_ax, label=cbar_label)
        cbar.set_label(cbar_label, fontsize=HEATMAP_CBAR_FONTSIZE)
        cbar.ax.tick_params(labelsize=HEATMAP_TICK_FONTSIZE)

    if title:
        fig.suptitle(title, fontsize=13, y=1.02)

    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
    return fig


def plot_metric_pair_heatmap(df, cols, output_path=None, suptitle=None, annotate=False):
    """Side-by-side heatmaps for two (or more) *different* metrics of the same config --
    e.g. unrec_blend_frac next to f1 for a paper figure. Unlike plot_heatmap_comparison
    (one metric, several configs, one shared colorbar), each panel here keeps its own
    colorbar since different metrics have different scales. One suptitle spans both
    panels instead of per-panel titles, and the x-axis (identical score_thresh grid
    either way) is shared."""
    n = len(cols)
    panel_w = max(5, df['score_thresh'].nunique() * 0.7 + 1.5)
    panel_h = max(3.5, df['nms_thresh'].nunique() * 0.55 + 1)
    fig, axes = plt.subplots(1, n, figsize=(panel_w * n + 1.5, panel_h + 1.2),
                             squeeze=False, sharex=True, sharey=True)
    axes = axes[0]

    for idx, col in enumerate(cols):
        metric = _metric_key(col)
        _, ax, im = plot_heatmap(df, col, ax=axes[idx], annotate=annotate)
        if idx > 0:
            ax.set_ylabel('')
        cb_label = METRIC_LABELS.get(metric, metric)
        cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02, label=cb_label)
        cbar.set_label(cb_label, fontsize=HEATMAP_CBAR_FONTSIZE)
        cbar.ax.tick_params(labelsize=HEATMAP_TICK_FONTSIZE)

    if suptitle:
        fig.suptitle(suptitle, fontsize=HEATMAP_TITLE_FONTSIZE)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
    return fig, axes


def plot_delta_heatmap(df, metric, ll, pipeline='dd', ax=None, annotate=True,
                       title_template=None, mag_limit=None, buffer=None):
    """DD - LSST heatmap for one metric/LL. Requires lsst_{metric}_{ll}. Cell values
    and the LSST reference in the title are both shown as percentage points."""
    dd_col = f'{pipeline}_{metric}_{ll}'
    lsst_col = f'lsst_{metric}_{ll}'
    if lsst_col not in df.columns or df[lsst_col].dropna().empty:
        print(f'Note: skipping delta heatmap for {metric}_{ll} (no {lsst_col})')
        return None

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(
            max(6, df['score_thresh'].nunique() * 0.9 + 2),
            max(4, df['nms_thresh'].nunique() * 0.7 + 1.5)))
    else:
        fig = ax.get_figure()

    lsst_val = df[lsst_col].dropna().iloc[0]
    pivot = df.pivot_table(values=dd_col, index='nms_thresh', columns='score_thresh',
                           aggfunc='mean')
    delta = pivot - lsst_val

    # avoids TwoSlopeNorm's vmin==vmax==vcenter error when DD==LSST on every cell
    vabs = max(np.nanmax(np.abs(delta.values)), 1e-6)
    norm = TwoSlopeNorm(vcenter=0, vmin=-vabs, vmax=vabs)
    cmap = _cmap_for(metric)
    im = ax.imshow(delta.values, cmap=cmap, aspect='auto', norm=norm)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f'{v:.2f}' for v in pivot.columns],
                       fontsize=HEATMAP_TICK_FONTSIZE, rotation=45, ha='right')
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f'{v:.2f}' for v in pivot.index], fontsize=HEATMAP_TICK_FONTSIZE)
    ax.set_xlabel('Score Threshold', fontsize=HEATMAP_AXIS_LABEL_FONTSIZE)
    ax.set_ylabel('NMS Threshold', fontsize=HEATMAP_AXIS_LABEL_FONTSIZE)

    if annotate:
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = delta.iloc[i, j]
                if np.isnan(val):
                    continue
                _annotate_cell(ax, j, i, f'{val * 100:+.1f}%')

    _mark_best_cell(ax, delta.values, metric)

    label = METRIC_LABELS.get(metric, metric)
    template = title_template or DEFAULT_DELTA_TITLE_TEMPLATE
    ax.set_title(template.format(metric=label, pipeline=pipeline.upper(), ll=ll,
                                 lsst=f'{lsst_val * 100:.1f}%',
                                 mag_limit=mag_limit, buffer=buffer),
                fontsize=HEATMAP_TITLE_FONTSIZE)

    if own_fig:
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label=f'{label} (DD-LSST)')
        cbar.set_label(f'{label} (DD-LSST)', fontsize=HEATMAP_CBAR_FONTSIZE)
        cbar.ax.tick_params(labelsize=HEATMAP_TICK_FONTSIZE)
        cbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        fig.tight_layout()

    return fig, ax, im


# ============================================================================
# Scatter
# ============================================================================
def plot_purity_vs_completeness(df, pipeline='dd', ll='1.0', ax=None,
                                title=None, show_lsst=True, mark_best=True, pareto=False):
    """Purity vs completeness for one linking length. One LL per figure -- overlaying
    several LLs on one plot reads as noisy, so callers loop this per LL instead."""
    _owns_fig = ax is None
    if _owns_fig:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()

    ll = str(ll)
    comp_col = f'{pipeline}_completeness_{ll}'
    pur_col = f'{pipeline}_purity_{ll}'
    f1_col = f'{pipeline}_f1_{ll}'

    sc = None
    star_label = None
    if comp_col not in df.columns or pur_col not in df.columns:
        print(f'Warning: Missing {comp_col} and/or {pur_col} columns for LL={ll}". Skipping.')
    else:
        sc = ax.scatter(df[comp_col], df[pur_col], c=df[f1_col], cmap='viridis',
                        marker='o', s=30, alpha=0.7, edgecolors='k', linewidth=0.3,
                        label=pipeline.upper())

        if show_lsst:
            lsst_comp_col, lsst_pur_col = f'lsst_completeness_{ll}', f'lsst_purity_{ll}'
            if (lsst_comp_col in df.columns and lsst_pur_col in df.columns
                    and df[lsst_comp_col].notna().any()):
                lsst_comp = df[lsst_comp_col].dropna().iloc[0]
                lsst_pur = df[lsst_pur_col].dropna().iloc[0]
                ax.scatter([lsst_comp], [lsst_pur], color=LSST_COLOR, marker='o',
                          s=80, edgecolors='k', linewidth=0.6, zorder=5, label=LSST_LABEL)
                ax.axvline(lsst_comp, color=LSST_COLOR, ls=':', alpha=0.4)
                ax.axhline(lsst_pur, color=LSST_COLOR, ls=':', alpha=0.4)

        if mark_best and f1_col in df.columns and df[f1_col].notna().any():
            best = df.loc[df[f1_col].idxmax()]
            star_label = (rf'Best F1: $\theta_{{\mathrm{{ROI}}}}$={best["score_thresh"]:.2f}, '
                         rf'$\theta_{{\mathrm{{NMS}}}}$={best["nms_thresh"]:.2f}')
            ax.scatter([best[comp_col]], [best[pur_col]], marker='*', s=220,
                      edgecolors='k', facecolors='gold', linewidth=1.2, zorder=6,
                      label=star_label)

        if pareto:
            pts = df[[comp_col, pur_col]].dropna().values
            if len(pts) > 0:
                front = _pareto_front(pts)
                front = front[np.argsort(front[:, 0])]
                ax.step(front[:, 0], front[:, 1], where='post', color='k',
                       ls='-', linewidth=1.2, alpha=0.7, zorder=4)

    ax.set_xlabel('Completeness', fontsize=12)
    ax.set_ylabel('Purity', fontsize=12)
    ax.set_title(title or f'{pipeline.upper()} Purity vs Completeness (LL={ll}")', fontsize=13)
    leg = ax.legend(fontsize=9)
    if star_label is not None:
        for handle, text in zip(leg.legend_handles, leg.get_texts()):
            if text.get_text() == star_label:
                handle.set_sizes([90])
                break
    ax.grid(True, alpha=0.3)
    if sc is not None:
        plt.colorbar(sc, ax=ax, label='F1')
    if _owns_fig:
        fig.tight_layout()
    return fig, ax


# ============================================================================
# CSV loading (shared)
# ============================================================================
def load_csv_infos(args):
    """Resolve --csv-files against --metrics-dir (unless --absolute-paths), read each,
    and pair it with its --mag-limits / --buffers label. Returns a list of dicts:
    {'path', 'df', 'mag_limit', 'buffer', 'label', 'output_dir'}. Validates
    len(mag_limits) == len(buffers) == len(csv_files)."""
    n = len(args.csv_files)
    if len(args.mag_limits) != n:
        raise ValueError(f'--mag-limits has {len(args.mag_limits)} entries, expected {n} '
                         f'(one per --csv-files)')
    if len(args.buffers) != n:
        raise ValueError(f'--buffers has {len(args.buffers)} entries, expected {n} '
                         f'(one per --csv-files)')

    if len(args.output_dir) == 1:
        output_dirs = args.output_dir * n
    elif len(args.output_dir) == n:
        output_dirs = args.output_dir
    else:
        raise ValueError(f'--output-dir has {len(args.output_dir)} entries, expected 1 or {n}')

    metrics_dir = Path(args.metrics_dir)
    csv_infos = []
    for csv_file, mag_limit, buffer, out_dir in zip(
            args.csv_files, args.mag_limits, args.buffers, output_dirs):
        csv_path = Path(csv_file) if args.absolute_paths else metrics_dir / csv_file
        output_path = Path(out_dir) if args.absolute_paths else metrics_dir / out_dir
        print(f'Loading {csv_path} ({mag_limit} buf{buffer}) ...')
        df = pd.read_csv(csv_path)
        print(f'  {len(df)} rows, {len(df.columns)} columns -> output_dir={output_path}')
        csv_infos.append({
            'path': csv_path,
            'df': df,
            'mag_limit': mag_limit,
            'buffer': buffer,
            'label': f'{mag_limit} buf{buffer}',
            'output_dir': output_path,
        })
    return csv_infos


# ============================================================================
# Driver functions
# ============================================================================
def plot_individual_heatmaps(csv_info, out_dir, lls, pipeline='dd', metrics=None,
                             absolute_paths=False, delta=False, annotate=True,
                             title_template=None, delta_title_template=None):
    """Per-LL layout: {out_dir}/{ll}/individual/ for heatmaps, {out_dir}/{ll}/delta/
    for DD-LSST delta heatmaps (when `delta` is set)."""
    metrics = metrics or METRICS
    df = csv_info['df']
    suffix = f"_{csv_info['mag_limit']}_buf{csv_info['buffer']}" if absolute_paths else ''
    template = title_template or DEFAULT_TITLE_TEMPLATE

    print(f"Individual heatmaps: {csv_info['label']} -> {out_dir} "
         f'({len(metrics)} metrics x {len(lls)} LLs)')
    n_saved = n_delta_saved = 0
    for ll in lls:
        ll_str = str(ll)
        ind_dir = Path(out_dir) / ll_str / 'individual'
        ind_dir.mkdir(parents=True, exist_ok=True)
        delta_dir = None
        if delta:
            delta_dir = Path(out_dir) / ll_str / 'delta'
            delta_dir.mkdir(parents=True, exist_ok=True)

        for metric in metrics:
            col = f'{pipeline}_{metric}_{ll_str}'
            if col not in df.columns:
                print(f'  skip {metric}_{ll_str}: column {col} not in CSV')
                continue
            label = METRIC_LABELS.get(metric, metric)
            title = template.format(metric=label, pipeline=pipeline.upper(), ll=ll_str,
                                    mag_limit=csv_info['mag_limit'], buffer=csv_info['buffer'])
            fig, _, _ = plot_heatmap(df, col, title=title, annotate=annotate)
            out_path = ind_dir / f'{pipeline}_{metric}{suffix}.png'
            fig.savefig(out_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            n_saved += 1

            if delta:
                result = plot_delta_heatmap(df, metric, ll_str, pipeline=pipeline,
                                            annotate=annotate,
                                            title_template=delta_title_template,
                                            mag_limit=csv_info['mag_limit'],
                                            buffer=csv_info['buffer'])
                if result is None:
                    continue
                dfig, _, _ = result
                dfig.savefig(delta_dir / f'{pipeline}_delta_{metric}{suffix}.png',
                            dpi=200, bbox_inches='tight')
                plt.close(dfig)
                n_delta_saved += 1
    print(f'  saved {n_saved} heatmaps' + (f', {n_delta_saved} delta heatmaps' if delta else ''))


def plot_purity_vs_comp_scatter(csv_info, out_dir, lls, pipeline='dd',
                                absolute_paths=False, pareto=False, title_template=None):
    """One purity-vs-completeness figure per LL, saved directly under {out_dir}/{ll}/."""
    df = csv_info['df']
    suffix = f"_{csv_info['mag_limit']}_buf{csv_info['buffer']}" if absolute_paths else ''
    template = title_template or DEFAULT_SCATTER_TITLE_TEMPLATE

    for ll in lls:
        ll_str = str(ll)
        ll_dir = Path(out_dir) / ll_str
        ll_dir.mkdir(parents=True, exist_ok=True)

        title = template.format(label=csv_info['label'], pipeline=pipeline.upper(), ll=ll_str,
                                mag_limit=csv_info['mag_limit'], buffer=csv_info['buffer'])
        fig, _ = plot_purity_vs_completeness(df, pipeline=pipeline, ll=ll_str,
                                             title=title, pareto=pareto)
        out_path = ll_dir / f'{pipeline}_purity_vs_completeness{suffix}.png'
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Purity vs completeness: {csv_info["label"]} LL={ll_str}" -> {out_path}')


def plot_metric_pair_heatmaps(csv_info, out_dir, lls, pipeline='dd', metric_pair=None,
                              absolute_paths=False, annotate=False, title_template=None):
    """One combined two-panel figure per LL: {out_dir}/{ll}/{pipeline}_pair_{m1}_{m2}.png.
    Runs independently of --no-individual so a single metric pair can be regenerated
    without redoing the full 12-metric x delta x scatter sweep."""
    if not metric_pair:
        return
    m1, m2 = metric_pair
    df = csv_info['df']
    suffix = f"_{csv_info['mag_limit']}_buf{csv_info['buffer']}" if absolute_paths else ''
    template = title_template or DEFAULT_TITLE_TEMPLATE
    combined_label = f'{METRIC_LABELS.get(m1, m1)} & {METRIC_LABELS.get(m2, m2)}'

    for ll in lls:
        ll_str = str(ll)
        cols = [f'{pipeline}_{m}_{ll_str}' for m in (m1, m2)]
        if not all(c in df.columns for c in cols):
            print(f'  skip metric pair {m1}/{m2}_{ll_str}: missing column(s)')
            continue
        suptitle = template.format(metric=combined_label, pipeline=pipeline.upper(),
                                   ll=ll_str, mag_limit=csv_info['mag_limit'],
                                   buffer=csv_info['buffer'])
        ll_dir = Path(out_dir) / ll_str
        ll_dir.mkdir(parents=True, exist_ok=True)
        out_path = ll_dir / f'{pipeline}_pair_{m1}_{m2}{suffix}.png'
        plot_metric_pair_heatmap(df, cols, output_path=out_path, suptitle=suptitle, annotate=annotate)
        print(f'Metric pair heatmap: {csv_info["label"]} LL={ll_str}" -> {out_path}')


def plot_comparison_grids(csv_infos, out_dir, pipeline='dd', metrics=None, lls=None,
                          title_template=None):
    metrics = metrics or METRICS
    lls = lls or [1.0]
    template = title_template or DEFAULT_TITLE_TEMPLATE
    comp_dir = Path(out_dir) / 'comparison'
    comp_dir.mkdir(parents=True, exist_ok=True)
    dfs = [ci['df'] for ci in csv_infos]
    labels = [ci['label'] for ci in csv_infos]

    print(f'Comparison grids: {len(dfs)} configs ({", ".join(labels)}) -> {comp_dir} '
         f'({len(metrics)} metrics x {len(lls)} LLs)')
    n_saved = 0
    for ll in lls:
        ll_str = str(ll)
        for metric in metrics:
            col = f'{pipeline}_{metric}_{ll_str}'
            if not any(col in df.columns for df in dfs):
                print(f'  skip {metric}_{ll_str}: column {col} not in any config')
                continue
            label = METRIC_LABELS.get(metric, metric)
            title = template.format(metric=label, pipeline=pipeline.upper(), ll=ll_str)
            out_path = comp_dir / f'compare_{pipeline}_{metric}_{ll_str}.png'
            fig = plot_heatmap_comparison(dfs, labels, col, output_path=out_path, title=title)
            if fig is None:
                print(f'  skip {metric}_{ll_str}: column {col} present but all-NaN')
                continue
            n_saved += 1
    print(f'  saved {n_saved} comparison grids')


def plot_purity_comp_comparison(csv_infos, out_dir, pipeline='dd', lls=None):
    """One side-by-side comparison figure per LL: compare_{pipeline}_purity_vs_completeness_{ll}.png"""
    n = len(csv_infos)
    if n == 0:
        return []
    lls = lls or [1.0]
    comp_dir = Path(out_dir) / 'comparison'
    comp_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for ll in lls:
        ll_str = str(ll)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
        for idx, csv_info in enumerate(csv_infos):
            plot_purity_vs_completeness(csv_info['df'], pipeline=pipeline, ll=ll_str,
                                        ax=axes[0, idx], title=csv_info['label'])
        fig.tight_layout()
        out_path = comp_dir / f'compare_{pipeline}_purity_vs_completeness_{ll_str}.png'
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        saved.append(out_path)
        print(f'Purity vs completeness comparison: {n} configs, LL={ll_str}" -> {out_path}')
    return saved


# ============================================================================
# CLI
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot heatmaps from grid search CSV results')
    parser.add_argument('--metrics-dir', required=True,
                        help='Base directory for relative CSV and output paths')
    parser.add_argument('--csv-files', nargs='+', default=['gold_buf2/gs_metrics.csv'],
                        help='Paths to CSV files (relative to --metrics-dir by default) '
                             "(default: gold_buf2/gs_metrics.csv)")
    parser.add_argument('--output-dir', nargs='+', default=['gold_buf2/gs_hmaps'],
                        help='Output directory/directories (one per CSV file, or single dir for all) '
                             "(default: gold_buf2/gs_hmaps)")
    parser.add_argument('--mag-limits', nargs='+', default=['gold'],
                        help='Magnitude limits for each CSV file (e.g., gold power_law nominal) '
                             "(default: gold)")
    parser.add_argument('--buffers', nargs='+', type=int, default=[2],
                        help='Buffer values for each CSV file (e.g., 0 1 2) (default: 2)')
    parser.add_argument('--absolute-paths', action='store_true',
                        help='Treat CSV and output paths as absolute instead of relative to --metrics-dir')
    parser.add_argument('--pipeline', default='dd', choices=['dd', 'lsst'],
                        help='Which pipeline to plot metrics for (default: dd)')
    parser.add_argument('--metrics', nargs='+', default=None,
                        help='Specific metrics to plot (e.g., completeness purity f1). '
                             'If not specified, plots all 12 metrics from blend_metrics + f1')
    parser.add_argument('--linking-lengths', nargs='+', type=float,
                        default=[0.5, 1.0],
                        help='FOF linking lengths in arcseconds (default: 0.5 1.0)')
    parser.add_argument('--no-individual', action='store_true',
                        help='Skip generating individual heatmaps for each CSV file')
    parser.add_argument('--metric-pair', nargs=2, default=None, metavar=('METRIC1', 'METRIC2'),
                        help='Emit one combined two-panel figure (per CSV file x LL) with '
                             'these two metrics side by side, sharing a y-axis. Runs '
                             'independently of --no-individual/--comparison, e.g. pass '
                             '--no-individual --no-comparison --metric-pair unrec_blend_frac f1 '
                             'to regenerate just this figure.')
    parser.add_argument('--comparison', action=argparse.BooleanOptionalAction, default=False,
                        help='Generate comparison grid figures across CSV files (default: False; '
                             'pass --comparison to enable, --no-comparison to force off)')
    parser.add_argument('--delta', action=argparse.BooleanOptionalAction, default=True,
                        help='Also emit DD-LSST delta heatmaps in individual mode '
                             '(default: True; pass --no-delta to turn off)')
    parser.add_argument('--pareto', action=argparse.BooleanOptionalAction, default=True,
                        help='Outline the Pareto front in the purity-vs-completeness scatter '
                             '(default: True; pass --no-pareto to turn off)')
    parser.add_argument('--annotate', action=argparse.BooleanOptionalAction, default=True,
                        help='Draw per-cell numeric labels on heatmaps and delta heatmaps '
                             '(default: True; pass --no-annotate to turn off)')
    parser.add_argument('--title-template', default=None,
                        help='Heatmap title format string. Placeholders: {metric}, {pipeline}, '
                             '{ll}, and (individual mode only) {mag_limit}, {buffer}. '
                             '(default: "%s")' % DEFAULT_TITLE_TEMPLATE)
    parser.add_argument('--delta-title-template', default=None,
                        help='Delta heatmap title format string. Placeholders: {metric}, '
                             '{pipeline}, {ll}, {lsst} (LSST reference, pre-formatted as a '
                             'percentage string), and (individual mode only) {mag_limit}, '
                             '{buffer}. (default: "%s")' % DEFAULT_DELTA_TITLE_TEMPLATE)
    parser.add_argument('--scatter-title-template', default=None,
                        help='Purity-vs-completeness scatter title format string. '
                             'Placeholders: {label} (e.g. "gold buf2"), {pipeline}, {ll}, '
                             '{mag_limit}, {buffer}. (default: "%s")'
                             % DEFAULT_SCATTER_TITLE_TEMPLATE)
    return parser.parse_args()


def main():
    t0 = time.time()
    args = parse_args()
    lls = args.linking_lengths
    metrics = args.metrics or METRICS

    if args.metric_pair:
        # metric-pair is a self-contained mode for one paper figure: single LL,
        # no individual/comparison sweep, no delta/pareto (not used by
        # plot_metric_pair_heatmap) or annotate clutter.
        lls = [1.0]
        args.no_individual = True
        args.comparison = False
        args.delta = False
        args.pareto = False
        args.annotate = False

    print('=' * 70)
    print('PLOT GRID SEARCH')
    print('=' * 70)
    print(f'  metrics-dir:  {args.metrics_dir}')
    print(f'  csv-files:    {args.csv_files}')
    print(f'  pipeline:     {args.pipeline}')
    print(f'  linking-lengths: {lls}')
    if args.metric_pair:
        print(f'  mode:         metric-pair ({args.metric_pair[0]} vs {args.metric_pair[1]})')
    else:
        print(f'  metrics:      {len(metrics)} ({"all" if args.metrics is None else "custom"})')
        print(f'  individual:   {not args.no_individual}   comparison: {args.comparison}')
        print(f'  delta:        {args.delta}   pareto: {args.pareto}   annotate: {args.annotate}')

    csv_infos = load_csv_infos(args)

    if not args.no_individual:
        for csv_info in csv_infos:
            out_dir = csv_info['output_dir']
            plot_individual_heatmaps(csv_info, out_dir, lls, pipeline=args.pipeline,
                                     metrics=metrics, absolute_paths=args.absolute_paths,
                                     delta=args.delta, annotate=args.annotate,
                                     title_template=args.title_template,
                                     delta_title_template=args.delta_title_template)
            plot_purity_vs_comp_scatter(csv_info, out_dir, lls, pipeline=args.pipeline,
                                        absolute_paths=args.absolute_paths,
                                        pareto=args.pareto,
                                        title_template=args.scatter_title_template)

    if args.metric_pair:
        for csv_info in csv_infos:
            plot_metric_pair_heatmaps(csv_info, csv_info['output_dir'], lls,
                                      pipeline=args.pipeline, metric_pair=args.metric_pair,
                                      absolute_paths=args.absolute_paths, annotate=args.annotate,
                                      title_template=args.title_template)

    if args.comparison:
        out_dir = csv_infos[0]['output_dir']
        plot_comparison_grids(csv_infos, out_dir, pipeline=args.pipeline,
                              metrics=metrics, lls=lls, title_template=args.title_template)
        plot_purity_comp_comparison(csv_infos, out_dir, pipeline=args.pipeline, lls=lls)

    print('=' * 70)
    print(f'Done in {time.time() - t0:.1f}s')
    print('=' * 70)


if __name__ == '__main__':
    main()
