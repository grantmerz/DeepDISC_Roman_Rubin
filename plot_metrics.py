"""
plot_metrics.py — All plotting functions for analysis

Import what you need:
    from plot_metrics import plot_binned_metric, plot_completeness_comparison, ...

Two categories of plots:

    A) Classified analysis DataFrame plots (from class_*.parquet files)

    B) Grid search CSV plots (from metrics_*.csv files):
        - plot_heatmap()                    — single metric heatmap
        - plot_heatmap_comparison()         — side-by-side heatmaps across configs

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from pathlib import Path

# ============================================================================
# Style constants
# ============================================================================
PIPELINE_COLORS = {'dd': 'tab:blue', 'lsst': 'tab:orange'}
LL_STYLES = {'1.0': '-', '2.0': '--', '0.5': ':'}

TRUTH_SCENARIO_COLORS = {
    'matched_isolated':       '#2ecc71',
    'matched_resolved_blend': '#27ae60',
    'matched_unrec_blend':    '#f39c12',
    'blended_away':           '#e74c3c',
    'missed':                 '#95a5a6',
    'missed_in_blend':        '#7f8c8d',
}

DET_SCENARIO_COLORS = {
    'matched_isolated':       '#2ecc71',
    'matched_resolved_blend': '#27ae60',
    'matched_unrec_blend':    '#f39c12',
    'shred_blend':            '#e67e22',
    'shred_isolated':         '#d35400',
    'spurious':               '#e74c3c',
}

# Grid search heatmap labels
METRIC_LABELS = {
    'completeness': 'Completeness',
    'purity': 'Purity',
    'f1': 'F1 Score',
    'blend_loss_frac': 'Blend Loss Fraction',
    'unrec_blend_frac_total': 'Unrec Blend Frac (all truths)',
    'unrec_blend_frac_blended': 'Unrec Blend Frac (blended only)',
    'unrec_blend_det_frac_total': 'Unrec Blend Det Frac (all dets)',
    'unrec_blend_det_frac_blended': 'Unrec Blend Det Frac (blended dets)',
    'resolved_rate': 'Resolved Blend Rate',
    'shred_frac': 'Shredding Fraction (truths)',
    'shred_det_frac': 'Shred Det Fraction',
    'spurious_frac': 'Spurious Fraction',
    'missed_frac': 'Missed Fraction',
}
LOWER_IS_BETTER = {
    'blend_loss_frac', 'unrec_blend_frac_total', 'unrec_blend_frac_blended',
    'unrec_blend_det_frac_total', 'unrec_blend_det_frac_blended',
    'shred_frac', 'shred_det_frac', 'spurious_frac', 'missed_frac',
}


# ============================================================================
# B) Grid search CSV plots
# ============================================================================
def plot_heatmap(df, col, ax=None, title=None, cmap=None,
                 fmt='.3f', vmin=None, vmax=None, annotate=True):
    """
    Plot a heatmap of a metric from grid search results.

    Parameters
    ----------
    df : pd.DataFrame
        Grid search CSV loaded as DataFrame.
    col : str
        Column to plot (e.g. 'dd_completeness_1.0').
    ax : matplotlib.axes.Axes, optional
    title : str, optional
    cmap : str, optional
        Auto-selects based on metric if None.
    fmt : str
        Annotation format.
    vmin, vmax : float, optional
    annotate : bool

    Returns
    -------
    fig, ax, im
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(
            max(6, df['score_thresh'].nunique() * 0.9 + 2),
            max(4, df['nms_thresh'].nunique() * 0.7 + 1.5)))
    else:
        fig = ax.get_figure()

    pivot = df.pivot_table(
        values=col, index='nms_thresh', columns='score_thresh', aggfunc='mean')

    # Auto-select colormap
    if cmap is None:
        metric_name = col.rsplit('_', 1)[0]  # strip linking length
        metric_name = '_'.join(metric_name.split('_')[1:])  # strip prefix
        cmap = 'RdYlGn_r' if metric_name in LOWER_IS_BETTER else 'RdYlGn'

    im = ax.imshow(pivot.values, cmap=cmap, aspect='auto',
                   vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f'{v:.2f}' for v in pivot.columns],
                       fontsize=8, rotation=45, ha='right')
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f'{v:.2f}' for v in pivot.index], fontsize=8)
    ax.set_xlabel('Score Threshold', fontsize=10)
    ax.set_ylabel('NMS Threshold', fontsize=10)

    if annotate:
        eff_vmin = vmin if vmin is not None else np.nanmin(pivot.values)
        eff_vmax = vmax if vmax is not None else np.nanmax(pivot.values)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.iloc[i, j]
                if np.isnan(val):
                    continue
                frac = (val - eff_vmin) / (eff_vmax - eff_vmin + 1e-9)
                color = 'white' if frac > 0.6 else 'black'
                ax.text(j, i, f'{val:{fmt}}', ha='center', va='center',
                        fontsize=7, color=color)

    if title:
        ax.set_title(title, fontsize=11)

    if own_fig:
        plt.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()

    return fig, ax, im


def plot_heatmap_comparison(dfs, labels, col, output_path=None, title=None,
                            cmap=None):
    """
    Side-by-side heatmaps of the same metric across multiple configs.

    Parameters
    ----------
    dfs : list of pd.DataFrame
        Grid search DataFrames (one per config).
    labels : list of str
        Panel titles (one per config).
    col : str
        Column to plot.
    output_path : str or Path, optional
    title : str, optional
        Suptitle.
    """
    n = len(dfs)
    if n == 0:
        return None

    # Shared color range
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
                             squeeze=False)

    last_im = None
    for idx, (df, label) in enumerate(zip(dfs, labels)):
        ax = axes[0, idx]
        if col not in df.columns:
            ax.set_visible(False)
            continue
        _, _, im = plot_heatmap(df, col, ax=ax, title=label, cmap=cmap,
                                vmin=vmin, vmax=vmax, annotate=(n <= 4))
        last_im = im
        if idx > 0:
            ax.set_ylabel('')
            ax.set_yticklabels([])

    if last_im is not None:
        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
        fig.colorbar(last_im, cax=cbar_ax)

    if title:
        fig.suptitle(title, fontsize=13, y=1.02)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig