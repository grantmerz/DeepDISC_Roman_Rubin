"""
plot_losses.py — Plot training and validation losses from saved loss JSON files.

Reads {run_dir}/{run_name}_losses.json and {run_dir}/{run_name}_val_losses.json
and saves plots under {run_dir}/loss_plots/.

Plots saved:
  - One plot per individual loss key (train + val overlaid if available)
  - All losses on one figure
  - Detection + keypoint losses on one figure
  - Total detection loss + contrastive loss on one figure

Usage:
    python plot_losses.py --run-dir ~/lsst_runs/clip5_30k_4h200_bs64_ep50
    python plot_losses.py --run-dir ~/lsst_runs/clip5_30k_4h200_bs64_ep50 --contrastive-xmin 7100 --contrastive-ymin 0.3 --contrastive-ymax 1.6
    python plot_losses.py --run-dir ~/lsst_runs/clip5_flatten_30k_4h200_bs64_ep15_resume --only-all-losses --all-losses-ymin 4
    
    python plot_losses.py --run-dir ~/lsst_runs/lsst5_30k_4h200_bs192_ep50
    python plot_losses.py --run-dir ~/lsst_runs/lsst5_30k_4h200_bs192_ep50 --all-losses-xmin 2365 --all-losses-ymax 5.2
    
    python plot_losses.py --run-dir ~/lsst_runs/clip5_30k_4h200_bs64_ep50 --smooth 10
    python plot_losses.py --run-dir ~/lsst_runs/clip5_30k_4h200_bs64_ep50 --no-val
"""

import argparse
import json
import os
import re

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ============================================================================
# Loss key groupings
# ============================================================================

# Detection losses: RPN + ROI head box/mask losses across all Cascade stages
# Matches: loss_cls_stage0/1/2, loss_box_reg_stage0/1/2, loss_mask,
#          loss_rpn_cls, loss_rpn_loc
DETECTION_LOSS_PATTERNS = [
    r'^loss_rpn_',
    r'^loss_cls',
    r'^loss_box_reg',
    r'^loss_mask$',
]

# Keypoint losses
KEYPOINT_LOSS_PATTERNS = [
    r'^loss_keypoint',
]

# Contrastive loss — matches roi_contrastive_loss from the CLIP run
CONTRASTIVE_LOSS_PATTERNS = [
    r'^roi_contrastive_loss$',
    r'^loss_contrastive',
    r'^contrastive_loss',
    r'^clip_loss',
]

# Keys to exclude from detection/keypoint/contrastive groups
# (e.g. logit_scale is a parameter, not a loss)
NON_LOSS_KEYS = {'logit_scale'}

# Human-readable labels for known loss keys
LOSS_LABELS = {
    # Cascade stage losses
    'loss_cls_stage0':             'Cls Loss (Stage 0)',
    'loss_cls_stage1':             'Cls Loss (Stage 1)',
    'loss_cls_stage2':             'Cls Loss (Stage 2)',
    'loss_box_reg_stage0':         'Box Reg Loss (Stage 0)',
    'loss_box_reg_stage1':         'Box Reg Loss (Stage 1)',
    'loss_box_reg_stage2':         'Box Reg Loss (Stage 2)',
    # Mask / keypoint
    'loss_mask':                   'Mask Loss',
    'loss_keypoint':               'Keypoint Loss',
    # RPN
    'loss_rpn_cls':                'RPN Cls Loss',
    'loss_rpn_loc':                'RPN Loc Loss',
    # Contrastive
    'roi_contrastive_loss':        'ROI Contrastive Loss',
    'loss_contrastive':            'Contrastive Loss',
    'contrastive_loss':            'Contrastive Loss',
    'clip_loss':                   'CLIP Loss',
    # Other
    'logit_scale':                 'Logit Scale',
}

# Plot style
TRAIN_COLOR  = '#2196F3'   # blue
VAL_COLOR    = '#F44336'   # red
ALPHA_RAW    = 0.25        # raw trace alpha
ALPHA_SMOOTH = 1.0         # smoothed trace alpha
FIGSIZE_SINGLE = (7, 4)
FIGSIZE_MULTI  = (14, 10)
FIGSIZE_GROUP  = (12, 7)
DPI = 150


# ============================================================================
# Helpers
# ============================================================================

def load_loss_json(path):
    """Load loss JSON → sorted list of (iter, loss_dict) tuples."""
    with open(path) as f:
        data = json.load(f)
    # Keys are string iteration numbers
    pairs = sorted((int(k), v) for k, v in data.items())
    return pairs


def load_loss_npy(path):
    """Load legacy .npy loss file → sorted list of (iter, loss_dict) tuples.

    The old format is a flat float32 array of total loss per iteration,
    saved as np.save(..., trainer.lossList). Iteration numbers are inferred
    as 1-based sequential indices (1, 2, 3, ..., N).
    Returns the same format as load_loss_json so the rest of the pipeline
    is unaffected: [(iter, {'total_loss': val}), ...]
    """
    arr = np.load(path).astype(float)
    return [(i + 1, {'total_loss': float(v)}) for i, v in enumerate(arr)]


def extract_series(pairs):
    """Convert [(iter, {key: val})] → {key: (iters_array, values_array)}."""
    if not pairs:
        return {}
    all_keys = set()
    for _, d in pairs:
        all_keys.update(d.keys())

    series = {k: ([], []) for k in all_keys}
    for it, d in pairs:
        for k in all_keys:
            if k in d:
                series[k][0].append(it)
                series[k][1].append(d[k])

    return {k: (np.array(iters), np.array(vals)) for k, (iters, vals) in series.items()}


def smooth(values, window):
    """Simple moving average smoothing."""
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode='valid')


def smooth_iters(iters, window):
    """Corresponding iteration indices after convolution shortening."""
    if window <= 1 or len(iters) < window:
        return iters
    return iters[window - 1:]


def apply_xlim(ax, xlim, x_margin=0.03, y_margin=0.05):
    """Apply xlim with padding and rescale y to only the visible data range.

    x_margin: fractional padding added to whichever x boundary the user set,
              matching matplotlib's default axis margin feel.
    y_margin: fractional padding above/below the visible y data range.
    """
    xmin_req, xmax_req = xlim
    if xmin_req is None and xmax_req is None:
        return

    # Compute x span from all plotted data to derive padding
    all_x = np.concatenate([line.get_xdata() for line in ax.get_lines()
                             if len(line.get_xdata()) > 0])
    if len(all_x) == 0:
        return
    x_span = all_x.max() - all_x.min()
    x_pad  = x_span * x_margin

    # Apply xlim with padding on the user-specified side(s).
    # For the unspecified side, pin to the actual data min/max (+ small pad)
    # rather than letting matplotlib autoscale, which adds its own 5% margin
    # and can push the left bound negative when data starts near 0.
    data_min = all_x.min()
    data_max = all_x.max()
    lo = (xmin_req - x_pad) if xmin_req is not None else max(0, data_min - x_pad)
    hi = (xmax_req + x_pad) if xmax_req is not None else (data_max + x_pad)
    ax.set_xlim(lo, hi)

    # Rescale y to data visible within the padded xlim
    xlo, xhi = ax.get_xlim()
    ymin, ymax = np.inf, -np.inf
    for line in ax.get_lines():
        xd = line.get_xdata()
        yd = line.get_ydata()
        if len(xd) == 0 or len(yd) == 0:
            continue
        mask = (xd >= xlo) & (xd <= xhi)
        if not np.any(mask):
            continue
        visible_y = yd[mask]
        visible_y = visible_y[np.isfinite(visible_y)]
        if len(visible_y) == 0:
            continue
        ymin = min(ymin, visible_y.min())
        ymax = max(ymax, visible_y.max())
    if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
        pad = (ymax - ymin) * y_margin
        ax.set_ylim(max(0, ymin - pad), ymax + pad)


def apply_ylim(ax, ylim):
    """Apply explicit y-axis limits if provided by the user."""
    ymin, ymax = ylim
    if ymin is not None or ymax is not None:
        lo = ymin if ymin is not None else ax.get_ylim()[0]
        hi = ymax if ymax is not None else ax.get_ylim()[1]
        ax.set_ylim(lo, hi)


def matches_any(key, patterns):
    return any(re.match(p, key) for p in patterns)


def sum_series(series_dict, key_list):
    """Sum multiple loss series into one, aligning on shared iterations."""
    all_iters = None
    for k in key_list:
        if k not in series_dict:
            continue
        iters, _ = series_dict[k]
        all_iters = iters if all_iters is None else np.intersect1d(all_iters, iters)
    if all_iters is None or len(all_iters) == 0:
        return None, None
    total = np.zeros(len(all_iters))
    for k in key_list:
        if k not in series_dict:
            continue
        iters, vals = series_dict[k]
        total += vals[np.isin(iters, all_iters)]
    return all_iters, total


def label_for(key):
    return LOSS_LABELS.get(key, key.replace('_', ' ').title())


def savefig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    print(f"  Saved → {path}")
    plt.close(fig)


# ============================================================================
# Plotting primitives
# ============================================================================

def plot_single_loss(ax, train_series, val_series, key, smooth_window, title=None, xlim=(None, None), ylim=(None, None)):
    """Plot one loss key (train + optional val) onto an axis."""
    plotted = False

    if key in train_series:
        iters, vals = train_series[key]
        ax.plot(iters, vals, color=TRAIN_COLOR, alpha=ALPHA_RAW, linewidth=0.8)
        if smooth_window > 1:
            sv = smooth(vals, smooth_window)
            si = smooth_iters(iters, smooth_window)
            ax.plot(si, sv, color=TRAIN_COLOR, alpha=ALPHA_SMOOTH,
                    linewidth=1.8, label='Train')
        else:
            ax.lines[-1].set_alpha(ALPHA_SMOOTH)
            ax.lines[-1].set_linewidth(1.8)
            ax.lines[-1].set_label('Train')
        plotted = True

    if val_series and key in val_series:
        iters, vals = val_series[key]
        ax.plot(iters, vals, color=VAL_COLOR, alpha=ALPHA_RAW, linewidth=0.8)
        if smooth_window > 1:
            sv = smooth(vals, smooth_window)
            si = smooth_iters(iters, smooth_window)
            ax.plot(si, sv, color=VAL_COLOR, alpha=ALPHA_SMOOTH,
                    linewidth=1.8, label='Val')
        else:
            ax.lines[-1].set_alpha(ALPHA_SMOOTH)
            ax.lines[-1].set_linewidth(1.8)
            ax.lines[-1].set_label('Val')
        plotted = True

    if not plotted:
        return

    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title(title or label_for(key), fontsize=12)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.grid(True, alpha=0.3)
    apply_xlim(ax, xlim)
    apply_ylim(ax, ylim)
    if (key in (train_series or {})) and (val_series and key in val_series):
        ax.legend(fontsize=9)


# ============================================================================
# Main plot functions
# ============================================================================

def plot_all_individual(train_series, val_series, keys, smooth_window, out_dir, xlim=(None, None), ylim=(None, None)):
    """One figure per loss key."""
    print("\n[1] Individual loss plots...")
    for key in sorted(keys):
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        plot_single_loss(ax, train_series, val_series, key, smooth_window, xlim=xlim, ylim=ylim)
        fig.tight_layout()
        fname = key.replace('/', '_').replace(' ', '_') + '.png'
        savefig(fig, os.path.join(out_dir, 'individual', fname))


def plot_overlaid(ax, train_series, val_series, keys, smooth_window, cmap, xlim=(None, None), ylim=(None, None)):
    """Plot multiple loss keys overlaid on a single axes, each with its own color.
    Drawing order: all raw traces (lowest zorder), then smoothed train, then smoothed val
    on top so val lines are never buried under the raw cloud.
    """
    # Pass 1 — raw faint traces (train + val), zorder=1
    for i, key in enumerate(keys):
        color = cmap(i)
        if key in train_series:
            iters, vals = train_series[key]
            ax.plot(iters, vals, color=color, alpha=ALPHA_RAW, linewidth=0.6, zorder=1)
        if val_series and key in val_series:
            iters, vals = val_series[key]
            ax.plot(iters, vals, color=color, alpha=ALPHA_RAW, linewidth=0.6,
                    linestyle='--', zorder=1)

    # Pass 2 — smoothed train lines, zorder=2
    for i, key in enumerate(keys):
        color = cmap(i)
        lbl   = label_for(key)
        if key in train_series:
            iters, vals = train_series[key]
            if smooth_window > 1 and len(vals) >= smooth_window:
                ax.plot(smooth_iters(iters, smooth_window), smooth(vals, smooth_window),
                        color=color, linewidth=1.8, label=lbl, zorder=2)
            else:
                ax.plot(iters, vals, color=color, linewidth=1.8, label=lbl, zorder=2)

    # Pass 3 — smoothed val lines on top, zorder=3, slightly darker color
    if val_series:
        for i, key in enumerate(keys):
            if key not in val_series:
                continue
            color = cmap(i)
            lbl   = label_for(key)
            iters, vals = val_series[key]
            if smooth_window > 1 and len(vals) >= smooth_window:
                ax.plot(smooth_iters(iters, smooth_window), smooth(vals, smooth_window),
                        color=color, linewidth=2.2, linestyle='--',
                        label=f'{lbl} (val)', zorder=3)
            else:
                ax.plot(iters, vals, color=color, linewidth=2.2, linestyle='--',
                        label=f'{lbl} (val)', zorder=3)

    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.grid(True, alpha=0.3)
    # Note: apply_xlim/apply_ylim are intentionally NOT called here.
    # Callers (plot_all_together, plot_detection_keypoint) apply axis limits
    # after all curves — including summaries — have been drawn.


def plot_all_together(train_series, val_series, keys, smooth_window, out_dir, run_name, xlim=(None, None), ylim=(None, None)):
    """All losses overlaid on one axes, plus total_loss and total_det_loss summary curves."""
    print("\n[2] All losses on one figure...")
    keys_sorted = sorted(keys)

    # tab20 for the individual keys, then two bold summary curves on top
    cmap = plt.colormaps['tab20'].resampled(len(keys_sorted))

    fig, ax = plt.subplots(figsize=(12, 5))
    plot_overlaid(ax, train_series, val_series, keys_sorted, smooth_window, cmap, xlim=xlim, ylim=ylim)

    # ---- summary curves ----
    det_keys = [k for k in keys if matches_any(k, DETECTION_LOSS_PATTERNS)]
    all_loss_keys = [k for k in keys
                     if not matches_any(k, CONTRASTIVE_LOSS_PATTERNS)]  # everything except contrastive

    SUMMARY_COLORS = {'total_loss': '#000000', 'total_det_loss': '#795548'}

    for label, key_list, color in [
        ('Total Loss',           all_loss_keys, SUMMARY_COLORS['total_loss']),
        ('Total Detection Loss', det_keys,      SUMMARY_COLORS['total_det_loss']),
    ]:
        # Skip if the key_list is just ['total_loss'] and this means the data
        # came from a .npy file and the summary would duplicate the
        # only curve already plotted by plot_overlaid
        if key_list == ['total_loss']:
            continue
        ti, tv = sum_series(train_series, key_list)
        if ti is None:
            continue
        ax.plot(ti, tv, color=color, alpha=ALPHA_RAW, linewidth=0.8)
        if smooth_window > 1 and len(tv) >= smooth_window:
            ax.plot(smooth_iters(ti, smooth_window), smooth(tv, smooth_window),
                    color=color, linewidth=2.5, linestyle='-',
                    label=label, zorder=5)
        else:
            ax.lines[-1].set_linewidth(2.5)
            ax.lines[-1].set_label(label)
            ax.lines[-1].set_zorder(5)

        if val_series:
            vi, vv = sum_series(val_series, key_list)
            if vi is not None:
                ax.plot(vi, vv, color=color, alpha=ALPHA_RAW, linewidth=0.8, linestyle='--')
                if smooth_window > 1 and len(vv) >= smooth_window:
                    ax.plot(smooth_iters(vi, smooth_window), smooth(vv, smooth_window),
                            color=color, linewidth=2.5, linestyle='--',
                            label=f'{label} (val)', zorder=5)
                else:
                    ax.lines[-1].set_linewidth(2.5)
                    ax.lines[-1].set_label(f'{label} (val)')
                    ax.lines[-1].set_zorder(5)

    ax.set_title(f'All Losses — {run_name}', fontsize=13)
    apply_xlim(ax, xlim)
    apply_ylim(ax, ylim)
    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1),
              borderaxespad=0, ncol=1)
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, 'all_losses.png'))


def plot_detection_keypoint(train_series, val_series, keys, smooth_window, out_dir, run_name, xlim=(None, None), ylim=(None, None)):
    """All detection + keypoint losses on a single overlaid axes."""
    print("\n[3] Detection + keypoint losses figure...")
    det_kp_keys = sorted(
        k for k in keys
        if matches_any(k, DETECTION_LOSS_PATTERNS + KEYPOINT_LOSS_PATTERNS)
    )

    if not det_kp_keys:
        print("  No detection/keypoint loss keys found — skipping.")
        return

    cmap = plt.colormaps['tab20'].resampled(len(det_kp_keys))
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_overlaid(ax, train_series, val_series, det_kp_keys, smooth_window, cmap, xlim=xlim, ylim=ylim)
    ax.set_title(f'Detection & Keypoint Losses — {run_name}', fontsize=13)
    apply_xlim(ax, xlim)
    apply_ylim(ax, ylim)
    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1),
              borderaxespad=0, ncol=1)
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, 'detection_keypoint_losses.png'))


def plot_total_det_vs_contrastive(train_series, val_series, keys, smooth_window, out_dir, run_name, xlim=(None, None), ylim=(None, None)):
    """Total detection loss (sum of all det losses) + contrastive loss side by side."""
    print("\n[4] Total detection loss + contrastive loss figure...")

    det_keys = [k for k in keys if matches_any(k, DETECTION_LOSS_PATTERNS)]
    ctr_keys = [k for k in keys if matches_any(k, CONTRASTIVE_LOSS_PATTERNS)]

    has_det = bool(det_keys)
    has_ctr = bool(ctr_keys)

    if not has_det and not has_ctr:
        print("  Neither detection nor contrastive loss keys found — skipping.")
        return

    # Build (label, train_iters, train_vals, val_iters, val_vals) for each curve
    curves = []

    if has_det:
        ti, tv = sum_series(train_series, det_keys)
        vi, vv = sum_series(val_series, det_keys) if val_series else (None, None)
        if ti is not None:
            curves.append(('Total Detection Loss', ti, tv, vi, vv))

    if has_ctr:
        for ck in ctr_keys:
            ti = train_series[ck][0] if ck in train_series else None
            tv = train_series[ck][1] if ck in train_series else None
            vi = val_series[ck][0] if (val_series and ck in val_series) else None
            vv = val_series[ck][1] if (val_series and ck in val_series) else None
            curves.append((label_for(ck), ti, tv, vi, vv))

    # Two curves at most (total det + contrastive), use fixed colors
    curve_colors = ['#2196F3', '#E91E63', '#FF9800', '#4CAF50']

    fig, ax = plt.subplots(figsize=(9, 5))

    # Pass 1 — raw faint traces, zorder=1
    for i, (lbl, ti, tv, vi, vv) in enumerate(curves):
        color = curve_colors[i % len(curve_colors)]
        if ti is not None:
            ax.plot(ti, tv, color=color, alpha=ALPHA_RAW, linewidth=0.6, zorder=1)
        if vi is not None and vv is not None:
            ax.plot(vi, vv, color=color, alpha=ALPHA_RAW, linewidth=0.6,
                    linestyle='--', zorder=1)

    # Pass 2 — smoothed train lines, zorder=2
    for i, (lbl, ti, tv, vi, vv) in enumerate(curves):
        color = curve_colors[i % len(curve_colors)]
        if ti is not None:
            if smooth_window > 1 and len(tv) >= smooth_window:
                ax.plot(smooth_iters(ti, smooth_window), smooth(tv, smooth_window),
                        color=color, linewidth=2.0, label=lbl, zorder=2)
            else:
                ax.plot(ti, tv, color=color, linewidth=2.0, label=lbl, zorder=2)

    # Pass 3 — smoothed val lines on top, zorder=3
    for i, (lbl, ti, tv, vi, vv) in enumerate(curves):
        color = curve_colors[i % len(curve_colors)]
        if vi is not None and vv is not None:
            if smooth_window > 1 and len(vv) >= smooth_window:
                ax.plot(smooth_iters(vi, smooth_window), smooth(vv, smooth_window),
                        color=color, linewidth=2.5, linestyle='--',
                        label=f'{lbl} (val)', zorder=3)
            else:
                ax.plot(vi, vv, color=color, linewidth=2.5, linestyle='--',
                        label=f'{lbl} (val)', zorder=3)

    ax.set_title(f'Total Detection Loss vs Contrastive Loss — {run_name}', fontsize=13)
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.grid(True, alpha=0.3)
    apply_xlim(ax, xlim)
    apply_ylim(ax, ylim)
    ax.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1.01, 1),
              borderaxespad=0)
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, 'total_det_vs_contrastive.png'))


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description='Plot training losses from saved JSON files.')
    p.add_argument('--run-dir', type=str, required=True,
                   help='Path to the run directory (e.g. ~/lsst_runs/clip5_30k_4h200_bs64_ep50)')
    p.add_argument('--run-name', type=str, default=None,
                   help='Run name override (default: basename of run-dir)')
    p.add_argument('--smooth', type=int, default=20,
                   help='Smoothing window size for moving average (default: 20, use 1 to disable)')
    p.add_argument('--no-val', action='store_true',
                   help='Skip validation loss even if the file exists')
    # Optional plot selectors — prefix 'only-' avoids ambiguity with axis range flags.
    # If none are set, all plots are generated (default behaviour).
    p.add_argument('--only-individual', dest='only_individual', action='store_true',
                   help='Generate only individual per-loss plots')
    p.add_argument('--only-all-losses', dest='only_all_losses', action='store_true',
                   help='Generate only the all-losses overlaid figure')
    p.add_argument('--only-det-kp', dest='only_det_kp', action='store_true',
                   help='Generate only the detection + keypoint figure')
    p.add_argument('--only-contrastive', dest='only_contrastive', action='store_true',
                   help='Generate only the total detection vs contrastive figure')
    # Per-plot x-axis range: --<plot>-xmin / --<plot>-xmax
    # Plots: individual, all-losses, det-kp, contrastive
    for plot in ('individual', 'all-losses', 'det-kp', 'contrastive'):
        dest = plot.replace('-', '_')
        p.add_argument(f'--{plot}-xmin', dest=f'{dest}_xmin', type=int, default=None,
                       metavar='N', help=f'x-axis min iteration for {plot} plot(s)')
        p.add_argument(f'--{plot}-xmax', dest=f'{dest}_xmax', type=int, default=None,
                       metavar='N', help=f'x-axis max iteration for {plot} plot(s)')
        p.add_argument(f'--{plot}-ymin', dest=f'{dest}_ymin', type=float, default=None,
                       metavar='V', help=f'y-axis min loss for {plot} plot(s)')
        p.add_argument(f'--{plot}-ymax', dest=f'{dest}_ymax', type=float, default=None,
                       metavar='V', help=f'y-axis max loss for {plot} plot(s)')
    return p.parse_args()


def main():
    args = parse_args()

    run_dir  = os.path.expanduser(args.run_dir)
    run_name = args.run_name or os.path.basename(run_dir.rstrip('/'))
    out_dir  = os.path.join(run_dir, 'loss_plots')

    print(f"Run dir  : {run_dir}")
    print(f"Run name : {run_name}")
    print(f"Output   : {out_dir}")
    print(f"Smooth   : {args.smooth}")
    # Build per-plot xlim tuples; fall back to (None, None) = auto
    xlims = {
        'individual':  (args.individual_xmin,  args.individual_xmax),
        'all_losses':  (args.all_losses_xmin,  args.all_losses_xmax),
        'det_kp':      (args.det_kp_xmin,      args.det_kp_xmax),
        'contrastive': (args.contrastive_xmin, args.contrastive_xmax),
    }
    ylims = {
        'individual':  (args.individual_ymin,  args.individual_ymax),
        'all_losses':  (args.all_losses_ymin,  args.all_losses_ymax),
        'det_kp':      (args.det_kp_ymin,      args.det_kp_ymax),
        'contrastive': (args.contrastive_ymin, args.contrastive_ymax),
    }
    # Determine which plots to generate
    _selectors = {
        'individual':  args.only_individual,
        'all_losses':  args.only_all_losses,
        'det_kp':      args.only_det_kp,
        'contrastive': args.only_contrastive,
    }
    selected_plots = {k for k, v in _selectors.items() if v} or set(_selectors)
    print(f"Plots    : {', '.join(sorted(selected_plots))}")

    for name, (xmin, xmax) in xlims.items():
        if xmin is not None or xmax is not None:
            print(f"X-axis ({name}): [{xmin}, {xmax}]")
    for name, (ymin, ymax) in ylims.items():
        if ymin is not None or ymax is not None:
            print(f"Y-axis ({name}): [{ymin}, {ymax}]")

    # ---- locate loss files (JSON preferred, .npy legacy fallback) ----
    train_path_json = os.path.join(run_dir, f'{run_name}_losses.json')
    train_path_npy  = os.path.join(run_dir, f'{run_name}_losses.npy')
    val_path_json   = os.path.join(run_dir, f'{run_name}_val_losses.json')
    val_path_npy    = os.path.join(run_dir, f'{run_name}_val_losses.npy')

    if os.path.exists(train_path_json):
        train_path, train_fmt = train_path_json, 'json'
    elif os.path.exists(train_path_npy):
        train_path, train_fmt = train_path_npy, 'npy'
    else:
        raise FileNotFoundError(
            f"No training loss file found. Tried:\n"
            f"  {train_path_json}\n  {train_path_npy}"
        )

    print(f"\nLoading train losses from: {train_path} ({train_fmt})")
    train_pairs  = load_loss_json(train_path) if train_fmt == 'json' else load_loss_npy(train_path)
    train_series = extract_series(train_pairs)
    print(f"  Keys: {sorted(train_series.keys())}")
    print(f"  Iterations: {len(train_pairs)}")

    val_series = None
    if not args.no_val:
        if os.path.exists(val_path_json):
            val_path, val_fmt = val_path_json, 'json'
        elif os.path.exists(val_path_npy):
            val_path, val_fmt = val_path_npy, 'npy'
        else:
            val_path, val_fmt = None, None

        if val_path:
            print(f"Loading val losses from:   {val_path} ({val_fmt})")
            val_pairs  = load_loss_json(val_path) if val_fmt == 'json' else load_loss_npy(val_path)
            val_series = extract_series(val_pairs)
            print(f"  Keys: {sorted(val_series.keys())}")
            print(f"  Iterations: {len(val_pairs)}")
        else:
            print(f"Val loss file not found, skipping val.")

    all_keys = set(train_series.keys())
    if val_series:
        all_keys.update(val_series.keys())

    # Separate out non-loss tracking scalars (e.g. logit_scale)
    loss_keys     = all_keys - NON_LOSS_KEYS
    tracking_keys = all_keys & NON_LOSS_KEYS

    # ---- generate selected plots (default: all) ----
    if 'individual' in selected_plots:
        plot_all_individual(train_series, val_series, loss_keys, args.smooth, out_dir,
                            xlim=xlims['individual'], ylim=ylims['individual'])
    if 'all_losses' in selected_plots:
        plot_all_together(train_series, val_series, loss_keys, args.smooth, out_dir, run_name,
                          xlim=xlims['all_losses'], ylim=ylims['all_losses'])
    if 'det_kp' in selected_plots:
        plot_detection_keypoint(train_series, val_series, loss_keys, args.smooth, out_dir, run_name,
                                xlim=xlims['det_kp'], ylim=ylims['det_kp'])
    if 'contrastive' in selected_plots:
        plot_total_det_vs_contrastive(train_series, val_series, loss_keys, args.smooth, out_dir, run_name,
                                      xlim=xlims['contrastive'], ylim=ylims['contrastive'])

    # Plot tracking scalars individually without smoothing
    if tracking_keys:
        print("\n[5] Tracking scalar plots (logit_scale, etc.)...")
        for key in sorted(tracking_keys):
            fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
            plot_single_loss(ax, train_series, val_series, key,
                             smooth_window=1, title=label_for(key))
            fig.tight_layout()
            savefig(fig, os.path.join(out_dir, 'individual', key + '.png'))

    print(f"\nDone. All plots saved under {out_dir}/")


if __name__ == '__main__':
    main()