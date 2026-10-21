"""
plot_binned_metrics.py — magnitude-binned blend metrics with beta-binomial CI bands.

One PNG per metric. Truth metrics binned by truth i, detection
metrics by observed i. Blend-conditional metrics (resolved/survivor/absorbed) get
a companion N_t^sig panel so per-pipeline denominators are visible. LSST curves are
run- and threshold-independent: computed once per (buffer, LL), drawn in LSST_COLOR
in every figure, and saved standalone. All counts/metrics/CIs come from
blend_metrics.py.

Notebook:
    from plot_binned_metrics import main, NotebookArgs
    main(NotebookArgs(run_pairs=[('runA','runB')],
                      score_thresholds=[0.65], nms_thresholds=[0.65],
                      lsst_source_run='runA'))
    # Single run (no pairwise comparison): use single_runs instead of run_pairs
    main(NotebookArgs(single_runs=['runA'],
                      score_thresholds=[0.65], nms_thresholds=[0.65],
                      lsst_source_run='runA'))
    # LSST-only (no DeepDISC runs): omit run_pairs/single_runs/score_thresholds/nms_thresholds
    main(NotebookArgs(lsst_source_run='runA'))
CLI:
    python plot_binned_metrics.py --run-pairs runA runB \
        --score-thresholds 0.65 --nms-thresholds 0.65 \
        --lsst-source-run runA --buffers 2 --linking-lengths 1.0
    python plot_binned_metrics.py --single-runs runA \
        --score-thresholds 0.65 --nms-thresholds 0.65 \
        --lsst-source-run runA --buffers 2 --linking-lengths 1.0
    python plot_binned_metrics.py --lsst-only --lsst-source-run runA \
        --buffers 2 --linking-lengths 1.0
"""
import argparse
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

import blend_metrics as bm

# --- style (Tableau colorblind10, with a couple of custom hues added for
# extra categorical contrast; run_color[2] below is Paul Tol's "Bright"
# purple, chosen to stay distinguishable from LSST_COLOR under red-green CVD
# -- unlike a pink/magenta, which shares too much of the same red channel) ---
LSST_COLOR = '#C85200'  # red-orange, reserved for LSST so it reads as distinct
RUN_COLORS = ['#006BA4', '#1baf7a', '#AA3377', '#898989',
              '#A2C8EC', '#FFBC79', '#ABABAB', '#595959', '#CFCFCF']
LSST_LABEL = 'LSST'
N_SIG_FLOOR = 0.5  # log-axis floor for the N_t^sig panel; zero-count bins clip here
Y_AXIS_PAD = 0.05  # visual headroom above 1.0 / below 0.0; no ticks placed there

# Per-bin N_t^sig floor for resolved/survivor/absorbed, empirically derived (not
# picked by feel): resolved_frac's fitted rho is 1.0 (within a significant-blend
# group, "resolved" is all-or-nothing), so DEFF = per-bin size-weighted m_bar.
# In the bins near this floor (i ~ 20.3-21.3, N_t^sig 21-35) m_bar is 1.08-1.35
# across all four paper pipelines at LL=1.0". At k=0 (the bright-end LSST bins
# this is meant to protect) the beta-binomial half-width crosses ~0.07 -- half
# the headline (LL=1.0", buffer=2, gold) DD resolved_frac effect size of ~0.14
# -- at n=28 (DEFF 1.2) to n=32 (DEFF 1.35). Cross-checked against real per-bin
# N_t^sig at both LL=0.5" and LL=1.0": thresholds of 30-32 keep the same bins
# (bright edge ~i=21). Re-derive if the headline effect size or population
# structure changes materially (ci_readme_numbers.py prints per-bin m_bar).
DEFAULT_MIN_BIN_COUNT = DEFAULT_MIN_BIN_COUNT_BLEND = 30


PLOT_METRICS = bm.TRUTH_METRICS + bm.DET_METRICS + bm.BLEND_COND_METRICS + bm.CLASS_METRICS

# Display names, matching tab:metrics_formulas.
METRIC_LABELS = {
    'completeness': 'Completeness',
    'purity': 'Purity',
    'missed_frac': 'Missed Fraction',
    'spurious_frac': 'Spurious Fraction',
    'sig_blend_frac': 'Significant Blend Fraction',
    'shredded_frac': 'Shredded Fraction',
    'shred_frac': 'Shred Fraction',
    'unrec_blend_frac': 'Unrecognized Blend Fraction',
    'resolved_frac': 'Resolved Fraction',
    'survivor_frac': 'Survivor Fraction',
    'absorbed_frac': 'Absorbed Fraction',
    'galaxy_recall': 'Galaxy Recall',
    'galaxy_precision': 'Galaxy Precision',
    'star_recall': 'Star Recall',
    'star_precision': 'Star Precision',
}

# Metrics that get a companion per-bin denominator-count panel (so a sparse
# bin's fraction can be read alongside how many objects it's based on), and
# the label for that panel's y-axis. The denominator plotted is always
# b[metric]['den'] from compute_binned -- METRIC_NUMDEN already pins that to
# N_t (completeness, sig_blend_frac), N_d (purity), or N_t_sig
# (resolved/survivor/absorbed_frac), so no separate lookup is needed.
COMPANION_PANEL_METRICS = {
    'completeness': r'$N_t$',
    'purity': r'$N_d$',
    'resolved_frac': r'$N_t^{\rm sig}$',
    'survivor_frac': r'$N_t^{\rm sig}$',
    'absorbed_frac': r'$N_t^{\rm sig}$',
    # Each STAR panel gets its own companion denominator strip: star_recall's is the
    # per-bin true-star count (N_ss + N_sg), star_precision's the predicted-star
    # count (N_ss + N_gs)
    'star_recall':    r'$N_{ss}+N_{sg}$',
    'star_precision': r'$N_{ss}+N_{gs}$',
}

def _merge_class_binned(binned, tcls, prefix, edges, min_count):
    """Fold compute_class_binned's four metric keys into the blend `binned`
    dict so each series carries every metric and plot_metric needs no change.
    edges/centers are already present from compute_binned."""
    cb = bm.compute_class_binned(tcls, prefix, edges, min_count)
    for m in bm.CLASS_METRICS:
        binned[m] = cb[m]
    return binned

# ---------------------------------------------------------------------------
def _draw_curve(ax, centers, edges, b, color, label, ls='-'):
    """Straight-line point estimate through bin centers, with the first/last
    valid bin's endpoint extended out to its true bin edge -- e.g. the last
    valid bin's right edge sits at m_cut + buffer, so the curve reaches that
    line instead of stopping a half-bin-width short at its center. Interior
    points are untouched, so NaN-masked (sparse) bins still break the line
    normally.

    CI: per-bin lo/hi drawn as vertical error bars (no caps) at each bin's
    true center, not interpolated -- CI width is a function of discrete
    per-bin N and can jump sharply at a bin boundary (e.g. right where a
    curve crosses into sparse territory), so smoothing it (spline or
    otherwise) would misrepresent a real discontinuity as a gradual taper.

    At k=0 (or k=n), betabinom_ci's flat-prior interval puts lo (or hi) a
    hair on the far side of val itself -- by design, see its docstring --
    so the whisker on that side is clipped to 0 rather than drawn negative,
    which matplotlib's errorbar rejects outright."""
    v, lo, hi = b['val'], b['lo'], b['hi']
    valid = np.where(np.isfinite(v))[0]
    if not valid.size:
        ax.plot(centers, v, color=color, ls=ls, lw=1.6, label=label)
        return
    i0, i1 = int(valid[0]), int(valid[-1])
    sl = slice(i0, i1 + 1)

    x = np.concatenate(([edges[i0]], centers[sl], [edges[i1 + 1]]))
    vv = np.concatenate(([v[i0]], v[sl], [v[i1]]))
    ax.plot(x, vv, color=color, ls=ls, lw=1.6, label=label)

    yerr = np.vstack([np.clip(v[sl] - lo[sl], 0, None),
                       np.clip(hi[sl] - v[sl], 0, None)])
    ax.errorbar(centers[sl], v[sl], yerr=yerr, fmt='none', ecolor=color,
                elinewidth=1.2, capsize=0, alpha=0.7)


def _check_unit_bounds(metric, label, b, tol=1e-9):
    """val/lo/hi are fractions and must stay within [0,1]; warn (don't crash) if
    a curve ever violates that, since it would signal a real bug upstream."""
    for name in ('val', 'lo', 'hi'):
        arr = b[name]
        finite = arr[np.isfinite(arr)]
        if finite.size and (finite.min() < -tol or finite.max() > 1.0 + tol):
            print(f'  WARNING: {metric} ({label}) {name} out of [0,1]: '
                  f'min={finite.min():.6f}, max={finite.max():.6f}')


DEAD_PAD_FRACTION = 0.05  # dead pad (past 0/1) as a share of the total axis
# span above which it reads as an obvious blank block rather than unnoticed
# margin. The absolute pad is roughly constant (~Y_AXIS_PAD), but metrics with
# a small dynamic range (blend-conditional fractions hovering near 0) have a
# small span too, so that same constant pad eats a much bigger share of the
# plot than it does for a wide-range metric like completeness.


def _tighten_dead_pad(ylim, threshold=DEAD_PAD_FRACTION):
    """Where the Y_AXIS_PAD headroom past 0/1 would make up more than
    `threshold` of the axis span shrink that side's pad instead of dropping it outright, 
    so the dead space lands right at `threshold` rather than at a fixed 0.05.
    Solving dead/(span) = threshold for the bottom (span = hi - lo,
    dead = -lo) gives dead = threshold*hi/(1-threshold); the top is the
    mirror image. Metrics whose dead pad already stays under the threshold
    keep their normal headroom untouched (the solved cap is provably >=
    their current pad in that case)."""
    lo, hi = ylim
    if lo < 0.0:
        max_dead = threshold * hi / (1.0 - threshold)
        lo = -min(-lo, max_dead)
    if hi > 1.0:
        max_dead = threshold * (1.0 - lo) / (1.0 - threshold)
        hi = 1.0 + min(hi - 1.0, max_dead)
    return lo, hi


DEFAULT_TITLE_TEMPLATE = '{metric} (LL={ll}", Δm={buffer})'


def _dynamic_ylim(series, metric, is_blend, min_total_blend_count):
    """Y-limits tight to the plotted values (val + CI band), clipped to
    [0,1] plus a little headroom (Y_AXIS_PAD) so a curve at the boundary
    isn't drawn flush against the axes edge -- except where that headroom
    would be a disproportionate, unreachable share of a narrow-range
    metric's axis (see _tighten_dead_pad)."""
    vals = []
    for s in series:
        b = s['binned']
        if is_blend and int(b['N_t_sig'].sum()) < min_total_blend_count:
            continue
        for arr in (b[metric]['val'], b[metric]['lo'], b[metric]['hi']):
            finite = arr[np.isfinite(arr)]
            if finite.size:
                vals.append(finite)
    if not vals:
        return -Y_AXIS_PAD, 1.0 + Y_AXIS_PAD
    allv = np.concatenate(vals)
    lo, hi = float(allv.min()), float(allv.max())
    pad = max(0.05, (hi - lo) * 0.1)
    ylim = max(-Y_AXIS_PAD, lo - pad), min(1.0 + Y_AXIS_PAD, hi + pad)
    return _tighten_dead_pad(ylim)


def plot_metric(metric, series, m_cut, buffer, ll, out_path, dpi=200, min_total_blend_count=0,
                title_template=None, dynamic_ylim=False, plot_mag_min=None):
    """series: list of dicts {label, color, ls, binned}. One figure -> out_path.

    title_template: format string with {metric}, {ll}, {buffer}, {m_cut} placeholders
    (metric is the display name from METRIC_LABELS); defaults to DEFAULT_TITLE_TEMPLATE.
    dynamic_ylim: zoom the y-axis to the plotted values instead of the fixed [0,1] range.
    plot_mag_min: visualization-only x-axis left cut -- does NOT rebin the data (unlike
    mag_min in NotebookArgs.mag_bins), just crops what's shown. Must fall within the
    existing bin range. Keeps the same left-edge margin matplotlib's autoscale would
    otherwise put before the leftmost bin edge, just measured from plot_mag_min instead.
    """
    is_det = metric in bm.DET_METRICS
    is_blend = metric in bm.BLEND_COND_METRICS
    has_count_panel = metric in COMPANION_PANEL_METRICS

    if has_count_panel:
        fig, (ax, axn) = plt.subplots(
            2, 1, figsize=(7, 5.5), sharex=True,
            gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.08})
    else:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        axn = None

    skipped = []
    n_drawn = 0
    for s in series:
        b = s['binned']
        centers = b['centers']
        if is_blend:
            total_sig = int(b['N_t_sig'].sum())
            if total_sig < min_total_blend_count:
                print(f'  {metric}: {s["label"]} reported scalar-only '
                      f'(N_t^sig = {total_sig} < {min_total_blend_count})')
                skipped.append((s['label'], total_sig))
                continue
        _check_unit_bounds(metric, s['label'], b[metric])
        _draw_curve(ax, centers, b['edges'], b[metric], s['color'], s['label'], s.get('ls', '-'))
        if axn is not None:
            counts = np.clip(b[metric]['den'], N_SIG_FLOOR, None)
            # stairs (not step(where='mid')): anchored to the true bin edges,
            # not inferred from the midpoint between adjacent centers -- the
            # last bin's segment reaches edges[-1] even when there's no
            # trailing bin past it to infer that midpoint from (e.g. when
            # mag_max is set right at the eval-limit cutoff).
            axn.stairs(counts, b['edges'], baseline=None, color=s['color'], lw=1.2)
        n_drawn += 1

    if skipped and n_drawn == 0:
        ax.text(0.5, 0.5, f'insufficient blend statistics\n(N_t^sig < {min_total_blend_count} for all series)',
                transform=ax.transAxes, ha='center', va='center', fontsize=9, color='0.4')
    elif skipped:
        note = '; '.join(f'{label} (N_t^sig={n})' for label, n in skipped)
        ax.text(0.05, 0.05, f'not shown: {note}', transform=ax.transAxes,
                ha='left', va='bottom', fontsize=6.5, color='0.4')

    eval_limit = m_cut + buffer
    vline_specs = [(m_cut, '--', r'$m_{\rm cut}$' + f' ({m_cut:.2f})'),
                   (eval_limit, ':', r'$m_{\rm cut} + \Delta m$' + f' ({eval_limit:.2f})')]
    for x, style, vlabel in vline_specs:
        ax.axvline(x, color='0.4', ls=style, lw=1, label=vlabel)
        if axn is not None:
            axn.axvline(x, color='0.4', ls=style, lw=1)
    ax.axvspan(m_cut, eval_limit, color='0.4', alpha=0.08, linewidth=0)
    if axn is not None:
        axn.axvspan(m_cut, eval_limit, color='0.4', alpha=0.08, linewidth=0)

    label = METRIC_LABELS.get(metric, metric)
    ax.set_ylabel(label)
    ylim = (_dynamic_ylim(series, metric, is_blend, min_total_blend_count)
            if dynamic_ylim else (-Y_AXIS_PAD, 1.0 + Y_AXIS_PAD))
    ax.set_ylim(*ylim)
    tick_tol = 1e-9  # tolerance for float step-arithmetic error in the auto
    # locator (e.g. 0.2*5 landing on 1.0000000000000002), so the 0.0/1.0
    # ticks aren't dropped for being a hair outside [0,1]
    ax.set_yticks([t for t in ax.get_yticks() if -tick_tol <= t <= 1.0 + tick_tol])
    ax.set_ylim(*ylim)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    target = axn if axn is not None else ax
    if is_det:
        target.set_xlabel('Observed $i$-band magnitude (AB)')
    else:
        target.set_xlabel('Truth $i$-band magnitude (AB)')
    if axn is not None:
        axn.set_ylabel(COMPANION_PANEL_METRICS[metric])
        axn.set_yscale('log')
        axn.set_ylim(bottom=N_SIG_FLOOR)
        axn.grid(alpha=0.2)
    template = title_template or DEFAULT_TITLE_TEMPLATE
    ax.set_title(template.format(metric=label, ll=ll, buffer=buffer, m_cut=m_cut), fontsize=9)

    if plot_mag_min is not None and series:
        edges = series[0]['binned']['edges']
        data_lo, data_hi = float(edges[0]), float(edges[-1])
        if not (data_lo <= plot_mag_min < data_hi):
            raise ValueError(
                f'plot_mag_min={plot_mag_min} is outside the bin range '
                f'[{data_lo}, {data_hi})')
        auto_left, auto_right = ax.get_xlim()  # matplotlib's default autoscale margins
        new_left = plot_mag_min - (data_lo - auto_left)
        # Right margin: matplotlib sized it once as a fraction of the *full*
        # bin range (data_lo to data_hi). Leaving that absolute margin
        # unchanged after cropping the left side shrinks the visible span
        # underneath it, so the same margin becomes a disproportionately
        # large chunk of the now-narrower window -- same failure mode
        # _tighten_dead_pad fixes on the y-axis. Fix: preserve the original
        # right-margin *fraction* of the span, not its absolute size, and
        # reapply that fraction to the new (cropped) span.
        right_margin_frac = (auto_right - data_hi) / (data_hi - data_lo)
        new_right = data_hi + right_margin_frac * (data_hi - new_left)
        ax.set_xlim(new_left, new_right)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'  {metric} -> {out_path}')


def _figset(series, m_cut, buffer, ll, out_dir, dpi, min_total_blend_count, title_template=None,
           dynamic_ylim=False, metrics=PLOT_METRICS, plot_mag_min=None):
    # one subfolder per linking length, so 0.5" and 1.0" figures never collide
    for metric in metrics:
        plot_metric(metric, series, m_cut, buffer, ll,
                    Path(out_dir) / str(ll) / f'{metric}_buf{buffer}.png',
                    dpi=dpi, min_total_blend_count=min_total_blend_count,
                    title_template=title_template, dynamic_ylim=dynamic_ylim,
                    plot_mag_min=plot_mag_min)


# ---------------------------------------------------------------------------
def run_pair_comparisons(
    root_run_dir: Path,
    run_pairs: Sequence[Sequence[str]],
    combos: Sequence[str],
    mapped_combo_pairs: Sequence[Sequence[str]],
    buffers: Sequence[int],
    linking_lengths: Sequence[str],
    mag_limit: str,
    mag_bins: np.ndarray,
    min_count: int,
    comparison_subdir: str,
    *,
    lsst_source_run: str,
    lsst_output_dir: Path,
    single_runs: Sequence[str] = (),
    output_subdir: str = 'metric_plots',
    min_bin_count_blend: Optional[int] = None,
    min_total_blend_count: int = 0,
    dpi: int = 200,
    title_template: Optional[str] = None,
    run_labels: Optional[Dict[str, str]] = None,
    dynamic_ylim: bool = False,
    lsst_mode: str = 'full',
    lsst_in_comparison: bool = True,
    hide_combo_in_label: bool = False,
    metrics: Sequence[str] = PLOT_METRICS,
    plot_mag_min: Optional[float] = None,
) -> None:
    """Per-run + N-way comparison + standalone-LSST figures for every metric.

    run_pairs: list of run groups; each group is 2+ runs drawn together
    on one comparison figure, plus individual per-run figures for each run.
    Pass identical runs with different combos via mapped_combo_pairs for
    a self-comparison figure (e.g. run_pairs=[('runX', 'runX')]).

    combos = shared sweep applied to every run in every group (and to every
    run in single_runs).  When mapped_combo_pairs is non-empty it must be a
    1:1 index-aligned list of combo tuples, one per run_pairs entry; each
    tuple must have the same length as its corresponding run group.
    Pass run_pairs=[] and combos=[] to produce only the standalone-LSST
    figures.

    single_runs: runs to plot individually (per-run figures only, no
    comparison) using the same combos sweep as run_pairs.
    title_template: format string with {metric}, {ll}, {buffer}, {m_cut}
    placeholders; defaults to DEFAULT_TITLE_TEMPLATE.
    run_labels: optional {run_name: display_name} override for legend
    entries (default is f'{run_name}/{combo}').
    min_bin_count_blend: per-bin denominator floor (N_t_sig) for
    resolved/survivor/absorbed only.
    dynamic_ylim: zoom each figure's y-axis to its plotted values instead
    of [0,1].
    lsst_mode: 'full' (default) draws LSST everywhere including standalone
    figures; 'overlay_only' keeps LSST in per-run/comparison overlays but
    skips the separate standalone-LSST-only figure; 'none' omits LSST
    entirely (DeepDISC-only plots).
    lsst_in_comparison: include LSST in comparison figures (default True);
    has no effect if lsst_mode is 'none'.
    hide_combo_in_label: drop the sX_nY combo tag from legend entries.
    plot_mag_min: visualization-only x-axis left cut, applied to every
    figure; does not change mag_bins/binning.
    """
    root = Path(root_run_dir).expanduser()
    m_cut = bm.MAG_LIMITS[mag_limit]
    edges = np.asarray(mag_bins, float)
    run_labels = run_labels or {}

    # stable per-run colors, in the order runs first appear
    order = list(dict.fromkeys(
        [r for group in run_pairs for r in group] + list(single_runs)))
    run_color = {r: RUN_COLORS[i % len(RUN_COLORS)] for i, r in enumerate(order)}

    @lru_cache(maxsize=None)
    def binned_dd(run, combo, ll, mag_limit_val):
        df = bm.load_analysis_df(root / run, 'dd', str(ll), combo, mag_limit_val)
        t, d = bm.split_truth_det(df, 'dd')
        bm.assert_maximality(t, d, 'dd')
        binned = bm.compute_binned(t, d, 'dd', edges, min_count, min_bin_count_blend)
        tcls = bm.attach_pred_class(t, d, 'dd')
        return _merge_class_binned(binned, tcls, 'dd', edges, min_count)

    @lru_cache(maxsize=None)
    def binned_lsst(buffer, ll, mag_limit_val):
        df = bm.load_analysis_df(root / lsst_source_run, 'lsst', str(ll),
                                 None, mag_limit_val)
        t, d = bm.split_truth_det(df, 'lsst')
        bm.assert_maximality(t, d, 'lsst')
        binned = bm.compute_binned(t, d, 'lsst', edges, min_count, min_bin_count_blend)
        tcls = bm.attach_pred_class(t, d, 'lsst')
        return _merge_class_binned(binned, tcls, 'lsst', edges, min_count)

    for buffer in buffers:
        mag_limit_val = m_cut + buffer
        for ll in linking_lengths:
            lsst_series = None
            if lsst_mode != 'none':
                lsst_b = binned_lsst(buffer, ll, mag_limit_val)
                lsst_series = {'label': LSST_LABEL, 'color': LSST_COLOR, 'ls': '-',
                               'binned': lsst_b}
                if lsst_mode == 'full':
                    _figset([lsst_series], m_cut, buffer, ll,
                            Path(lsst_output_dir).expanduser(), dpi, min_total_blend_count,
                            title_template=title_template, dynamic_ylim=dynamic_ylim, metrics=metrics,
                            plot_mag_min=plot_mag_min)

            combo_groups = list(mapped_combo_pairs) if mapped_combo_pairs \
                else [tuple(c for _ in group) for group in run_pairs for c in combos]

            for group_idx, runs in enumerate(run_pairs):
                for cg in combo_groups:
                    if len(cg) != len(runs):
                        continue  # skip mismatched combo group

                    series_list = []
                    for run, combo in zip(runs, cg):
                        b = binned_dd(run, combo, ll, mag_limit_val)
                        name = run_labels.get(run, run)
                        label = name if hide_combo_in_label else f'{name}/{combo}'
                        s = {'label': label, 'color': run_color[run],
                             'ls': '-', 'binned': b}
                        series_list.append((run, combo, s))

                    # per-run figures ({run_dir}/metric_plots/)
                    for run, combo, s in series_list:
                        _figset([x for x in (lsst_series, s) if x], m_cut, buffer, ll,
                                root / run / output_subdir, dpi, min_total_blend_count,
                                title_template=title_template, dynamic_ylim=dynamic_ylim,
                                metrics=metrics, plot_mag_min=plot_mag_min)

                    # comparison figures (skip trivial 1-run groups)
                    if len(runs) > 1:
                        comp_dir = '__vs__'.join(runs)
                        comp = root / runs[0] / comparison_subdir / comp_dir
                        comp_lsst = lsst_series if lsst_in_comparison else None
                        _figset([x for x in (comp_lsst,) + tuple(s for _, _, s in series_list) if x],
                                m_cut, buffer, ll, comp, dpi, min_total_blend_count,
                                title_template=title_template, dynamic_ylim=dynamic_ylim,
                                metrics=metrics, plot_mag_min=plot_mag_min)

            # single-run figures (no pairwise comparison)
            for run in single_runs:
                for combo in combos:
                    b = binned_dd(run, combo, ll, mag_limit_val)
                    name = run_labels.get(run, run)
                    label = name if hide_combo_in_label else f'{name}/{combo}'
                    s = {'label': label, 'color': run_color[run],
                         'ls': '-', 'binned': b}
                    _figset([x for x in (lsst_series, s) if x], m_cut, buffer, ll,
                            root / run / output_subdir, dpi, min_total_blend_count,
                            title_template=title_template, dynamic_ylim=dynamic_ylim,
                            metrics=metrics, plot_mag_min=plot_mag_min)


# ---------------------------------------------------------------------------
@dataclass
class NotebookArgs:
    lsst_source_run: str = 'lsst5_30k_4h200_bs192_ep50_musigma'
    run_pairs: Sequence[Sequence[str]] = field(default_factory=list)
    single_runs: Sequence[str] = field(default_factory=list)
    score_thresholds: Sequence[float] = field(default_factory=list)
    nms_thresholds: Sequence[float] = field(default_factory=list)
    root_run_dir: str = '~/lsst_runs'
    mapped_combo_pairs: Sequence[Sequence[str]] = field(default_factory=list)
    buffers: Sequence[int] = (2,)
    linking_lengths: Sequence[str] = ('1.0',)
    mag_limit: str = 'gold'
    mag_min: float = 18.5
    mag_max: float = 27.5
    mag_bin_width: float = 0.5
    min_count: int = DEFAULT_MIN_BIN_COUNT
    min_bin_count_blend: Optional[int] = DEFAULT_MIN_BIN_COUNT_BLEND
    min_total_blend_count: int = 0
    lsst_output_dir: str = './lsst_metric_plots'
    output_subdir: str = 'metric_plots'
    comparison_subdir: str = 'comparisons'
    dpi: int = 200
    title_template: Optional[str] = None
    run_labels: Dict[str, str] = field(default_factory=dict)
    dynamic_ylim: bool = False
    lsst_mode: str = 'full'
    lsst_in_comparison: bool = True
    hide_combo_in_label: bool = False
    classification_only: bool = False
    metrics: Sequence[str] = field(default_factory=lambda: list(PLOT_METRICS))
    plot_mag_min: Optional[float] = None

    @property
    def mag_bins(self):
        """Bin edges anchored so m_cut falls exactly on an edge (not mid-bin).
        buffers are always integers and mag_bin_width (default 0.5) evenly
        divides 1.0, so anchoring to m_cut alone puts every m_cut + buffer on
        an edge too -- no bin ever straddles an eval-limit line, which used to
        make the N_t^sig step plot visually spill past m_cut + buffer even
        though no truth objects actually exist past that cut."""
        m_cut = bm.MAG_LIMITS[self.mag_limit]
        n_below = np.ceil((m_cut - self.mag_min) / self.mag_bin_width)
        lo = m_cut - n_below * self.mag_bin_width
        return np.arange(lo, self.mag_max + self.mag_bin_width / 2, self.mag_bin_width)

    @property
    def combos(self):
        if len(self.score_thresholds) != len(self.nms_thresholds):
            raise ValueError('score_thresholds and nms_thresholds must have the same length')
        return [bm.combo_tag(s, n) for s, n in zip(self.score_thresholds, self.nms_thresholds)]


def main(args: NotebookArgs):
    metrics = list(bm.CLASS_METRICS) if args.classification_only else list(args.metrics)
    run_pair_comparisons(
        root_run_dir=args.root_run_dir,
        run_pairs=args.run_pairs,
        single_runs=args.single_runs,
        combos=args.combos,
        mapped_combo_pairs=args.mapped_combo_pairs,
        buffers=args.buffers,
        linking_lengths=args.linking_lengths,
        mag_limit=args.mag_limit,
        mag_bins=args.mag_bins,
        min_count=args.min_count,
        min_bin_count_blend=args.min_bin_count_blend,
        comparison_subdir=args.comparison_subdir,
        lsst_source_run=args.lsst_source_run,
        lsst_output_dir=args.lsst_output_dir,
        output_subdir=args.output_subdir,
        min_total_blend_count=args.min_total_blend_count,
        dpi=args.dpi,
        title_template=args.title_template,
        run_labels=args.run_labels,
        dynamic_ylim=args.dynamic_ylim,
        lsst_mode=args.lsst_mode,
        lsst_in_comparison=args.lsst_in_comparison,
        hide_combo_in_label=args.hide_combo_in_label,
        metrics=metrics,
        plot_mag_min=args.plot_mag_min,
    )

def _parse_run_labels(raw_maps):
    labels = {}
    for raw in raw_maps or []:
        if ':' not in raw:
            raise SystemExit(f"Invalid --run-labels entry '{raw}'. Use format: run_name:Display Name")
        run, label = raw.split(':', 1)
        labels[run.strip()] = label.strip()
    return labels


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root-run-dir', default='~/lsst_runs',
                   help='Base directory containing each run subdirectory')
    p.add_argument('--lsst-only', action='store_true',
                   help='Skip DeepDISC runs; only produce the standalone LSST figures. '
                        'Makes --run-pairs/--single-runs/--score-thresholds/'
                        '--nms-thresholds optional.')
    p.add_argument('--run-pairs', action='append', nargs='+', default=None,
                   help='Repeatable: --run-pairs runA runB [--run-pairs runC runD ...] '
                        'Each invocation = one comparison group (2+ runs drawn together). '
                        '(required unless --lsst-only or --single-runs is given)')
    p.add_argument('--single-runs', nargs='+', default=None,
                   help='run(s) to plot individually, no comparison: '
                        'runA [runB ...] (required unless --lsst-only or '
                        '--run-pairs is given)')
    p.add_argument('--score-thresholds', nargs='+', type=float, default=None,
                   help='score threshold per combo, or a single shared value '
                        '(required unless --lsst-only)')
    p.add_argument('--nms-thresholds', nargs='+', type=float, default=None,
                   help='NMS threshold per combo, or a single shared value '
                        '(required unless --lsst-only)')
    p.add_argument('--mapped-combo-pairs', action='append', nargs='+', default=None,
                   help='Repeatable: --mapped-combo-pairs s0.5_n0.6 s0.6_n0.65 '
                        'One invocation per run_pairs group (raw sX_nY tags, '
                        'overrides --score-thresholds/--nms-thresholds)')
    p.add_argument('--lsst-source-run', default='lsst5_30k_4h200_bs192_ep50_musigma',
                   help='Run whose lsst/ parquet supplies the LSST curve '
                        '(threshold-independent, loaded once per buffer/LL)')
    p.add_argument('--buffers', nargs='+', type=int, default=[2],
                   help='Δm values added to the mag-limit cut for the evaluation '
                        'range (mag_limit_val = m_cut + buffer)')
    p.add_argument('--linking-lengths', nargs='+', default=['1.0'],
                   help='FoF linking-length tags to plot separately, matching the '
                        'analysis_cats subdirectory names (e.g. 0.5 1.0)')
    p.add_argument('--mag-limit', default='gold', choices=bm.MAG_LIMITS,
                   help='Which blend_metrics.MAG_LIMITS entry defines m_cut')
    p.add_argument('--mag-min', type=float, default=18.5,
                   help='Lower edge of the magnitude bin range')
    p.add_argument('--mag-max', type=float, default=27.5,
                   help='Upper edge of the magnitude bin range')
    p.add_argument('--mag-bin-width', type=float, default=0.5,
                   help='Width of each magnitude bin')
    p.add_argument('--min-count', type=int, default=DEFAULT_MIN_BIN_COUNT,
                   help='Per-bin denominator floor; bins below this are NaN in the '
                        'plotted curve for every metric except resolved/survivor/'
                        'absorbed (see --min-bin-count-blend)')
    p.add_argument('--min-bin-count-blend', type=int, default=DEFAULT_MIN_BIN_COUNT_BLEND,
                   help='Per-bin N_t_sig floor for resolved/survivor/absorbed only '
                        f'(default: {DEFAULT_MIN_BIN_COUNT_BLEND}, empirically derived -- '
                        'see DEFAULT_MIN_BIN_COUNT_BLEND). Not --min-total-blend-count.')
    p.add_argument('--min-total-blend-count', type=int, default=0,
                   help='Skip drawing a whole curve if its total N_t_sig, summed '
                        'across all bins, is below this (default: 0, never skip)')
    p.add_argument('--lsst-output-dir', default='./lsst_metric_plots',
                   help='Output directory for the standalone LSST-only figures')
    p.add_argument('--output-subdir', default='metric_plots',
                   help='Subdirectory name, under each run dir, for that run\'s own figures')
    p.add_argument('--comparison-subdir', default='comparisons',
                   help='Subdirectory name, under run_a\'s dir, for pairwise comparison figures')
    p.add_argument('--dpi', type=int, default=200,
                   help='Output figure resolution (passed to fig.savefig(dpi=))')
    p.add_argument('--metrics', nargs='+', default=None, choices=PLOT_METRICS,
                   help='Subset of metrics to plot (default: all of PLOT_METRICS). '
                        'e.g. --metrics completeness for just that one figure.')
    p.add_argument('--plot-mag-min', type=float, default=None,
                   help='Visualization-only x-axis left cut (does not change binning); '
                        'must fall within the mag_bins range')
    p.add_argument('--title', dest='title_template', default=None,
                   help='Custom title template, supports {metric}, {ll}, {buffer}, {m_cut} '
                        'placeholders (default: "%s")' % DEFAULT_TITLE_TEMPLATE)
    p.add_argument('--run-labels', nargs='+', default=None,
                   help='Optional legend name overrides, flat list: '
                        'run_name:"Display Name" [...]')
    p.add_argument('--dynamic-ylim', action='store_true',
                   help='Zoom each figure y-axis to its plotted values instead of [0,1]')
    p.add_argument('--lsst-mode', default='full',
                   choices=['full', 'overlay_only', 'none'],
                   help="'full' (default): LSST everywhere, standalone figures too. "
                        "'overlay_only': LSST in per-run/comparison overlays, no "
                        "standalone figures. 'none': omit LSST entirely.")
    p.add_argument('--lsst-in-comparison', action='store_true', default=True,
                   dest='lsst_in_comparison',
                   help='Include LSST in comparison figures (default: True)')
    p.add_argument('--no-lsst-in-comparison', action='store_false',
                   dest='lsst_in_comparison',
                   help='Exclude LSST from comparison figures')
    p.add_argument('--hide-combo-in-label', action='store_true',
                   help='Drop the sX_nY combo tag from legend entries, showing just '
                        'the run name (default legend entry is "run_name/sX_nY")')
    p.add_argument('--classification-only', '--cls-only', action='store_true',
                   help='Plot only the four classification metrics '
                        '(galaxy_recall, galaxy_precision, star_recall, star_precision)')
    return p.parse_args()


def _cli():
    a = parse_args()
    has_targets = a.run_pairs or a.single_runs
    if not a.lsst_only and not (has_targets and a.score_thresholds and a.nms_thresholds):
        raise SystemExit('--run-pairs and/or --single-runs, plus --score-thresholds and '
                          '--nms-thresholds, are required unless --lsst-only is set')
    args = NotebookArgs(
        run_pairs=a.run_pairs or [],
        single_runs=a.single_runs or [],
        score_thresholds=a.score_thresholds or [],
        nms_thresholds=a.nms_thresholds or [],
        lsst_source_run=a.lsst_source_run,
        root_run_dir=a.root_run_dir,
        mapped_combo_pairs=a.mapped_combo_pairs,
        buffers=a.buffers,
        linking_lengths=a.linking_lengths,
        mag_limit=a.mag_limit,
        mag_min=a.mag_min, mag_max=a.mag_max, mag_bin_width=a.mag_bin_width,
        min_count=a.min_count, min_bin_count_blend=a.min_bin_count_blend,
        min_total_blend_count=a.min_total_blend_count,
        lsst_output_dir=a.lsst_output_dir, output_subdir=a.output_subdir,
        comparison_subdir=a.comparison_subdir, dpi=a.dpi,
        title_template=a.title_template,
        run_labels=_parse_run_labels(a.run_labels),
        dynamic_ylim=a.dynamic_ylim,
        lsst_mode=a.lsst_mode,
        lsst_in_comparison=a.lsst_in_comparison,
        hide_combo_in_label=a.hide_combo_in_label,
        classification_only=a.classification_only,
        metrics=a.metrics or list(PLOT_METRICS),
        plot_mag_min=a.plot_mag_min,
    )
    main(args)


if __name__ == '__main__':
    _cli()