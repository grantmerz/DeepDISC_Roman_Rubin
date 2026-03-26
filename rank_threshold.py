"""
rank_thresholds.py — Find the most robust (score, nms) threshold combo
across all mag limit/buffer evaluation configs

For each combo, computes a percentile rank within each config's metric,
then aggregates across configs using minimax, mean, or min value strats

Usage:
    # With metrics-dir as base (paths relative to metrics-dir):
    python rank_thresholds.py \
        --metrics-dir ~/lsst_runs/lsst5_30k_4h200_bs192_ep50/metrics \
        --csv-files gold_buf0/gs_metrics.csv gold_buf1/gs_metrics.csv gold_buf2/gs_metrics.csv \
        --mag-limits gold gold gold 
        --buffers 0 1 2

    # With absolute paths:
    python rank_thresholds.py --absolute-paths \
        --csv-files /path/to/gs_metrics.csv \
        --mag-limits gold --buffers 0

    # Custom optimization:
    python rank_thresholds.py \
        --metrics-dir ~/lsst_runs/my_run/metrics \
        --csv-files gold_buf0/gs_metrics.csv gold_buf1/gs_metrics.csv gold_buf2/gs_metrics.csv \
        --mag-limits gold gold gold --buffers 0 1 2 \
        --primary-metric f1 \
        --constraint "purity >= 0.85" \
        --constraint "shred_frac <= 0.05"

   ===== LSST Baseline and CLIP 30k ======
    Gold Mag Limit + Power Law Mag Limit
    ------------------------------------
    python rank_thresholds.py --metrics-dir ~/lsst_runs/lsst5_30k_4h200_bs192_ep50/metrics --csv-files gold_buf0/gs_metrics.csv gold_buf1/gs_metrics.csv gold_buf2/gs_metrics.csv pl_buf0/gs_metrics.csv pl_buf1/gs_metrics.csv pl_buf2/gs_metrics.csv --mag-limits gold gold gold power_law power_law power_law --buffers 0 1 2 0 1 2
    python rank_thresholds.py --metrics-dir ~/lsst_runs/clip5_30k_4h200_bs64_ep50/metrics --csv-files gold_buf0/gs_metrics.csv gold_buf1/gs_metrics.csv gold_buf2/gs_metrics.csv pl_buf0/gs_metrics.csv pl_buf1/gs_metrics.csv pl_buf2/gs_metrics.csv --mag-limits gold gold gold power_law power_law power_law --buffers 0 1 2 0 1 2
    
    > ~/lsst_runs/lsst5_30k_4h200_bs192_ep50/metrics/rank_outs/rank_thresh.txt 2>&1
    > ~/lsst_runs/clip5_30k_4h200_bs64_ep50/metrics/rank_outs/rank_thresh.txt 2>&1
    
    ===== LSST Baseline All ======
    Gold Mag Limit + Power Law Mag Limit
    --------------
    python rank_thresholds.py --metrics-dir ~/lsst_runs/lsst5_all_4h200_bs192_ep50/metrics --csv-files gold_buf0/gs_metrics.csv gold_buf1/gs_metrics.csv gold_buf2/gs_metrics.csv pl_buf0/gs_metrics.csv pl_buf1/gs_metrics.csv pl_buf2/gs_metrics.csv --mag-limits gold gold gold power_law power_law power_law --buffers 0 1 2 0 1 2
    python rank_thresholds.py --metrics-dir ~/lsst_runs/clip5_all_4h200_bs64_ep50/metrics --csv-files gold_buf0/gs_metrics.csv gold_buf1/gs_metrics.csv gold_buf2/gs_metrics.csv pl_buf0/gs_metrics.csv pl_buf1/gs_metrics.csv pl_buf2/gs_metrics.csv --mag-limits gold gold gold power_law power_law power_law --buffers 0 1 2 0 1 2

    > ~/lsst_runs/lsst5_all_4h200_bs192_ep50/metrics/rank_outs/rank_thresh.txt 2>&1
    > ~/lsst_runs/clip5_all_4h200_bs64_ep50/metrics/rank_outs/rank_thresh.txt 2>&1
    
"""

import argparse
import os
import re
import numpy as np
import pandas as pd


# ============================================================================
# Constants
# ============================================================================
MAG_LIMITS = {
    'power_law': 26.22,
    'gold': 25.3,
    'nominal': 26.42,
}

# Metrics where lower is better
LOWER_IS_BETTER = {
    'blend_loss_frac', 'unrec_blend_frac_total', 'unrec_blend_frac_blended', 'unrec_blend_frac_matched',
    'unrec_blend_det_frac_total', 'unrec_blend_det_frac_blended',
    'shred_frac', 'shred_det_frac', 'spurious_frac', 'missed_frac',
}
def format_display_label(label):
    """Return a cleaner display-only label so it becomes gold (i <= 25.30) instead of gold + 0 (i <= 25.30)"""
    match = re.match(r'^\s*([^+]+?)\s*\+\s*\d+\s*\((i\s*<=\s*[^)]+)\)\s*$', label)
    if match:
        mag_name = match.group(1).strip()
        cutoff = match.group(2).strip()
        return f'{mag_name} ({cutoff})'
    return label

def format_absolute_delta(value, baseline):
    """Format absolute change from baseline as +X.XXXX / -X.XXXX."""
    if np.isnan(value) or np.isnan(baseline):
        return 'N/A'
    return f'{(value - baseline):+.4f}'

def format_value_with_delta(value, baseline):
    """Format value plus its absolute delta vs baseline."""
    if np.isnan(value):
        return 'N/A'
    delta_str = format_absolute_delta(value, baseline)
    if delta_str == 'N/A':
        return f'{value:.4f}'
    return f'{value:.4f} ({delta_str})'

# ============================================================================
# Core ranking logic
# ============================================================================
def load_and_merge_configs(csv_infos, pipeline='dd', linking_length='1.0'):
    """
    Load all CSVs and merge into a single DataFrame with one row per
    (score_thresh, nms_thresh) and columns for each config's metrics.

    Returns
    -------
    merged : pd.DataFrame
        Indexed by (score_thresh, nms_thresh).
    config_labels : list of str
    """
    dfs = []
    labels = []
    for info in csv_infos:
        df = pd.read_csv(info['path'])
        label = info['label']
        labels.append(label)
        # Extract just the columns for this pipeline + LL
        cols_to_keep = ['score_thresh', 'nms_thresh']
        metric_cols = {}
        for col in df.columns:
            if col.startswith(f'{pipeline}_') and col.endswith(f'_{linking_length}'):
                # dd_completeness_1.0 -> completeness
                metric_name = col[len(pipeline)+1:-(len(linking_length)+1)]
                new_col = f'{metric_name}_{label}'
                metric_cols[col] = new_col
                cols_to_keep.append(col)

        subset = df[cols_to_keep].copy()
        subset.rename(columns=metric_cols, inplace=True)
        subset = subset.set_index(['score_thresh', 'nms_thresh'])
        dfs.append(subset)

    merged = pd.concat(dfs, axis=1)
    return merged, labels

def apply_constraints(merged, config_labels, constraints):
    """
    Filter combos that violate any constraint across ANY config.

    Parameters
    ----------
    constraints : list of str
        e.g. ["purity >= 0.85", "shred_frac <= 0.05"]

    Returns
    -------
    mask : pd.Series of bool (True = passes all constraints)
    """
    mask = pd.Series(True, index=merged.index)
    for constraint_str in constraints:
        # Parse "metric_name >= value" or "metric_name <= value"
        parts = constraint_str.strip().split()
        if len(parts) != 3:
            print(f"WARNING: couldn't parse constraint '{constraint_str}', skipping")
            continue
        metric_name, op, threshold = parts[0], parts[1], float(parts[2])
        for label in config_labels:
            col = f'{metric_name}_{label}'
            if col not in merged.columns:
                continue
            if op == '>=':
                mask &= merged[col] >= threshold
            elif op == '<=':
                mask &= merged[col] <= threshold
            elif op == '>':
                mask &= merged[col] > threshold
            elif op == '<':
                mask &= merged[col] < threshold
            else:
                print(f"WARNING: unknown operator '{op}' in constraint '{constraint_str}'")
    return mask

def compute_percentile_ranks(merged, config_labels, metric='f1'):
    """
    For each config, rank all combos by the given metric (as percentile 0-100)
    Higher percentile = better unless it's a LOWER_IS_BETTER metric, in which case we invert the ranking
    Returns
    -------
    rank_df : pd.DataFrame
        Columns: one per config label, values are percentile ranks.
    """
    rank_cols = {}
    for label in config_labels:
        col = f'{metric}_{label}'
        if col not in merged.columns:
            continue
        values = merged[col]
        if metric in LOWER_IS_BETTER:
            # Lower is better --> invert ranking
            rank_cols[label] = values.rank(ascending=False, pct=True) * 100 # getting percentile w/ pct=True
        else:
            rank_cols[label] = values.rank(ascending=True, pct=True) * 100
    return pd.DataFrame(rank_cols, index=merged.index)


def rank_combos(merged, config_labels, primary_metric='f1',
                strategy='minimax', constraints=None):
    """
    Rank all (score, nms) combos by robustness across configs.

    Parameters
    ----------
    merged : pd.DataFrame
        From load_and_merge_configs().
    config_labels : list of str
    primary_metric : str
        Metric to optimize (e.g. 'f1', 'completeness').
    strategy : str
        'minimax' — rank by worst-case percentile across configs. for example, 
            Step 1 (Min over configs): Combo A gets 100 on Gold and 0 on Power Law (min = 0)
                                        Combo B gets 50 on Gold and 50 on Power Law (min = 50)
            Step 2 (Max over combos): We compare Combo A's worst-case (0) and 
                                        Combo B's worst-case (50) and we want to maximize this value, 
                                        so we pick Combo B
            This strat basically prioritizes combos that perform reasonably well across all mag limits, 
            rather than excelling in some and performing poorly in others.
        
        'mean' — rank by mean percentile across configs. for example,
            Step 1 (Mean over configs): Combo B gets 65 on Gold and 65 on Power Law (mean = 65)
                                        Combo C gets 90 on Gold and 50 on Power Law (mean = 70)
            Step 2 (Max over combos): We compare Combo B's average (65) and 
                                        Combo C's average (70) and we want to maximize this value, 
                                        so we pick Combo C
            This strat basically prioritizes the best overall average performance, even if it 
            means accepting a poor result at one specific mag limit (like faint objects) because 
            it excels so much on another
            
        'min_value' — rank by worst raw value across configs
            Step 1 (Min raw value over configs): Combo B's raw F1 is 0.82 on Gold and 0.75 on Power Law (min = 0.75)
                                                Combo C's raw F1 is 0.87 on Gold and 0.60 on Power Law (min = 0.60)
            Step 2 (Max over combos): We compare Combo B's worst raw score (0.75) and 
                                        Combo C's worst raw score (0.60) and we want to maximize this value, 
                                        so we pick Combo B (if lower is better, worst is max, and we minimize it).
            This strat completely ignores relative percentiles and acts as a strict safety net, 
            guaranteeing that your actual metric (e.g., F1 score) never drops below (or exceeds) a certain 
            absolute baseline value at any mag limit
            
    constraints : list of str or None
        e.g.: ["purity >= 0.85", "shred_frac <= 0.05"]

    Returns
    -------
    results : pd.DataFrame
        Sorted by rank (best first), with columns for each config's
        metric value, percentile, and the aggregate score.
    """
    # Apply constraints
    if constraints:
        mask = apply_constraints(merged, config_labels, constraints)
        n_filtered = (~mask).sum()
        if n_filtered > 0:
            print(f"  Filtered out {n_filtered}/{len(merged)} combos "
                  f"that violate constraints")
        working = merged[mask].copy()
    else:
        working = merged.copy()
    if len(working) == 0:
        print("ERROR: No combos pass all constraints!")
        return pd.DataFrame()
    # Get percentile ranks
    ranks = compute_percentile_ranks(working, config_labels, primary_metric)
    # Get raw values
    raw_values = {}
    for label in config_labels:
        col = f'{primary_metric}_{label}'
        if col in working.columns:
            raw_values[label] = working[col]
    # Aggregate
    if strategy == 'minimax':
        working['aggregate_score'] = ranks.min(axis=1)
        score_label = 'Worst-case percentile'
    elif strategy == 'mean':
        working['aggregate_score'] = ranks.mean(axis=1)
        score_label = 'Mean percentile'
    elif strategy == 'min_value':
        raw_df = pd.DataFrame(raw_values)
        if primary_metric in LOWER_IS_BETTER:
            working['aggregate_score'] = -raw_df.max(axis=1)  # negate so sorting works (-0.08 > -0.12)
        else:
            working['aggregate_score'] = raw_df.min(axis=1)
        score_label = 'Worst raw value'
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    # Build results table
    result_cols = {}
    for label in config_labels:
        col = f'{primary_metric}_{label}'
        if col in working.columns:
            result_cols[f'{primary_metric} ({label})'] = working[col]
        if label in ranks.columns:
            result_cols[f'pctile ({label})'] = ranks[label]

    result_cols[score_label] = working['aggregate_score']
    results = pd.DataFrame(result_cols, index=working.index)
    results = results.sort_values(score_label, ascending=False)
    return results

def load_compare_baseline_values(csv_infos, pipeline='lsst', metric='f1', linking_length='1.0'):
    """Load one baseline value per config for a comparison pipeline metric."""
    baseline_vals = {}
    target_col = f'{pipeline}_{metric}_{linking_length}'
    for info in csv_infos:
        label = info['label']
        df = pd.read_csv(info['path'])
        if target_col not in df.columns:
            baseline_vals[label] = np.nan
            continue
        vals = df[target_col].copy()
        baseline_vals[label] = float(vals.iloc[0]) if not vals.empty else np.nan
    return baseline_vals


def load_compare_baseline_metrics(csv_infos, pipeline='lsst', linking_length='1.0'):
    """Load baseline values for all SECONDARY_METRICS by config label."""
    metric_baselines = {metric: {} for metric in SECONDARY_METRICS}
    for info in csv_infos:
        label = info['label']
        df = pd.read_csv(info['path'])
        for metric in SECONDARY_METRICS:
            col = f'{pipeline}_{metric}_{linking_length}'
            if col not in df.columns:
                metric_baselines[metric][label] = np.nan
                continue
            vals = df[col].dropna()
            metric_baselines[metric][label] = float(vals.iloc[0]) if not vals.empty else np.nan
    return metric_baselines

# ============================================================================
# Secondary metrics report
# ============================================================================
SECONDARY_METRICS = [
    'completeness', 'purity', 'f1',
    'blend_loss_frac', 'unrec_blend_frac_total', 'unrec_blend_frac_blended', 'unrec_blend_frac_matched',
    'unrec_blend_det_frac_total', 'unrec_blend_det_frac_blended',
    'shred_frac', 'shred_det_frac', 'spurious_frac', 'missed_frac', 'resolved_frac'
]

def print_secondary_metrics(merged, config_labels, score_thresh, nms_thresh,
                            compare_metric_baselines=None, compare_pipeline=None):
    """Print all metrics for a specific combo across all configs."""
    if (score_thresh, nms_thresh) not in merged.index:
        print(f"  Combo ({score_thresh}, {nms_thresh}) not found!")
        return
    row = merged.loc[(score_thresh, nms_thresh)]
    display_labels = [format_display_label(label) for label in config_labels]
    value_col_width = max(18, max(len(label) for label in display_labels))

    if compare_metric_baselines is not None:
        max_cell_len = 0
        for metric in SECONDARY_METRICS:
            for label in config_labels:
                col = f'{metric}_{label}'
                val = row.get(col, np.nan)
                baseline_val = compare_metric_baselines.get(metric, {}).get(label, np.nan)
                cell = format_value_with_delta(val, baseline_val)
                max_cell_len = max(max_cell_len, len(cell))
        value_col_width = max(value_col_width, max_cell_len)

    print(f"\n  All metrics for s={score_thresh}, n={nms_thresh}:")
    if compare_metric_baselines is not None and compare_pipeline is not None:
        print(f"  (values shown as DD (delta vs {compare_pipeline.upper()} baseline))")
    print(f"  {'Metric':<30}", end='')
    for label in display_labels:
        print(f"  {label:>{value_col_width}}", end='')
    print()
    print(f"  {'─'*30}", end='')
    for _ in config_labels:
        print(f"  {'─'*value_col_width}", end='')
    print()
    for metric in SECONDARY_METRICS:
        print(f"  {metric:<30}", end='')
        for label in config_labels:
            col = f'{metric}_{label}'
            val = row.get(col, np.nan)
            if compare_metric_baselines is not None:
                baseline_val = compare_metric_baselines.get(metric, {}).get(label, np.nan)
                cell = format_value_with_delta(val, baseline_val)
                print(f"  {cell:>{value_col_width}}", end='')
            else:
                if np.isnan(val):
                    print(f"  {'N/A':>{value_col_width}}", end='')
                else:
                    print(f"  {val:>{value_col_width}.4f}", end='')
        print()

# ============================================================================
# Stability check
# ============================================================================
def check_stability(merged, config_labels, score_thresh, nms_thresh, primary_metric='f1'):
    """
    Check how stable a combo is by looking at neighbors. This just tells us
    if our chosen combo sits on a plateau or a sharp peak. Because 
    if it's a sharp peak, small changes in thresholds cause big metric swings and the combo 
    won't generalize well to unseen data.
    
    Returns the metric values for the combo and its immediate neighbors
    """
    all_scores = sorted(merged.index.get_level_values('score_thresh').unique())
    all_nms = sorted(merged.index.get_level_values('nms_thresh').unique())

    s_idx = all_scores.index(score_thresh) if score_thresh in all_scores else -1
    n_idx = all_nms.index(nms_thresh) if nms_thresh in all_nms else -1

    if s_idx < 0 or n_idx < 0:
        return None
    # Collect neighbors (+- 1 step in each direction)
    neighbor_vals = {}
    for ds in [-1, 0, 1]:
        for dn in [-1, 0, 1]:
            si = s_idx + ds
            ni = n_idx + dn
            if 0 <= si < len(all_scores) and 0 <= ni < len(all_nms):
                s, n = all_scores[si], all_nms[ni]
                if (s, n) in merged.index:
                    vals = {}
                    for label in config_labels:
                        col = f'{primary_metric}_{label}'
                        if col in merged.columns:
                            vals[label] = merged.loc[(s, n), col]
                    neighbor_vals[(s, n)] = vals
    return neighbor_vals


def print_stability_report(neighbor_vals, score_thresh, nms_thresh, config_labels):
    """Pretty-print stability of a combo and its neighbors."""
    center = (score_thresh, nms_thresh)
    display_labels = [format_display_label(label) for label in config_labels]
    value_col_width = max(18, max(len(label) for label in display_labels))

    print(f"\n  Stability neighborhood for s={score_thresh}, n={nms_thresh}:")
    print(f"  {'(score, nms)':<16}", end='')
    for label in display_labels:
        print(f"  {label:>{value_col_width}}", end='')
    print()
    print(f"  {'─'*16}", end='')
    for _ in config_labels:
        print(f"  {'─'*value_col_width}", end='')
    print()

    all_vals = {label: [] for label in config_labels}
    for (s, n), vals in sorted(neighbor_vals.items()):
        marker = ' *' if (s, n) == center else '  '
        print(f"{marker}({s:.2f}, {n:.2f})  ", end='')
        for label in config_labels:
            v = vals.get(label, np.nan)
            print(f"  {v:>{value_col_width}.4f}", end='')
            if not np.isnan(v):
                all_vals[label].append(v)
        print()

    print(f"\n  Neighbor spread (std):", end='')
    for label in config_labels:
        if all_vals[label]:
            print(f"  {np.std(all_vals[label]):>{value_col_width}.4f}", end='')
    print()


# ============================================================================
# CLI
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description='Find the most robust threshold combo across evaluation configs')
    parser.add_argument('--metrics-dir', required=True,
                        help='Base directory for relative CSV paths')
    parser.add_argument('--csv-files', nargs='+', required=True,
                        help='Paths to CSV files (relative to --metrics-dir by default)')
    parser.add_argument('--mag-limits', nargs='+', required=True,
                        help='Magnitude limits for each CSV file (e.g., gold power_law nominal)')
    parser.add_argument('--buffers', nargs='+', type=int, required=True,
                        help='Buffer values for each CSV file (e.g., 0 1 2)')
    parser.add_argument('--absolute-paths', action='store_true',
                        help='Treat CSV and output paths as absolute instead of relative to --metrics-dir')
    parser.add_argument('--pipeline', default='dd', choices=['dd', 'lsst'],
                        help='Which pipeline to plot metrics for (default: dd)')
    parser.add_argument('--compare-pipeline', default='lsst', choices=['dd', 'lsst'],
                        help='Optional pipeline to display as a fixed comparison baseline')
    parser.add_argument('--linking-length', default='1.0',
                        help='Linking length (default: 1.0)')
    parser.add_argument('--primary-metric', default='f1',
                        help='Metric to optimize (default: f1)')
    parser.add_argument('--strategy', default='minimax',
                        choices=['minimax', 'mean', 'min_value'],
                        help='Aggregation strategy (default: minimax)')
    parser.add_argument('--constraint', action='append', default=[],
                        dest='constraints',
                        help='Constraint like "purity >= 0.85" (repeatable)')
    parser.add_argument('--top', type=int, default=15,
                        help='Number of top combos to show (default: 15)')
    return parser.parse_args()


def main():
    args = parse_args()
    # --- Build csv_infos ---
    if args.mag_limits and len(args.mag_limits) != len(args.csv_files):
        print(f"Error: Number of mag_limits ({len(args.mag_limits)}) must match "
              f"number of CSV files ({len(args.csv_files)}).")
        return
    if args.buffers and len(args.buffers) != len(args.csv_files):
        print(f"Error: Number of buffers ({len(args.buffers)}) must match "
              f"number of CSV files ({len(args.csv_files)}).")
        return
    # Root dir for relative paths
    if not args.absolute_paths and args.metrics_dir:
        root_dir = os.path.expanduser(args.metrics_dir)
    else:
        root_dir = None
    csv_infos = []
    for idx, path in enumerate(args.csv_files):
        # Resolve path (relative to root_dir or absolute)
        if root_dir and not args.absolute_paths:
            full_path = os.path.join(root_dir, path)
        else:
            full_path = os.path.expanduser(path)
        # mag_limit and buffer from args
        ml = args.mag_limits[idx]
        buf = args.buffers[idx]
        # label
        mag_val = MAG_LIMITS.get(ml, None)
        label = (f'{ml} + {buf} (i <= {mag_val + buf:.2f})'
                 if mag_val else f'{ml} buf{buf}')
        csv_infos.append({'path': full_path, 'mag_limit': ml,
                          'buffer': buf, 'label': label})
    if not csv_infos:
        print("No CSV files found.")
        return

    print("=" * 70)
    print("THRESHOLD ROBUSTNESS RANKING")
    print("=" * 70)
    print(f"\nConfigs ({len(csv_infos)}):")
    for info in csv_infos:
        print(f"  {info['label']}: {info['path']}")
    print(f"\nSettings:")
    print(f"  Pipeline:        {args.pipeline.upper()}")
    if args.compare_pipeline:
        print(f"  Compare:         {args.compare_pipeline.upper()} ({args.primary_metric})")
    print(f"  Linking length:  {args.linking_length}\"")
    print(f"  Primary metric:  {args.primary_metric}")
    print(f"  Strategy:        {args.strategy}")
    if args.constraints:
        print(f"  Constraints:     {args.constraints}")

    # --- Load and merge ---
    merged, config_labels = load_and_merge_configs(
        csv_infos, pipeline=args.pipeline,
        linking_length=args.linking_length
    )
    print(f"\nLoaded {len(merged)} unique (score, nms) combos")

    # --- Rank ---
    results = rank_combos(
        merged, config_labels,
        primary_metric=args.primary_metric,
        strategy=args.strategy,
        constraints=args.constraints if args.constraints else None
    )
    if len(results) == 0:
        return

    compare_baseline = None
    compare_metric_baselines = None
    if args.compare_pipeline:
        compare_baseline = load_compare_baseline_values(
            csv_infos,
            pipeline=args.compare_pipeline,
            metric=args.primary_metric,
            linking_length=args.linking_length,
        )
        compare_metric_baselines = load_compare_baseline_metrics(
            csv_infos,
            pipeline=args.compare_pipeline,
            linking_length=args.linking_length,
        )
        compare_vals = [compare_baseline.get(label, np.nan) for label in config_labels]
        if not np.all(np.isnan(compare_vals)):
            compare_mean = float(np.nanmean(compare_vals))

    # --- Print top combos ---
    top_n = min(args.top, len(results))
    print(f"\n{'=' * 70}")
    print(f"TOP {top_n} COMBOS (by {args.strategy} of {args.primary_metric})")
    print(f"{'=' * 70}")

    display_labels = [format_display_label(label) for label in config_labels]
    col_width = max(15, max(len(label) for label in display_labels))
    value_col_width = max(col_width, 20) if compare_baseline is not None else col_width

    print(f"\n{'Rank':<6} {'Score':<8} {'NMS':<8}", end='')
    for label in display_labels:
        print(f"  {label:>{value_col_width}}", end='')
    agg_col = results.columns[-1]  # last col is the aggregate score
    print(f"  {agg_col:>20}")

    print(f"{'─'*6} {'─'*8} {'─'*8}", end='')
    for _ in config_labels:
        print(f"  {'─'*value_col_width}", end='')
    print(f"  {'─'*20}")
    if compare_baseline is not None:
        print(f"{args.compare_pipeline.upper():<6} {'-':<8} {'-':<8}", end='')
        for label in config_labels:
            v = compare_baseline.get(label, np.nan)
            if np.isnan(v):
                print(f"  {'N/A':>{value_col_width}}", end='')
            else:
                print(f"  {v:>{value_col_width}.4f}", end='')
        print(f"  {'-':>20}")
    for rank, ((score, nms), row) in enumerate(results.head(top_n).iterrows(), 1):
        print(f"{rank:<6} {score:<8.2f} {nms:<8.2f}", end='')
        for label in config_labels:
            col = f'{args.primary_metric} ({label})'
            val = row.get(col, np.nan)
            if compare_baseline is not None:
                baseline_val = compare_baseline.get(label, np.nan)
                disp_val = format_value_with_delta(val, baseline_val)
                print(f"  {disp_val:>{value_col_width}}", end='')
            else:
                print(f"  {val:>{value_col_width}.4f}", end='')
        print(f"  {row[agg_col]:>20.1f}")

    # --- Detailed report for #1 ---
    best_score, best_nms = results.index[0]
    print(f"\n{'=' * 70}")
    print(f"RECOMMENDED: score={best_score}, nms={best_nms}")
    print(f"{'=' * 70}")

    # Secondary metrics
    print_secondary_metrics(
        merged,
        config_labels,
        best_score,
        best_nms,
        compare_metric_baselines=compare_metric_baselines,
        compare_pipeline=args.compare_pipeline,
    )
    # Stability
    neighbors = check_stability(merged, config_labels, best_score, best_nms,
                                primary_metric=args.primary_metric)
    if neighbors:
        print_stability_report(neighbors, best_score, best_nms, config_labels)

    # --- Also check #2 and #3 if they exist ---
    for runner_up in range(1, min(3, len(results))):
        s, n = results.index[runner_up]
        print(f"\n{'─' * 70}")
        print(f"Runner-up #{runner_up + 1}: score={s}, nms={n}")
        print_secondary_metrics(merged, config_labels, s, n)
        neighbors = check_stability(merged, config_labels, s, n,
                                    primary_metric=args.primary_metric)
        if neighbors:
            print_stability_report(neighbors, s, n, config_labels)

    print(f"\n{'=' * 70}")
    print(f"This is the recommended threshold for test set inference:")
    print(f"  --score-thresh {best_score} --nms-thresh {best_nms}")
    print(f"{'=' * 70}")

if __name__ == '__main__':
    main()