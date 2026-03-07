"""
plot_grid_search.py — CLI for plotting heatmaps from grid search CSV results

Produces:
    1. Individual heatmap PNGs per (metric, mag_limit + buffer, linking_length)
    2. Comparison grid figures per (metric, linking_length) with subplots
       for each mag_limit + buffer combo side by side

Usage:
    # With metrics-dir as base (paths relative to metrics-dir by default)
    python plot_grid_search.py --metrics-dir ~/lsst_runs/lsst5_30k_4h200_bs192_ep50/metrics \
                                --csv-files gold_buf0/gs_metrics.csv gold_buf1/gs_metrics.csv gold_buf2/gs_metrics.csv \
                                --output-dir gold_buf0/gs_hmaps gold_buf1/gs_hmaps gold_buf2/gs_hmaps \
                                --mag-limits gold gold gold \
                                --buffers 0 1 2

    python plot_grid_search.py --metrics-dir ~/lsst_runs/lsst5_30k_4h200_bs192_ep50/metrics --csv-files gold_buf0/gs_metrics.csv gold_buf1/gs_metrics.csv gold_buf2/gs_metrics.csv --output-dir gold_buf0/gs_hmaps gold_buf1/gs_hmaps gold_buf2/gs_hmaps --mag-limits gold gold gold --buffers 0 1 2 --no-comparison
    python plot_grid_search.py --metrics-dir ~/lsst_runs/lsst5_30k_4h200_bs192_ep50/metrics --csv-files gold_buf0/gs_metrics.csv gold_buf1/gs_metrics.csv gold_buf2/gs_metrics.csv --output-dir gold_cmp --mag-limits gold gold gold --buffers 0 1 2 --metrics completeness purity f1 --no-individual

    python plot_grid_search.py --metrics-dir ~/lsst_runs/clip5_30k_4h200_bs32_ep50/metrics --csv-files gold_buf0/gs_metrics.csv gold_buf1/gs_metrics.csv gold_buf2/gs_metrics.csv --output-dir gold_buf0/gs_hmaps gold_buf1/gs_hmaps gold_buf2/gs_hmaps --mag-limits gold gold gold --buffers 0 1 2 --no-comparison
    python plot_grid_search.py --metrics-dir ~/lsst_runs/clip5_30k_4h200_bs32_ep50/metrics --csv-files gold_buf0/gs_metrics.csv gold_buf1/gs_metrics.csv gold_buf2/gs_metrics.csv --output-dir gold_cmp --mag-limits gold gold gold --buffers 0 1 2 --metrics completeness purity f1 --no-individual

    # With absolute paths (use --absolute-paths flag)
    python plot_grid_search.py --csv-files /path/to/file1.csv /path/to/file2.csv \
                                --output-dir /path/to/output --absolute-paths \
                                --mag-limits gold nominal \
                                --buffers 0 1
"""

import argparse
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from plot_metrics import (
    plot_heatmap, plot_heatmap_comparison,
    METRIC_LABELS
)
from grid_search import FRACTION_METRICS

# ============================================================================
# Constants
# ============================================================================
MAG_LIMITS = {
    'power_law': 26.22,
    'gold': 25.3,
    'nominal': 26.42,
}

# ============================================================================
# Individual heatmaps
# ============================================================================
def plot_individual_heatmaps(csv_info, output_dir, linking_lengths, 
                             pipeline='dd', metrics=None, absolute_paths=False):
    """Generate one heatmap PNG per (metric, linking_length) for a single CSV"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_info['path'])
    label = csv_info['label']
    ml = csv_info['mag_limit']
    buf = csv_info['buffer']

    if metrics is None:
        metrics = FRACTION_METRICS
    for metric in metrics:
        for ll in linking_lengths:
            col = f'{pipeline}_{metric}_{ll}'
            if col not in df.columns:
                print(f"  Skipping {metric} (LL={ll}\") - column '{col}' not found in CSV")
                continue
            metric_label = METRIC_LABELS.get(metric, metric)
            title = f'{pipeline.upper()} {metric_label}\nLL={ll}" | {label}'
            fig, _, _ = plot_heatmap(df, col, title=title)
            if absolute_paths:
                out_path = output_dir / f'{pipeline}_{metric}_{ll}_{ml}_buf{buf}.png'
            else:
                out_path = output_dir / f'{pipeline}_{metric}_{ll}.png'
            fig.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'  Saved: {out_path.name}')


# ============================================================================
# Comparison grid figures
# ============================================================================
def plot_comparison_grids(csv_infos, output_dir, pipeline='dd',
                          metrics=None, linking_lengths=None):
    """
    For each (metric, linking_length), create a figure with one subplot
    per mag_limit + buffer combo, side by side for easy comparison.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if metrics is None:
        metrics = FRACTION_METRICS

    loaded = []
    for info in csv_infos:
        df = pd.read_csv(info['path'])
        loaded.append((info, df))
    if not loaded:
        print("No CSV files to plot.")
        return

    for metric in metrics:
        for ll in linking_lengths:
            col = f'{pipeline}_{metric}_{ll}'
            valid = [(info, df) for info, df in loaded if col in df.columns]
            if not valid:
                print(f"  Skipping {metric} (LL={ll}\") - column '{col}' not found in any CSV")
                continue

            dfs = [df for _, df in valid]
            labels = [info['label'] for info, _ in valid]
            metric_label = METRIC_LABELS.get(metric, metric)
            title = f'{pipeline.upper()} {metric_label} (LL={ll}")'

            out_path = output_dir / f'compare_{pipeline}_{metric}_{ll}.png'
            plot_heatmap_comparison(dfs, labels, col,
                                   output_path=str(out_path), title=title)
            print(f'  Saved: {out_path.name}')


# ============================================================================
# CLI
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot heatmaps from grid search CSV results')
    parser.add_argument('--metrics-dir',
                        help='Base directory for relative CSV and output paths')
    parser.add_argument('--csv-files', nargs='+', required=True,
                        help='Paths to CSV files (relative to --metrics-dir by default)')
    parser.add_argument('--output-dir', nargs='+', required=True,
                        help='Output directory/directories (one per CSV file, or single dir for all)')
    parser.add_argument('--mag-limits', nargs='+', required=True,
                        help='Magnitude limits for each CSV file (e.g., gold power_law nominal)')
    parser.add_argument('--buffers', nargs='+', type=int, required=True,
                        help='Buffer values for each CSV file (e.g., 0 1 2)')
    parser.add_argument('--absolute-paths', action='store_true',
                        help='Treat CSV and output paths as absolute instead of relative to --metrics-dir')
    parser.add_argument('--pipeline', default='dd', choices=['dd', 'lsst'],
                        help='Which pipeline to plot metrics for (default: dd)')
    parser.add_argument('--metrics', nargs='+', default=None,
                        help='Specific metrics to plot (e.g., completeness purity f1). '
                             'If not specified, plots all FRACTION_METRICS from grid_search module')
    parser.add_argument('--linking-lengths', nargs='+', type=float,
                        default=[1.0],
                        help='FOF linking lengths in arcseconds (default: 1.0)')
    parser.add_argument('--no-individual', action='store_true',
                        help='Skip generating individual heatmaps for each CSV file')
    parser.add_argument('--no-comparison', action='store_true',
                        help='Skip generating comparison grid figures across CSV files')
    return parser.parse_args()


def main():
    args = parse_args()
    # Validate mag_limits and buffers if provided
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
    # Process CSV files
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

    # Handle output directories
    if args.output_dir:
        output_dirs = args.output_dir
        # If single output dir provided, use it for all CSVs
        if len(output_dirs) == 1 and len(csv_infos) > 1:
            output_dirs = [output_dirs[0]] * len(csv_infos)
        elif len(output_dirs) != len(csv_infos):
            print(f"Error: Number of output dirs ({len(output_dirs)}) must match "
                  f"number of CSV files ({len(csv_infos)}) or be 1.")
            return
    else:
        # Default output directory
        if root_dir:
            output_dirs = [os.path.join(root_dir, 'gs_hmaps')] * len(csv_infos)
        else:
            output_dirs = ['./gs_hmaps'] * len(csv_infos)
    
    # Resolve output directory paths
    resolved_output_dirs = []
    for out_dir in output_dirs:
        if root_dir and not args.absolute_paths:
            resolved_output_dirs.append(os.path.join(root_dir, out_dir))
        else:
            resolved_output_dirs.append(os.path.expanduser(out_dir))

    print(f"Found {len(csv_infos)} CSV file(s):")
    for info, out_dir in zip(csv_infos, resolved_output_dirs):
        print(f"  {info['label']}: {info['path']}")
        print(f"    Output: {out_dir}")
    print(f"Pipeline: {args.pipeline.upper()}")

    if not args.no_individual:
        print(f"\nGenerating individual heatmaps...")
        for info, out_dir in zip(csv_infos, resolved_output_dirs):
            subdir = Path(out_dir) / 'individual'
            print(f"\n  [{info['label']}]")
            plot_individual_heatmaps(info, subdir, linking_lengths=args.linking_lengths, 
                                        pipeline=args.pipeline,
                                        metrics=args.metrics,
                                        absolute_paths=args.absolute_paths)

    if not args.no_comparison and len(csv_infos) > 1:
        print(f"\nGenerating comparison grids...")
        # Use first output directory for comparison plots
        comp_dir = Path(resolved_output_dirs[0]) / 'comparison'
        plot_comparison_grids(csv_infos, comp_dir, pipeline=args.pipeline,
                              metrics=args.metrics,
                              linking_lengths=args.linking_lengths)
    elif not args.no_comparison and len(csv_infos) == 1:
        print("\nSkipping comparison grids (only 1 CSV file).")

    print(f"\nDone! Heatmaps saved.")

if __name__ == '__main__':
    main()
