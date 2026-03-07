"""
grid_search.py — Grid search over score/NMS thresholds for various metrics

Runs FOF + Stage 1/2 classification for every (score_thresh, nms_thresh) combo,
computes scalar metrics, and saves results to a CSV for later plotting/analysis

Usage:
    python grid_search.py \
        --run-dir ~/lsst_runs/lsst5_30k_4h200_bs192_ep50 \
        --test-cats-dir ~/lsst_data/test_cats_lvl5/val_4k/ \
        --output ~/lsst_runs/lsst5_30k_4h200_bs192_ep50/metrics/gold_buf0/gs_metrics.csv \
        --mag-limit gold \
        --buffer 0

    # Background with logging:
    python grid_search.py [args] > gs_out.txt 2>&1 &
"""

import argparse
import json
import os
import itertools
import time
import warnings

import numpy as np
import pandas as pd
from multiprocessing import Lock, Pool
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

from metrics import compute_scalar_metrics
from fof_classify import (
    find_matches, build_table, get_stage1_counts,
    classify_stg1, classify_stage2
)
warnings.filterwarnings('ignore')
file_lock = Lock()
MAG_LIMITS = {
    'power_law': 26.22,
    'gold': 25.3,
    'nominal': 26.42,
}
# ============================================================================
# Helper functions for CSV output
# ============================================================================
# CSV column definitions
FRACTION_METRICS = [
    'completeness', 'purity', 'f1',
    'blend_loss_frac',
    'unrec_blend_frac_total', 'unrec_blend_frac_blended',
    'unrec_blend_det_frac_total', 'unrec_blend_det_frac_blended',
    'resolved_rate',
    'shred_frac', 'shred_det_frac',
    'spurious_frac', 'missed_frac',
]
COUNT_METRICS = ['n_truth', 'n_det', 'n_matched_truth', 'n_matched_det']

def metrics_to_csv_row(score_thresh, nms_thresh, all_metrics, elapsed,
                       pipelines=('dd',), linking_lengths=('1.0', '2.0'),
                       include_counts=False):
    """
    Format a single CSV row from the metrics dictionaries.

    Parameters
    ----------
    all_metrics : dict
        {(prefix, ll_str): metrics_dict} as returned by compute_grid_metrics.
    """
    metrics_list = FRACTION_METRICS + (COUNT_METRICS if include_counts else [])
    parts = [f'{score_thresh}', f'{nms_thresh}']
    for prefix in pipelines:
        for ll in linking_lengths:
            m = all_metrics.get((prefix, ll), {})
            for metric_name in metrics_list:
                val = m.get(metric_name, np.nan)
                if isinstance(val, (int, np.integer)):
                    parts.append(f'{val}')
                else:
                    parts.append(f'{val:.6f}')
    parts.append(f'{elapsed:.2f}')
    return ','.join(parts) + '\n'

def csv_header(pipelines=('dd',), linking_lengths=('1.0', '2.0'),
               include_counts=False):
    """Return the CSV header line (with newline)."""
    cols = ['score_thresh', 'nms_thresh']
    metrics = FRACTION_METRICS + (COUNT_METRICS if include_counts else [])
    for prefix in pipelines:
        for ll in linking_lengths:
            for m in metrics:
                cols.append(f'{prefix}_{m}_{ll}')
    cols.append('elapsed_sec')
    return ','.join(cols) + '\n'

# ============================================================================
# Column config
# ============================================================================

def get_col_config():
    """Return the column config dict for build_table()
    Since we're not saving any intermediate outputs, 
    we literally just need to give RA and DEC from each catalog 
    to fof_classify.py
    """
    return {
        'dd_det': {
            'ra': 'ra',
            'dec': 'dec'
        },
        'lsst_det': {
            'ra': 'ra',
            'dec': 'dec'
        },
        'lsst_truth': {
            'ra': 'ra',
            'dec': 'dec'
        }
    }


# ============================================================================
# LSST pre-computation (run once)
# ============================================================================
def compute_lsst_metrics(lsst_det_cat, truth_cat, col_config,
                         linking_lengths):
    """
    Pre-compute LSST metrics for all linking lengths.
    Since LSST catalogs are constant across all DD thresholds,
    we only need to run this once.

    Returns
    -------
    dict : {linking_length_str: metrics_dict}
    """
    cats = {
        'lsst_det': lsst_det_cat,
        'lsst_truth': truth_cat,
    }
    lsst_results = {}
    for ll in linking_lengths:
        ll_str = str(ll)
        print(f"  LSST FOF for linking length {ll_str}\"...")
        fof_result = find_matches(cats, linking_length=ll, max_friends=None)
        results_df = fof_result.to_pandas()
        analysis_df = build_table(results_df, cats, col_config)
        counts = get_stage1_counts(
            analysis_df,
            expected_keys=['lsst_det', 'lsst_truth'],
            rename_map={'lsst_truth': 'n_truth', 'lsst_det': 'n_lsst'}
        )
        counts = classify_stg1(counts, prefix='lsst',
                               det_col_name='n_lsst', truth_col_name='n_truth')
        analysis_df = classify_stage2(analysis_df, counts, det_prefix='lsst', match_rad=ll / 2, verbose=False)
        metrics = compute_scalar_metrics(analysis_df, 'lsst')
        lsst_results[ll_str] = metrics
        print(f"    Comp={metrics['completeness']:.4f}, "
              f"Purity={metrics['purity']:.4f}, "
              f"F1={metrics['f1']:.4f}")
    return lsst_results

# ============================================================================
# Per-combo worker
# ============================================================================
def process_single_combo(score_thresh, nms_thresh, truth_cat,
                         run_dir, output_file, col_config, lsst_results,
                         linking_lengths):
    """Process a single (score_thresh, nms_thresh) combination."""
    try:
        start_time = time.time()
        if run_dir == '/u/yse2/lsst_runs/lsst5_30k_4h200_bs192_ep50':
            pred_fn = os.path.join(run_dir, 'preds', 'eval', 'with_mags',
                            f'pred_s{score_thresh}_n{nms_thresh}.json')
        else:
            pred_fn = os.path.join(run_dir, 'preds', 'eval',
                            f'pred_s{score_thresh}_n{nms_thresh}.json')
        if not os.path.exists(pred_fn):
            print(f"WARNING: File not found: {pred_fn}", flush=True)
            return None

        dd_det_cat = pd.read_json(pred_fn)
        dd_det_cat = dd_det_cat.rename(columns={'ra_kp': 'ra', 'dec_kp': 'dec'})
        cats = {
            'dd_det': dd_det_cat,
            'lsst_truth': truth_cat,
        }

        ll_strs = [str(ll) for ll in linking_lengths]
        results = {}
        for ll in linking_lengths:
            ll_str = str(ll)
            print(f"  s={score_thresh}, n={nms_thresh} | LL={ll_str}\"...", flush=True)
            results_df = find_matches(
                cats, linking_length=ll, max_friends=None
            ).to_pandas()
            analysis_df = build_table(results_df, cats, col_config)
            counts = get_stage1_counts(
                analysis_df,
                expected_keys=['dd_det', 'lsst_truth'],
                rename_map={'lsst_truth': 'n_truth', 'dd_det': 'n_dd'}
            )
            counts = classify_stg1(counts, prefix='dd',
                                   det_col_name='n_dd', truth_col_name='n_truth')
            analysis_df = classify_stage2(analysis_df, counts, det_prefix='dd', match_rad=ll / 2, verbose=False)
            metrics = compute_scalar_metrics(analysis_df, 'dd')
            results[('dd', ll_str)] = metrics
            results[('lsst', ll_str)] = lsst_results[ll_str]

        elapsed = time.time() - start_time

        # Write CSV row (thread-safe)
        with file_lock:
            with open(output_file, 'a') as f:
                f.write(metrics_to_csv_row(
                    score_thresh, nms_thresh, results, elapsed,
                    pipelines=('dd', 'lsst'), linking_lengths=tuple(ll_strs)
                ))

        # Log summary
        parts = [f"DONE: s={score_thresh}, n={nms_thresh}"]
        for ll_str in ll_strs:
            dd_c = results[('dd', ll_str)]['completeness']
            lsst_c = results[('lsst', ll_str)]['completeness']
            parts.append(f"DD({ll_str}\")={dd_c:.4f}")
            parts.append(f"LSST({ll_str}\")={lsst_c:.4f}")
        parts.append(f"{elapsed:.1f}s")
        print(" | ".join(parts), flush=True)
        return results
    except Exception as e:
        print(f"ERROR processing s={score_thresh}, n={nms_thresh}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# CLI
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description='Grid search over score/NMS thresholds for various metrics'
    )
    parser.add_argument('--run-dir', required=True,
                        help='Path to the run directory (e.g. ~/lsst_runs/run_name/)')
    parser.add_argument('--test-cats-dir', required=True,
                        help='Path to test catalogs directory')
    parser.add_argument('--output', required=True,
                        help='Output CSV path (e.g. ~/lsst_runs/run_name/metrics/gold_buf1/val_gs_metrics.csv)')
    parser.add_argument('--mag-limit', default='gold', choices=MAG_LIMITS.keys(),
                        help='Magnitude limit key (default: gold)')
    parser.add_argument('--buffer', type=int, default=1, choices=[0, 1, 2],
                        help='Buffer to add to mag limit (default: 1)')
    parser.add_argument('--linking-lengths', nargs='+', type=float,
                        default=[1.0, 2.0],
                        help='FOF linking lengths in arcseconds (default: 1.0 2.0)')
    parser.add_argument('--score-thresholds', nargs='+', type=float,
                        default=[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                                 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
                        help='Score thresholds to search')
    parser.add_argument('--nms-thresholds', nargs='+', type=float,
                        default=[0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                                 0.55, 0.6, 0.65, 0.7, 0.75],
                        help='NMS thresholds to search')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: min(num_cpus / 2, n_combos))')
    return parser.parse_args()

def main():
    args = parse_args()
    run_dir = os.path.expanduser(args.run_dir)
    test_cats_dir = os.path.expanduser(args.test_cats_dir)
    truth_mag_limit = MAG_LIMITS[args.mag_limit] + args.buffer
    output_file = os.path.expanduser(args.output)
    linking_lengths = args.linking_lengths

    threshold_combos = list(itertools.product(
        args.score_thresholds, args.nms_thresholds
    ))
    cpu_count = len(os.sched_getaffinity(0))
    n_workers = args.workers or min(cpu_count // 2, len(threshold_combos))

    print("=" * 80)
    print("GRID SEARCH FOR OPTIMAL DETECTION COMPLETENESS")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Run directory:    {run_dir}")
    print(f"  Test cats dir:    {test_cats_dir}")
    print(f"  Mag limit:        {args.mag_limit} ({MAG_LIMITS[args.mag_limit]}) "
          f"+ buffer {args.buffer} = {truth_mag_limit:.2f}")
    print(f"  Linking lengths:  {linking_lengths}")
    print(f"  Output file:      {output_file}")
    print(f"  Score thresholds: {len(args.score_thresholds)} values")
    print(f"  NMS thresholds:   {len(args.nms_thresholds)} values")
    print(f"  Total combos:     {len(threshold_combos)}")
    print(f"  CPUs available:   {cpu_count}")
    print(f"  Workers:          {n_workers}")

    # --- Load catalogs ---
    print(f"\nLoading shared catalogs...")
    truth_fn = os.path.join(test_cats_dir, f'test_truth_cat_maglim_{truth_mag_limit:.2f}.parquet')
    print(f"  Truth catalog: {truth_fn}")
    lsst_truth_cat = pd.read_parquet(truth_fn)
    print(f"    {len(lsst_truth_cat):,} truth objects")

    lsst_det_fn = os.path.join(test_cats_dir, 'test_det_cat.json')
    print(f"  LSST detections: {lsst_det_fn}")
    lsst_det_cat = pd.read_json(lsst_det_fn)
    print(f"    {len(lsst_det_cat):,} LSST detections")

    col_config = get_col_config()

    # --- Pre-compute LSST metrics ---
    print(f"\nPre-computing LSST metrics (run once, reuse for all "
          f"{len(threshold_combos)} combos)...")
    lsst_start = time.time()
    lsst_results = compute_lsst_metrics(
        lsst_det_cat, lsst_truth_cat, col_config, linking_lengths
    )
    print(f"LSST pre-computation done in {time.time() - lsst_start:.1f}s")

    # --- Write CSV header ---
    ll_strs = tuple(str(ll) for ll in linking_lengths)
    with open(output_file, 'w') as f:
        f.write(csv_header(
            pipelines=('dd', 'lsst'), linking_lengths=ll_strs
        ))

    # --- Run grid search ---
    print(f"\n{'=' * 80}")
    print("Starting grid search...")
    print(f"{'=' * 80}\n")

    start_time = time.time()
    process_func = partial(
        process_single_combo,
        truth_cat=lsst_truth_cat,
        run_dir=run_dir,
        output_file=output_file,
        col_config=col_config,
        lsst_results=lsst_results,
        linking_lengths=linking_lengths
    )
    print(f"Using {n_workers} parallel workers\n")
    with Pool(processes=n_workers) as pool:
        results = pool.starmap(process_func, threshold_combos)
    # with ProcessPoolExecutor(max_workers=n_workers) as executor:
    #     score_vals = [score for score, _ in threshold_combos]
    #     nms_vals = [nms for _, nms in threshold_combos]
    #     results = list(executor.map(process_func, score_vals, nms_vals))
    results = [r for r in results if r is not None]

    total_time = time.time() - start_time

    print(f"\n{'=' * 80}")
    print("Grid search complete!")
    print(f"{'=' * 80}")
    print(f"Total time: {total_time / 60:.1f} minutes")
    print(f"Successful combinations: {len(results)}/{len(threshold_combos)}")
    print(f"Results saved to: {output_file}")
    return results

if __name__ == "__main__":
    results = main()