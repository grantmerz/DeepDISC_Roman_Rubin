"""
grid_search.py — blend_metrics grid search over score/NMS thresholds.

For one run, one mag-limit, one buffer: sweep the score x NMS grid. For every
combo, run FOF + Stage 1/2 + significance fully in memory, compute all 12
blend_metrics metrics for DeepDISC and the fixed LSST reference, append one
wide CSV row, then discard the classified table. No parquets are written.

Usage:
    python grid_search.py \
        --run-dir ~/lsst_runs/lsst5_30k_4h200_bs192_ep50 \
        --test-cats-dir ~/lsst_data/test_cats_lvl5/val_4k/ \
        --output ~/lsst_runs/lsst5_30k_4h200_bs192_ep50/metrics/gold_buf2/gs_metrics.csv \
        --mag-limit gold \
        --buffer 2

    # Background with logging:
    python grid_search.py [args] > gs_out.txt 2>&1 &
"""

import argparse
import itertools
import os
import time
import warnings
from functools import partial
from multiprocessing import Lock, Pool

import pandas as pd

import blend_metrics as bm
from fof_classify import (
    find_matches, build_table, get_stage1_counts,
    classify_stg1, classify_stage2, add_sig_blend_flag,
)

warnings.filterwarnings('ignore')

file_lock = Lock()   # module-level; shared with workers via fork inheritance (see below)

METRICS = list(bm.METRIC_NUMDEN) + ['f1']   # 12, in this order
COUNTS = ['N_t', 'N_t_matched', 'N_t_missed', 'N_t_abs_all', 'N_t_sig',
          'N_t_res', 'N_t_surv', 'N_t_abs', 'N_t_shredded',
          'N_d', 'N_d_matched', 'N_d_spurious', 'N_d_shred', 'N_d_abr',
          'n_det_nan_mag', 'f_det_nan_mag']   # only if --include-counts


# ============================================================================
# Column config
# ============================================================================
def get_col_config():
    """blend_metrics only ever touches group_id, catalog_key, row_index, ra,
    dec, mag_i, so pulling only those three source columns keeps per-worker
    memory low across 165 combos."""
    cols = {'ra': 'ra', 'dec': 'dec', 'mag_i': 'mag_i'}
    return {'dd_det': dict(cols), 'lsst_det': dict(cols), 'lsst_truth': dict(cols)}

# ============================================================================
# CSV schema
# ============================================================================
def _pipelines(skip_lsst):
    return ('dd',) if skip_lsst else ('dd', 'lsst')


def csv_header(lls, include_counts, skip_lsst):
    pipelines = _pipelines(skip_lsst)
    ll_strs = [str(ll) for ll in lls]
    cols = ['score_thresh', 'nms_thresh']
    for prefix in pipelines:
        for ll in ll_strs:
            for metric in METRICS:
                cols.append(f'{prefix}_{metric}_{ll}')
    if include_counts:
        for prefix in pipelines:
            for ll in ll_strs:
                for count in COUNTS:
                    cols.append(f'{prefix}_{count}_{ll}')
    cols.append('elapsed_sec')
    return ','.join(cols) + '\n'


def format_row(score, nms, results, lls, include_counts, skip_lsst, elapsed):
    pipelines = _pipelines(skip_lsst)
    ll_strs = [str(ll) for ll in lls]
    parts = [f'{score}', f'{nms}']
    for prefix in pipelines:
        for ll in ll_strs:
            m, _ = results[(prefix, ll)]
            for metric in METRICS:
                parts.append(f'{m[metric]:.6f}')
    if include_counts:
        for prefix in pipelines:
            for ll in ll_strs:
                _, c = results[(prefix, ll)]
                for count in COUNTS:
                    parts.append(f'{c[count]:.6f}' if count.startswith('f_') else f'{c[count]}')
    parts.append(f'{elapsed:.2f}')
    return ','.join(parts) + '\n'

# ============================================================================
# Core: classify one pipeline at one linking length
# ============================================================================
def classify_and_metrics(cats, det_prefix, ll, m_cut, dmag_sig, match_rad, col_config):
    mr = ll if match_rad is None else match_rad
    det_key = 'dd_det' if det_prefix == 'dd' else 'lsst_det'
    det_col = 'n_dd' if det_prefix == 'dd' else 'n_lsst'

    results_df = find_matches(cats, linking_length=ll, max_friends=None).to_pandas()
    analysis_df = build_table(results_df, cats, col_config)

    counts = get_stage1_counts(
        analysis_df, expected_keys=[det_key, 'lsst_truth'],
        rename_map={'lsst_truth': 'n_truth', det_key: det_col})
    counts = classify_stg1(counts, prefix=det_prefix,
                           det_col_name=det_col, truth_col_name='n_truth')

    analysis_df = classify_stage2(analysis_df, counts, det_prefix=det_prefix,
                                  match_rad=mr, verbose=False)
    analysis_df = add_sig_blend_flag(analysis_df, det_prefix, dmag_sig=dmag_sig)

    truths, dets = bm.split_truth_det(analysis_df, det_prefix)
    bm.assert_maximality(truths, dets, det_prefix)          # before compute_counts
    mt = truths[bm.MAG_COL] < m_cut
    md = dets[bm.MAG_COL] < m_cut
    c = bm.compute_counts(truths, dets, det_prefix, mt, md)  # asserts 1-3 run here
    m = bm.compute_metrics(c)                                # 11 ratios + f1
    return m, c


# ============================================================================
# LSST precompute (once per buffer, all linking lengths)
# ============================================================================
def compute_lsst(lsst_det, truth, col_config, lls, m_cut, dmag_sig, match_rad):
    out = {}
    for ll in lls:
        cats = {'lsst_det': lsst_det, 'lsst_truth': truth}
        out[str(ll)] = classify_and_metrics(cats, 'lsst', ll, m_cut,
                                             dmag_sig, match_rad, col_config)
    return out   # {ll_str: (metrics, counts)}


# ============================================================================
# Per-combo worker
# ============================================================================
def process_combo(score, nms, truth, run_dir, preds_dir, output_file, col_config,
                  lsst_results, lls, m_cut, dmag_sig, match_rad,
                  include_counts, skip_lsst):
    t0 = time.time()
    tag = bm.combo_tag(score, nms)                       # 's{score}_n{nms}'
    pred_fn = os.path.join(run_dir, 'preds', preds_dir, f'pred_{tag}.json')
    if not os.path.exists(pred_fn):
        print(f'WARNING: missing pred file, skipping: {pred_fn}', flush=True)
        return None

    dd = pd.read_json(pred_fn).rename(columns={'ra_kp': 'ra', 'dec_kp': 'dec'})
    if bm.MAG_COL not in dd.columns:                     # mag_i presence check
        print(f'WARNING: {bm.MAG_COL} absent in {pred_fn}, skipping combo', flush=True)
        return None

    cats = {'dd_det': dd, 'lsst_truth': truth}
    results = {}
    for ll in lls:
        results[('dd', str(ll))] = classify_and_metrics(
            cats, 'dd', ll, m_cut, dmag_sig, match_rad, col_config)
        if not skip_lsst:
            results[('lsst', str(ll))] = lsst_results[str(ll)]

    row = format_row(score, nms, results, lls, include_counts, skip_lsst,
                     elapsed=time.time() - t0)
    with file_lock:
        with open(output_file, 'a') as f:
            f.write(row)
    print(f'DONE: s={score}, n={nms} ({time.time() - t0:.1f}s)', flush=True)
    return results




# ============================================================================
# CLI
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description='Grid search over score/NMS thresholds using blend_metrics.'
    )
    parser.add_argument('--run-dir', required=True,
                        help='Path to the run directory (e.g. ~/lsst_runs/run_name/)')
    parser.add_argument('--test-cats-dir', required=True,
                        help='Path to test catalogs directory')
    parser.add_argument('--output', required=True,
                        help='Output CSV path')
    parser.add_argument('--mag-limit', default='gold', choices=list(bm.MAG_LIMITS.keys()),
                        help='Magnitude limit key (default: gold)')
    parser.add_argument('--buffer', type=int, default=2, choices=[0, 1, 2],
                        help='Buffer added to mag limit to select the truth catalog '
                             '(default: 2). Never enters the mag cut itself.')
    parser.add_argument('--linking-lengths', nargs='+', type=float,
                        default=[0.5, 1.0],
                        help='FOF linking lengths in arcseconds (default: 0.5 1.0)')
    parser.add_argument('--score-thresholds', nargs='+', type=float,
                        default=[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                                 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
                        help='Score thresholds to search')
    parser.add_argument('--nms-thresholds', nargs='+', type=float,
                        default=[0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                                 0.55, 0.6, 0.65, 0.7, 0.75],
                        help='NMS thresholds to search')
    parser.add_argument('--preds-dir', default='eval',
                        help='Subdirectory under preds/ containing DD prediction files')
    parser.add_argument('--match-rad', type=float, default=None,
                        help='Stage 2 matching radius in arcseconds. '
                             'Default: use each linking length value')
    parser.add_argument('--dmag-sig', type=float, default=2.0,
                        help='Significant-blend magnitude gap threshold (default: 2.0)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers '
                             '(default: min(num_cpus // 2, n_combos))')
    parser.add_argument('--include-counts', action='store_true',
                        help='Also write raw N_* counts per pipeline/ll to the CSV')
    parser.add_argument('--skip-lsst', action='store_true',
                        help='Skip the LSST reference pipeline (DD columns only)')
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = os.path.expanduser(args.run_dir)
    test_cats_dir = os.path.expanduser(args.test_cats_dir)
    output_file = os.path.expanduser(args.output)
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    m_cut = bm.MAG_LIMITS[args.mag_limit]
    truth_mag_limit = m_cut + args.buffer

    truth_fn = os.path.join(test_cats_dir,
                            f'test_truth_cat_maglim_{truth_mag_limit:.2f}.parquet')
    lsst_det_fn = os.path.join(test_cats_dir, 'test_det_cat.json')

    combos = list(itertools.product(args.score_thresholds, args.nms_thresholds))
    cpu_count = len(os.sched_getaffinity(0))
    n_workers = args.workers or min(cpu_count // 2, len(combos))

    print('=' * 80)
    print('GRID SEARCH — blend_metrics')
    print('=' * 80)
    print('\nConfiguration:')
    print(f'  Run directory:    {run_dir}')
    print(f'  Test cats dir:    {test_cats_dir}')
    print(f'  Mag limit:        {args.mag_limit} ({m_cut}); buffer {args.buffer} '
          f'-> truth cat maglim {truth_mag_limit:.2f}')
    print(f'  Linking lengths:  {args.linking_lengths}')
    print(f'  Match radius:     {args.match_rad if args.match_rad is not None else "per-linking-length"}')
    print(f'  dmag_sig:         {args.dmag_sig}')
    print(f'  Output file:      {output_file}')
    print(f'  Score thresholds: {len(args.score_thresholds)} values')
    print(f'  NMS thresholds:   {len(args.nms_thresholds)} values')
    print(f'  Total combos:     {len(combos)}')
    print(f'  Include counts:   {args.include_counts}')
    print(f'  Skip LSST:        {args.skip_lsst}')
    print(f'  CPUs available:   {cpu_count}')
    print(f'  Workers:          {n_workers}')

    print(f'\nLoading truth catalog: {truth_fn}')
    truth_cat = pd.read_parquet(truth_fn)
    print(f'  {len(truth_cat):,} truth objects')

    lsst_det_cat = None
    if not args.skip_lsst:
        print(f'Loading LSST detections: {lsst_det_fn}')
        lsst_det_cat = pd.read_json(lsst_det_fn)
        print(f'  {len(lsst_det_cat):,} LSST detections')

    col_config = get_col_config()

    lsst_results = {}
    if not args.skip_lsst:
        print(f'\nPre-computing LSST metrics for {len(args.linking_lengths)} '
              f'linking length(s)...')
        t0 = time.time()
        lsst_results = compute_lsst(lsst_det_cat, truth_cat, col_config,
                                    args.linking_lengths, m_cut, args.dmag_sig,
                                    args.match_rad)
        print(f'LSST pre-computation done in {time.time() - t0:.1f}s')

    with open(output_file, 'w') as f:
        f.write(csv_header(args.linking_lengths, args.include_counts, args.skip_lsst))

    print(f'\n{"=" * 80}')
    print('Starting grid search...')
    print(f'{"=" * 80}\n')

    process_func = partial(
        process_combo,
        truth=truth_cat,
        run_dir=run_dir,
        preds_dir=args.preds_dir,
        output_file=output_file,
        col_config=col_config,
        lsst_results=lsst_results,
        lls=args.linking_lengths,
        m_cut=m_cut,
        dmag_sig=args.dmag_sig,
        match_rad=args.match_rad,
        include_counts=args.include_counts,
        skip_lsst=args.skip_lsst,
    )

    t0 = time.time()
    with Pool(processes=n_workers) as pool:
        results = pool.starmap(process_func, combos)
    results = [r for r in results if r is not None]
    total_time = time.time() - t0

    print(f'\n{"=" * 80}')
    print('Grid search complete!')
    print(f'{"=" * 80}')
    print(f'Successful combinations: {len(results)}/{len(combos)}')
    print(f'Wall time: {total_time / 60:.1f} minutes')
    print(f'Output: {output_file}')


if __name__ == '__main__':
    main()
