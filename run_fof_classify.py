"""
run_fof_classify.py — Run FOF matching and classification for a single threshold combo.

This is the CLI equivalent of running fof_and_classify.ipynb end-to-end for one
(score_thresh, nms_thresh) combination. It saves all intermediate artifacts
(raw FOF, analysis table, classified table) and prints summary metrics.

Output directory structure (under {run_dir}/analysis_cats/):
    dd/{ll}/fof_{mag_limit}.ecsv         — raw FOF result (astropy Table)
    dd/{ll}/raw_{mag_limit}.parquet      — analysis table (after build_table)
    dd/{ll}/class_{mag_limit}.parquet    — classified table (after stage 2)
    lsst/{ll}/fof_{mag_limit}.ecsv
    lsst/{ll}/raw_{mag_limit}.parquet
    lsst/{ll}/class_{mag_limit}.parquet

Usage:
    python run_fof_classify.py \
        --run-name ~/lsst_runs/lsst5_30k_4h200_bs192_ep50 \
        --test-cats-dir ~/lsst_data/test_cats_lvl5/val_4k/ \
        --score-thresh 0.5 --nms-thresh 0.5 \
        --mag-limit gold --buffer 2 \
        --linking-lengths 1.0 2.0

python run_fof_classify.py --run-name ~/lsst_runs/lsst5_30k_4h200_bs192_ep50 --test-cats-dir ~/lsst_data/test_cats_lvl5/val_4k/ --score-thresh 0.6 --nms-thresh 0.6 --mag-limit gold --buffer 1 --linking-lengths 1.0 2.0 > /u/yse2/lsst_runs/lsst5_30k_4h200_bs192_ep50/metrics/gold_buf1/run_s0.6_n0.6.log 2>&1
python run_fof_classify.py --run-name ~/lsst_runs/clip5_30k_4h200_bs64_ep50 --test-cats-dir ~/lsst_data/test_cats_lvl5/val_4k/ --score-thresh 0.6 --nms-thresh 0.6 --mag-limit gold --buffer 1 --linking-lengths 1.0 2.0 > /u/yse2/lsst_runs/clip5_30k_4h200_bs64_ep50/metrics/gold_buf1/run_s0.6_n0.6.log 2>&1
                    
AFTER RANKING
python run_fof_classify.py --run-name ~/lsst_runs/lsst5_30k_4h200_bs192_ep50 --test-cats-dir ~/lsst_data/test_cats_lvl5/val_4k/ --score-thresh 0.55 --nms-thresh 0.55 --mag-limit gold --buffer 1 --linking-lengths 1.0 2.0 > /u/yse2/lsst_runs/lsst5_30k_4h200_bs192_ep50/metrics/gold_buf1/run_s0.55_n0.55.log 2>&1
python run_fof_classify.py --run-name ~/lsst_runs/clip5_30k_4h200_bs64_ep50 --test-cats-dir ~/lsst_data/test_cats_lvl5/val_4k/ --score-thresh 0.55 --nms-thresh 0.55 --mag-limit gold --buffer 1 --linking-lengths 1.0 2.0 > /u/yse2/lsst_runs/clip5_30k_4h200_bs64_ep50/metrics/gold_buf1/run_s0.55_n0.55.log 2>&1

FOR TEST SET (8k):
python run_fof_classify.py --run-name ~/lsst_runs/lsst5_30k_4h200_bs192_ep50 --preds-dir test --test-cats-dir ~/lsst_data/test_cats_lvl5/test_8k/ --score-thresh 0.4 --nms-thresh 0.55 --mag-limit gold --buffer 1 --linking-lengths 1.0 2.0 > /u/yse2/lsst_runs/lsst5_30k_4h200_bs192_ep50/metrics/gold_buf1/test_s0.4_n0.55.log 2>&1
python run_fof_classify.py --run-name ~/lsst_runs/lsst5_30k_4h200_bs192_ep50 --preds-dir test --test-cats-dir ~/lsst_data/test_cats_lvl5/test_8k/ --score-thresh 0.4 --nms-thresh 0.55 --mag-limit gold --buffer 2 --linking-lengths 1.0 2.0 > /u/yse2/lsst_runs/lsst5_30k_4h200_bs192_ep50/metrics/gold_buf2/test_s0.4_n0.55.log 2>&1

python run_fof_classify.py --run-name ~/lsst_runs/lsst5_30k_4h200_bs192_ep50 --preds-dir test --test-cats-dir ~/lsst_data/test_cats_lvl5/test_8k/ --score-thresh 0.5 --nms-thresh 0.6 --mag-limit gold --buffer 1 --linking-lengths 1.0 2.0 > /u/yse2/lsst_runs/lsst5_30k_4h200_bs192_ep50/metrics/gold_buf1/test_s0.5_n0.6.log 2>&1
python run_fof_classify.py --run-name ~/lsst_runs/lsst5_30k_4h200_bs192_ep50 --preds-dir test --test-cats-dir ~/lsst_data/test_cats_lvl5/test_8k/ --score-thresh 0.5 --nms-thresh 0.6 --mag-limit gold --buffer 2 --linking-lengths 1.0 2.0 > /u/yse2/lsst_runs/lsst5_30k_4h200_bs192_ep50/metrics/gold_buf2/test_s0.5_n0.6.log 2>&1

python run_fof_classify.py --run-name ~/lsst_runs/clip5_30k_4h200_bs64_ep50 --preds-dir test --test-cats-dir ~/lsst_data/test_cats_lvl5/test_8k/ --score-thresh 0.4 --nms-thresh 0.55 --mag-limit gold --buffer 1 --linking-lengths 1.0 2.0 > /u/yse2/lsst_runs/clip5_30k_4h200_bs64_ep50/metrics/gold_buf1/test_s0.4_n0.55.log 2>&1
python run_fof_classify.py --run-name ~/lsst_runs/clip5_30k_4h200_bs64_ep50 --preds-dir test --test-cats-dir ~/lsst_data/test_cats_lvl5/test_8k/ --score-thresh 0.4 --nms-thresh 0.55 --mag-limit gold --buffer 2 --linking-lengths 1.0 2.0 > /u/yse2/lsst_runs/clip5_30k_4h200_bs64_ep50/metrics/gold_buf2/test_s0.4_n0.55.log 2>&1

python run_fof_classify.py --run-name ~/lsst_runs/clip5_30k_4h200_bs64_ep50 --preds-dir test --test-cats-dir ~/lsst_data/test_cats_lvl5/test_8k/ --score-thresh 0.5 --nms-thresh 0.6 --mag-limit gold --buffer 1 --linking-lengths 1.0 2.0 > /u/yse2/lsst_runs/clip5_30k_4h200_bs64_ep50/metrics/gold_buf1/test_s0.5_n0.6.log 2>&1
python run_fof_classify.py --run-name ~/lsst_runs/clip5_30k_4h200_bs64_ep50 --preds-dir test --test-cats-dir ~/lsst_data/test_cats_lvl5/test_8k/ --score-thresh 0.5 --nms-thresh 0.6 --mag-limit gold --buffer 2 --linking-lengths 1.0 2.0 > /u/yse2/lsst_runs/clip5_30k_4h200_bs64_ep50/metrics/gold_buf2/test_s0.5_n0.6.log 2>&1

FOR TEST SET (ALL):
python run_fof_classify.py --run-name ~/lsst_runs/lsst5_all_4h200_bs192_ep20 --preds-dir test --test-cats-dir ~/lsst_data/test_cats_lvl5/test_all/ --score-thresh 0.45 --nms-thresh 0.65 --mag-limit gold --buffer 1 --linking-lengths 1.0 2.0 > /u/yse2/lsst_runs/lsst5_all_4h200_bs192_ep20/metrics/gold_buf1/test_s0.45_n0.65.log 2>&1
python run_fof_classify.py --run-name ~/lsst_runs/lsst5_all_4h200_bs192_ep20 --preds-dir test --test-cats-dir ~/lsst_data/test_cats_lvl5/test_all/ --score-thresh 0.45 --nms-thresh 0.65 --mag-limit gold --buffer 2 --linking-lengths 1.0 2.0 > /u/yse2/lsst_runs/lsst5_all_4h200_bs192_ep20/metrics/gold_buf2/test_s0.45_n0.65.log 2>&1

python run_fof_classify.py --run-name ~/lsst_runs/lsst5_all_4h200_bs192_ep20 --preds-dir test --test-cats-dir ~/lsst_data/test_cats_lvl5/test_all/ --score-thresh 0.55 --nms-thresh 0.65 --mag-limit gold --buffer 1 --linking-lengths 1.0 2.0 > /u/yse2/lsst_runs/lsst5_all_4h200_bs192_ep20/metrics/gold_buf1/test_s0.55_n0.65.log 2>&1
python run_fof_classify.py --run-name ~/lsst_runs/lsst5_all_4h200_bs192_ep20 --preds-dir test --test-cats-dir ~/lsst_data/test_cats_lvl5/test_all/ --score-thresh 0.55 --nms-thresh 0.65 --mag-limit gold --buffer 2 --linking-lengths 1.0 2.0 > /u/yse2/lsst_runs/lsst5_all_4h200_bs192_ep20/metrics/gold_buf2/test_s0.55_n0.65.log 2>&1

"""

import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd

from fof_classify import (
    find_matches, build_table, get_stage1_counts,
    classify_stg1, classify_stage2, stage2_summary
)
from metrics import compute_scalar_metrics, print_scalar_metrics

warnings.filterwarnings('ignore')


# ============================================================================
# Config
# ============================================================================
MAG_LIMITS = {
    'power_law': 26.22,
    'gold': 25.3,
    'nominal': 26.42,
}
def get_col_config(dd_det_cols):
    """Column config for build_table()."""
    # cols we want to pull from each cat {'og_col_name': 'new_col_name'}
    return {
        # copy all cols
        'dd_det': {col: col for col in dd_det_cols},
        'lsst_det': {
            'objectId': 'id',  # rename to match DD and LSST truth's 'id'
            'parentObjectId': 'parentObjectId',
            'ra': 'ra',
            'dec': 'dec',
            'tract': 'tract',
            'patch': 'patch',
            # 'Ixx_pixel': 'Ixx_pixel',
            # 'Iyy_pixel': 'Iyy_pixel',
            # 'Ixy_pixel': 'Ixy_pixel',
            # 'I_flag': 'I_flag',
            # 'IxxPSF_pixel': 'IxxPSF_pixel',
            # 'IyyPSF_pixel': 'IyyPSF_pixel',
            # 'IxyPSF_pixel': 'IxyPSF_pixel',
            # ideally separate mags for stars and galaxies (use mag_i_cModel for galaxies but ~300 galaxies have NaN cModel mag_i)
            'mag_u': 'mag_u',
            'mag_g': 'mag_g',
            'mag_r': 'mag_r',
            'mag_i': 'mag_i',
            'mag_z': 'mag_z',
            'mag_y': 'mag_y', 
            # 'psFlux_u': 'psFlux_u',
            # 'psFlux_g': 'psFlux_g',
            # 'psFlux_r': 'psFlux_r',
            # 'psFlux_i': 'psFlux_i',
            # 'psFlux_z': 'psFlux_z',
            # 'psFlux_y': 'psFlux_y',
            # 'cModelFlux_u': 'cModelFlux_u',
            # 'cModelFlux_g': 'cModelFlux_g',
            # 'cModelFlux_r': 'cModelFlux_r',
            # 'cModelFlux_i': 'cModelFlux_i',
            # 'cModelFlux_z': 'cModelFlux_z',
            # 'cModelFlux_y': 'cModelFlux_y',
            'snr_u_cModel': 'snr_u',
            'snr_g_cModel': 'snr_g',
            'snr_r_cModel': 'snr_r',
            'snr_i_cModel': 'snr_i',
            'snr_z_cModel': 'snr_z',
            'snr_y_cModel': 'snr_y',
            'psf_fwhm_u': 'psf_fwhm_u',
            'psf_fwhm_g': 'psf_fwhm_g',
            'psf_fwhm_r': 'psf_fwhm_r',
            'psf_fwhm_i': 'psf_fwhm_i',
            'psf_fwhm_z': 'psf_fwhm_z',
            'psf_fwhm_y': 'psf_fwhm_y',
            'extendedness': 'class', # rename to 'class' to match DD (0 is star, 1 is galaxy)
            'blendedness': 'blendedness',
            'x': 'x',
            'y': 'y',
            'cutout_x': 'cutout_x',
            'cutout_y': 'cutout_y',
            'file_name': 'file_name'
        },
        'lsst_truth': {
            'id': 'id',
            'ra': 'ra',
            'dec': 'dec',
            'truth_type': 'class',  # rename to match DD (2 is star, 1 is galaxy)
            'tract': 'tract',
            'patch': 'patch',
            'mag_u': 'mag_u',
            'mag_g': 'mag_g',
            'mag_r': 'mag_r',
            'mag_i': 'mag_i',
            'mag_z': 'mag_z',
            'mag_y': 'mag_y',
            'flux_u': 'flux_u',
            'flux_g': 'flux_g',
            'flux_r': 'flux_r',
            'flux_i': 'flux_i',
            'flux_z': 'flux_z',
            'flux_y': 'flux_y',
            # sizes
            'size_true': 'size',
            'size_disk_true': 'size_disk',
            'size_bulge_true': 'size_bulge',
            'size_minor_true': 'size_minor',
            'size_minor_disk_true': 'size_minor_disk',
            'size_minor_bulge_true': 'size_minor_bulge',
            # ellipticity
            'ellipticity_1_true': 'e1',
            'ellipticity_2_true': 'e2',
            'ellipticity_1_disk_true': 'e1_disk',
            'ellipticity_2_disk_true': 'e2_disk',
            'ellipticity_1_bulge_true': 'e1_bulge',
            'ellipticity_2_bulge_true': 'e2_bulge',
            # orientation & profile
            'position_angle_true': 'pa_true',
            'position_angle_true_dc2': 'pa_true_dc2',
            'bulge_to_total_ratio_u': 'bttr_u',
            'bulge_to_total_ratio_g': 'bttr_g',
            'bulge_to_total_ratio_r': 'bttr_r',
            'bulge_to_total_ratio_i': 'bttr_i',
            'bulge_to_total_ratio_z': 'bttr_z',
            'bulge_to_total_ratio_y': 'bttr_y',
            # dust/extinction
            'av': 'av',
            'A_v': 'A_v',
            'rv': 'rv',
            'R_v': 'R_v',
            # SDSS filter luminosities (stellar)
            'SDSS_filters/diskLuminositiesStellar:SDSS_u:observed': 'SDSS_diskLum_u',
            'SDSS_filters/diskLuminositiesStellar:SDSS_g:observed': 'SDSS_diskLum_g',
            'SDSS_filters/diskLuminositiesStellar:SDSS_r:observed': 'SDSS_diskLum_r',
            'SDSS_filters/diskLuminositiesStellar:SDSS_i:observed': 'SDSS_diskLum_i',
            'SDSS_filters/diskLuminositiesStellar:SDSS_z:observed': 'SDSS_diskLum_z',
            'SDSS_filters/spheroidLuminositiesStellar:SDSS_u:observed': 'SDSS_spheroidLum_u',
            'SDSS_filters/spheroidLuminositiesStellar:SDSS_g:observed': 'SDSS_spheroidLum_g',
            'SDSS_filters/spheroidLuminositiesStellar:SDSS_r:observed': 'SDSS_spheroidLum_r',
            'SDSS_filters/spheroidLuminositiesStellar:SDSS_i:observed': 'SDSS_spheroidLum_i',
            'SDSS_filters/spheroidLuminositiesStellar:SDSS_z:observed': 'SDSS_spheroidLum_z',
            # LSST filter luminosities (stellar)
            'LSST_filters/diskLuminositiesStellar:LSST_u:observed': 'LSST_diskLum_u',
            'LSST_filters/diskLuminositiesStellar:LSST_y:observed': 'LSST_diskLum_y',
            'LSST_filters/spheroidLuminositiesStellar:LSST_u:observed': 'LSST_spheroidLum_u',
            'LSST_filters/spheroidLuminositiesStellar:LSST_y:observed': 'LSST_spheroidLum_y',
            'shear_1': 'shear_1',
            'shear_2': 'shear_2',
            'convergence': 'convergence',
            'host_galaxy': 'host_galaxy',
            'cosmodc2_hp': 'cosmodc2_hp',
            'cosmodc2_id': 'cosmodc2_id',
            'redshift': 'z',
            # Cutout info
            'image_id': 'image_id',
            'height': 'height',
            'width': 'width',
            'tile': 'tile',
            'det_cat_path': 'det_cat_path',
            'truth_cat_path': 'truth_cat_path',
            'wcs': 'wcs',
            'bbox': 'bbox',
            'bbox_mode': 'bbox_mode',
            'segmentation': 'segmentation',
            'keypoints': 'keypoints',
            'cutout_x': 'cutout_x',
            'cutout_y': 'cutout_y',
            'file_name': 'file_name'
        }
    }


# ============================================================================
# Pipeline for one (pipeline, linking_length)
# ============================================================================
def run_single_pipeline(det_cat, truth_cat, col_config, det_prefix,
                        linking_length, out_dir, truth_mag_limit,
                        threshold_tag=None, match_rad=None, verbose=True):
    """
    Run full FOF + classify pipeline for one detector at one linking length.
    Saves all intermediate artifacts.

    Parameters
    ----------
    det_cat : pd.DataFrame
        Detection catalog.
    truth_cat : pd.DataFrame
        Truth catalog.
    col_config : dict
        Column config for build_table().
    det_prefix : str
        'dd' or 'lsst'.
    linking_length : float
        FOF linking length in arcseconds.
    out_dir : str
        Base output directory ({run_dir}/analysis_cats/).
    truth_mag_limit : float
        For naming output files.
    threshold_tag : str or None
        Subdirectory for threshold-dependent results (e.g. 's0.5_n0.5').
        Used for DD to separate results by score/NMS combo.
        LSST results are threshold-independent, so pass None.
    verbose : bool
        Print progress.

    Returns
    -------
    analysis_df : pd.DataFrame
        Classified analysis table.
    metrics : dict
        Summary metrics.
    """
    match_rad = linking_length if match_rad is None else match_rad
    ll_str = str(linking_length)
    det_cat_key = 'dd_det' if det_prefix == 'dd' else 'lsst_det'

    cats = {
        det_cat_key: det_cat,
        'lsst_truth': truth_cat,
    }

    if det_prefix == 'dd':
        expected_keys = ['dd_det', 'lsst_truth']
        rename_map = {'lsst_truth': 'n_truth', 'dd_det': 'n_dd'}
        det_col_name = 'n_dd'
    else:
        expected_keys = ['lsst_det', 'lsst_truth']
        rename_map = {'lsst_truth': 'n_truth', 'lsst_det': 'n_lsst'}
        det_col_name = 'n_lsst'

    # Create output directory
    # DD: {out_dir}/dd/{ll}/s{score}_n{nms}/
    # LSST: {out_dir}/lsst/{ll}/
    if threshold_tag:
        save_dir = os.path.join(out_dir, det_prefix, ll_str, threshold_tag)
    else:
        save_dir = os.path.join(out_dir, det_prefix, ll_str)
    os.makedirs(save_dir, exist_ok=True)

    # --- Step 1: FOF matching ---
    if verbose:
        print(f"\n[{det_prefix.upper()} LL={ll_str}\"] FOF matching...")
    t0 = time.time()
    fof_result = find_matches(cats, linking_length=linking_length, max_friends=None)
    if verbose:
        print(f"  FOF done in {time.time() - t0:.1f}s")

    # Save raw FOF result
    fof_path = os.path.join(save_dir, f'fof_{truth_mag_limit:.2f}.ecsv')
    fof_result.write(fof_path, overwrite=True)
    if verbose:
        print(f"  Saved: {fof_path}")

    # --- Step 2: Build analysis table ---
    results_df = fof_result.to_pandas()
    analysis_df = build_table(results_df, cats, col_config)

    # Save raw analysis table
    raw_path = os.path.join(save_dir, f'raw_{truth_mag_limit:.2f}.parquet')
    analysis_df.to_parquet(raw_path)
    if verbose:
        print(f"  Saved: {raw_path} ({len(analysis_df):,} rows)")

    # --- Step 3: Stage 1 ---
    counts = get_stage1_counts(analysis_df, expected_keys=expected_keys,
                               rename_map=rename_map)
    counts = classify_stg1(counts, prefix=det_prefix,
                           det_col_name=det_col_name, truth_col_name='n_truth')

    if verbose:
        stg1_col = f'{det_prefix}_stg1'
        print(f"\n  Stage 1 counts ({det_prefix.upper()}):")
        for label, cnt in counts[stg1_col].value_counts().items():
            print(f"    {label}: {cnt:,}")

    # --- Step 4: Stage 2 ---
    if verbose:
        print(f"\n  Stage 2 classification (match_rad={match_rad}\")...")
    t0 = time.time()
    analysis_df = classify_stage2(analysis_df, counts, det_prefix,
                                  match_rad, verbose=verbose)
    if verbose:
        print(f"  Stage 2 done in {time.time() - t0:.1f}s")

    # Save classified table
    class_path = os.path.join(save_dir, f'class_{truth_mag_limit:.2f}.parquet')
    analysis_df.to_parquet(class_path)
    if verbose:
        print(f"  Saved: {class_path}")

    # --- Summary & metrics ---
    stage2_summary(analysis_df, det_prefix)
    metrics = compute_scalar_metrics(analysis_df, det_prefix)
    print_scalar_metrics(metrics, f'{det_prefix.upper()} LL={ll_str}"')

    return analysis_df, metrics


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Run FOF matching and classification for a single threshold combo.'
    )
    parser.add_argument('--root-run-dir', type=str, default='~/lsst_runs/')
    parser.add_argument('--run-name', required=True,
                        help='Run name (subfolder under --root-run-dir)')
    parser.add_argument('--preds-dir', default='eval',
                        help='Subdirectory under preds/ containing DD prediction files')
    parser.add_argument('--test-cats-dir', required=True,
                        help='Path to test catalogs directory')
    parser.add_argument('--score-thresh', type=float, required=True,
                        help='Score threshold for DD predictions')
    parser.add_argument('--nms-thresh', type=float, required=True,
                        help='NMS threshold for DD predictions')
    parser.add_argument('--mag-limit', default='gold', choices=MAG_LIMITS.keys(),
                        help='Magnitude limit key (default: gold)')
    parser.add_argument('--buffer', type=int, default=1, choices=[0, 1, 2],
                        help='Buffer to add to mag limit (default: 1)')
    parser.add_argument('--linking-lengths', nargs='+', type=float,
                        default=[1.0, 2.0],
                        help='FOF linking lengths in arcseconds (default: 1.0 2.0)')
    parser.add_argument('--match-rad', type=float, default=None,
                        help=('Stage 2 matching radius in arcseconds. '
                              'Default: use each linking length value'))
    parser.add_argument('--skip-lsst', action='store_true',
                        help='Skip LSST pipeline (only run DD)')
    return parser.parse_args()


def main():
    args = parse_args()
    root_run_dir = os.path.expanduser(args.root_run_dir)
    run_dir = os.path.join(root_run_dir, args.run_name)
    test_cats_dir = os.path.expanduser(args.test_cats_dir)
    truth_mag_limit = MAG_LIMITS[args.mag_limit] + args.buffer
    out_dir = os.path.join(run_dir, 'analysis_cats')
    threshold_tag = f's{args.score_thresh}_n{args.nms_thresh}'
    truth_fn = os.path.join(
        test_cats_dir,
        f'test_truth_cat_maglim_{truth_mag_limit:.2f}.parquet',
    )
    pred_fn = os.path.join(
        run_dir, 'preds', args.preds_dir,
        f'pred_{threshold_tag}.json',
    )
    lsst_det_fn = os.path.join(test_cats_dir, 'test_det_cat.json')

    print("=" * 80)
    print("FOF MATCHING AND CLASSIFICATION")
    print("=" * 80)
    print(f"\nArguments used (CLI):")
    print(f"  root_run_dir:     {args.root_run_dir}")
    print(f"  run_name:         {args.run_name}")
    print(f"  preds_dir:        {args.preds_dir}")
    print(f"  test_cats_dir:    {args.test_cats_dir}")
    print(f"  score_thresh:     {args.score_thresh}")
    print(f"  nms_thresh:       {args.nms_thresh}")
    print(f"  mag_limit:        {args.mag_limit}")
    print(f"  buffer:           {args.buffer}")
    print(f"  linking_lengths:  {args.linking_lengths}")
    print(f"  match_rad:        {args.match_rad}")
    print(f"  skip_lsst:        {args.skip_lsst}")

    print(f"\nResolved parameters used at runtime:")
    print(f"  run_dir:          {run_dir}")
    print(f"  test_cats_dir:    {test_cats_dir}")
    print(f"  output_dir:       {out_dir}")
    print(f"  threshold_tag:    {threshold_tag}")
    print(f"  mag_limit_value:  {MAG_LIMITS[args.mag_limit]}")
    print(f"  truth_mag_limit:  {truth_mag_limit:.2f}")
    print(f"  truth_fn:         {truth_fn}")
    print(f"  pred_fn:          {pred_fn}")
    if not args.skip_lsst:
        print(f"  lsst_det_fn:      {lsst_det_fn}")
    print("  per-LL match radius:")
    for ll in args.linking_lengths:
        effective_match = ll if args.match_rad is None else args.match_rad
        print(f"    LL={ll}: match_rad={effective_match}")

    # --- Load catalogs ---
    print(f"\nLoading catalogs...")
    print(f"  Truth: {truth_fn}")
    lsst_truth_cat = pd.read_parquet(truth_fn)
    print(f"    {len(lsst_truth_cat):,} truth objects")

    print(f"  DD predictions: {pred_fn}")
    dd_det_cat = pd.read_json(pred_fn)
    dd_det_cat = dd_det_cat.rename(columns={'ra_kp': 'ra', 'dec_kp': 'dec'})
    print(f"    {len(dd_det_cat):,} DD detections")    
    # Check for mag cols in dd preds since we need them for mag bin plots
    if not [col for col in dd_det_cat.columns if 'mag' in col.lower()]:
        print(f"\n WARNING: No magnitude columns found in DD predictions!")
        print(f"       Expected columns like: mag_u, mag_g, mag_r, mag_i, mag_z, mag_y")
        print(f"       Run add_dd_mags.py to add mags to predictions\n")

    if not args.skip_lsst:
        print(f"  LSST detections: {lsst_det_fn}")
        lsst_det_cat = pd.read_json(lsst_det_fn)
        print(f"    {len(lsst_det_cat):,} LSST detections")

    col_config = get_col_config(dd_det_cat.columns)

    # --- Run pipelines ---
    total_start = time.time()
    all_metrics = {}

    for ll in args.linking_lengths:
        # DD pipeline
        _, dd_metrics = run_single_pipeline(
            dd_det_cat, lsst_truth_cat, col_config,
            det_prefix='dd', linking_length=ll,
            out_dir=out_dir, truth_mag_limit=truth_mag_limit,
            threshold_tag=threshold_tag, match_rad=args.match_rad
        )
        all_metrics[('dd', str(ll))] = dd_metrics

        # LSST pipeline
        if not args.skip_lsst:
            _, lsst_metrics = run_single_pipeline(
                lsst_det_cat, lsst_truth_cat, col_config,
                det_prefix='lsst', linking_length=ll,
                out_dir=out_dir, truth_mag_limit=truth_mag_limit,
                match_rad=args.match_rad
            )
            all_metrics[('lsst', str(ll))] = lsst_metrics

    total_time = time.time() - total_start

    # --- Final summary ---
    print(f"\n{'=' * 80}")
    print("ALL DONE")
    print(f"{'=' * 80}")
    print(f"Total time: {total_time:.1f}s")
    print(f"\nResults saved to: {out_dir}")
    print(f"  Per pipeline/linking_length/threshold subdirectory:")
    print(f"    fof_{truth_mag_limit:.2f}.ecsv    — raw FOF groups")
    print(f"    raw_{truth_mag_limit:.2f}.parquet  — analysis table")
    print(f"    class_{truth_mag_limit:.2f}.parquet — classified table")

    # Comparison table
    print(f"\n{'─' * 60}")
    print(f"{'Pipeline':>8} {'LL':>5} {'Comp':>8} {'Purity':>8} "
          f"{'F1':>8} {'BlendLoss':>10} {'ShredFrac':>10}")
    print(f"{'─' * 60}")
    for (prefix, ll_str), m in all_metrics.items():
        print(f"{prefix.upper():>8} {ll_str:>5} "
              f"{m['completeness']:>8.4f} {m['purity']:>8.4f} "
              f"{m['f1']:>8.4f} {m['blend_loss_frac']:>10.4f} "
              f"{m['shred_frac']:>10.4f}")
    print(f"{'─' * 60}")


if __name__ == '__main__':
    main()