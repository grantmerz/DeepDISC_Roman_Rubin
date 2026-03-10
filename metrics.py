"""
metrics.py — Comprehensive metrics for Stage 2 classified analysis DataFrames

The single source of truth for all metrics so we can just import from here
rather than copy-pasting code from script to script

Provides:
    Helpers:
        - split_truth_det()         — split analysis_df into truth/det subframes
        - get_matched_truth_mag()   — look up truth mag_i for matched detections

    Scalar metrics:
        - compute_scalar_metrics()  — all scalar metrics in one dict

"""

import numpy as np
import pandas as pd

# ============================================================================
# Helpers
# ============================================================================
def split_truth_det(analysis_df, prefix):
    """
    Split analysis DataFrame into truth and detection subframes.

    Parameters
    ----------
    analysis_df : pd.DataFrame
        Classified long-format analysis table.
    prefix : str
        'dd' or 'lsst'.

    Returns
    -------
    truths, dets : pd.DataFrame
    """
    det_key = 'dd_det' if prefix == 'dd' else 'lsst_det'
    truths = analysis_df[analysis_df['catalog_key'] == 'lsst_truth'].copy()
    dets = analysis_df[analysis_df['catalog_key'] == det_key].copy()
    return truths, dets

def get_matched_truth_mag(dets, truths, prefix, mag_col='mag_i'):
    """
    For each matched detection, look up the matched truth's magnitude.

    Parameters
    ----------
    dets : pd.DataFrame
        Detection rows.
    truths : pd.DataFrame
        Truth rows.
    prefix : str
        'dd' or 'lsst'.
    mag_col : str
        Magnitude column in truths. Default 'mag_i'.

    Returns
    -------
    pd.Series
        Aligned with dets index. NaN for unmatched detections.
    """
    # Build truth mag lookup: row_index -> mag_i
    truth_mag_lookup = truths.set_index('row_index')[mag_col]
    matched_id_col = f'{prefix}_matched_id'
    matched_mask = dets[f'{prefix}_is_matched']

    matched_truth_mags = dets.loc[matched_mask, matched_id_col].map(truth_mag_lookup)
    result = pd.Series(np.nan, index=dets.index)
    result.loc[matched_mask] = matched_truth_mags.values
    return result


def _safe_div(num, denom):
    """Safe division returning 0.0 when denominator is 0."""
    return num / denom if denom > 0 else 0.0


# ============================================================================
# Scalar metrics
# ============================================================================
def compute_scalar_metrics(analysis_df, prefix):
    """
    Compute all scalar summary metrics from a classified analysis DataFrame.

    Parameters
    ----------
    analysis_df : pd.DataFrame
        Classified long-format table from classify_stage2.
    prefix : str
        'dd' or 'lsst'.

    Returns
    -------
    dict : metric_name -> value
        Contains counts and fractions
    """
    p = prefix
    truths, dets = split_truth_det(analysis_df, prefix)

    n_truth = len(truths)
    n_det = len(dets)

    # --- Raw counts ---
    n_matched_truth = int(truths[f'{p}_is_matched'].sum())
    n_matched_det = int(dets[f'{p}_is_matched'].sum())
    assert n_matched_det == n_matched_truth, (
        f"Matched count mismatch: "
        f"truth={n_matched_det}, det={n_matched_truth}"
    )
    # --- Unrec blend among matched ---
    # each matched pair in an unrec blend group contributes one to each side so both counts shld match
    n_unrec_blend_matched = int(
        (truths[f'{p}_is_matched'] & truths[f'{p}_is_unrec_blend']).sum()
    )
    _n_unrec_blend_matched_det = int(
        (dets[f'{p}_is_matched'] & dets[f'{p}_is_unrec_blend']).sum()
    )
    assert n_unrec_blend_matched == _n_unrec_blend_matched_det, (
        f"Unrec blend matched count mismatch: "
        f"truth={n_unrec_blend_matched}, det={_n_unrec_blend_matched_det}"
    )
    n_spurious = int(dets[f'{p}_is_spurious'].sum())
    n_shred = int(dets[f'{p}_is_shred'].sum())
    n_missed = int(truths[f'{p}_is_missed'].sum())
    n_blended_away = int(truths[f'{p}_is_blended_away'].sum())
    n_shredded = int(truths[f'{p}_is_shredded'].sum())
    n_unrec_blend_truth = int(truths[f'{p}_is_unrec_blend'].sum())
    n_unrec_blend_det = int(dets[f'{p}_is_unrec_blend'].sum())
    n_resolved_truth = int(truths[f'{p}_is_resolved_blend'].sum())
    n_resolved_det = int(dets[f'{p}_is_resolved_blend'].sum())
    n_part_of_blend_truth = int(truths[f'{p}_is_part_of_blend'].sum())
    n_part_of_blend_det = int(dets[f'{p}_is_part_of_blend'].sum())
    n_partial_deblend_truth = int(truths[f'{p}_is_partial_deblend'].sum())
    n_partial_deblend_det = int(dets[f'{p}_is_partial_deblend'].sum())

    # --- Core fractions ---
    completeness = _safe_div(n_matched_truth, n_truth)
    purity = _safe_div(n_matched_det, n_det)
    f1 = (_safe_div(2 * completeness * purity, completeness + purity)
          if (completeness + purity) > 0 else 0.0)
    
    # --- Blend fractions (truth side) ---
    blend_loss_frac = _safe_div(n_blended_away, n_truth)
    unrec_blend_frac_total = _safe_div(n_unrec_blend_truth, n_truth)
    unrec_blend_frac_blended = _safe_div(n_unrec_blend_truth, n_part_of_blend_truth)
    unrec_blend_frac_matched = _safe_div(n_unrec_blend_matched, n_matched_truth)
    resolved_frac = _safe_div(n_resolved_truth, n_part_of_blend_truth)

    # --- Blend fractions (det side) ---
    unrec_blend_det_frac_total = _safe_div(n_unrec_blend_det, n_det)
    unrec_blend_det_frac_blended = _safe_div(n_unrec_blend_det, n_part_of_blend_det)

    # --- Shredding ---
    shred_frac = _safe_div(n_shredded, n_truth)
    shred_det_frac = _safe_div(n_shred, n_det)

    # --- Spurious / Missed ---
    spurious_frac = _safe_div(n_spurious, n_det)
    missed_frac = _safe_div(n_missed, n_truth)

    return {
        # counts
        'n_truth': n_truth,
        'n_det': n_det,
        'n_matched_truth': n_matched_truth,
        'n_matched_det': n_matched_det,
        'n_spurious': n_spurious,
        'n_shred': n_shred,
        'n_missed': n_missed,
        'n_blended_away': n_blended_away,
        'n_shredded': n_shredded,
        'n_unrec_blend_truth': n_unrec_blend_truth,
        'n_unrec_blend_det': n_unrec_blend_det,
        'n_unrec_blend_matched': n_unrec_blend_matched,
        'n_resolved_truth': n_resolved_truth,
        'n_resolved_det': n_resolved_det,
        'n_part_of_blend_truth': n_part_of_blend_truth,
        'n_part_of_blend_det': n_part_of_blend_det,
        'n_partial_deblend_truth': n_partial_deblend_truth,
        'n_partial_deblend_det': n_partial_deblend_det,
        # core fractions
        'completeness': completeness,
        'purity': purity,
        'f1': f1,
        # blend fractions (truth)
        'blend_loss_frac': blend_loss_frac,
        'unrec_blend_frac_total': unrec_blend_frac_total,
        'unrec_blend_frac_blended': unrec_blend_frac_blended,
        'unrec_blend_frac_matched': unrec_blend_frac_matched,
        'resolved_frac': resolved_frac,
        # blend fractions (det)
        'unrec_blend_det_frac_total': unrec_blend_det_frac_total,
        'unrec_blend_det_frac_blended': unrec_blend_det_frac_blended,
        # shredding
        'shred_frac': shred_frac,
        'shred_det_frac': shred_det_frac,
        # spurious / missed
        'spurious_frac': spurious_frac,
        'missed_frac': missed_frac,
    }


def print_scalar_metrics(metrics, label=''):
    """Pretty-print scalar metrics dict."""
    m = metrics
    print(f"\n{'=' * 60}")
    print(f"Metrics: {label}")
    print(f"{'=' * 60}")
    print(f"  Truths:              {m['n_truth']:>10,d}")
    print(f"  Detections:          {m['n_det']:>10,d}")
    print(f"  Completeness:        {m['completeness']:>10.4f}  "
          f"({m['completeness']*100:.2f}%)")
    print(f"  Purity:              {m['purity']:>10.4f}  "
          f"({m['purity']*100:.2f}%)")
    print(f"  F1:                  {m['f1']:>10.4f}")
    print(f"  Matched truths:      {m['n_matched_truth']:>10,d}")
    print(f"  Matched dets:        {m['n_matched_det']:>10,d}")
    print(f"  Missed:              {m['n_missed']:>10,d}  "
          f"({m['missed_frac']*100:.2f}%)")
    print(f"  Spurious:            {m['n_spurious']:>10,d}  "
          f"({m['spurious_frac']*100:.2f}%)")
    print(f"  Blended away:        {m['n_blended_away']:>10,d}  "
          f"({m['blend_loss_frac']*100:.2f}%)")
    print(f"  Shredded truths:     {m['n_shredded']:>10,d}  "
          f"({m['shred_frac']*100:.2f}%)")
    print(f"  Shred dets:          {m['n_shred']:>10,d}  "
          f"({m['shred_det_frac']*100:.2f}%)")
    print(f"  Unrec blend truths:  {m['n_unrec_blend_truth']:>10,d}  "
          f"({m['unrec_blend_frac_total']*100:.2f}%)")
    print(f"  Resolved truths:     {m['n_resolved_truth']:>10,d}")
    print(f"  Part of blend:       {m['n_part_of_blend_truth']:>10,d}")
    print(f"  Resolved rate:       {m['resolved_frac']:>10.4f}  "
          f"({m['resolved_frac']*100:.2f}%)")