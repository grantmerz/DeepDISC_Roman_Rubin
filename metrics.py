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

    Binned metrics (vs magnitude):
        - binned_fraction()         — generic: fraction of flagged objects per mag bin
        - compute_completeness()    — matched_truth / total_truth vs mag
        - compute_purity()          — matched_det / total_det vs matched truth mag
        - compute_blend_loss()      — blended_away / total_truth vs mag
        - compute_unrec_blend_frac()— unrec_blend / total_truth vs mag
        - compute_shred_frac()      — shredded / total_truth vs mag
        - compute_missed_frac()     — missed / total_truth vs mag
        - compute_resolved_frac()   — resolved / part_of_blend vs mag
        - compute_spurious_frac()   — spurious / total_det vs mag
        Bootstrap:
        - bootstrap_binned_metric()   — bootstrap CIs for any binned metric function

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
    print(f"  Resolved frac:       {m['resolved_frac']:>10.4f}  "
          f"({m['resolved_frac']*100:.2f}%)")
    print(f"  Unrec BL frac (matched): {m['unrec_blend_frac_matched']:>7.4f}  "
          f"({m['unrec_blend_frac_matched']*100:.2f}%)")


# ============================================================================
# Binned metrics (vs magnitude)
# ============================================================================
def binned_fraction(flag_series, mag_series, mag_bins, min_count=1,
                    confidence_level=0.95, interval='wilson',
                    compute_ci=True):
    """
    Compute fraction of True values in each magnitude bin with Astropy CIs

    Parameters
    ----------
    flag_series : array-like of bool
        Boolean flag (e.g., is_matched).
    mag_series : array-like of float
        Magnitude values for binning.
    mag_bins : array-like
        Bin edges.
    min_count : int
        Minimum objects per bin for valid fraction.
    confidence_level : float
        Confidence level for binomial confidence interval.
    interval : str
        Astropy interval mode for binom_conf_interval.
    compute_ci : bool
        If True, compute confidence intervals. If False, only fractions and
        counts are computed and CI arrays remain NaN.

    Returns
    -------
    dict
        Keys: bin_centers, fractions, ci_lo, ci_hi, counts, flagged_counts.
    """
    flag_arr = np.asarray(flag_series, dtype=bool)
    mag_arr = np.asarray(mag_series, dtype=float)

    valid_input = np.isfinite(mag_arr)
    flag_arr = flag_arr[valid_input]
    mag_arr = mag_arr[valid_input]

    n_bins = len(mag_bins) - 1
    bin_centers = 0.5 * (mag_bins[:-1] + mag_bins[1:])
    fractions = np.full(n_bins, np.nan, dtype=float)
    ci_lo = np.full(n_bins, np.nan, dtype=float)
    ci_hi = np.full(n_bins, np.nan, dtype=float)
    counts = np.zeros(n_bins, dtype=int)
    flagged_counts = np.zeros(n_bins, dtype=int)

    bin_idx = np.digitize(mag_arr, mag_bins) - 1
    in_range = (bin_idx >= 0) & (bin_idx < n_bins)
    if not np.any(in_range):
        return {
            'bin_centers': bin_centers,
            'fractions': fractions,
            'ci_lo': ci_lo,
            'ci_hi': ci_hi,
            'counts': counts,
            'flagged_counts': flagged_counts,
        }
    bin_idx = bin_idx[in_range]
    flags = flag_arr[in_range].astype(int)

    counts = np.bincount(bin_idx, minlength=n_bins)
    flagged_counts = np.bincount(bin_idx, weights=flags, minlength=n_bins)
    flagged_counts = flagged_counts.astype(int)

    valid_bins = counts >= min_count
    if np.any(valid_bins):
        k_valid = flagged_counts[valid_bins]
        n_valid = counts[valid_bins]
        fr_valid = k_valid / n_valid
        fractions[valid_bins] = fr_valid
        if compute_ci:
            bounds = binom_conf_interval(
                k_valid,
                n_valid,
                confidence_level=confidence_level,
                interval=interval,
            )
            ci_lo[valid_bins] = bounds[0]
            ci_hi[valid_bins] = bounds[1]

    return {
        'bin_centers': bin_centers,
        'fractions': fractions,
        'ci_lo': ci_lo,
        'ci_hi': ci_hi,
        'counts': counts,
        'flagged_counts': flagged_counts,
    }


# ============================================================================
# Bootstrap
# ============================================================================
def bootstrap_binned_metric(compute_func, n_bootstrap=1000, seed=42,
                            confidence=0.95, **kwargs):
    """
    Bootstrap confidence intervals for any binned metric function.

    Parameters
    ----------
    compute_func : callable
        A function like compute_completeness that takes a DataFrame as its
        first argument and returns a binned_fraction dict.
    n_bootstrap : int
        Number of bootstrap iterations.
    seed : int
        Random seed.
    confidence : float
        Confidence level for bootstrap quantile bounds.
    **kwargs :
        Additional keyword arguments passed to compute_func.
        Must include the DataFrame as 'data' keyword.
        If compute_func supports 'compute_ci', it is set to False during
        resamples to avoid redundant CI computation.

    Returns
    -------
    dict with keys:
        fractions_median, ci_lo, ci_hi : arrays
        all_fractions : (n_bootstrap x n_bins) array
        bin_centers : array
    """
    data = kwargs.pop('data')
    n = len(data)
    if n == 0:
        raise ValueError("Cannot bootstrap empty data.")
    alpha = (1.0 - confidence) / 2.0

    call_kwargs = dict(kwargs)
    signature = inspect.signature(compute_func)
    supports_compute_ci = (
        'compute_ci' in signature.parameters
        or any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in signature.parameters.values()
        )
    )
    if supports_compute_ci:
        call_kwargs['compute_ci'] = False

    result0 = compute_func(data, **call_kwargs)
    base_indices = np.arange(n)

    def _bootfunc(sampled_indices):
        idx = np.asarray(sampled_indices, dtype=int)
        boot_data = data.iloc[idx]
        result = compute_func(boot_data, **call_kwargs)
        return np.asarray(result['fractions'], dtype=float)

    # Astropy bootstrap uses numpy's global RNG. Save and restore state.
    rng_state = np.random.get_state()
    np.random.seed(seed)
    try:
        all_fracs = bootstrap(
            base_indices,
            bootnum=n_bootstrap,
            samples=n,
            bootfunc=_bootfunc,
        )
    finally:
        np.random.set_state(rng_state)

    n_bins = all_fracs.shape[1]
    fractions_median = np.full(n_bins, np.nan, dtype=float)
    ci_lo = np.full(n_bins, np.nan, dtype=float)
    ci_hi = np.full(n_bins, np.nan, dtype=float)

    valid_cols = np.any(np.isfinite(all_fracs), axis=0)
    if np.any(valid_cols):
        valid_fracs = all_fracs[:, valid_cols]
        fractions_median[valid_cols] = np.nanmedian(valid_fracs, axis=0)
        ci_lo[valid_cols] = np.nanquantile(valid_fracs, alpha, axis=0)
        ci_hi[valid_cols] = np.nanquantile(valid_fracs, 1.0 - alpha, axis=0)

    return {
        'fractions_median': fractions_median,
        'ci_lo': ci_lo,
        'ci_hi': ci_hi,
        'all_fractions': all_fracs,
        'bin_centers': result0['bin_centers'],
    }

def compute_completeness(truths, prefix, mag_bins, subset_mask=None,
                         mag_col='mag_i', min_count=1, compute_ci=True):
    """
    Completeness = matched_truth / total_truth, binned by truth magnitude

    Parameters
    ----------
    truths : pd.DataFrame
        Truth rows from split_truth_det().
    prefix : str
        'dd' or 'lsst'.
    mag_bins : array-like
        Bin edges.
    subset_mask : array-like of bool, optional
        Additional mask to apply (e.g., isolated only).
    mag_col : str
        Magnitude column. Default 'mag_i'.
    min_count : int
        Minimum per bin.
    compute_ci : bool
        If True, compute confidence intervals in binned_fraction.

    Returns
    -------
    dict
        Output from binned_fraction().
    """
    t = truths[subset_mask] if subset_mask is not None else truths
    return binned_fraction(
        t[f'{prefix}_is_matched'],
        t[mag_col],
        mag_bins,
        min_count=min_count,
        compute_ci=compute_ci,
    )

def compute_purity(dets, truths, prefix, mag_bins, mag_col='mag_i',
                   min_count=1):
    """
    Purity = matched_det / total_det, binned by matched truth magnitude.

    Only detections with matched truth magnitude are included in the bins.

    Parameters
    ----------
    dets : pd.DataFrame
        Detection rows from split_truth_det().
    truths : pd.DataFrame
        Truth rows from split_truth_det().
    prefix : str
        'dd' or 'lsst'.
    mag_bins : array-like
        Bin edges.
    mag_col : str
        Magnitude column in truths. Default 'mag_i'.
    min_count : int
        Minimum per bin.

    Returns
    -------
    dict
        Output from binned_fraction().
    """
    truth_mags = get_matched_truth_mag(dets, truths, prefix, mag_col)
    valid = np.isfinite(truth_mags.values)
    return binned_fraction(
        dets[f'{prefix}_is_matched'].values[valid],
        truth_mags.values[valid],
        mag_bins,
        min_count=min_count,
    )


def compute_purity_by_det_mag(dets, prefix, mag_bins, mag_col='mag_i',
                              min_count=1):
    """
    Purity = matched_det / total_det, binned by detection magnitude.

    Parameters
    ----------
    dets : pd.DataFrame
        Detection rows.
    prefix : str
        'dd' or 'lsst'.
    mag_bins : array-like
        Bin edges.
    mag_col : str
        Magnitude column in detections. Default 'mag_i'.
    min_count : int
        Minimum per bin.

    Returns
    -------
    dict
        Output from binned_fraction().
    """
    valid = np.isfinite(dets[mag_col].values)
    return binned_fraction(
        dets[f'{prefix}_is_matched'].values[valid],
        dets[mag_col].values[valid],
        mag_bins,
        min_count=min_count,
    )


def compute_binned_f1(truths, dets, prefix, mag_bins, mag_col='mag_i', min_count=1):
    """
    Compute F1 score per magnitude bin using binned completeness and purity.

    Parameters
    ----------
    truths : pd.DataFrame
        Truth rows from split_truth_det().
    dets : pd.DataFrame
        Detection rows from split_truth_det().
    prefix : str
        'dd' or 'lsst'.
    mag_bins : array-like
        Bin edges.
    mag_col : str
        Magnitude column. Default 'mag_i'.
    min_count : int
        Minimum per bin.

    Returns
    -------
    dict
        Output with bin_centers, f1, ci_lo, ci_hi, counts.
    """
    # Completeness: matched_truth / total_truth, binned by truth mag
    comp = compute_completeness(truths, prefix, mag_bins, mag_col=mag_col, min_count=min_count)
    # Purity: matched_det / total_det, binned by matched truth mag
    pur = compute_purity(dets, truths, prefix, mag_bins, mag_col=mag_col, min_count=min_count)
    # F1 = 2 * (comp * pur) / (comp + pur)
    completeness = comp['fractions']
    purity = pur['fractions']
    with np.errstate(invalid='ignore', divide='ignore'):
        f1 = (2 * completeness * purity) / (completeness + purity)
    # CIs: propagate as min/max of endpoints (not exact, but informative)
    ci_lo = np.fmax(0, 2 * comp['ci_lo'] * pur['ci_lo'] / (comp['ci_lo'] + pur['ci_lo']))
    ci_hi = np.fmin(1, 2 * comp['ci_hi'] * pur['ci_hi'] / (comp['ci_hi'] + pur['ci_hi']))
    return {
        'bin_centers': comp['bin_centers'],
        'fractions': f1,
        'ci_lo': ci_lo,
        'ci_hi': ci_hi,
        'counts': comp['counts'],
        'flagged_counts': comp['flagged_counts'],
    }

def compute_blend_loss(truths, prefix, mag_bins, mag_col='mag_i',
                       min_count=1):
    """
    Blend loss = blended_away / total_truth, binned by truth magnitude.

    Parameters
    ----------
    truths : pd.DataFrame
        Truth rows.
    prefix : str
        'dd' or 'lsst'.
    mag_bins : array-like
        Bin edges.
    mag_col : str
        Magnitude column. Default 'mag_i'.
    min_count : int
        Minimum per bin.

    Returns
    -------
    dict
        Output from binned_fraction().
    """
    return binned_fraction(
        truths[f'{prefix}_is_blended_away'],
        truths[mag_col],
        mag_bins,
        min_count=min_count,
    )


def compute_unrec_blend_frac(truths, prefix, mag_bins, mag_col='mag_i',
                             min_count=1):
    """
    Unrecognized blend fraction = unrec_blend / total_truth.

    Includes both matched survivors and blended-away victims.

    Parameters
    ----------
    truths : pd.DataFrame
        Truth rows.
    prefix : str
        'dd' or 'lsst'.
    mag_bins : array-like
        Bin edges.
    mag_col : str
        Magnitude column. Default 'mag_i'.
    min_count : int
        Minimum per bin.

    Returns
    -------
    dict
        Output from binned_fraction().
    """
    return binned_fraction(
        truths[f'{prefix}_is_unrec_blend'],
        truths[mag_col],
        mag_bins,
        min_count=min_count,
    )


def compute_shred_frac(truths, prefix, mag_bins, mag_col='mag_i',
                       min_count=1):
    """
    Shred fraction = shredded / total_truth, binned by truth magnitude.

    Parameters
    ----------
    truths : pd.DataFrame
        Truth rows.
    prefix : str
        'dd' or 'lsst'.
    mag_bins : array-like
        Bin edges.
    mag_col : str
        Magnitude column. Default 'mag_i'.
    min_count : int
        Minimum per bin.

    Returns
    -------
    dict
        Output from binned_fraction().
    """
    return binned_fraction(
        truths[f'{prefix}_is_shredded'],
        truths[mag_col],
        mag_bins,
        min_count=min_count,
    )


def compute_missed_frac(truths, prefix, mag_bins, mag_col='mag_i',
                        min_count=1):
    """
    Missed fraction = missed / total_truth, binned by truth magnitude.

    Parameters
    ----------
    truths : pd.DataFrame
        Truth rows.
    prefix : str
        'dd' or 'lsst'.
    mag_bins : array-like
        Bin edges.
    mag_col : str
        Magnitude column. Default 'mag_i'.
    min_count : int
        Minimum per bin.

    Returns
    -------
    dict
        Output from binned_fraction().
    """
    return binned_fraction(
        truths[f'{prefix}_is_missed'],
        truths[mag_col],
        mag_bins,
        min_count=min_count,
    )


def compute_spurious_frac(dets, prefix, mag_bins, mag_col='mag_i',
                          min_count=1):
    """
    Spurious fraction = spurious / total_det, by detection magnitude.

    Parameters
    ----------
    dets : pd.DataFrame
        Detection rows.
    prefix : str
        'dd' or 'lsst'.
    mag_bins : array-like
        Bin edges.
    mag_col : str
        Magnitude column in detections. Default 'mag_i'.
    min_count : int
        Minimum per bin.

    Returns
    -------
    dict
        Output from binned_fraction().
    """
    valid = np.isfinite(dets[mag_col].values)
    return binned_fraction(
        dets[f'{prefix}_is_spurious'].values[valid],
        dets[mag_col].values[valid],
        mag_bins,
        min_count=min_count,
    )


def compute_resolved_frac(truths, prefix, mag_bins, mag_col='mag_i',
                          min_count=1):
    """
    Resolved fraction = resolved_blend / part_of_blend, by truth magnitude.

    Only includes truth objects that participate in a blend.

    Parameters
    ----------
    truths : pd.DataFrame
        Truth rows.
    prefix : str
        'dd' or 'lsst'.
    mag_bins : array-like
        Bin edges.
    mag_col : str
        Magnitude column. Default 'mag_i'.
    min_count : int
        Minimum per bin.

    Returns
    -------
    dict
        Output from binned_fraction(). If there are no blended truths,
        arrays are returned with NaN fractions/CIs and zero counts.
    """
    blend_mask = truths[f'{prefix}_is_part_of_blend'].values.astype(bool)
    t = truths[blend_mask]
    if len(t) == 0:
        n_bins = len(mag_bins) - 1
        return {
            'bin_centers': 0.5 * (mag_bins[:-1] + mag_bins[1:]),
            'fractions': np.full(n_bins, np.nan),
            'ci_lo': np.full(n_bins, np.nan),
            'ci_hi': np.full(n_bins, np.nan),
            'counts': np.zeros(n_bins, dtype=int),
            'flagged_counts': np.zeros(n_bins, dtype=int),
        }
    return binned_fraction(
        t[f'{prefix}_is_resolved_blend'],
        t[mag_col],
        mag_bins,
        min_count=min_count,
    )

def compute_shred_det_frac(dets, prefix, mag_bins, mag_col='mag_i',
                           min_count=1):
    """
    Shred det fraction = is_shred / total_det, by detection magnitude.

    Parameters
    ----------
    dets : pd.DataFrame
        Detection rows.
    prefix : str
        'dd' or 'lsst'.
    mag_bins : array-like
        Bin edges.
    mag_col : str
        Magnitude column in detections. Default 'mag_i'.
    min_count : int
        Minimum per bin.

    Returns
    -------
    dict
        Output from binned_fraction().
    """
    valid = np.isfinite(dets[mag_col].values)
    return binned_fraction(
        dets[f'{prefix}_is_shred'].values[valid],
        dets[mag_col].values[valid],
        mag_bins,
        min_count=min_count,
    )


def compute_all_binned_metrics(analysis_df, prefix, mag_bins, mag_col='mag_i',
                               min_count=1):
    """
    Compute all binned metrics at once.

    Parameters
    ----------
    analysis_df : pd.DataFrame
        Classified analysis table.
    prefix : str
        'dd' or 'lsst'.
    mag_bins : array-like
        Bin edges.
    mag_col : str
        Magnitude column. Default 'mag_i'.
    min_count : int
        Minimum per bin.

    Returns
    -------
    dict of {str: dict}
        Mapping of metric name -> binned_fraction-style output.
    """
    truths, dets = split_truth_det(analysis_df, prefix)
    p = prefix

    isolated_mask = ~truths[f'{p}_is_part_of_blend'].values.astype(bool)
    blended_mask = truths[f'{p}_is_part_of_blend'].values.astype(bool)

    return {
        'completeness': compute_completeness(
            truths, p, mag_bins, mag_col=mag_col, min_count=min_count),
        'completeness_isolated': compute_completeness(
            truths, p, mag_bins, subset_mask=isolated_mask,
            mag_col=mag_col, min_count=min_count),
        'completeness_blended': compute_completeness(
            truths, p, mag_bins, subset_mask=blended_mask,
            mag_col=mag_col, min_count=min_count),
        'purity_by_truth_mag': compute_purity(
            dets, truths, p, mag_bins, mag_col=mag_col, min_count=min_count),
        'purity_by_det_mag': compute_purity_by_det_mag(
            dets, p, mag_bins, mag_col=mag_col, min_count=min_count),
        'blend_loss': compute_blend_loss(
            truths, p, mag_bins, mag_col=mag_col, min_count=min_count),
        'unrec_blend_frac': compute_unrec_blend_frac(
            truths, p, mag_bins, mag_col=mag_col, min_count=min_count),
        'shred_frac': compute_shred_frac(
            truths, p, mag_bins, mag_col=mag_col, min_count=min_count),
        'missed_frac': compute_missed_frac(
            truths, p, mag_bins, mag_col=mag_col, min_count=min_count),
        'spurious_frac': compute_spurious_frac(
            dets, p, mag_bins, mag_col=mag_col, min_count=min_count),
        'resolved_frac': compute_resolved_frac(
            truths, p, mag_bins, mag_col=mag_col, min_count=min_count),
        'shred_det_frac': compute_shred_det_frac(
            dets, p, mag_bins, mag_col=mag_col, min_count=min_count),
    }