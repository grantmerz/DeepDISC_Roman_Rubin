"""
blend_metrics.py — single source of truth for the DeepDISC blend metrics.

Every population count, metric formula, partition assert, and CI helper lives
here. compute_scalar_table.py computes each metric once over the evaluation-cut
sample; plot_binned_metrics.py computes the same metric per magnitude bin. A
scalar is literally the one-bin case of the binned computation — neither script
re-implements a mask or a formula.

Flag columns are '{prefix}_{flag}' (prefix = 'dd' or 'lsst'); truth rows are
catalog_key == 'lsst_truth', detection rows == '{prefix}_det'. Definitions
follow subsec:catalog_matching / subsec:metrics and the authoritative
fof_classify.py + run_fof_classify.py. Does NOT import the project-space
metrics.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAG_LIMITS = {'power_law': 26.22, 'gold': 25.3, 'nominal': 26.42}
MAG_COL = 'mag_i'
MIN_MAG_BIN = 18.5
MAX_MAG_BIN = 27.5
TRUTH_KEY = 'lsst_truth'
DET_KEYS = {'dd': 'dd_det', 'lsst': 'lsst_det'}
CLASS_COL = 'class' # truth truth_type, DD label, LSST extendedness all renamed to this
# raw integer code -> canonical 'g'/'s', keyed by catalog_key. These encodings
# differ across catalogs and MUST be mapped before any comparison:
#   lsst_truth : truth_type    2 = star, 1 = galaxy
#   dd_det     : DD label      0 = galaxy, 1 = star
#   lsst_det   : extendedness  0 = star, 1 = galaxy   (NaN where the measurement failed)
CLASS_MAPS = {
    'lsst_truth': {1: 'g', 2: 's'},
    'dd_det':     {0: 'g', 1: 's'},
    'lsst_det':   {0: 's', 1: 'g'},
}

# per-object flags every classified parquet must carry (prefixed with '{p}_'),
# is_sig_blend included: it is written by add_sig_blend_flag AFTER classify_stage2,
# so a parquet from before that step loads fine but silently NaNs every blend
# metric. Guard at load instead (see load_analysis_df).
_REQUIRED_FLAGS = [
    'is_matched', 'is_missed', 'is_absorbed', 'is_part_of_blend',
    'is_resolved_blend', 'is_survivor', 'is_shredded',
    'is_spurious', 'is_shred', 'is_absorber', 'is_sig_blend',
]

# metric -> (numerator_key, denominator_key) into the compute_counts dict.
# Single source for scalar values, binned values, and beta-binomial CIs. Matches
# tab:metrics_formulas exactly. F1 handled separately (function of C and P).
METRIC_NUMDEN = {
    'completeness':      ('N_t_matched', 'N_t'),
    'missed_frac':       ('N_t_missed', 'N_t'),
    'sig_blend_frac':    ('N_t_sig', 'N_t'),
    'shredded_frac':     ('N_t_shredded', 'N_t_matched'),
    'resolved_frac':     ('N_t_res', 'N_t_sig'),
    'survivor_frac':     ('N_t_surv', 'N_t_sig'),
    'absorbed_frac':     ('N_t_abs', 'N_t_sig'),
    'purity':            ('N_d_matched', 'N_d'),
    'spurious_frac':     ('N_d_spurious', 'N_d'),
    'shred_frac':        ('N_d_shred', 'N_d'),
    'unrec_blend_frac':  ('N_d_abr', 'N_d_matched'),
}
TRUTH_METRICS = ['completeness', 'missed_frac', 'sig_blend_frac', 'shredded_frac',
                 'resolved_frac', 'survivor_frac', 'absorbed_frac']
DET_METRICS = ['purity', 'spurious_frac', 'shred_frac', 'unrec_blend_frac']
BLEND_COND_METRICS = ['resolved_frac', 'survivor_frac', 'absorbed_frac']

CLASS_METRICS = ['galaxy_recall', 'galaxy_precision', 'star_recall', 'star_precision']
# metric -> (truth_class, predicted_class) for the (population, indicator) split:
#   recall(c):    population = classified truths with truth_cls == c,
#                 indicator  = pred_cls == c
#   precision(c): population = classified truths with pred_cls  == c,
#                 indicator  = truth_cls == c
# encoded as (axis, class_label): axis 't' selects population by truth_cls (recall),
# axis 'p' selects population by pred_cls (precision); indicator is the OTHER axis == class_label.
_CLASS_SPEC = {
    'galaxy_recall':    ('t', 'g'),
    'galaxy_precision': ('p', 'g'),
    'star_recall':      ('t', 's'),
    'star_precision':   ('p', 's'),
}
# metric -> (numerator_confusion_cell, denominator_confusion_cells) for reporting
# counts and the long-format CSV. Cells are ('N_gg','N_gs','N_sg','N_ss').
CLASS_METRIC_CELLS = {
    'galaxy_recall':    ('N_gg', ('N_gg', 'N_gs')),
    'galaxy_precision': ('N_gg', ('N_gg', 'N_sg')),
    'star_recall':      ('N_ss', ('N_ss', 'N_sg')),
    'star_precision':   ('N_ss', ('N_ss', 'N_gs')),
}

# Short abbreviations for filenames / compact display
METRIC_ABBREV = {
    'f1': 'f1', 'unrec_blend_frac': 'ubf', 'purity': 'p', 'completeness': 'c',
    'resolved_frac': 'rf', 'spurious_frac': 'spf', 'shred_frac': 'shf',
    'missed_frac': 'mf', 'absorbed_frac': 'af', 'survivor_frac': 'suf',
    'sig_blend_frac': 'sbf', 'shredded_frac': 'sdf',
}
CLASS_METRIC_ABBREV = {
    'galaxy_recall': 'gr', 'galaxy_precision': 'gp',
    'star_recall': 'sr', 'star_precision': 'sp',
}


# ---------------------------------------------------------------------------
# User helpers (reused verbatim; load_analysis_df lightly adapted to raise +
# guard columns instead of returning None)
# ---------------------------------------------------------------------------
def combo_tag(score_thresh: float, nms_thresh: float) -> str:
    """Threshold subdir tag, matching run_fof_classify.py's naming exactly
    (f's{score_thresh}_n{nms_thresh}' on the parsed floats)."""
    return f's{score_thresh}_n{nms_thresh}'


def split_truth_det(analysis_df, prefix):
    """Split a classified long-format table into (truths, dets) subframes."""
    det_key = DET_KEYS[prefix]
    truths = analysis_df[analysis_df['catalog_key'] == TRUTH_KEY].copy()
    dets = analysis_df[analysis_df['catalog_key'] == det_key].copy()
    return truths, dets


def get_matched_truth_mag(dets, truths, prefix, mag_col=MAG_COL):
    """Matched truth's magnitude for each detection, aligned with dets.index (NaN if unmatched)."""
    truth_mag = truths.set_index('row_index')[mag_col]
    matched = dets[f'{prefix}_is_matched'].astype(bool)
    out = pd.Series(np.nan, index=dets.index)
    out.loc[matched] = dets.loc[matched, f'{prefix}_matched_id'].map(truth_mag).values
    return out


def _safe_div(num, denom):
    """Safe division returning 0.0 when denominator is 0."""
    return num / denom if denom > 0 else 0.0


def load_analysis_df(run_dir: Path, prefix: str, linking_length: str,
                     combo: Optional[str], mag_limit_val: float) -> pd.DataFrame:
    """Load one classified analysis parquet for DD or LSST.

    combo=None omits the threshold subdir (LSST is threshold-independent, matching
    run_fof_classify.py's threshold_tag=None path). Raises on a missing file or a
    parquet that predates add_sig_blend_flag.
    """
    run_dir = Path(run_dir)
    cats_dir = run_dir / 'analysis_cats'
    if prefix == 'dd':
        parquet = cats_dir / 'dd' / linking_length / combo / f'class_{mag_limit_val:.2f}.parquet'
    else:
        parquet = cats_dir / 'lsst' / linking_length / f'class_{mag_limit_val:.2f}.parquet'
    if not parquet.exists():
        raise FileNotFoundError(f'classified parquet not found: {parquet}')
    df = pd.read_parquet(parquet)
    missing = [f'{prefix}_{f}' for f in _REQUIRED_FLAGS
               if f'{prefix}_{f}' not in df.columns]
    if MAG_COL not in df.columns:
        missing.append(MAG_COL)
    if missing:
        raise KeyError(
            f'{parquet} is missing columns {missing}. If {prefix}_is_sig_blend is '
            f'among them the parquet predates add_sig_blend_flag; re-run '
            f'run_fof_classify.py.'
        )
    return df

def _map_class(codes, catalog_key):
    """Map a Series of raw integer class codes to canonical 'g'/'s' via
    CLASS_MAPS[catalog_key]. NaN stays NaN (undefined class -> unclassified).
    Any non-NaN code not in the map raises: this catches unexpected encodings
    (e.g. truth_type == 3 or 0) loudly rather than silently dropping them,
    so an exclusion is always an explicit decision.
    """
    m = CLASS_MAPS[catalog_key]
    out = codes.map(m)
    bad = codes.notna() & out.isna()
    if bad.any():
        vals = sorted(pd.unique(codes[bad]))
        raise ValueError(
            f"{catalog_key}: class codes {vals} are not in CLASS_MAPS[{catalog_key!r}] "
            f"= {m}. Add them explicitly (or filter upstream) rather than dropping them.")
    return out

def attach_pred_class(truths, dets, prefix):
    """Return the truth subframe with three canonical-class columns added:
 
      truth_cls          : 'g'/'s' from lsst_truth CLASS_COL
      pred_cls           : 'g'/'s' from the MATCHED detection's CLASS_COL,
                           routed through the det-side CLASS_MAPS; NaN when the
                           truth is unmatched (matched_id == -1) or the matched
                           detection has no defined class (LSST extendedness NaN)
      {p}_is_classified  : is_matched & truth_cls.notna() & pred_cls.notna()
 
    Uses {p}_matched_id on truth rows (a det row_index, -1 for no match) to pull
    the matched detection's class. The parquet is the buffered table, so the
    matched detection is present even when it fails m_cut -- a truth object is
    classified via its match regardless of the det's own magnitude. 
    No magnitude cut is applied here; callers cut afterward.
    """
    p = prefix
    det_key = DET_KEYS[prefix]
    mid_col = f'{p}_matched_id'
    for col, frame, name in [(CLASS_COL, truths, 'truth'),
                             (CLASS_COL, dets, det_key),
                             (mid_col, truths, 'truth')]:
        if col not in frame.columns:
            raise KeyError(
                f"classification needs {col!r} on {name} rows but it is absent. "
                f"For DD confirm the pred JSON carried a class column; for LSST/truth "
                f"confirm the col_config renamed extendedness/truth_type to {CLASS_COL!r}.")
    if not dets['row_index'].is_unique:
        raise ValueError(f"{det_key}: row_index is not unique; cannot index matched_id "
                         f"into the detection class column.")
 
    t = truths.copy()
    t['truth_cls'] = _map_class(t[CLASS_COL], 'lsst_truth')
 
    det_cls_raw = dets.set_index('row_index')[CLASS_COL]
    mid = t[mid_col]
    matched_det_code = mid.map(det_cls_raw)          # NaN where mid == -1 (no such row_index)
    t['pred_cls'] = _map_class(matched_det_code, det_key)
 
    t[f'{p}_is_classified'] = (
        t[f'{p}_is_matched'] & t['truth_cls'].notna() & t['pred_cls'].notna())
    return t

# ---------------------------------------------------------------------------
# Counts
# ---------------------------------------------------------------------------
def compute_counts(truths, dets, prefix, mag_mask_t, mag_mask_d, mag_col=MAG_COL):
    """Return every N_* for one pipeline under a magnitude selection.

    mag_mask_t / mag_mask_d are boolean Series aligned to the full truth / det
    subframes. A half-open interval mask (lo <= mag < hi) is False for NaN mags,
    so NaN-mag detections drop out of N_d and every detection numerator without a
    separate step. n_det_nan_mag / f_det_nan_mag are computed on the full det
    subframe (pre-mask) for reporting.

    Asserts 1-3 (the disjoint partitions) run on the returned counts. They hold
    bin-by-bin because every object carries exactly one flag from each partition.
    Maximality asserts 4-5 are full-frame properties; see assert_maximality.
    """
    p = prefix
    t = truths[mag_mask_t]
    d = dets[mag_mask_d]

    sig_t = t[f'{p}_is_sig_blend'] & t[f'{p}_is_part_of_blend']
    c = {
        'N_t':          len(t),
        'N_t_matched':  int(t[f'{p}_is_matched'].sum()),
        'N_t_missed':   int(t[f'{p}_is_missed'].sum()),
        'N_t_abs_all':  int(t[f'{p}_is_absorbed'].sum()),
        'N_t_sig':      int(sig_t.sum()),
        'N_t_res':      int((t[f'{p}_is_resolved_blend'] & t[f'{p}_is_sig_blend']).sum()),
        'N_t_surv':     int((t[f'{p}_is_survivor'] & t[f'{p}_is_sig_blend']).sum()),
        'N_t_abs':      int((t[f'{p}_is_absorbed'] & t[f'{p}_is_sig_blend']).sum()),
        'N_t_shredded': int(t[f'{p}_is_shredded'].sum()),
        'N_d':          len(d),
        'N_d_matched':  int(d[f'{p}_is_matched'].sum()),
        'N_d_spurious': int(d[f'{p}_is_spurious'].sum()),
        'N_d_shred':    int(d[f'{p}_is_shred'].sum()),
        'N_d_abr':      int((d[f'{p}_is_absorber'] & d[f'{p}_is_sig_blend']).sum()),
    }
    n_nan = int(dets[mag_col].isna().sum())
    c['n_det_nan_mag'] = n_nan
    c['f_det_nan_mag'] = n_nan / len(dets) if len(dets) else 0.0

    # 1. blend-conditional partition — catches significance-broadcast / component bugs
    assert c['N_t_res'] + c['N_t_surv'] + c['N_t_abs'] == c['N_t_sig'], (
        f"blend partition broken: res({c['N_t_res']}) + surv({c['N_t_surv']}) + "
        f"abs({c['N_t_abs']}) != sig({c['N_t_sig']})")
    # 2. truth catalog partition
    assert c['N_t_matched'] + c['N_t_missed'] + c['N_t_abs_all'] == c['N_t'], (
        f"truth partition broken: matched({c['N_t_matched']}) + "
        f"missed({c['N_t_missed']}) + abs_all({c['N_t_abs_all']}) != N_t({c['N_t']})")
    # 3. detection catalog partition
    assert c['N_d_matched'] + c['N_d_shred'] + c['N_d_spurious'] == c['N_d'], (
        f"det partition broken: matched({c['N_d_matched']}) + shred({c['N_d_shred']}) "
        f"+ spurious({c['N_d_spurious']}) != N_d({c['N_d']})")
    return c

def assert_maximality(truths, dets, prefix):
    """Asserts 4-5: every absorber is matched, every shredded truth is matched.

    Full-frame properties of the Hungarian assignment; run once per pipeline, not
    per magnitude bin.
    """
    p = prefix
    absorber = dets[f'{p}_is_absorber']
    assert (absorber & dets[f'{p}_is_matched']).equals(absorber), (
        f"{prefix}: {(absorber & ~dets[f'{p}_is_matched']).sum()} absorbers are unmatched")
    shredded = truths[f'{p}_is_shredded']
    assert (shredded & truths[f'{p}_is_matched']).equals(shredded), (
        f"{prefix}: {(shredded & ~truths[f'{p}_is_matched']).sum()} shredded truths are unmatched")

def compute_class_counts(tcls, prefix, mag_mask):
    """Confusion cells + N_t_cls + n_matched_uncls for one pipeline under a
    truth magnitude selection.
 
    tcls  : output of attach_pred_class (truth rows with truth_cls/pred_cls/
            {p}_is_classified).
    mag_mask : boolean Series aligned to tcls (truth mag_i < m_cut, or a bin mask).
 
    Returns dict:
      N_t_cls, N_gg, N_gs, N_sg, N_ss, n_matched_uncls
    N_t_cls == N_gg+N_gs+N_sg+N_ss (asserted). n_matched_uncls = matched truths
    within the cut whose det class is undefined (the LSST-extendedness exclusion
    reported in the table caption); it is NOT part of N_t_cls.
    """
    iscls_col = f'{prefix}_is_classified'
    matched_col = f'{prefix}_is_matched'
    sel = mag_mask.to_numpy() if hasattr(mag_mask, 'to_numpy') else np.asarray(mag_mask)
 
    cls = tcls[sel & tcls[iscls_col].to_numpy()]
    tc = cls['truth_cls'].to_numpy()
    pc = cls['pred_cls'].to_numpy()
    counts = {
        'N_gg': int(((tc == 'g') & (pc == 'g')).sum()),
        'N_gs': int(((tc == 'g') & (pc == 's')).sum()),
        'N_sg': int(((tc == 's') & (pc == 'g')).sum()),
        'N_ss': int(((tc == 's') & (pc == 's')).sum()),
    }
    counts['N_t_cls'] = counts['N_gg'] + counts['N_gs'] + counts['N_sg'] + counts['N_ss']
    assert counts['N_t_cls'] == len(cls), (
        f"class partition broken: N_t_cls={counts['N_t_cls']} != classified rows={len(cls)} "
        f"(a classified truth has a class outside {{g,s}})")
 
    matched = tcls[sel & tcls[matched_col].to_numpy()]
    counts['n_matched_uncls'] = int((~matched[iscls_col].to_numpy()).sum())
    return counts

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(counts):
    """All scalar/bin metrics as ratios of the counts dict, matching
    tab:metrics_formulas. resolved/survivor/absorbed share the single N_t_sig in
    counts, so they sum to 1 (assert 1). F1 = harmonic mean of C and P."""
    m = {name: _safe_div(counts[num], counts[den])
         for name, (num, den) in METRIC_NUMDEN.items()}
    c, p = m['completeness'], m['purity']
    m['f1'] = _safe_div(2 * c * p, c + p)
    return m

def compute_class_metrics(counts):
    """Four ratios from the confusion cells, matching tab:class_metrics.
    _safe_div returns 0.0 on an empty denominator (a class with no members in
    this bin), consistent with the blend metrics."""
    out = {}
    for name, (num_cell, den_cells) in CLASS_METRIC_CELLS.items():
        num = counts[num_cell]
        den = sum(counts[c] for c in den_cells)
        out[name] = _safe_div(num, den)
    return out

# ---------------------------------------------------------------------------
# Per-metric populations (shared by the scalar and binned beta-binomial CIs)
# ---------------------------------------------------------------------------
def _metric_populations(t, d, prefix):
    """(population_df, numerator_indicator) per metric in METRIC_NUMDEN, for an
    already magnitude-masked (t, d). One definition of "what population and
    indicator does metric X's ratio come from", reused by scalar_cis
    (population = whole evaluation-cut sample) and compute_binned (population =
    the full range passed in, for fitting rho/m_bar once per metric) so both CI
    paths compute exactly over the objects each metric's ratio is actually
    computed over.

    Population sizes match compute_counts' denominators exactly (e.g.
    len(t[sig]) == N_t_sig) provided the data invariant compute_counts' own
    partition asserts already require holds: is_resolved_blend/is_survivor/
    is_absorbed implies is_part_of_blend. The self-check cross-checks this.
    """
    p = prefix
    sig = t[f'{p}_is_sig_blend'] & t[f'{p}_is_part_of_blend']
    t_matched = t[t[f'{p}_is_matched']]
    t_sig = t[sig]
    d_matched = d[d[f'{p}_is_matched']]
    return {
        'completeness':     (t, t[f'{p}_is_matched']),
        'missed_frac':      (t, t[f'{p}_is_missed']),
        'sig_blend_frac':   (t, sig),
        'shredded_frac':    (t_matched, t_matched[f'{p}_is_shredded']),
        'resolved_frac':    (t_sig, t_sig[f'{p}_is_resolved_blend']),
        'survivor_frac':    (t_sig, t_sig[f'{p}_is_survivor']),
        'absorbed_frac':    (t_sig, t_sig[f'{p}_is_absorbed']),
        'purity':           (d, d[f'{p}_is_matched']),
        'spurious_frac':    (d, d[f'{p}_is_spurious']),
        'shred_frac':       (d, d[f'{p}_is_shred']),
        'unrec_blend_frac': (d_matched, d_matched[f'{p}_is_absorber'] & d_matched[f'{p}_is_sig_blend']),
    }

def _class_populations(tcls, prefix):
    """(population_df, indicator) per class metric, over an already
    magnitude-masked, classified-only truth frame. One definition reused by the
    scalar and binned CIs, mirroring _metric_populations. Every population is
    truth rows carrying group_id, so estimate_icc / mean_cluster_size apply.
 
    recall(c):    population = classified truths with truth_cls == c; indicator = pred_cls == c
    precision(c): population = classified truths with pred_cls  == c; indicator = truth_cls == c
    """
    cls = tcls[tcls[f'{prefix}_is_classified']]
    out = {}
    for name, (axis, class_label) in _CLASS_SPEC.items():
        if axis == 't':                      # recall: population selected by truth class
            pop = cls[cls['truth_cls'] == class_label]
            ind = (pop['pred_cls'] == class_label)
        else:                                # precision: population selected by predicted class
            pop = cls[cls['pred_cls'] == class_label]
            ind = (pop['truth_cls'] == class_label)
        out[name] = (pop, ind)
    return out
# ---------------------------------------------------------------------------
# Confidence intervals — beta-binomial with a design effect for within-group
# (cluster) correlation, uniformly by group_id
# ---------------------------------------------------------------------------
# group_id already partitions every object (an isolated object is a singleton
# group, a blend is a multi-object group). A per-metric intra-cluster
# correlation rho and size-weighted mean cluster size m_bar give a design effect
# DEFF = 1 + rho*(m_bar - 1). rho is fit once per metric on the full population
# (unstable per bin when k ~ 0); m_bar is always taken on the population the
# interval is for (scalar sample or single bin) -- it doesn't depend on the
# indicator, so it's stable everywhere. Shrinking both k and n by DEFF before taking
# Beta(k+1, n-k+1) quantiles widens the interval for correlated metrics
# (resolved/survivor/absorbed) while leaving singleton-dominated catalog
# metrics (completeness, purity) at ~DEFF=1, the independent Cameron/flat
# interval. Deterministic, no resampling -- this replaces the old cluster
# bootstrap and the Wilson interval alike.
def estimate_icc(group_ids, indicator):
    """Fleiss-Cochran ANOVA-type intra-cluster correlation for a binary
    indicator, clustered by group_id, robust to unequal cluster sizes and
    singletons.

    rho = 1 - MSW / (p_hat (1 - p_hat)),  where
      MSW = [ sum_g s_g (m_g - s_g) / m_g ] / [ sum_g (m_g - 1) ]

    Singletons (m_g = 1): the numerator term s_g(m_g - s_g)/m_g = 0 (since
    s_g in {0,1} => s_g(1 - s_g) = 0) and the dof term (m_g - 1) = 0, so they
    contribute nothing to either sum and cancel structurally. Under pure
    independence this returns 0 regardless of singleton fraction.

    Returns (rho, rho_raw): rho is clamped to [0, inf) via max(0, rho_raw) --
    that's what feeds design_effect/betabinom_ci, since a design effect below 1
    isn't meaningful. rho_raw is the unclamped ANOVA estimate, so a caller can
    tell "genuinely ~0" (rho_raw ~ 0, e.g. catalog metrics like completeness)
    apart from "anti-correlated, floored at 0" (rho_raw < 0). The latter is
    real and expected for survivor/absorbed: within a blend group those
    indicators trade off (if one member survives, the others are absorbed),
    which is negative within-cluster correlation -- unlike resolved, which is
    all-or-nothing within a group (rho_raw ~= 1, positive).
    """
    group_ids = np.asarray(group_ids)
    y = np.asarray(indicator, dtype=float)
    if y.size == 0:
        return 0.0, 0.0
    p_hat = y.mean()
    if p_hat <= 0.0 or p_hat >= 1.0:
        return 0.0, 0.0                 # no variance to partition

    _, inv = np.unique(group_ids, return_inverse=True)
    m_g = np.bincount(inv)              # cluster sizes (>= 1)
    s_g = np.bincount(inv, weights=y)   # successes per cluster

    msw_num = (s_g * (m_g - s_g) / m_g).sum()   # m_g never 0; singleton term = 0
    dof = (m_g - 1).sum()                        # singletons contribute 0
    if dof <= 0:
        return 0.0, 0.0                          # all singletons: no clustering info
    rho_raw = 1.0 - (msw_num / dof) / (p_hat * (1.0 - p_hat))
    return max(0.0, rho_raw), rho_raw


def mean_cluster_size(group_ids):
    """Size-weighted mean cluster size m_bar = sum m_g^2 / sum m_g over the
    groups present in the population the interval is FOR (the evaluation-cut
    sample for a scalar, one bin for a binned CI).

    Size-weighted, not the simple mean: with unequal cluster sizes Var(k/n)
    scales with sum m_g^2 / n (Kish), so at rho = 1 DEFF equals this exactly;
    the simple mean understates it (e.g. LSST resolved_frac, i < 25.3: 1.31
    simple vs 1.49 weighted). Singleton-dominated populations still give
    m_bar ~ 1, so DEFF ~ 1 and the interval collapses to Cameron/flat."""
    group_ids = np.asarray(group_ids)
    if group_ids.size == 0:
        return 1.0
    _, counts = np.unique(group_ids, return_counts=True)
    return float((counts ** 2).sum() / counts.sum())


def design_effect(rho, m_bar):
    """DEFF = 1 + rho (m_bar - 1). rho is clamped >= 0 and m_bar >= 1, so
    DEFF >= 1. Assert as a tripwire."""
    deff = 1.0 + rho * (m_bar - 1.0)
    assert deff >= 1.0 - 1e-12, f"DEFF < 1 ({deff}); rho or m_bar invalid"
    return max(1.0, deff)


def betabinom_ci(k, n, rho, m_bar, conf=0.95):
    """Design-effect beta interval. Shrinks both k and n by DEFF to preserve
    the point estimate k/n, then takes Beta(k_eff + 1, n_eff - k_eff + 1)
    quantiles.

    - Point estimate unchanged (caller reports raw k/n separately).
    - rho = 0 or m_bar = 1  => DEFF = 1 => reduces exactly to Cameron/flat
      Beta(k + 1, n - k + 1).
    - k = 0 => k_eff = 0 => Beta(1, n_eff + 1): a proper, non-degenerate
      interval with hi > 0; lo is a small positive number just above 0
      (this equal-tailed flat-prior interval, unlike Clopper-Pearson, never
      hard-clips to exactly 0/1 -- that is the correct, intended behavior).
    - k = n symmetric: Beta(n_eff + 1, 1), lo just below 1, hi < 1.
    """
    if n == 0:
        return np.nan, np.nan
    deff = design_effect(rho, m_bar)
    k_eff = k / deff
    n_eff = n / deff
    a = k_eff + 1.0
    b = n_eff - k_eff + 1.0
    alpha = 1.0 - conf
    lo = float(beta_dist.ppf(alpha / 2.0, a, b))
    hi = float(beta_dist.ppf(1.0 - alpha / 2.0, a, b))
    return lo, hi


def _f1(c, p):
    if c <= 0.0 or p <= 0.0:
        return 0.0                      # harmonic mean with a zero arg -> 0 (avoids 0/0)
    return 2.0 * c * p / (c + p)


def f1_interval(c_lo, c_hi, p_lo, p_hi):
    """Conservative F1 interval via monotone endpoint propagation of the
    completeness and purity intervals. F1 = 2CP/(C+P) is strictly increasing
    in both C and P, so the image of [C_lo,C_hi] x [P_lo,P_hi] under F1 is
    exactly [F1(C_lo,P_lo), F1(C_hi,P_hi)]. Over-covers unless the C and P
    errors are perfectly correlated -- not a 95% CI in the same sense as the
    beta-binomial intervals; label it as conservative in any output/caption.
    """
    return _f1(c_lo, p_lo), _f1(c_hi, p_hi)


def scalar_cis(truths, dets, prefix, m_cut, conf=0.95):
    """Beta-binomial CI per scalar metric: fit rho and m_bar once on each
    metric's own population (§2.7 table, via _metric_populations), then
    design-effect-widen the flat Beta interval around that metric's own
    (k, n). Returns metric -> dict(lo, hi, rho, rho_raw, deff).

    rho_raw is the unclamped ICC (see estimate_icc) -- carried through so a
    reporting table can distinguish genuinely-uncorrelated (rho_raw ~ 0) from
    anti-correlated-and-floored (rho_raw < 0, rho = 0, DEFF = 1), which is the
    expected story for survivor/absorbed.

    F1 has no rho/deff of its own -- its interval comes from f1_interval
    (monotone endpoint propagation over completeness and purity), so its
    rho/rho_raw/deff entries are NaN.
    """
    p = prefix
    tc = truths[truths[MAG_COL] < m_cut]
    dc = dets[(dets[MAG_COL] < m_cut) & dets[MAG_COL].notna()]

    pops = _metric_populations(tc, dc, p)
    out = {}
    for name, (pop, ind) in pops.items():
        gids, y = pop['group_id'].to_numpy(), ind.to_numpy()
        k, n = int(y.sum()), y.size
        rho, rho_raw = estimate_icc(gids, y)
        m_bar = mean_cluster_size(gids)
        lo, hi = betabinom_ci(k, n, rho, m_bar, conf)
        out[name] = {'lo': lo, 'hi': hi, 'rho': rho, 'rho_raw': rho_raw,
                     'deff': design_effect(rho, m_bar)}

    f1_lo, f1_hi = f1_interval(out['completeness']['lo'], out['completeness']['hi'],
                               out['purity']['lo'], out['purity']['hi'])
    out['f1'] = {'lo': f1_lo, 'hi': f1_hi, 'rho': np.nan, 'rho_raw': np.nan, 'deff': np.nan}
    return out

def class_scalar_cis(tcls, prefix, m_cut, conf=0.95):
    """Beta-binomial CI per class metric: fit rho and m_bar once on each
    metric's own population (_class_populations), then design-effect-widen the
    flat Beta interval around that metric's (k, n). Mirrors scalar_cis exactly.
    Returns metric -> dict(lo, hi, rho, rho_raw, deff).
    """
    cut = tcls[tcls[MAG_COL] < m_cut]
    pops = _class_populations(cut, prefix)
    out = {}    
    for name, (pop, ind) in pops.items():
        gids, y = pop['group_id'].to_numpy(), ind.to_numpy().astype(float)
        k, n = int(y.sum()), y.size
        rho, rho_raw = estimate_icc(gids, y)
        m_bar = mean_cluster_size(gids)
        lo, hi = betabinom_ci(k, n, rho, m_bar, conf)
        out[name] = {'lo': lo, 'hi': hi, 'rho': rho, 'rho_raw': rho_raw,
                     'deff': design_effect(rho, m_bar)}
    return out

# ---------------------------------------------------------------------------
# Binned computation (shared by the plotting script)
# ---------------------------------------------------------------------------
def compute_binned(truths, dets, prefix, edges, min_count=1, min_bin_count_blend=None,
                   conf=0.95, mag_col=MAG_COL):
    """Per-bin metrics + beta-binomial CIs + per-bin numerator/denominator
    counts.

    Truth metrics are binned by truth mag_i, detection metrics by observed mag_i,
    using the same edges. Bins whose denominator < min_count are set to NaN for
    plotting (counts are left untouched, so the asserts inside compute_counts run
    on raw counts, never on masked values).

    rho is fit ONCE per metric on the full population passed in (truths, dets --
    not re-fit per bin, which is unstable at the bright end where k ~ 0). m_bar
    is computed PER BIN on that bin's population: a significant blend's members
    usually land in different 0.5-mag bins, so the full-population m_bar (~2.2)
    would overstate each bin's clustering (per-bin ~1.0-1.35). Same rule as
    scalar_cis (m_bar on the population the interval is for), so binned and
    scalar CIs share one method. Intervals are pointwise. Wilson intervals
    and bootstrap are no longer used anywhere in this module.

    min_bin_count_blend: separate, usually much larger, threshold for the
    blend-conditional metrics (BLEND_COND_METRICS), whose denominator is N_t_sig
    and can be single digits per bin even where N_t/N_d are in the thousands.
    Defaults to min_count (i.e. one global threshold) when not given.

    Returns dict: metric -> {'val','lo','hi','num','den'} arrays of length n_bins,
    plus 'edges', 'centers', and 'N_t_sig' (per-bin, for the companion panel).
    """
    if min_bin_count_blend is None:
        min_bin_count_blend = min_count
    edges = np.asarray(edges, float)
    n = len(edges) - 1
    centers = 0.5 * (edges[:-1] + edges[1:])
    tmag = truths[mag_col].to_numpy()
    dmag = dets[mag_col].to_numpy()

    full_pops = _metric_populations(truths, dets, prefix)
    rho_g = {m: estimate_icc(pop['group_id'].to_numpy(), ind.to_numpy())[0]
             for m, (pop, ind) in full_pops.items()}

    out = {m: {'val': np.full(n, np.nan), 'lo': np.full(n, np.nan),
               'hi': np.full(n, np.nan), 'num': np.zeros(n, int),
               'den': np.zeros(n, int)} for m in METRIC_NUMDEN}
    n_t_sig = np.zeros(n, int)

    for b in range(n):
        mask_t = pd.Series((tmag >= edges[b]) & (tmag < edges[b + 1]), index=truths.index)
        mask_d = pd.Series((dmag >= edges[b]) & (dmag < edges[b + 1]), index=dets.index)
        c = compute_counts(truths, dets, prefix, mask_t, mask_d, mag_col)  # asserts 1-3 here
        met = compute_metrics(c)
        n_t_sig[b] = c['N_t_sig']
        pops_b = _metric_populations(truths[mask_t], dets[mask_d], prefix)
        for m, (nk, dk) in METRIC_NUMDEN.items():
            num, den = c[nk], c[dk]
            out[m]['num'][b] = num
            out[m]['den'][b] = den
            thresh = min_bin_count_blend if m in BLEND_COND_METRICS else min_count
            if den >= thresh:  # mask AFTER asserts, never before
                out[m]['val'][b] = met[m]
                pop_b = pops_b[m][0]
                assert len(pop_b) == den, f'{m} bin {b}: population {len(pop_b)} != den {den}'
                mbar_b = mean_cluster_size(pop_b['group_id'].to_numpy())
                out[m]['lo'][b], out[m]['hi'][b] = betabinom_ci(
                    num, den, rho_g[m], mbar_b, conf)
    out['edges'] = edges
    out['centers'] = centers
    out['N_t_sig'] = n_t_sig
    return out

def compute_class_binned(tcls, prefix, edges, min_count=1, conf=0.95, mag_col=MAG_COL):
    """Per-bin class metrics + CIs + per-bin numerator/denominator counts, all
    binned by truth mag_i. Mirrors compute_binned: rho is fit ONCE per metric on
    the full classified population, m_bar per bin on that bin's population.
    Bins with denominator < min_count are NaN'd
    for plotting (counts left untouched).
 
    'den' is exactly the per-bin recall/precision denominator (true-star count
    for star_recall, predicted-star count for star_precision, etc.), which is
    what the companion sub-panel in fig:class_binned draws.
 
    Returns dict: metric -> {'val','lo','hi','num','den'} arrays, plus
    'edges','centers'.
    """
    edges = np.asarray(edges, float)
    n = len(edges) - 1
    centers = 0.5 * (edges[:-1] + edges[1:])
 
    full_pops = _class_populations(tcls, prefix)
    rho_g, gid_g, ind_g, mag_g = {}, {}, {}, {}
    for m, (pop, ind) in full_pops.items():
        gid_g[m] = pop['group_id'].to_numpy()
        rho_g[m], _ = estimate_icc(gid_g[m], ind.to_numpy().astype(float))
        ind_g[m] = ind.to_numpy().astype(bool)
        mag_g[m] = pop[mag_col].to_numpy()          # truth mag of this metric's population
 
    out = {m: {'val': np.full(n, np.nan), 'lo': np.full(n, np.nan),
               'hi': np.full(n, np.nan), 'num': np.zeros(n, int),
               'den': np.zeros(n, int)} for m in CLASS_METRICS}
 
    for m in CLASS_METRICS:
        pm, ind = mag_g[m], ind_g[m]
        for b in range(n):
            in_bin = (pm >= edges[b]) & (pm < edges[b + 1])
            den = int(in_bin.sum())
            num = int((in_bin & ind).sum())
            out[m]['num'][b] = num
            out[m]['den'][b] = den
            if den >= min_count:
                out[m]['val'][b] = _safe_div(num, den)
                out[m]['lo'][b], out[m]['hi'][b] = betabinom_ci(
                    num, den, rho_g[m], mean_cluster_size(gid_g[m][in_bin]), conf)
    out['edges'] = edges
    out['centers'] = centers
    return out

# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------
def _selfcheck():
    rng = np.random.default_rng(0)

    # (a) rho -> 0 limit reproduces Cameron/flat Beta(k+1, n-k+1)
    k, n = 30, 200
    lo_bb, hi_bb = betabinom_ci(k, n, rho=0.0, m_bar=2.0)   # DEFF=1 regardless of m_bar
    lo_ref = float(beta_dist.ppf(0.025, k + 1, n - k + 1))
    hi_ref = float(beta_dist.ppf(0.975, k + 1, n - k + 1))
    assert abs(lo_bb - lo_ref) < 1e-9 and abs(hi_bb - hi_ref) < 1e-9, "rho=0 != Cameron"

    # (b) k=0 -> proper non-degenerate interval, lo near 0 but > 0 (this is
    # the equal-tailed flat-prior Beta(k+1,n-k+1) interval, not Clopper-Pearson,
    # so it never hard-clips to exactly 0/1 -- see betabinom_ci docstring);
    # k=n is the symmetric case.
    lo0, hi0 = betabinom_ci(0, 50, rho=0.5, m_bar=2.0)
    assert 0.0 < lo0 < 1e-3 and 0.0 < hi0 < 1.0, f"k=0 boundary bad ({lo0},{hi0})"
    lon, hin = betabinom_ci(50, 50, rho=0.5, m_bar=2.0)
    assert 1.0 - 1e-3 < hin < 1.0 and lon < 1.0, f"k=n boundary bad ({lon},{hin})"

    # (c) DEFF>1 widens vs independence (same k,n; rho>0, m_bar>1)
    lo_i, hi_i = betabinom_ci(60, 200, rho=0.0, m_bar=2.0)
    lo_c, hi_c = betabinom_ci(60, 200, rho=0.6, m_bar=2.0)
    assert (hi_c - lo_c) > (hi_i - lo_i), "clustering did not widen interval"

    # (d) synthetic rho recovery: known-overdispersed clusters -> clear positive rho
    n_groups, m = 4000, 2
    p_group = rng.beta(2, 2, n_groups)          # cluster-level prob -> overdispersion
    gids, ys = [], []
    for g in range(n_groups):
        gids += [g] * m
        ys += list((rng.random(m) < p_group[g]).astype(float))
    rho_hat, _ = estimate_icc(np.array(gids), np.array(ys, float))
    assert rho_hat > 0.15, f"ICC recovery too low: {rho_hat}"

    # (e) all-singletons -> rho EXACTLY 0, m_bar == 1 (no hedge); rho_raw is also
    # exactly 0 here (not just floored), since singletons carry no clustering info
    sgids = np.arange(500)
    sind = (rng.random(500) < 0.1).astype(float)
    assert mean_cluster_size(sgids) == 1.0
    # size-weighted m_bar: groups of 1 and 3 -> (1 + 9) / 4 = 2.5, not the simple mean 2
    assert mean_cluster_size([0, 1, 1, 1]) == 2.5
    rho_s, rho_s_raw = estimate_icc(sgids, sind)
    assert rho_s == 0.0 and rho_s_raw == 0.0, "singletons must give rho=0 exactly"

    # (f) MIXED singleton+pair, independent -> rho ~ 0  (the test that catches the bug)
    g_ids, y = [], []
    for i in range(900):                        # 900 singleton groups
        g_ids.append(i); y.append(float(rng.random() < 0.5))
    for j in range(100):                        # 100 independent pairs
        g_ids += [900 + j, 900 + j]
        y += [float(rng.random() < 0.5), float(rng.random() < 0.5)]
    rho_mix, _ = estimate_icc(np.array(g_ids), np.array(y, float))
    assert rho_mix < 0.05, f"mixed independent population gave spurious rho={rho_mix}"

    # (h) anti-correlation is exposed via rho_raw, not silently floored away:
    # within each pair, exactly one success (perfect negative within-cluster
    # correlation) -> rho_raw << 0 but the clamped rho used for DEFF is 0
    ag_ids, ay = [], []
    for g in range(500):
        ag_ids += [g, g]
        first = rng.random() < 0.5
        ay += [1.0 if first else 0.0, 0.0 if first else 1.0]
    rho_anti, rho_anti_raw = estimate_icc(np.array(ag_ids), np.array(ay, float))
    assert rho_anti == 0.0, f"anti-correlated rho must be floored to 0, got {rho_anti}"
    assert rho_anti_raw < -0.5, f"anti-correlated rho_raw should be strongly negative, got {rho_anti_raw}"

    # (g) F1 interval: monotone endpoint propagation, conservative, zero-guard
    assert f1_interval(0.90, 0.92, 0.97, 0.98) == (_f1(0.90, 0.97), _f1(0.92, 0.98))
    assert f1_interval(0.0, 0.5, 0.0, 0.5)[0] == 0.0, "zero-corner F1 must be 0, not NaN"

    print("beta-binomial CI self-check passed")

def _class_selfcheck():
    """Confusion-table fixture for the classification block. Mirrors _selfcheck:
    a hand-built table with known cells, exercised through the same functions the
    scripts call. Covers all three catalog encodings, the buffered-match semantics
    (a truth matched to a past-cut detection is still classified), the NaN-class
    exclusion feeding n_matched_uncls, the drift guard between _class_populations
    and compute_class_counts, scalar/binned agreement, and the strict-map guard.

    DD prefix throughout (encodings: DD det 0=gal/1=star, truth 1=gal/2=star).
    """
    def T(cls, mag, gid, matched, mid):
        return dict(catalog_key=TRUTH_KEY, row_index=-1, **{
            CLASS_COL: cls, MAG_COL: mag, 'group_id': gid,
            'dd_is_matched': matched, 'dd_matched_id': mid})

    def D(rowidx, cls):
        return dict(catalog_key=DET_KEYS['dd'], row_index=rowidx, **{
            CLASS_COL: cls, MAG_COL: 20.0, 'group_id': 0})

    # Known within-cut confusion (m_cut = 25.3, truth mag): gg=3, gs=1, sg=1, ss=2.
    # Plus one matched-to-NaN-class (n_matched_uncls) and one unmatched truth.
    truth_rows = [
        T(1, 21.0, 10, True, 100),   # gal -> det gal   gg
        T(1, 21.2, 10, True, 101),   # gal -> det gal   gg
        T(1, 24.9, 11, True, 102),   # gal -> det star  gs
        T(1, 25.1, 11, True, 103),   # gal -> det gal   gg  (det past cut, still classified)
        T(2, 20.5, 12, True, 104),   # star -> det gal  sg
        T(2, 22.0, 12, True, 105),   # star -> det star ss
        T(2, 23.0, 13, True, 106),   # star -> det star ss
        T(1, 22.5, 14, True, 107),   # matched, det class NaN -> unclassified
        T(2, 26.0, 15, False, -1),   # unmatched -> not classified (and past cut)
    ]
    det_rows = [D(100, 0), D(101, 0), D(102, 1), D(103, 0),
                D(104, 0), D(105, 1), D(106, 1), D(107, np.nan)]
    df = pd.DataFrame(truth_rows + det_rows)
    truths, dets = split_truth_det(df, 'dd')

    tcls = attach_pred_class(truths, dets, 'dd')
    m_cut = 25.3
    mask = tcls[MAG_COL] < m_cut

    # (a) confusion cells + partition + the caption's exclusion count
    c = compute_class_counts(tcls, 'dd', mask)
    assert (c['N_gg'], c['N_gs'], c['N_sg'], c['N_ss']) == (3, 1, 1, 2), c
    assert c['N_t_cls'] == 7
    assert c['n_matched_uncls'] == 1          # the 22.5 matched-to-NaN, within cut

    # (b) ratios match tab:class_metrics
    met = compute_class_metrics(c)
    assert abs(met['galaxy_recall']    - 3 / 4) < 1e-12   # gg/(gg+gs)
    assert abs(met['galaxy_precision'] - 3 / 4) < 1e-12   # gg/(gg+sg)
    assert abs(met['star_recall']      - 2 / 3) < 1e-12   # ss/(ss+sg)
    assert abs(met['star_precision']   - 2 / 3) < 1e-12   # ss/(ss+gs)

    # (c) drift guard: _class_populations reproduces compute_class_counts exactly
    cut = tcls[tcls[MAG_COL] < m_cut]
    pops = _class_populations(cut, 'dd')
    for name, (num_cell, den_cells) in CLASS_METRIC_CELLS.items():
        pop, ind = pops[name]
        assert len(pop) == sum(c[x] for x in den_cells), (name, len(pop))
        assert int(ind.sum()) == c[num_cell], (name, int(ind.sum()))

    # (d) scalar CIs bracket the point estimate (or the class is too small to assert)
    cis = class_scalar_cis(tcls, 'dd', m_cut)
    for k in CLASS_METRICS:
        den = sum(c[x] for x in CLASS_METRIC_CELLS[k][1])
        assert (cis[k]['lo'] <= met[k] <= cis[k]['hi']) or den < 5, (k, cis[k], met[k])

    # (e) binned totals over the full range == the no-cut confusion (scalar == aggregate)
    edges = np.arange(MIN_MAG_BIN, MAX_MAG_BIN + 0.01, 0.5)
    b = compute_class_binned(tcls, 'dd', edges, min_count=1)
    full = compute_class_counts(tcls, 'dd', pd.Series(True, index=tcls.index))
    for k in CLASS_METRICS:
        num_cell, den_cells = CLASS_METRIC_CELLS[k]
        assert int(b[k]['num'].sum()) == full[num_cell], k
        assert int(b[k]['den'].sum()) == sum(full[x] for x in den_cells), k

    # (f) strict-map guard: an unexpected truth code (e.g. SNe truth_type=3) must
    # raise, not silently drop
    bad = truths.copy()
    bad.loc[bad.index[0], CLASS_COL] = 3
    try:
        attach_pred_class(bad, dets, 'dd')
        raise SystemExit('FAIL: unmapped class code did not raise')
    except ValueError as e:
        assert 'not in CLASS_MAPS' in str(e)

    # (g) the det encodings really are inverted between DD and LSST
    s = pd.Series([0, 1, np.nan])
    assert list(_map_class(s, 'lsst_det').fillna('N')) == ['s', 'g', 'N']
    assert list(_map_class(s, 'dd_det').fillna('N')) == ['g', 's', 'N']

    print('classification self-check OK')


if __name__ == '__main__':
    def _row(cat, mag, p='dd', **flags):
        base = {f'{p}_{f}': False for f in _REQUIRED_FLAGS}
        base.update({'catalog_key': cat, MAG_COL: mag, 'group_id': flags.pop('g', 0)})
        base.update({f'{p}_{k}': v for k, v in flags.items()})
        return base

    T, D = 'lsst_truth', 'dd_det'
    rows = [
        # isolated match
        _row(T, 22.0, is_matched=True, g=1),
        _row(D, 22.0, is_matched=True, g=1),
        # missed truth, spurious det
        _row(T, 23.0, is_missed=True, g=2),
        _row(D, 24.0, is_spurious=True, g=3),
        # resolved significant blend (2 components, both matched)
        _row(T, 21.0, is_matched=True, is_part_of_blend=True, is_resolved_blend=True, is_sig_blend=True, g=4),
        _row(T, 21.5, is_matched=True, is_part_of_blend=True, is_resolved_blend=True, is_sig_blend=True, g=4),
        _row(D, 21.0, is_matched=True, is_part_of_blend=True, is_resolved_blend=True, is_sig_blend=True, g=4),
        _row(D, 21.5, is_matched=True, is_part_of_blend=True, is_resolved_blend=True, is_sig_blend=True, g=4),
        # unrecognized significant blend: survivor + absorbed + absorber
        _row(T, 20.0, is_matched=True, is_part_of_blend=True, is_survivor=True, is_sig_blend=True, g=5),
        _row(T, 20.5, is_absorbed=True, is_part_of_blend=True, is_sig_blend=True, g=5),
        _row(D, 20.0, is_matched=True, is_part_of_blend=True, is_absorber=True, is_sig_blend=True, g=5),
        # shredded truth + shred det
        _row(T, 19.0, is_matched=True, is_shredded=True, g=6),
        _row(D, 19.0, is_matched=True, g=6),
        _row(D, 19.1, is_shred=True, g=6),
        # NaN-mag det (excluded everywhere, still counted for reporting)
        _row(D, np.nan, is_matched=True, g=7),
        _row(T, 19.5, is_matched=True, g=7),
    ]
    df = pd.DataFrame(rows)
    truths, dets = split_truth_det(df, 'dd')
    assert_maximality(truths, dets, 'dd')
    m_cut = 25.3
    mt = pd.Series(truths[MAG_COL] < m_cut, index=truths.index)
    md = pd.Series(dets[MAG_COL] < m_cut, index=dets.index)
    c = compute_counts(truths, dets, 'dd', mt, md)
    met = compute_metrics(c)

    # _metric_populations must reproduce compute_counts' own numerator/denominator
    # counts exactly -- this is the drift guard for having two independent
    # definitions of "what population does metric X use" (compute_counts' inline
    # sums, kept untouched for zero risk; _metric_populations for the beta-binomial
    # CIs). If this ever fails, the resolved/survivor/absorbed-implies-
    # part_of_blend invariant _metric_populations relies on has been violated.
    pops = _metric_populations(truths[mt], dets[md], 'dd')
    for name, (num_key, den_key) in METRIC_NUMDEN.items():
        pop, ind = pops[name]
        assert int(ind.sum()) == c[num_key], (
            f'{name}: population numerator {int(ind.sum())} != compute_counts {num_key}={c[num_key]}')
        assert len(pop) == c[den_key], (
            f'{name}: population size {len(pop)} != compute_counts {den_key}={c[den_key]}')

    assert c['N_t_sig'] == 4 and c['N_t_res'] == 2 and c['N_t_surv'] == 1 and c['N_t_abs'] == 1
    assert c['N_t_abs_all'] == 1 and c['N_d_abr'] == 1
    assert c['n_det_nan_mag'] == 1
    assert abs(met['resolved_frac'] + met['survivor_frac'] + met['absorbed_frac'] - 1.0) < 1e-9
    assert c['N_d'] == 7  # 8 det rows, 1 NaN excluded

    cis = scalar_cis(truths, dets, 'dd', m_cut)
    assert cis['completeness']['lo'] <= met['completeness'] <= cis['completeness']['hi'] or c['N_t'] < 5

    binned = compute_binned(truths, dets, 'dd', np.arange(MIN_MAG_BIN, MAX_MAG_BIN + 0.01, 0.5), min_count=1)
    assert np.nanmax(binned['completeness']['val']) <= 1.0

    # min_bin_count_blend applies only to BLEND_COND_METRICS: at mag [20.0, 20.5) this
    # fixture has N_t == N_t_sig == 1, so min_count=1 keeps completeness but
    # min_bin_count_blend=2 must NaN the blend-conditional metrics in the same bin.
    edges = np.arange(MIN_MAG_BIN, MAX_MAG_BIN + 0.01, 0.5)
    b_idx = np.searchsorted(edges, 20.0, side='right') - 1
    split = compute_binned(truths, dets, 'dd', edges, min_count=1, min_bin_count_blend=2)
    assert not np.isnan(split['completeness']['val'][b_idx])
    assert np.isnan(split['resolved_frac']['val'][b_idx])

    # corrupting a flag must trip assert 1
    bad = df.copy()
    bad.loc[bad.index[8], 'dd_is_survivor'] = False  # drop a survivor -> res+surv+abs != sig
    bt, bd = split_truth_det(bad, 'dd')
    try:
        compute_counts(bt, bd, 'dd',
                       pd.Series(bt[MAG_COL] < m_cut, index=bt.index),
                       pd.Series(bd[MAG_COL] < m_cut, index=bd.index))
        raise SystemExit('FAIL: corrupt survivor did not trip assert 1')
    except AssertionError as e:
        assert 'blend partition broken' in str(e)

    print('blend_metrics self-check OK')

    _selfcheck()
    _class_selfcheck()