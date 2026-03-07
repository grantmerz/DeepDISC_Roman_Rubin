"""
fof_classify.py — FOF catalog matching and Stage 1/2 classification module

Extracted from fof_and_classify.ipynb for reuse in grid search and other scripts

Core pipeline:
    1. find_matches()          — FOF spatial matching between det + truth catalogs
    2. build_table()           — join FOF results back to source catalog columns
    3. get_stage1_counts()     — pivot group counts by catalog_key
    4. classify_stg1()         — label groups by count-based rules
    5. build_bipartite_and_match() — Hungarian 1-to-1 matching within groups
    6. classify_group_stage2() — per-group flag assignment
    7. classify_stage2()       — orchestrate stage 2 across all groups

Utilities:
    - stage2_summary()         — print human-readable summary

"""

import numpy as np
import pandas as pd
import FoFCatalogMatching
from scipy.optimize import linear_sum_assignment
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table


# ============================================================================
# Constants
# ============================================================================

ALL_FLAG_COLS = [
    'is_matched', 'sep_dist', 'matched_id',
    'is_spurious', 'is_shred', 'is_shredded',
    'is_missed', 'is_blended_away', 'is_unrec_blend', 'is_part_of_blend',
    'is_resolved_blend', 'is_partial_deblend'
]

# ============================================================================
# Stage 0: FOF Matching
# ============================================================================

def find_matches(cats, linking_length=1.0, max_friends=None):
    """
    Find matches between catalogs using the Friends-of-Friends algorithm.

    Parameters
    ----------
    cats : dict of {str: pd.DataFrame or astropy.table.Table}
        Named catalogs to match. Each must have 'ra' and 'dec' columns.
    linking_length : float, optional
        Linking length in arcseconds. Default is 1.0.
    max_friends : int or None, optional
        Maximum number of friends allowed in a group.
        Use None for no limit. Default is None.

    Returns
    -------
    astropy.table.Table
        FOF matched catalog with group IDs, catalog keys, and row indices.
    """
    print(f"Processing {len(cats)} catalogs:", cats.keys())
    cat_tables = {}
    for name, cat in cats.items():
        if isinstance(cat, pd.DataFrame):
            cat_tables[name] = Table.from_pandas(cat[['ra', 'dec']])
        else:
            cat_tables[name] = cat
    linking_lengths = {linking_length: max_friends}
    return FoFCatalogMatching.match(cat_tables, linking_lengths)


def build_table(results_df, cats, col_config):
    """
    Build a long-format table by joining requested columns from source catalogs.

    Parameters
    ----------
    results_df : pd.DataFrame
        Matched results with columns: row_index, catalog_key, group_id.
    cats : dict of {str: pd.DataFrame}
        Source catalogs keyed by catalog name.
    col_config : dict
        Column mapping per catalog.
        Format: {'catalog_key': {'orig_col_name': 'new_col_name', ...}}

    Returns
    -------
    pd.DataFrame
        Long-format table with columns from all catalogs merged and renamed.
    """
    chunks = []
    for cat_key in results_df['catalog_key'].unique():
        if cat_key not in cats:
            print(f"Didn't recognize catalog key: {cat_key}, skipping...")
            continue
        subset = results_df[results_df['catalog_key'] == cat_key].copy()
        # # FoFCatalogMatching can return row_index as object dtype;
        # # coerce to int so the merge key matches the source catalog's index
        # subset['row_index'] = subset['row_index'].astype(int)
        source_cat = cats[cat_key]
        config = col_config.get(cat_key, {})
        # print(f"Processing {len(subset)} entries for catalog key: {cat_key}")
        # print(f"Config columns: {config}")
        cols_to_fetch = list(config.keys())
        if not cols_to_fetch:
            chunks.append(subset)
            continue
        # now we merge which replaces idxing into the catalog manually
        merged = subset.merge(
            source_cat[cols_to_fetch],
            left_on='row_index',
            right_index=True,
            how='left'
        )
        # rename cols so they align (e.g. objectId -> id)
        merged.rename(columns=config, inplace=True)
        chunks.append(merged)
    final_df = pd.concat(chunks, axis=0).reset_index(drop=True)
    return final_df


# ============================================================================
# Stage 1: Count-Based Classification
# ============================================================================

def get_stage1_counts(analysis, expected_keys=None, rename_map=None):
    """
    Pivot FOF group membership counts by catalog key.

    Parameters
    ----------
    analysis : pd.DataFrame
        Long-format analysis table with 'group_id' and 'catalog_key'.
    expected_keys : list of str, optional
        Catalog keys to ensure exist as columns (filled with 0 if absent).
    rename_map : dict, optional
        Rename catalog key columns, e.g. {'lsst_truth': 'n_truth'}.

    Returns
    -------
    pd.DataFrame
        Indexed by group_id, with one column per catalog key (renamed).
    """
    counts = (
        analysis.groupby(['group_id', 'catalog_key'])
        .size()
        .unstack(fill_value=0)
    )
    # ensure all cols exist (in case a batch has 0 detections total)
    if expected_keys is None:
        expected_keys = ['dd_det', 'lsst_det', 'lsst_truth']
    for col in expected_keys:
        if col not in counts.columns:
            counts[col] = 0
    if rename_map is None:
        rename_map = {
            'lsst_truth': 'n_truth',
            'lsst_det': 'n_lsst',
            'dd_det': 'n_dd'
        }
    counts.rename(columns=rename_map, inplace=True)
    return counts


def classify_stg1(counts, prefix, det_col_name, truth_col_name='n_truth'):
    """
    Apply Stage 1 classification rules based on detection/truth counts per group.

    Parameters
    ----------
    counts : pd.DataFrame
        Output of get_stage1_counts, indexed by group_id.
    prefix : str
        Pipeline prefix for the label column (e.g. 'dd' or 'lsst').
    det_col_name : str
        Column name for detection counts (e.g. 'n_dd' or 'n_lsst').
    truth_col_name : str
        Column name for truth counts. Default 'n_truth'.

    Returns
    -------
    pd.DataFrame
        Input with new column '{prefix}_stg1' containing stage 1 labels.
    """
    label_col = f"{prefix}_stg1"
    n_t = counts[truth_col_name]
    n_d = counts[det_col_name]

    conditions = [
        (n_t == 1) & (n_d == 1),       # 1-to-1: isolated match
        (n_t > 1) & (n_d == n_t),       # N-to-N: candidate resolved blend
        (n_t > n_d) & (n_d == 0),       # N-to-0: missed (FN)
        (n_d > n_t) & (n_t == 0),       # 0-to-N: spurious (FP)
        (n_t > n_d) & (n_d == 1),       # N-to-1: unrecognized blend
        (n_d > n_t) & (n_t == 1),       # 1-to-N: shredded
        (n_t > n_d) & (n_d > 1),        # N-to-M where N > M > 1
        (n_d > n_t) & (n_t > 1)         # N-to-M where M > N > 1
    ]
    classes = [
        'isolated_match',
        'candidate_res_blend',
        'missed_false_neg',
        'spurious_false_pos',
        'unrec_blend',
        'shredded',
        'complex_merge',
        'complex_shred'
    ]

    counts[label_col] = np.select(conditions, classes, default='n/a')
    return counts


# ============================================================================
# Bipartite Matching (Hungarian Algorithm)
# ============================================================================
def build_bipartite_and_match(sep_matrix, match_rad):
    """
    Build adjacency from sep matrix and run optimal 1-to-1 matching

    Parameters
    ----------
    sep_matrix : np.ndarray, shape (n_det, n_truth)
        Pairwise separations in arcseconds. Here's an example with 3 detections and 3 truths:
                truth_0   truth_1   truth_2
        det_0 [  0.12"    0.87"     1.20"  ]
        det_1 [  0.95"    0.23"     0.44"  ]
        det_2 [  1.10"    0.67"     0.08"  ]
    match_rad : float
        Maximum separation for a valid match, in arcseconds.

    Returns
    -------
    matched_pairs : list of (det_local_idx, truth_local_idx, sep_dist)
    adj : np.ndarray (n_det x n_truth), binary (1 if within radius)
    """
    # Build binary adjacency matrix: entry is 1 if the detection-truth pair
    # falls within the matching radius, 0 otherwise. This defines which edges
    # exist in the bipartite graph. int8 keeps memory low for large groups.
    adj = (sep_matrix <= match_rad).astype(np.int8)
    # no edges exist at all --> no matching is possible
    # Return empty pairs and the all-zero adjacency so we can
    # immediately flag all dets as spurious and all truths as missed
    if not adj.any():
        return [], adj
    
    # Build the cost matrix for the Hungarian algorithm.
    # Valid pairs (within radius) use their actual separation as cost,
    # so the optimizer naturally prefers closer matches.
    # Invalid pairs (outside radius) get a large sentinel value (1e9)
    # instead of np.inf, because linear_sum_assignment requires finite values.
    # The sentinel is large enough that the optimizer will never choose an
    # out-of-radius pair when a valid one is available.
    SENTINEL = 1e9
    cost_matrix = np.where(adj, sep_matrix, SENTINEL)
    
    # Run Hungarian algo with scipy's linear_sum_assignment
    # Returns det_idxs and truth_idxs as parallel arrays of row/col idxs
    # forming the optimal assignment. The algorithm minimizes total cost,
    # so it finds the globally best 1-to-1 pairing — not greedy nearest-neighbor.
    # and it can handle non-square matrices, so n_det != n_truth is a non-issue
    det_idxs, truth_idxs = linear_sum_assignment(cost_matrix)
    # get seps for each assigned pair
    seps = sep_matrix[det_idxs, truth_idxs]
    # filter out sentinel assignments: the Hungarian algorithm may have been
    # forced to assign a det to an out-of-radius truth (when n_det > n_truth,
    # or when no valid truth was available). Discard any pair whose separation
    # exceeds the match radius bc these aren't real matches
    valid = seps <= match_rad
    
    # results is gonna be a list of (det_local_idx, truth_local_idx, sep_arcsec)
    # Indices are local to this group (0-based), not global df pos
    # We'll be responsible for mapping back to global iloc/row_index
    matched_pairs = [
        (int(di), int(ti), float(s))
        for di, ti, s in zip(det_idxs[valid], truth_idxs[valid], seps[valid])
    ]
    return matched_pairs, adj


# ============================================================================
# Stage 2: Per-Object Classification
# ============================================================================

def classify_group_stage2(n_det, n_truth, matched_pairs, adj):
    """
    Classify all objects in one FOF group based on bipartite matching results.

    Group-level blend flags are derived from the adjacency graph,
    NOT from FOF group membership. A truth with zero edges is simply
    "missed" and does not make nearby detections into "unrec blends".

    Parameters
    ----------
    n_det, n_truth : int
        Number of detections and truth objects in this group.
    matched_pairs : list of (det_local_idx, truth_local_idx, sep_dist)
    adj : np.ndarray (n_det x n_truth), binary

    Returns
    -------
    det_r : dict of numpy arrays (length n_det)
    truth_r : dict of numpy arrays (length n_truth)
    """
    det_r = {
        'is_matched': np.zeros(n_det, dtype=bool),
        'matched_id_local': np.full(n_det, -1, dtype=int),
        'sep_dist': np.full(n_det, np.nan),
        'is_spurious': np.zeros(n_det, dtype=bool),
        'is_shred': np.zeros(n_det, dtype=bool),
        'is_unrec_blend': np.zeros(n_det, dtype=bool),
        'is_part_of_blend': np.zeros(n_det, dtype=bool),
        'is_resolved_blend': np.zeros(n_det, dtype=bool),
        'is_partial_deblend': np.zeros(n_det, dtype=bool),
    }
    truth_r = {
        'is_matched': np.zeros(n_truth, dtype=bool),
        'matched_id_local': np.full(n_truth, -1, dtype=int),
        'sep_dist': np.full(n_truth, np.nan),
        'is_missed': np.zeros(n_truth, dtype=bool),
        'is_blended_away': np.zeros(n_truth, dtype=bool),
        'is_shredded': np.zeros(n_truth, dtype=bool),
        'is_unrec_blend': np.zeros(n_truth, dtype=bool),
        'is_part_of_blend': np.zeros(n_truth, dtype=bool),
        'is_resolved_blend': np.zeros(n_truth, dtype=bool),
        'is_partial_deblend': np.zeros(n_truth, dtype=bool),
    }

    # --- Record matched pairs ---
    matched_det_set = set()
    matched_truth_set = set()
    for di, ti, sep in matched_pairs:
        det_r['is_matched'][di] = True
        det_r['matched_id_local'][di] = ti
        det_r['sep_dist'][di] = sep
        truth_r['is_matched'][ti] = True
        truth_r['matched_id_local'][ti] = di
        truth_r['sep_dist'][ti] = sep
        matched_det_set.add(di)
        matched_truth_set.add(ti)

    n_matched = len(matched_pairs)

    # --- Per-object: unmatched detections ---
    for di in range(n_det):
        if di not in matched_det_set:
            if adj[di, :].any():
                det_r['is_shred'][di] = True
            else:
                det_r['is_spurious'][di] = True

    # --- Per-object: truth objects ---
    # For is_shredded: only count edges from UNMATCHED dets.
    # A matched det assigned to a different truth doesn't indicate shredding.
    unmatched_det_mask = np.ones(n_det, dtype=bool)
    for di in matched_det_set:
        unmatched_det_mask[di] = False
    unmatched_adj = adj[unmatched_det_mask, :] if unmatched_det_mask.any() else np.zeros((0, n_truth), dtype=np.int8)
    unmatched_edge_counts = unmatched_adj.sum(axis=0)
    edge_counts_per_truth = adj.sum(axis=0)

    for ti in range(n_truth):
        total_edges = int(edge_counts_per_truth[ti])
        unmatched_edges = int(unmatched_edge_counts[ti]) if len(unmatched_adj) > 0 else 0
        if ti not in matched_truth_set:
            if total_edges == 0: # if this unmatched truth has zero edges to any det, it's simply missed
                truth_r['is_missed'][ti] = True
            if unmatched_edges >= 1: # if this unmatched truth has 1+ edges from unmatched dets, it's shredded
                truth_r['is_shredded'][ti] = True
        else:
            # Matched truth is shredded if 1+ unmatched dets also pile on it
            if unmatched_edges >= 1:
                truth_r['is_shredded'][ti] = True

    # --- Group-level blend flags (adj-based) ---
    # connected_truths: truths with at least one detection edge within match_rad
    connected_truths = set(
        ti for ti in range(n_truth) if edge_counts_per_truth[ti] > 0
    )
    unmatched_connected_truths = connected_truths - matched_truth_set
    n_connected = len(connected_truths)
    n_unmatched_connected = len(unmatched_connected_truths)

    # A "real blend" requires 2+ truths with detection edges
    is_real_blend = (n_connected >= 2)
    fully_resolved = is_real_blend and (n_unmatched_connected == 0) and (n_matched == n_connected)
    
    has_unrec_blend = is_real_blend and (n_unmatched_connected > 0)
    has_partial_deblend = has_unrec_blend and (n_matched > 0)

    connected_dets = [di for di in range(n_det) if adj[di, :].any()] if is_real_blend else []

    # is_part_of_blend: broadest blend flag, all blend scenarios
    if is_real_blend:
        for di in connected_dets:
            det_r['is_part_of_blend'][di] = True
        for ti in connected_truths:
            truth_r['is_part_of_blend'][ti] = True

    # Fully resolved: deblender correctly separated every source
    if fully_resolved:
        for di in matched_det_set:
            det_r['is_resolved_blend'][di] = True
        for ti in connected_truths:
            truth_r['is_resolved_blend'][ti] = True

    # Unrecognized blend: at least one truth was absorbed
    if has_unrec_blend:
        for di in connected_dets:
            det_r['is_unrec_blend'][di] = True
        for ti in connected_truths:
            truth_r['is_unrec_blend'][ti] = True
        for ti in unmatched_connected_truths:
            truth_r['is_blended_away'][ti] = True

    # Partial deblend: strict subset of unrec_blend where some progress was made
    if has_partial_deblend:
        for di in connected_dets:
            det_r['is_partial_deblend'][di] = True
        for ti in connected_truths:
            truth_r['is_partial_deblend'][ti] = True

    return det_r, truth_r


def classify_stage2(analysis_df, counts_df, det_prefix,
                    match_rad, verbose=True):
    """
    Run Stage 2 classification on all groups for one detection pipeline.

    Parameters
    ----------
    analysis_df : pd.DataFrame
        Long-format analysis table with columns:
        group_id, catalog_key, row_index, ra, dec.
    counts_df : pd.DataFrame
        Stage 1 counts table, indexed by group_id. Must contain
        the '{det_prefix}_stg1' label column.
    det_prefix : str
        Detection pipeline identifier: 'dd' or 'lsst'.
    match_rad : float
        Matching radius in arcseconds.
    verbose : bool
        Print progress updates.

    Returns
    -------
    analysis_df : pd.DataFrame
        Input dataframe with new columns: '{det_prefix}_{flag}' for each flag.
        matched_id is int64 (-1 = no match). For dets it refers to a truth
        row_index; for truths it refers to a det row_index.
    """
    det_cat_key = 'dd_det' if det_prefix == 'dd' else 'lsst_det'
    truth_cat_key = 'lsst_truth'
    stg1_col = f"{det_prefix}_stg1"
    prefix = det_prefix
    N = len(analysis_df)

    # 1. Initialize numpy flag arrays
    flags = {}
    for col in ALL_FLAG_COLS:
        if col == 'sep_dist':
            flags[col] = np.full(N, np.nan, dtype=np.float64)
        elif col == 'matched_id':
            flags[col] = np.full(N, -1, dtype=np.int64)
        else:
            flags[col] = np.zeros(N, dtype=bool)

    # 2. Build group --> iloc index (single O(N) pass)
    if verbose:
        print(f"[Stage 2 {prefix.upper()}] Building group index for {N:,} rows...")
    group_id_arr = analysis_df['group_id'].values
    cat_key_arr = analysis_df['catalog_key'].values
    row_idx_arr = analysis_df['row_index'].values
    ra_arr = analysis_df['ra'].values.astype(np.float64)
    dec_arr = analysis_df['dec'].values.astype(np.float64)

    group_index = {}
    for iloc_pos in range(N):
        gid = group_id_arr[iloc_pos]
        ckey = cat_key_arr[iloc_pos]
        if gid not in group_index:
            group_index[gid] = {
                'det_iloc': [], 'det_rowid': [], 'det_ra': [], 'det_dec': [],
                'truth_iloc': [], 'truth_rowid': [], 'truth_ra': [], 'truth_dec': [],
            }
        gi = group_index[gid]
        if ckey == det_cat_key:
            gi['det_iloc'].append(iloc_pos)
            gi['det_rowid'].append(row_idx_arr[iloc_pos])
            gi['det_ra'].append(ra_arr[iloc_pos])
            gi['det_dec'].append(dec_arr[iloc_pos])
        elif ckey == truth_cat_key:
            gi['truth_iloc'].append(iloc_pos)
            gi['truth_rowid'].append(row_idx_arr[iloc_pos])
            gi['truth_ra'].append(ra_arr[iloc_pos])
            gi['truth_dec'].append(dec_arr[iloc_pos])

    if verbose:
        print(f"[Stage 2 {prefix.upper()}] Indexed {len(group_index):,} groups.")

    # 3. Bulk: missed & spurious groups (no coordinate math needed)
    stg1_labels = counts_df[stg1_col]
    missed_gids = set(stg1_labels[stg1_labels == 'missed_false_neg'].index)
    spurious_gids = set(stg1_labels[stg1_labels == 'spurious_false_pos'].index)

    n_missed_objs = 0
    n_spurious_objs = 0
    for gid in missed_gids:
        if gid in group_index:
            # for each truth in this group, mark it as missed
            for ip in group_index[gid]['truth_iloc']:
                flags['is_missed'][ip] = True
            n_missed_objs += len(group_index[gid]['truth_iloc'])

    for gid in spurious_gids:
        if gid in group_index:
            # for each det in this group, mark it as spurious
            for ip in group_index[gid]['det_iloc']:
                flags['is_spurious'][ip] = True
            n_spurious_objs += len(group_index[gid]['det_iloc'])

    if verbose:
        print(f"[Stage 2 {prefix.upper()}] Bulk: {len(missed_gids):,} missed groups "
              f"({n_missed_objs:,} truths), {len(spurious_gids):,} spurious groups "
              f"({n_spurious_objs:,} dets)")

    # 4. Batch isolated matches (single SkyCoord call)
    isolated_gids = stg1_labels[stg1_labels == 'isolated_match'].index
    n_isolated = 0

    if len(isolated_gids) > 0:
        iso_det_ilocs = []
        iso_truth_ilocs = []
        iso_det_ra = []
        iso_det_dec = []
        iso_truth_ra = []
        iso_truth_dec = []
        iso_det_rowids = []
        iso_truth_rowids = []

        for gid in isolated_gids:
            if gid not in group_index:
                continue
            gi = group_index[gid]
            if len(gi['det_iloc']) == 0 or len(gi['truth_iloc']) == 0:
                continue
            iso_det_ilocs.append(gi['det_iloc'][0])
            iso_truth_ilocs.append(gi['truth_iloc'][0])
            iso_det_ra.append(gi['det_ra'][0])
            iso_det_dec.append(gi['det_dec'][0])
            iso_truth_ra.append(gi['truth_ra'][0])
            iso_truth_dec.append(gi['truth_dec'][0])
            iso_det_rowids.append(gi['det_rowid'][0])
            iso_truth_rowids.append(gi['truth_rowid'][0])

        n_isolated = len(iso_det_ilocs)

        if n_isolated > 0:
            det_coords = SkyCoord(ra=iso_det_ra, dec=iso_det_dec, unit=u.deg)
            truth_coords = SkyCoord(ra=iso_truth_ra, dec=iso_truth_dec, unit=u.deg)
            seps = det_coords.separation(truth_coords).arcsec

            iso_det_ilocs = np.array(iso_det_ilocs)
            iso_truth_ilocs = np.array(iso_truth_ilocs)
            iso_det_rowids = np.array(iso_det_rowids, dtype=np.int64)
            iso_truth_rowids = np.array(iso_truth_rowids, dtype=np.int64)

            matched = seps <= match_rad
            unmatched = ~matched

            flags['is_matched'][iso_det_ilocs[matched]] = True
            flags['matched_id'][iso_det_ilocs[matched]] = iso_truth_rowids[matched]
            flags['sep_dist'][iso_det_ilocs[matched]] = seps[matched]
            flags['is_matched'][iso_truth_ilocs[matched]] = True
            flags['matched_id'][iso_truth_ilocs[matched]] = iso_det_rowids[matched]
            flags['sep_dist'][iso_truth_ilocs[matched]] = seps[matched]

            flags['is_spurious'][iso_det_ilocs[unmatched]] = True
            flags['is_missed'][iso_truth_ilocs[unmatched]] = True

    if verbose:
        print(f"[Stage 2 {prefix.upper()}] Batched {n_isolated:,} isolated matches.")

    # 5. Complex groups: batch SkyCoord, then match per group
    skip_labels = {'missed_false_neg', 'spurious_false_pos', 'isolated_match'}
    complex_gids = stg1_labels[~stg1_labels.isin(skip_labels)].index
    n_complex = 0

    if verbose:
        print(f"[Stage 2 {prefix.upper()}] Processing {len(complex_gids):,} complex groups...")

    if len(complex_gids) > 0:
        all_det_ra_tiles = []
        all_det_dec_tiles = []
        all_truth_ra_tiles = []
        all_truth_dec_tiles = []
        group_slices = []
        total_pairs = 0

        for gid in complex_gids:
            if gid not in group_index:
                continue
            gi = group_index[gid]
            n_d = len(gi['det_iloc'])
            n_t = len(gi['truth_iloc'])
            if n_d == 0 or n_t == 0:
                continue
            det_ra = np.array(gi['det_ra'])
            det_dec = np.array(gi['det_dec'])
            truth_ra = np.array(gi['truth_ra'])
            truth_dec = np.array(gi['truth_dec'])

            all_det_ra_tiles.append(np.repeat(det_ra, n_t))
            all_det_dec_tiles.append(np.repeat(det_dec, n_t))
            all_truth_ra_tiles.append(np.tile(truth_ra, n_d))
            all_truth_dec_tiles.append(np.tile(truth_dec, n_d))
            group_slices.append((gid, total_pairs, n_d, n_t))
            total_pairs += n_d * n_t

        if total_pairs > 0:
            big_det = SkyCoord(
                ra=np.concatenate(all_det_ra_tiles),
                dec=np.concatenate(all_det_dec_tiles),
                unit=u.deg
            )
            big_truth = SkyCoord(
                ra=np.concatenate(all_truth_ra_tiles),
                dec=np.concatenate(all_truth_dec_tiles),
                unit=u.deg
            )
            all_seps = big_det.separation(big_truth).arcsec
            del all_det_ra_tiles, all_det_dec_tiles
            del all_truth_ra_tiles, all_truth_dec_tiles
            del big_det, big_truth

            if verbose:
                print(f"[Stage 2 {prefix.upper()}] Batched {total_pairs:,} pairwise separations.")

            for gid, start_idx, n_d, n_t in group_slices:
                n_complex += 1
                gi = group_index[gid]
                sep_matrix = all_seps[start_idx:start_idx + n_d * n_t].reshape(n_d, n_t)
                matched_pairs, adjacency = build_bipartite_and_match(
                    sep_matrix, match_rad=match_rad
                )
                det_r, truth_r = classify_group_stage2(
                    n_d, n_t, matched_pairs, adjacency
                )

                det_ilocs = gi['det_iloc']
                truth_ilocs = gi['truth_iloc']
                det_rowids = gi['det_rowid']
                truth_rowids = gi['truth_rowid']

                for local_i, iloc_pos in enumerate(det_ilocs):
                    flags['is_matched'][iloc_pos] = det_r['is_matched'][local_i]
                    mid = det_r['matched_id_local'][local_i]
                    if mid >= 0:
                        flags['matched_id'][iloc_pos] = truth_rowids[mid]
                    flags['sep_dist'][iloc_pos] = det_r['sep_dist'][local_i]
                    flags['is_spurious'][iloc_pos] = det_r['is_spurious'][local_i]
                    flags['is_shred'][iloc_pos] = det_r['is_shred'][local_i]
                    flags['is_unrec_blend'][iloc_pos] = det_r['is_unrec_blend'][local_i]
                    flags['is_part_of_blend'][iloc_pos] = det_r['is_part_of_blend'][local_i]
                    flags['is_resolved_blend'][iloc_pos] = det_r['is_resolved_blend'][local_i]
                    flags['is_partial_deblend'][iloc_pos] = det_r['is_partial_deblend'][local_i]

                for local_j, iloc_pos in enumerate(truth_ilocs):
                    flags['is_matched'][iloc_pos] = truth_r['is_matched'][local_j]
                    mid = truth_r['matched_id_local'][local_j]
                    if mid >= 0:
                        flags['matched_id'][iloc_pos] = det_rowids[mid]
                    flags['sep_dist'][iloc_pos] = truth_r['sep_dist'][local_j]
                    flags['is_missed'][iloc_pos] = truth_r['is_missed'][local_j]
                    flags['is_blended_away'][iloc_pos] = truth_r['is_blended_away'][local_j]
                    flags['is_shredded'][iloc_pos] = truth_r['is_shredded'][local_j]
                    flags['is_unrec_blend'][iloc_pos] = truth_r['is_unrec_blend'][local_j]
                    flags['is_part_of_blend'][iloc_pos] = truth_r['is_part_of_blend'][local_j]
                    flags['is_resolved_blend'][iloc_pos] = truth_r['is_resolved_blend'][local_j]
                    flags['is_partial_deblend'][iloc_pos] = truth_r['is_partial_deblend'][local_j]

    if verbose:
        print(f"[Stage 2 {prefix.upper()}] Done: {n_isolated:,} isolated, "
              f"{n_complex:,} complex groups matched.")

    # 6. Bulk write all flags to DataFrame
    for col in ALL_FLAG_COLS:
        analysis_df[f'{prefix}_{col}'] = flags[col]

    return analysis_df


# ============================================================================
# Summary & Metrics
# ============================================================================

def stage2_summary(analysis_df, det_prefix):
    """Print Stage 2 classification summary for one pipeline."""
    prefix = det_prefix
    det_cat_key = 'dd_det' if det_prefix == 'dd' else 'lsst_det'

    dets = analysis_df[analysis_df['catalog_key'] == det_cat_key]
    truths = analysis_df[analysis_df['catalog_key'] == 'lsst_truth']

    print(f"\n=== Stage 2 Summary for {det_prefix.upper()} ===")
    print(f"Total detections: {len(dets):,}")
    print(f"  Matched:          {dets[f'{prefix}_is_matched'].sum():,}")
    print(f"  Spurious:         {dets[f'{prefix}_is_spurious'].sum():,}")
    print(f"  Shreds:           {dets[f'{prefix}_is_shred'].sum():,}")
    print(f"  Unrec blend:      {dets[f'{prefix}_is_unrec_blend'].sum():,}")
    print(f"  Resolved blend:   {dets[f'{prefix}_is_resolved_blend'].sum():,}")
    print(f"  Partial deblend:  {dets[f'{prefix}_is_partial_deblend'].sum():,}")
    print()
    print(f"Total truth objects: {len(truths):,}")
    print(f"  Matched:          {truths[f'{prefix}_is_matched'].sum():,}")
    print(f"  Missed:           {truths[f'{prefix}_is_missed'].sum():,}")
    print(f"  Blended away:     {truths[f'{prefix}_is_blended_away'].sum():,}")
    print(f"  Shredded:         {truths[f'{prefix}_is_shredded'].sum():,}")
    print(f"  Unrec blend:      {truths[f'{prefix}_is_unrec_blend'].sum():,}")
    print(f"  Resolved blend:   {truths[f'{prefix}_is_resolved_blend'].sum():,}")
    print(f"  Partial deblend:  {truths[f'{prefix}_is_partial_deblend'].sum():,}")
    print()