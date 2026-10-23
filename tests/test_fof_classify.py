"""
test_fof_classify.py — Comprehensive tests for fof_classify.py against the
Catalog Matching section of the paper.

Coverage map (paper -> tests):
  Table stage1_labels        -> TestStage1Labels
  Bipartite/Hungarian text   -> TestBipartiteMatching
  Table object_flags         -> TestStage2ObjectFlags
  Table stage2_labels +
    adjacency-based blends   -> TestGroupBlendFlags
  Significant blend defn     -> TestSignificantBlend
  classify_stage2 plumbing
    (bulk missed/spurious,
     isolated batch, complex
     batch, flag writing)    -> TestEndToEndClassifyStage2
  Maximality claims
    ("absorbed points to a
     matched det", "every
     unrec blend retains a
     survivor", trichotomy)  -> TestInvariantsRandomized

Group membership is injected directly (no FOF call): Stage 2 logic is
independent of how groups were formed, and FoFCatalogMatching is an
external dependency we treat as trusted.

Run:  pytest test_fof_classify.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for the analysis modules

from fof_classify import (
    ALL_FLAG_COLS,
    build_bipartite_and_match,
    build_table,
    classify_group_stage2,
    classify_stg1,
    classify_stage2,
    get_stage1_counts,
    add_group_blend_props,
    add_sig_blend_flag,
)

MATCH_RAD = 1.0


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def run_group(sep_matrix, match_rad=MATCH_RAD):
    """Run bipartite matching + per-group classification on a sep matrix."""
    sep_matrix = np.asarray(sep_matrix, dtype=float)
    n_d, n_t = sep_matrix.shape
    pairs, adj = build_bipartite_and_match(sep_matrix, match_rad)
    det_r, truth_r = classify_group_stage2(n_d, n_t, pairs, adj)
    return pairs, adj, det_r, truth_r


def pair_dict(pairs):
    """{det_local_idx: truth_local_idx}"""
    return {di: ti for di, ti, _ in pairs}


# ----------------------------------------------------------------------------
# Bipartite matching (Hungarian + sentinel)
# ----------------------------------------------------------------------------

class TestBipartiteMatching:

    def test_no_edges_returns_empty(self):
        pairs, adj = build_bipartite_and_match(np.array([[5.0, 3.0]]), MATCH_RAD)
        assert pairs == []
        assert not adj.any()

    def test_single_valid_pair(self):
        pairs, adj = build_bipartite_and_match(np.array([[0.3]]), MATCH_RAD)
        assert len(pairs) == 1
        di, ti, sep = pairs[0]
        assert (di, ti) == (0, 0)
        assert sep == pytest.approx(0.3)

    def test_boundary_sep_equal_to_radius_is_matched(self):
        # adjacency and validity both use <=
        pairs, adj = build_bipartite_and_match(np.array([[MATCH_RAD]]), MATCH_RAD)
        assert adj[0, 0] == 1
        assert len(pairs) == 1

    def test_prefers_closer_match(self):
        # 2 dets, 1 truth: closer det wins, one-to-one enforced
        pairs, _ = run_group([[0.2], [0.5]])[0:2]
        assert pair_dict(pairs) == {0: 0}

    def test_out_of_radius_forced_assignment_discarded(self):
        # 2x2 square forces 2 assignments; the >radius one must be dropped
        pairs, _ = build_bipartite_and_match(
            np.array([[0.5, 5.0], [5.0, 5.0]]), MATCH_RAD
        )
        assert pair_dict(pairs) == {0: 0}
        assert len(pairs) == 1

    def test_maximum_cardinality_beats_greedy(self):
        # Greedy nearest-neighbor would take D0-T0 (0.1) and strand D1.
        # Maximal matching takes D0-T1 (0.9), D1-T0 (0.2): 2 matches.
        sep = np.array([[0.1, 0.9],
                        [0.2, 5.0]])
        pairs, _ = build_bipartite_and_match(sep, MATCH_RAD)
        assert pair_dict(pairs) == {0: 1, 1: 0}

    def test_nonsquare_matrices(self):
        # more truths than dets and vice versa both work
        pairs, _ = build_bipartite_and_match(np.array([[0.1, 0.6, 0.9]]), MATCH_RAD)
        assert pair_dict(pairs) == {0: 0}
        pairs, _ = build_bipartite_and_match(
            np.array([[0.1], [0.6], [0.9]]), MATCH_RAD
        )
        assert pair_dict(pairs) == {0: 0}

    def test_rectangular_group_leaves_unmatchable_det_unassigned(self):
        # 3 dets, 2 truths: rectangular assignment only covers min(3,2)=2
        # rows. Det 2 has no valid edge anywhere and is simply excluded
        # from the assignment rather than forced onto a bad pair.
        sep = np.array([
            [0.2, 5.0],
            [5.0, 0.3],
            [5.0, 5.0],
        ])
        pairs, _ = build_bipartite_and_match(sep, MATCH_RAD)
        assert pair_dict(pairs) == {0: 0, 1: 1}
        assert len(pairs) == 2

    def test_square_group_forces_and_discards_invalid_pair(self):
        # 3x3 square: Hungarian must produce a FULL assignment even for a
        # det with no real candidate anywhere; that sentinel-cost pair has
        # to survive the algorithm and then get filtered by the match_rad
        # check afterward.
        sep = np.array([
            [0.2, 5.0, 5.0],
            [5.0, 0.3, 5.0],
            [5.0, 5.0, 5.0],   # det 2 has no valid edge to any truth
        ])
        pairs, adj = build_bipartite_and_match(sep, MATCH_RAD)
        assert pair_dict(pairs) == {0: 0, 1: 1}
        assert len(pairs) == 2
        assert not adj[2, :].any()

    def test_symmetric_tie_still_yields_valid_maximal_matching(self):
        # all four separations equal -> both possible 1-to-1 assignments
        # have identical total cost. Don't assert which one scipy picks,
        # only that whichever it returns is a valid maximal matching.
        pairs, _ = build_bipartite_and_match(np.array([[0.5, 0.5], [0.5, 0.5]]), MATCH_RAD)
        assert len(pairs) == 2
        assert {p[0] for p in pairs} == {0, 1}
        assert {p[1] for p in pairs} == {0, 1}

    def test_zero_separation_self_match(self):
        pairs, _ = build_bipartite_and_match(np.array([[0.0]]), MATCH_RAD)
        assert pair_dict(pairs) == {0: 0}

    def test_match_rad_zero_boundary(self):
        # exact coincidence still matches at radius 0; anything off matches nothing
        pairs, _ = build_bipartite_and_match(np.array([[0.0, 0.05]]), match_rad=0.0)
        assert pair_dict(pairs) == {0: 0}
        pairs, _ = build_bipartite_and_match(np.array([[0.05]]), match_rad=0.0)
        assert pairs == []


# ----------------------------------------------------------------------------
# build_table (col-config merge/rename, untested join path)
# ----------------------------------------------------------------------------

class TestBuildTable:

    def test_merges_and_renames_configured_columns(self):
        results = pd.DataFrame({
            'row_index': [0, 1], 'catalog_key': ['lsst_truth', 'lsst_truth'],
            'group_id': [0, 0],
        })
        cats = {'lsst_truth': pd.DataFrame({'objectId': [10, 11], 'mag_i': [22.0, 23.0]})}
        out = build_table(results, cats, {'lsst_truth': {'objectId': 'id'}})
        assert list(out['id']) == [10, 11]
        assert 'mag_i' not in out.columns  # only configured cols are pulled in

    def test_unrecognized_catalog_key_is_skipped(self, capsys):
        results = pd.DataFrame({
            'row_index': [0, 1], 'catalog_key': ['unknown_cat', 'dd_det'],
            'group_id': [0, 0],
        })
        out = build_table(results, {'dd_det': pd.DataFrame({'mag_i': [21.0]})}, {})
        assert list(out['catalog_key']) == ['dd_det']
        assert 'unknown_cat' in capsys.readouterr().out

    def test_all_keys_unrecognized_raises(self):
        # ponytail: every catalog_key skipped -> chunks is empty -> pd.concat([])
        # raises ValueError instead of returning an empty frame. Latent gap;
        # this test documents current behavior rather than papering over it.
        results = pd.DataFrame({
            'row_index': [0], 'catalog_key': ['unknown_cat'], 'group_id': [0],
        })
        with pytest.raises(ValueError):
            build_table(results, {}, {})


# ----------------------------------------------------------------------------
# Stage 1 labels (paper Table: stage1_labels)
# ----------------------------------------------------------------------------

class TestStage1Labels:

    CASES = [
        (1, 1, 'isolated_match'),        # n_t = 1 and n_d = 1
        (3, 3, 'candidate_res_blend'),   # n_t > 1 and n_d = n_t
        (2, 0, 'missed_false_neg'),      # n_t > n_d and n_d = 0
        (1, 0, 'missed_false_neg'),
        (0, 2, 'spurious_false_pos'),    # n_d > n_t and n_t = 0
        (0, 1, 'spurious_false_pos'),
        (3, 1, 'unrec_blend'),           # n_t > n_d and n_d = 1
        (2, 1, 'unrec_blend'),
        (1, 3, 'shredded'),              # n_d > n_t and n_t = 1
        (1, 2, 'shredded'),
        (5, 3, 'complex_merge'),         # n_t > n_d and n_d > 1
        (3, 5, 'complex_shred'),         # n_d > n_t and n_t > 1
    ]

    def _counts(self, rows):
        df = pd.DataFrame(rows, columns=['n_truth', 'n_dd'])
        return classify_stg1(df, prefix='dd', det_col_name='n_dd')

    def test_all_table_rows(self):
        rows = [(t, d) for t, d, _ in self.CASES]
        out = self._counts(rows)
        for (n_t, n_d, expected), got in zip(self.CASES, out['dd_stg1']):
            assert got == expected, f"(n_t={n_t}, n_d={n_d}) -> {got}"

    def test_rules_are_exhaustive(self):
        # every nonempty group gets a real label (default 'n/a' unreachable)
        rows = [(t, d) for t in range(0, 7) for d in range(0, 7)
                if (t, d) != (0, 0)]
        out = self._counts(rows)
        assert (out['dd_stg1'] != 'n/a').all()


# ----------------------------------------------------------------------------
# Stage 2 object flags (paper Table: object_flags)
# ----------------------------------------------------------------------------

class TestStage2ObjectFlags:

    def test_isolated_style_match(self):
        _, _, det_r, truth_r = run_group([[0.3]])
        assert det_r['is_matched'][0] and truth_r['is_matched'][0]
        assert det_r['matched_id_local'][0] == 0
        assert truth_r['matched_id_local'][0] == 0
        assert det_r['sep_dist'][0] == pytest.approx(0.3)
        assert not det_r['is_spurious'][0] and not det_r['is_shred'][0]
        assert not truth_r['is_missed'][0]

    def test_no_edges_spurious_and_missed(self):
        _, _, det_r, truth_r = run_group([[3.0]])
        assert det_r['is_spurious'][0] and not det_r['is_matched'][0]
        assert truth_r['is_missed'][0] and not truth_r['is_matched'][0]
        assert np.isnan(det_r['sep_dist'][0])

    def test_shred_detection_and_shredded_truth(self):
        # paper: unmatched det with >=1 edge = shred (fell on a claimed object);
        # matched truth with >=1 adjacent unmatched det = shredded
        _, _, det_r, truth_r = run_group([[0.2], [0.5]])
        assert det_r['is_matched'][0]
        assert det_r['is_shred'][1] and not det_r['is_spurious'][1]
        assert truth_r['is_matched'][0] and truth_r['is_shredded'][0]
        # 1 connected truth only -> NOT a blend, no blend flags anywhere
        for r in (det_r, truth_r):
            assert not r['is_part_of_blend'].any()
            assert not r['is_unrec_blend'].any()
            assert not r['is_resolved_blend'].any()

    def test_absorbed_truth(self):
        # paper: unmatched truth with an edge = absorbed (is_absorbed),
        # and it points at a matched detection (maximality)
        _, adj, det_r, truth_r = run_group([[0.1, 0.5]])
        assert truth_r['is_matched'][0]              # closer truth wins
        assert truth_r['is_absorbed'][1]
        assert not truth_r['is_missed'][1]
        # the absorbed truth's only edge is to the matched detection
        assert adj[0, 1] == 1 and det_r['is_matched'][0]
        # survivor = the matched component of the unrecognized blend
        assert truth_r['is_survivor'][0]
        assert not truth_r['is_survivor'][1]


# ----------------------------------------------------------------------------
# Group-level blend flags (paper Table: stage2_labels)
# ----------------------------------------------------------------------------

class TestGroupBlendFlags:

    def test_resolved_blend(self):
        # 2 truths, 2 dets, chain geometry, each matched to its own det
        # T0=0, D0=0.4, T1=0.8, D1=1.2  (1-D positions in arcsec)
        d = np.array([0.4, 1.2]); t = np.array([0.0, 0.8])
        sep = np.abs(d[:, None] - t[None, :])
        pairs, _, det_r, truth_r = run_group(sep)
        assert pair_dict(pairs) == {0: 0, 1: 1}
        assert truth_r['is_resolved_blend'].all()
        assert det_r['is_resolved_blend'].all()
        assert truth_r['is_part_of_blend'].all()
        assert not truth_r['is_unrec_blend'].any()
        assert not truth_r['is_survivor'].any()
        assert not truth_r['is_absorbed'].any()

    def test_unrecognized_blend_flags_broadcast(self):
        _, _, det_r, truth_r = run_group([[0.1, 0.5]])
        # all connected objects carry blend + unrec flags
        assert det_r['is_part_of_blend'][0] and det_r['is_unrec_blend'][0]
        assert truth_r['is_part_of_blend'].all()
        assert truth_r['is_unrec_blend'].all()
        assert not truth_r['is_resolved_blend'].any()

    def test_edgeless_truth_does_not_create_blend(self):
        # paper: "a truth object with no edges ... does not turn its
        # neighbors into a blend"
        _, _, det_r, truth_r = run_group([[0.2, 3.0]])
        assert truth_r['is_matched'][0]
        assert truth_r['is_missed'][1]
        assert not truth_r['is_part_of_blend'].any()
        assert not det_r['is_part_of_blend'].any()

    def test_clustered_dets_equal_counts_not_resolved(self):
        # paper: "Equal counts do not imply correct correspondence."
        # 3 dets cluster near T0; T1, T2 edgeless.
        d = np.array([0.1, 0.3, 0.5]); t = np.array([0.0, 10.0, 20.0])
        sep = np.abs(d[:, None] - t[None, :])
        pairs, _, det_r, truth_r = run_group(sep)
        assert len(pairs) == 1
        assert truth_r['is_matched'][0] and truth_r['is_shredded'][0]
        assert truth_r['is_missed'][1] and truth_r['is_missed'][2]
        assert det_r['is_shred'].sum() == 2
        # only 1 connected truth -> not a blend
        assert not truth_r['is_part_of_blend'].any()

    def test_complex_merge_survivors_and_absorbed(self):
        # T0=0, T1=0.7, T2=1.4; D0=0.1, D1=1.3
        d = np.array([0.1, 1.3]); t = np.array([0.0, 0.7, 1.4])
        sep = np.abs(d[:, None] - t[None, :])
        pairs, _, det_r, truth_r = run_group(sep)
        assert pair_dict(pairs) == {0: 0, 1: 2}
        assert truth_r['is_survivor'][0] and truth_r['is_survivor'][2]
        assert truth_r['is_absorbed'][1]
        assert truth_r['is_part_of_blend'].all()
        assert truth_r['is_unrec_blend'].all()
        assert det_r['is_unrec_blend'].all()

    def test_disjoint_isolated_pairs_in_one_group_flagged_as_blend(self):
        # Two FOF-chained but physically unrelated pairs share one group:
        # T0-D0 (sep 0.1) and T1-D1 (sep 0.1) have zero cross edges. Blend
        # flags run over the group's WHOLE adjacency matrix rather than
        # per connected component, so both matched pairs get flagged as a
        # "resolved blend" even though neither pair has an edge to the
        # other. Documents a known limitation, not a fix.
        d = np.array([0.1, 100.1]); t = np.array([0.0, 100.0])
        sep = np.abs(d[:, None] - t[None, :])
        pairs, adj, det_r, truth_r = run_group(sep)
        assert pair_dict(pairs) == {0: 0, 1: 1}
        assert adj[0, 1] == 0 and adj[1, 0] == 0   # no cross edges at all
        assert truth_r['is_resolved_blend'].all()  # flagged anyway
        assert truth_r['is_part_of_blend'].all()

    def test_blend_contaminates_unrelated_isolated_pair_in_same_group(self):
        # T0,T1 form a real unrecognized blend (D0 only reaches T0); T2-D1
        # is a separate, fully disconnected isolated match chained into the
        # same FOF group. D1 has zero edge into the blend, yet still gets
        # is_unrec_blend/is_survivor -- group-wide connected-truth counting
        # contaminates an otherwise-unrelated match in the same group.
        d = np.array([0.1, 500.1]); t = np.array([0.0, 0.6, 500.0])
        sep = np.abs(d[:, None] - t[None, :])
        pairs, adj, det_r, truth_r = run_group(sep)
        assert pair_dict(pairs) == {0: 0, 1: 2}
        assert adj[1, 0] == 0 and adj[1, 1] == 0   # D1 has no edge into the blend
        assert truth_r['is_absorbed'][1]           # T1: the actual absorbed truth
        assert truth_r['is_survivor'][0] and truth_r['is_survivor'][2]
        assert det_r['is_unrec_blend'][1]          # D1 contaminated too
        assert truth_r['is_unrec_blend'][2]        # T2 contaminated too

    def test_resolved_component_mislabeled_unrec_by_disconnected_component(self):
        # Two real, mutually disconnected blends chained into ONE FOF group:
        # T0/T1/D0 is an unrecognized blend (T1 absorbed); T2/T3/D1/D2 is a
        # blend that is FULLY resolved on its own (both truths matched, no
        # unmatched connected truth, zero edges to the other component).
        # fully_resolved/has_unrec_blend are computed once over the group's
        # WHOLE adjacency graph, so the unrelated absorbed T1 drags the
        # resolved component down too -- it never gets is_resolved_blend
        # despite being perfectly deblended in isolation. Worse than the
        # isolated-pair contamination above: a genuinely resolved blend
        # gets mislabeled as unrecognized.
        d = np.array([0.1, 1000.1, 1000.9])
        t = np.array([0.0, 0.6, 1000.0, 1000.8])
        sep = np.abs(d[:, None] - t[None, :])
        pairs, adj, det_r, truth_r = run_group(sep)
        assert pair_dict(pairs) == {0: 0, 1: 2, 2: 3}
        assert adj[0, 2] == 0 and adj[1, 0] == 0   # zero cross edges between components
        assert truth_r['is_matched'][2] and truth_r['is_matched'][3]
        assert not truth_r['is_resolved_blend'][2]
        assert not truth_r['is_resolved_blend'][3]
        assert truth_r['is_unrec_blend'][2] and truth_r['is_unrec_blend'][3]
        assert det_r['is_unrec_blend'][1] and det_r['is_unrec_blend'][2]

    def test_resolved_blend_with_shred_coexist(self):
        # paper: "A single group can contain both shredding and a blend"
        # T0=0, T1=2.0; D0=0.1, D1=2.1, D2=0.4 (edge to T0 only)
        d = np.array([0.1, 2.1, 0.4]); t = np.array([0.0, 2.0])
        sep = np.abs(d[:, None] - t[None, :])
        pairs, _, det_r, truth_r = run_group(sep)
        assert pair_dict(pairs) == {0: 0, 1: 1}
        assert det_r['is_shred'][2] and det_r['is_part_of_blend'][2]
        assert not det_r['is_resolved_blend'][2]   # only matched dets flagged
        assert det_r['is_resolved_blend'][0] and det_r['is_resolved_blend'][1]
        assert truth_r['is_resolved_blend'].all()
        assert truth_r['is_shredded'][0]           # shredded AND resolved
        assert not truth_r['is_shredded'][1]       # D2 has no edge to T1


# ----------------------------------------------------------------------------
# Significant blends (add_group_blend_props / add_sig_blend_flag)
# ----------------------------------------------------------------------------

def _blend_df(groups):
    """groups: {gid: (truth_mags, n_dets)} -> long-format frame."""
    rows = []
    for gid, (mags, n_det) in groups.items():
        for m in mags:
            rows.append({'group_id': gid, 'catalog_key': 'lsst_truth',
                         'mag_i': m})
        for _ in range(n_det):
            rows.append({'group_id': gid, 'catalog_key': 'dd_det',
                         'mag_i': np.nan})
    return pd.DataFrame(rows)


class TestSignificantBlend:

    def test_counts_and_min_gap(self):
        df = _blend_df({0: ([24.0, 24.4, 26.0], 1), 1: ([22.0], 1)})
        out = add_group_blend_props(df)
        g0 = out[out['group_id'] == 0]
        assert (g0['n_truth_in_group'] == 3).all()      # dets excluded from count
        assert g0['min_truth_dmag_i'].iloc[0] == pytest.approx(0.4)
        g1 = out[out['group_id'] == 1]
        assert (g1['n_truth_in_group'] == 1).all()
        assert g1['min_truth_dmag_i'].isna().all()

    def test_broadcast_to_detection_rows(self):
        df = _blend_df({0: ([23.0, 23.5], 2)})
        out = add_sig_blend_flag(add_group_blend_props(df), dmag_sig=1.0)
        det_rows = out[out['catalog_key'] == 'dd_det']
        assert (det_rows['n_truth_in_group'] == 2).all()
        assert det_rows['is_sig_blend'].all()

    def test_strict_inequality(self):
        # paper: "less than dmag_sig" -> gap exactly equal is NOT significant
        df = _blend_df({0: ([23.0, 24.0], 0)})
        out = add_sig_blend_flag(add_group_blend_props(df), dmag_sig=1.0)
        assert not out['is_sig_blend'].any()

    def test_single_faint_straggler_does_not_mask_bright_pair(self):
        # paper: min pairwise gap, not full spread
        df = _blend_df({0: ([21.0, 21.3, 27.0], 0)})
        out = add_sig_blend_flag(add_group_blend_props(df), dmag_sig=1.0)
        assert out['is_sig_blend'].all()

    def test_significance_is_detection_independent(self):
        # a fully missed group (0 dets) can still be a significant blend
        df = _blend_df({0: ([24.0, 24.2], 0)})
        out = add_sig_blend_flag(add_group_blend_props(df), dmag_sig=1.0)
        assert out['is_sig_blend'].all()

    def test_nan_mags(self):
        df = _blend_df({0: ([23.0, np.nan], 0), 1: ([23.0, np.nan, 23.6], 0)})
        out = add_sig_blend_flag(add_group_blend_props(df), dmag_sig=1.0)
        assert not out[out['group_id'] == 0]['is_sig_blend'].any()   # <2 finite
        assert out[out['group_id'] == 1]['is_sig_blend'].all()       # gap 0.6

    def test_missing_mag_col_raises(self):
        df = _blend_df({0: ([23.0, 23.5], 0)}).drop(columns=['mag_i'])
        with pytest.raises(KeyError):
            add_group_blend_props(df)


# ----------------------------------------------------------------------------
# End-to-end classify_stage2 (bulk paths, batching, flag writing, matched_id)
# ----------------------------------------------------------------------------

def _build_e2e():
    """
    10 groups covering every Stage 1 label and every Stage 2 scenario.
    1-D geometry: offsets (arcsec) along the equator; base RA separates groups.
    Truth row_index 1000+, det row_index 2000+.
    """
    rows = []
    trid, drid = [1000], [2000]

    def add(gid, key, off_arcsec, mag=np.nan):
        rid = trid if key == 'lsst_truth' else drid
        rows.append({
            'group_id': gid, 'catalog_key': key, 'row_index': rid[0],
            'ra': 10.0 + gid * 1.0 + off_arcsec / 3600.0, 'dec': 0.0,
            'mag_i': mag,
        })
        rid[0] += 1
        return rows[-1]['row_index']

    ids = {}
    # g0 isolated match (sep 0.3)
    ids['g0_T'] = add(0, 'lsst_truth', 0.0, 22.0)
    ids['g0_D'] = add(0, 'dd_det', 0.3)
    # g1 isolated pair beyond radius (sep 1.5) -> spurious + missed
    ids['g1_T'] = add(1, 'lsst_truth', 0.0, 22.0)
    ids['g1_D'] = add(1, 'dd_det', 1.5)
    # g2 missed group (2 truths, 0 dets), significant (gap 0.4)
    ids['g2_T0'] = add(2, 'lsst_truth', 0.0, 24.0)
    ids['g2_T1'] = add(2, 'lsst_truth', 0.5, 24.4)
    # g3 spurious group (0 truths, 2 dets)
    ids['g3_D0'] = add(3, 'dd_det', 0.0)
    ids['g3_D1'] = add(3, 'dd_det', 0.5)
    # g4 unrecognized blend (T0=0, T1=0.6, D=0.1), significant (gap 0.5)
    ids['g4_T0'] = add(4, 'lsst_truth', 0.0, 23.0)
    ids['g4_T1'] = add(4, 'lsst_truth', 0.6, 23.5)
    ids['g4_D'] = add(4, 'dd_det', 0.1)
    # g5 shred (T=0; D0=0.2, D1=0.5)
    ids['g5_T'] = add(5, 'lsst_truth', 0.0, 23.0)
    ids['g5_D0'] = add(5, 'dd_det', 0.2)
    ids['g5_D1'] = add(5, 'dd_det', 0.5)
    # g6 resolved blend (T0=0, D0=0.4, T1=0.8, D1=1.2), NOT significant (gap 3)
    ids['g6_T0'] = add(6, 'lsst_truth', 0.0, 22.0)
    ids['g6_T1'] = add(6, 'lsst_truth', 0.8, 25.0)
    ids['g6_D0'] = add(6, 'dd_det', 0.4)
    ids['g6_D1'] = add(6, 'dd_det', 1.2)
    # g7 complex merge (T0=0, T1=0.7, T2=1.4; D0=0.1, D1=1.3), sig (gap 0.3)
    ids['g7_T0'] = add(7, 'lsst_truth', 0.0, 21.0)
    ids['g7_T1'] = add(7, 'lsst_truth', 0.7, 21.3)
    ids['g7_T2'] = add(7, 'lsst_truth', 1.4, 24.0)
    ids['g7_D0'] = add(7, 'dd_det', 0.1)
    ids['g7_D1'] = add(7, 'dd_det', 1.3)
    # g8 Stage1 unrec_blend that Stage2 resolves into match+missed
    #    (T0=0, T1=3.0 edgeless, D=0.2), significant (gap 0.5)
    ids['g8_T0'] = add(8, 'lsst_truth', 0.0, 22.0)
    ids['g8_T1'] = add(8, 'lsst_truth', 3.0, 22.5)
    ids['g8_D'] = add(8, 'dd_det', 0.2)
    # g9 resolved blend + shred (T0=0, T1=2.0; D0=0.1, D1=2.1, D2=0.4), sig
    ids['g9_T0'] = add(9, 'lsst_truth', 0.0, 20.0)
    ids['g9_T1'] = add(9, 'lsst_truth', 2.0, 20.9)
    ids['g9_D0'] = add(9, 'dd_det', 0.1)
    ids['g9_D1'] = add(9, 'dd_det', 2.1)
    ids['g9_D2'] = add(9, 'dd_det', 0.4)

    return pd.DataFrame(rows), ids


@pytest.fixture(scope='module')
def e2e():
    analysis, ids = _build_e2e()
    counts = get_stage1_counts(
        analysis,
        expected_keys=['dd_det', 'lsst_truth'],
        rename_map={'lsst_truth': 'n_truth', 'dd_det': 'n_dd'},
    )
    counts = classify_stg1(counts, prefix='dd', det_col_name='n_dd')
    out = classify_stage2(analysis.copy(), counts, 'dd',
                          match_rad=MATCH_RAD, verbose=False)
    out = add_group_blend_props(out)
    out = add_sig_blend_flag(out, dmag_sig=1.0)
    return out, counts, ids


def _row(df, rid):
    return df[df['row_index'] == rid].iloc[0]


class TestEndToEndClassifyStage2:

    def test_stage1_labels(self, e2e):
        _, counts, _ = e2e
        expected = {
            0: 'isolated_match', 1: 'isolated_match', 2: 'missed_false_neg',
            3: 'spurious_false_pos', 4: 'unrec_blend', 5: 'shredded',
            6: 'candidate_res_blend', 7: 'complex_merge',
            8: 'unrec_blend', 9: 'complex_shred',
        }
        for gid, label in expected.items():
            assert counts.loc[gid, 'dd_stg1'] == label

    def test_all_flag_columns_written(self, e2e):
        out, _, _ = e2e
        for col in ALL_FLAG_COLS:
            assert f'dd_{col}' in out.columns

    def test_isolated_match_and_matched_id_crossref(self, e2e):
        out, _, ids = e2e
        d, t = _row(out, ids['g0_D']), _row(out, ids['g0_T'])
        assert d['dd_is_matched'] and t['dd_is_matched']
        assert d['dd_matched_id'] == ids['g0_T']
        assert t['dd_matched_id'] == ids['g0_D']
        assert d['dd_sep_dist'] == pytest.approx(0.3, abs=1e-3)

    def test_isolated_beyond_radius(self, e2e):
        out, _, ids = e2e
        assert _row(out, ids['g1_D'])['dd_is_spurious']
        assert _row(out, ids['g1_T'])['dd_is_missed']
        assert not _row(out, ids['g1_D'])['dd_is_matched']
        assert _row(out, ids['g1_D'])['dd_matched_id'] == -1

    def test_bulk_missed_and_spurious_groups(self, e2e):
        out, _, ids = e2e
        assert _row(out, ids['g2_T0'])['dd_is_missed']
        assert _row(out, ids['g2_T1'])['dd_is_missed']
        assert _row(out, ids['g3_D0'])['dd_is_spurious']
        assert _row(out, ids['g3_D1'])['dd_is_spurious']

    def test_unrecognized_blend_group(self, e2e):
        out, _, ids = e2e
        d = _row(out, ids['g4_D'])
        t0, t1 = _row(out, ids['g4_T0']), _row(out, ids['g4_T1'])
        assert d['dd_is_matched'] and d['dd_matched_id'] == ids['g4_T0']
        assert d['dd_sep_dist'] == pytest.approx(0.1, abs=1e-3)
        assert t0['dd_is_survivor'] and t0['dd_is_matched']
        assert t1['dd_is_absorbed'] and not t1['dd_is_missed']
        for r in (d, t0, t1):
            assert r['dd_is_part_of_blend'] and r['dd_is_unrec_blend']

    def test_shred_group(self, e2e):
        out, _, ids = e2e
        assert _row(out, ids['g5_D0'])['dd_is_matched']
        assert _row(out, ids['g5_D1'])['dd_is_shred']
        t = _row(out, ids['g5_T'])
        assert t['dd_is_matched'] and t['dd_is_shredded']
        assert not t['dd_is_part_of_blend']

    def test_resolved_blend_group(self, e2e):
        out, _, ids = e2e
        for k in ('g6_T0', 'g6_T1', 'g6_D0', 'g6_D1'):
            r = _row(out, ids[k])
            assert r['dd_is_matched'] and r['dd_is_resolved_blend']
            assert r['dd_is_part_of_blend'] and not r['dd_is_unrec_blend']
        assert _row(out, ids['g6_D0'])['dd_matched_id'] == ids['g6_T0']
        assert _row(out, ids['g6_D1'])['dd_matched_id'] == ids['g6_T1']

    def test_complex_merge_group(self, e2e):
        out, _, ids = e2e
        assert _row(out, ids['g7_T0'])['dd_is_survivor']
        assert _row(out, ids['g7_T2'])['dd_is_survivor']
        assert _row(out, ids['g7_T1'])['dd_is_absorbed']
        for k in ('g7_T0', 'g7_T1', 'g7_T2', 'g7_D0', 'g7_D1'):
            assert _row(out, ids[k])['dd_is_unrec_blend']

    def test_stage1_unrec_downgraded_by_stage2(self, e2e):
        # g8: Stage 1 says unrec_blend, Stage 2 finds match + edgeless miss
        out, _, ids = e2e
        assert _row(out, ids['g8_D'])['dd_is_matched']
        assert _row(out, ids['g8_T0'])['dd_is_matched']
        assert _row(out, ids['g8_T1'])['dd_is_missed']
        for k in ('g8_D', 'g8_T0', 'g8_T1'):
            r = _row(out, ids[k])
            assert not r['dd_is_part_of_blend'] and not r['dd_is_unrec_blend']

    def test_resolved_plus_shred_group(self, e2e):
        out, _, ids = e2e
        d2 = _row(out, ids['g9_D2'])
        assert d2['dd_is_shred'] and d2['dd_is_part_of_blend']
        assert not d2['dd_is_resolved_blend']
        t0 = _row(out, ids['g9_T0'])
        assert t0['dd_is_shredded'] and t0['dd_is_resolved_blend']
        assert not _row(out, ids['g9_T1'])['dd_is_shredded']

    def test_sig_blend_flags(self, e2e):
        out, _, ids = e2e
        assert _row(out, ids['g4_D'])['dd_is_matched']  # sanity
        assert _row(out, ids['g4_T0'])['is_sig_blend']
        assert _row(out, ids['g4_D'])['is_sig_blend']        # broadcast to det
        assert not _row(out, ids['g6_T0'])['is_sig_blend']   # gap 3.0
        assert _row(out, ids['g2_T0'])['is_sig_blend']       # missed but sig
        assert _row(out, ids['g8_T0'])['is_sig_blend']       # sig, unengaged
        assert not _row(out, ids['g0_T'])['is_sig_blend']    # single truth

    def test_engaged_components_partition(self, e2e):
        # engaged component = truth & is_part_of_blend & is_sig_blend;
        # exactly one of {resolved, survivor, absorbed} (paper: sum to 1)
        out, _, _ = e2e
        truths = out[out['catalog_key'] == 'lsst_truth']
        eng = truths[truths['dd_is_part_of_blend'] & truths['is_sig_blend']]
        assert len(eng) > 0
        cats = (eng['dd_is_resolved_blend'].astype(int)
                + eng['dd_is_survivor'].astype(int)
                + eng['dd_is_absorbed'].astype(int))
        assert (cats == 1).all()


# ----------------------------------------------------------------------------
# Randomized invariants (paper's maximality claims and flag exclusivity)
# ----------------------------------------------------------------------------

class TestInvariantsRandomized:

    N_TRIALS = 300

    def _random_group(self, rng):
        n_d = rng.integers(1, 7)
        n_t = rng.integers(1, 7)
        d = rng.uniform(0.0, 4.0, size=n_d)
        t = rng.uniform(0.0, 4.0, size=n_t)
        sep = np.abs(d[:, None] - t[None, :])
        return sep

    def test_invariants(self):
        rng = np.random.default_rng(42)
        for _ in range(self.N_TRIALS):
            sep = self._random_group(rng)
            pairs, adj, det_r, truth_r = run_group(sep)
            n_d, n_t = sep.shape

            # one-to-one
            dets = [p[0] for p in pairs]
            truths = [p[1] for p in pairs]
            assert len(dets) == len(set(dets))
            assert len(truths) == len(set(truths))

            # matched pairs valid: within radius, along an edge
            for di, ti, s in pairs:
                assert s <= MATCH_RAD and adj[di, ti] == 1
                assert s == pytest.approx(sep[di, ti])

            m_d = det_r['is_matched']
            m_t = truth_r['is_matched']

            # MAXIMALITY: no unmatched det shares an edge with unmatched truth
            for di in range(n_d):
                for ti in range(n_t):
                    if adj[di, ti] and not m_d[di]:
                        assert m_t[ti], "maximality violated"

            # det trichotomy: exactly one of matched/shred/spurious
            det_cats = (m_d.astype(int) + det_r['is_shred'].astype(int)
                        + det_r['is_spurious'].astype(int))
            assert (det_cats == 1).all()

            # truth trichotomy: exactly one of matched/missed/absorbed
            tr_cats = (m_t.astype(int) + truth_r['is_missed'].astype(int)
                       + truth_r['is_absorbed'].astype(int))
            assert (tr_cats == 1).all()

            # shredded only ever fires on matched truths (table object_flags;
            # the unmatched-shredded branch is unreachable under Hungarian)
            assert not (truth_r['is_shredded'] & ~m_t).any()

            # shred det implies an edge; spurious implies no edges
            for di in range(n_d):
                if det_r['is_shred'][di]:
                    assert adj[di, :].any()
                if det_r['is_spurious'][di]:
                    assert not adj[di, :].any()

            # blend membership: truth is part_of_blend iff connected and the
            # group has >=2 connected truths
            connected = adj.sum(axis=0) > 0
            n_conn = int(connected.sum())
            for ti in range(n_t):
                expect = bool(connected[ti]) and n_conn >= 2
                assert bool(truth_r['is_part_of_blend'][ti]) == expect

            # resolved vs unrecognized are group-exclusive
            assert not (truth_r['is_resolved_blend'].any()
                        and truth_r['is_unrec_blend'].any())

            # blend-component partition (paper: resolved+survivor+absorbed=1)
            in_blend = truth_r['is_part_of_blend']
            comp = (truth_r['is_resolved_blend'].astype(int)
                    + truth_r['is_survivor'].astype(int)
                    + truth_r['is_absorbed'].astype(int))
            assert (comp[in_blend] == 1).all()
            assert (comp[~in_blend] == 0).all()

            # every unrecognized blend retains >=1 survivor (maximality)
            if truth_r['is_absorbed'].any():
                assert truth_r['is_survivor'].any()

            # survivor => matched & unrec; absorbed => unmatched & unrec
            assert not (truth_r['is_survivor'] & ~m_t).any()
            assert not (truth_r['is_survivor'] & ~truth_r['is_unrec_blend']).any()
            assert not (truth_r['is_absorbed'] & m_t).any()
            assert not (truth_r['is_absorbed']
                        & ~truth_r['is_unrec_blend']).any()

            # # partial_deblend == unrec_blend under maximal matching
            # assert (truth_r['is_partial_deblend']
            #         == truth_r['is_unrec_blend']).all()
            # assert (det_r['is_partial_deblend']
            #         == det_r['is_unrec_blend']).all()


# ----------------------------------------------------------------------------
# get_stage1_counts plumbing
# ----------------------------------------------------------------------------

class TestStage1Counts:
    def test_missing_catalog_key_filled_with_zero(self):
        df = pd.DataFrame({
            'group_id': [0, 0], 'catalog_key': ['lsst_truth', 'lsst_truth'],
        })
        counts = get_stage1_counts(
            df, expected_keys=['dd_det', 'lsst_truth'],
            rename_map={'lsst_truth': 'n_truth', 'dd_det': 'n_dd'},
        )
        assert counts.loc[0, 'n_truth'] == 2
        assert counts.loc[0, 'n_dd'] == 0

    def test_default_expected_keys_and_rename_map(self):
        # no expected_keys/rename_map passed -> exercises the built-in defaults
        df = pd.DataFrame({'group_id': [0], 'catalog_key': ['lsst_truth']})
        counts = get_stage1_counts(df)
        assert set(counts.columns) == {'n_truth', 'n_dd', 'n_lsst'}
        assert counts.loc[0, 'n_truth'] == 1
        assert counts.loc[0, 'n_dd'] == 0
        assert counts.loc[0, 'n_lsst'] == 0


class TestClassifyStage2LsstPrefix:
    # every other classify_stage2 test uses det_prefix='dd'; 'lsst' selects a
    # different det_cat_key branch ('lsst_det') that was otherwise never hit.

    def test_lsst_prefix_uses_lsst_det_key(self):
        analysis = pd.DataFrame([
            {'group_id': 0, 'catalog_key': 'lsst_truth', 'row_index': 100,
             'ra': 10.0, 'dec': 0.0},
            {'group_id': 0, 'catalog_key': 'lsst_det', 'row_index': 200,
             'ra': 10.0 + 0.3 / 3600, 'dec': 0.0},
        ])
        counts = get_stage1_counts(
            analysis, expected_keys=['lsst_det', 'lsst_truth'],
            rename_map={'lsst_truth': 'n_truth', 'lsst_det': 'n_lsst'},
        )
        counts = classify_stg1(counts, prefix='lsst', det_col_name='n_lsst')
        out = classify_stage2(analysis.copy(), counts, 'lsst',
                               match_rad=MATCH_RAD, verbose=False)
        row = out[out['catalog_key'] == 'lsst_det'].iloc[0]
        assert row['lsst_is_matched']


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))