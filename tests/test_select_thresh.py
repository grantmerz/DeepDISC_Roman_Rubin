"""Tests for select_thresh's ranking and operating-point selection. Run: python -m pytest tests/test_select_thresh.py"""
import operator
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for the analysis modules
from select_thresh import (apply_constraints, mean_percentile_across_lls, parse_constraints, pareto_set,
                           percentile, select_operating_point, stability)


def _grid(**cols):
    """Three combos along score_thresh at one nms_thresh, plus metric columns."""
    return pd.DataFrame({'score_thresh': [0.1, 0.2, 0.3], 'nms_thresh': [0.5, 0.5, 0.5], **cols})


def test_percentile_direction():
    df = _grid(**{'dd_completeness_1.0': [0.5, 0.9, 0.7], 'dd_missed_frac_1.0': [0.5, 0.9, 0.7]})
    assert percentile(df, 'dd', 'completeness', '1.0').iloc[1] == 100.0  # higher-is-better: max raw -> 100
    assert percentile(df, 'dd', 'missed_frac', '1.0').iloc[0] == 100.0   # lower-is-better: min raw -> 100


def test_mean_percentile_across_lls_ranks_best_first():
    # (0.1) is best at LL 0.5 but worst at LL 1.0; (0.2) is second and first, so it wins on the mean
    # percentiles: (0.1) 100 & 33.3 -> 66.7, (0.2) 66.7 & 100 -> 83.3, (0.3) 33.3 & 66.7 -> 50
    df = _grid(**{'dd_f1_0.5': [0.9, 0.8, 0.4], 'dd_f1_1.0': [0.5, 0.9, 0.6]})
    ranked = mean_percentile_across_lls(df, 'dd', 'f1', [0.5, 1.0])
    assert list(ranked.index.names) == ['score_thresh', 'nms_thresh']
    assert ranked.index[0] == (0.2, 0.5)
    assert ranked.index[-1] == (0.3, 0.5)


def test_select_operating_point_mean_and_min():
    # f1 favours (0.1); unrec_blend_frac (lower is better) favours (0.3)
    df = _grid(**{'dd_f1_1.0': [0.9, 0.8, 0.7], 'dd_unrec_blend_frac_1.0': [0.12, 0.10, 0.08]})
    sel = select_operating_point(df, 'dd', ['f1', 'unrec_blend_frac'], '1.0')
    assert sel['mean_pct'].is_monotonic_decreasing                      # sorted best-first
    assert (sel['min_pct'] <= sel[['f1', 'unrec_blend_frac']].min(axis=1)).all()
    assert sel.loc[(0.1, 0.5), 'unrec_blend_frac'] < sel.loc[(0.3, 0.5), 'unrec_blend_frac']


def test_pareto_set_respects_metric_direction():
    # (0.2) is worse than (0.1) on both f1 and unrec_blend_frac -> dominated;
    # (0.3) trades f1 for a lower unrec_blend_frac -> kept
    df = _grid(**{'dd_f1_1.0': [0.9, 0.8, 0.7], 'dd_unrec_blend_frac_1.0': [0.10, 0.12, 0.08]})
    assert list(pareto_set(df, 'dd', ['f1', 'unrec_blend_frac'], '1.0')) == [True, False, True]


def test_constraints_filter_and_reject_bad_input():
    df = _grid(**{'dd_purity_1.0': [0.80, 0.90, 0.95]})
    kept = apply_constraints(df, 'dd', parse_constraints(['purity >= 0.9']), '1.0')
    assert list(kept['score_thresh']) == [0.2, 0.3]
    with pytest.raises(ValueError):
        parse_constraints(['purity at least 0.9'])
    with pytest.raises(KeyError):
        apply_constraints(df, 'dd', [('completeness', operator.ge, 0.5)], '1.0')


def test_stability_span_is_neighbourhood_range():
    scores, nmss = [0.1, 0.2, 0.3], [0.4, 0.5, 0.6]
    rows = [{'score_thresh': s, 'nms_thresh': n, 'dd_f1_1.0': 0.80 + 0.01 * i}
            for i, (s, n) in enumerate((s, n) for s in scores for n in nmss)]
    block, span, _ = stability(pd.DataFrame(rows), 'dd', 'f1', '1.0', (0.2, 0.5))
    assert block.shape == (3, 3)
    assert span == pytest.approx(0.08)                                  # 0.88 - 0.80 over the 3x3 block


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
