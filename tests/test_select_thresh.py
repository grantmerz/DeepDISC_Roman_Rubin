"""Self-check for select_thresh's direction handling. Run: python tests/test_select_thresh.py"""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for the analysis modules
from select_thresh import percentile, robustness


def test_percentile_direction():
    df = pd.DataFrame({'score_thresh': [0.1, 0.2, 0.3], 'nms_thresh': [0.1, 0.1, 0.1],
                       'dd_completeness_1.0': [0.5, 0.9, 0.7],
                       'dd_missed_frac_1.0': [0.5, 0.9, 0.7]})
    assert percentile(df, 'dd', 'completeness', '1.0').iloc[1] == 100.0  # higher-is-better: max raw -> 100
    assert percentile(df, 'dd', 'missed_frac', '1.0').iloc[0] == 100.0   # lower-is-better: min raw -> 100


def test_robustness_min_value_maximin():
    df_a = pd.DataFrame({'score_thresh': [0.1, 0.2], 'nms_thresh': [0.1, 0.1],
                        'dd_completeness_1.0': [0.9, 0.4]})
    df_b = pd.DataFrame({'score_thresh': [0.1, 0.2], 'nms_thresh': [0.1, 0.1],
                        'dd_completeness_1.0': [0.3, 0.8]})
    # worst-case: (0.1,0.1)->min(0.9,0.3)=0.3 ; (0.2,0.1)->min(0.4,0.8)=0.4 -> maximin picks (0.2,0.1)
    rob = robustness([df_a, df_b], 'dd', 'completeness', '1.0', 'min_value')
    assert rob.index[0] == (0.2, 0.1)


if __name__ == '__main__':
    test_percentile_direction()
    test_robustness_min_value_maximin()
    print('OK')
