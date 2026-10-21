"""
compute_scalar_table.py — scalar blend metrics with beta-binomial CIs.

One markdown table (stdout) + one long-format CSV per (buffer, linking length),
columns = LSST then each DD run. All formulas, masks, and asserts come from
blend_metrics.py; this script only loads, cuts at m_cut, and formats.

Example:
    python compute_scalar_table.py \
        --run-names comb_30k_4h200_bs144_ep50 \
        --score-thresholds 0.65 --nms-thresholds 0.65 \
        --lsst-source-run comb_30k_4h200_bs144_ep50 \
        --buffers 2 --linking-lengths 1.0 --mag-limit gold \
        --output-dir ./scalar_tables
    
    python compute_scalar_table.py --run-names lsst5_30k_4h200_bs192_ep50_musigma --score-thresholds 0.65 --nms-thresholds 0.65 --linking-lengths 0.5 1.0
"""
import argparse
import csv
from pathlib import Path

import blend_metrics as bm

def _pipeline_row(truths, dets, prefix, m_cut):
    """counts, metrics, beta-binomial CIs for one loaded pipeline."""
    mt = (truths[bm.MAG_COL] < m_cut)
    md = (dets[bm.MAG_COL] < m_cut)
    bm.assert_maximality(truths, dets, prefix)
    counts = bm.compute_counts(truths, dets, prefix, mt, md)
    metrics = bm.compute_metrics(counts)
    cis = bm.scalar_cis(truths, dets, prefix, m_cut)
    return counts, metrics, cis

def _class_row(truths, dets, prefix, m_cut):
    """class confusion counts, metrics, beta-binomial CIs for one loaded pipeline.
    Mirrors _pipeline_row: takes the full buffered (truths, dets), attaches the
    matched-detection class, cuts at m_cut internally."""
    tcls = bm.attach_pred_class(truths, dets, prefix)
    mask = tcls[bm.MAG_COL] < m_cut
    counts = bm.compute_class_counts(tcls, prefix, mask)
    metrics = bm.compute_class_metrics(counts)
    cis = bm.class_scalar_cis(tcls, prefix, m_cut)
    return counts, metrics, cis

def _fmt(v, lo, hi):
    if v != v:  # NaN
        return '—'
    return f'{v:.4f} [{lo:.4f}, {hi:.4f}]'


def _fmt_rd(rho, rho_raw, deff):
    if rho != rho:  # NaN (F1, which has no rho/deff of its own)
        return '—'
    # rho_raw < 0 means the ANOVA estimate was negative (anti-correlated within
    # blend, e.g. survivor/absorbed) and floored to rho=0 for the design effect --
    # surface that distinctly from a genuinely-uncorrelated rho_raw ~ 0.
    if rho_raw < 0:
        return f'ρ=0.000 (raw={rho_raw:.3f}, floored), DEFF={deff:.3f}'
    return f'ρ={rho:.3f}, DEFF={deff:.3f}'


METRIC_ORDER = list(bm.METRIC_NUMDEN) + ['f1']
COUNT_ORDER = ['N_t', 'N_t_matched', 'N_t_missed', 'N_t_abs_all', 'N_t_sig',
               'N_t_res', 'N_t_surv', 'N_t_abs', 'N_t_shredded',
               'N_d', 'N_d_matched', 'N_d_spurious', 'N_d_shred', 'N_d_abr',
               'n_det_nan_mag', 'f_det_nan_mag']
CLASS_METRIC_ORDER = list(bm.CLASS_METRICS)                     # 4 metrics
CLASS_COUNT_ORDER = ['N_t_cls', 'N_gg', 'N_gs', 'N_sg', 'N_ss', 'n_matched_uncls']


def _emit(results, class_results, buffer, ll, mag_limit, out_dir):
    """results, class_results: ordered lists of (label, counts, metrics, cis). Print markdown, write CSV."""
    labels = [r[0] for r in results]
    assert [r[0] for r in class_results] == labels, 'class_results must match results order/labels'
    print(f'\n### Scalar metrics — buffer={buffer}, LL={ll}", mag_limit={mag_limit}\n')

    # metrics table
    print('| Metric | ' + ' | '.join(labels) + ' |')
    print('|' + '---|' * (len(labels) + 1))
    for m in METRIC_ORDER:
        cells = [_fmt(mt[m], ci[m]['lo'], ci[m]['hi']) for _, _, mt, ci in results]
        print(f'| {m} | ' + ' | '.join(cells) + ' |')

    # ICC / design-effect table
    print(f'\n**ICC / design effect** (buffer={buffer}, LL={ll}")\n')
    print('| Metric | ' + ' | '.join(labels) + ' |')
    print('|' + '---|' * (len(labels) + 1))
    for m in METRIC_ORDER:
        cells = [_fmt_rd(ci[m]['rho'], ci[m]['rho_raw'], ci[m]['deff']) for _, _, _, ci in results]
        print(f'| {m} | ' + ' | '.join(cells) + ' |')

    # counts table
    print(f'\n**Counts** (buffer={buffer}, LL={ll}")\n')
    print('| Count | ' + ' | '.join(labels) + ' |')
    print('|' + '---|' * (len(labels) + 1))
    for k in COUNT_ORDER:
        fmt = '{:.4f}' if k.startswith('f_') else '{:,}'
        print(f'| {k} | ' + ' | '.join(fmt.format(c[k]) for _, c, _, _ in results) + ' |')

    # ----- classification block (mirrors the three tables above) -----
    print(f'\n### Classification — buffer={buffer}, LL={ll}", mag_limit={mag_limit}\n')
    print('| Metric | ' + ' | '.join(labels) + ' |')
    print('|' + '---|' * (len(labels) + 1))
    for m in CLASS_METRIC_ORDER:
        cells = [_fmt(mt[m], ci[m]['lo'], ci[m]['hi']) for _, _, mt, ci in class_results]
        print(f'| {m} | ' + ' | '.join(cells) + ' |')
 
    print(f'\n**Classification ICC / design effect** (buffer={buffer}, LL={ll}")\n')
    print('| Metric | ' + ' | '.join(labels) + ' |')
    print('|' + '---|' * (len(labels) + 1))
    for m in CLASS_METRIC_ORDER:
        cells = [_fmt_rd(ci[m]['rho'], ci[m]['rho_raw'], ci[m]['deff']) for _, _, _, ci in class_results]
        print(f'| {m} | ' + ' | '.join(cells) + ' |')
 
    print(f'\n**Classification counts** (buffer={buffer}, LL={ll}")\n')
    print('| Count | ' + ' | '.join(labels) + ' |')
    print('|' + '---|' * (len(labels) + 1))
    for k in CLASS_COUNT_ORDER:
        print(f'| {k} | ' + ' | '.join('{:,}'.format(c[k]) for _, c, _, _ in class_results) + ' |')

    # CSV (long, self-contained)
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f'scalar_metrics_buf{buffer}_ll{ll}.csv'
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['metric', 'pipeline', 'value', 'ci_lo', 'ci_hi', 'rho', 'rho_raw',
                        'deff', 'numerator', 'denominator'])
            for (label, counts, metrics, cis), (clabel, ccounts, cmetrics, ccis) in zip(
                    results, class_results):
                assert label == clabel
                # blend metrics
                for m in METRIC_ORDER:
                    if m in bm.METRIC_NUMDEN:
                        nk, dk = bm.METRIC_NUMDEN[m]
                        num, den = counts[nk], counts[dk]
                    else:  # f1
                        num, den = '', ''
                    ci = cis[m]
                    w.writerow([m, label, metrics[m], ci['lo'], ci['hi'], ci['rho'],
                                ci['rho_raw'], ci['deff'], num, den])
                for k in COUNT_ORDER:
                    w.writerow([f'count:{k}', label, counts[k], '', '', '', '', '', '', ''])
                # ----- classification metrics (namespaced class:) + counts -----
                for m in CLASS_METRIC_ORDER:
                    num_cell, den_cells = bm.CLASS_METRIC_CELLS[m]
                    num = ccounts[num_cell]
                    den = sum(ccounts[x] for x in den_cells)
                    ci = ccis[m]
                    w.writerow([f'class:{m}', label, cmetrics[m], ci['lo'], ci['hi'], ci['rho'],
                                ci['rho_raw'], ci['deff'], num, den])
                for k in CLASS_COUNT_ORDER:
                    w.writerow([f'count:{k}', label, ccounts[k], '', '', '', '', '', '', ''])
        print(f'\n_CSV → {path}_')

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root-run-dir', default='~/lsst_runs',
                   help='Base directory containing each run subdirectory')
    p.add_argument('--run-names', nargs='+', required=True,
                   help='DeepDISC run directory names to include as table columns')
    p.add_argument('--score-thresholds', nargs='+', type=float, required=True,
                   help='score threshold per run, or a single shared value')
    p.add_argument('--nms-thresholds', nargs='+', type=float, required=True,
                   help='NMS threshold per run, or a single shared value')
    p.add_argument('--lsst-source-run', default='lsst5_30k_4h200_bs192_ep50_musigma',
                   help='run whose lsst/ parquet supplies the LSST column')
    p.add_argument('--buffers', nargs='+', type=int, default=[2],
                   help='Δm values added to the mag-limit cut for the evaluation '
                        'range (mag_limit_val = m_cut + buffer)')
    p.add_argument('--linking-lengths', nargs='+', default=['0.5','1.0'],
                   help='FoF linking-length tags to report separately, matching the '
                        'analysis_cats subdirectory names (e.g. 0.5 1.0)')
    p.add_argument('--mag-limit', default='gold', choices=bm.MAG_LIMITS,
                   help='Which blend_metrics.MAG_LIMITS entry defines m_cut')
    p.add_argument('--output-dir', default=None,
                   help='Write a CSV per (buffer, linking length) here; '
                        'omit to print markdown tables only')
    return p.parse_args()


def main():
    a = parse_args()
    root = Path(a.root_run_dir).expanduser()
    runs = a.run_names
    if len(a.score_thresholds) != len(a.nms_thresholds):
        raise SystemExit('--score-thresholds and --nms-thresholds must have the same length')
    combos = [bm.combo_tag(s, n) for s, n in zip(a.score_thresholds, a.nms_thresholds)]
    combos = combos if len(combos) == len(runs) else combos * len(runs)
    if len(combos) != len(runs):
        raise SystemExit('--score-thresholds/--nms-thresholds must be one shared '
                         'pair or one per run')
    m_cut = bm.MAG_LIMITS[a.mag_limit]

    for buffer in a.buffers:
        mag_limit_val = m_cut + buffer
        for ll in a.linking_lengths:
            results, class_results = [], []
            # LSST once (threshold-independent, combo=None)
            lsst_df = bm.load_analysis_df(root / a.lsst_source_run, 'lsst',
                                          str(ll), None, mag_limit_val)
            lt, ld = bm.split_truth_det(lsst_df, 'lsst')
            results.append(('LSST', *_pipeline_row(lt, ld, 'lsst', m_cut)))
            class_results.append(('LSST', *_class_row(lt, ld, 'lsst', m_cut)))
            # each DD run
            for run, combo in zip(runs, combos):
                dd_df = bm.load_analysis_df(root / run, 'dd', str(ll), combo, mag_limit_val)
                dt, dd = bm.split_truth_det(dd_df, 'dd')
                label = f'{run}/{combo}'
                results.append((label, *_pipeline_row(dt, dd, 'dd', m_cut)))
                class_results.append((label, *_class_row(dt, dd, 'dd', m_cut)))
            _emit(results, class_results, buffer, ll, a.mag_limit, a.output_dir)

if __name__ == '__main__':
    main()