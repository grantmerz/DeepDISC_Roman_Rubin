import sys
from pathlib import Path
import numpy as np
from scipy.stats import norm
sys.path.insert(0, '/u/yse2/DeepDISC_Roman_Rubin')
import blend_metrics as bm
from plot_binned_metrics import NotebookArgs

root = Path('~/lsst_runs').expanduser()
LL = '1.0'
m_cut = bm.MAG_LIMITS['gold']
mlv = m_cut + 2
edges = NotebookArgs().mag_bins
RUNS = [('LSST', 'lsst5_30k_4h200_bs192_ep50_musigma', 'lsst', None),
        ('Rubin', 'lsst5_30k_4h200_bs192_ep50', 'dd', 's0.5_n0.6'),
        ('RubinUps', 'lsst5_ups_30k_4h200_bs192_ep50', 'dd', 's0.6_n0.65'),
        ('Rubin+Roman', 'comb_30k_4h200_bs144_ep50', 'dd', 's0.6_n0.7')]
rng = np.random.default_rng(0)

def wilson(k, n, z=1.959964):
    if n == 0: return np.nan, np.nan
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d; h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return c - h, c + h

def cluster_boot(gids, y, B=2000):
    _, inv = np.unique(gids, return_inverse=True)
    s = np.bincount(inv, weights=y); m = np.bincount(inv)
    G = len(m)
    idx = rng.integers(0, G, size=(B, G))
    fr = s[idx].sum(1) / m[idx].sum(1)
    return np.percentile(fr, [2.5, 97.5])

for label, run, p, combo in RUNS:
    df = bm.load_analysis_df(root / run, p, LL, combo, mlv)
    t, d = bm.split_truth_det(df, p)
    print(f'\n===== {label} ({run} {combo}) LL={LL}')
    # full-group composition of significant blends (no mag cut, i.e. <27.3)
    sigall = t[t[f'{p}_is_sig_blend'] & t[f'{p}_is_part_of_blend']]
    sz = sigall.groupby('group_id').size()
    print(f'sig groups (<{mlv}): {len(sz)}, truths {len(sigall)}, size dist',
          sz.value_counts().sort_index().head(6).to_dict())
    # scalar population (< m_cut)
    tc = t[t[bm.MAG_COL] < m_cut]; dc = d[(d[bm.MAG_COL] < m_cut) & d[bm.MAG_COL].notna()]
    pops = bm._metric_populations(tc, dc, p)
    for m in ['resolved_frac', 'survivor_frac', 'absorbed_frac', 'completeness']:
        pop, ind = pops[m]; g = pop['group_id'].to_numpy(); y = ind.to_numpy().astype(float)
        k, n = int(y.sum()), y.size
        rho, raw = bm.estimate_icc(g, y); mb = bm.mean_cluster_size(g); deff = bm.design_effect(rho, mb)
        lo, hi = bm.betabinom_ci(k, n, rho, mb); lo0, hi0 = bm.betabinom_ci(k, n, 0, 1)
        wl, wh = wilson(k, n)
        print(f'  {m:14s} k={k:6d} n={n:6d} groups={len(np.unique(g)):6d} m_bar={mb:.3f} '
              f'rho_raw={raw:+.3f} DEFF={deff:.3f} | BB+DEFF hw={(hi-lo)/2:.4f} '
              f'BBflat hw={(hi0-lo0)/2:.4f} Wilson hw={(wh-wl)/2:.4f}')
    if m == 'completeness':
        sgs = pops['resolved_frac'][0].groupby('group_id').size()
        print('  sig pop (<25.3) members-per-group dist', sgs.value_counts().sort_index().to_dict())
    # binned: intervals straight from compute_binned (global rho, per-bin m_bar)
    full = bm._metric_populations(t, d, p)
    pop, ind = full['resolved_frac']
    g_all = pop['group_id'].to_numpy(); rho, _ = bm.estimate_icc(g_all, ind.to_numpy())
    print(f'  binned resolved_frac: global rho={rho:.3f}, full-pop m_bar={bm.mean_cluster_size(g_all):.3f} (not used per bin)')
    b = bm.compute_binned(t, d, p, edges, 1, 1)
    print(f'  {"bin":>11s} {"N_sig":>6s} {"k_res":>6s} {"grps":>5s} {"m_bar":>6s} {"BB+DEFF":>17s} {"Wilson":>17s} {"clusterBoot":>17s}')
    mag = pop[bm.MAG_COL].to_numpy()
    for i in range(len(edges) - 1):
        n = b['resolved_frac']['den'][i]; k = b['resolved_frac']['num'][i]
        if n == 0: continue
        sel = (mag >= edges[i]) & (mag < edges[i+1])
        gi, yi = g_all[sel], ind.to_numpy()[sel].astype(float)
        lo, hi = b['resolved_frac']['lo'][i], b['resolved_frac']['hi'][i]
        wl, wh = wilson(k, n); bl, bh = cluster_boot(gi, yi)
        print(f'  {edges[i]:5.2f}-{edges[i+1]:5.2f} {n:6d} {k:6d} {len(np.unique(gi)):5d} {bm.mean_cluster_size(gi):6.3f} '
              f'[{lo:.3f},{hi:.3f}] [{wl:.3f},{wh:.3f}] [{bl:.3f},{bh:.3f}]')
    # example: LSST 3-truth sig groups with mixed survivor/absorbed, all members < m_cut
    if label in ('LSST', 'Rubin+Roman'):
        cand = []
        for gid, grp in sigall.groupby('group_id'):
            if len(grp) == 3 and grp[f'{p}_is_absorbed'].sum() == 1 and grp[f'{p}_is_survivor'].sum() == 2 \
               and (grp[bm.MAG_COL] < m_cut).all():
                cand.append(gid)
        print(f'  3-member (all <{m_cut}) groups with 2 survivors + 1 absorbed: {len(cand)}')
        for gid in cand[:2]:
            grp = sigall[sigall.group_id == gid]
            print(grp[['group_id', 'id', bm.MAG_COL, f'{p}_is_survivor', f'{p}_is_absorbed', f'{p}_is_matched']].to_string(index=False))
