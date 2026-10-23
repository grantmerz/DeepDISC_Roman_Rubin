"""
matching_diagram.py — the matching-procedure figure (paper Fig. `fig:matching_procedure`).

Each panel is a small hand-placed configuration of truth objects and detections
(positions in arcsec) that is run through the pipeline's own matching code,
fof_classify.build_bipartite_and_match() and classify_group_stage2(), plus the
Stage 1 rules of classify_stg1(). Every line and label in the figure comes from
that output, and each panel asserts that
    - all of its objects form one FOF group at the linking length,
    - the computed Stage 2 outcome is the one the panel is meant to show,
so a change to the matching definitions makes this script fail instead of
drawing a stale figure.

Writes one PDF per panel, a standalone legend.pdf, and panels.tex (the \\subfloat
grid, with subcaptions from the computed outcomes) to <paper>/figures/matching/.
main.tex keeps the figure environment, \\input{figures/matching/panels.tex}, the
legend and the caption.

    python paper_figures/matching_diagram.py --preview-dir /tmp/matching_preview
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.transforms import Bbox

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for the analysis modules
from fof_classify import build_bipartite_and_match, classify_group_stage2, classify_stg1

LINK = 1.0        # FOF linking length (arcsec)
MATCH_RAD = 1.0   # Stage 2 matching radius (arcsec), equal to the linking length

# classify_stg1 codes -> names used in the paper's Stage 1 table (tab:stage1_labels)
STAGE1_NAMES = {
    'isolated_match': 'Isolated Match', 'missed_false_neg': 'Missed',
    'spurious_false_pos': 'Spurious', 'candidate_res_blend': 'Candidate Resolved Blend',
    'unrec_blend': 'Candidate Unrecognized Blend', 'shredded': 'Single Shred',
    'complex_merge': 'Complex Merge', 'complex_shred': 'Complex Shred',
}

# expect = the Stage 2 outcome the panel must produce (checked below).
# tl / dl = label side ('l', 'r', 'u', 'd') for each truth / detection that gets a flag label.
PANELS = [
    dict(expect='Isolated Match', truths=[(0.0, 0.0)], dets=[(0.3, 0.2)], scalebar=True),
    dict(expect='Missed', truths=[(0.0, 0.0)], dets=[], tl=['d']),
    dict(expect='Spurious', truths=[], dets=[(0.0, 0.0)], dl=['d']),
    # cross edges (dashed) exist but the assignment pairs each detection with its own truth
    dict(expect='Resolved Blend', truths=[(-0.45, 0.0), (0.45, 0.0)], dets=[(-0.35, 0.3), (0.35, -0.3)]),
    dict(expect='Unrecognized Blend', truths=[(-0.35, -0.1), (0.35, 0.15)], dets=[(-0.1, 0.25)],
         tl=['d', 'r'], dl=['u']),
    dict(expect='Shredded', truths=[(0.0, 0.0)], dets=[(0.2, 0.25), (0.55, -0.35)], tl=['l'], dl=['r', 'r']),
    # FOF links the two truths (0.9" apart), but only one of them has an edge
    dict(expect='Not a Blend', truths=[(-0.45, 0.0), (0.45, 0.0)], dets=[(-0.6, 0.35)], tl=['d', 'r']),
    # nearest-neighbour matching pairs D1 with T2 and leaves T1 unmatched;
    # the Hungarian assignment matches both
    dict(expect='Resolved Blend', truths=[(0.0, 0.0), (1.05, 0.0)], dets=[(0.7, 0.55), (1.85, 0.45)],
         hungarian=True),
    dict(expect='Resolved Blend + Shred', truths=[(-0.45, 0.0), (0.45, 0.0)],
         dets=[(-0.35, 0.3), (0.35, -0.3), (0.95, 0.45)], tl=['l', 'r'], dl=['r', 'r', 'u']),
]
LABEL_DIRS = {'l': ((-1, 0), 'right', 'center'), 'r': ((1, 0), 'left', 'center'),
              'u': ((0, 1), 'center', 'bottom'), 'd': ((0, -1), 'center', 'top')}

TRUTH_C, DET_C = '#0072B2', '#E69F00'   # Okabe-Ito blue / orange
INK, MUTED = '#222222', '#6b6b6b'
HALO = [pe.withStroke(linewidth=2.5, foreground='white')]


def seps(dets, truths):
    """(n_det, n_truth) flat-sky separations in arcsec."""
    if len(dets) == 0 or len(truths) == 0:
        return np.zeros((len(dets), len(truths)))
    d, t = np.asarray(dets, float), np.asarray(truths, float)
    return np.hypot(*(d[:, None, :] - t[None, :, :]).transpose(2, 0, 1))


def n_fof_groups(points):
    """Number of friends-of-friends groups at LINK among all objects (union-find)."""
    parent = list(range(len(points)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    p = np.asarray(points, float)
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            if np.hypot(*(p[i] - p[j])) <= LINK:
                parent[find(i)] = find(j)
    return len({find(i) for i in range(len(p))})


def greedy_nn_matches(sep):
    """Match count from greedy nearest-neighbour pairing (closest pairs first)."""
    pairs = sorted((sep[i, j], i, j) for i in range(sep.shape[0]) for j in range(sep.shape[1])
                   if sep[i, j] <= MATCH_RAD)
    used_d, used_t, n = set(), set(), 0
    for _, i, j in pairs:
        if i not in used_d and j not in used_t:
            used_d.add(i); used_t.add(j); n += 1
    return n


def stage1_name(n_t, n_d):
    c = classify_stg1(pd.DataFrame({'n_truth': [n_t], 'n_det': [n_d]}), 'x', 'n_det')
    return STAGE1_NAMES[c['x_stg1'].iloc[0]]


def stage2_outcome(n_t, n_d, adj, det_r, truth_r):
    """Group outcome in the terms of the paper's Stage 2 table, from the code's flags."""
    if n_d == 0:
        return 'Missed'
    if n_t == 0:
        return 'Spurious'
    shred = ' + Shred' if det_r['is_shred'].any() else ''
    if (adj.sum(axis=0) > 0).sum() >= 2:   # >= 2 connected truths: a blend
        return ('Unrecognized Blend' if truth_r['is_absorbed'].any() else 'Resolved Blend') + shred
    if n_t == 1 and n_d == 1 and truth_r['is_matched'].all():
        return 'Isolated Match'
    if det_r['is_shred'].any():
        return 'Shredded'
    if n_t >= 2 and truth_r['is_missed'].any():
        return 'Not a Blend'
    raise ValueError('configuration does not map to a Stage 2 outcome')


def flag_labels(det_r, truth_r):
    """Short text for each object's notable flags (plain matches get no label)."""
    t_lab = []
    for i in range(len(truth_r['is_matched'])):
        f = [n for k, n in (('is_missed', 'missed'), ('is_absorbed', 'absorbed'),
                            ('is_survivor', 'survivor'), ('is_shredded', 'shredded')) if truth_r[k][i]]
        t_lab.append(', '.join(f))
    d_lab = []
    for i in range(len(det_r['is_matched'])):
        f = [n for k, n in (('is_spurious', 'spurious'), ('is_shred', 'shred'),
                            ('is_absorber', 'absorber')) if det_r[k][i]]
        d_lab.append(', '.join(f))
    return t_lab, d_lab


def run_panel(p):
    truths, dets = p['truths'], p['dets']
    n_t, n_d = len(truths), len(dets)
    assert n_fof_groups(truths + dets) == 1, f"{p['expect']}: objects are not one FOF group"
    sep = seps(dets, truths)
    matched, adj = build_bipartite_and_match(sep, MATCH_RAD) if n_t and n_d else ([], np.zeros((n_d, n_t), int))
    det_r, truth_r = classify_group_stage2(n_d, n_t, matched, adj)
    out = stage2_outcome(n_t, n_d, adj, det_r, truth_r)
    assert out == p['expect'], f"panel meant to show {p['expect']!r} computes as {out!r}"
    if p.get('hungarian'):
        assert greedy_nn_matches(sep) < len(matched), 'nearest-neighbour does as well as Hungarian'
    return dict(sep=sep, adj=adj, matched=matched, det_r=det_r, truth_r=truth_r, outcome=out,
                stage1=stage1_name(n_t, n_d), n_t=n_t, n_d=n_d)


TRUTH_MS, DET_MS = 11, 9   # marker sizes (pt): filled circle / filled square
SCALEBAR_ROOM = 0.3        # arcsec reserved at the bottom (and top) of a row holding the scale bar


def draw_label(ax, xy, text, side, r=0.2):
    (dx, dy), ha, va = LABEL_DIRS[side]
    ax.text(xy[0] + r * dx, xy[1] + r * dy, text, fontsize=8, color=INK, ha=ha, va=va, zorder=5,
            path_effects=HALO)


def content_bounds(p, pad=0.06):
    """Data limits (arcsec) that hold every matching-radius circle and truth marker in full,
    so each panel is cropped to its own content instead of a fixed square."""
    boxes = [(x - MATCH_RAD, y - MATCH_RAD, x + MATCH_RAD, y + MATCH_RAD) for x, y in p['dets']]
    boxes += [(x - 0.25, y - 0.25, x + 0.25, y + 0.25) for x, y in p['truths']]
    if not p['dets']:   # a lone truth: symmetric room for its label, so the object stays centred
        boxes += [(x - 0.6, y - 0.45, x + 0.6, y + 0.45) for x, y in p['truths']]
    x0, y0 = min(b[0] for b in boxes) - pad, min(b[1] for b in boxes) - pad
    x1, y1 = max(b[2] for b in boxes) + pad, max(b[3] for b in boxes) + pad
    return x0, x1, y0, y1


def row_bounds(panels, per_row=3):
    """Give every panel in a row the same height, content centred vertically. \\subfloat
    aligns a row on its bottom edges, so unequal heights would push short panels (e.g. a
    lone missed truth) to the bottom of the row. A row with the scale bar gets room for it
    at both top and bottom, so its contents stay level with the rest of the row."""
    cb = [content_bounds(p) for p in panels]
    out = []
    for r0 in range(0, len(panels), per_row):
        row = range(r0, min(r0 + per_row, len(panels)))
        H = max(cb[i][3] - cb[i][2] for i in row)
        if any(panels[i].get('scalebar') for i in row):
            H += 2 * SCALEBAR_ROOM
        for i in row:
            x0, x1, y0, y1 = cb[i]
            cy = (y0 + y1) / 2
            out.append((x0, x1, cy - H / 2, cy + H / 2))
    return out


def draw_panel(ax, p, res, bounds):
    truths, dets = np.asarray(p['truths'], float).reshape(-1, 2), np.asarray(p['dets'], float).reshape(-1, 2)

    for d in dets:   # matching radius around each detection
        ax.add_patch(Circle(d, MATCH_RAD, fill=False, ec=DET_C, lw=0.8, ls=(0, (2, 2)), alpha=0.6, zorder=1))
    matched = {(di, ti) for di, ti, _ in res['matched']}
    for di in range(len(dets)):   # edges: solid = Stage 2 match, dashed = edge not matched
        for ti in range(len(truths)):
            if res['adj'][di, ti]:
                m = (di, ti) in matched
                ax.plot(*zip(dets[di], truths[ti]), color=INK if m else MUTED, lw=1.6 if m else 1.1,
                        ls='-' if m else (0, (4, 2)), zorder=2)
    ax.plot(truths[:, 0], truths[:, 1], 'o', ms=TRUTH_MS, color=TRUTH_C, mec='white', mew=0.8,
            ls='none', zorder=3)
    ax.plot(dets[:, 0], dets[:, 1], 's', ms=DET_MS, color=DET_C, mec='white', mew=0.8, ls='none', zorder=4)

    t_lab, d_lab = flag_labels(res['det_r'], res['truth_r'])
    for pts, labs, sides in ((truths, t_lab, p.get('tl', [])), (dets, d_lab, p.get('dl', []))):
        for i, (xy, lab) in enumerate(zip(pts, labs)):
            if lab:
                assert i < len(sides), f"{p['expect']}: no label side given for flagged object {i} ({lab})"
                draw_label(ax, xy, lab, sides[i])
    if p.get('hungarian'):
        # mark the nearest-neighbour pair (D1, T2) that the assignment did not use
        mid = (dets[0] + truths[1]) / 2
        ax.annotate('nearest neighbour,\nnot used', xy=mid, xytext=mid + np.array([0.25, -0.75]),
                    fontsize=7.5, color=INK, ha='center', va='top', path_effects=HALO,
                    arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8), zorder=5)

    x0, x1, y0, y1 = bounds
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect('equal')
    ax.axis('off')   # no frame: each panel is cropped to its own content
    if p.get('scalebar'):
        ax.plot([x0 + 0.05, x0 + 1.05], [y0 + 0.08, y0 + 0.08], color=INK, lw=1.5)
        ax.text(x0 + 0.55, y0 + 0.13, '1″', ha='center', va='bottom', fontsize=8, color=INK)


def slug(i, outcome):
    return f"{i + 1:02d}_" + outcome.lower().replace(' + ', '_and_').replace(' ', '_')


SLOT = 0.33   # each panel sits centred in a slot this fraction of \textwidth wide


def panels_tex(results, rel_dir, per_row=3):
    """The \\subfloat grid for main.tex, with subcaptions taken from the computed outcomes.
    Panels are included at their natural size, so 1 arcsec is the same length in all of them;
    the fixed-width slot keeps each subcaption from widening its row."""
    lines = []
    for i, r in enumerate(results):
        cap = (rf"{{\shortstack{{{r['outcome']} \\ \scriptsize Stage 1: {r['stage1']} \\ "
               rf"\scriptsize $n_t = {r['n_t']}$, $n_d = {r['n_d']}$}}}}")
        sep = r' \\' if (i + 1) % per_row == 0 else r' \hfill'
        img = rf"\includegraphics{{{rel_dir}/{slug(i, r['outcome'])}.pdf}}"
        lines.append(rf"  \subfloat[{cap}]{{\makebox[{SLOT}\textwidth]{{{img}}}}}{sep}")
    return ('% Generated by DeepDISC_Roman_Rubin/matching_diagram.py -- do not edit by hand.\n'
            + '\n'.join(lines) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--paper-dir', default='/u/yse2/roman_rubin_blending_paper')
    ap.add_argument('--rel-dir', default='figures/matching',
                    help='output folder relative to --paper-dir, as referenced from main.tex')
    ap.add_argument('--preview-dir', default=None, help='also write PNG previews here')
    ap.add_argument('--scale', type=float, default=0.7, help='inches per arcsec, shared by all panels')
    ap.add_argument('--textwidth', type=float, default=7.42, help='paper text width in inches (fit check)')
    a = ap.parse_args()

    out_dir = Path(a.paper_dir) / a.rel_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    prev_dir = Path(a.preview_dir) if a.preview_dir else None
    if prev_dir:
        prev_dir.mkdir(parents=True, exist_ok=True)

    results = [run_panel(p) for p in PANELS]
    # each panel sized to its own content at a common scale (inches per arcsec), axes filling
    # the figure, so the same arcsec is the same printed length in every panel
    for i, (p, res, b) in enumerate(zip(PANELS, results, row_bounds(PANELS))):
        w, h = (b[1] - b[0]) * a.scale, (b[3] - b[2]) * a.scale
        fig = plt.figure(figsize=(w, h))
        ax = fig.add_axes([0, 0, 1, 1])
        draw_panel(ax, p, res, b)
        # grow the canvas symmetrically for labels that overhang the circles: symmetric so the
        # group stays centred in its slot and the row heights stay equal; never rescales
        tb = fig.get_tightbbox(fig.canvas.get_renderer())
        dx = max(0.0, -tb.x0, tb.x1 - w) + 0.02
        dy = max(0.0, -tb.y0, tb.y1 - h)
        box = Bbox.from_extents(-dx, -dy, w + dx, h + dy)
        assert box.width < SLOT * a.textwidth, \
            f'panel {i + 1} is {box.width:.2f} in wide, over its slot; lower --scale'
        fig.savefig(out_dir / f'{slug(i, res["outcome"])}.pdf', bbox_inches=box)
        if prev_dir:
            fig.savefig(prev_dir / f'{slug(i, res["outcome"])}.png', dpi=200, bbox_inches=box)
        plt.close(fig)

    handles = [
        Line2D([], [], marker='o', ms=TRUTH_MS, color=TRUTH_C, mec='white', mew=0.8, ls='none',
               label='Ground Truth'),
        Line2D([], [], marker='s', ms=DET_MS, color=DET_C, mec='white', mew=0.8, ls='none', label='Detection'),
        Line2D([], [], color=INK, lw=1.6, label='Stage 2 match'),
        Line2D([], [], color=MUTED, lw=1.1, ls=(0, (4, 2)), label='Edge, not matched'),
        Line2D([], [], color=DET_C, lw=0.8, ls=(0, (2, 2)), alpha=0.6, label='1″ matching radius'),
    ]
    fig = plt.figure(figsize=(7.0, 0.35))
    fig.legend(handles=handles, loc='center', ncols=5, frameon=False, fontsize=8,
               handlelength=2.2, columnspacing=1.2)
    fig.savefig(out_dir / 'legend.pdf', bbox_inches='tight')
    if prev_dir:
        fig.savefig(prev_dir / 'legend.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    (out_dir / 'panels.tex').write_text(panels_tex(results, a.rel_dir))
    for i, r in enumerate(results):
        print(f"({'abcdefghi'[i]}) {r['outcome']:24s} Stage 1: {r['stage1']:28s} matches={len(r['matched'])}")
    print(f'wrote {len(results)} panels, legend.pdf and panels.tex to {out_dir}')


if __name__ == '__main__':
    main()
