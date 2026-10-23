"""
window_diagram.py — Swin Stage-1 attention windows on the four resampling-ablation inputs
(paper §3.1.2, figure `fig:ups_windows`).

One test cutout (the scene of the paper's `fig:scene`) is prepared the way each model
receives it, and the backbone's Stage-1 window grid is drawn on top. The grid is derived
from each model's patch size with the same padding rules the model applies:

    canvas  = square_pad rounded up to a multiple of the coarsest backbone stride
              (patch * 2**3, the FPN size_divisibility), e.g. 512 -> 520 for patch 13
    tokens  = canvas / patch per side, padded by Swin to a multiple of the window size (7)
    windows = padded tokens / 7 per side

and the window counts are checked against the paper's (36 and 361). The upsampled inputs
use the cached Lanczos remap to the Roman grid that training and inference used.

Writes one PDF per panel, a standalone legend.pdf and panels.tex (the \\subfloat row, with
subcaptions from the computed geometry) to <paper>/figures/windows/.

    python paper_figures/window_diagram.py --preview-dir /tmp/window_preview
"""
import argparse
import hashlib
import math
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from astropy.visualization import make_lupton_rgb

FN = '/u/yse2/lsst_data/truth/53.71_-41.5/c110_53.71_-41.5.npy'   # fig:scene's cutout (test set)
CACHE_DIR = '/work/hdd/bfhm/g4merz/wcs_map_cache/test_8k_keypoints_wcs'
WINDOW = 7            # Swin window size M, in tokens
N_MERGES = 3          # patch-merging stages: coarsest backbone stride = patch * 2**3
UPS_SIZE = 512

# order = left-to-right in the paper: baseline window count first, then the increased count
MODELS = [
    dict(key='baseline', title='Baseline', patch=4, square_pad=160, content='native', windows=36),
    dict(key='ps13', title='Coarser patch size (PS13)', patch=13, square_pad=512, content='interpolated',
         windows=36),
    dict(key='padded', title='Padded', patch=4, square_pad=512, content='zero-padded', windows=361),
    dict(key='ups', title='Lanczos upsampled', patch=4, square_pad=512, content='interpolated', windows=361),
]

BG, GRID_C, HILITE_C, HATCH_C = 'black', 'white', 'cyan', '#5a5a5a'
PAD_HATCH, TOKEN_HATCH = '///', '...'
plt.rcParams['hatch.linewidth'] = 0.4   # thin hatching, so the window grid stays readable over it


def geometry(m):
    div = m['patch'] * 2 ** N_MERGES
    canvas = math.ceil(m['square_pad'] / div) * div
    assert canvas % m['patch'] == 0, f"{m['key']}: canvas {canvas} not a multiple of patch {m['patch']}"
    tokens = canvas // m['patch']
    padded_tokens = math.ceil(tokens / WINDOW) * WINDOW
    n_win = padded_tokens // WINDOW
    assert n_win ** 2 == m['windows'], f"{m['key']}: {n_win}x{n_win} windows, paper says {m['windows']}"
    return dict(canvas=canvas, tokens=tokens, padded_tokens=padded_tokens, n_win=n_win,
                window_px=WINDOW * m['patch'], grid_px=padded_tokens * m['patch'])


def lanczos_upsample(native):
    """The training/inference remap of a Rubin cutout onto the 512x512 Roman grid (cached maps)."""
    rub = FN.replace('/u/', '/work/hdd/bdsp/')          # the key_mapper path the cache is keyed on
    rom = rub.replace('lsst_data', 'truth-roman').replace('/truth/', '/')
    map_x, map_y = np.load(f"{CACHE_DIR}/{hashlib.md5((rub + rom).encode()).hexdigest()}.npy")
    out = np.stack([cv2.remap(c, map_x, map_y, interpolation=cv2.INTER_LANCZOS4,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0) for c in native])
    assert out.shape[1:] == (UPS_SIZE, UPS_SIZE)
    return out


def make_rgb(img, bands=(3, 2, 1), Q=8):
    """i/r/g Lupton composite with a per-image percentile stretch, as in scene_plots.ipynb."""
    r, g, b = img[bands[0]], img[bands[1]], img[bands[2]]
    pmin = np.percentile([r, g, b], 1)
    stretch = np.percentile([r, g, b], 99.5) - pmin
    return make_lupton_rgb(r, g, b, minimum=pmin, stretch=stretch, Q=Q)


def draw_panel(ax, img, geo):
    """Array orientation (row 0 at the top), matching how the model pads: zeros are added on
    the right and bottom of the image, and Swin pads tokens on the right and bottom."""
    h, w = img.shape[1:]
    c, g, wp = geo['canvas'], geo['grid_px'], geo['window_px']
    ax.set_facecolor(BG)
    ax.imshow(make_rgb(img), extent=(0, w, h, 0), interpolation='nearest', zorder=1)
    # zero padding inside the input canvas (right and bottom of the image)
    for x, y, rw, rh in ((w, 0, c - w, c), (0, h, w, c - h)):
        if rw > 0 and rh > 0:
            ax.add_patch(Rectangle((x, y), rw, rh, facecolor=BG, edgecolor=HATCH_C, hatch=PAD_HATCH,
                                   lw=0, zorder=2))
    # Swin's internal token padding beyond the canvas, up to the last full window
    for x, y, rw, rh in ((c, 0, g - c, g), (0, c, c, g - c)):
        if rw > 0 and rh > 0:
            ax.add_patch(Rectangle((x, y), rw, rh, facecolor=BG, edgecolor=HATCH_C, hatch=TOKEN_HATCH,
                                   lw=0, zorder=2))
    # Stage-1 window grid
    for k in range(geo['n_win'] + 1):
        ax.plot([k * wp, k * wp], [0, g], color=GRID_C, lw=0.5, alpha=0.8, zorder=3)
        ax.plot([0, g], [k * wp, k * wp], color=GRID_C, lw=0.5, alpha=0.8, zorder=3)
    # one window's 7x7 patch tokens
    p = wp / WINDOW
    for k in range(1, WINDOW):
        ax.plot([k * p, k * p], [0, wp], color=HILITE_C, lw=0.35, alpha=0.8, zorder=4)
        ax.plot([0, wp], [k * p, k * p], color=HILITE_C, lw=0.35, alpha=0.8, zorder=4)
    ax.add_patch(Rectangle((0, 0), wp, wp, fill=False, ec=HILITE_C, lw=1.2, zorder=5))
    ax.set_xlim(0, g)
    ax.set_ylim(g, 0)
    ax.set_aspect('equal')
    ax.axis('off')


def subcaption(m, geo):
    n = geo['n_win']
    return (rf"{{\shortstack{{{m['title']} \\ \scriptsize {m['content'].capitalize()} content \\ "
            rf"\scriptsize ${geo['canvas']}^2$ input, {m['patch']}-px patches \\ "
            rf"\scriptsize ${n} \times {n}$ windows}}}}")


def panels_tex(rows, rel_dir, width=r'0.235\textwidth'):
    lines = [rf"  \subfloat[{subcaption(m, geo)}]{{\includegraphics[width={width}]{{{rel_dir}/{m['key']}.pdf}}}}"
             + (r' \hfill' if i < len(rows) - 1 else '') for i, (m, geo) in enumerate(rows)]
    return ('% Generated by DeepDISC_Roman_Rubin/window_diagram.py -- do not edit by hand.\n'
            + '\n'.join(lines) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--paper-dir', default='/u/yse2/roman_rubin_blending_paper')
    ap.add_argument('--rel-dir', default='figures/windows')
    ap.add_argument('--preview-dir', default=None, help='also write PNG previews here')
    ap.add_argument('--size', type=float, default=1.9, help='panel size in inches')
    a = ap.parse_args()

    out_dir = Path(a.paper_dir) / a.rel_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    prev = Path(a.preview_dir) if a.preview_dir else None
    if prev:
        prev.mkdir(parents=True, exist_ok=True)

    native = np.load(FN)
    ups = lanczos_upsample(native)
    rows = []
    for m in MODELS:
        geo = geometry(m)
        img = native if m['key'] in ('baseline', 'padded') else ups
        fig = plt.figure(figsize=(a.size, a.size), facecolor=BG)
        ax = fig.add_axes([0, 0, 1, 1])
        draw_panel(ax, img, geo)
        fig.savefig(out_dir / f"{m['key']}.pdf", facecolor=BG)
        if prev:
            fig.savefig(prev / f"{m['key']}.png", dpi=200, facecolor=BG)
        plt.close(fig)
        rows.append((m, geo))
        print(f"{m['key']:9s} canvas {geo['canvas']:3d}  tokens {geo['tokens']:3d}->{geo['padded_tokens']:3d}  "
              f"windows {geo['n_win']}x{geo['n_win']} of {geo['window_px']} px  "
              f"span {geo['grid_px']} px")

    # white-background legend: the white and cyan lines get a thin black outline so they stay
    # visible on the page while still reading as the lines drawn in the panels
    outline = [pe.Stroke(linewidth=2.6, foreground='black'), pe.Normal()]
    handles = [Line2D([], [], color=GRID_C, lw=1.2, path_effects=outline, label='Stage-1 attention window'),
               Line2D([], [], color=HILITE_C, lw=1.2, path_effects=outline,
                      label='7×7 patch tokens of one window'),
               Patch(facecolor=BG, edgecolor=HATCH_C, hatch=PAD_HATCH, label='Zero padding of the input'),
               Patch(facecolor=BG, edgecolor=HATCH_C, hatch=TOKEN_HATCH, label='Swin token padding')]
    fig = plt.figure(figsize=(7.0, 0.35))
    fig.legend(handles=handles, loc='center', ncols=4, frameon=False, fontsize=8, labelcolor='#222222',
               handlelength=2.0, columnspacing=1.4)
    fig.savefig(out_dir / 'legend.pdf', bbox_inches='tight')
    if prev:
        fig.savefig(prev / 'legend.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    (out_dir / 'panels.tex').write_text(panels_tex(rows, a.rel_dir))
    print(f'wrote {len(rows)} panels, legend.pdf and panels.tex to {out_dir}')


if __name__ == '__main__':
    main()
