"""
Add magnitudes to DeepDISC prediction files for a given run

For every (score_thresh, nms_thresh) combination, reads the existing
prediction file at:

    <run_dir>/preds/<preds_dir>/pred_s{score}_n{nms}.json

calculates PSF-weighted magnitudes for all bands (u, g, r, i, z, y)
using a zero-point of 27 (DC2 coadds), and writes a new file with
the mag columns appended:

    <run_dir>/preds/<preds_dir>/pred_s{score}_n{nms}_psf.json

PSF photometry uses the BTK LSST survey PSF rendered to a 35x35 pixel
stamp (https://dp1.lsst.io/tutorials/notebook/205/notebook-205-2.html) 
at 0.2 arcsec/pix. For each detection, the keypoint centroid
(ra_kp, dec_kp) is converted to pixel coordinates via the cutout WCS,
and the PSF-weighted flux is computed as:

    f_psf = sum(I * P) / sum(P^2)

where I is the image patch and P is the normalized PSF stamp

Formula comes from https://github.com/lsst/meas_base/blob/main/include/lsst/meas/base/PsfFlux.h:

 *  @brief A measurement algorithm that estimates instFlux using a linear least-squares fit with the Psf model
 *
 *  The PsfFlux algorithm is extremely simple: we do a least-squares fit of the Psf model (evaluated
 *  at a given position) to the data.  For point sources, this p rovides the optimal instFlux measurement
 *  in the limit where the Psf model is correct.  We do not use per-pixel weights in the fit, as this
 *  results in bright stars being fit with a different effective profile than faint stairs.

Usage
-----
# Use default threshold grids:
python add_dd_mags.py --run_name lsst5_30k_4h200_bs192_ep50

# Custom threshold grids:
python add_dd_mags.py --run_name lsst5_30k_4h200_bs192_ep50 \\
    --score_thresholds 0.4 0.45 0.5 \\
    --nms_thresholds 0.3 0.35 0.4

# Single combo:
python add_dd_mags.py --run_name lsst5_30k_4h200_bs192_ep50 \\
    --score_thresholds 0.45 --nms_thresholds 0.3
    
Also for 
"""
import argparse
import gc
import itertools
import json
import logging
import os
import sys
import time
import btk
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, shared_memory
from typing import Dict, List, Optional, Tuple
from astropy.wcs import WCS
import numpy as np
import pandas as pd
import pycocotools.mask as mask_util
# allows us to import from the custom configs directory w/o affecting deepdisc library imports
sys.path.insert(0, '/u/yse2/deepdisc/configs')
from custom.transforms import LanczosResizeTransform

logger = logging.getLogger(__name__)

# Default threshold grids
DEFAULT_SCORE_THRESHOLDS = [
    0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
    0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9,
]
DEFAULT_NMS_THRESHOLDS = [
    0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75,
]
BANDS: List[str] = ['u', 'g', 'r', 'i', 'z', 'y']
N_BANDS: int = len(BANDS)
ZP: float = 27.0  # DC2 zero-point
# we're gonna pack all the images into a single contingous shared-memory block
# make sure that you have enough memory on the node (ex. 15k 6-band imgs of 160x160 is ~8GB)
# the index we'll use is file_name --> (byte_offset, shape, dtype_str)
# within that block. So, each worker will open the block once using only 1 file descriptor (FD)
# and then creates zero-copy numpy views by offset
# STAMP_SIZE: int = 35 # ~7x7
# INFO - Building PSF stamp cache (BTK LSST survey)...
# INFO -   u-band PSF: FWHM=0.9052" = 4.53px, enclosed=0.9904, sum(P^2)=0.015720
# INFO -   g-band PSF: FWHM=0.8667" = 4.33px, enclosed=0.9907, sum(P^2)=0.017036
# INFO -   r-band PSF: FWHM=0.8186" = 4.09px, enclosed=0.9909, sum(P^2)=0.018935
# INFO -   i-band PSF: FWHM=0.8003" = 4.00px, enclosed=0.9908, sum(P^2)=0.019695
# INFO -   z-band PSF: FWHM=0.7819" = 3.91px, enclosed=0.9907, sum(P^2)=0.020522
# INFO -   y-band PSF: FWHM=0.7733" = 3.87px, enclosed=0.9905, sum(P^2)=0.020895
STAMP_SIZE: int = 25  # ~5x5 arcsecs smaller stamp to reduce neighbor contamination at edges
# INFO -   u-band PSF: FWHM=0.9052" = 4.53px, enclosed=0.9829, sum(P^2)=0.015961
# INFO -   g-band PSF: FWHM=0.8667" = 4.33px, enclosed=0.9835, sum(P^2)=0.017283
# INFO -   r-band PSF: FWHM=0.8186" = 4.09px, enclosed=0.9843, sum(P^2)=0.019191
# INFO -   i-band PSF: FWHM=0.8003" = 4.00px, enclosed=0.9842, sum(P^2)=0.019959
# INFO -   z-band PSF: FWHM=0.7819" = 3.91px, enclosed=0.9842, sum(P^2)=0.020793
# INFO -   y-band PSF: FWHM=0.7733" = 3.87px, enclosed=0.9840, sum(P^2)=0.021172
HALF_STAMP: int = STAMP_SIZE // 2  # 17

# Large stamp size --> the outer pixels of I will contain flux from a neighboring obj
# evwn though the PSF model $P$ has very low values at those edges, we are still multiplying 
# neighbor's bright flux by a non-zero PSF value and adding it to our total sum. 
# So calculated flux $f_{PSF}$ will be artificially inflated by neighbor's flux and give artifically brighter mags (lower mag val) 
# not great photometry of deblended components

_shm_index: Dict[str, Tuple[int, tuple, str]] = {}
_shm_block: Optional[shared_memory.SharedMemory] = None
_psf_stamps: Optional[np.ndarray] = None    # (6, 35, 35) normalized PSF stamps
_psf_norm_sq: Optional[np.ndarray] = None   # (6,) precomputed sum(P^2) per band
# UPSAMPLE_LSST: bool = False
WCS_LOOKUP: Optional[Dict[str, Dict]] = None
    
def _init_worker(shm_name: str, shm_index: dict,
                 psf_stamps: np.ndarray, psf_norm_sq: np.ndarray,
                #  upsample_lsst: bool, 
                 wcs_lookup: dict) -> None:
    """Pool initializer: open the single SHM block and store the index"""
    global _shm_block, _shm_index, _psf_stamps, _psf_norm_sq, UPSAMPLE_LSST, WCS_LOOKUP
    _shm_block = shared_memory.SharedMemory(name=shm_name, create=False)
    _shm_index = shm_index
    _psf_stamps = psf_stamps
    _psf_norm_sq = psf_norm_sq
    # UPSAMPLE_LSST = upsample_lsst
    WCS_LOOKUP = wcs_lookup

# PSF cache builder
def build_psf_cache() -> Tuple[np.ndarray, np.ndarray]:
    """
    Render and normalize BTK LSST PSF stamps for all 6 bands
    Returns
    -------
    psf_stamps : np.ndarray
        Shape (6, 31, 31), normalized so each stamp sums to 1
    psf_norm_sq : np.ndarray
        Shape (6,), precomputed sum(P^2) for each band
    """
    survey = btk.survey.get_surveys("LSST")
    pixel_scale = survey.pixel_scale.to_value("arcsec")
    psf_stamps = np.zeros((N_BANDS, STAMP_SIZE, STAMP_SIZE), dtype=np.float64)
    psf_norm_sq = np.zeros(N_BANDS, dtype=np.float64)
    for i, band in enumerate(BANDS):
        psf = survey.get_filter(band).psf
        img = psf.drawImage(nx=STAMP_SIZE, ny=STAMP_SIZE, scale=pixel_scale)
        P = img.array.astype(np.float64)
        P /= P.sum()  # normalize so sum(P) = 1
        psf_stamps[i] = P
        psf_norm_sq[i] = np.sum(P ** 2)
        fwhm = psf.calculateFWHM()
        logger.info(
            f"  {band}-band PSF: FWHM={fwhm:.4f}\" = {fwhm/pixel_scale:.2f}px, "
            f"enclosed={img.array.sum():.4f}, sum(P^2)={psf_norm_sq[i]:.6f}"
        )
    return psf_stamps, psf_norm_sq

# ── Shared-memory registry setup and grabbing all unique imgs ────────────────────────────────────────────
def build_shm_block(
    unique_files: set,
    io_threads: int = 16,
) -> Tuple[Optional[shared_memory.SharedMemory], Dict[str, Tuple[int, tuple, str]], int]:
    def _load_npy(fn):
        try:
            img = np.load(fn)
            assert img.shape[0] == N_BANDS
            return fn, img
        except Exception as e:
            logger.error(f"Failed to load {fn}: {e}")
            return fn, None

    imgs = {}
    with ThreadPoolExecutor(max_workers=io_threads) as executor:
        for fn, img in executor.map(_load_npy, sorted(unique_files)):
            if img is not None:
                imgs[fn] = img

    total_bytes = sum(img.nbytes for img in imgs.values())
    logger.info(f"Packing {len(imgs)} images ({total_bytes / 1e6:.1f} MB) into SHM")

    shm = shared_memory.SharedMemory(create=True, size=total_bytes)
    index = {}
    offset = 0
    for fn in sorted(imgs):
        img = imgs[fn]
        np.copyto(np.ndarray(img.shape, dtype=img.dtype, buffer=shm.buf, offset=offset), img)
        index[fn] = (offset, img.shape, img.dtype.str)
        offset += img.nbytes
    del imgs
    return shm, index, total_bytes
        
def release_shm(shm: Optional[shared_memory.SharedMemory]) -> None:
    """Unlink a shared memory segment"""
    if shm is None:
        return
    try:
        shm.close()
        shm.unlink()
    except Exception:
        pass

# --- Grabbing imgs from SHM ---
def _get_img_from_shm(file_name: str) -> Optional[np.ndarray]:
    """
    Return a zero-copy read-only numpy view of an image from shared memory
    
    Uses the single pre-opened SHM block and byte-offset indexing to 
    create a numpy view of the requested image that workers can use
    to compute mags
    
    Parameters
    ----------
    file_name : str
        The image path used as key in the SHM index.

    Returns
    -------
    np.ndarray or None
        Shape (C, H, W) float32 image, or None if not found.
    """
    info = _shm_index.get(file_name)
    if info is None:
        return None
    # otherwise, we need to open the SHM segment and cache it
    offset, shape, dtype_str = info
    return np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=_shm_block.buf, offset=offset)

# ── Resume helper ────────────────────────────────────────────────────────────
def _is_combo_complete(pred_fn: str) -> bool:
    """
    Return True if pred_fn already has all mag columns with at least some
    non-NaN values.  Uses pd.read_json to match the same logic used for
    post-run integrity checks.
    """
    if not os.path.exists(pred_fn):
        return False
    try:
        df = pd.read_json(pred_fn)
        missing = [c for c in BANDS if f'mag_psf_{c}' not in df.columns]
        all_nan = [c for c in BANDS if f'mag_psf_{c}' in df.columns and df[f'mag_psf_{c}'].isna().all()]
        return len(missing) == 0 and len(all_nan) == 0
    except Exception:
        return False

# ── Magnitude calculation ──────────────────────────────────────────────────────
def _compute_mags_for_group(
    px: np.ndarray,
    py: np.ndarray,
    lsst_img: np.ndarray,
) -> np.ndarray:
    """
    Compute PSF-weighted magnitudes for all detections sharing one image
        * px, py are the pixel coordinates of the detections in the LSST image
        * lsst_img is the (C,H,W) image array from SHM for that file_name
    
        For each detection, we cut out a 35x35 patch around the centroid, 
        compute the PSF-weighted flux using the pre-rendered BTK PSF stamps,
        and convert to magnitude with the DC2 zero-point. If a patch goes
        out of bounds of the image, we pad with zeros, which can result in
        underestimated fluxes or NaN mags after thresholding at f_psf > 0
    
    Parameters
    ----------
    px : np.ndarray
        Shape (N,) x-pixel coordinates of detections.
    py : np.ndarray
        Shape (N,) y-pixel coordinates of detections.
    lsst_img : np.ndarray
        Shape (C, H, W) multi-band image.

    Returns
    -------
    np.ndarray
        Shape (N, 6) PSF magnitudes. Invalid fluxes (<=0) are NaN.
    """
    N = len(px)
    _, H, W = lsst_img.shape
    # if a centroid is too close to the edge of cutout
    # (e.g., x < 17 or x > width - 17), a 35x35 slice will go out of bounds 
    # so let's just pad the image with zeros w/ np.pad
    # We just gotta note that this zero padding dilutes the flux for edge detections
    padded = np.pad(
        lsst_img,
        ((0, 0), (HALF_STAMP, HALF_STAMP), (HALF_STAMP, HALF_STAMP)),
        mode='constant',
        constant_values=0,
    )
    mags = np.full((N, N_BANDS), np.nan, dtype=np.float64) # (N, C) — one row per detection
    # after padding, we have to shift the pixel coords by HALF_STAMP 
    # so they're in the coord system of the padded image
    # We also need to clip because of this case:
    # dets whose np.rint(px) equals W (one pixel past the last valid index), s
    # o x_idx.max() = W + STAMP_SIZE - 1 = W + 2*HALF_STAMP, 
    # which is one past the padded array's last valid index
    # so we clamp cx/cy to a valid range
    # a det w/ centroid outside cutout gets measured at the nearest edge pixel,
    # with most of its PSF stamp falling on the zero-padded region so the flux will be underestimated 
    # and likely produce a faint/NaN mag which is expected for edge dets
    cx = np.clip(np.rint(px).astype(np.int64), 0, W - 1) + HALF_STAMP
    cy = np.clip(np.rint(py).astype(np.int64), 0, H - 1) + HALF_STAMP
    # batched idx grids for STAMP_SIZE x STAMP_SIZE cutouts
    x0, y0 = cx - HALF_STAMP, cy - HALF_STAMP
    x_idx = x0[:, None] + np.arange(STAMP_SIZE)[None, :]  # (N, STAMP_SIZE) = (N, 35)
    y_idx = y0[:, None] + np.arange(STAMP_SIZE)[None, :] # (N, STAMP_SIZE) = (N, 35)
    
    for j in range(N_BANDS):
        # patches shape: (N, STAMP_SIZE, STAMP_SIZE)
        # patch = padded[j, cy - HALF_STAMP : cy + HALF_STAMP + 1,
        #                   cx - HALF_STAMP : cx + HALF_STAMP + 1]
        patches = padded[j, y_idx[:, :, None], x_idx[:, None, :]]  # (N, 35, 35)
        # quick sanity check
        if patches.shape != (N, STAMP_SIZE, STAMP_SIZE):
            raise RuntimeError(
                "Unexpected patch tensor shape: "
                f"{patches.shape}, expected "
                f"({N}, {STAMP_SIZE}, {STAMP_SIZE})"
            )
        # f_psf = np.sum(patch * _psf_stamps[j]) / _psf_norm_sq[j]
        f_psf = np.einsum('nij,ij->n', patches, _psf_stamps[j]) / _psf_norm_sq[j]
        pos = f_psf > 0
        if np.any(pos):
            mags[pos, j] = ZP - 2.5 * np.log10(f_psf[pos])
    return mags

def process_combo(combo_args: tuple) -> str:
    """
    Add PSF magnitude columns to one prediction file

    Reads the JSON, computes mags for every detection using pre-loaded
    shared-memory imgs, appends mag_psf_u..mag_psf_y columns, and writes
    to a new file with a `_psf.json` suffix.

    Parameters
    ----------
    combo_args : tuple
        (score_thresh, nms_thresh, run_dir, preds_dir)

    Returns
    -------
    str
        'Success: ...' or 'Failed ...: <error>' message.
    """
    score_thresh, nms_thresh, run_dir, preds_dir = combo_args
    label = f"s{score_thresh}_n{nms_thresh}"
    pred_fn = os.path.join(run_dir, 'preds', preds_dir, f'pred_{label}.json')
    logger.info(f"Processing combo: {label}")
    combo_start = time.perf_counter()
    try:
        dd_det_cat = pd.read_json(pred_fn).reset_index(drop=True)
        n_dets = len(dd_det_cat)
        # Pre-allocate mag array (NaN default for missing imgs)──────
        all_mags = np.full((n_dets, N_BANDS), np.nan, dtype=np.float64)
        # Compute mags per image group ───────────────────────────────────
        for file_name, group in dd_det_cat.groupby('file_name'):
            lsst_img = _get_img_from_shm(file_name)
            if lsst_img is None:
                logger.warning(f"No shared image for {file_name} — NaN mags")
                continue
            wcs_lsst = WCS(WCS_LOOKUP[file_name]['wcs']) # wcs_roman = None
            ra_kps = group['ra_kp'].values.astype(np.float64)
            dec_kps = group['dec_kp'].values.astype(np.float64)
            px, py = wcs_lsst.wcs_world2pix(ra_kps, dec_kps, 0)
            if not (np.all(np.isfinite(px)) and np.all(np.isfinite(py))):
                raise ValueError("All detections must have finite world-to-pixel coordinates")
            assert len(px) == len(py) == len(ra_kps), "All detections must have x, y"            
            # if UPSAMPLE_LSST:
            #     cache_file_name = file_name.replace("/u/","/work/hdd/bdsp/")
            #     cache_dir = '/work/hdd/bfhm/g4merz/wcs_map_cache/val_4k_keypoints_wcs' if preds_dir == 'eval' else '/work/hdd/bfhm/g4merz/wcs_map_cache/test_8k_keypoints_wcs'
            #     wcs_roman = WCS(WCS_LOOKUP[file_name]['wcs_roman'])
            #     # SHM image is C,H,W -> H,W,C for transform API
            #     lsst_hwc = np.ascontiguousarray(lsst_img.transpose(1, 2, 0))
            #     tfm = LanczosResizeTransform(
            #         h=lsst_hwc.shape[0],
            #         w=lsst_hwc.shape[1],
            #         new_h=512,
            #         new_w=512,
            #         wcs_rubin=wcs_lsst,
            #         wcs_roman=wcs_roman,
            #         rubin_fns=cache_file_name,
            #         roman_fns=cache_file_name.replace('lsst_data', 'truth-roman').replace('/truth/', '/'),
            #         cache_dir=cache_dir,
            #     )
            #     lsst_512_hwc = tfm.apply_image(lsst_hwc)   # (512,512,6)
            #     coords_512 = tfm.apply_coords(np.stack([px, py], axis=1)) # (N, 2) in 512x512 space
            #     px, py = coords_512[:, 0], coords_512[:, 1]
            #     lsst_img = np.ascontiguousarray(lsst_512_hwc.transpose(2, 0, 1))  # (6,512,512)
            try:
                mags = _compute_mags_for_group(px, py, lsst_img)
                all_mags[group.index.values] = mags
            except Exception as e:
                logger.warning(f"PSF magnitude computation failed for {file_name} : {e}")

        # Assign mag columns to dd_det_cat
        for j, band in enumerate(BANDS):
            dd_det_cat[f'mag_psf_{band}'] = all_mags[:, j]
        # if all_mags is all Nans, then don't overwrite the file, since it means something went wrong with SHM access 
        # But if we have at least some valid mags, then we can overwrite the file with the new columns added.
        if not np.all(np.isnan(all_mags)):
            # dd_det_cat.to_json(pred_fn.replace('eval', 'eval/mags'))
            # dd_det_cat.to_json(pred_fn)
            dd_det_cat.to_json(pred_fn.replace('.json', '_psf_25.json'))
        else:
            logger.warning(f"Skipping file write for {pred_fn} — all NaN mags")

        elapsed = time.perf_counter() - combo_start
        logger.info(f"DONE {label} in {elapsed:.3f}s ({n_dets} detections)")
        return f"Success: {label}"

    except Exception as e:
        logger.error(f"X Failed {label}: {e}", exc_info=True)
        return f"Failed {label}: {e}"

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    overall_start = time.perf_counter()
    logger.info("Starting script")
    parser = argparse.ArgumentParser(
        description='Add PSF magnitudes to DeepDISC prediction files and save as *_psf.json',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--root_run_dir', type=str, default='~/lsst_runs/',
    )
    parser.add_argument(
        '--preds_dir', type=str, default='eval',
        help='Subdirectory under preds/ containing prediction files',
    )
    parser.add_argument(
        '--run_name', type=str, required=True,
        help='Run name (subfolder under root_run_dir)',
    )
    parser.add_argument(
        '--io_threads', type=int, default=16,
        help='Threads for parallel image loading into SHM',
    )
    # t = pd.read_json('lsst_data/annotations_lvl5/val_4k_keypoints.json')
    # uniq = set(t['file_name'])
    # pd.Series(sorted(uniq), name="file_name").to_csv("val4k_fns.csv", index=False)
    # use the above code to create val4k_fns.csv from anns file
    # if wcs is also needed (for combined LSST+Roman data), then you need to do the below:
    # t = pd.read_json('lsst_data/annotations_lvl5/val_4k_keypoints_wcs.json')
    # meta = (
    #     t[['file_name', 'wcs', 'wcs_roman']]
    #     .drop_duplicates(subset=['file_name'])
    #     .sort_values('file_name')
    # )
    # meta.to_csv('val4k_fns_wcs.csv', index=False)
    parser.add_argument(
        '--filenames_csv', type=str, default='~/val4k_fns_wcs.csv',
        help='Path to CSV with a file_name column containing all image fns that inference was done for',
    )
    parser.add_argument(
        '--score_thresholds', type=float, nargs='+',
        default=DEFAULT_SCORE_THRESHOLDS,
    )
    parser.add_argument(
        '--nms_thresholds', type=float, nargs='+',
        default=DEFAULT_NMS_THRESHOLDS,
    )
    parser.add_argument(
        '--no_combo', action='store_true',
        help='Pair score and NMS thresholds by index instead of full cartesian product',
    )
    parser.add_argument(
        '--n_processes', type=int, default=None,
        help='Worker processes (default: all available CPUs)',
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='Skip combos whose prediction file already contains mag columns',
    )
    args = parser.parse_args()
    root_run_dir = os.path.expanduser(args.root_run_dir)
    run_dir = os.path.join(root_run_dir, args.run_name)
    n_processes = args.n_processes or len(os.sched_getaffinity(0))
    if not os.path.isdir(run_dir):
        logger.error(f"Run directory does not exist: {run_dir}")
        return

    if len(args.score_thresholds) == 0 or len(args.nms_thresholds) == 0:
        logger.error("Score and NMS threshold lists cannot be empty")
        sys.exit(1)

    if args.no_combo:
        if len(args.score_thresholds) != len(args.nms_thresholds):
            logger.error(
                "--no_combo requires equal list lengths: "
                f"{len(args.score_thresholds)} score values vs "
                f"{len(args.nms_thresholds)} nms values"
            )
            sys.exit(1)
        threshold_combos = list(zip(args.score_thresholds, args.nms_thresholds))
        threshold_mode = 'index-paired'
    else:
        threshold_combos = list(
            itertools.product(args.score_thresholds, args.nms_thresholds)
        )
        threshold_mode = 'full cartesian product'

    combo_args = [(s, n, run_dir, args.preds_dir) for s, n in threshold_combos]

    if args.resume:
        preds_base = os.path.join(run_dir, 'preds', args.preds_dir)
        remaining = []
        skipped = 0
        for ca in combo_args:
            s, n, _, _ = ca
            label = f"s{s}_n{n}"
            pred_fn = os.path.join(preds_base, f'pred_{label}.json')
            psf_pred_fn = pred_fn.replace('.json', '_psf.json')
            if os.path.exists(psf_pred_fn) and _is_combo_complete(psf_pred_fn):
                logger.info(f"Skipping (already done): {label}")
                skipped += 1
            else:
                remaining.append(ca)
        logger.info(f"Resume: skipping {skipped} completed combos, {len(remaining)} remaining")
        combo_args = remaining

    logger.info(f"Run dir:          {run_dir}")
    logger.info(f"Threshold mode:   {threshold_mode}")
    logger.info(f"Threshold combos: {len(combo_args)}")
    logger.info(f"Worker processes: {n_processes}")

    if len(combo_args) == 0:
        logger.info("No remaining combos to process. Exiting before SHM setup.")
        logger.info(f"Finished in {time.perf_counter() - overall_start:.3f}s total")
        return

    # build PSF cache once and share across workers
    logger.info("Building PSF stamp cache (BTK LSST survey)...")
    psf_stamps, psf_norm_sq = build_psf_cache()
    logger.info(f"PSF cache ready: {STAMP_SIZE}x{STAMP_SIZE} stamps for {N_BANDS} bands")

    # ── Phase 1: Pre-load all images into shared memory ──────────────────
    shm_load_start = time.perf_counter()
    csv_path = os.path.expanduser(args.filenames_csv)
    df = pd.read_csv(csv_path)
    unique_files = set(df['file_name'].tolist())
    logger.info(f"Loaded {len(unique_files)} filenames from {csv_path}")
    shm_block, shm_index, total_bytes = build_shm_block(
        unique_files, io_threads=args.io_threads
    )
    if shm_block is None:
        logger.error("Shared memory setup failed — aborting")
        return
    
    total_mb = total_bytes / 1e6
    logger.info(f"Shared memory ready: {len(shm_index)} images loaded in {time.perf_counter() - shm_load_start:.3f}s ({total_mb:.1f} MB total)")
    
    wcs_lookup = df.set_index('file_name').to_dict('index')
    logger.info(f"WCS lookup built for {len(wcs_lookup)} images")
    
    # ── Phase 2: Distribute combo work across workers ────────────────────
    try:
        pool_start = time.perf_counter()
        with Pool(
            processes=n_processes,
            initializer=_init_worker,
            initargs=(shm_block.name, shm_index, psf_stamps, psf_norm_sq, wcs_lookup),
        ) as pool:
            results = list(pool.imap_unordered(process_combo, combo_args))
        pool_time = time.perf_counter() - pool_start
    finally:
        release_shm(shm_block)
        logger.info("Shared memory released")

    successful = sum(1 for r in results if r.startswith("Success"))
    failed = [r for r in results if r.startswith("Failed")]

    logger.info(f"Processing complete: {successful}/{len(combo_args)} successful in {pool_time:.3f}s")
    if failed:
        logger.warning(f"Failed combos ({len(failed)}):")
        for msg in failed:
            logger.warning(f"  {msg}")

    gc.collect()
    logger.info(f"Finished in {time.perf_counter() - overall_start:.3f}s total")

if __name__ == "__main__":
    main()