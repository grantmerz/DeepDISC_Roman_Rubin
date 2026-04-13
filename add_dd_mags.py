"""
Add magnitudes to DeepDISC prediction files for a given run

For every (score_thresh, nms_thresh) combination, reads the existing
prediction file at:

    <run_dir>/preds/eval/pred_s{score}_n{nms}.json

calculates instrumental magnitudes for all bands (u, g, r, i, z, y)
using a zero-point of 27 (DC2 coadds), and overwrites the file in-place
with the mag columns appended

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
import json, ijson
import logging
import os
import sys
import time
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
_shm_index: Dict[str, Tuple[int, tuple, str]] = {}
_shm_block: Optional[shared_memory.SharedMemory] = None
UPSAMPLE_LSST: bool = False
WCS_LOOKUP: Optional[Dict[str, Dict]] = None  # populated if UPSAMPLE_LSST is True
    
def _init_worker(shm_name: str, shm_index: dict,
                 upsample_lsst: bool, wcs_lookup: dict) -> None:
    """Pool initializer: open the single SHM block and store the index"""
    global _shm_block, _shm_index, UPSAMPLE_LSST, WCS_LOOKUP
    _shm_block = shared_memory.SharedMemory(name=shm_name, create=False)
    _shm_index = shm_index
    UPSAMPLE_LSST = upsample_lsst
    WCS_LOOKUP = wcs_lookup

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
        missing = [c for c in BANDS if f'mag_{c}' not in df.columns]
        all_nan = [c for c in BANDS if f'mag_{c}' in df.columns and df[f'mag_{c}'].isna().all()]
        return len(missing) == 0 and len(all_nan) == 0
    except Exception:
        return False

# ── Magnitude calculation ──────────────────────────────────────────────────────
def _compute_mags_for_group(
    rle_masks: List[dict],
    lsst_img: np.ndarray,
) -> np.ndarray:
    """
    Compute instrumental magnitudes for all detections sharing one image

    Uses vectorized batch RLE decode and a single matrix multiply to
    compute flux in all bands simultaneously.  Returns a plain numpy
    array to avoid per-group DataFrame overhead

    Parameters
    ----------
    rle_masks : list of dict
        RLE mask dicts (with str 'counts') for each detection.
    lsst_img : np.ndarray
        Shape (C, H, W) multi-band image.

    Returns
    -------
    np.ndarray
        Shape (N, 6) magnitudes array (one row per detection, one col
        per band).  Invalid fluxes (<=0) are NaN.
    """
    # Encode RLE counts to bytes for pycocotools
    rle_dicts = [
        {'size': r['size'], 'counts': r['counts'].encode('utf-8')}
        for r in rle_masks
    ]
    # Batch decode: returns uint8 (H, W, N) and convert to float32 
    masks = mask_util.decode(rle_dicts).astype(np.float32)  # (H, W, N)
    H, W, N = masks.shape
    # (C, H * W) @ (H * W, N) => (C, N)
    fluxes = lsst_img.reshape(N_BANDS, H * W) @ masks.reshape(H * W, N)
    # Convert flux to magnitudes: mag = ZP - 2.5 * log10(flux)
    with np.errstate(divide='ignore', invalid='ignore'):
        safe_flux = np.where(fluxes > 0, fluxes, np.nan)
        mags = ZP - 2.5 * np.log10(safe_flux)  # (C, N)
    return mags.T  # (N, C) — one row per detection

def process_combo(combo_args: tuple) -> str:
    """
    Add magnitude columns to one prediction file in-place

    Reads the JSON, computes mags for every detection using pre-loaded
    shared-memory imgs, appends mag_u..mag_y columns, and writes
    the file back

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
    pred_fn = os.path.join(
        run_dir, 'preds', preds_dir, f'pred_{label}.json'
    )
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
            if UPSAMPLE_LSST:
                cache_file_name = file_name.replace("/u/","/work/hdd/bdsp/")
                cache_dir = '/work/hdd/bfhm/g4merz/wcs_map_cache/val_4k_keypoints_wcs' if preds_dir == 'eval' else '/work/hdd/bfhm/g4merz/wcs_map_cache/test_8k_keypoints_wcs'
                wcs_lsst, wcs_roman = WCS(WCS_LOOKUP[file_name]['wcs']), WCS(WCS_LOOKUP[file_name]['wcs_roman'])
                # SHM image is C,H,W -> H,W,C for transform API
                lsst_hwc = np.ascontiguousarray(lsst_img.transpose(1, 2, 0))
                tfm = LanczosResizeTransform(
                    h=lsst_hwc.shape[0],
                    w=lsst_hwc.shape[1],
                    new_h=512,
                    new_w=512,
                    wcs_rubin=wcs_lsst,
                    wcs_roman=wcs_roman,
                    rubin_fns=cache_file_name,
                    roman_fns=cache_file_name.replace('lsst_data', 'truth-roman').replace('/truth/', '/'),
                    cache_dir=cache_dir,
                )
                lsst_512_hwc = tfm.apply_image(lsst_hwc)   # (512,512,6)
                lsst_img = np.ascontiguousarray(lsst_512_hwc.transpose(2, 0, 1))  # (6,512,512)
            try:
                mags = _compute_mags_for_group(
                    group['rle_masks'].tolist(), lsst_img
                )
                all_mags[group.index.values] = mags
            except Exception as e:
                logger.warning(f"Mag computation failed for {file_name} : {e}")

        # Assign mag columns to dd_det_cat
        for j, band in enumerate(BANDS):
            dd_det_cat[f'mag_{band}'] = all_mags[:, j]
        # if all_mags is all Nans, then don't overwrite the file, since it means something went wrong with SHM access 
        # But if we have at least some valid mags, then we can overwrite the file with the new columns added.
        if not np.all(np.isnan(all_mags)):
            # dd_det_cat.to_json(pred_fn.replace('eval', 'eval/mags'))
            dd_det_cat.to_json(pred_fn)
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
        description='Add magnitudes to DeepDISC prediction files in-place',
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
        '--filenames_csv', type=str, default='~/val4k_fns.csv',
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
    parser.add_argument(
        '--upsample', action='store_true', default=UPSAMPLE_LSST,
        help='Whether to upsample LSST imgs to 512x512 using LanczosResize before mag comp  \
        Needed for combined LSST+Roman models w/ WCS-based transforms'
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
            if _is_combo_complete(pred_fn):
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

    # ── Phase 1: Pre-load all images into shared memory ──────────────────
    shm_load_start = time.perf_counter()    
    csv_path = os.path.expanduser(args.filenames_csv)
    unique_files = set(pd.read_csv(csv_path, usecols=['file_name'])['file_name'].tolist())
    logger.info(f"Loaded {len(unique_files)} filenames from {csv_path}")
    shm_block, shm_index, total_bytes = build_shm_block(
        unique_files, io_threads=args.io_threads
    )
    # shm_block, shm_index, total_bytes = build_shm_block(
    #     combo_args, io_threads=args.io_threads
    # )
    if shm_block is None:
        logger.error("Shared memory setup failed — aborting")
        return
    
    total_mb = total_bytes / 1e6
    logger.info(f"Shared memory ready: {len(shm_index)} images loaded in {time.perf_counter() - shm_load_start:.3f}s ({total_mb:.1f} MB total)")
    
    logger.info(f"Upsampling {'enabled' if args.upsample else 'disabled'} for LSST images to 512x512 with LanczosResizeTransform")
    wcs_lookup = pd.read_csv(csv_path).set_index('file_name').to_dict('index') if args.upsample else None
    
    # ── Phase 2: Distribute combo work across workers ────────────────────
    try:
        pool_start = time.perf_counter()
        with Pool(
            processes=n_processes,
            initializer=_init_worker,
            initargs=(shm_block.name, shm_index, args.upsample, wcs_lookup),
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