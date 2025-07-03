import re, os, cv2, time, gc
import warnings
import requests
import logging

import numpy as np
import pandas as pd

from pathlib import Path
from urllib.parse import urljoin
from collections import Counter
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.table import Table
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord

from detectron2.structures import BoxMode
from deepdisc.data_format.conversions import convert_to_json

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_ra_dec(filename):
    matches = re.search(r'dc2_(det|seg|index)_(\d+\.\d+)_(\-\d+\.\d+)', filename)
    if matches:
        return matches.group(2), matches.group(3)
    return None

def create_multiband_coadd(ra_dec):
    target_file = f"roman_data/truth/{ra_dec[0]}_{ra_dec[1]}/full_{ra_dec[0]}_{ra_dec[1]}.npy"
    # we still load in f184_img so we can grab the wcs
    f184_img = fits.open(f'roman_data/original_fits/{ra_dec[0]}_{ra_dec[1]}/F184_{ra_dec[0]}_{ra_dec[1]}.fits')
    w = WCS(f184_img[1].header)
    if os.path.exists(target_file):
        print(f"Multiband coadd for {ra_dec[0]}_{ra_dec[1]} has already been created!")
        full_img_data = np.load(target_file)
        return full_img_data, w
    h158_img = fits.open(f'roman_data/original_fits/{ra_dec[0]}_{ra_dec[1]}/H158_{ra_dec[0]}_{ra_dec[1]}.fits')
    y106_img = fits.open(f'roman_data/original_fits/{ra_dec[0]}_{ra_dec[1]}/Y106_{ra_dec[0]}_{ra_dec[1]}.fits')
    j129_img = fits.open(f'roman_data/original_fits/{ra_dec[0]}_{ra_dec[1]}/J129_{ra_dec[0]}_{ra_dec[1]}.fits')
    full_img_data = np.stack((f184_img[1].data, h158_img[1].data, y106_img[1].data, j129_img[1].data))
    
    np.save(target_file, full_img_data)    
    
    f184_img.close()
    h158_img.close()
    y106_img.close()
    j129_img.close()
    
    print(f"Multiband coadd saved to {target_file}")
    return full_img_data, w

def create_cutouts(seg_file, full_img_data, ra_dec, w):
    seg = fits.open(seg_file)
    cutout_size = 512
    overlap_pixels = 500
    coadd_size = seg[0].data.shape
    # only use the core region avoiding 500px overlap on each edge
    usable_width = coadd_size[1] - 2 * overlap_pixels  # 7825
    usable_height = coadd_size[0] - 2 * overlap_pixels  # 7825
    start_x, start_y = overlap_pixels, overlap_pixels  # 500, 500
    
    nx_cutouts = usable_width // cutout_size
    ny_cutouts = usable_height // cutout_size
        
    # calc spacing to distribute cutouts evenly
    if nx_cutouts > 1:
        x_spacing = (usable_width - cutout_size) // (nx_cutouts - 1)
    else:
        x_spacing = 0

    if ny_cutouts > 1:
        y_spacing = (usable_height - cutout_size) // (ny_cutouts - 1)
    else:
        y_spacing = 0
    
    print(f"Creating cutouts for {ra_dec[0]}_{ra_dec[1]}...")
    print(f"Image size: {coadd_size[1]}x{coadd_size[0]}")
    print(f"Usable region: {usable_width}x{usable_height} (starting at {start_x},{start_y})")
    print(f"Cutouts: {nx_cutouts}x{ny_cutouts} = {nx_cutouts * ny_cutouts} total")
    print(f"Spacing: x={x_spacing}, y={y_spacing}")
    
    counter = 0
    seg_cutouts = []
    for i in range(ny_cutouts):
        for j in range(nx_cutouts):
           # cutout center pos
            if ny_cutouts == 1:
                y_center = start_y + cutout_size // 2
            else:
                y_center = start_y + cutout_size // 2 + i * y_spacing
            
            if nx_cutouts == 1:
                x_center = start_x + cutout_size // 2
            else:
                x_center = start_x + cutout_size // 2 + j * x_spacing
            
            seg_cutout = Cutout2D(seg[0].data, position=(x_center, y_center), 
                                size=cutout_size, wcs=w, mode='partial', fill_value=0)
            seg_cutouts.append(seg_cutout)
            
            full_cutout_path = f'roman_data/truth/{ra_dec[0]}_{ra_dec[1]}/full_c{counter}_{ra_dec[0]}_{ra_dec[1]}.npy'
            if not os.path.exists(full_cutout_path):
                raw_cutout_f184 = Cutout2D(full_img_data[0], position=(x_center, y_center), 
                                         size=cutout_size, wcs=w, mode='partial', fill_value=0)
                raw_cutout_h158 = Cutout2D(full_img_data[1], position=(x_center, y_center), 
                                         size=cutout_size, wcs=w, mode='partial', fill_value=0)
                raw_cutout_y106 = Cutout2D(full_img_data[2], position=(x_center, y_center), 
                                         size=cutout_size, wcs=w, mode='partial', fill_value=0)
                raw_cutout_j129 = Cutout2D(full_img_data[3], position=(x_center, y_center), 
                                         size=cutout_size, wcs=w, mode='partial', fill_value=0)
                full_raw_cutout = np.stack((raw_cutout_f184.data, raw_cutout_h158.data, 
                                          raw_cutout_y106.data, raw_cutout_j129.data)) 
                np.save(full_cutout_path, full_raw_cutout)
            
            # debug info for first few cutouts
            if counter < 3:
                print(f"  Cutout {counter}: center=({x_center}, {y_center}), "
                      f"bbox=({x_center-cutout_size//2}, {y_center-cutout_size//2}, "
                      f"{x_center+cutout_size//2}, {y_center+cutout_size//2})")
            
            counter += 1
    seg.close()
    print(f"Created {len(seg_cutouts)} cutouts")
    return seg_cutouts

def get_cutout_truth_cat(seg_cutouts, truth_df, ra_dec, verbose=False):
    """
    We obtain a subset of the truth catalog for each cutout by first converting the 4 corners of the 512x512 cutout to RAs/DECs and then
    filtering the truth catalog to only objects within those RA/Dec ranges. For convience, we also convert the filtered objects' coordinates 
    to cutout pixel coords. If no objs in cutout, file still gets saved for consistency.   
    """
    truth_coords = SkyCoord(ra=truth_df['ra'].values*u.degree, dec=truth_df['dec'].values*u.degree)
    truth_ras = truth_coords.ra.degree
    truth_decs = truth_coords.dec.degree
    # 4 corners of the 512x512 cutout
    corners_pix = np.array([
        [0, 0],        # bottom left
        [511, 0],      # bottom right  
        [511, 511],    # top right
        [0, 511]       # top left
    ])
    
    cutout_truth_filenames = []
    cutout_truths = []

    for imgid, seg_cutout in enumerate(seg_cutouts):
        truth_cutout_path = f'roman_data/truth/{ra_dec[0]}_{ra_dec[1]}/truth_c{imgid}_{ra_dec[0]}_{ra_dec[1]}.json'
        if not os.path.exists(truth_cutout_path):
            corners_world = seg_cutout.wcs.pixel_to_world(corners_pix[:, 0], corners_pix[:, 1])
            corner_ras = [coord.ra.degree for coord in corners_world]
            corner_decs = [coord.dec.degree for coord in corners_world]
            ra_min = min(corner_ras)
            ra_max = max(corner_ras)
            dec_min = min(corner_decs)
            dec_max = max(corner_decs)

            if verbose:
                print(f"Cutout {imgid} boundaries:")
                print(f"  RA: {ra_min:.6f}° to {ra_max:.6f}° (span: {(ra_max-ra_min)*3600:.2f}\")")
                print(f"  Dec: {dec_min:.6f}° to {dec_max:.6f}° (span: {(dec_max-dec_min)*3600:.2f}\")")
                break

            ra_mask = (truth_ras >= ra_min) & (truth_ras <= ra_max)
            dec_mask = (truth_decs >= dec_min) & (truth_decs <= dec_max)
            within_bounds = ra_mask & dec_mask
            if not np.any(within_bounds):
                print(f"No truth objects found within cutout {imgid}")
                cutout_truth = pd.DataFrame(columns=[
                    *truth_df.columns,
                    'cutout_x', 'cutout_y', 'cutout_id'
                ])
            else:
                cutout_truth = truth_df[within_bounds].copy()

                cutout_truth_coords = truth_coords[within_bounds]
                pix_coords = seg_cutout.wcs.world_to_pixel(cutout_truth_coords)
                cutout_truth['cutout_x'] = pix_coords[0]
                cutout_truth['cutout_y'] = pix_coords[1]
                cutout_truth['cutout_id'] = imgid

                if verbose:
                    print(f"  Found {len(cutout_truth)} truth objects within cutout {imgid} boundaries")
                    print(f"  Pixel coordinate ranges:")
                    print(f"    X: {cutout_truth['cutout_x'].min():.2f} to {cutout_truth['cutout_x'].max():.2f}")
                    print(f"    Y: {cutout_truth['cutout_y'].min():.2f} to {cutout_truth['cutout_y'].max():.2f}")
                    break

            cutout_truth.to_json(truth_cutout_path, orient='records')
        else:
            cutout_truth = pd.read_json(truth_cutout_path)
        
        cutout_truth_filenames.append(truth_cutout_path)
        cutout_truths.append(cutout_truth)
        # break
    
    total_objects = sum(len(result) for result in cutout_truths)
    non_empty_cutouts = sum(1 for result in cutout_truths if len(result) > 0)
    empty_cutouts = len(cutout_truths) - non_empty_cutouts 
    empty_cutout_ids = [imgid for imgid, cutout_truth in enumerate(cutout_truths) if len(cutout_truth) == 0]

    print(f"Completed truth catalog processing for {len(seg_cutouts)} cutouts")
    print(f"  Total truth objects assigned: {total_objects}")
    print(f"  Non-empty cutouts: {non_empty_cutouts}")
    print(f"  Empty cutouts: {empty_cutouts} ({empty_cutout_ids})")
    print(f"  Avg num of truth objects per non-empty cutout: {total_objects/non_empty_cutouts if non_empty_cutouts > 0 else 0:.1f}")
   
    return cutout_truth_filenames, cutout_truths

def get_cutout_det_cat(seg_cutouts, det_df, ra_dec, verbose=False):
    """
    We obtain a subset of the det catalog for each cutout from the segmentation map and indexing into the detection catalog with those ids
    For convience, we also convert the det objects' coords to cutout pixel coords.
    """
    cutout_det_filenames = []
    cutout_dets = []
    corners_pix = np.array([
        [0, 0],        # bottom left
        [511, 0],      # bottom right  
        [511, 511],    # top right
        [0, 511]       # top left
    ])
    for imgid, seg_cutout in enumerate(seg_cutouts):
        det_cutout_path = f'roman_data/truth/{ra_dec[0]}_{ra_dec[1]}/det_c{imgid}_{ra_dec[0]}_{ra_dec[1]}.json'
        if not os.path.exists(det_cutout_path):
            # unique objs from segm cutout
            seg_objs = []
            seg_img_cut = seg_cutout.data
            for s in np.unique(seg_img_cut):
                if s == 0:  # background
                    continue
                seg_objs.append(s)
            seg_objs = np.asarray(seg_objs)

            if len(seg_objs) == 0:
                print(f"No detected objects in this cutout {imgid}")
                cutout_det = pd.DataFrame(columns=[
                    *det_df.columns,
                    'cutout_x', 'cutout_y', 'seg_id', 'cutout_id'
                ])
            else:
                det_objs = det_df.iloc[seg_objs-1].copy()  # -1 because segmentation IDs are 1-indexed and we match to number col in det df
                det_coords = SkyCoord(ra=det_objs['alphawin_j2000'].values*u.degree, 
                                     dec=det_objs['deltawin_j2000'].values*u.degree)
                det_ras = det_coords.ra.degree
                det_decs = det_coords.dec.degree

                # now we further filter the det coords by ensuring that we only take the objects whose center ra/dec is within the cutout
                # some objs are detected in the segmentation map but their center ra/dec is outside of the cutout so we choose to exclude
                # these objects despite there being a mask for them since we want to stay consistent with the filtering for truth catalog
                corners_world = seg_cutout.wcs.pixel_to_world(corners_pix[:, 0], corners_pix[:, 1])
                corner_ras = [coord.ra.degree for coord in corners_world]
                corner_decs = [coord.dec.degree for coord in corners_world]
                ra_min = min(corner_ras)
                ra_max = max(corner_ras)
                dec_min = min(corner_decs)
                dec_max = max(corner_decs)

                ra_mask = (det_ras >= ra_min) & (det_ras <= ra_max)
                dec_mask = (det_decs >= dec_min) & (det_decs <= dec_max)
                within_bounds = ra_mask & dec_mask
                if not np.any(within_bounds):
                    print(f"No det objects found within cutout {imgid} after filtering by cutout boundaries")
                    cutout_det = pd.DataFrame(columns=[
                        *det_df.columns,
                        'cutout_x', 'cutout_y', 'seg_id', 'cutout_id'
                    ])
                else:
                    cutout_det = det_objs[within_bounds].copy()
                    cutout_det_coords = det_coords[within_bounds]
                    cutout_seg_objs = seg_objs[within_bounds]
                    pix_coords = seg_cutout.wcs.world_to_pixel(cutout_det_coords)
                    cutout_det['cutout_x'] = pix_coords[0]
                    cutout_det['cutout_y'] = pix_coords[1]
                    cutout_det['seg_id'] = cutout_seg_objs
                    cutout_det['cutout_id'] = imgid
            
            cutout_det.to_json(det_cutout_path, orient='records')
        else:
            cutout_det = pd.read_json(det_cutout_path)
            
        cutout_det_filenames.append(det_cutout_path)
        cutout_dets.append(cutout_det)
#         break
    total_objects = sum(len(result) for result in cutout_dets)
    non_empty_cutouts = sum(1 for result in cutout_dets if len(result) > 0)
    empty_cutouts = len(cutout_dets) - non_empty_cutouts
    empty_cutout_ids = [imgid for imgid, cutout_det in enumerate(cutout_dets) if len(cutout_det) == 0]
    
    print(f"Completed det catalog processing for {len(seg_cutouts)} cutouts")
    print(f"  Total det objects assigned: {total_objects}")
    print(f"  Non-empty cutouts: {non_empty_cutouts}")
    print(f"  Empty cutouts: {empty_cutouts} ({empty_cutout_ids})")
    print(f"  Avg num of det objects per non-empty cutout: {total_objects/non_empty_cutouts if non_empty_cutouts > 0 else 0:.1f}")
    return cutout_det_filenames, cutout_dets

def cross_match_objects(det_df, truth_df, max_sep_arcsec=0.0575):
    """
    Cross-match detection catalog and truth catalog for a cutout
    """
    if len(det_df) == 0 or len(truth_df) == 0:
        return pd.DataFrame(), {}
    
    det_coords_np = np.column_stack([det_df['alphawin_j2000'].values, 
                                 det_df['deltawin_j2000'].values])
    truth_coords_np = np.column_stack([truth_df['ra'].values, 
                                   truth_df['dec'].values])
    # vectorized coord creation
    det_coords = SkyCoord(ra=det_coords_np[:, 0]*u.degree, 
                            dec=det_coords_np[:, 1]*u.degree)
    truth_coords = SkyCoord(ra=truth_coords_np[:, 0]*u.degree, 
                              dec=truth_coords_np[:, 1]*u.degree)
    
    # 0.0575 same as coadd pixel scale from https://academic.oup.com/mnras/article/522/2/2801/7076879?login=false
    max_sep = max_sep_arcsec * u.arcsec
    idx_truth, d2d, d3d = det_coords.match_to_catalog_sky(truth_coords)
    sep_constraint = d2d <= max_sep
    
    if not np.any(sep_constraint):
        print(f"Detection catalog has objects but none of them match with the truth objects within {max_sep} arcsecs")
        return pd.DataFrame(), {}
    
    matched_objs = pd.DataFrame()
    seg_truth_mapping = {}
    
    if np.any(sep_constraint):
        matched_det_idxs = np.where(sep_constraint)[0]
        matched_truth_idxs = idx_truth[sep_constraint]
        matched_seps = d2d[sep_constraint]
        
        unique_truth_idxs, counts = np.unique(matched_truth_idxs, return_counts=True)
        
        duplicate_truth_idxs = unique_truth_idxs[counts > 1]
        
        if len(duplicate_truth_idxs) > 0:
            print(f"Found {len(duplicate_truth_idxs)} truth objects matched by multiple detections:")
            for truth_idx in duplicate_truth_idxs:
                det_matches = matched_det_idxs[matched_truth_idxs == truth_idx]
                seps = matched_seps[matched_truth_idxs == truth_idx]
                print(f"  Truth index {truth_idx}: matched by {len(det_matches)} detections")
                for i, sep in enumerate(seps.to(u.arcsec).value):
                    print(f" Separation for det {det_matches[i]} : {sep:.4f} arcsec")
        
        final_det_idxs = []
        final_truth_idxs = []
        final_seps = []
        n_competing_dets = []
        match_quality_flags = []
    
        for truth_idx in unique_truth_idxs:
            # all dets matching this truth obj
            matching_det_mask = matched_truth_idxs == truth_idx
            matching_det_idxs = matched_det_idxs[matching_det_mask]
            matching_seps = matched_seps[matching_det_mask]
            # how many dets competed for this truth obj
            n_competitors = len(matching_det_idxs)

            # keeping closest match
            closest_idx = np.argmin(matching_seps)
            chosen_det_idx = matching_det_idxs[closest_idx]
            chosen_sep = matching_seps[closest_idx]

            final_det_idxs.append(chosen_det_idx)
            final_truth_idxs.append(truth_idx)
            final_seps.append(chosen_sep.to(u.arcsec).value)
            n_competing_dets.append(n_competitors)
            if n_competitors == 1:
                match_quality_flags.append('unique')
            else:
                match_quality_flags.append('closest_of_multiple')
        
        final_det_idxs = np.array(final_det_idxs)
        final_truth_idxs = np.array(final_truth_idxs)
        final_seps = np.array(final_seps)
        n_competing_dets = np.array(n_competing_dets) 
        
        matched_dets = det_df.iloc[final_det_idxs].copy().reset_index(drop=True)
        matched_truths = truth_df.iloc[final_truth_idxs].copy().reset_index(drop=True)
        
        matched_objs = matched_dets.copy()
        for col in matched_truths.columns:
            if col not in ['cutout_id', 'cutout_x', 'cutout_y']:
                matched_objs[f'{col}'] = matched_truths[col].values
            elif col in ['cutout_x', 'cutout_y']:
                matched_objs[f'truth_{col}'] = matched_truths[col].values
            
        matched_objs['sep_arcsec'] = [sep for sep in final_seps]
        matched_objs['sep_pixels'] = matched_objs['sep_arcsec'] / 0.0575  # roman pixel scale
        matched_objs['n_competing_dets'] = n_competing_dets
        matched_objs['is_ambiguous_match'] = n_competing_dets > 1
        matched_objs['match_quality'] = match_quality_flags
        
        # mapping seg ID to truth classification and other truth info
        for i, (det_idx, truth_idx) in enumerate(zip(final_det_idxs, final_truth_idxs)):
            seg_id = det_df.iloc[det_idx]['seg_id']
            truth_row = truth_df.iloc[truth_idx]
            seg_truth_mapping[seg_id] = (int(truth_row['gal_star']), {
                "mag_F184": truth_row['mag_F184'],
                "mag_H158": truth_row['mag_H158'],
                "mag_J129": truth_row['mag_J129'],
                "mag_Y106": truth_row['mag_Y106'],
                "ra": truth_row['ra'],
                "dec": truth_row['dec'],
                "sep_arcsec": final_seps[i],
                "n_competing_dets": int(n_competing_dets[i]),
                "match_quality": match_quality_flags[i]
            })
    
    n_unique = np.sum(np.array(match_quality_flags) == 'unique')
    n_ambiguous = np.sum(np.array(match_quality_flags) == 'closest_of_multiple')
    median_sep = np.median([sep for sep in final_seps])

#     print(f"Match quality summary:")
#     print(f"  Unique matches: {n_unique}")
#     print(f"  Ambiguous matches (closest selected): {n_ambiguous}")
#     print(f"  Median separation: {median_sep:.4f} arcsec")
#     print(f"  Max competing detections for single truth: {np.max(n_competing_dets)}")
    
    return matched_objs, seg_truth_mapping

def get_metadata(seg_cutout_data, cutout_id, seg_truth_mapping):
    """Optimized version of get_metadata with vectorization"""
    anns = []
    height, width = seg_cutout_data.shape
    
    if len(seg_truth_mapping) == 0:
        print(f"Cutout {cutout_id}: Empty seg truth mapping! No annotations")
    else:
        # unique objs (vectorized)
        unique_objs = np.unique(seg_cutout_data)
        unique_objs = unique_objs[unique_objs > 0]  # removes background

        for s in unique_objs:
            # skip if no truth mapping
            if s not in seg_truth_mapping:
                continue
            # obj class and info from truth matching
            obj_class = seg_truth_mapping[s][0]
            obj_info = seg_truth_mapping[s][1]

            # skip if no matching truth (class 2)
            if obj_class == 2:
                continue
            # using boolean indexing directly avoiding np.where
            # convert boolean arr (True -> 1 and False -> 0) using .astype
            mask = (seg_cutout_data == s).astype(np.uint8)

            y_coords, x_coords = np.where(mask)
            if len(y_coords) == 0:
                continue

            x0, x1 = x_coords.min(), x_coords.max()
            y0, y1 = y_coords.min(), y_coords.max()
            w, h = int(x1 - x0), int(y1 - y0)

            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            segmentation = []
            for contour in contours:
                contour = contour.flatten()
                if len(contour) > 4:
                    segmentation.append(contour.tolist())

            if len(segmentation) == 0:
                continue

            obj = {
                "bbox": [int(x0), int(y0), w, h],
                "bbox_mode": BoxMode.XYWH_ABS,
                "area": w * h,
                "segmentation": segmentation,
                "category_id": obj_class,
                "obj_id": int(s),
                "mag_F184": obj_info['mag_F184'],
                "mag_H158": obj_info['mag_H158'],
                "mag_J129": obj_info['mag_J129'],
                "mag_Y106": obj_info['mag_Y106'],
                "ra": obj_info['ra'],
                "dec": obj_info['dec'],
                "sep_arcsec": obj_info["sep_arcsec"],
                "n_competing_dets": obj_info["n_competing_dets"],
                "match_quality": obj_info["match_quality"]
            }
            anns.append(obj)
    
    return {
        "annotations": anns,
        'height': height,
        'width': width,
        "image_id": cutout_id,
    }

def process_single_cutout(args):
    """Just process a single cutout"""
    cutout_id, seg_cutout_data, det_df, truth_df, ra_dec, seg_cutout_wcs = args
    if len(det_df) == 0:
        print(f"Cutout {cutout_id}: Empty detection catalog")
    if len(truth_df) == 0:
        print(f"Cutout {cutout_id}: Empty truth catalog")
    matched_objs, seg_truth_mapping = cross_match_objects(det_df, truth_df)
 
    metadata = get_metadata(seg_cutout_data, cutout_id, seg_truth_mapping)
    metadata["file_name"] = f'./roman_data/truth/{ra_dec[0]}_{ra_dec[1]}/full_c{cutout_id}_{ra_dec[0]}_{ra_dec[1]}.npy'
    metadata["wcs"] = seg_cutout_wcs
    metadata["det_cat_path"] = f'./roman_data/truth/{ra_dec[0]}_{ra_dec[1]}/det_c{cutout_id}_{ra_dec[0]}_{ra_dec[1]}.json'
    metadata["truth_cat_path"] = f'./roman_data/truth/{ra_dec[0]}_{ra_dec[1]}/truth_c{cutout_id}_{ra_dec[0]}_{ra_dec[1]}.json'
    metadata["matched_det_path"] = f'./roman_data/truth/{ra_dec[0]}_{ra_dec[1]}/matched_c{cutout_id}_{ra_dec[0]}_{ra_dec[1]}.json'
    metadata["num_matched"] = len(matched_objs)
    metadata["num_dets"] = len(det_df)
    metadata["num_truth"] = len(truth_df)
    
    return {
        'cutout_id': cutout_id,
        'metadata': metadata,
        'matched_objs': matched_objs,
        'success': True
    }

def create_cutout_metadata_parallel(seg_cutouts, det_dfs, truth_dfs, ra_dec, n_workers=64):
    """Parallel version with 16 workers"""
    
    print(f"Creating metadata for {len(seg_cutouts)} cutouts using {n_workers} workers...")
    start_time = time.time()
    
    # prep args for parallel processing
    args_list = []
    for cutout_id, (seg_cutout, det_df, truth_df) in enumerate(zip(seg_cutouts, det_dfs, truth_dfs)):
        args_list.append((
            cutout_id,
            seg_cutout.data,  # Pass data, not obj (for pickling)
            det_df,
            truth_df,
            ra_dec,
            seg_cutout.wcs.to_header_string()
        ))
    
    all_metadata = []
    successful_cutouts = 0
    failed_cutouts = 0
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # submit all tasks
        future_to_cutout = {
            executor.submit(process_single_cutout, args): args[0] 
            for args in args_list
        }
        # collect results with progress tracking
        for future in tqdm(as_completed(future_to_cutout), 
                          total=len(args_list), 
                          desc="Processing cutouts",
                          leave=False):
            
            cutout_id = future_to_cutout[future]
            try:
                result = future.result()
                
                if result['success']:
                    all_metadata.append(result['metadata'])
                    
                    # save matched objects
                    matched_file = f'./roman_data/truth/{ra_dec[0]}_{ra_dec[1]}/matched_c{cutout_id}_{ra_dec[0]}_{ra_dec[1]}.json'
                    result['matched_objs'].to_json(matched_file, orient='records')
                    successful_cutouts += 1
                else:
                    print(f"Failed cutout {cutout_id}: {result['error']}")
                    failed_cutouts += 1
                    
            except Exception as exc:
                print(f'Cutout {cutout_id} generated an exception: {exc}')
                failed_cutouts += 1
    
    # sort metadata by image_id to maintain order
    all_metadata.sort(key=lambda x: x['image_id'])
    
    os.makedirs('./roman_data/annotations', exist_ok=True)
    metadata_filename = f"./roman_data/annotations/{ra_dec[0]}_{ra_dec[1]}.json"
    convert_to_json(all_metadata, metadata_filename)
    
    elapsed_time = time.time() - start_time
    print(f"Metadata creation completed in {elapsed_time:.2f}s")
    print(f"Successfully processed {successful_cutouts}/{len(seg_cutouts)} cutouts")
    if failed_cutouts > 0:
        print(f"Failed cutouts: {failed_cutouts}")
    
    print(f"Metadata saved to {metadata_filename}")

detection_dir = 'roman_data/detection_fits'
segmentation_dir = 'roman_data/segmentation_fits'
truth_dir = 'roman_data/truth_fits'
# the specific tilenames we want 
det_files = ['roman_data/detection_fits/dc2_det_50.93_-42.0.fits.gz','roman_data/detection_fits/dc2_det_51.34_-41.3.fits.gz',
            'roman_data/detection_fits/dc2_det_51.37_-38.3.fits.gz','roman_data/detection_fits/dc2_det_51.53_-40.0.fits.gz',
            'roman_data/detection_fits/dc2_det_52.31_-41.6.fits.gz','roman_data/detection_fits/dc2_det_52.93_-40.8.fits.gz',
            'roman_data/detection_fits/dc2_det_53.25_-41.8.fits.gz','roman_data/detection_fits/dc2_det_53.75_-38.9.fits.gz',
            'roman_data/detection_fits/dc2_det_54.24_-38.3.fits.gz','roman_data/detection_fits/dc2_det_54.31_-41.6.fits.gz',
             'roman_data/detection_fits/dc2_det_55.03_-41.9.fits.gz','roman_data/detection_fits/dc2_det_56.06_-39.8.fits.gz',
            'roman_data/detection_fits/dc2_det_50.93_-38.8.fits.gz', 'roman_data/detection_fits/dc2_det_52.49_-39.1.fits.gz',
            'roman_data/detection_fits/dc2_det_52.40_-41.1.fits.gz', 'roman_data/detection_fits/dc2_det_55.54_-41.9.fits.gz']

# det_files = ['roman_data/detection_fits/dc2_det_52.40_-41.1.fits.gz']

# det_files = ['roman_data/detection_fits/dc2_det_51.53_-40.0.fits.gz', 'roman_data/detection_fits/dc2_det_52.31_-41.6.fits.gz',
#              'roman_data/detection_fits/dc2_det_52.93_-40.8.fits.gz',
#             'roman_data/detection_fits/dc2_det_53.25_-41.8.fits.gz','roman_data/detection_fits/dc2_det_53.75_-38.9.fits.gz',
#             'roman_data/detection_fits/dc2_det_54.24_-38.3.fits.gz','roman_data/detection_fits/dc2_det_54.31_-41.6.fits.gz',
#              'roman_data/detection_fits/dc2_det_55.03_-41.9.fits.gz','roman_data/detection_fits/dc2_det_56.06_-39.8.fits.gz',
#             'roman_data/detection_fits/dc2_det_50.93_-38.8.fits.gz', 'roman_data/detection_fits/dc2_det_52.49_-39.1.fits.gz']

total_start_time = time.time()
print(f"Processing {len(det_files)} tiles...")

for i, det_file in enumerate(tqdm(det_files, desc="Processing Tiles", unit="tile")):
#     if i == 0 or i == 1:
#         continue
    tile_start_time = time.time()
    ra_dec = extract_ra_dec(det_file)
    os.makedirs(f'roman_data/truth/{ra_dec[0]}_{ra_dec[1]}', exist_ok=True)
    print(f"\n{'='*50}")
    print(f"Processing tile {i+1}/{len(det_files)}: {ra_dec[0]}_{ra_dec[1]}")
    print(f"{'='*50}")
    # Step 1: Multiband Coadd
    step_start = time.time()
    full_img_data, w = create_multiband_coadd(ra_dec)
    coadd_time = time.time() - step_start
    print(f"Time Multiband coadd: {coadd_time:.2f}s")
    
    # Step 2: Creating Cutouts
    step_start = time.time()
    seg_file = f'{segmentation_dir}/dc2_seg_{ra_dec[0]}_{ra_dec[1]}.fits.gz'
    truth_file = f'{truth_dir}/dc2_index_{ra_dec[0]}_{ra_dec[1]}.fits.gz'
    seg_cutouts = create_cutouts(seg_file, full_img_data, ra_dec, w) # took ~ 4 minutes just to make all the cutouts for each tile
    cutouts_time = time.time() - step_start
    print(f"Time Create cutouts: {cutouts_time:.2f}s")
    
    # Step 3: Load and process truth cat
    step_start = time.time()
    # now, we get the truth catalog information for every cutout in this tile
    truth = fits.open(truth_file)
    truth_df = Table.read(truth,hdu=1).to_pandas()
    truth.close()
    print(f"\nTruth catalog loaded: {len(truth_df)} objects")
    cutout_truth_filenames, cutout_truth_dfs = get_cutout_truth_cat(seg_cutouts, truth_df, ra_dec)
    truth_processing_time = time.time() - step_start
    print(f"Time Truth catalog processing: {truth_processing_time:.2f}s")
    
    # Step 4: Load and process det cat
    # now, we get the detection catalog info for every cutout in this tile
    step_start = time.time()
    det = fits.open(det_file)
    det_df = Table.read(det, hdu=1).to_pandas()
    det.close()
    print(f"\nDetection catalog loaded: {len(det_df)} objects")
    cutout_det_filenames, cutout_det_dfs = get_cutout_det_cat(seg_cutouts, det_df, ra_dec)
    det_processing_time = time.time() - step_start
    print(f"Time  Detection catalog processing: {det_processing_time:.2f}s")
    
    # Step 5: Metadata and Annotations
    step_start = time.time()
    # now we need to create the DeepDISC annotations from the matched detections and metadata
    create_cutout_metadata_parallel(seg_cutouts, cutout_det_dfs, cutout_truth_dfs, ra_dec)
    metadata_time = time.time() - step_start
    print(f"Time  Metadata creation: {metadata_time:.2f}s")
    
    del full_img_data, seg_cutouts, truth_df, det_df
    del cutout_truth_dfs, cutout_det_dfs
    gc.collect()
    
    tile_total_time = time.time() - tile_start_time
    print(f"\nTILE SUMMARY:")
    print(f"   Multiband coadd:     {coadd_time:>8.2f}s ({coadd_time/tile_total_time*100:>5.1f}%)")
    print(f"   Create cutouts:      {cutouts_time:>8.2f}s ({cutouts_time/tile_total_time*100:>5.1f}%)")
    print(f"   Truth processing:    {truth_processing_time:>8.2f}s ({truth_processing_time/tile_total_time*100:>5.1f}%)")
    print(f"   Detection processing:{det_processing_time:>8.2f}s ({det_processing_time/tile_total_time*100:>5.1f}%)")
    print(f"   Metadata creation:   {metadata_time:>8.2f}s ({metadata_time/tile_total_time*100:>5.1f}%)")
    print(f"   TOTAL TILE TIME:     {tile_total_time:>8.2f}s")
    
#     if i > 0:
#         elapsed_total = time.time() - total_start_time
#         avg_time_per_tile = elapsed_total / (i + 1)
#         remaining_tiles = len(det_files) - (i + 1)
#         estimated_remaining = avg_time_per_tile * remaining_tiles
#         print(f"   📈 Avg per tile:      {avg_time_per_tile:>8.2f}s")
#         print(f"   ⏳ Est. remaining:    {estimated_remaining/60:>8.1f}m ({estimated_remaining/3600:>5.1f}h)")
    
#     speedup = 76.75 / tile_total_time
#     print(f"   SPEEDUP:             {speedup:.1f}x faster than baseline")
    
#     break

total_elapsed = time.time() - total_start_time
print(f"\n🏁 PROCESSING COMPLETE")
print(f"   Total time: {total_elapsed:.2f}s ({total_elapsed/60:.1f}m)")
print(f"   Tiles processed: {min(i+1, len(det_files))}")
if i+1 < len(det_files):
    print(f"   Remaining tiles: {len(det_files) - (i+1)}")