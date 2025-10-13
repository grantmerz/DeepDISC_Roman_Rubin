import os, re, json, time, gc, sys, glob, random, traceback, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import astropy.units as u
from scipy.ndimage import zoom
from collections import defaultdict, namedtuple

from lsst.afw.image import exposure
from lsst.afw.image import ExposureF
from lsst.afw.fits import readMetadata
import lsst.geom as geom
from lsst.afw.fits import makeLimitedFitsHeader, readMetadata
from lsst.afw.geom import makeSkyWcs


import multiprocessing as mp
from functools import partial
from tqdm import tqdm
pd.options.mode.chained_assignment = None

def find_tract_patch(ra, dec, skymap):
    """
    Finds LSST tract and patch for a given RA & Dec
    Args:
        ra (np.ndarray) in degrees
        dec (np.ndarray) in degrees
        skymap: LSST skymap object obtained from Butler

    Returns:
        tract_ids & patch_idxs (np.ndarray)
    """
    try:
        # generator instead to avoid creating an intermediate list
        spherePoints = (geom.SpherePoint(r * geom.degrees, d * geom.degrees) for r, d in zip(ra, dec))
        tracts = [skymap.findTract(spherePoint) for spherePoint in spherePoints]
        # generator can only be used once so we have to re-create it here 
        spherePoints = (geom.SpherePoint(r * geom.degrees, d * geom.degrees) for r, d in zip(ra, dec))
        patches = [patchInfo.getIndex() for spherePoint, tractInfo in zip(spherePoints, tracts) 
                      for patchInfo in [tractInfo.findPatch(spherePoint)]]
        patch_idxs = [",".join(map(str, idx)) for idx in patches]
        tract_ids = [tractInfo.getId() for tractInfo in tracts]

        return np.array(tract_ids), np.array(patch_idxs)
    except Exception as e: 
        print(f"Warning: Could not find tract/patch for some RA, Dec: {e}. Not within this skymap")
        return None, None

def cutout_metadata_map(tile, roman_dir, skymap, exp_map):
    """
    Creates a map linking each Roman cutout ID to the LSST tract(s) 
    and patch(es) it overlaps with as well as the RA/DEC boundaries

    Args:
        tile (str): Identifier for the tile being processed
        roman_dir (str): Root directory for the Roman truth data
        skymap: LSST skyMap object from the Butler
        exp_map: A pre-loaded dictionary of {filename: ExposureF} objects. 
    Returns:
        dict: Dict mapping cutout_id to a dict of 
            "tract_patches": list of unique (tract, patch) tuples,
            "bounds": RA/DEC min/max
    """
    cutout_map = {}
    print(f"Generating cutout-to-patch map for tile: {tile}...")
    for cutout_id in tqdm(range(225), desc="Mapping Cutouts"):
        start_time = time.time()
        
        roman_truth_path = f'{roman_dir}{tile}/truth_c{cutout_id}_{tile}.json'
        if not os.path.exists(roman_truth_path):
            cutout_map[cutout_id] = {'tract_patches': [], 'bounds': None}
            continue
        
        roman_truth_cat = pd.read_json(roman_truth_path, orient='records')
        if roman_truth_cat.empty:
            cutout_map[cutout_id] = {'tract_patches': [], 'bounds': None}
            print(f"Roman truth catalog empty! {tile}: {cutout_id}")
            del roman_truth_cat
            continue
        io_end_time = time.time()
        
        ra_min, ra_max = roman_truth_cat['ra'].min(), roman_truth_cat['ra'].max()
        dec_min, dec_max = roman_truth_cat['dec'].min(), roman_truth_cat['dec'].max()
        bounds = {
            'ra_min': ra_min,
            'ra_max': ra_max,
            'dec_min': dec_min,
            'dec_max': dec_max,
        }
        # corner points of bbox
        corners = [
            (ra_min, dec_min), # bottom left
            (ra_min, dec_max), # top left
            (ra_max, dec_max), # top right
            (ra_max, dec_min), # bottom right
        ]
        corner_ras, corner_decs = zip(*corners)
        tracts, patches = find_tract_patch(np.array(corner_ras), np.array(corner_decs), skymap)
        if tracts is None or patches is None:
            cutout_map[cutout_id] = {'tract_patches': [], 'bounds': bounds}
            continue

        # combine and find unique (tract, patch) pairs
        # auto handles duplicates if corners fall in same patch
        unique_pairs = sorted(list(set(zip([int(t) for t in tracts], 
                                           [str(p) for p in patches]
                                          ))))

        anchor_tract_patch = (None, None)
        # Even if a cutout has corners that correspond to multipe tracts/patches, 
        # can its entire sky area bounding box fit completely inside just one of those candidate patches?
        # we ask this because all patches overlap one another, so there could be a case where all the corners 
        # are found in multiple tracts/patches but because of the overlap, they're the exact same coadd data.
        # another case could be that the bbox gets cutoff in one patch but is fully present in another patch.
        # The above case is usually near the edges of a tract.
        if len(unique_pairs) > 1:
            single_patch = False
            # we just select the first pair in the list because it's impossible for all 4 corners
            # to exist completely in one patch but not another patch.
            for tract, patch in unique_pairs:
                key = f"tract{tract}_patch{patch.replace(',', '_')}_bandi.fits"
                lsst_exp = exp_map.get(key)
                if not lsst_exp:
                    raise ValueError(f"LSST exposure not found for key {key}")
                corners_geom = [geom.SpherePoint(ra * geom.degrees, dec * geom.degrees) for ra, dec in corners]
                # cutout bbox using this patch's WCS
                wcs = lsst_exp['wcs']
                corners_pixel = [wcs.skyToPixel(corner) for corner in corners_geom]
                x_coords = [p.getX() for p in corners_pixel]
                y_coords = [p.getY() for p in corners_pixel]
                min_x, max_x = int(np.floor(min(x_coords))), int(np.ceil(max(x_coords)))
                min_y, max_y = int(np.floor(min(y_coords))), int(np.ceil(max(y_coords)))
                cutout_bbox = geom.Box2I(geom.Point2I(min_x, min_y), geom.Point2I(max_x, max_y))
                # if our patch bbox fully contains the cutout_bbox, we keep this patch
                # and can discard the rest of the pairs in the list
                if lsst_exp['bbox'].contains(cutout_bbox):
                    single_patch = True
                    unique_pairs = [(tract, patch)]
                    break
            
            # if after checking all candidates, none contained the full bbox, it's a true multi-patch case
            # then only do we calculate the anchor tract/patch
            if not single_patch:
                center_ra = (ra_min + ra_max) / 2.0
                center_dec = (dec_min + dec_max) / 2.0
                center_tract_arr, center_patch_arr = find_tract_patch(np.array([center_ra]), np.array([center_dec]), skymap)            
                if center_tract_arr is not None:
                    anchor_tract_patch = (int(center_tract_arr[0]), str(center_patch_arr[0]))
        
        cutout_map[cutout_id] = {
            "tract_patches": unique_pairs,
            "bounds": bounds,
            "anchor_tract_patch": anchor_tract_patch
        }
        compute_end_time = time.time()
        print(f"Cutout {cutout_id}: I/O took {io_end_time - start_time:.2f}s, Compute took {compute_end_time - io_end_time:.2f}s")
        del roman_truth_cat
        gc.collect()
    return cutout_map

def check_tile(tile, global_skymap, root_dir, roman_dir, lsst_dir, coadd_dir):
    """Processes a single tile and returns any true multi-patch cutout IDs."""
    try:
        start_time = time.time()
        # load the per-tile truth catalog to see which patches are needed
        tile_truth_path = f'{lsst_dir}{tile}/full_truth_{tile}.parquet'
        if not os.path.exists(tile_truth_path):
            print(f"ERROR: Truth file not found at {tile_truth_path}")
            return tile, f"ERROR: Truth file not found at {tile_truth_path}"
        tile_lsst_truth = pd.read_parquet(tile_truth_path)
        if tile_lsst_truth.empty:
            print(f"ERROR: Truth file empty at {tile_truth_path}")
            del tile_lsst_truth
            return tile, [] # No objects, so no multi-patch issues

        tile_tract_patch = tile_lsst_truth.groupby('tract')['patch'].apply(lambda x: sorted(x.unique())).to_dict()
        del tile_lsst_truth
        gc.collect()
        catalog_read_time = time.time()
        
        # build tile-specific exp_map
        exp_map = {}
        for tract, patches in tile_tract_patch.items():
            for patch in patches:
                # We only need the i-band for geometry checks
                fname = f"tract{tract}_patch{patch.replace(',', '_')}_bandi.fits"
                fpath = os.path.join(coadd_dir, fname)
                if os.path.exists(fpath):
                    metadata = readMetadata(fpath, hdu=1)  # Skip ExposureF constructor
                    wcs = makeSkyWcs(metadata)                    
                    # image dims from metadata
                    width = metadata.getScalar("NAXIS1")  # width
                    height = metadata.getScalar("NAXIS2")  # height
                    origin_x = int(-metadata['LTV1']) # x minimum
                    origin_y = int(-metadata['LTV2']) # y minimum
                    bbox = geom.Box2I(geom.Point2I(origin_x, origin_y), geom.Extent2I(width, height))
                    exp_map[fname] = {
                        'wcs': wcs,
                        'bbox': bbox
                    }
                else:
                    print(f"EXPOSURE DOESN'T EXIS {fpath}")
        io_end_time = time.time()
        # run optimized mapping function
        cutout_map = cutout_metadata_map(tile, roman_dir, global_skymap, exp_map)
        compute_end_time = time.time()
        del exp_map
        gc.collect()
        
        # perform check and return the results
        multi_patch_ids = [cid for cid, data in cutout_map.items() if len(data['tract_patches']) > 1]
        print(f"Tile {tile}: Catalog I/O took {catalog_read_time - start_time:.2f}s, Exp_Map Creation I/O took {io_end_time - catalog_read_time:.2f}s, Compute took {compute_end_time - io_end_time:.2f}s")
        return tile, multi_patch_ids

    except Exception as e:
        return tile, f"ERROR: {e}"

if __name__ == "__main__":    
    ROOT_DIR = '/pscratch/sd/y/yaswante/MyQuota/roman_lsst/'
    LSST_DIR = f'{ROOT_DIR}truth-lsst/'
    ROMAN_DIR = f'{ROOT_DIR}truth-roman/'
    COADD_DIR = f'{ROOT_DIR}full_coadd_butler_striped/'
    TILE_LIST_FILE = f'{ROOT_DIR}700_tiles.txt'

    print("Loading global skymap...")
    with open('/pscratch/sd/y/yaswante/MyQuota/roman_lsst/dr6_deepCoadd_skyMap.pkl', 'rb') as f:
        skymap = pickle.load(f)

    print("Loading tile list...")
    with open(TILE_LIST_FILE, 'r') as f:
        tiles = [line.strip() for line in f if line.strip()]

    tasks = [(tile, skymap, ROOT_DIR, ROMAN_DIR, LSST_DIR, COADD_DIR) for tile in tiles]

    num_processes = 256
    print(f"\nStarting verification for {len(tiles)} tiles using {num_processes} processes...")
    results = {}
    start_time = time.time()
    with mp.Pool(processes=num_processes) as pool:
        pbar = tqdm(total=len(tasks))
        for tile, multi_patch_ids in pool.starmap(check_tile, tasks):
            results[tile] = multi_patch_ids
            pbar.update(1)
        pbar.close()
    end_time = time.time()
    print(f"\nVerification complete in {end_time - start_time:.2f} seconds.")

    print("\n--- Verification Report ---")
    true_multi_patch_tiles = {tile: ids for tile, ids in results.items() if isinstance(ids, list) and len(ids) > 0}
    error_tiles = {tile: msg for tile, msg in results.items() if isinstance(msg, str)}
    
    if not true_multi_patch_tiles:
        print("SUCCESS: No true multi-patch cutouts were found in any of the 700 tiles.")
        print("Your hypothesis is correct. You can proceed without the GetTemplateTask logic.")
    else:
        print(f"WARNING: Found {len(true_multi_patch_tiles)} tiles with true multi-patch cutouts:")
        for tile, ids in true_multi_patch_tiles.items():
            print(f"  - Tile {tile}: {len(ids)} cutouts (IDs: {ids})")

    if error_tiles:
        print(f"Found {len(error_tiles)} tiles that failed with an error:")
        for tile, msg in error_tiles.items():
            print(f"  - Tile {tile}: {msg}")
# took 239.60 seconds with 32 process on login node