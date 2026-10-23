import os, re, json, time, gc, sys, glob, random, traceback, pickle
import numpy as np
import pandas as pd

from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import astropy.units as u
from astropy.io import fits
from collections import defaultdict, namedtuple

from lsst.afw.image import ExposureF
import lsst.geom as geom
from lsst.afw.fits import makeLimitedFitsHeader, readMetadata
from lsst.afw.geom import makeSkyWcs

import multiprocessing as mp
from functools import partial
from tqdm import tqdm
pd.options.mode.chained_assignment = None

skymap = None

def find_tract_patch(ra, dec):
    global skymap
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
        spherePoints = [geom.SpherePoint(r * geom.degrees, d * geom.degrees) for r, d in zip(ra, dec)]
        tracts = [skymap.findTract(spherePoint) for spherePoint in spherePoints]
        patches = [patchInfo.getIndex() for spherePoint, tractInfo in zip(spherePoints, tracts) 
                      for patchInfo in [tractInfo.findPatch(spherePoint)]]
        patch_idxs = [",".join(map(str, idx)) for idx in patches]
        tract_ids = [tractInfo.getId() for tractInfo in tracts]

        return np.array(tract_ids), np.array(patch_idxs)
    except Exception as e: 
        print(f"Warning: Could not find tract/patch for some RA, Dec: {e}. Not within this skymap")
        return None, None

def cutout_metadata_map(tile, roman_dir, exp_map):
    global skymap
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
        roman_truth_path = f'{roman_dir}{tile}/truth_c{cutout_id}_{tile}.json'
        if not os.path.exists(roman_truth_path):
            cutout_map[cutout_id] = {'tract_patches': [], 'bounds': None}
            continue
        
        roman_truth_cat = pd.read_json(roman_truth_path, orient='records')
        if roman_truth_cat.empty:
            cutout_map[cutout_id] = {'tract_patches': [], 'bounds': None}
            continue
        
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
        tracts, patches = find_tract_patch(np.array(corner_ras), np.array(corner_decs))
        if tracts is None or patches is None:
            cutout_map[cutout_id] = {'tract_patches': [], 'bounds': bounds}
            continue

        # combine and find unique (tract, patch) pairs
        # auto handles duplicates if corners fall in same patch
        unique_pairs = sorted(list(set(zip([int(t) for t in tracts], 
                                           [str(p) for p in patches]
                                          ))))
        
        # Even if a cutout has corners that correspond to multipe tracts/patches, 
        # can its entire sky area bounding box fit completely inside just one of those candidate patches?
        # we ask this because all patches overlap one another, so there could be a case where all the corners 
        # are found in multiple tracts/patches but because of the overlap, they're the exact same coadd data.
        # another case could be that the bbox gets cutoff in one patch but is fully present in another patch (usually happens near the edge of a tract).
        if len(unique_pairs) > 1:
            single_patch = False
            for tract, patch in unique_pairs:
                key = f"tract{tract}_patch{patch.replace(',', '_')}_bandi.fits"
                lsst_exp = exp_map.get(key)
                if not lsst_exp:
                    raise(f"LSST exposure not found for {key}")
                corners_geom = [geom.SpherePoint(ra * geom.degrees, dec * geom.degrees) for ra, dec in corners]
                # cutout bbox using this patch's WCS
                wcs = lsst_exp.getWcs()
                corners_pixel = [wcs.skyToPixel(corner) for corner in corners_geom]
                x_coords = [p.getX() for p in corners_pixel]
                y_coords = [p.getY() for p in corners_pixel]
                min_x, max_x = int(np.floor(min(x_coords))), int(np.ceil(max(x_coords)))
                min_y, max_y = int(np.floor(min(y_coords))), int(np.ceil(max(y_coords)))
                cutout_bbox = geom.Box2I(geom.Point2I(min_x, min_y), geom.Point2I(max_x, max_y))
                # if our patch bbox fully contains the cutout_bbox, we keep this patch
                # and can discard the rest of the pairs in the list
                if lsst_exp.getBBox().contains(cutout_bbox):
                    single_patch = True
                    unique_pairs = [(tract, patch)]
                    break
            
            # if after checking all candidates, none contained the full bbox, it's a true multi-patch case
            # then only do we calculate the anchor tract/patch
            if not single_patch:
                print(f"OVerlapping Patches for {tile}!!")
        
        cutout_map[cutout_id] = {
            "tract_patches": unique_pairs,
            "bounds": bounds
        }
    return cutout_map

def process_cutout(cutout_id, cutout_map, exp_map, tile_cats, tile_info):
    tract_patch_pairs = cutout_map[cutout_id]['tract_patches']
    if not tract_patch_pairs:
        print(f"Cutout {cutout_id} in {tile_info['tile']}: No overlapping tract_patch pairs found. Skipped.")
        return cutout_id, None, False
    if len(tract_patch_pairs) > 1:
        # this case should be impossible as of Sep 24, 2025
        print(f"Cutout {cutout_id} in {tile_info['tile']}: Skipped - Spans {len(tract_patch_pairs)} patches.")
        return cutout_id, None, False
        
    tract, patch = tract_patch_pairs[0]
    # getting min/max RA/DEC to cut lsst image and lsst catalogs
    bounds = cutout_map[cutout_id]['bounds']
    if not bounds:
        print(f"Cutout {cutout_id} in {tile_info['tile']}: Skipped - No boundaries found in map.")
        return cutout_id, None, False
    
    ra_min, ra_max, dec_min, dec_max = (
        bounds['ra_min'],
        bounds['ra_max'],
        bounds['dec_min'],
        bounds['dec_max']
    )
    corners = [
        (ra_min, dec_min), # bottom left
        (ra_min, dec_max), # top left
        (ra_max, dec_max), # top right
        (ra_max, dec_min), # bottom right
    ]
    corners_geom = [geom.SpherePoint(ra * geom.degrees, dec * geom.degrees) for ra, dec in corners]
    # getting the lsst image in all bands and its corresponding WCS
    bands = ['u', 'g', 'r', 'i', 'z', 'y']
    all_imgs = []
    cutout_wcs, cutout_wcs_header = None, None
    cutout_origin = None
    bbox = None
    try:
        for band in bands:
            key = f"tract{tract}_patch{patch.replace(',', '_')}_band{band}.fits"
            lsst_exp = exp_map[key]
            if not lsst_exp:
                raise FileNotFoundError(f"Cached exposure {key} not found in exp_map.")
            if bbox is None:
                wcs = lsst_exp.getWcs()
                corners_pixel = [wcs.skyToPixel(corner) for corner in corners_geom]
                # min/max pixel coords to define the bounding box
                x_coords = [p.getX() for p in corners_pixel]
                y_coords = [p.getY() for p in corners_pixel]
                # guarantee that resulting integer bbox completely encloses entire continuous area
                min_x, max_x = int(np.floor(min(x_coords))), int(np.ceil(max(x_coords))) 
                min_y, max_y = int(np.floor(min(y_coords))), int(np.ceil(max(y_coords)))
                bbox = geom.Box2I(geom.Point2I(min_x, min_y), geom.Point2I(max_x, max_y))
                # print(f"LSST Cutout BBox: {bbox}")
            if bbox.isEmpty():
                print("The requested RA/Dec region is entirely outside the exposure's footprint.")
                all_imgs.extend([np.zeros((0, 0), dtype=np.float32) for _ in range(6)])
                break
            # grabbing cutout using subset method and saving the wcs
            cutout = lsst_exp.subset(bbox)
            all_imgs.append(cutout.image.array)
            if cutout_wcs is None:
                cutout_wcs = WCS(cutout.getWcs().getFitsMetadata())
                subimage_origin = cutout.getXY0()
                cutout_origin = (subimage_origin.getX(), subimage_origin.getY())
                # print("Cutout Origin and Dimensions: ", cutout_origin, cutout.getDimensions())
                # to save the right WCS relative to cutout, we need to reset the origin
                cutout.setXY0(geom.PointI(0, 0))
                cutout_wcs_header = makeLimitedFitsHeader(cutout.getWcs().getFitsMetadata())
    
        # saving the cutouts 
        lsst_tile_dir = f"{tile_info['lsst_dir']}{tile_info['tile']}/"
        os.makedirs(lsst_tile_dir, exist_ok=True)
        np.save(f"{lsst_tile_dir}c{cutout_id}_{tile_info['tile']}.npy", np.asarray(all_imgs))
        del all_imgs
        # saving catalogs
        for cat_type, cat_df in tile_cats.items():
            mask = ( (cat_df['ra'] >= ra_min) & (cat_df['ra'] <= ra_max) &
                     (cat_df['dec'] >= dec_min) & (cat_df['dec'] <= dec_max) )
            cutout_df = cat_df[mask].copy()
            # print(f"Number of {cat_type} objects within this cutout:", len(cutout_df))
            # adding cutout_x and cutout_y coords 
            if not cutout_df.empty:
                sky_coords = SkyCoord(ra=cutout_df['ra'].values * u.degree, dec=cutout_df['dec'].values * u.degree)
                xs, ys = cutout_wcs.world_to_pixel(sky_coords)
                xs -= cutout_origin[0]
                ys -= cutout_origin[1]
                cutout_df['cutout_x'] = xs
                cutout_df['cutout_y'] = ys
            
            cutout_df.to_json(f"{lsst_tile_dir}{cat_type}_c{cutout_id}_{tile_info['tile']}.json", orient='records')
            del cutout_df

        return cutout_id, cutout_wcs_header, True

    except Exception as e:
        print(f"TILE: {tile_info['tile']}: Cutout {cutout_id}: Failed with error - {e}")
        traceback.print_exc()
        return cutout_id, cutout_wcs_header, False

def process_tile(tile_info):
    global skymap
    start_time = time.time()
    tile, roman_dir, lsst_dir, coadd_dir =  tile_info['tile'], tile_info['roman_dir'], tile_info['lsst_dir'], tile_info['coadd_dir']

    load_cats_start_time = time.time()
    tile_truth_path = f'{lsst_dir}{tile}/full_truth_{tile}.parquet'
    if not os.path.exists(tile_truth_path):
        print(f"ERROR: Truth file not found at {tile_truth_path}")
        return tile, False
    tile_lsst_truth = pd.read_parquet(tile_truth_path)
    if tile_lsst_truth.empty:
        print(f"ERROR: Truth file empty at {tile_truth_path}")
        del tile_lsst_truth
        return tile, False

    tile_det_path = f'{lsst_dir}{tile}/full_det_{tile}.parquet'
    if not os.path.exists(tile_det_path):
        print(f"ERROR: Det file not found at {tile_det_path}")
        return tile, False
    tile_lsst_det = pd.read_parquet(tile_det_path)
    if tile_lsst_det.empty:
        print(f"ERROR: Det file empty at {tile_det_path}")
        del tile_lsst_det
        return tile, False
    load_cats_time = time.time() - load_cats_start_time
    tile_cats = {
        "truth": tile_lsst_truth,
        "det": tile_lsst_det,
    }
    
    tile_tract_patch = tile_lsst_truth.groupby('tract')['patch'].apply(lambda x: sorted(x.unique())).to_dict()
    load_exp_start_time = time.time()
    bands = ['u', 'g', 'r', 'i', 'z', 'y']
    exp_map = {
        f"tract{tract}_patch{patch.replace(',', '_')}_band{band}.fits": 
        ExposureF(f"{coadd_dir}tract{tract}_patch{patch.replace(',', '_')}_band{band}.fits")
        for band in bands
        for tract, patches in tile_tract_patch.items()
        for patch in patches
    }
    load_exp_time = time.time() - load_exp_start_time
    # print(f"We have {len(exp_map)} dataIds loaded from cache.")

    cutout_md_start_time = time.time()
    cutout_map = cutout_metadata_map(tile, roman_dir, exp_map)
    cutout_md_time = time.time() - cutout_md_start_time

    num_cutouts = 225
    cutout_ids = range(num_cutouts)
    cutouts_start_time = time.time()
    results = [
        process_cutout(cid, cutout_map, exp_map, tile_cats, tile_info) for cid in cutout_ids
    ]
    cutouts_time = time.time() - cutouts_start_time
    print(f"Processed {len(cutout_ids)} cutouts in serial in {cutouts_time:.4f} seconds.")
    successes = sum(1 for _, _, r in results if r)
    print(f"{successes} cutouts processed successfully.")
    # Saving WCS Data
    wcs_headers = {}
    for cid, wcs_header, success in results:
        if success and wcs_header:  # only save if cutout was successful and has WCS data
            wcs_headers[f"c{cid}"] = wcs_header
    
    if wcs_headers:  # only save if we have any WCS headers
        wcs_output_path = f"{lsst_dir}{tile}/wcs_{tile}.json"
        os.makedirs(os.path.dirname(wcs_output_path), exist_ok=True)
        with open(wcs_output_path, 'w') as f:
            json.dump(wcs_headers, f, indent=2)
        print(f"Saved {len(wcs_headers)} WCS headers to {wcs_output_path}")
    
    total_time = time.time() - start_time
    # print("\n" + "="*60)
    # print(f"TIMING SUMMARY FOR TILE {tile}")
    # print("="*60)
    # print(f"{'Stage':<25} {'Time (s)':<10} {'Percentage':<10}")
    # print("-" * 50)
    # print(f"{'Load Catalogs':<25} {load_cats_time:<10.4f} {100*load_cats_time/total_time:<10.1f}%")
    # print(f"{'Load Exposures':<25} {load_exp_time:<10.4f} {100*load_exp_time/total_time:<10.1f}%")
    # print(f"{'Cutout Metadata':<25} {cutout_md_time:<10.4f} {100*cutout_md_time/total_time:<10.1f}%")
    # print(f"{'Process Cutouts':<25} {cutouts_time:<10.4f} {100*cutouts_time/total_time:<10.1f}%")
    # print("-" * 50)
    # print(f"{'TOTAL':<25} {total_time:<10.4f} {'100.0%':<10}")
    # print("="*60)

    if successes != num_cutouts:
        return tile, False

    del cutout_map, exp_map, tile_lsst_truth, tile_lsst_det, tile_cats
    gc.collect()
    
    return tile, True

if __name__ == "__main__":    
    ROOT_DIR = '/pscratch/sd/y/yaswante/MyQuota/roman_lsst/'
    LSST_DIR = f'{ROOT_DIR}truth-lsst/'
    ROMAN_DIR = f'{ROOT_DIR}truth-roman/'
    COADD_DIR = f'{ROOT_DIR}full_coadd_butler_striped/' # striped folder containing our exposures
    TILE_LIST_FILE = f'{ROOT_DIR}700_tiles.txt'

    print("Loading global skymap...")
    with open('/pscratch/sd/y/yaswante/MyQuota/roman_lsst/dr6_deepCoadd_skyMap.pkl', 'rb') as f:
        skymap = pickle.load(f)

    print("Loading tile list...")
    with open(TILE_LIST_FILE, 'r') as f:
        tiles = [line.strip() for line in f if line.strip()]

    all_tile_info = [
        {
            "tile": tile,
            "roman_dir": ROMAN_DIR,
            "lsst_dir": LSST_DIR,
            "coadd_dir": COADD_DIR
        }
        for tile in tiles
    ]
    
    num_workers = 64
    print(f"Starting job. Processing {len(tiles)} tiles with {num_workers} parallel workers.")
    start_time = time.time()
    
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(process_tile, all_tile_info)
    
    end_time = time.time()
    print(f"\nFinished processing all tiles in {end_time - start_time:.2f} seconds.")
    
    successful_jobs = [res for res in results if res[1]]
    print(f"Successfully processed {len(successful_jobs)} out of {len(tiles)} tiles.")