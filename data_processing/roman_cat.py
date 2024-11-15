import os
import gc
import json
import numpy as np
import pandas as pd
import argparse
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import multiprocessing as mp
from functools import partial

def match_with_truth(det_chunk, truth_cat, max_sep=0.5*u.arcsec):
    """
    Match detection catalog with truth catalog using sky coordinates
    Only keep 1-to-1 matches within max_sep
    """
    if len(det_chunk) == 0:
        return pd.DataFrame()

    # Create SkyCoord objects
    det_coords = SkyCoord(ra=det_chunk['alphawin_j2000']*u.deg, 
                         dec=det_chunk['deltawin_j2000']*u.deg)
    truth_coords = SkyCoord(ra=truth_cat['ra']*u.deg, 
                           dec=truth_cat['dec']*u.deg)
    
    # Find matches
    idx_det, idx_truth, d2d, _ = det_coords.search_around_sky(truth_coords, max_sep)
    
    # Find unique matches (1-to-1 correspondence)
    unique_det, det_counts = np.unique(idx_det, return_counts=True)
    unique_truth, truth_counts = np.unique(idx_truth, return_counts=True)
    
    # Keep only 1-to-1 matches
    good_det = unique_det[det_counts == 1]
    good_truth = unique_truth[truth_counts == 1]
    
    # Find intersection of good matches
    final_matches = np.intersect1d(idx_det[np.isin(idx_det, good_det)],
                                 idx_det[np.isin(idx_truth, good_truth)])
    
    # Return matched detections
    return det_chunk.iloc[final_matches].copy()

# def match_to_truth_chunk(det_chunk, truth_file, truth_chunksize=50000, max_sep=0.5*u.arcsec):
#     """
#     matching dets to truth cat while processing it in chunks
#     keeps track of all potential matches and selects the best ones
    
#     Parameters:
#     -----------
#     det_chunk : pd.DataFrame
#         Chunk of det cat
#     truth_file : str
#         Path to truth cat HDF5 file
#     truth_chunksize : int
#         Size of truth cat chunks to process
#     max_sep : astropy.units.Quantity
#         Max sep for matching
        
#     Returns:
#     --------
#     tuple (pd.DataFrame, pd.DataFrame)
#         Filtered det cat and corresponding truth cat with only 1-to-1 matches
#     """
#     if len(det_chunk) == 0:
#         return pd.DataFrame()
    
#     det_coords = SkyCoord(ra=det_chunk['alphawin_j2000']*u.deg,
#                          dec=det_chunk['deltawin_j2000']*u.deg)
    
#     # storing all matches for each det
#     # {det idx: list of (truth_idx, dist)}
#     all_matches = {}
#     truth_rows = {}  # Store truth cat rows by idx
    
#     with pd.HDFStore(truth_file, mode='r') as store:
#         key = store.keys()[0]
#         total_rows = store.get_storer(key).nrows
        
#         for start in range(0, total_rows, truth_chunksize):
#             end = min(start + truth_chunksize, total_rows)
#             truth_chunk = pd.read_hdf(truth_file, key=key, start=start, stop=end)
            
#             truth_coords = SkyCoord(ra=truth_chunk['ra']*u.deg,
#                                   dec=truth_chunk['dec']*u.deg)
            
#             # match this chunk
#             idx_det, idx_truth, d2d, _ = det_coords.search_around_sky(truth_coords, max_sep)
            
#             # stroing all matches w/ their dists and truth rows
#             for det_idx, truth_idx, dist in zip(idx_det, idx_truth, d2d):
#                 if det_idx not in all_matches:
#                     all_matches[det_idx] = []
#                 abs_truth_idx = truth_idx + start
#                 all_matches[det_idx].append((abs_truth_idx, dist.to(u.arcsec).value))
#                 # storing truth row if we haven't seen it before
#                 if abs_truth_idx not in truth_rows:
#                     truth_rows[abs_truth_idx] = truth_chunk.iloc[truth_idx]
            
#             del truth_chunk
#             gc.collect()
    
#     if not all_matches:  # no matches
#         return pd.DataFrame(), pd.DataFrame()
    
#     # now we have to find the best matches
#     best_matches = {}  # det_idx -> (truth_idx, distance)
#     used_truth = set()  # used truth idxs
    
#     # sorting all dets by their best match dist
#     det_best_distances = {det_idx: min(matches, key=lambda x: x[1])[1] 
#                          for det_idx, matches in all_matches.items()}
#     sorted_dets = sorted(det_best_distances.keys(), 
#                         key=lambda x: det_best_distances[x])
    
#     # finding best matches greedily as efficiency doesn't really matter here
#     for det_idx in sorted_dets:
#         # sorting matches for this det by dist
#         sorted_matches = sorted(all_matches[det_idx], key=lambda x: x[1])
        
#         # trying to find closest unused truth match
#         for truth_idx, dist in sorted_matches:
#             if truth_idx not in used_truth:
#                 best_matches[det_idx] = (truth_idx, dist)
#                 used_truth.add(truth_idx)
#                 break
    
#     # creating matched det cat
#     matched_mask = np.zeros(len(det_chunk), dtype=bool)
#     matched_det_indices = list(best_matches.keys())
#     matched_mask[matched_det_indices] = True
#     matched_det_catalog = det_chunk[matched_mask].copy()
    
#     # creating matched truth cat
#     matched_truth_indices = [best_matches[det_idx][0] for det_idx in matched_det_indices]
#     matched_truth_rows = [truth_rows[idx] for idx in matched_truth_indices]
#     matched_truth_catalog = pd.DataFrame(matched_truth_rows)
    
#     # adding match quality info for later use
#     match_distances = [best_matches[det_idx][1] for det_idx in matched_det_indices]
#     matched_det_catalog['match_distance'] = match_distances
#     matched_truth_catalog['match_distance'] = match_distances
    
#     return matched_det_catalog, matched_truth_catalog

def process_chunk(chunk_data, all_bounds, truth_cat):
    """
    processing a single chunk of the catalog data and matching to truth
    """
    chunk, _ = chunk_data
    
    # Create SkyCoord object for the chunk
    chunk_coords = SkyCoord(ra=chunk['alphawin_j2000']*u.deg, 
                           dec=chunk['deltawin_j2000']*u.deg)
    
    # Find objects within bounds
    chunk_mask = np.zeros(len(chunk_coords), dtype=bool)
    for bounds in all_bounds:
        ra_min, ra_max = np.min(bounds.ra), np.max(bounds.ra)
        dec_min, dec_max = np.min(bounds.dec), np.max(bounds.dec)
        
        chunk_mask |= ((chunk_coords.ra >= ra_min) & 
                      (chunk_coords.ra <= ra_max) &
                      (chunk_coords.dec >= dec_min) & 
                      (chunk_coords.dec <= dec_max))
    
    filtered_chunk = chunk[chunk_mask]
    
    if len(filtered_chunk) > 0:
        # Match with truth catalog
        matched_chunk = match_with_truth(filtered_chunk, truth_cat)
        return matched_chunk
    
    return pd.DataFrame()

# def process_chunk(chunk_data, all_bounds, truth_file): #  cat_type='det'
#     """
#     processing a single chunk of the catalog data and matching to truth
#     """
#     start, end, roman_cat_file, key = chunk_data
    
#     chunk = pd.read_hdf(roman_cat_file, key=key, start=start, stop=end)
#     chunk_coords = SkyCoord(ra=chunk['alphawin_j2000']*u.deg, 
#                            dec=chunk['deltawin_j2000']*u.deg)
# #     if cat_type == 'det':
# #         chunk_coords = SkyCoord(ra=chunk['alphawin_j2000']*u.deg, 
# #                               dec=chunk['deltawin_j2000']*u.deg)
# #     else: # roman_truth
# #         chunk_coords = SkyCoord(ra=chunk['ra']*u.deg, 
# #                               dec=chunk['dec']*u.deg)
    
#     # filtering by spatial bounds first
#     chunk_mask = np.zeros(len(chunk_coords), dtype=bool)
#     for bounds in all_bounds:
#         ra_min, ra_max = np.min(bounds.ra), np.max(bounds.ra)
#         dec_min, dec_max = np.min(bounds.dec), np.max(bounds.dec)
#         chunk_mask |= ((chunk_coords.ra >= ra_min) & 
#                       (chunk_coords.ra <= ra_max) &
#                       (chunk_coords.dec >= dec_min) & 
#                       (chunk_coords.dec <= dec_max))
    
#     filtered_chunk = chunk[chunk_mask]

#     if len(filtered_chunk) > 0:
#         # macth to truth cat and keep only 1-to-1 matches
#         matched_det, matched_truth = match_to_truth_chunk(filtered_chunk, truth_file)
#         return matched_det, matched_truth        
# #         if cat_type == 'truth':
# #             return filtered_chunk
# #         else:
# #             pass
# #         if cat_type == 'det':
# #             keep_cols = ['alphawin_j2000', 'deltawin_j2000', 'mag_auto_F184', ]
# #             keep_cols = [col for col in keep_cols if col in filtered_chunk.columns]
# #             return filtered_chunk[keep_cols]
    
#     return pd.DataFrame(), pd.DataFrame()

def get_bounds(test_data):
    all_bounds = []
    truth_info_cache = {}
    
    for d in test_data:
        imid = d['image_id']
        subpatch = d['subpatch']
        
        if subpatch not in truth_info_cache:
            truth_info_filename = f'./lsst_data/truth/{subpatch}/{subpatch}_info.json'
            with open(truth_info_filename) as json_data:
                truth_info_cache[subpatch] = json.load(json_data)
        
        truth_info = truth_info_cache[subpatch]
        entry = next(entry for entry in truth_info if entry['image_id'] == imid)
        wcs = WCS(entry['wcs'])
        height = entry['height']
        width = entry['width']
        
        corners = wcs.pixel_to_world([0, width-1, 0, width-1], 
                                   [0, 0, height-1, height-1])
        all_bounds.append(corners)
    
    return all_bounds

def load_json_chunks(file_path, chunk_size=50000):
    """
    Stream large JSON file in chunks using ijson
    """
    chunk = []
    parser = ijson.parse(open(file_path, 'rb'))
    
    try:
        # Handle both array of objects and newline-delimited JSON
        for prefix, event, value in parser:
            if event == 'start_map':  # Start of an object
                chunk.append({})
            elif event == 'map_key':  # Field name
                current_key = value
            elif event == 'string' or event == 'number' or event == 'boolean':  # Field value
                chunk[-1][current_key] = value
            
            if len(chunk) >= chunk_size:
                yield pd.DataFrame(chunk)
                chunk = []
                
        # Yield remaining items
        if chunk:
            yield pd.DataFrame(chunk)
            
    except ijson.common.IncompleteJSONError as e:
        print(f"Warning: Incomplete JSON detected: {e}")
        if chunk:
            yield pd.DataFrame(chunk)

def load_truth_catalog(truth_file, chunksize=50000):
    """
    Load truth catalog using ijson and return as DataFrame
    """
    print(f"Loading truth catalog from {truth_file}...")
    truth_data = []
    
    for chunk in load_json_chunks(truth_file, chunksize):
        truth_data.append(chunk)
    
    return pd.concat(truth_data, ignore_index=True)

def main():
    parser = argparse.ArgumentParser(description='Processing Roman detection catalog')
#     parser.add_argument('--test-data', type=str, required=True,
#                       help='Path to test data JSON file')
#     parser.add_argument('--cat-type', type=str, required=True,
#                         help='Type of Roman cat file')
    parser.add_argument('--det-file', type=str, required=True,
                      help='Path to Roman detection JSON file')
    parser.add_argument('--truth-file', type=str, required=True,
                      help='Path to Roman truth JSON file')
    parser.add_argument('--output-det', type=str, required=True,
                      help='Path to output filtered and matched detection catalog')
#     parser.add_argument('--output-truth', type=str, required=True,
#                       help='Path to output matched truth catalog')
#     parser.add_argument('--output', type=str, required=True,
#                       help='Path to output filtered catalog')
    parser.add_argument('--chunksize', type=int, default=50000,
                      help='Chunk size for processing')
    args = parser.parse_args()
    
    test_data_fi = './lsst_data/annotationsc-ups/test.json'
    print(f"\nLoading in test data from {test_data_fi}...")
    with open(test_data_fi, 'r') as f:
        test_data = json.load(f)

    print("Calculating bounds from test data...")
    all_bounds = get_bounds(test_data)
    
    truth_cat = load_truth_catalog(args.truth_file, args.chunksize)
       
    # det cat in chunks
    print(f"Processing det cat from {args.det_file}...")
    chunks = enumerate(load_json_chunks(args.det_file, args.chunksize))

    # number of CPUs
    num_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', mp.cpu_count()))
    print(f"Using {num_cpus} CPUs")

    print(f"Processing {len(chunks)} chunks...")
    with mp.Pool(num_cpus) as pool:
        process_chunk_partial = partial(process_chunk, 
                                     all_bounds=all_bounds, 
                                     truth_cat=truth_cat)
        filtered_chunks = list(tqdm(
            pool.imap(process_chunk_partial, chunks),
            desc="Processing chunks"
        ))
    
    print("Combining results...")
    combined_det = pd.concat([chunk for chunk in filtered_chunks if not chunk.empty], 
                                ignore_index=True)
#     det_chunks, truth_chunks = zip(*[r for r in results if not r[0].empty])
#     combined_det = pd.concat(det_chunks, ignore_index=True)
#     combined_truth = pd.concat(truth_chunks, ignore_index=True)
    
    print("Deduplicating results...")
    # for dets, keep match w/ smallest match_distance when duplicates exist
#     final_det = combined_det.sort_values('match_distance').drop_duplicates(subset=['alphawin_j2000', 
#                                                                                    'deltawin_j2000'])
    final_det = combined_det.drop_duplicates(subset=['alphawin_j2000', 'deltawin_j2000'])
    # get corresponding truth entries
#     final_truth = combined_truth[combined_truth.index.isin(final_det.index)]
#     if cat_type == 'det':
#         final_catalog = combined_filtered.drop_duplicates(subset=['alphawin_j2000', 
#                                                             'deltawin_j2000'])
#     else: # truth
#         final_catalog = combined_filtered.drop_duplicates(subset=['ra', 
#                                                             'dec'])
    
    print(f"Saving results to {args.output}...")
    final_det.to_json(args.output, orient='records')
    
    
#     print(f"Saving results to {args.output_det} and {args.output_truth}...")
#     final_det.to_json(args.output_det)
#     final_truth.to_json(args.output_truth)
    
    print("Done!")

if __name__ == '__main__':
    main()





