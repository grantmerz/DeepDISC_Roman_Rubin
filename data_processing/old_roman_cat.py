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

def process_chunk(chunk_data, all_bounds, cat_type='det'):
    """
    processing a single chunk of the catalog data
    """
    start, end, roman_cat_file, key = chunk_data
    
    chunk = pd.read_hdf(roman_cat_file, key=key, start=start, stop=end)
    
    if cat_type == 'det':
        chunk_coords = SkyCoord(ra=chunk['alphawin_j2000']*u.deg, 
                              dec=chunk['deltawin_j2000']*u.deg)
    else: # roman_truth
        chunk_coords = SkyCoord(ra=chunk['ra']*u.deg, 
                              dec=chunk['dec']*u.deg)
    
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
        return filtered_chunk
#         if cat_type == 'det':
#             keep_cols = ['alphawin_j2000', 'deltawin_j2000', 'mag_auto_F184', ]
#             keep_cols = [col for col in keep_cols if col in filtered_chunk.columns]
#             return filtered_chunk[keep_cols]
    
    return pd.DataFrame()

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

def main():
    parser = argparse.ArgumentParser(description='Processing Roman detection catalog')
#     parser.add_argument('--test-data', type=str, required=True,
#                       help='Path to test data JSON file')
    parser.add_argument('--cat-type', type=str, required=True,
                        help='Type of Roman cat file')
    parser.add_argument('--cat-file', type=str, required=True,
                      help='Path to Roman detection HDF5 file')
    parser.add_argument('--output', type=str, required=True,
                      help='Path to output filtered catalog')
    parser.add_argument('--chunksize', type=int, default=50000,
                      help='Chunk size for processing')
    args = parser.parse_args()
    
    test_data_fi = './lsst_data/annotationsc-ups/test.json'
    print(f"\nLoading in test data from {test_data_fi}...")
    with open(test_data_fi, 'r') as f:
        test_data = json.load(f)

    print("Calculating bounds from test data...")
    all_bounds = get_bounds(test_data)

    with pd.HDFStore(args.cat_file, mode='r') as store:
        key = store.keys()[0]
        total_rows = store.get_storer(key).nrows
    
    chunks = []
    for start in range(0, total_rows, args.chunksize):
        end = min(start + args.chunksize, total_rows)
        chunks.append((start, end, args.cat_file, key))

    # number of CPUs
    num_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', mp.cpu_count()))
    print(f"Using {num_cpus} CPUs")

    print(f"Processing {len(chunks)} chunks...")
    with mp.Pool(num_cpus) as pool:
        process_chunk_partial = partial(process_chunk, all_bounds=all_bounds, cat_type=cat_type)
        filtered_chunks = list(pool.imap(process_chunk_partial, chunks))
    
    print("Combining results...")
    combined_filtered = pd.concat([chunk for chunk in filtered_chunks if not chunk.empty], 
                                ignore_index=True)
    
    print("Deduplicating results...")
    if cat_type == 'det':
        final_catalog = combined_filtered.drop_duplicates(subset=['alphawin_j2000', 
                                                            'deltawin_j2000'])
    else: # truth
        final_catalog = combined_filtered.drop_duplicates(subset=['ra', 
                                                            'dec'])
    
    print(f"Saving results to {args.output}...")
    final_catalog.to_json(args.output)
    
    print("Done!")

if __name__ == '__main__':
    main()