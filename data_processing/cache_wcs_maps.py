import os
import json
import numpy as np
import argparse
import multiprocessing as mp
from tqdm import tqdm
import pandas as pd
import time
from deepdisc.data_format.conversions import convert_to_json
from deepdisc.data_format.file_io import get_data_from_json
from astropy.wcs import WCS


def lsst_key_mapper(dataset_dict):
    
    key = dataset_dict["file_name"]
    im = key.split('truth/')[1]
    k = os.path.join("/work/hdd/bdsp/yse2/lsst_data/truth/",im)
    return k

def build_remap(wcs_rubin, wcs_roman, src_shape, dst_shape):
    """
    Precompute float32 remap arrays for cv2.remap.
    dst pixel → sky → src pixel
    """
    new_h, new_w = dst_shape
    # Grid of all Roman pixel coords
    yg, xg = np.indices((512, 512))
    ra, dec = wcs_roman.wcs_pix2world(xg.ravel(), yg.ravel(), 0)
    src_x, src_y = wcs_rubin.wcs_world2pix(ra, dec, 0)
    

    map_x = src_x.reshape(new_h, new_w).astype(np.float32)
    map_y = src_y.reshape(new_h, new_w).astype(np.float32)
    return map_x, map_y


import hashlib
def cache_maps(args):

    d, cache_dir=args
    
    rubin_fns = lsst_key_mapper(d)
    roman_fns = rubin_fns.replace('lsst_data', 'truth-roman').replace('/truth/','/')
    patch = os.path.dirname(roman_fns).split('/')[-1]
    subpatch = int(os.path.basename(roman_fns).split('_')[0][1:])
    troman_fn = os.path.join('/projects/bfhm/yse2/annotations_roman/',patch)+'.json'
    wcs_rubin = WCS(d['wcs'])
    wcs_roman = WCS(d['wcs_roman'])

    # Generate a unique hash for this specific Rubin-Roman pair
    cache_key = hashlib.md5((rubin_fns + roman_fns).encode()).hexdigest()
    cache_path = os.path.join(cache_dir, f"{cache_key}.npy")

    mapxy = build_remap(wcs_rubin, wcs_roman, (d['height'],d['width']), (512,512))
    np.save(cache_path, mapxy)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Add keypoints to existing annotation files.")
    parser.add_argument("--dset", type=str, help="The name of the annotated dataset to process (e.g., train,test,val)")
    parser.add_argument("--snr", type=int, default=5, help="SNR level (default: 5)")
    parser.add_argument('--anns-dir', type=str, required=True,
                       help='Base name for the annotations directory (e.g., "annotations", "annotations_ups", "annotationsc_ups").')
    parser.add_argument('--root-dir', type=str, default='/work/hdd/bdsp/yse2/lsst_data/', 
                       help='Root directory of the data (e.g., ./lsst_data/)')
    parser.add_argument('--cache-dir', type=str, default='/work/hdd/bfhm/g4merz/wcs_map_cache/', 
                       help='Root directory of the data as stored in the annotations (e.g., /u/yse2/lsst_data/truth/)')

    args = parser.parse_args()
    
    
    annotations_dir = f'{args.root_dir}{args.anns_dir}_lvl{args.snr}/'
    truth_dir = os.path.join(args.root_dir,'truth/')
    metadata_path = os.path.join(annotations_dir, f'{args.dset}.json')
    cache_dir = os.path.join(args.cache_dir,args.dset)
    
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found at {metadata_path}")

    else:
        print(f'Loading {metadata_path}')
        with open(metadata_path, 'r') as f:
            data = json.load(f)
    
        num_processes = int(os.environ.get("SLURM_CPUS_ON_NODE", len(os.sched_getaffinity(0)) or 64))

        pool_args = [(data[i],cache_dir) for i in range(len(data))]

    
        print(f"Adding roman wcs for {metadata_path}.")
        
        start_time = time.time()    
        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(cache_maps,pool_args)
        end_time = time.time()
    
        print(f"\Cached WCS maps in {cache_dir}. Took {end_time - start_time:.2f} seconds.")


    

    
   
