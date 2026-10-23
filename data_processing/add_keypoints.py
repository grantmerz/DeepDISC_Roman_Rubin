import os
import json
import numpy as np
import argparse
import multiprocessing as mp
from tqdm import tqdm
import pandas as pd
import time
from deepdisc.data_format.conversions import convert_to_json

def annotations_to_truth_mapper(dataset_dict,truth_dir):
    '''
    Get the path to the truth file for the given annotations

    '''
    
    truth_path = dataset_dict["truth_cat_path"]
    truth_path = truth_path.replace('/u/yse2/lsst_data/truth/',truth_dir)
    return truth_path


def make_truth_lookup(ann_d,truth_dir):
    '''
    Make a lookup table that maps the obj_ids to the cutout_x,cutout_y values from the truth file
    This lets us quickly assign the keypoints in the annotation file based on obj_ids
    '''
    
    truth_path = annotations_to_truth_mapper(ann_d,truth_dir)
    with open(truth_path, 'r') as f:
        dtruth = json.load(f)
    truth_indexes_lookup = {}
    for i,a in enumerate(dtruth):
        truth_indexes_lookup[a['id']]=[a['cutout_x'],a['cutout_y']]

    return truth_indexes_lookup


def add_keypoints(args):
    '''
    Add keypoints for a single dict annotation file based on its truth file companion
    Returns the index to keep preserve the order of the dicts in the original file 
    '''
    
    d,index,truth_dir = args    
    truth_indexes_lookup = make_truth_lookup(d,truth_dir)
    for a in d['annotations']:
        oid = a['obj_id']
        a['keypoints'] = [*truth_indexes_lookup[oid],2]
    return d, index


    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Add keypoints to existing annotation files.")
    parser.add_argument("--dset", type=str, help="The name of the annotated dataset to process (e.g., train,test,val)")
    parser.add_argument("--snr", type=int, default=5, help="SNR level (default: 5)")
    parser.add_argument('--anns-dir', type=str, required=True,
                       help='Base name for the annotations directory (e.g., "annotations", "annotations_ups", "annotationsc_ups").')
    parser.add_argument('--root-dir', type=str, default='/work/hdd/bdsp/yse2/lsst_data/', 
                       help='Root directory of the data (e.g., ./lsst_data/)')
    parser.add_argument('--stored-root-dir', type=str, default='/u/yse2/lsst_data/truth/', 
                       help='Root directory of the data as stored in the annotations (e.g., /u/yse2/lsst_data/truth/)')

    args = parser.parse_args()
    
    
    annotations_dir = f'{args.root_dir}{args.anns_dir}_lvl{args.snr}/'
    truth_dir = os.path.join(args.root_dir,'truth/')
    metadata_path = os.path.join(annotations_dir, f'{args.dset}.json')
    
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found at {metadata_path}")

    else:
        print(f'Loading {metadata_path}')
        with open(metadata_path, 'r') as f:
            data = json.load(f)
    
        pool_args = [(data[i],i,truth_dir) for i in range(len(data))]
        num_processes = int(os.environ.get("SLURM_CPUS_ON_NODE", len(os.sched_getaffinity(0)) or 64))
    
        print(f"Adding keypoints for {metadata_path}.")
        
        start_time = time.time()    
        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(add_keypoints,pool_args)
        end_time = time.time()
    
        print(f"\nKeypoints added for {metadata_path}. Took {end_time - start_time:.2f} seconds.")

        #Map the randomized output from the multiprocessing back to the original order of the dicts (not really necessary but nice to maintain)
        dicts_with_kp = [res[0] for res in results if res[0] is not None]
        processed_inds = [res[1] for res in results if res[1] is not None]
        mapped_inds = {}
        for i in range(len(data)):
            mapped_inds[processed_inds[i]] = i
    
        updated_dicts = []
        for i in range(len(data)):
            updated_dicts.append(dicts_with_kp[mapped_inds[i]])
    
    
        metadata_path_new = os.path.join(annotations_dir, f'{args.dset}_keypoints.json')
        print(f"Saving dict to {metadata_path_new}. Took {end_time - start_time:.2f} seconds.")
        convert_to_json(updated_dicts,metadata_path_new)
    
    

    
   
