"""
Calculate pixel mean and standard deviation for model normalization.
"""

import os
import json
import numpy as np
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def get_stats_for_file(filename, num_channels):
    """
    Calculates mean and std for a single image file.
    Args:
        filename (str): Path to the image file.
        num_channels (int): Number of channels in the image.
        
    Returns:
        A tuple (mean, std) or (None, None) if an error occurs.
    """
    if not os.path.exists(filename):
        # print(f"Skipping non-existent file: {filename}")
        return None, None
    try:
        img_data = np.load(filename)
        if img_data.shape[0] != num_channels or img_data.shape[1] == 0 or img_data.shape[2] == 0:
            # print(f"Skipping malformed or empty image: {filename}")
            return None, None
            
        # axis=(1, 2) because shape is (channels, height, width)
        means_for_filters = np.mean(img_data, axis=(1, 2))
        stds_for_filters = np.std(img_data, axis=(1, 2))
        
        return means_for_filters, stds_for_filters
    except Exception as e:
        # print(f"Error processing file {filename}: {e}")
        return None, None


def calculate_pixel_stats(root_dir, anns_dir, snr_lvl, num_channels, max_workers):
    """
    Calculate pixel stats from image data.
    
    Args:
        root_dir (str): Root directory of the data
        anns_dir (str): Base name for the annotations directory
        snr_lvl (int): SNR level for annotations directory
        num_channels (int): Number of channels in the image data
        max_workers (int): Number of CPU cores to use
    """
    annotations_dir = f'{root_dir}{anns_dir}_lvl{snr_lvl}/'
    train_metadata_path = os.path.join(annotations_dir, 'train.json')
    print(f"Calculating the mean and std using image filenames from {train_metadata_path}")
    if not os.path.exists(train_metadata_path):
        print(f"Error: Metadata file not found at {train_metadata_path}")
        return
    
    with open(train_metadata_path, 'r') as f:
        data = json.load(f)
    
    image_files = [entry['file_name'] for entry in data]
    total_files = len(image_files)
    print(f"Found {total_files} images to process.")
    print(f"Using {max_workers} worker processes.")
    
    means_list = []
    stds_list = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # all jobs to the pool
        future_to_file = {executor.submit(get_stats_for_file, filename, num_channels): filename for filename in image_files}
        # processing results as completed
        for future in tqdm(as_completed(future_to_file), total=total_files, desc="Processing images"):
            mean, std = future.result()
            if mean is not None and std is not None:
                means_list.append(mean)
                stds_list.append(std)

    if not means_list:
        print("No valid images were processed. Cannot calculate stats.")
        return
    
    means_arr = np.array(means_list)
    stds_arr = np.array(stds_list)

    # for serial processing    
    # means_arr = np.empty((0, num_channels))
    # stds_arr = np.empty((0, num_channels))
    # for entry in data:
    #     filename = entry['file_name']
    #     if not os.path.exists(filename):
    #         print(f"Skipping non-existent file: {filename}")
    #         continue

    #     if entry['height'] == 0 or entry['width'] == 0:
    #         print(f"Skipping empty image: {filename}")
    #         continue

    #     img_data = np.load(filename)
    #     means_for_filters = np.mean(img_data, axis=(2, 1))
    #     means_for_filters = np.reshape(means_for_filters, (1, num_channels)) # necessary to keep 2D array for concat later
    #     stds_for_filters = np.std(img_data, axis=(2, 1))
    #     stds_for_filters = np.reshape(stds_for_filters, (1, num_channels))

    #     means_arr = np.concatenate((means_arr, means_for_filters), axis=0)
    #     stds_arr = np.concatenate((stds_arr, stds_for_filters), axis=0)

    pixel_mean = means_arr.mean(axis=0)
    pixel_std = stds_arr.mean(axis=0)
    print("Calculation Complete")
    print(f"Processed {len(means_arr)} / {total_files} valid images.")
    
    print("\nmodel.pixel_mean = [")
    for value in pixel_mean:
        print(f"    {value},")
    print("]")

    print("\nmodel.pixel_std = [")
    for value in pixel_std:
        print(f"    {value},")
    print("]")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate pixel mean and std for model normalization.')
    parser.add_argument('--root_dir', type=str, default='/u/yse2/lsst_data/', 
                       help='Root directory of the data (e.g., ./lsst_data/)')
    parser.add_argument('--anns_dir', type=str, required=True,
                       help='Base name for the annotations directory (e.g., "annotations", "annotations_ups", "annotationsc_ups").')
    parser.add_argument('--snr_lvl', type=int, default=5, 
                       help='SNR level for annotations directory')
    parser.add_argument('--num_channels', type=int, required=True, 
                       help='Number of channels in the image data')
    parser.add_argument('--max_workers', type=int, default=16,
                       help='Maximum number of worker processes to use. Defaults to the number of CPUs.')

    args = parser.parse_args()

    calculate_pixel_stats(args.root_dir, args.anns_dir, args.snr_lvl, args.num_channels, args.max_workers)

# python data_processing/calc_pixel_mean_std.py --root_dir /u/yse2/lsst_data/ --anns_dir annotations --snr_lvl 5 --num_channels 6 --max_workers 16
