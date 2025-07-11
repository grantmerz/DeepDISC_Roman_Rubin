"""
Calculate pixel mean and standard deviation for model normalization.
"""

import os
import json
import numpy as np
import argparse

def calculate_pixel_stats(root_dir, snr_lvl, num_channels):
    """
    Calculate pixel statistics from image data.
    
    Args:
        root_dir (str): Root directory of the data
        snr_lvl (int): SNR level for annotations directory
        num_channels (int): Number of channels in the image data
    """
    annotations_dir = f'{root_dir}annotations_lvl{snr_lvl}/'
    clean_metadata_path = os.path.join(annotations_dir, 'all_metadata.json')

    with open(clean_metadata_path, 'r') as f:
        data = json.load(f)

    means_arr = np.empty((0, num_channels))
    stds_arr = np.empty((0, num_channels))

    for entry in data:
        filename = entry['file_name']
        if not os.path.exists(filename):
            print(f"Skipping non-existent file: {filename}")
            continue

        if entry['height'] == 0 or entry['width'] == 0:
            print(f"Skipping empty image: {filename}")
            continue

        img_data = np.load(filename)
        means_for_filters = np.mean(img_data, axis=(2, 1))
        means_for_filters = np.reshape(means_for_filters, (1, num_channels))
        stds_for_filters = np.std(img_data, axis=(2, 1))
        stds_for_filters = np.reshape(stds_for_filters, (1, num_channels))

        means_arr = np.concatenate((means_arr, means_for_filters), axis=0)
        stds_arr = np.concatenate((stds_arr, stds_for_filters), axis=0)

    pixel_mean = means_arr.mean(axis=0)
    pixel_std = stds_arr.mean(axis=0)

    print("model.pixel_mean = [")
    for value in pixel_mean:
        print(f"    {value},")
    print("]")

    print("\nmodel.pixel_std = [")
    for value in pixel_std:
        print(f"    {value},")
    print("]")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate pixel mean and std for model normalization.')
    parser.add_argument('--root_dir', type=str, default='./lsst_data/', 
                       help='Root directory of the data (e.g., ./lsst_data/)')
    parser.add_argument('--snr_lvl', type=int, default=5, 
                       help='SNR level for annotations directory')
    parser.add_argument('--num_channels', type=int, required=True, 
                       help='Number of channels in the image data')

    args = parser.parse_args()

    calculate_pixel_stats(args.root_dir, args.snr_lvl, args.num_channels)