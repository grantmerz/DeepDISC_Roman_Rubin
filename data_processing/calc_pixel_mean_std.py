"""
Calculate pixel mean and standard deviation for model normalization.
"""

import os, json, sys, argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
from typing import Dict, Optional
from astropy.wcs import WCS
# allows us to import from the custom configs directory w/o affecting deepdisc library imports
CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'configs'))
sys.path.insert(0, CONFIG_DIR)
from custom.transforms import LanczosResizeTransform
WCS_LOOKUP: Optional[Dict[str, Dict]] = None

def _init_worker(wcs_lookup: dict) -> None:
    """Pool initializer, ensure wcs lookup is initialized for every worker"""
    global WCS_LOOKUP
    WCS_LOOKUP = wcs_lookup

def get_stats_for_file(filename, num_channels, 
                       upsample_lsst, cache_dir, 
                       new_w, new_h, using_roman):
    """
    Calculates mean and std for a single image file.
    Args:
        filename (str): Path to the image file.
        num_channels (int): Number of channels in the image.
        upsample_lsst (bool): Whether to apply upsampling to LSST images
        cache_dir (str): Directory to use for caching WCS maps if upsampling is applied
        new_w (int): New width for upsampling LSST images
        new_h (int): New height for upsampling LSST images
        using_roman (bool): Whether to read Roman cutouts on non-upsample path via file_name.replace('lsst_data', 'roman_data')
        
    Returns:
        A tuple (channel_sum, channel_sq_sum, pixel_count)
        or (None, None, None) if an error occurs.
    """
    # Roman mode on non-upsample path: read Roman cutouts derived from LSST paths
    source_fn = filename
    if using_roman and not upsample_lsst:
        source_fn = filename.replace('lsst_data', 'roman_data')

    if not os.path.exists(source_fn):
        print(f"Skipping non-existent file: {source_fn}")
        return None, None, None
    try:
        img = np.load(source_fn)
        if img.shape[0] != num_channels or img.shape[1] == 0 or img.shape[2] == 0:
            # print(f"Skipping malformed or empty image: {filename}")
            return None, None, None
        
        if upsample_lsst:
            cache_fn = filename.replace("/u/","/work/hdd/bdsp/")
            wcs_lsst = WCS(WCS_LOOKUP[filename]['wcs'])
            wcs_roman = WCS(WCS_LOOKUP[filename]['wcs_roman'])
            # img is C,H,W -> H,W,C for transform API
            lsst_hwc = np.ascontiguousarray(img.transpose(1, 2, 0))
            tfm = LanczosResizeTransform(
                h=lsst_hwc.shape[0],
                w=lsst_hwc.shape[1],
                new_h=new_h,
                new_w=new_w,
                wcs_rubin=wcs_lsst,
                wcs_roman=wcs_roman,
                rubin_fns=cache_fn,
                roman_fns=cache_fn.replace('lsst_data', 'truth-roman').replace('/truth/', '/'),
                cache_dir=cache_dir,
            )
            lsst_512_hwc = tfm.apply_image(lsst_hwc)  # (512,512,6)
            img = np.ascontiguousarray(lsst_512_hwc.transpose(2, 0, 1))  # (6,512,512)
        # axis=(1, 2) because shape is (channels, height, width)        
        # means_for_filters = np.mean(img, axis=(1, 2))
        # stds_for_filters = np.std(img, axis=(1, 2))
        # axis=(1, 2) because shape is (channels, height, width)
        channel_sum = np.sum(img, axis=(1, 2))
        channel_sq_sum = np.sum(np.square(img), axis=(1, 2))
        pixel_count = img.shape[1] * img.shape[2]
        return channel_sum, channel_sq_sum, pixel_count
    except Exception as e:
        # print(f"Error processing file {filename}: {e}")
        return None, None, None


def calculate_pixel_stats(root_dir, anns_dir, snr_lvl, num_channels, 
                          max_workers, upsample_lsst, filenames_csv, 
                          cache_dir, new_w, new_h, using_roman):
    """
    Calculate pixel stats from image data.
    
    Args:
        root_dir (str): Root directory of the data
        anns_dir (str): Base name for the annotations directory
        snr_lvl (int): SNR level for annotations directory
        num_channels (int): Number of channels in the image data
        max_workers (int): Number of CPU cores to use
        upsample_lsst (bool): Whether to upsample LSST images
        filenames_csv (str): Path to CSV with a file_name column containing all image fns used for calculation (used if we can't find the .json file in the expected location)
        cache_dir (str): Directory to use for caching WCS maps if upsampling is applied
        new_w (int): New width for upsampling LSST images
        new_h (int): New height for upsampling LSST images
        using_roman (bool): Whether to read Roman cutouts on non-upsample path via file_name.replace('lsst_data', 'roman_data')
    """
    if not upsample_lsst:
        annotations_dir = f'{root_dir}{anns_dir}_lvl{snr_lvl}/'
        train_metadata_path = os.path.join(annotations_dir, 'train_30k_keypoints.json')
    else:
        train_metadata_path = os.path.expanduser(filenames_csv)
    # /u/yse2/lsst_data/annotations_lvl5/train_30k_keypoints_wcs.json
    print(f"Calculating the mean and std using image filenames from {train_metadata_path}")
    if not os.path.exists(train_metadata_path):
        print(f"Error: Metadata file not found at {train_metadata_path}")
        return
    
    # with open(train_metadata_path, 'r') as f:
    #     data = json.load(f)
    # image_files = [entry['file_name'] for entry in data]
    data = pd.read_json(train_metadata_path) if not upsample_lsst else pd.read_csv(train_metadata_path)
    unique_img_files = set(data['file_name'].tolist())
    total_files = len(unique_img_files)
    wcs_lookup = {}
    if upsample_lsst:
        required_cols = {'file_name', 'wcs', 'wcs_roman'}
        missing_cols = required_cols.difference(set(data.columns))
        if missing_cols:
            print(f"Error: upsample mode requires columns {sorted(required_cols)} in {train_metadata_path}. "
                  f"Missing: {sorted(missing_cols)}")
            return
        wcs_lookup = data.set_index('file_name').to_dict('index')
        print(f"WCS lookup built for {len(wcs_lookup)} images")
    if using_roman and not upsample_lsst:
        example_input = next(iter(unique_img_files), None)
        if example_input is not None:
            example_roman = example_input.replace('lsst_data', 'roman_data')
            print("Roman mode enabled: reading Roman files via file_name.replace('lsst_data', 'roman_data').")
            print(f"Example Roman file: {example_roman}")
        else:
            print("Roman mode enabled: reading Roman files via file_name.replace('lsst_data', 'roman_data').")
    print(f"Found {total_files} images to process.")
    print(f"Using {max_workers} worker processes.")
    # means_list = []
    # stds_list = []
    global_sum = np.zeros(num_channels, dtype=np.float64)
    global_sq_sum = np.zeros(num_channels, dtype=np.float64)
    total_pixels = 0
    valid_files = 0
    
    with ProcessPoolExecutor(max_workers=max_workers, 
                             initializer=_init_worker, 
                             initargs=(wcs_lookup,)) as executor:
        # all jobs to the pool
        future_to_file = {executor.submit(get_stats_for_file,
                                          fn, num_channels, upsample_lsst,
                                          cache_dir, new_w, new_h, using_roman): fn for fn in unique_img_files}
        # processing results as completed
        for future in tqdm(as_completed(future_to_file), total=total_files, desc="Processing images"):
            # mean, std = future.result()
            # if mean is not None and std is not None:
            #     means_list.append(mean)
            #     stds_list.append(std)
            channel_sum, channel_sq_sum, pixel_count = future.result()
            if channel_sum is not None and channel_sq_sum is not None:
                global_sum += channel_sum
                global_sq_sum += channel_sq_sum
                total_pixels += pixel_count
                valid_files += 1
    # if not means_list:
    if valid_files == 0 or total_pixels == 0: 
        print("No valid images were processed. Cannot calculate stats.")
        return
    # means_arr = np.array(means_list)
    # stds_arr = np.array(stds_list)
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
    # pixel_mean = means_arr.mean(axis=0)
    # pixel_std = stds_arr.mean(axis=0)
    pixel_mean = global_sum / total_pixels
    # var = E(X^2) - [E(X)]^2 where E(X) is mean
    # var can't be neg but just incase of numerical issues we clip to 0 before sqrt
    pixel_var = np.maximum(global_sq_sum / total_pixels - np.square(pixel_mean), 0.0)
    pixel_std = np.sqrt(pixel_var)
    print("Calculation Complete")
    # print(f"Processed {len(means_arr)} / {total_files} valid images.")
    print(f"Processed {valid_files} / {total_files} valid images.")
    
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
    parser.add_argument('--upsample_lsst', default=False, action='store_true',
                        help='Whether to apply upsampling to LSST images (if calculating stats for upsampled data)')
    # should adapt this for all training sets with this being used if we can't find the .json file in the expected location
    parser.add_argument('--filenames_csv', type=str, default='~/train30k_fns_wcs.csv',
                        help='Path to CSV with a file_name column containing all image fns used for calculation')
    parser.add_argument('--cache_dir', type=str, default='/work/hdd/bfhm/g4merz/wcs_map_cache/train_30k_keypoints_wcs',
                       help='Directory to use for caching WCS maps if upsampling is applied')
    parser.add_argument('--new_w', type=int, default=512,
                       help='New width for upsampling LSST images')
    parser.add_argument('--new_h', type=int, default=512,
                       help='New height for upsampling LSST images')
    parser.add_argument('--using_roman', default=False, action='store_true',
                        help="Use Roman cutouts on non-upsample path via file_name.replace('lsst_data', 'roman_data')")
    args = parser.parse_args()

    calculate_pixel_stats(args.root_dir, args.anns_dir, args.snr_lvl, args.num_channels, 
                          args.max_workers, args.upsample_lsst, args.filenames_csv, 
                          args.cache_dir, args.new_w, args.new_h, args.using_roman)

# python data_processing/calc_pixel_mean_std.py --root_dir /u/yse2/lsst_data/ --anns_dir annotations --snr_lvl 5 --num_channels 6 --max_workers 16
# python data_processing/calc_pixel_mean_std.py --root_dir /u/yse2/lsst_data/ --anns_dir annotations --snr_lvl 5 --num_channels 6 --max_workers 16 --using_roman