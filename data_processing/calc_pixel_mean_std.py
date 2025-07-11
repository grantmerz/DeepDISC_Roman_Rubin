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


# if all_metadata not available
# import numpy as np
# # train - 2264
# means1 = [0.97712864, 1.39985861, 1.1705819, 1.32836647, 0.07292949, 0.05212027,
#           0.07686396, 0.10820146, 0.13678103, 0.21416727] 
# std1 = [25.09916028, 33.55411201, 32.27604826, 33.65564667,  1.23393619,  0.62527912,
#         0.90964217,  1.29218721,  1.68141347,  2.53358636,]
# # val - 645
# means2 = [1.04510643, 1.48554965, 1.22554864, 1.38715854, 0.06708084, 0.05461708, 
#           0.07907751, 0.11093077, 0.13886542, 0.22121676] 
# std2 = [23.96581063, 32.31719949, 30.87609945, 32.35426318,  1.18710183,  0.6282187, 
#         0.87981039,  1.22820512,  1.61456256,  2.56517648]
# # test - 325
# means3 = [0.97581576, 1.38375226, 1.15473818, 1.29755389, 0.08251515, 0.07355598, 
#           0.08711831, 0.12756675, 0.17699495, 0.33186062] 
# std3 = [27.50737869, 37.16229961, 34.60873065, 36.63573593, 1.51284636, 0.87668429, 
#         0.94773068, 1.46231145, 2.2127573, 4.31327333]

# total_means = []
# total_stds = []
# total_samples = 2264 + 645 + 325 - 3
# for i in range(10):
#     final_mean = (means1[i] + means2[i] + means3[i])/3
#     total_means.append(final_mean)
#     weighted_sum = ((2264 - 1) * std1[i]**2) + ((645 - 1) * std2[i]**2) + ((325 - 1) * std1[i]**2)
#     # weighted standard deviation
#     weighted_std = np.sqrt(weighted_sum / total_samples)
#     total_stds.append(weighted_std)
    
# np.asarray(total_means), np.asarray(total_stds)