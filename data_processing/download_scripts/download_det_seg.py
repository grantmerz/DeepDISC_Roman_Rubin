import os

with open("det_seg_file_names.txt", 'r') as file:
    hrefs = [line.strip() for line in file]

for url in hrefs:
    full_download_url = "https://irsa.ipac.caltech.edu/data/theory/Roman/Troxel2023/detection/" + url
    if 'det' in url:
        os.system(f'wget -P roman_data/detection_fits/ {full_download_url}')
    else: # segmentation fits files
        os.system(f'wget -P roman_data/segmentation_fits/ {full_download_url}')
    