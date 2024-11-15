import os

with open("truth_file_names.txt", 'r') as file:
    hrefs = [line.strip() for line in file]

for url in hrefs:
    full_download_url = "https://irsa.ipac.caltech.edu/data/theory/Roman/Troxel2023/truth/coadd/" + url
    os.system(f'wget -P roman_data/truth_fits/ {full_download_url}')
    