import os, csv

ras_decs = []
with open("original_img_file_names.csv", 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        ra, dec = map(float, row)
        ras_decs.append((ra, dec))

for coords in ras_decs:
    ra = coords[0]
    dec = coords[1]
    page_url = "https://irsa.ipac.caltech.edu/data/theory/Roman/Troxel2023/images/simple_model/coadd/"
    f184_image = f"dc2_F184_{ra}_{dec}.fits"
    os.system(f'wget -nc -P roman_data/original_fits/ {page_url+f184_image}')
    h158_image = f"dc2_H158_{ra}_{dec}.fits"
    os.system(f'wget -nc -P roman_data/original_fits/ {page_url+h158_image}')
    j129_image = f"dc2_J129_{ra}_{dec}.fits"
    os.system(f'wget -nc -P roman_data/original_fits/ {page_url+j129_image}')
    y106_image = f"dc2_Y106_{ra}_{dec}.fits"
    os.system(f'wget -nc -P roman_data/original_fits/ {page_url+y106_image}')
    