# Setting up imports
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.table import Table
from astropy import wcs
from astropy.coordinates import SkyCoord
from detectron2.structures import BoxMode
from deepdisc.data_format.conversions import convert_to_json

import cv2
import glob
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import astropy.units as u

# Extract RA and DEC from filename with regex
def extract_ra_dec(filename):
    matches = re.search(r'dc2_(det|seg|index)_(\d+\.\d+)_(\-\d+\.\d+)', filename)
    if matches:
        return matches.group(2), matches.group(3)
    return None

def create_cutouts(seg_file, full_img_data, ra_dec, w):
    seg = fits.open(seg_file)
    # Create Cutouts
    cutout_size = 512
    # Find step size
    x_steps = np.arange(0, seg[0].data.shape[1], cutout_size)
    y_steps = np.arange(0, seg[0].data.shape[0], cutout_size)
    # Finding the actual cutouts
    seg_cutouts = []
    counter = 0
    for y in y_steps:
        for x in x_steps:
            # partial mode and set fill_value to zero since we will ignore it anyway
            seg_cutout = Cutout2D(seg[0].data, position=(x + cutout_size/2, y + cutout_size/2), size=cutout_size, wcs=w, mode='partial', fill_value=0)
            seg_cutouts.append(seg_cutout)
            raw_cutout_f184 = Cutout2D(full_img_data[0], position=(x + cutout_size/2, y + cutout_size/2), size=cutout_size, wcs=w, mode='partial', fill_value=0)
            raw_cutout_h158 = Cutout2D(full_img_data[1], position=(x + cutout_size/2, y + cutout_size/2), size=cutout_size, wcs=w, mode='partial', fill_value=0)
            raw_cutout_y106 = Cutout2D(full_img_data[2], position=(x + cutout_size/2, y + cutout_size/2), size=cutout_size, wcs=w, mode='partial', fill_value=0)
            raw_cutout_j129 = Cutout2D(full_img_data[3], position=(x + cutout_size/2, y + cutout_size/2), size=cutout_size, wcs=w, mode='partial', fill_value=0)
            full_raw_cutout = np.stack((raw_cutout_f184.data, raw_cutout_h158.data, raw_cutout_y106.data, raw_cutout_j129.data)) 
            np.save(f'roman_data/truth/dc2_{ra_dec[0]}_{ra_dec[1]}/full_c{counter}_{ra_dec[0]}_{ra_dec[1]}.npy', full_raw_cutout)
            counter += 1
    
    return seg_cutouts 

# Creating annotations for each image using the detection file's RA and DEC
def create_annotations(det_file, seg_cutouts, truth_file, ra_dec):
    det = fits.open(det_file)
    det_df = Table.read(det,hdu=1).to_pandas()
    truth = fits.open(truth_file)
    truth_df = Table.read(truth,hdu=1).to_pandas()
    # Cross-Matching
    all_matched_objs_info = []
    all_metadata = []
    for imgid, seg_cutout in enumerate(seg_cutouts):
        # Getting All Unique Objects in Segmap Cutout
        seg_objs = []
        seg_img_cut = seg_cutout.data
        for s in np.unique(seg_img_cut):
            if s == 0: # background which we don't care about
                continue
            seg_objs.append(s)
        seg_objs = np.asarray(seg_objs)
        # Grabbing the DECs and RAs of the unique objs in this cutout
        det_objs = det_df.iloc[seg_objs-1] # -1 because we match these objs with the number col NOT index 
        ras = det_objs['alphawin_j2000'].values
        decs = det_objs['deltawin_j2000'].values
        det_SkyCoords = SkyCoord(ra=ras*u.degree, dec=decs*u.degree)
        truth_coords = SkyCoord(ra=truth_df['ra'].values*u.degree, dec=truth_df['dec'].values*u.degree)
        # Enforcing a separation constraint matched 
        # so we get sources in det and truth that are separated by less than .0575"
        max_sep = 0.0575 * u.arcsec # same as coadd pixel scale from https://academic.oup.com/mnras/article/522/2/2801/7076879?login=false
        idx_cutout, d2d, d3d = det_SkyCoords.match_to_catalog_sky(truth_coords)
        sep_constraint = d2d < max_sep
        if len(idx_cutout[sep_constraint]) > 0: # only take cutouts that pass separation constraint
            objects_info = truth_df.loc[idx_cutout[sep_constraint]].copy()
            # We need to store the wcs info for each cutout so we can find the sky coordinates for a given pixel in the cutout array. like so: from astropy.wcs.utils import pixel_to_skycoord pixel_to_skycoord(x_cutout, y_cutout, cutout.wcs)
            truth_info = { "image_id": imgid,
                          "file_name": f'./roman_data/truth/dc2_{ra_dec[0]}_{ra_dec[1]}/full_c{imgid}_{ra_dec[0]}_{ra_dec[1]}.npy',
                          "wcs": seg_cutout.wcs.to_header_string(),
                          "objects_info": objects_info.to_json(orient='records')
                         }
            all_matched_objs_info.append(truth_info)
            # Formatting Segmaps for DeepDisc
            matched_truths = {} # s : 0 or 1 whether it's a galaxy or star
            for idx, seg_obj in enumerate(seg_objs):
                # this obj has a separation less than .0575" and thus has a corressponding truth val
                if sep_constraint[idx]:
                    matched_truths[seg_obj] = int(truth_df.iloc[idx_cutout[idx]]['gal_star'])
            # getting metadata for this cutout
            cutout_metadata = get_metadata(seg_img_cut, imgid, matched_truths)
#             cutout_metadata["ra_dec"] = f'{ra_dec[0]}_{ra_dec[1]}' 
            cutout_metadata["file_name"] = f'./roman_data/truth/dc2_{ra_dec[0]}_{ra_dec[1]}/full_c{imgid}_{ra_dec[0]}_{ra_dec[1]}.npy'
            all_metadata.append(cutout_metadata)
    
    # Convert Metadata to JSON File
    metadata_filename = f"roman_data/annotations/dc2_{ra_dec[0]}_{ra_dec[1]}.json"
    objs_info_filename = f"roman_data/truth/dc2_{ra_dec[0]}_{ra_dec[1]}/dc2_{ra_dec[0]}_{ra_dec[1]}_info.json"
    convert_to_json(all_metadata, metadata_filename)
    convert_to_json(all_matched_objs_info, objs_info_filename)

# Creating metadata in the correct format for DeepDisc
def get_metadata(cutout_data, imid, matched_truth):
    d = {}
    annos = []
    for s in np.unique(cutout_data):
        mask = np.zeros(cutout_data.shape)
        if s == 0:
            continue
        s0i = np.where(cutout_data == s)
        mask[s0i] =1

        x0 = s0i[1].min()
        x1 = s0i[1].max()
        y0 = s0i[0].min()
        y1 = s0i[0].max()

        h = int(y1-y0)
        w = int(x1-x0)
        

        contours, hierarchy = cv2.findContours(
                (mask).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )

        segmentation = []
        for contour in contours:
            # contour = [x1, y1, ..., xn, yn]
            contour = contour.flatten()
            if len(contour) > 4:
                segmentation.append(contour.tolist())
        
        #use the truth matched indices to get object classes
        obj_class = matched_truth.get(s, 2) # -1 or 2 if no matching truth
        
        # No valid contours or no matching truth
        if len(segmentation) == 0 or obj_class == 2:
            continue
       
        obj = {
            "bbox": [int(x0), int(y0), w, h],
            "bbox_mode": BoxMode.XYWH_ABS,
            "area": w * h,
            "segmentation": segmentation,
            "category_id": obj_class,
            "obj_id": int(s),
        }
        annos.append(obj)

    height, width = mask.shape

    d["annotations"] = annos
    d['height'] = height
    d['width'] = width
    d["image_id"] = imid
    
    return d

# MAIN

detection_dir = 'roman_data/detection_fits'
segmentation_dir = 'roman_data/segmentation_fits'
truth_dir = 'roman_data/truth_fits'
# # Grab the filenames of 1 detection files
# det_files = [os.path.join(detection_dir, f) for idx, f in enumerate(os.listdir(detection_dir)) if f.endswith('fits.gz') if idx < 10]

det_files = ['roman_data/detection_fits/dc2_det_50.93_-42.0.fits.gz','roman_data/detection_fits/dc2_det_51.34_-41.3.fits.gz',
            'roman_data/detection_fits/dc2_det_51.37_-38.3.fits.gz','roman_data/detection_fits/dc2_det_51.53_-40.0.fits.gz',
            'roman_data/detection_fits/dc2_det_52.31_-41.6.fits.gz','roman_data/detection_fits/dc2_det_52.93_-40.8.fits.gz',
            'roman_data/detection_fits/dc2_det_53.25_-41.8.fits.gz','roman_data/detection_fits/dc2_det_53.75_-38.9.fits.gz',
            'roman_data/detection_fits/dc2_det_54.24_-38.3.fits.gz','roman_data/detection_fits/dc2_det_54.31_-41.6.fits.gz',
             'roman_data/detection_fits/dc2_det_55.03_-41.9.fits.gz','roman_data/detection_fits/dc2_det_56.06_-39.8.fits.gz'] 
# det_files = ['roman_data/detection_fits/dc2_det_50.93_-42.0.fits.gz', 
#             'roman_data/detection_fits/dc2_det_52.93_-40.8.fits.gz',
#             'roman_data/detection_fits/dc2_det_53.25_-41.8.fits.gz',
#             'roman_data/detection_fits/dc2_det_53.75_-38.9.fits.gz',
#             'roman_data/detection_fits/dc2_det_56.06_-39.8.fits.gz', roman_data/detection_fits/dc2_det_51.37_-38.3.fits.gz, roman_data/detection_fits/dc2_det_54.31_-41.6.fits.gz, (52.31, -41.6), (51.34, -41.3), (54.24, -38.3) (51.53, -40.0)]

# det_files = ['roman_data/detection_fits/dc2_det_51.53_-40.0.fits.gz']

for det_file in det_files:
    ra_dec = extract_ra_dec(det_file)
    os.makedirs(f'roman_data/truth/dc2_{ra_dec[0]}_{ra_dec[1]}', exist_ok=True)
    f184_img = fits.open(f'roman_data/original_fits/dc2_F184_{ra_dec[0]}_{ra_dec[1]}.fits')
    h158_img = fits.open(f'roman_data/original_fits/dc2_H158_{ra_dec[0]}_{ra_dec[1]}.fits')
    y106_img = fits.open(f'roman_data/original_fits/dc2_Y106_{ra_dec[0]}_{ra_dec[1]}.fits')
    j129_img = fits.open(f'roman_data/original_fits/dc2_J129_{ra_dec[0]}_{ra_dec[1]}.fits')
    full_img_data = np.stack((f184_img[1].data, h158_img[1].data, y106_img[1].data, j129_img[1].data))
    np.save(f'roman_data/truth/dc2_{ra_dec[0]}_{ra_dec[1]}/full_{ra_dec[0]}_{ra_dec[1]}.npy', full_img_data)
    w = wcs.WCS(f184_img[1].header)
    seg_file = f'{segmentation_dir}/dc2_seg_{ra_dec[0]}_{ra_dec[1]}.fits.gz'
    truth_file = f'{truth_dir}/dc2_index_{ra_dec[0]}_{ra_dec[1]}.fits.gz'
    seg_cutouts = create_cutouts(seg_file, full_img_data, ra_dec, w)
    create_annotations(det_file, seg_cutouts, truth_file, ra_dec)

# 110 seconds for each image

# need to download 53.00_-40.6, 55.80_-38.0, 52.50_-41.3