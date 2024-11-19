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
    for y in y_steps:
        for x in x_steps:
            # partial mode and set fill_value to zero since we will ignore it anyway
            seg_cutout = Cutout2D(seg[0].data, position=(x + cutout_size/2, y + cutout_size/2), size=cutout_size, wcs=w, mode='partial', fill_value=0)
            seg_cutouts.append(seg_cutout)    
    return seg_cutouts 

def create_annotations(det_file, seg_cutouts, truth_file, ra_dec):
    det = fits.open(det_file)
    det_df = Table.read(det,hdu=1).to_pandas()
    truth = fits.open(truth_file)
    truth_df = Table.read(truth,hdu=1).to_pandas()
    # Cross-Matching
    all_matched_objs_info = []
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
            # Include all objects from this cutout in truth_info, regardless of separation constraint
            objects_info = truth_df.loc[idx_cutout].copy()
            objects_info['passes_constraint'] = sep_constraint  # indicate if it passes the constraint

            truth_info = {
                "image_id": imgid,
                "file_name": f'./roman_data/truth/dc2_{ra_dec[0]}_{ra_dec[1]}/full_c{imgid}_{ra_dec[0]}_{ra_dec[1]}.npy',
                "wcs": seg_cutout.wcs.to_header_string(),
                "objects_info": objects_info.to_json(orient='records')
            }
            all_matched_objs_info.append(truth_info)
    
    # Convert Truth Info to JSON File
    objs_info_filename = f"roman_data/truth/dc2_{ra_dec[0]}_{ra_dec[1]}/dc2_{ra_dec[0]}_{ra_dec[1]}_info_all.json"
    convert_to_json(all_matched_objs_info, objs_info_filename)

    
detection_dir = 'roman_data/detection_fits'
segmentation_dir = 'roman_data/segmentation_fits'
truth_dir = 'roman_data/truth_fits'
det_files = ['roman_data/detection_fits/dc2_det_50.93_-42.0.fits.gz','roman_data/detection_fits/dc2_det_51.34_-41.3.fits.gz',
            'roman_data/detection_fits/dc2_det_51.37_-38.3.fits.gz','roman_data/detection_fits/dc2_det_51.53_-40.0.fits.gz',
            'roman_data/detection_fits/dc2_det_52.31_-41.6.fits.gz','roman_data/detection_fits/dc2_det_52.93_-40.8.fits.gz',
            'roman_data/detection_fits/dc2_det_53.25_-41.8.fits.gz','roman_data/detection_fits/dc2_det_53.75_-38.9.fits.gz',
            'roman_data/detection_fits/dc2_det_54.24_-38.3.fits.gz','roman_data/detection_fits/dc2_det_54.31_-41.6.fits.gz',
             'roman_data/detection_fits/dc2_det_55.03_-41.9.fits.gz','roman_data/detection_fits/dc2_det_56.06_-39.8.fits.gz']

for det_file in det_files:
    ra_dec = extract_ra_dec(det_file)
    seg_file = f'{segmentation_dir}/dc2_seg_{ra_dec[0]}_{ra_dec[1]}.fits.gz'
    truth_file = f'{truth_dir}/dc2_index_{ra_dec[0]}_{ra_dec[1]}.fits.gz'

    f184_img = fits.open(f'roman_data/original_fits/dc2_F184_{ra_dec[0]}_{ra_dec[1]}.fits')
    w = wcs.WCS(f184_img[1].header)
    full_img_data = np.load(f'roman_data/truth/dc2_{ra_dec[0]}_{ra_dec[1]}/full_{ra_dec[0]}_{ra_dec[1]}.npy')
    seg_cutouts = create_cutouts(seg_file, full_img_data, ra_dec, w)
    create_annotations(det_file, seg_cutouts, truth_file, ra_dec)