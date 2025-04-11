# In case you need to point to pre-existing scarlet install
import sys
# change these paths to your specific directories where deepdisc and detectron2 are stored
sys.path.insert(0, '/home/yse2/deepdisc/src')
sys.path.insert(0, '/home/yse2/detectron2')
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
# warnings.filterwarnings("ignore", category=DtypeWarning)
import deepdisc
import detectron2
print(deepdisc.__file__)
print(detectron2.__file__)
from detectron2.data import MetadataCatalog, DatasetCatalog

# Standard imports
import os, json
import numpy as np
import pandas as pd
import time
import math
import glob
import scarlet
import cv2
import argparse
# for multiprocessing
import multiprocessing
from functools import partial
import psutil

# astropy
import astropy.io.fits as fits
import astropy.units as u
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy.table import Table
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.visualization import make_lupton_rgb

# Astrodet imports
from deepdisc.preprocessing.get_data import get_cutout
from deepdisc.astrodet.hsc import get_tract_patch_from_coord, get_hsc_data
from deepdisc.astrodet.visualizer import ColorMode
from deepdisc.astrodet.visualizer import Visualizer

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

from galcheat.utilities import mag2counts, mean_sky_level
from btk.survey import Filter, Survey, make_wcs
import galsim
import btk
from astropy.stats import gaussian_fwhm_to_sigma


def parse_args():
    parser = argparse.ArgumentParser(description="Create custom ground truth annotations")   
    parser.add_argument('--data-path', type=str, default='/home/yse2/lsst_data/truth/', help='Path to the LSST cutouts and truth catalogs')
    args = parser.parse_args()
    return args

def e1e2_to_ephi(e1,e2):
    
    pa = np.arctan(e2/e1)
    
    return pa

def dcut_reformat(cat):
    L0 = 3.0128e28
    for band in ['u', 'g', 'r', 'i', 'z', 'y']:
        cat[f'{band}_ab'] = cat[f'mag_true_{band}']
        total_flux = L0 * 10**(-0.4*cat[f'mag_true_{band}'])
        bulge_to_total_ratio = cat[f'bulge_to_total_ratio_{band}']

        cat[f'fluxnorm_bulge_{band}'] = total_flux * bulge_to_total_ratio
        cat[f'fluxnorm_disk_{band}'] = total_flux * (1-bulge_to_total_ratio)
        cat[f'fluxnorm_agn_{band}'] = np.zeros(total_flux.shape)

    cat['a_b'] = cat['size_bulge_true']
    cat['b_b'] = cat['size_minor_bulge_true']

    cat['a_d'] = cat['size_disk_true']
    cat['b_d'] = cat['size_minor_disk_true']

    cat['pa_bulge'] = e1e2_to_ephi(cat['ellipticity_1_bulge_true'],cat['ellipticity_2_bulge_true']) * 180.0/np.pi

    cat['pa_disk'] = e1e2_to_ephi(cat['ellipticity_1_disk_true'],cat['ellipticity_2_disk_true']) * 180.0/np.pi
    
    cat['pa_tot'] = e1e2_to_ephi(cat['ellipticity_1_true'],cat['ellipticity_2_true']) * 180.0/np.pi

    cat['g1'] = cat['shear_1']
    cat['g2'] = cat['shear_2']
    
    return cat

seed = 8312
rng = np.random.RandomState(seed)
grng = galsim.BaseDeviate(rng.randint(0, 2**30))

def get_star_gsparams(mag, flux, noise):
    """
    Get appropriate gsparams given flux and noise

    Parameters
    ----------
    mag: float
        mag of star
    flux: float
        flux of star
    noise: float
        noise of image

    Returns
    --------
    GSParams, isbright where isbright is true for stars with mag less than 18
    """
    do_thresh = do_acc = False
    if mag < 18:
        do_thresh = True
    if mag < 15:
        do_acc = True

    if do_thresh or do_acc:
        isbright = True

        kw = {}
        if do_thresh:

            # this is designed to quantize the folding_threshold values,
            # so that there are fewer objects in the GalSim C++ cache.
            # With continuous values of folding_threshold, there would be
            # a moderately largish overhead for each object.

            folding_threshold = noise/flux
            folding_threshold = np.exp(
                np.floor(np.log(folding_threshold))
            )
            kw['folding_threshold'] = min(folding_threshold, 0.005)

        if do_acc:
            kw['kvalue_accuracy'] = 1.0e-8
            kw['maxk_threshold'] = 1.0e-5

        gsparams = galsim.GSParams(**kw)
    else:
        gsparams = None
        isbright = False

    return gsparams, isbright


def make_star(entry, survey, filt):
    """
    Parameters
    ----------
    survey: WLDeblendSurvey or BasicSurvey
        The survey object
    band: string
        Band string, e.g. 'r'
    i: int
        Index of object
    noise: float
        The noise level, needed for setting gsparams

    Returns
    -------
    galsim.GSObject
    """    
    #https://pipelines.lsst.io/v/DM-22499/cpp-api/file/_photo_calib_8h.html
#     mag = -2.5*np.log10(entry[f'flux_{filt.name}']*1e-9/(1e23*10**(48.6/-2.5)))
    mag = entry[f'mag_{filt.name}']
    flux = mag2counts(mag,survey,filt).to_value("electron")
#     flux = entry[f'flux_{filt.name}']
    noise = mean_sky_level(survey, filt).to_value('electron') # gain = 1
    gsparams, isbright = get_star_gsparams(mag, flux, noise)
    star = galsim.Gaussian(
        fwhm=1.0e-4,
        flux=flux,
        gsparams=gsparams,
    )
    return star, gsparams, flux

def make_galaxy(entry, survey, filt, no_disk= False, no_bulge = False, no_agn = True):
    components = []
    total_flux = mag2counts(entry[filt.name + "_ab"], survey, filt).to_value("electron")
    # Calculate the flux of each component in detected electrons.
    total_fluxnorm = entry["fluxnorm_disk_"+filt.name] + entry["fluxnorm_bulge_"+filt.name] + entry["fluxnorm_agn_"+filt.name]
    disk_flux = 0.0 if no_disk else entry["fluxnorm_disk_"+filt.name] / total_fluxnorm * total_flux
    bulge_flux = 0.0 if no_bulge else entry["fluxnorm_bulge_"+filt.name] / total_fluxnorm * total_flux
    agn_flux = 0.0 if no_agn else entry["fluxnorm_agn_"+filt.name] / total_fluxnorm * total_flux

    if disk_flux + bulge_flux + agn_flux == 0:
        raise SourceNotVisible

    if disk_flux > 0:
        a_d, b_d = entry["a_d"], entry["b_d"]
        disk_hlr_arcsecs=entry['size_disk_true']
        
        
        disk_q = entry['size_minor_disk_true']/entry['size_disk_true']
        pa = np.pi*entry['position_angle_true_dc2']/180
        
        epsilon_disk = (1 - disk_q) / (1 + disk_q)
        
        e1_disk = epsilon_disk * np.cos(2 * pa)
        e2_disk = epsilon_disk * np.sin(2 * pa)

        disk = galsim.Exponential(flux=disk_flux, half_light_radius=disk_hlr_arcsecs).shear(
            e1=-e1_disk, e2=e2_disk
        )
        
        components.append(disk)
        
        
    if bulge_flux > 0:
        a_b, b_b = entry["a_b"], entry["b_b"]
        bulge_hlr_arcsecs = np.sqrt(a_b * b_b)

        bulge_q = entry['size_minor_bulge_true']/entry['size_bulge_true']

        pa = np.pi*entry['position_angle_true_dc2']/180

        
        epsilon_bulge = (1 - bulge_q) / (1 + bulge_q)
        
        e1_bulge = epsilon_bulge * np.cos(2 * pa)
        e2_bulge = epsilon_bulge * np.sin(2 * pa)
        
        bulge = galsim.DeVaucouleurs(flux=bulge_flux, half_light_radius=bulge_hlr_arcsecs).shear(
           e1=-e1_bulge, e2=e2_bulge
        )
        components.append(bulge)

    if agn_flux > 0:
        agn = galsim.Gaussian(flux=agn_flux, sigma=1e-8)
        components.append(agn)

    profile = galsim.Add(components)
    return profile

def make_im(entry, survey, filt, lvl, nx=128, ny=128,get_gso=False):
    psf = survey.get_filter(filt).psf
    sky_level = mean_sky_level(survey, filt).to_value('electron') # gain = 1
    obj_type = entry['truth_type'] # 1 for galaxies, 2 for stars
    im = None
    if obj_type == 1:
        gal = make_galaxy(entry, survey, survey.get_filter(filt))
        gal = gal.shear(g1=entry["g1"], g2=entry["g2"])
        conv_gal = galsim.Convolve(gal, psf)
        im = conv_gal.drawImage(
            nx=nx,
            ny=ny,
            scale=survey.pixel_scale.to_value("arcsec")
        )
        
        if get_gso:
            return im, conv_gal
        else:
            return im
    
    else:
        star, gsparams, flux = make_star(entry, survey, survey.get_filter(filt))
        max_n_photons = 10_000_000
        # 0 means use the flux for n_photons 
        n_photons = 0 if flux < max_n_photons else max_n_photons
        # n_photons = 0 if entry[f'flux_{filt}'] < max_n_photons else max_n_photons
        conv_star = galsim.Convolve(star, psf)
        im = conv_star.drawImage(
            nx=nx,
            ny=ny,
            scale=survey.pixel_scale.to_value("arcsec"),
            method="phot",
            n_photons=n_photons,
            poisson_flux=True,
            maxN=1_000_000,  # shoot in batches this size
            rng=grng
        )
        
        if get_gso:
            return im, conv_star
        else:
            return im   
#     imd = np.expand_dims(np.expand_dims(im.array,0),0)
#     # thresh for mask set relative to the bg noise level which is what sigma_noise is
#     # so lower the thresh for the star to include more of its light
#     # so lower sigma_noise, bigger masks and higher lvl, smaller masks bc it'll only capture very brightest central part of star
# #     if obj_type == 2: # if star, 
# #         segs = btk.metrics.utils.get_segmentation(imd, sky_level, sigma_noise=lvl * 0.02)
# #     else:
# #         segs = btk.metrics.utils.get_segmentation(imd, sky_level, sigma_noise=lvl) 
#     segs = btk.metrics.utils.get_segmentation(imd, sky_level, sigma_noise=lvl) 
#     return segs[0][0], mag

def get_bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin-4, rmax+4, cmin-4, cmax+4

def create_metadata(img_shape, cat, imid, sub_patch, filename, survey, filters, SE, lvl=2):

    """ Code to format the metadatain to a dict.  It takes the i-band and makes a footprint+bounding boxes
    from thresholding to sn*sky_level
    
    Parameters
    
    blend_batch: BTK blend batch
        BTK batch of blends
    sky_level: float
        The background sky level in the i-band
    sn: int
        The signal-to-noise ratio for thresholding
    idx:
        The index of the blend in the blend_batch
        
    Returns
        ddict: dict
            The dictionary of metadata for the idx'th blend in the batch 
    
    """
    

    ddict = {}

    ddict[f"file_name"] = filename
    ddict["image_id"] = imid # need to change this to use truth_info imid
    ddict["height"] = img_shape[0]
    ddict["width"] = img_shape[1]
    ddict["subpatch"] = sub_patch # need to use subpatch dir here
    
    t = Table.from_pandas(cat)
    n = len(cat)
    objs = []
    for j in range(n):

        obj = t[j]
        x = int(obj['new_x'])
        y = int(obj['new_y'])
        
        segs = []
        for filt in ['u','g','r','i','z','y']:
            im,im_conv  = make_im(obj, survey, filt, lvl=lvl, nx=128,ny=128, get_gso=True)
            psf = survey.get_filter(filt).psf
            #convolve by the psf and threshold with noise multiplied by psf area (Bosch 2018)
            im_conv2 = galsim.Convolve(im_conv, psf)
            image = galsim.Image(128,128, scale=survey.pixel_scale.to_value("arcsec"))
            im2 = im_conv2.drawImage(image,scale=scale,method='no_pixel')
            #estimate of PSF area
            #psf_fac = np.pi*psf.calculateMomentRadius()**2
            psf_fac = np.pi*(gaussian_fwhm_to_sigma*psf.calculateFWHM())**2
            imd = np.expand_dims(np.expand_dims(im2.array,0),0)
            sky_level = mean_sky_level(survey, filt).to_value('electron')
            maskf = btk.metrics.utils.get_segmentation(imd, sky_level*psf_fac, sigma_noise=lvl)[0][0]
            #dilate by the psf size
            maskf = cv2.morphologyEx(maskf, cv2.MORPH_DILATE, SE)   
            segs.append(maskf)
        #add masks in separate bands
        mask = np.clip(np.sum(segs,axis=0), a_min=0, a_max=1)

        #final check to remove too small masks
        if np.sum(mask)==0 or np.sum(mask)<12:
            continue
        
        bbox = get_bbox(mask)
        x0 = bbox[2]
        x1 = bbox[3]
        y0 = bbox[0]
        y1 = bbox[1]
        
        w = x1-x0
        h = y1-y0
        
        bbox = [x-w/2, y-h/2, w, h]     

        redshift = obj['redshift']

        contours, hierarchy = cv2.findContours(
                    (mask).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                )


        segmentation = []
        for contour in contours:
            # contour = [x1, y1, ..., xn, yn]
            contour = contour.flatten()
            if len(contour) > 4:
                contour[::2] += (int(np.rint(x))-x0-w//2)
                contour[1::2] += (int(np.rint(y))-y0-h//2)
                segmentation.append(contour.tolist())
        # No valid countors
        if len(segmentation) == 0:
            print(j)
            continue
        name = f"mag_r"
        obj = {
            "obj_id": j, # will be used to access specific obj's morphological/other info
            "bbox": bbox,
            "area": w*h,
            "bbox_mode": 1,
            "segmentation": segmentation,
            "category_id": 1 if obj['truth_type'] == 2 else 0,
            "redshift": redshift,
            name: obj['mag_r'] # ab mag
        }
        objs.append(obj)

    ddict['annotations'] = objs

    return ddict

def get_num_processes():
#     num_cpus = 8 # for jupyter notebook
    num_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK'))
#     total_mem = 4 # GB for jupyter notebook
    total_mem = int(os.environ.get('SLURM_MEM_PER_NODE')) / 1024
    
    # estimated mem per process
    mem_per_process = 0.5  # GB
    # num of processes
    num_processes = int(total_mem // mem_per_process)
    # optimal process count
    num_optimal_processes = min(num_cpus, num_processes)
#     print(num_optimal_processes)
    # use at least 1/4 of CPUs but not less than 1
    return max(num_optimal_processes, max(1, num_cpus // 4))
#     return max(1, num_optimal_processes)

def process_subpatch(sub_patch, truth_dict, filters, dirpath, survey, SE):
#     print(f"\nProcessing Sub-Patch: {sub_patch}")
    subpatch_metadata = []
    # grab each cutout and their corresponding truth info
    for entry in truth_dict:
        filename = dirpath + f"{sub_patch}/full_c{entry['image_id']}_{sub_patch[4:]}.npy"
        img_shape = (entry['height'], entry['width']) # height, width
        cat = pd.read_json(entry['obj_catalog'], orient='records')
        dcut = dcut_reformat(cat)
        ddict = create_metadata(img_shape, dcut, entry['image_id'], sub_patch, filename, survey, filters, SE, lvl=5)
        subpatch_metadata.append(ddict)
    
    df = pd.DataFrame(subpatch_metadata)
#     output_file = f'/home/shared/hsc/roman_lsst/lsst_data/annotations/{sub_patch}.json'
    output_file = f'/home/yse2/lsst_data/annotations/{sub_patch}.json'
    df.to_json(output_file, orient='records')
#     with open(output_file, 'w') as f:
#         json.dump(subpatch_metadata, f, indent=2)
    return f"Processed {sub_patch}"
    
def main(args):
    filters = ['u','g','r','i','z','y']
    dirpath = args.data_path
    survey = btk.survey.get_surveys("LSST")
    sub_patches = ['dc2_55.03_-41.9', 'dc2_56.06_-39.8']
    
    #Dilates the masks by a kernel the size of the PSF 
    #The psf variations are small between bands, so just using i-band is ok
    fwhm = survey.get_filter('i').psf.calculateFWHM()
    sig = gaussian_fwhm_to_sigma*fwhm/.2
    SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(2*sig),int(2*sig)))
    
    
    print("\nReading in the Truth Catalogs for each subpatch")
    print("------------------")
    prepared_data = {}
    for sub_patch in sub_patches:
        with open(dirpath + f'{sub_patch}/{sub_patch}_info.json', 'r') as f:
            truth_dict = json.load(f)
        prepared_data[sub_patch] = truth_dict
    
    num_processes = get_num_processes()
    print(f"Using {num_processes} processes")
    
    # partial funcs like in cs421!
    process_subpatch_partial = partial(process_subpatch, filters=filters, dirpath=dirpath, survey=survey, SE=SE)

    with multiprocessing.Pool(processes=2) as pool:
        results = pool.starmap(process_subpatch_partial, prepared_data.items())
    
    for result in results:
        print(result)
    
if __name__ == "__main__":
    start_time = time.time()
    os.environ["OMP_NUM_THREADS"] = "1" # ensures each process created by mp uses only one thread
    args = parse_args()
    main(args)
    end_time = time.time()
    print(end_time - start_time, " seconds")
    # 1022.3751366138458 s ~ 17 mins for one subpatch that had 280 entries

# def add_roman_to_lsst(file, lsst_img, root_dir):
#     # grabbing selected Roman image data
#     curr_rimg_filename = file.replace('./roman_data/', root_dir)
#     roman_im = np.load(curr_rimg_filename)
#     # print("Old File: ", file)
    
#     new_filename = file.replace('truth', 'truth-combined')
#     cutout_filename = (re.search(r'roman_data/(.+)', new_filename)).group(1)
#     # print(cutout_filename)
#     full_cutout_filename = f'{root_dir}{cutout_filename}'
#     # print("Roman: ", roman_im.shape)
#     # print(roman_im[0, :, :])
#     # print("LSST: ", np.asarray(upsampled_lsst_imgs_lz).shape)
#     # print(np.asarray(upsampled_lsst_imgs_lz)[-1, :, :])
    
#     combined_data = np.concatenate((roman_im, upsampled_lsst_img), axis=0)
#     np.save(full_cutout_filename, combined_data)
