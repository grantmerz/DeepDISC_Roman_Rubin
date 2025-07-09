# In case you need to point to pre-existing scarlet install
import sys
# change these paths to your specific directories where deepdisc and detectron2 are stored
sys.path.insert(0, '/home/yse2/deepdisc/src')
sys.path.insert(0, '/home/yse2/detectron2')

import os, json, glob
import time
import argparse
import multiprocessing as mp
import warnings
import numpy as np
import pandas as pd

import cv2
import deepdisc
import detectron2
print(deepdisc.__file__)
print(detectron2.__file__)
import scarlet
import sep
# Print the versions to test the imports and so we know what works
print(scarlet.__version__)
print(np.__version__)
print(sep.__version__)

import galsim
import btk
from astropy.stats import gaussian_fwhm_to_sigma
#from galcheat.utilities import mag2counts, mean_sky_level
from astropy.stats import sigma_clipped_stats


# --- Configuration ---
root_dir = './lsst_data/'
lsst_dir = f'{root_dir}truth/'
rejected_mask_dir = f'{root_dir}rejected_objs_lvl5/rejected_masks/'

cutouts_per_tile = 225 # we shld prob just dynamically calculate this by looking through the truth folders
seed = 8312
rng = np.random.RandomState(seed)
grng = galsim.BaseDeviate(rng.randint(0, 2**30))


class SourceNotVisible(Exception):
    """Raised when source has no visible flux"""
    pass

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
    # pos angle
    cat['pa_bulge'] = e1e2_to_ephi(cat['ellipticity_1_bulge_true'],cat['ellipticity_2_bulge_true']) * 180.0/np.pi

    cat['pa_disk'] = e1e2_to_ephi(cat['ellipticity_1_disk_true'],cat['ellipticity_2_disk_true']) * 180.0/np.pi
    
    cat['pa_tot'] = e1e2_to_ephi(cat['ellipticity_1_true'],cat['ellipticity_2_true']) * 180.0/np.pi

    cat['g1'] = cat['shear_1']
    cat['g2'] = cat['shear_2']
    
    return cat

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
    #flux = mag2counts(mag,survey,filt).to_value("electron")
    delta_m = mag - 27
    flux = 10 ** (-delta_m / 2.5)
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
    #total_flux = mag2counts(entry[filt.name + "_ab"], survey, filt).to_value("electron")
    delta_m = entry[filt.name + "_ab"] - 27
    total_flux = 10 ** (-delta_m / 2.5)
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

def make_im(entry, survey, filt, nx=128, ny=128, get_gso=False):
    psf = survey.get_filter(filt).psf
    #sky_level = mean_sky_level(survey, filt).to_value('electron') # gain = 1
    obj_type = entry['truth_type'] # 1 for galaxies, 2 for stars
    im = None
    if obj_type == 1:
        gal = make_galaxy(entry, survey, survey.get_filter(filt))
        gal = gal.shear(g1=entry["g1"], g2=entry["g2"])
        conv_gal = galsim.Convolve(gal, psf)
#         im = conv_gal.drawImage(
#             nx=nx,
#             ny=ny,
#             scale=survey.pixel_scale.to_value("arcsec")
#         )
        if get_gso:
            return conv_gal, psf
    else:
        star, gsparams, flux = make_star(entry, survey, survey.get_filter(filt))
        max_n_photons = 10_000_000
        # 0 means use the flux for n_photons 
        n_photons = 0 if flux < max_n_photons else max_n_photons
        # n_photons = 0 if entry[f'flux_{filt}'] < max_n_photons else max_n_photons
        conv_star = galsim.Convolve(star, psf)
#         im = conv_star.drawImage(
#             nx=nx,
#             ny=ny,
#             scale=survey.pixel_scale.to_value("arcsec"),
#             method="phot",
#             n_photons=n_photons,
#             poisson_flux=True,
#             maxN=1_000_000,  # shoot in batches this size
#             rng=grng
#         )
        if get_gso:
            return conv_star, psf
    return im, psf

# uses combined mask
def get_bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmasx = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin-4, rmax+4, cmin-4, cmax+4

def load_cutout_data(tile, cutout_id, wcs_header=None):
    """Load essential truth catalog data for a specific cutout"""
    base_filename = f'c{cutout_id}_{tile}'
    paths = {
        'lsst_img': f'{lsst_dir}dc2_{tile}/full_{base_filename}.npy',
        'truth_cat': f'{lsst_dir}dc2_{tile}/truth_{base_filename}.json',
        'det_cat': f'{lsst_dir}dc2_{tile}/det_{base_filename}.json',
        'matched_det': f'{lsst_dir}dc2_{tile}/matched_{base_filename}.json'
    }

    if not os.path.exists(paths['lsst_img']):
        # we can handle this in the main loop by creating empty dictionaries
        print(f"Warning: Image file for cutout {cutout_id} from tile {tile} does not exist! Skipping.")
        return None

    try:
        lsst_img = np.load(paths['lsst_img'])
        height, width = lsst_img.shape[1], lsst_img.shape[2]
        #estimate sig-clipped background noise
        noise = np.array([sigma_clipped_stats(img)[-1] for img in lsst_img])
        del lsst_img
        # load truth cat if it exists, otherwise create empty dfs
        truth_cat = pd.read_json(paths['truth_cat'], orient='records') if os.path.exists(paths['truth_cat']) else pd.DataFrame()
        return {
            'tile': tile,
            'cutout_id': cutout_id,
            'truth_cat': truth_cat,
            'width': width,
            'height': height,
            'wcs': wcs_header,
            'paths': paths,
            'noise': noise
        }
    except Exception as e:
        print(f"Error loading cutout {cutout_id} from tile {tile}: {e}")
        return None

def process_object(obj_entry, survey, se_kernel, obj_idx, tile, cutout_id, noise, snr_lvl=5,):
    """
    Processes a single object from the truth catalog to generate an annotation or a rejection log.
    """
    filters = ['u', 'g', 'r', 'i', 'z', 'y']
    segs = []

    # simulate the object and create segmentation masks for each band
    for i,filt in enumerate(filters):
        try:
            # isolated image of object
            im_conv, psf = make_im(obj_entry, survey, filt, nx=128, ny=128, get_gso=True)
            #convolve by the psf and threshold with noise multiplied by psf area (Bosch 2018)
            im_conv2 = galsim.Convolve(im_conv, psf)
            image = galsim.Image(128, 128, scale=survey.pixel_scale.to_value("arcsec"))
            im2 = im_conv2.drawImage(image, scale=survey.pixel_scale.to_value("arcsec"), method='no_pixel')
            # estimate of PSF area
            #psf_fac = np.pi * (gaussian_fwhm_to_sigma * psf.calculateFWHM())**2
            psfim = psf.drawImage(nx=64,ny=64)
            psf_fac = np.sum(psfim.array**2)
            imd = np.expand_dims(np.expand_dims(im2.array, 0), 0)
            #sky_level = mean_sky_level(survey, filt).to_value('electron') # gain = 1
            #noise from the cutout image
            var = noise[i]**2
            # segmentation mask using the fixed SNR threshold
            maskf = btk.metrics.utils.get_segmentation(imd, var * psf_fac, sigma_noise=snr_lvl)[0][0]
            # dilate the mask by the psf size
            maskf = cv2.morphologyEx(maskf, cv2.MORPH_DILATE, se_kernel)   
            segs.append(maskf)
        except Exception as e:
            # assume no Galsim failures.
            # For now, we create an empty mask to trigger rejection.
            print(f"Galsim failed for {obj_idx} in {filt}! Below is the error:\n {e}")
            segs.append(np.zeros((128, 128), dtype=bool))

    # add masks in separate bands
    combined_mask = np.clip(np.sum(segs,axis=0), a_min=0, a_max=1)
    obj_id = int(obj_entry['id']) if obj_entry['truth_type'] == 2 else int(obj_entry['cosmodc2_id'])
    
    # now when an obj is rejected, save its mask and store the path
    def handle_rejection(reason):
        mask_dir = os.path.join(rejected_mask_dir, tile, f"c{cutout_id}")
        os.makedirs(mask_dir, exist_ok=True)
        mask_path = os.path.join(mask_dir, f"mask_{obj_idx}.npy")
        np.save(mask_path, combined_mask)
        rejected_obj = {
            'obj_id': obj_id,
            'obj_truth_idx': obj_idx,
            'category_id': 1 if obj_entry['truth_type'] == 2 else 0,
            'ra': obj_entry['ra'],
            'dec': obj_entry['dec'],
            'redshift': obj_entry['redshift'],
            'size_true': obj_entry['size_true'],
            'ellipticity_1_true': obj_entry['ellipticity_1_true'],
            'ellipticity_2_true': obj_entry['ellipticity_2_true'],
            'reason': reason,
            'mask_path': mask_path
        }
        for filt in filters:
            rejected_obj[f'mag_{filt}'] = obj_entry[f'mag_{filt}'] # AB mag values
            rejected_obj[f'flux_{filt}'] = obj_entry[f'flux_{filt}'] # flux values (nJy)
        return 'rejected', rejected_obj
    
    if np.sum(combined_mask) == 0:
        return handle_rejection('empty_mask')

    if np.sum(combined_mask) < 12:
        return handle_rejection('small_mask_12px')

    bbox_coords = get_bbox(combined_mask)
    if bbox_coords is None:
        return handle_rejection('invalid_bbox') # Should be caught by empty mask, but just in case
    
    y0, y1, x0, x1 = bbox_coords
    w, h = x1 - x0, y1 - y0
    
    # bbox relative to full cutout coordinates
    cutout_x, cutout_y = int(obj_entry['cutout_x']), int(obj_entry['cutout_y'])
    bbox = [cutout_x - w/2, cutout_y - h/2, w, h]

    # contours for segmentation
    contours, _ = cv2.findContours((combined_mask).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    for contour in contours:
        contour = contour.flatten()
        if len(contour) > 4: # Must have at least 3 points
            # Adjust contour coordinates to be relative to the full cutout
            contour[::2] += int(np.rint(cutout_x)) - (x0 + w//2)
            contour[1::2] += int(np.rint(cutout_y)) - (y0 + h//2)
            segmentation.append(contour.tolist())

    if len(segmentation) == 0:
        return handle_rejection('invalid_contours')

    # successful ann
    obj_md = {
        'obj_id': obj_id,
        'obj_truth_idx': obj_idx,     
        'category_id': 1 if obj_entry['truth_type'] == 2 else 0, # 2 is star, 1 is galaxy truth_type
        'bbox': bbox,
        'bbox_mode': 1, # BoxMode.XYWH_ABS
        'area': w * h,
        'segmentation': segmentation,
        'ra': obj_entry['ra'],
        'dec': obj_entry['dec'],
        'redshift': obj_entry['redshift'],
        'size_true': obj_entry['size_true'],
        'ellipticity_1_true': obj_entry['ellipticity_1_true'],
        'ellipticity_2_true': obj_entry['ellipticity_2_true']  
    }
    for filt in filters:
        obj_md[f'mag_{filt}'] = obj_entry[f'mag_{filt}'] # AB mag values
        obj_md[f'flux_{filt}'] = obj_entry[f'flux_{filt}'] # flux values (nJy)
        
    return 'success', obj_md

def process_cutout(cutout_data, snr_lvl=5):
    """Generates annotation dicts for all objects in a given cutout."""
    # cutout file does not exist but we still need an empty entry
    if cutout_data is None:
        return {}, {}

    truth_cat = cutout_data['truth_cat']
    
    base_dict = {
        "file_name": cutout_data['paths']['lsst_img'],
        "image_id": cutout_data['cutout_id'],
        "height": cutout_data['height'],
        "width": cutout_data['width'],
        "tile": cutout_data['tile'],
        "det_cat_path": cutout_data['paths']['det_cat'],
        "truth_cat_path": cutout_data['paths']['truth_cat'],
        "matched_det_path": cutout_data['paths']['matched_det'],
        "wcs": cutout_data['wcs']
    }

    if truth_cat.empty:
        # empty dicts 
        print(f"Skipping cutout {cutout_data['cutout_id']} with empty truth catalog.")
        success_dict = {**base_dict, "annotations": []}
        rejected_dict = {**base_dict, "rejected_objs": []}
        return success_dict, rejected_dict

    survey = btk.survey.get_surveys("LSST")
    # structuring element for mask dilation
    # The psf variations are small between bands, so just using i-band is ok
    fwhm = survey.get_filter('i').psf.calculateFWHM()
    sig = gaussian_fwhm_to_sigma * fwhm / survey.pixel_scale.to_value("arcsec")
    se_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(2 * sig), int(2 * sig)))

    dcut = dcut_reformat(truth_cat)
    
    annotations = []
    rejected_objs = []
    for idx, obj in dcut.iterrows():
        status, result_dict = process_object(obj, survey, se_kernel, idx, 
                                             cutout_data['tile'], 
                                             cutout_data['cutout_id'], 
                                             cutout_data['noise'],snr_lvl)
        if status == 'success':
            annotations.append(result_dict)
        else:
            rejected_objs.append(result_dict)
    
    success_dict = {**base_dict, "annotations": annotations}
    rejected_dict = {**base_dict, "rejected_objs": rejected_objs}
    
    return success_dict, rejected_dict

def process_single_cutout_wrapper(args):
    """
    A helper function to unpack args for use with multiprocessing.Pool
    It loads and processes a single cutout
    """
    tile_name, cutout_id, snr_lvl, wcs_header = args
    print(f"Processing {tile_name} | Cutout {cutout_id}...")
    cutout_data = load_cutout_data(tile_name, cutout_id, wcs_header)
    success_ddict, rejected_ddict = process_cutout(cutout_data, snr_lvl)
    return success_ddict, rejected_ddict

def process_and_save_tile(tile_name, snr_lvl=5):
    """
    Processes all cutouts for a given tile in parallel and saves the results.
    """
    print(f"--- Starting parallel processing for tile: {tile_name} ---")
    
    # lookup dict for wcs headers
    wcs_df = pd.read_json(f"{lsst_dir}dc2_{tile_name}/wcs_{tile_name}.json")
    wcs_lookup = wcs_df.set_index('cutout_id')['wcs_header'].to_dict()

    # args for each task
    tasks = [(tile_name, cutout_id, snr_lvl, wcs_lookup.get(cutout_id, None)) 
             for cutout_id in range(cutouts_per_tile)]
    
    num_processes = int(os.environ.get("SLURM_CPUS_ON_NODE", 16))
    print(f"Creating a pool of {num_processes} worker processes.")
    with mp.Pool(processes=num_processes) as pool:
        # pool.map distributes the tasks and blocks until all are complete
        results = pool.map(process_single_cutout_wrapper, tasks)
        pool.close()
        pool.join()

    tile_md = [res[0] for res in results if res[0]]
    tile_rejected_md = [res[1] for res in results if res[1]]

    print(f"\n--- Finished processing for tile: {tile_name} ---")
    print(f"Aggregated {len(tile_md)} successful annotation sets.")
    print(f"Aggregated {len(tile_rejected_md)} rejected object sets.")

    output_dir_ann = f"./lsst_data/annotations_lvl{snr_lvl}/"
    output_dir_rej = f"./lsst_data/rejected_objs_lvl{snr_lvl}/"
    os.makedirs(output_dir_ann, exist_ok=True)
    os.makedirs(output_dir_rej, exist_ok=True)

    ann_path = os.path.join(output_dir_ann, f"dc2_{tile_name}.json")
    rej_path = os.path.join(output_dir_rej, f"dc2_{tile_name}.json")

    if tile_md:
        print(f"Saving successful annotations to {ann_path}")
        pd.DataFrame(tile_md).to_json(ann_path, orient='records', indent=4)
    if tile_rejected_md:
        print(f"Saving rejected objects to {rej_path}")
        pd.DataFrame(tile_rejected_md).to_json(rej_path, orient='records', indent=4)

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    parser = argparse.ArgumentParser(description="Process a single LSST DC2 tile for ground truth annotations.")
    parser.add_argument("tile_name", type=str, help="The name of the tile to process (e.g., '53.25_-41.8').")
    parser.add_argument("--snr", type=int, default=5, help="SNR level (default: 5)")
    args = parser.parse_args()
    
    start_time = time.time()
    process_and_save_tile(args.tile_name, snr_lvl=args.snr)
    end_time = time.time()
    
    print(f"\nPipeline finished for tile {args.tile_name}. Took {end_time - start_time:.2f} seconds.")
