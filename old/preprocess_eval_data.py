# Common Libraries
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import os
import sys
import json
import torch
import argparse
import gc
import time
import glob
import random
import ssl
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Astropy imports
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
from astropy.table import Table, Column
from astropy.nddata import Cutout2D
from astropy.visualization import make_lupton_rgb

# Detectron2 setup
import detectron2.utils.comm as comm
from detectron2.utils.logger import setup_logger
setup_logger()  # Setup Detectron2 logger
from detectron2.config import LazyConfig, get_cfg
from detectron2.engine import launch
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2 import structures
from detectron2.structures import BoxMode, Boxes
import detectron2.data as d2data

# DeepDisc imports
from deepdisc.data_format.image_readers import RomanImageReader
from deepdisc.data_format.register_data import register_data_set
from deepdisc.inference.predictors import return_predictor_transformer, get_predictions
from deepdisc.astrodet.visualizer import Visualizer, ColorMode
import deepdisc.astrodet.astrodet as toolkit  # For COCOEvaluatorRecall

# Utils to analyze model
from plot_eval_utils import plot_losses, plot_det_gt, evaluate_model, plot_ap, get_pr

# Matplotlib and other libraries
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from tqdm import tqdm

# Additional imports
import FoFCatalogMatching


# === helper Functions ===
def get_scale_factors(orig_height, orig_width, target_size=512):
    """Calculates scale factors for upsampled images."""
    scale_x = orig_width / target_size
    scale_y = orig_height / target_size
    return scale_x, scale_y

def adjust_coordinates(x, y, scale_x, scale_y):
    """Scales down coordinates from upsampled image size to original WCS size."""
    return x * scale_x, y * scale_y

def img_key_mapper(dataset_dict):
    """Maps image keys for data loader."""
    return dataset_dict["file_name"]

def autofit_text(ax, text, x, y):
    """Fits text within a cell in the FoF plot."""
    fontsize = 11
    bbox_props = dict(boxstyle="square,pad=0.3", fc="white", ec="none", alpha=0)
    while fontsize > 1:
        t = ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
                    color='white', bbox=bbox_props)
        r = ax.figure.canvas.get_renderer()
        bb = t.get_window_extent(renderer=r).transformed(ax.transData.inverted())
        if bb.width < 1 and bb.height < 1:
            return
        t.remove()
        fontsize -= 1
    ax.text(x, y, '.', ha='center', va='center', fontsize=8, color='white')
    
def get_truth_test_cat(test_data, scale=True):
    """
    Gets the RAs & Decs for the ground truth annotations from the test data

    Args:
        test_data (list): A list of dictionaries, where each dictionary contains
            image information and annotations
        scale (bool, optional): A flag indicating whether to apply scaling to the
            ground truth coordinates based on original image dimensions. default is True

    Returns:
        truth_cat (dict): dict w/ 'ra' and 'dec' keys containing RA and Dec coords of gt_objs
    """
    gt_ras = []
    gt_decs = []

    truth_info_cache = {}
    for d in test_data:
        imid = d['image_id']
        subpatch = d['subpatch']
        # temporary sol before I move all the corresponding correct WCS to the ups and pad dirs
        # right now only the lsst_data/truth dir has the updated WCS
        if subpatch not in truth_info_cache:
            truth_info_filename = f'./lsst_data/truth/{subpatch}/{subpatch}_info.json'
            with open(truth_info_filename) as json_data:
                truth_info_cache[subpatch] = json.load(json_data)

        truth_info = truth_info_cache[subpatch]
        entry = next(entry for entry in truth_info if entry['image_id'] == imid)
        if entry is None:
            print(f"Warning: No truth info found for image_id {imid} in subpatch {subpatch}")
            continue
        wcs = WCS(entry['wcs'])
        orig_height, orig_width = entry['height'], entry['width']

        # grab ground truth
        gt_boxes = np.array([a["bbox"] for a in d["annotations"]])
        if gt_boxes.shape[0] != 0:
            gt_boxes = BoxMode.convert(gt_boxes, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
            gt_boxes = Boxes(torch.Tensor(gt_boxes))
            centers_gt = gt_boxes.get_centers().cpu().numpy()

            if scale:
                scale_x, scale_y = get_scale_factors(orig_height, orig_width)
                adj_centers_gt = np.array([adjust_coordinates(x, y, scale_x, scale_y) for x, y in centers_gt])
                gt_coords = wcs.pixel_to_world(adj_centers_gt[:, 0], adj_centers_gt[:, 1])
            else:
                gt_coords = wcs.pixel_to_world(centers_gt[:, 0], centers_gt[:, 1])

            gt_ras.append(gt_coords.ra.degree)
            gt_decs.append(gt_coords.dec.degree)
        else:
            print(f"{d['file_name']} has no objects in its annotations!")

    # turn them into dicts
    truth_cat = {'ra': np.hstack(gt_ras), 'dec': np.hstack(gt_decs)}
    
    return truth_cat

def match_with_truth(det_cat, truth_cat, max_sep=0.5*u.arcsec):
    """
    Match detection catalog with truth catalog with search around sky
    Only keep 1-to-1 matches within max_sep
    """
    if len(det_cat) == 0:
        return pd.DataFrame(), pd.DataFrame()

    det_coords = SkyCoord(ra=det_cat['ra']*u.deg, 
                         dec=det_cat['dec']*u.deg)
    truth_coords = SkyCoord(ra=truth_cat['ra'].to_numpy()*u.deg, 
                           dec=truth_cat['dec'].to_numpy()*u.deg)
    
    idx_det, d2d, _ = det_coords.match_to_catalog_sky(truth_coords)
    good_sep = d2d <= max_sep
    # count how many truth objs match to each det and keep 1-to-1 matches
    unique_truth, truth_counts = np.unique(idx_det[good_sep], return_counts=True)
    good_truth = unique_truth[truth_counts == 1]
    
    final_mask = good_sep & np.isin(idx_det, good_truth)
    matched_det_indices = np.where(final_mask)[0]
    matched_truth_indices = idx_det[final_mask]
    
    matched_det = det_cat.iloc[matched_det_indices].copy()
    matched_truth = truth_cat.iloc[matched_truth_indices].copy()
    
    print(f"Found {len(matched_det)} matched pairs")
    
#     print(final_matches)
    return matched_det, matched_truth

class ModelEvaluator:
    def __init__(self, run_dir, output_dir, test_data_file, cfg_file, model_path):
        """Initialize the ModelEvaluator.
        Args:
            run_dir (str): Directory containing model checkpoints
            output_dir (str): Directory for evaluation outputs
            test_data_file (str): Path to test data JSON file
            cfg_file (str): Path to model config file
        """
        self.run_dir = run_dir
        self.output_dir = output_dir
        self.test_data_file = test_data_file
        self.cfg_file = cfg_file
        self.model_path = model_path
        self.imreader = RomanImageReader()
        
        # default params
        self.topk_per_img = 2000
        self.score_thresh = 0.3
        self.nms_thresh = 0.4
        
        self.cfg = None
        self.predictor = None
        self.test_data = None
        self.registered_test_data = None
        


    def set_thresholds(self, topk=None, score=None, nms=None):
        """Set detection thresholds.
        
        Args (Defaults set in instantiation):
            topk (int): Top-k predictions per image. 
            score (float): Score threshold
            nms (float): NMS threshold
        """
        if topk is not None:
            self.topk_per_img = topk
        if score is not None:
            self.score_thresh = score
        if nms is not None:
            self.nms_thresh = nms

    def load_model(self):
        """Load model configuration and checkpoint."""
        if self.model_path is not None:
            model_path = self.model_path
        else:
            # finding model checkpoint
            pth_files = glob.glob(os.path.join(self.run_dir, '*.pth'))
            model_path = None
            for pth in pth_files:
                if "instances" not in pth:
                    model_path = pth
                    break
#             model_path = './lsst_runs/run5_sm_dlvl5/lsst_dlvl5.pth' # lsst_dlvl5.pth 
#         model_path = './lsst_runs/run5_ups_roman_dlvl5/lsstc_ups_dlvl5.pth'

        if not model_path:
            raise ValueError(f"No valid checkpoint file found in {self.run_dir}")
            
        print(f"Using checkpoint file: {model_path}")
        
        # loading config
        print(f"\nLoading configs from {self.cfg_file}...")
        self.cfg = LazyConfig.load(self.cfg_file)
        for key in self.cfg.get("MISC", dict()).keys():
            self.cfg[key] = self.cfg.MISC[key]
            
        # model params
        self.cfg.train.init_checkpoint = model_path
        for box_predictor in self.cfg.model.roi_heads.box_predictors:
            box_predictor.test_topk_per_image = self.topk_per_img
            box_predictor.test_score_thresh = self.score_thresh
            box_predictor.test_nms_thresh = self.nms_thresh
            
        # initialize predictor
        self.predictor = return_predictor_transformer(self.cfg)
    
    def load_test_data(self):
        """Load and register test dataset."""
        print(f"\nLoading test data from {self.test_data_file}...")
        with open(self.test_data_file, 'r') as f:
            self.test_data = json.load(f)
        
        # Force re-registration
        # Remove the dataset and its metadata first, if they exist.
        # This prevents the AssertionError when registering again.
        dataset_name = 'test'
        print(f"Attempting to remove existing registration for '{dataset_name}'...")
        if dataset_name in DatasetCatalog.list():
            DatasetCatalog.remove(dataset_name)
            print(f"Removed '{dataset_name}' from DatasetCatalog.")
        # It's important to remove from MetadataCatalog too!
        if dataset_name in MetadataCatalog.list():
            MetadataCatalog.remove(dataset_name)
            print(f"Removed '{dataset_name}' from MetadataCatalog.")
        try:
            custom_colors = [
                (0, 255, 0),    # green for galaxies
                (0, 0, 255),    # blue for stars
            ]
            print(f"Registering dataset '{dataset_name}'...")
            self.registered_test_data = register_data_set('test', 
                            self.test_data_file, 
                            thing_classes=["galaxy", "star"]).set(thing_colors=custom_colors)
        except AssertionError:
            print("\nTest dataset already registered")
            

    
    def get_test_data(self):
        if not self.test_data:
            raise ValueError("Load in the test data and register the dataset first! You can use load_test_data().")
        return self.test_data
    
    def evaluate(self):
        """Evaluate model performance on test dataset.
        
        Returns:
            dict: Evaluation results
        """
        if not all([self.cfg, self.predictor]):
            raise ValueError("Model not loaded. Call load_model() first.")
            
        mapper = self.cfg.dataloader.train.mapper(
            self.cfg.dataloader.imagereader, 
            img_key_mapper
        ).map_data
        
        evaluator = toolkit.COCOEvaluatorRecall(
            'test',
            use_fast_impl=True,
            output_dir=self.output_dir,
            allow_cached_coco=False
        )
        
        eval_loader = d2data.build_detection_test_loader(
            self.cfg,
            'test',
            mapper=mapper
        )
        
        results = inference_on_dataset(
            self.predictor.model,
            eval_loader,
            evaluator
        )
        
        return results
    
    
    def get_predictions(self, scale=False, trunc=False):
        """Uses model to predict on the test data 
        
        Args:
            scale (boolean): Default is false since the WCS is for the smaller images. But, we need this to be able to scale up the RAs and DECs accordingly for the upsampled images
        
        Returns:
            predictions (dict): A catalog containing RAs, DECs of every predicted object along with the filenames, imgid and subpatch
        """
        if not all([self.imreader, self.test_data, self.cfg, self.predictor]):
            raise ValueError("Model not loaded. Call load_model() first. \
                             OR ModelEvaluator not instantiated. Create an ModelEvaluator object first \
                             OR Load in test data using load_test_data()")    
        pred_ras = []
        pred_decs = []
        imgids = []
        subpatches = []
        filenames = []
        full_outputs = []

        truth_info_cache = {}
        for d in tqdm(self.test_data, desc="Getting predictions"):
#         for d in self.test_data:
            imid = d['image_id']
            filename = d['file_name']
            # temporary sol before I move all the corresponding correct WCS to the ups and pad dirs
            # right now only the truth dir has the updated WCS
            subpatch = d['subpatch']
            if trunc and d['wcs'] is not None:
                wcs = WCS(d['wcs'])
                if scale:
                    orig_height, orig_width = d['height'], d['width']
                    scale_x, scale_y = get_scale_factors(orig_height, orig_width)
            else:
                if subpatch not in truth_info_cache:
                    truth_info_filename = f'./lsst_data/truth/{subpatch}/{subpatch}_info.json'
                    with open(truth_info_filename) as json_data:
                        truth_info_cache[subpatch] = json.load(json_data)

                truth_info = truth_info_cache[subpatch]
                # grab the WCS
                entry = next(entry for entry in truth_info if entry['image_id'] == imid)
                wcs = WCS(entry['wcs'])
                if scale:
                    orig_height, orig_width = entry['height'], entry['width']
                    scale_x, scale_y = get_scale_factors(orig_height, orig_width)
            # print(wcs)

            # grab model's predictions
            outputs = get_predictions(d, self.imreader, img_key_mapper, self.predictor)
            raw_instances = outputs['instances']
            centers_pred = raw_instances.pred_boxes.get_centers().cpu().numpy()
            if scale:
                adj_centers_pred = np.array([adjust_coordinates(x, y, scale_x, scale_y) for x, y in centers_pred])
                pred_coords = wcs.pixel_to_world(adj_centers_pred[:,0], adj_centers_pred[:,1])
            else:
                pred_coords = wcs.pixel_to_world(centers_pred[:,0],centers_pred[:,1])
            # extend all lists with prediction info
            n_preds = len(centers_pred)
            pred_ras.extend(pred_coords.ra.degree)
            pred_decs.extend(pred_coords.dec.degree)
            imgids.extend([imid] * n_preds)
            subpatches.extend([subpatch] * n_preds)
            filenames.extend([filename] * n_preds)
            
            serializable_instance_data = {}
            if raw_instances.has('pred_boxes'):
                serializable_instance_data['pred_boxes'] = raw_instances.pred_boxes.tensor.cpu().numpy().tolist()
            if raw_instances.has('scores'):
                serializable_instance_data['scores'] = raw_instances.scores.cpu().numpy().tolist()
            if raw_instances.has('pred_classes'):
                serializable_instance_data['pred_classes'] = raw_instances.pred_classes.cpu().numpy().tolist()
#             if raw_instances.has('pred_masks'):
#                 # size implications for masks
#                 serializable_instance_data['pred_masks'] = raw_instances.pred_masks.cpu().numpy().tolist()
            serializable_instance_data['image_id'] = imid 
            serializable_instance_data['file_name'] = filename
            full_outputs.append(serializable_instance_data)

        # turn them into dicts
        dd_det_cat = {
            'ra': np.array(pred_ras),
            'dec': np.array(pred_decs),
            'image_id': np.array(imgids),
            'subpatch': np.array(subpatches),
            'file_name': np.array(filenames)
        }
        
        return dd_det_cat, full_outputs
    
    def get_random_prediction(self, random_img):
        outputs = get_predictions(random_img,  self.imreader, img_key_mapper, self.predictor)
        return outputs
    
    def get_lsst_det_mask(self, lsst_det_coords, scale=False):
        all_bounds = []
        truth_info_cache = {}
        for d in tqdm(self.test_data, desc="Getting LSST Detections within each cutout"):
            imid = d['image_id']
            filename = d['file_name']
            subpatch = d['subpatch']
            scale_x = None
            scale_y = None
            wcs = None
            height = d['height']
            width = d['width']
            if "wcs" in d and d['wcs'] is not None:
                wcs = WCS(d['wcs'])
                if scale:
                    orig_height, orig_width = d['height'], d['width']
                    scale_x, scale_y = get_scale_factors(orig_height, orig_width)
            else:
                if subpatch not in truth_info_cache:
                    truth_info_filename = f'./lsst_data/truth/{subpatch}/{subpatch}_info.json'
                    with open(truth_info_filename) as json_data:
                        truth_info_cache[subpatch] = json.load(json_data)

                truth_info = truth_info_cache[subpatch]
                # grab the WCS
                entry = next(entry for entry in truth_info if entry['image_id'] == imid)
                wcs = WCS(entry['wcs'])
                if scale:
                    orig_height, orig_width = entry['height'], entry['width']
                    scale_x, scale_y = get_scale_factors(orig_height, orig_width)
            
            # corner coords
            # top left (0,0) top right (511, 0) bottom left (0, 511) bottom right (511, 511)
            corners_x = [0, width - 1, 0, width - 1]  # bc upsampled img is 512x512
            corners_y = [0, 0, height - 1, height - 1]
            if scale:
                adj_corners_x, adj_corners_y = zip(*[adjust_coordinates(x, y, scale_x, scale_y) for x, y in zip(corners_x, corners_y)])
                corners = wcs.pixel_to_world(adj_corners_x, adj_corners_y)
            else:
                corners = wcs.pixel_to_world(corners_x, corners_y)

            all_bounds.append(corners)      

        mask = np.zeros(len(lsst_det_coords), dtype=bool)
        for bounds in all_bounds:
            ra_min, ra_max = np.min(bounds.ra), np.max(bounds.ra)
            dec_min, dec_max = np.min(bounds.dec), np.max(bounds.dec)

            mask |= ((lsst_det_coords.ra >= ra_min) & (lsst_det_coords.ra <= ra_max) &
                     (lsst_det_coords.dec >= dec_min) & (lsst_det_coords.dec <= dec_max))

        return mask
        
    def run_full_evaluation(self):
        """Run full evaluation pipeline.
        
        Returns:
            dict: Evaluation results
        """
        self.load_model()
        self.load_test_data()
        return self.evaluate()

    




def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run evaluation on test data.')
    parser.add_argument('--folder', type=str, default='annotationsc-ups',
                        help='Data folder name (default: annotationsc-ups)')
    
    parser.add_argument('--scale', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='Whether to scale coordinates for upsampled images (default: True)')
    
    parser.add_argument('--run_dir', type=str, default='./lsst_runs/run5_ups_roman_dlvl5',
                        help='Directory containing model checkpoints (default: ./lsst_runs/run5_ups_roman_dlvl5)')
    
    parser.add_argument('--config_file', type=str, default='./deepdisc/configs/solo/swinc_lsst_ups.py',
                        help='Model config file path (default: ./deepdisc/configs/solo/swinc_lsst_ups.py)')
    
    parser.add_argument('--model_path', type=str, default=None,
                        help='Exact path to the model checkpoint (.pth). Overrides automatic search if provided.')
    
    parser.add_argument('--test_score_thresh', type=float, default=0.3,
                    help='Test Score Threshold for making predictions with DeepDISC (default: 0.3)')
    parser.add_argument('--nms', type=float, default=0.5,
                    help='Non-Maximum Suppression (default: 0.5)')
    
    return parser.parse_args()

# -------------- MAIN -----------------
if __name__ == "__main__":
    main_start_time = time.perf_counter()
    args = parse_args()

    folder = args.folder
    # scale based on random shape size in test data or put in metadata
    scale = args.scale
    run_dir = args.run_dir
    cfg_file = args.config_file
    model_path = args.model_path
    test_score_thresh = args.test_score_thresh
    nms = args.nms
    
    
#     folder = 'annotationsc-ups'
#     scale = True # upsampled - True, lsst - False
    print(f"Running with parameters:")
    print(f"  folder: {folder}")
    print(f"  scale: {scale}")
    print(f"  run_dir: {run_dir}")
    print(f"  config_file: {cfg_file}")
    
    test_data_fi = f'lsst_data/{folder}/test.json'
    with open(test_data_fi, 'r') as f:
        test_data = json.load(f)

    # test_truth_cat = get_truth_test_cat(test_data, scale=False)
    test_truth_cat = get_truth_test_cat(test_data, scale=scale)
    test_truth_cat_df = pd.DataFrame(test_truth_cat)

    # saving RAs and DECs of test data
    file_path = f'./lsst_runs/{run_dir}/test_truth_radec.json'
    if not os.path.exists(file_path):
        test_truth_cat_df.to_json(file_path, orient='records')
        print(f"File created: {file_path}")
    else:
        print(f"File already exists: {file_path}")


# ---------------- Subset of LSST Truth/Det Catalog ---------------------
# lsst_truth = pd.read_json('/home/shared/hsc/roman_lsst/lsst_catalogs/overlap_lsst_truth_cat.json')
    lsst_truth = pd.read_json('/home/shared/hsc/roman_lsst/lsst_catalogs/lsst_truth_cat_all_otp_morph.json')
    print(f"Successfully loaded in LSST Truth Catalog overlapping with all of Roman Truth Catalog: {len(lsst_truth)}")

    lsst_det = pd.read_json('/home/shared/hsc/roman_lsst/lsst_catalogs/lsst_det_cat_all_overlap_patches.json')
    print(f"Successfully loaded in LSST Det Catalog overlapping with all of Roman Truth Catalog: {len(lsst_det)}")
    
    # Match Test Truth with LSST Truth
    matched_test_truth_filename = f'./lsst_runs/{run_dir}/test_truth11_cat.json'
    if not os.path.exists(matched_test_truth_filename):
        t0 = time.perf_counter()
        matched_test, matched_lsst_truth = match_with_truth(test_truth_cat_df, lsst_truth)
        t1 = time.perf_counter()
        print(f"Time to match test truth with LSST truth (getting subset of LSST Truth catalog): {t1 - t0:.2f} seconds")
        
        matched_lsst_truth.to_json(matched_test_truth_filename, orient='records')
        print(f"File created: {matched_test_truth_filename}")
    else:
        print(f"File already exists: {matched_test_truth_filename}")
        matched_lsst_truth = pd.read_json(matched_test_truth_filename)
    
    # Match the above subset of LSST Truth catalog with the overall LSST Detection catalog
    # to get the LSST Detection catalog that essentially covers the same region as the test set
    matched_test_det_filename = f'./lsst_runs/{run_dir}/test_lsst_det11_info.json'
    matched_test_det_truth_filename = f'./lsst_runs/{run_dir}/test_lsst_det_truth11_info.json'
    
    if not os.path.exists(matched_test_det_filename):
        t0 = time.perf_counter()
        matched_lsst_det, matched_lsst_det_truth = match_with_truth(lsst_det, matched_lsst_truth)
        t1 = time.perf_counter()

        print(f"Time to get the subset of LSST det catalog from test set: {t1 - t0:.2f} seconds")
        matched_lsst_det.to_json(matched_test_det_filename, orient='records')
        matched_lsst_det_truth.to_json(matched_test_det_truth_filename, orient='records')
        print(f"Files created: {matched_test_det_filename} and {matched_test_det_truth_filename}")
    else:
        matched_lsst_det = pd.read_json(matched_test_det_filename)
        matched_lsst_det_truth = pd.read_json(matched_test_det_truth_filename)
        print(f"Files already exist: {matched_test_det_filename} and {matched_test_det_truth_filename}")



# ------------ Model Predictions ------------
#     run_dir = './lsst_runs/run5_ups_roman_dlvl5'
    output_dir = run_dir
    test_data_file = f'./lsst_data/{folder}/test.json'
#     cfg_file = "./deepdisc/configs/solo/swin_lsst.py"
#     cfg_file = "./deepdisc/configs/solo/swinc_lsst_ups.py"
    # run_name = 

    evaluator = ModelEvaluator(
        run_dir=run_dir,
        output_dir=output_dir,
        test_data_file=test_data_file,
        cfg_file=cfg_file,
        model_path=model_path
    )
    # 0.45 0.5 was used for that AstroFest plot comparison of unrecognized blends
    evaluator.set_thresholds(topk=2000, score=test_score_thresh, nms=nms)
    evaluator.load_model()
    evaluator.load_test_data()
    # test_data = evaluator.get_test_data()
    
    full_pred_output_file = f'./lsst_runs/{run_dir}/full_pred_output_s{test_score_thresh}_n{nms}.json'
    predictions_file = f'./lsst_runs/{run_dir}/pred_s{test_score_thresh}_n{nms}.json'
    dd_det_file = f'./lsst_runs/{run_dir}/pred_s{test_score_thresh}_n{nms}_radec.json'

    if not os.path.exists(full_pred_output_file):
        # assuming if above file doesn't exist none of the other two exist
        print("Generating Model Predictions:")
        predictions, all_outputs = evaluator.get_predictions(scale=scale)
        predictions_df = pd.DataFrame(predictions)
        predictions_df.to_json(predictions_file, orient='records', indent=4)
        
        # list_of_serializable_outputs
        with open(full_pred_output_file, 'w') as f:
            json.dump(all_outputs, f, indent=4)
        print(f"Saved full prediction outputs (from Instances) to {full_pred_output_file}")

        print(f"Saved DeepDISC Predictions to {predictions_file} and {full_pred_output_file}")
        dd_det_df = pd.DataFrame({
            'ra': predictions['ra'],
            'dec': predictions['dec']
        })
        dd_det_df.to_json(dd_det_file, orient='records')
        print(f"Saved DeepDISC Predictions RA and DEC to {dd_det_file}")
    else:
        predictions = pd.read_json(predictions_file, orient='records')
        dd_det_df = pd.read_json(dd_det_file, orient='records')
    
    matched_dd_truth_filename = f'./lsst_runs/{run_dir}/dd_truth11_info_s{test_score_thresh}_n{nms}.json'
    matched_dd_det_filename = f'./lsst_runs/{run_dir}/dd_det11_s{test_score_thresh}_n{nms}_radec.json'
    
    if not os.path.exists(matched_dd_truth_filename):
        matched_dd_det, matched_dd_truth = match_with_truth(dd_det_df, matched_lsst_truth)
        matched_dd_det.to_json(matched_dd_det_filename, orient='records')
        matched_dd_truth.to_json(matched_dd_truth_filename, orient='records')
        print(f"File created: {matched_dd_truth_filename} and {matched_dd_det_filename}")
    else:
        print(f"File already exists: {matched_dd_truth_filename} and {matched_dd_det_filename}")

# ----Getting Roman and LSST Detection Catalog Manually rather than SkyCoord matching Predictions -------

    # LSST First (from original metrics.ipynb)
    lsst_all_test_det_filename = f'./lsst_runs/{run_dir}/test_lsst_det_fof_info.json'
    if not os.path.exists(lsst_all_test_det_filename):
        lsst_det_coords = SkyCoord(ra=lsst_det['ra']*u.deg, dec=lsst_det['dec']*u.deg)
        mask = evaluator.get_lsst_det_mask(lsst_det_coords, scale=scale)
        filtered_lsst_det = lsst_det[mask]
        lsst_all_test_det = filtered_lsst_det.drop_duplicates(subset=['ra', 'dec'])
        lsst_all_test_det.to_json(lsst_all_test_det_filename, orient='records')
    else:
        print(f"Files already exist: {lsst_all_test_det_filename}!")
    
    main_final_time = time.perf_counter()
    print(f"Time for whole script to run: {main_final_time - main_start_time:.2f} seconds")