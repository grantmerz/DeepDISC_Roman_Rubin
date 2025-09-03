from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib import colors
from astropy.visualization import make_lupton_rgb
import numpy as np
import pandas as pd
from astropy.nddata import Cutout2D
from astropy.wcs import WCS

import cv2
from detectron2.structures import BoxMode
from astropy.table import Table
import glob
from astropy.coordinates import SkyCoord  # High-level coordinates
from detectron2.config import LazyConfig, get_cfg, instantiate
import os
import scipy.stats as stats
import h5py
import json
import astropy.units as u
from astropy.coordinates import SkyCoord

import warnings
import time

from astropy.wcs import FITSFixedWarning
warnings.filterwarnings("ignore", category=FITSFixedWarning)
import torch
import torch.nn.functional as F
from detectron2.data import detection_utils as utils
import pickle
import detectron2.data as d2data

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cfgfile', type=str,
                    help='path to config file')
parser.add_argument('--run-name', type=str,
                    help='run name')
parser.add_argument('--dropfilt', type=int, default=-1,
                    help='filter dropped during inference')
parser.add_argument('--output-dir', type=str,
                    help='directory that has saved model')

args = parser.parse_args()

trainfile = "/home/shared/hsc/roman_lsst/lsst_data/annotations_lvl2/train.json"
testfile = "/home/shared/hsc/roman_lsst/lsst_data/annotations_lvl2/test.json"

with open('/home/shared/hsc/roman_lsst/lsst_data/annotations_lvl2/test.json', 'r') as f:
    test_ddicts = json.load(f)



def rubin_key_mapper(dataset_dict):
    '''
    args
        dataset_dict: [dict]
            A dictionary of metadata
    
    returns
        fn: str
            The filepath to the corresponding image
    
    '''
    fn = dataset_dict['file_name']
    return os.path.join('/home/shared/hsc/roman_lsst/',fn)



            
dtruth = pd.read_csv('/home/g4merz/DeepDISC_Roman_Rubin/data/dtest_ups_truth.csv')

cfgfile = args.cfgfile
run_name = args.run_name
dropfilt = args.dropfilt
output_dir = args.output_dir
    
#import deepdisc.model.models as roiheads
#import deepdisc.model.loaders as loaders

from deepdisc.inference.predictors import AstroPredictor

cfg = LazyConfig.load(cfgfile)

cfg.train.init_checkpoint = os.path.join(output_dir, run_name + ".pth")
cfg.TEST.DETECTIONS_PER_IMAGE = 3000
cfg.model.proposal_generator.anchor_generator.sizes = [[8], [16], [32], [64], [128]]
cfg.model.roi_heads.batch_size_per_image=1024
cfg.model.proposal_generator.post_nms_topk=[6000,3000]
cfg.model.proposal_generator.batch_size_per_image = 1024
for box_predictor in cfg.model.roi_heads.box_predictors:
    box_predictor.test_topk_per_image = 3000
    box_predictor.test_score_thresh = 0.5
    box_predictor.test_nms_thresh = 0.3
   
gf=False
cfg.model.roi_heads.output_features = gf
    
def matched_catalog(dall, outputs, wcs):
    allcatalog = SkyCoord(ra=dall['RA'].values*u.degree, dec=dall['DEC'].values*u.degree)
        
    centers = outputs['instances'].pred_boxes.get_centers().cpu().numpy()
    c = wcs.pixel_to_world(centers[:,0],centers[:,1])

    #avoid duplicate matches by having allcatalog first
    idx, d2d, d3d = allcatalog.match_to_catalog_sky(c)
    minds = np.where(d2d.to(u.arcsec).value<=0.5)[0]
    
    return dall.iloc[minds], idx[minds]

    
from deepdisc.inference.match_objects import get_matched_object_inds

def iou_matched_catalog(d,outputs,dcat=dtruth):

    gi_sz=[]
    di_sz = []

    gi, di = get_matched_object_inds(d,outputs,IOUthresh=0.3)

    for i,gii in enumerate(gi):
        if d['annotations'][gii]['obj_id'] in dcat.object_id.values:
            gi_sz.append(d['annotations'][gii]['obj_id'])
            di_sz.append(di[i])

    dm = dcat.iloc[np.array([np.where(dcat.object_id.values==gi) for gi in gi_sz ]).flatten()]
    m = di_sz
    
    return dm,m



def get_outputs_withwcs(predictor, d, image):
    #d = metadata[i]
    wcs = d['wcs']
    with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
        # Apply pre-processing to image.
        # image = self.aug.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32"))
        inputs = {'image':image, 'wcs':wcs, 'height':image.shape[1], 'width':image.shape[2]}
        out = predictor.model([inputs])

    return out



def get_matched_dects_by_area(d, outputs, get_features=False):
        
    xs = np.linspace(0, 14, 1401)
    ys = np.zeros_like(xs)
    wcsi = WCS(d['wcs'])
 
    if len(outputs['instances'])==0:
        return None
    
    dm,m = iou_matched_catalog(d,outputs)
        
    if len(dm)==0:
        return None

        
    #pdfs = np.exp(outputs['instances'].pred_redshift_pdf.cpu().numpy())[m]
    
    gmm = outputs['instances'].pred_gmm.cpu()
    ws = gmm[..., :5]
    ws = F.softmax(ws, dim=-1).numpy()
    mus = gmm[..., 5:10]
    stds = torch.exp(gmm[..., 10:])

    scores = outputs['instances'].scores.cpu().numpy()[m]

    zpreds = mus[m,np.argmax(ws[m],axis=1)]
    ztrues = dm.z.values
    #ztrues = np.ones(len(dm))
    ids = dm.object_id.values
    
    
    gmms = np.transpose(np.array([ws[m],mus[m].numpy(),stds[m].numpy()]),axes=(1,0,2))

    if len(scores) != len(dm):
        print('mismatch')
 

    if get_features is False:
        return zpreds, ztrues, ids, scores, gmms
    else:
        features = outputs['instances'].features.cpu().numpy()[m]
        return zpreds, ztrues, ids, scores, gmms, features




def map_inds(i):
    d=test_ddicts[i]
    wcs = d['wcs']
    filename = rubin_key_mapper(d)
    image = np.load(filename)    
    with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
        # Apply pre-processing to image.
        # image = self.aug.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32"))
        inputs = {'image':image, 'wcs':wcs, 'height':image.shape[1], 'width':image.shape[2], 'annotations':d['annotations']}
        #out = predictor.model([inputs])
    return inputs


def get_res(dloader,predictor,get_features=False):
    zps = []
    zts = []
    ids = []
    scores = []
    gmms = []
    if get_features:
        features = []
    with torch.no_grad():
        for i, dataset_dicts in enumerate(dloader):
            print(i)
            batched_outputs = predictor.model(dataset_dicts)
            for i,(d,outputs) in enumerate(zip(dataset_dicts,batched_outputs)):
                outs = get_matched_dects_by_area(d,outputs,get_features)
                if outs is not None: 
                    if get_features:
                        zp, zt, idi, si, gmmi, feati = outs
                        features.appends(feati)
                    else:
                        zp, zt, idi, si, gmmi = outs
                    zps.append(zp)
                    zts.append(zt)
                    ids.append(idi)
                    scores.append(si)
                    gmms.append(gmmi)
    zps = np.hstack(zps)
    zts = np.hstack(zts)
    ids = np.hstack(ids)
    scores = np.hstack(scores)
    gmms = np.vstack(gmms)
    if get_features:
        features = np.hstack(features)
    
    clean_inds = []
    for idi in np.unique(ids):
        inds = np.where(ids==idi)[0]
        if len(inds)==1:
            clean_inds.append(inds[0])
        else:
            si = np.argmax(scores[inds])
            clean_inds.append(inds[si])
    clean_inds = np.array(clean_inds)
    zts = zts[clean_inds]
    zps = zps[clean_inds]
    ids = ids[clean_inds]
    gmms = gmms[clean_inds]
    if get_features:
        features = np.hstack(features)
        return zps,zts,ids,scores, gmms, features
    else:
         return zps,zts,ids,scores, gmms


predictor = AstroPredictor(cfg)


loader = d2data.build_detection_test_loader(
    np.arange(len(test_ddicts)), mapper=map_inds, batch_size=1
)


outs = get_res(loader,predictor, gf)


if gf:
    zps,zts,ids,scores,gmms,features = outs
    zps_trunc,zts_trunc,ids_trunc,scores_trunc,gmms_trunc,features_trunc = outs_trunc

    test_dict={'z_pred':zps, 'z_spec':zts, 'ids':ids, 'scores':scores, 'gmms':gmms, 'features':features}
    test_dict_trunc={'z_pred':zps_trunc, 'z_spec':zts_trunc, 'ids':ids_trunc, 'scores':scores_trunc, 'gmms':gmms_trunc, 'features':features_trunc}

else:
    zps,zts,ids,scores,gmms = outs

    test_dict={'z_pred':zps, 'z_spec':zts, 'ids':ids, 'scores':scores, 'gmms':gmms}


with open(f'/home/g4merz/DeepDISC_Roman_Rubin/estimation/{run_name}_test_outs_phot.npy', 'wb') as fp:
    pickle.dump(test_dict, fp)



