""" This is a demo "solo config" file for use in solo_test_run_transformers.py.

This uses template configs cascade_mask_rcnn_swin_b_in21k_50ep and yaml_style_defaults."""

from omegaconf import OmegaConf
import numpy as np
# ---------------------------------------------------------------------------- #
# Local variables and metadata
# ---------------------------------------------------------------------------- #
epoch=566 #(TRAINING NUM OF IMAGES / BATCH SIZE 280/4 2351/4) 588 for roman 566 for combined
bs=4 # 1 or 2 times the number of gpus
metadata = OmegaConf.create() 
metadata.classes = ["galaxy", "star"]

numclasses = len(metadata.classes)

# ---------------------------------------------------------------------------- #
# Standard config (this has always been the LazyConfig/.py-style config)
# ---------------------------------------------------------------------------- #
# Get values from templates
from ..COCO.cascade_mask_rcnn_swin_b_in21k_50ep import dataloader, model, train, lr_multiplier, optimizer
import deepdisc.model.loaders as loaders
from deepdisc.data_format.augment_image import dc2_train_augs, dc2_train_augs_full, roman_train_augs
from deepdisc.data_format.image_readers import RomanImageReader

# Overrides

# configuring focal loss
from detectron2.config import LazyCall as L
from detectron2.modeling.roi_heads import (
    FastRCNNOutputLayers
)
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.layers import ShapeSpec

dataloader.augs = dc2_train_augs
dataloader.train.total_batch_size = bs

model.proposal_generator.anchor_generator.sizes = [[8], [16], [32], [64], [128]]
model.roi_heads.num_classes = numclasses
model.roi_heads.batch_size_per_image = 512

# # implementing focal loss
# model.roi_heads.use_focal_ce = True
# model.roi_heads.focal_alpha = 0.25  # apply same weight to the minority class (stars) & 1 - alpha to the majority class (galaxies)
# model.roi_heads.focal_gamma = 2.0

# just for roman
# model.backbone.bottom_up.in_chans = 4
# for combined lsst and roman data
model.backbone.bottom_up.in_chans = 10

# one patch (50.93 -42.0)
# model.pixel_mean = [
#         0.72566669,
#         1.08332919,
#         0.89876418,
#         1.081059
# ]
# model.pixel_std = [
#         21.29308954,
#         28.89818123,
#         27.25925706,
#         28.90273031
# ]

# all patches
# model.pixel_mean = [
#         0.96572415,
#         1.3813639,
#         1.1515364,
#         1.30524278
# ]
# model.pixel_std = [
#         24.60855278,
#         33.10368844,
#         31.73495475,
#         33.16013963
# ]

# combined lsst and roman data
model.pixel_mean = [
        0.99935028, 
        1.42305351, 
        1.18362291, 
        1.33769297, 
        0.07417516, 0.06009778, 0.08101993, 
        0.11556633, 0.15088047, 0.25574822
]
model.pixel_std = [
        24.87738243, 
        33.31123686, 
        32.00189893, 
        33.40030245,  
        1.22474413, 0.62586614, 0.9037747,  
        1.27968965, 1.66830258, 2.53991423
]
#model.roi_heads.num_components = 3
#model.roi_heads.zloss_factor = 1
#model.roi_heads.zbins = np.linspace(0,5,200)
#model.roi_heads.weights = np.load('/home/g4merz/rail_deepdisc/configs/solo/zweights.npy')
#model.roi_heads._target_ = RedshiftPDFCasROIHeads
model.proposal_generator.nms_thresh = 0.3

for box_predictor in model.roi_heads.box_predictors:
    box_predictor.test_topk_per_image = 2000
    box_predictor.test_score_thresh = 0.5
    box_predictor.test_nms_thresh = 0.3
#     box_predictor.use_focal_ce = True
#     box_predictor.focal_alpha = 0.25  # apply same weight to the minority class (stars) & 1 - alpha to the majority class (galaxies)
#     box_predictor.focal_gamma = 2.0
    
# train.init_checkpoint = "./detectron2/projects/ViTDet/model_final_246a82.pkl"
# from https://github.com/kakaobrain/torchlars, using Layer-wise Adaptive Rate Scaling

optimizer.lr = 0.001
dataloader.test.mapper = loaders.DictMapper
dataloader.train.mapper = loaders.DictMapper
reader = RomanImageReader()
dataloader.imagereader = reader
dataloader.epoch=epoch
# ---------------------------------------------------------------------------- #
# Yaml-style config (was formerly saved as a .yaml file, loaded to cfg_loader)
# ---------------------------------------------------------------------------- #
# Get values from template
from .yacs_style_defaults import MISC, DATALOADER, DATASETS, GLOBAL, INPUT, MODEL, SOLVER, TEST

# Overrides
SOLVER.IMS_PER_BATCH = bs

DATASETS.TRAIN = "astro_train"
DATASETS.TEST = "astro_val"

SOLVER.BASE_LR = 0.001
SOLVER.CLIP_GRADIENTS.ENABLED = True
# Type of gradient clipping, currently 2 values are supported:
# - "value": the absolute values of elements of each gradients are clipped
# - "norm": the norm of the gradient for each parameter is clipped thus
#   affecting all elements in the parameter
SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
# Maximum absolute value used for clipping gradients
# Floating point number p for L-p norm to be used with the "norm"
# gradient clipping type; for L-inf, please specify .inf
SOLVER.CLIP_GRADIENTS.NORM_TYPE = 5.0


e1 = epoch * 15
e2 = epoch * 25
e3 = epoch * 30
efinal = epoch * 50

SOLVER.STEPS = [e1,e2,e3]  # do not decay learning rate for retraining
SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
SOLVER.WARMUP_ITERS = 0
SOLVER.MAX_ITER = efinal  # for DefaultTrainer
