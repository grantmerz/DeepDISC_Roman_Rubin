""" This is a demo "solo config" file for use in solo_test_run_transformers.py.

This uses template configs cascade_mask_rcnn_swin_b_in21k_50ep and yaml_style_defaults."""

from omegaconf import OmegaConf
import numpy as np
# ---------------------------------------------------------------------------- #
# Local variables and metadata
# ---------------------------------------------------------------------------- #
epoch=448 #(TRAINING NUM OF IMAGES / BATCH SIZE 280/4 diff img set - 2351/4 = 588, same size img set - 2264/4=566, 1792/4 )
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
dataloader.augs = dc2_train_augs
dataloader.train.total_batch_size = bs

model.proposal_generator.anchor_generator.sizes = [[8], [16], [32], [64], [128]]
model.roi_heads.num_classes = numclasses
model.roi_heads.batch_size_per_image = 512

model.backbone.bottom_up.in_chans = 6

# LSST 3828 Tract (1,1-2,5 Patches) - 6 Filters
model.pixel_mean = [
    0.05707236, 
    0.0508133,
    0.07755638, 
    0.10909662, 
    0.14604638,
    0.21117774
] 

model.pixel_std = [
    1.57212144, 
    1.06318582, 
    1.53453935, 
    2.19896594, 
    2.76794921, 
    4.03117939
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
    
# train.init_checkpoint = "./detectron2/projects/ViTDet/model_final_246a82.pkl"

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
efinal = epoch * 50 # 22400 took 17330s ~ 4.8138889 hrs

SOLVER.STEPS = [e1,e2,e3]  # do not decay learning rate for retraining
SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
SOLVER.WARMUP_ITERS = 0
SOLVER.MAX_ITER = efinal  # for DefaultTrainer