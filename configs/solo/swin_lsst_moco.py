""" This is a demo "solo config" file for use in solo_test_run_transformers.py.

This uses template configs cascade_mask_rcnn_swin_b_in21k_50ep and yaml_style_defaults."""

from omegaconf import OmegaConf
import numpy as np
import math
# ---------------------------------------------------------------------------- #
# Local variables and metadata
# ---------------------------------------------------------------------------- #
# for lsst_data of 3822 tract
# epoch=448 #(TRAINING NUM OF IMAGES / BATCH SIZE 280/4 diff img set - 2351/4 = 588, same size img set - 2264/4=566, 1792/4 2298/4 )
epoch= 582 # 2331/4 # total batch size spread across multiple GPUs
bs=4 # 1 or 2 times the number of gpus batch size and make sure it's divisible by the number of GPUs
# change until memory error

metadata = OmegaConf.create() 
metadata.classes = ["galaxy", "star"]
numclasses = len(metadata.classes)

# ---------------------------------------------------------------------------- #
# Standard config (this has always been the LazyConfig/.py-style config)
# ---------------------------------------------------------------------------- #
# Get values from templates
from ..COCO.cascade_mask_rcnn_swin_b_in21k_50ep import dataloader, model, train, lr_multiplier, optimizer
from detectron2.modeling import SwinTransformer

from ..custom.image_readers import RomanRubinImageReader, MoCoRomanRubinImageReader
from ..custom.mappers import MocoDictMapper
import deepdisc.model.loaders as loaders
from deepdisc.data_format.augment_image import dc2_train_augs
from deepdisc.data_format.image_readers import NumpyImageReader
from detectron2.config import LazyCall as L
from ..custom.meta_arch import GeneralizedRCNNMoco

# # Overrides
#train.init_checkpoint = "/projects/bdsp/yse2/model_final_246a82.pkl"
train.init_checkpoint='/projects/bdsp/yse2/cascade_mask_rcnn_swin_b_in21k_model.pkl'
dataloader.augs = dc2_train_augs
dataloader.train.total_batch_size = bs

model.proposal_generator.anchor_generator.sizes = [[8], [16], [32], [64], [128]]
model.roi_heads.num_classes = numclasses
model.roi_heads.batch_size_per_image = 512


model._target_ = GeneralizedRCNNMoco
model.backbone_q = model.backbone
model.backbone_k = model.backbone
model.pop('backbone')
model.pop('pixel_mean')
model.pop('pixel_std')

model.backbone_q.square_pad = 160
model.backbone_k.square_pad = 512
model.backbone_q.bottom_up.in_chans = 6
model.backbone_k.bottom_up.in_chans = 3
model.backbone_q.bottom_up.patch_size=4
model.backbone_k.bottom_up.patch_size = math.ceil(model.backbone_k.square_pad/model.backbone_q.square_pad*model.backbone_q.bottom_up.patch_size)

# LSST Data in 6 filters for 700 tiles
model.lsst_pixel_mean = [
    0.0570717453956604,
    0.05500221252441406,
    0.07863432914018631,
    0.11082269251346588,
    0.13925790786743164,
    0.21512146294116974,
]

model.lsst_pixel_std = [
    0.9746726155281067,
    0.6917527318000793,
    0.9822555184364319,
    1.382053017616272,
    1.8204920291900635,
    2.6324615478515625,
]


model.roman_pixel_mean = [
    0.0570717453956604,
    0.05500221252441406,
    0.07863432914018631,
]

model.roman_pixel_std = [
    0.9746726155281067,
    0.6917527318000793,
    0.9822555184364319,
]

# LSST Data in 6 filters for 16 tiles
# 7/11/25
# model.pixel_mean = [
#     0.05766211653019801,
#     0.05522824341264653,
#     0.08055171695300539,
#     0.11272945612787254,
#     0.14285408247959108,
#     0.22691514861918002,
# ]
# model.pixel_std = [
#     1.0329147035160937,
#     0.6753845510365868,
#     0.9406882743716739,
#     1.3125461429047487,
#     1.7310270468969295,
#     2.7391247463323487,
# ]

model.proposal_generator.nms_thresh = 0.3
for box_predictor in model.roi_heads.box_predictors:
    box_predictor.test_topk_per_image = 2000
    box_predictor.test_score_thresh = 0.5
    box_predictor.test_nms_thresh = 0.3
    
# change this function depending on the metadata format
# needs to return where the cutout image data for each cutout is stored
def roman_key_mapper(dataset_dict):
    fn = dataset_dict["file_name"]
    return fn

dataloader.key_mapper = roman_key_mapper

#dataloader.test.mapper = loaders.DictMapper
#dataloader.train.mapper = loaders.DictMapper

dataloader.test.mapper = MocoDictMapper
dataloader.train.mapper = MocoDictMapper

reader = MoCoRomanRubinImageReader()
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

# # 22400 took 17330s ~ 4.8138889 hrs

SOLVER.STEPS = []  # do not decay learning rate for retraining
SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
SOLVER.WARMUP_ITERS = 0
