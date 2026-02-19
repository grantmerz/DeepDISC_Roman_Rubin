""" This is a demo "solo config" file for use in solo_test_run_transformers.py.

This uses template configs cascade_mask_rcnn_swin_b_in21k_50ep and yaml_style_defaults."""

from omegaconf import OmegaConf
# ---------------------------------------------------------------------------- #
# Local variables and metadata
# ---------------------------------------------------------------------------- #
# bs: total batch size spread across multiple GPUs & make sure it's divisible by the number of GPUs
bs = 192 # for 4 H200 GPUS with 140G memory each as each GPU can handle bs=48
# steps_per_epoch = num of steps per epoch = num of training images / bs
# a training step/iteration is when the model weights are updated (once per batch)
num_imgs = 109782
steps_per_epoch = num_imgs // bs

metadata = OmegaConf.create() 
metadata.classes = ["galaxy", "star"]
numclasses = len(metadata.classes)

# ---------------------------------------------------------------------------- #
# Standard config (this has always been the LazyConfig/.py-style config)
# ---------------------------------------------------------------------------- #
# Get values from templates
from ..COCO.cascade_mask_rcnn_swin_b_in21k_50ep import dataloader, model, train, lr_multiplier, optimizer
from ..custom.image_readers import RomanRubinImageReader

import deepdisc.model.loaders as loaders
from deepdisc.data_format.augment_image import dc2_train_augs

# Overrides of the template COCO config
# This is the cascade mask rcnn of an ImageNet-swin_base_patch4_window7_224_21k model
train.init_checkpoint = "/projects/bdsp/yse2/cascade_mask_rcnn_swin_b_in21k_model.pkl"
# for TimedLazyAstroTrainer (all in iters)
train.timing_report_period = steps_per_epoch // 2 # report every n iters (for testing, 5)
train.timing_rolling_window_size = steps_per_epoch // 2 # average over last n iters (for testing, 5)
train.timing_save_period = steps_per_epoch # save timing to disk every n iters (for testing, 10)

dataloader.augs = dc2_train_augs
dataloader.train.total_batch_size = bs
# dataloader.train.num_workers = 16
# when bs=96, bs * 6 but when bs=144, bs * 6 crashed so using bs * 4 and for bs=192, bs * 2
dataloader.test.total_batch_size = bs * 2 # higher since no gradients being calculated
dataloader.test.num_workers = 16
model.proposal_generator.anchor_generator.sizes = [[8], [16], [32], [64], [128]]
model.roi_heads.num_classes = numclasses
model.roi_heads.batch_size_per_image = 512

"""
Since the in_features are ("p0", "p1", "p2", "p3"), 
Swin's initial default patch embedding has stride 4,
and then 3 patch merging layers (which each double the stride) making our final stride 32.
Our strides are then [4, 8, 16, 32] and since the Detectron2's FPN implementation uses 
the last/largest stride (strides[-1]) of our bottom-up backbone's output features, size_divisibility is set to 32.
    (You can double check with model.backbone.size_divisibility after instantiating the model with this config file)

Thus, it's preferred that our image dimensions are divisible by 32.
Our max LSST size is 151x151 so we pad to 160x160 (32*5). 
You can theoretically set square_pad to whatever you want, but setting it to a value not divisible by size_divisibility 
just results in extra padding being added to make it divisible anyway.
"""

model.backbone.square_pad = 160
# using gradient checkpointing to save memory
# works by not storing intermediate activations for each layer, 
# instead recomputing them during the backward pass. 
# Reduces memory usage at the cost of extra computation. 
# But, it lets us use larger batch sizes
model.backbone.bottom_up.use_checkpoint = True
# for 6 channels (ugrizy)
model.backbone.bottom_up.in_chans = 6

# from training data of 109,782 imgs
model.pixel_mean = [
    0.057071752846241,
    0.05500221624970436,
    0.07863432168960571,
    0.11082268506288528,
    0.13925790786743164,
    0.21512141823768616,
]

model.pixel_std = [
    0.9746726155281067,
    0.6917526721954346,
    0.9822554588317871,
    1.382053017616272,
    1.8204922676086426,
    2.6324615478515625,
]





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
dataloader.test.mapper = loaders.DictMapper
dataloader.train.mapper = loaders.DictMapper
reader = RomanRubinImageReader()
dataloader.imagereader = reader
dataloader.steps_per_epoch = steps_per_epoch
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

SOLVER.STEPS = []  # do not decay learning rate for retraining
SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
SOLVER.WARMUP_ITERS = 0
