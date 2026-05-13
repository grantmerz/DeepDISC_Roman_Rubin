""" This is a demo "solo config" file for use in solo_test_run_transformers.py.

This uses template configs cascade_mask_rcnn_swin_b_in21k_50ep and yaml_style_defaults."""

from omegaconf import OmegaConf
import math
# ---------------------------------------------------------------------------- #
# Local variables and metadata
# ---------------------------------------------------------------------------- #
# bs: total batch size spread across multiple GPUs & make sure it's divisible by the number of GPUs
bs = 192 # for 4 H200 GPUS with 140G memory each as each GPU can handle bs=48
# bs = 16 for 2xA100 GPUs so each GPU can handle bs = 8
# steps_per_epoch = num of steps per epoch = num of training images / bs
# a training step/iteration is when the model weights are updated (once per batch)
num_imgs = 30000
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
from ..custom.mappers import ResizeMapper

from deepdisc.data_format.augment_image import dc2_train_augs
from detectron2.layers import ShapeSpec
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import KRCNNConvDeconvUpsampleHead
from detectron2.config import LazyCall as L
from ..custom.meta_arch import DynamicSwinTransformer
from ..custom.poolers import NonPowerOf2ROIPooler, CustomLastLevelMaxPool

# Overrides of the template COCO config
# This is the cascade mask rcnn of an ImageNet-swin_base_patch4_window7_224_21k model
# train.init_checkpoint = "/projects/bdsp/yse2/cascade_mask_rcnn_swin_b_in21k_model.pkl"
train.init_checkpoint = "/projects/bdsp/yse2/cascade_mask_rcnn_swin_b_in21k_model_ps13.pkl"
# for TimedLazyAstroTrainer (all in iters)
train.timing_report_period = steps_per_epoch // 2 # report every n iters (for testing, 5)
train.timing_rolling_window_size = steps_per_epoch // 2 # average over last n iters (for testing, 5)
train.timing_save_period = steps_per_epoch # save timing to disk every n iters (for testing, 10)

dataloader.augs = dc2_train_augs
dataloader.train.total_batch_size = bs
# dataloader.train.num_workers = 16
# when bs=96, bs * 6 but when bs=144, bs * 6 crashed so using bs * 4 and for bs=192, bs * 2
dataloader.test.total_batch_size = bs # higher since no gradients being calculated
# dataloader.test.num_workers = 16 keeps crashing during validation when bs=192
dataloader.test.num_workers = 12

# w/ patch_size=13, DynamicSwinTransformer gives us backbone strides [13, 26, 52, 104]
# use _target_ to only swap the class type but keeps existing args from parent config
model.backbone.bottom_up._target_ = DynamicSwinTransformer
# default patch size is 4 for SWIN, so feature size for 160x160 cutouts is 160/4 = 40
baseline_feature_size = 160 // 4
model.backbone.bottom_up.patch_size = math.ceil(512 / baseline_feature_size)  # ceil(512/40) = 13

# FPN names outputs using int(log2(stride)): int(log2(13))=3 --> p3, ..., int(log2(104))=6 --> p6
# The top block (CustomLastLevelMaxPool) adds p7 with stride 208 (even though the FPN metadata will have the stride as 128)
# ALL in_features lists and ROI pooler scales must reflect these new names/strides
ps13_features = ["p3", "p4", "p5", "p6"]
ps13_strides = [13, 26, 52, 104, 208]
ps13_scales = tuple(1.0 / s for s in ps13_strides[:4])  # (1/13, 1/26, 1/52, 1/104) for ROI heads
# RPN: update which FPN levels are used and anchor placement strides
model.proposal_generator.in_features = ps13_features + ["p7"]
model.proposal_generator.anchor_generator.strides = ps13_strides
model.proposal_generator.anchor_generator.sizes = [[8], [16], [32], [64], [128]]

model.roi_heads.num_classes = numclasses
model.roi_heads.batch_size_per_image = 512

# Box ROI pooler
model.roi_heads.box_in_features = ps13_features
model.roi_heads.box_pooler = L(NonPowerOf2ROIPooler)(
    output_size=7,
    scales=ps13_scales,
    sampling_ratio=0,
    pooler_type="ROIAlignV2",
    canonical_level=6,
)
# Mask ROI pooler
model.roi_heads.mask_in_features = ps13_features
model.roi_heads.mask_pooler = L(NonPowerOf2ROIPooler)(
    output_size=14,
    scales=ps13_scales,
    sampling_ratio=0,
    pooler_type="ROIAlignV2",
    canonical_level=6,
)
# Keypoint ROI head
model.roi_heads.update(
    keypoint_in_features=ps13_features,
    keypoint_pooler=L(NonPowerOf2ROIPooler)(
        output_size=14,
        scales=ps13_scales,
        sampling_ratio=0,
        pooler_type="ROIAlignV2",
        canonical_level=6,
    ),
    keypoint_head=L(KRCNNConvDeconvUpsampleHead)(
        input_shape=ShapeSpec(channels=256, width=14, height=14),
        num_keypoints=1,
        conv_dims=[512] * 8,
        loss_normalizer="visible",
    ),
)
#Keypoint AP degrades (though box AP improves) when using plain L1 loss
for box_predictor in model.roi_heads.box_predictors:
    box_predictor.smooth_l1_beta = 0.5

# Default LastLevelMaxPool takes p5 with ps13 FPN outputs ["p3","p4","p5","p6"],
# but p5 has stride 52 --> maxpool --> 5×5, which duplicates p6 (stride 104, also 5×5).
# CustomLastLevelMaxPool takes p6 (stride 104)
model.backbone.top_block = L(CustomLastLevelMaxPool)(in_feature="p6")

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
model.backbone.square_pad = 512
# using gradient checkpointing to save memory
# works by not storing intermediate activations for each layer, 
# instead recomputing them during the backward pass. 
# Reduces memory usage at the cost of extra computation. 
# But, it lets us use larger batch sizes
model.backbone.bottom_up.use_checkpoint = True
# for 6 channels (ugrizy)
model.backbone.bottom_up.in_chans = 6

# for 30k training set
# model.pixel_mean = [
#     0.05976027995347977,
#     0.056569650769233704,
#     0.0808037668466568,
#     0.11346549540758133,
#     0.14247749745845795,
#     0.22078551352024078,
# ]
# model.pixel_std = [
#     1.0054351091384888,
#     0.7062947750091553,
#     1.0013556480407715,
#     1.4049317836761475,
#     1.8567354679107666,
#     2.689509153366089,
# ]
# using the new means and stds of upsampled lsst from calc_pixel_mean_std.py 4/21/26
model.pixel_mean = [
    0.059525945963155996,
    0.05607563838058074,
    0.08002280354759617,
    0.11240181841218273,
    0.14113771833545105,
    0.21905372687133592,
]
model.pixel_std = [
    8.543412165904854,
    3.220351274634394,
    3.971109699288376,
    5.308931779536128,
    7.139499556348322,
    13.41088501316321,
]

model.proposal_generator.nms_thresh = 0.3
for box_predictor in model.roi_heads.box_predictors:
    box_predictor.test_topk_per_image = 2000
    box_predictor.test_score_thresh = 0.5
    box_predictor.test_nms_thresh = 0.3
    
# change this function depending on the metadata format
# needs to return where the cutout image data for each cutout is stored
def key_mapper(dataset_dict):
    key = dataset_dict["file_name"]
    return key.replace("/u/","/work/hdd/bdsp/")

dataloader.key_mapper = key_mapper
dataloader.train.mapper = ResizeMapper
reader = RomanRubinImageReader()
dataloader.train.imagereader = reader
dataloader.train.keypoint_hflip_indices=[0]

dataloader.test.mapper = ResizeMapper
dataloader.test.imagereader = reader
dataloader.test.keypoint_hflip_indices=[0]

dataloader.train.cache_dir='/work/hdd/bfhm/g4merz/wcs_map_cache/train_30k_keypoints_wcs'
dataloader.test.cache_dir='/work/hdd/bfhm/g4merz/wcs_map_cache/val_4k_keypoints_wcs'

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
