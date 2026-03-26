""" This is a demo "solo config" file for use in solo_test_run_transformers.py.

This uses template configs cascade_mask_rcnn_swin_b_in21k_50ep and yaml_style_defaults."""

from omegaconf import OmegaConf
import numpy as np
import math
# ---------------------------------------------------------------------------- #
# Local variables and metadata
# ---------------------------------------------------------------------------- #
# bs: total batch size spread across multiple GPUs & make sure it's divisible by the number of GPUs
# 4 A100x4 GPUs --> bs = 64 (16 per gpu) (Max GPU util and ~85% GPU memory), test_bs = bs * 2
# we must also ensure that per gpu batch size is divisible by K used in meta_arch.py
# K=8192 for 30k set and 8192/16 = 512 so we can set bs = 64
# for stage 1 freeze, using same bs means only 50% GPU memory usage so in the future, we could change queue size K to get higher bs as 32/GPU runs out of memory
#bs = 192
#bs = 24 # for testing on A100

# 4 H200 GPUS --> bs = 256 (64 per gpu) (Max GPU Util and ~87-95% GPU memory), test_bs = bs * 2 
# K = 8192 for 30k and 8192/64 = 128 so we can set bs = 256
# bs = 256
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
from ..custom.image_readers import DualRubinRubinImageReader, DualRomanRubinImageReader, RomanRubinImageReader
from ..custom.mappers import ResizeCombinedandLSSTMapper, CLIPEvalMapper 
import deepdisc.model.loaders as loaders

from deepdisc.data_format.augment_image import dc2_train_augs
from ..custom.meta_arch_test import GeneralizedRCNNMultimodal, DynamicSwinTransformer
from ..custom.roiheads_distill import DistillCascadeROIHeads
from detectron2.layers import ShapeSpec
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import KRCNNConvDeconvUpsampleHead
from detectron2.config import LazyCall as L


# Overrides of the template COCO config
# This is the cascade mask rcnn of an ImageNet-swin_base_patch4_window7_224_21k model
# train.init_checkpoint = "/projects/bdsp/yse2/cascade_mask_rcnn_swin_b_in21k_moco_model.pkl"
train.init_checkpoint = "/work/hdd/bdsp/pretrained_models/cascade_mask_rcnn_swin_b_in21k_student_teacher_model.pkl"
#train.init_checkpoint = None

# change to the frozen swin checkpoint for stage 2 of training
# train.init_checkpoint = "/projects/bdsp/yse2/cascade_mask_rcnn_swin_b_in21k_frozen_moco_model.pkl"

# for TimedLazyAstroTrainer (all in iters)
train.timing_report_period = steps_per_epoch // 2 # report every n iters (for testing, 5)
train.timing_rolling_window_size = steps_per_epoch // 2 # average over last n iters (for testing, 5)
train.timing_save_period = steps_per_epoch # save timing to disk every n iters (for testing, 10)

dataloader.augs = dc2_train_augs
dataloader.train.total_batch_size = bs
# when bs=96, bs * 6 but when bs=144, bs * 6 crashed so using bs * 4 and for bs=192, bs * 2
# dataloader.test.total_batch_size = bs * 2 # higher since no gradients being calculated
# two encoders double the memory usage, so reducing test batch size compared to baseline run
# dataloader.test.total_batch_size = bs * 2 # higher since no gradients being calculated
dataloader.test.total_batch_size = bs # higher since no gradients being calculated
dataloader.test.num_workers = 16

model.proposal_generator.anchor_generator.sizes = [[8], [16], [32], [64], [128]]
model.roi_heads.num_classes = numclasses
model.roi_heads.batch_size_per_image = 512



#Keypoint ROI head taken from the config on detectron2 repo 
model.roi_heads.update(
    keypoint_in_features=["p2", "p3", "p4", "p5"],
    keypoint_pooler=L(ROIPooler)(
        output_size=14,
        scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
        sampling_ratio=0,
        pooler_type="ROIAlignV2",
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


# use _target_ to only swap the class type but keeps existing args from parent config
model.roi_heads._target_= DistillCascadeROIHeads
model.roi_heads.kd_weight=1.0


#--------------------------------------------------------
#model.backbone.bottom_up._target_ = DynamicSwinTransformer 
model._target_ = GeneralizedRCNNMultimodal
model.backbone_student = model.backbone # query is Rubin
model.backbone_teacher = model.backbone # key is Roman
model.pop('backbone')
model.pop('pixel_mean')
model.pop('pixel_std')

# change the param weight names in the checkpoint to match our architecture

model.backbone_student.bottom_up.in_chans = 6 # ugrizy for Rubin
model.backbone_teacher.bottom_up.in_chans = 9 # Y106, J129, H158 for Roman
# using gradient checkpointing to save memory
# works by not storing intermediate activations for each layer, 
# instead recomputing them during the backward pass. 
# Reduces memory usage at the cost of extra computation. 
# But, it lets us use larger batch sizes
model.backbone_student.bottom_up.use_checkpoint = True
model.backbone_teacher.bottom_up.use_checkpoint = True
# set patch sizes to be proportional to square_pad sizes
"""
in_features (list[str]): names of the input feature maps coming
    from the backbone to which FPN is attached.
The FPN in_features from the Swin Transformer backbones (the encoders for both Rubin and Roman here)
are ("p0", "p1", "p2", "p3") and we set Rubin's patch size to Swin's default patch size/stride of 4
giving us a stride of 4 with 3 patch merging layers (which each double the stride) making our final stride 32.
Our strides are then [4, 8, 16, 32] and since the Detectron2's FPN implementation (detectron2/modeling/backbone/fpn.py) uses 
the last/largest stride (strides[-1]) of our bottom-up backbone's output features, size_divisibility is set to 32.
    (You can double check with model.backbone_student.size_divisibility after instantiating the model with this config file)
Thus, it's preferred that our image dimensions are divisible by 32.
Our max Rubin size is 151x151 so we pad to 160x160 (32*5). 
You can theoretically set square_pad to whatever you want, but setting it to a value not divisible by size_divisibility 
just results in extra padding being added to make it divisible anyway.
"""
model.backbone_student.bottom_up.patch_size = 4
query_size_div = 32 # query_patch_size * 2 ** (num of swin stages p0 p1 p2 p3 - 1)
max_query_img_size = 151
model.backbone_student.square_pad = math.ceil(max_query_img_size / query_size_div) * query_size_div

'''
Do not rescale the Roman feature maps by altering the stride.  Instead, we will rescale the ROIs during ROI pooling 

'''
max_key_img_size = 512
model.backbone_teacher.bottom_up.patch_size = 4  # ceil(512/40) = 13
model.backbone_teacher.square_pad = max_key_img_size  # 40*13 = 520


model.beta = 1.0 # supervised loss weight


# from rubin training data of 109,782 imgs
# model.pixel_mean = [
#     0.057071752846241,
#     0.05500221624970436,
#     0.07863432168960571,
#     0.11082268506288528,
#     0.13925790786743164,
#     0.21512141823768616,
# ]

# model.pixel_std = [
#     0.9746726155281067,
#     0.6917526721954346,
#     0.9822554588317871,
#     1.382053017616272,
#     1.8204922676086426,
#     2.6324615478515625,
# ]

# for 30k training set
model.rubin_pixel_mean = [
    0.05976027995347977,
    0.056569650769233704,
    0.0808037668466568,
    0.11346549540758133,
    0.14247749745845795,
    0.22078551352024078,
]
model.rubin_pixel_std = [
    1.0054351091384888,
    0.7062947750091553,
    1.0013556480407715,
    1.4049317836761475,
    1.8567354679107666,
    2.689509153366089,
]

# for 30k training set
model.combined_pixel_mean = [
    0.05976027995347977,
    0.056569650769233704,
    0.0808037668466568,
    0.11346549540758133,
    0.14247749745845795,
    0.22078551352024078,
    1.0947377681732178,
    1.2559534311294556,
    1.3356200456619263,
]
model.combined_pixel_std = [
    1.0054351091384888,
    0.7062947750091553,
    1.0013556480407715,
    1.4049317836761475,
    1.8567354679107666,
    2.689509153366089,
    31.218294143676758,
    32.78688049316406,
    32.67300796508789,
]


model.proposal_generator.nms_thresh = 0.3
for box_predictor in model.roi_heads.box_predictors:
    box_predictor.test_topk_per_image = 2000
    box_predictor.test_score_thresh = 0.5
    box_predictor.test_nms_thresh = 0.3
    
# change this function depending on the metadata format
# needs to return where the cutout image data for each cutout is stored

def lsst_key_mapper(dataset_dict):
    key = dataset_dict["file_name"]
    k = key.replace("/u/","/work/hdd/bdsp/")
    return k

dataloader.key_mapper = lsst_key_mapper
dataloader.train.mapper = ResizeCombinedandLSSTMapper
dataloader.test.mapper = CLIPEvalMapper
reader = DualRomanRubinImageReader()
eval_reader = RomanRubinImageReader()
dataloader.train.imagereader = reader
dataloader.test.imagereader = eval_reader
dataloader.steps_per_epoch = steps_per_epoch
dataloader.train.keypoint_hflip_indices=[0]
dataloader.test.keypoint_hflip_indices=[0]
dataloader.train.cache_dir='/work/hdd/bfhm/g4merz/wcs_map_cache/train_30k_keypoints_wcs'


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

# 22400 took 17330s ~ 4.8138889 hrs
SOLVER.STEPS = []  # do not decay learning rate for retraining
SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
SOLVER.WARMUP_ITERS = 0
