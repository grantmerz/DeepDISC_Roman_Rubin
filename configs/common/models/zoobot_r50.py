from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2 import model_zoo
from detectron2.solver import WarmupParamScheduler
from fvcore.common.param_scheduler import MultiStepParamScheduler
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.roi_heads import FastRCNNOutputLayers, FastRCNNConvFCHead, CascadeROIHeads
from ..coco_loader_lsj import dataloader
from .mask_rcnn_fpn import model
from omegaconf import OmegaConf
import numpy as np


# Initialization and trainer settings
train = model_zoo.get_config("common/train.py").train
train.amp.enabled = True
train.ddp.fp16_compression = True

# Schedule
# 100 ep = 184375 iters * 64 images/iter / 118000 images/ep
train.max_iter = 184375
lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[163889, 177546],
        num_updates=train.max_iter,
    ),
    warmup_length=250 / train.max_iter,
    warmup_factor=0.001,
)

optimizer = model_zoo.get_config("common/optim.py").AdamW
optimizer.params.overrides = {"pos_embed": {"weight_decay": 0.0}}
optimizer.lr = 8e-5


# arguments that don't exist for Cascade R-CNN
[model.roi_heads.pop(k) for k in ["box_head", "box_predictor", "proposal_matcher"]]

model.roi_heads.update(
    _target_=CascadeROIHeads,
    box_heads=[
        L(FastRCNNConvFCHead)(
            input_shape=ShapeSpec(channels=256, height=7, width=7),
            conv_dims=[],
            fc_dims=[1024, 1024],
        )
        for k in range(3)
    ],
    box_predictors=[
        L(FastRCNNOutputLayers)(
            input_shape=ShapeSpec(channels=1024),
            test_score_thresh=0.05,
            box2box_transform=L(Box2BoxTransform)(weights=(w1, w1, w2, w2)),
            cls_agnostic_bbox_reg=True,
            num_classes="${...num_classes}",
        )
        for (w1, w2) in [(10, 5), (20, 10), (30, 15)]
    ],
    proposal_matchers=[
        L(Matcher)(thresholds=[th], labels=[0, 1], allow_low_quality_matches=False)
        for th in [0.5, 0.6, 0.7]
    ],
)


# ---------------------------------------------------------------------------- #
# Local variables and metadata
# ---------------------------------------------------------------------------- #
bs = 1

metadata = OmegaConf.create() 
metadata.classes = ["object"]

numclasses = len(metadata.classes)

# Overrides
dataloader.augs = dc2_train_augs
dataloader.train.total_batch_size = bs

model.proposal_generator.anchor_generator.sizes = [[8], [16], [32], [64], [128]]
model.roi_heads.num_classes = numclasses
model.roi_heads.batch_size_per_image = 512

model.roi_heads.num_classes = numclasses
model.roi_heads.batch_size_per_image = 512
model.backbone.bottom_up.stem.in_channels=6

model.pixel_mean = [0.05485338, 0.05625433, 0.08389594, 0.11460936, 0.15186928, 0.22200608]
model.pixel_std =[2.7777975, 2.009476, 2.780979, 3.7311685, 4.749582, 7.093875 ]


model.roi_heads.num_components = 5
model.roi_heads.zloss_factor = 1
model.roi_heads.cmax = 5
model.roi_heads.coarse_bins = np.linspace(0,3,300)

#model.roi_heads.weights = np.load('/home/g4merz/rail_deepdisc/configs/solo/zweights.npy')

model.roi_heads._target_ = RedshiftPDFCasROIHeadsGold

model.proposal_generator.pre_nms_topk=[30000,10000]
model.proposal_generator.post_nms_topk=[30000,10000]
#model.proposal_generator.post_nms_topk=[6000,1000]
#model.proposal_generator.nms_thresh = 0.3
for box_predictor in model.roi_heads.box_predictors:
    box_predictor.test_topk_per_image = 3000
    box_predictor.test_score_thresh = 0.3
    box_predictor.test_nms_thresh = 0.5
