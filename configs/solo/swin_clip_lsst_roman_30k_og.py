from .swin_clip_lsst_roman_30k import dataloader, model, train, lr_multiplier, optimizer, metadata
from .swin_clip_lsst_roman_30k import MISC, DATALOADER, DATASETS, GLOBAL, INPUT, MODEL, SOLVER, TEST

from ..custom.roiheads import ContrastiveCascadeROIHeads
model.roi_heads._target_ = ContrastiveCascadeROIHeads
model.roi_heads.contrastive_dim = 128
model.roi_heads.contrastive_hidden_dim = 1024
model.roi_heads.contrastive_weight = 1.0
model.roi_heads.temperature = 0.07