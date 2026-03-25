from typing import List

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from detectron2.modeling.poolers import ROIPooler
from .poolers import ScaledROIPooler

from detectron2.modeling.roi_heads import CascadeROIHeads, select_foreground_proposals
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
import torch


class DistillCascadeROIHeads(CascadeROIHeads):
    """
    Extends Detectron2's CascadeROIHeads with an ROI-level CLIP-style contrastive
    loss between Rubin (query) and Roman (key) encoder features

    Detection flow (standard Cascade R-CNN):
        Rubin FPN features -> box head (x3 cascade stages) -> cls + bbox losses
                           -> mask head -> mask loss
                           -> keypoint head -> keypoint loss

    Contrastive flow (ROI-level InfoNCE, only during training with paired Roman data):
        1. Augment proposals with ground-truth boxes so every labeled object
           participates in the contrastive loss
        2. Filter down to foreground (positive) proposals only
        3. ROI-align both Rubin and Roman feature pyramids at those box locations
           -> (N, 256, 7, 7) tensors
        4. AdaptiveMaxPool2d to (N, 256, 1, 1) -> flatten -> (N, 256)
        5. Pass through separate proj_q / proj_k MLP heads -> L2-normalized
           embeddings z_q, z_k of shape (N, contrastive_dim)
        6. Compute symmetric InfoNCE loss: treat each (z_q_i, z_k_i) pair (diagonals) as a
           positive and all other pairs within the batch as negatives
           loss = 0.5 * (CE(z_q @ z_k.T * scale, arange) +
                         CE(z_k @ z_q.T * scale, arange))
        7. Scale by contrastive_weight and return under key 'roi_contrastive_loss'

    The logit_scale (inverse temperature) is a learnable scalar initialized to
    log(1/T) so exp(logit_scale) ~ 1/T, matching CLIP's learnable temp
    """
    def __init__(
        self,
        kd_weight: float, # weight applied to contrastive loss term
        *,
        box_in_features: List[str],
        box_pooler: ScaledROIPooler,
        box_heads: List[nn.Module],
        box_predictors: List[nn.Module],
        proposal_matchers,
        **kwargs,
    ):
        # Initialize the standard CascadeROIHeads (box, mask, keypoint heads)
        super().__init__(
            box_in_features=box_in_features,
            box_pooler=box_pooler,
            box_heads=box_heads,
            box_predictors=box_predictors,
            proposal_matchers=proposal_matchers,
            **kwargs,
        )
        # weight so we can balance the contrastive loss against det losses
        # default 1.0 means treat it as equally important; can be tuned based on loss curves/perf
        self.kd_weight = kd_weight
 
        # ROI pooler for contrastive features (separate from box pooler)
        # Uses same FPN level scales as standard Cascade R-CNN box pooler:
        #   p2 -> scale 1/4, p3 -> 1/8, p4 -> 1/16, p5 -> 1/32
        # output_size=7 -> each ROI becomes a 7x7 spatial feature map before pooling
        # ROIAlignV2 (aligned=True) avoids 0.5-pixel quantization misalignment of
        # OG ROIPool, giving better-localized features
        self.student_feature_pooler = ROIPooler(
            output_size=7,
            scales=tuple(k for k in [0.25, 0.125, 0.0625, 0.03125]),
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        )
        
        # Because the key feature maps have a different size as the query feature maps, we
        # must rescale the features by a scale_factor.  This ensures the boxes are rescaled correctly,
        # and ensures the same feature level assignment as the default base-2 strides.
        self.teacher_feature_pooler = ScaledROIPooler(
            output_size=7,
            scales=tuple(k*3.478 for k in [0.25, 0.125, 0.0625, 0.03125]),
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
            scale_factor = 1/3.478,
        )
        # infer box head output dim
        self.in_channels = 256 # TODO: grab this dynmically from the box head output channels instead of hardcoding


    def forward_distill(self, features_student, features_teacher, instances, targets=None):
        """
        Distill knowledge between a frozen "teacher" model and a learnable "student" model

        The key idea is that the teacher model is trained with Rubin+Roman, 
        whereas the student is only trained with Rubin.  We try to match the ROI features of the Rubin model 
        to the ROI features of the teacher model

        Args:
            features_student (dict[str, Tensor]): FPN feature maps from the Rubin (query)
                backbone, keyed by feature level name (e.g. "p2", "p3", "p4", "p5").
            features_teacher (dict[str, Tensor]): FPN feature maps from the Roman (key)
                backbone, same keys as features_student after feature-name alignment in
                GeneralizedRCNNMultimodal
            instances (list[Instances]): Per-image proposal boxes from the RPN,
                already matched to ground-truth (labeled by label_and_sample_proposals)
            targets (list[Instances]): Ground-truth instances per image

        Returns:
            dict[str, Tensor]: {"roi_contrastive_loss": scalar tensor}
        """

        if self.training:
            # Augment RPN proposals with every GT box to guarantee all labeled
            # objects participate in the contrastive loss. Bc RPN may completely
            # miss some objects (e.g., faint/small galaxies with no proposal
            # exceeding the IoU threshold) giving them no contrastive
            # signal at all, we add GT boxes to directly address it
            proposals = add_ground_truth_to_proposals(targets, instances)
            # keep only foreground proposals (those with a positive class label) since
            # background proposals don't contribute to the contrastive loss and wld just add noise
            instances, _ = select_foreground_proposals(proposals, self.num_classes)

        # if no foreground instances exist in this batch. return 0 loss
        if sum(len(p) for p in instances) == 0:
            return {"distillation_loss": torch.tensor(0.0, device=list(features_student.values())[0].device)}

        if (self.student_feature_pooler is not None):
            # Select the FPN levels that the box head uses (box_in_features)
            # ROIAlignV2 automatically assigns each box to the appropriate FPN
            # level based on its area (following the FPN level assignment rule)
            features_student = [features_student[f] for f in self.box_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            # Pool Rubin features: (N_total_rois, 256, 7, 7)
            features_student = self.student_feature_pooler(features_student, boxes)

            features_teacher = [features_teacher[f] for f in self.box_in_features]
            # Reuse exactly the same box coords for Roman so that
            # spatial regions being compared are aligned across modalities
            # Roman feature pooler must be a ScaledPooler since the Roman feature maps are not the same size as Rubin 
            features_teacher = self.teacher_feature_pooler(features_teacher, boxes)  

        #def normalize_per_roi(x):
        #    mean = x.mean(dim=[2,3], keepdim=True)
        #    std = x.std(dim=[2,3], keepdim=True) + 1e-6
        #    return (x - mean) / std

        #features_student_norm = normalize_per_roi(features_student)
        #features_teacher_norm = normalize_per_roi(features_teacher)

        #loss_distill = F.mse_loss(features_student_norm, features_teacher_norm)
        loss_distill = F.mse_loss(features_student, features_teacher.detach())


        if self.training:
            return {
                "loss_distillation": loss_distill * self.kd_weight
            }

    def forward(self, images, features_student, proposals, targets=None, features_teacher=None):
        """ 
        Ex: self.roi_heads(images_q_rubin, features_student, proposals, gt_instances, features_teacher) calls this forward function
        During training:
            1. label_and_sample_proposals: assigns GT labels to each RPN proposal
               and sub-samples to batch_size_per_image (e.g. 512) with a fixed
               foreground/background ratio
            2. forward_distill: runs knowledge distillation between student and teacher models
            3. _forward_box: runs the 3-stage Cascade R-CNN box regression and
               classification heads, returning classification + regression losses
            4. _forward_mask: runs the mask head on foreground proposals,
               returning mask loss
            5. _forward_keypoint: runs the keypoint head on foreground proposals,
               returning keypoint loss
            

        During inference:
            1. _forward_box: predicts boxes and class scores, applies NMS
            2. forward_with_given_boxes: runs mask and keypoint heads on the
               surviving detections to produce final per-instance predictions

        Args:
            images: unused (deleted immediately); kept for API compatibility with
                the parent CascadeROIHeads signature
            features_student (dict[str, Tensor]): Rubin FPN features (query encoder)
            proposals (list[Instances]): RPN proposals per image
            targets (list[Instances] | None): GT instances; required during training
        Returns:
            training: (proposals, losses_dict)
                proposals are the labeled/sampled proposals (returned so that
                GeneralizedRCNNMultimodal can reuse them for the contrastive loss)
            inference: (pred_instances, {})
        """
        del images  # not used; images were already preprocessed upstream
        if self.training:
            # Assigns GT class labels and box regression targets to each proposal
            # and sub-samples them to a fixed budget (batch_size_per_image)
            proposals = self.label_and_sample_proposals(proposals, targets)

        if self.training:
            # targets must be forwarded to the cascade box head for computing
            # regression targets at each stage
            # run distillation loss first
            losses = self.forward_distill(features_student,features_teacher,proposals,targets)
            losses.update(self._forward_box(features_student, proposals, targets))
            losses.update(self._forward_mask(features_student, proposals))
            losses.update(self._forward_keypoint(features_student, proposals))
            return proposals, losses
        else:
            # box head: runs 3 cascade reg stages, merges preds, applies class-specific NMS
            # returning surviving instances
            pred_instances = self._forward_box(features_student, proposals)
            # mask + keypoint heads: run on surviving boxes after NMS
            pred_instances = self.forward_with_given_boxes(features_student, pred_instances)
            return pred_instances, {}


