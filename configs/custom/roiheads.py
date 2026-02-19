from typing import List

import numpy as np
import torch
from astropy.wcs import WCS
from detectron2.layers import ShapeSpec, cat
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import CascadeROIHeads, select_foreground_proposals
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from torch import nn
from torch.distributions.categorical import Categorical
from torch.distributions.independent import Independent
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.normal import Normal


from typing import List
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import cat
from detectron2.modeling.roi_heads import CascadeROIHeads
from detectron2.modeling.poolers import ROIPooler


class ROIContrastiveHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=1024, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        z = self.net(x)
        return F.normalize(z, dim=1)


class ContrastiveCascadeROIHeads(CascadeROIHeads):
    """
    Cascade ROI heads with ROI-level contrastive loss between
    query/key encoder features.
    """

    def __init__(
        self,
        contrastive_dim: int,
        contrastive_hidden_dim: int,
        contrastive_weight: float,
        temperature: float,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_heads: List[nn.Module],
        box_predictors: List[nn.Module],
        proposal_matchers,
        **kwargs,
    ):
        super().__init__(
            box_in_features=box_in_features,
            box_pooler=box_pooler,
            box_heads=box_heads,
            box_predictors=box_predictors,
            proposal_matchers=proposal_matchers,
            **kwargs,
        )

        self.contrastive_weight = contrastive_weight
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/temperature))

        self.feature_pooler = ROIPooler(
            output_size=7,
            scales=tuple(k for k in [0.25, 0.125, 0.0625, 0.03125]),
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        )


        # infer box head output dim
        self.in_channels = 256


        self.proj_q = ROIContrastiveHead(
            self.in_channels, contrastive_hidden_dim, contrastive_dim
        )
        self.proj_k = ROIContrastiveHead(
            self.in_channels, contrastive_hidden_dim, contrastive_dim
        )



    def forward_contrastive(self, features_q, features_k, proposals, targets):
        """
        Compute ROI contrastive loss using SAME proposals.
        """

        if self.training:
            #Add all gt bounding boxes for contrastive loss
            proposals = add_ground_truth_to_proposals(targets, proposals)
            instances, _ = select_foreground_proposals(proposals, self.num_classes)


        if sum(len(p) for p in instances) == 0:
            return {"roi_contrastive_loss": torch.tensor(0.0, device=list(features_q.values())[0].device)}


        if self.feature_pooler is not None:
            features_q = [features_q[f] for f in self.box_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features_q = self.feature_pooler(features_q, boxes)

            features_k = [features_k[f] for f in self.box_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features_k = self.feature_pooler(features_k, boxes)


        features_q = nn.AdaptiveMaxPool2d((1, 1))(features_q)
        features_q = torch.flatten(features_q, 1)

        features_k = nn.AdaptiveMaxPool2d((1, 1))(features_k)
        features_k = torch.flatten(features_k, 1)

        z_q = self.proj_q(features_q)
        z_k = self.proj_k(features_k)

        logits = (z_q @ z_k.T) * self.logit_scale
        labels = torch.arange(len(z_q), device=z_q.device)

        loss_qk = F.cross_entropy(logits, labels)
        loss_kq = F.cross_entropy(logits.T, labels)

        loss = 0.5 * (loss_qk + loss_kq)

        return {"roi_contrastive_loss": loss * self.contrastive_weight}

    def forward(self, images, features_q, proposals, targets=None, image_wcs=None):
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)

        if self.training:
            # Need targets to box head
            losses = self._forward_box(features_q, proposals, targets)
            losses.update(self._forward_mask(features_q, proposals))
            losses.update(self._forward_keypoint(features_q, proposals))
            #losses.update(self.forward_contrastive(features_q, features_k, proposals, targets))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features_q, proposals)
            pred_instances = self.forward_with_given_boxes(features_q, pred_instances)
            return pred_instances, {}
