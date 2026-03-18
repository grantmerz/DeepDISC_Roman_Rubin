from typing import List

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from detectron2.modeling.poolers import ROIPooler
from .poolers import ScaledROIPooler

from detectron2.modeling.roi_heads import CascadeROIHeads, select_foreground_proposals
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals

class ROIContrastiveHead(nn.Module):
    """
    Two-layer MLP projection head used to map ROI-pooled features into a
    shared contrastive embedding space for both the query (Rubin) and key
    (Roman) encoders

    Architecture: Linear(in_dim -> hidden_dim) -> ReLU -> Linear(hidden_dim -> out_dim)
    
    We gotta L2-normalize output embeddings so that dot products equal cosine similarities,
    which is required for the symmetric InfoNCE loss

    Args:
        in_dim (int): Input feature dimension. Should match the number of channels
            coming out of the ROI pooler (default is 256 from box head)
        hidden_dim (int): Hidden layer width. Default 1024
        out_dim (int): Output embedding dimension. Default 128 (same as MoCo v2)
    """
    def __init__(self, in_dim, hidden_dim=1024, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        z = self.net(x)
        # L2-normalize along embedding dim so cosine similarity == dot prod
        return F.normalize(z, dim=1)

        # # MoCo params
        # m: float= 0.999, 
        # T: float= 0.07,
        # hidden_dim: int = 2048,
        # dim: int = 128,

class ContrastiveCascadeROIHeadsFlatten(CascadeROIHeads):
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
        contrastive_dim: int, # output embedding dimension (e.g. 128)
        contrastive_hidden_dim: int, # MLP hidden dimension (e.g. 1024)
        contrastive_weight: float, # weight applied to contrastive loss term
        temperature: float, # softmax temperature that scales the similarities in the embedding space; lower = sharper distribution
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
        self.contrastive_weight = contrastive_weight
        # Learnable log-scale temperature param, initialized to log(1/T)
        # Keeping it learnable lets the model adapt the sharpness (peakness) of the similarity
        # distribution over training, as done in CLIP (https://arxiv.org/abs/2103.00020)
        # https://naokishibuya.github.io/blog/2023-08-13-clip-2103/#the-structure-of-clip 
        # also has a nice explanation of why CLIP uses a learnable temp and how it evolves during training
        #self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/temperature))
        
        self.register_buffer("logit_scale", torch.tensor(np.log(1/temperature), dtype=torch.float32))
        
        # ROI pooler for contrastive features (separate from box pooler)
        # Uses same FPN level scales as standard Cascade R-CNN box pooler:
        #   p2 -> scale 1/4, p3 -> 1/8, p4 -> 1/16, p5 -> 1/32
        # output_size=7 -> each ROI becomes a 7x7 spatial feature map before pooling
        # ROIAlignV2 (aligned=True) avoids 0.5-pixel quantization misalignment of
        # OG ROIPool, giving better-localized features
        self.query_feature_pooler = ROIPooler(
            output_size=7,
            scales=tuple(k for k in [0.25, 0.125, 0.0625, 0.03125]),
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        )
        
        # Because the key feature maps have strides that are not base 2, we 
        # must rescale the features by a scale_factor.  This ensures the boxes are rescaled correctly,
        # and ensures the same feature level assignment as the default base-2 strides
        #self.key_feature_pooler = ScaledROIPooler(
        #    output_size=7,
        #    scales=tuple(k for k in [1/13, 1/26, 1/52, 1/104]),
        #    sampling_ratio=0,
        #    pooler_type="ROIAlignV2",
        #    scale_factor = 13/4,
        #)
        # infer box head output dim
        self.in_channels = 256 # TODO: grab this dynmically from the box head output channels instead of hardcoding

        # Separate projection heads for query (Rubin) and key (Roman) encoders
        # Having separate weights allows each modality to learn its own projection
        # into the shared embedding space like CLIP-models do
        self.proj_q = ROIContrastiveHead(
            256*7*7, contrastive_hidden_dim, contrastive_dim
        )
        self.proj_k = ROIContrastiveHead(
            256*7*7, contrastive_hidden_dim, contrastive_dim
        )

    def forward_contrastive(self, features_q, features_k, instances, targets=None):
        """
        Compute the ROI-level symmetric InfoNCE (CLIP-style) contrastive loss
        using the SAME set of proposals applied to both encoders

        The key idea is that a Rubin ROI crop and the spatially corresponding
        Roman ROI crop are a positive pair, while all other cross-image or
        cross-object combinations within the batch are negatives

        Args:
            features_q (dict[str, Tensor]): FPN feature maps from the Rubin (query)
                backbone, keyed by feature level name (e.g. "p2", "p3", "p4", "p5").
            features_k (dict[str, Tensor]): FPN feature maps from the Roman (key)
                backbone, same keys as features_q after feature-name alignment in
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
            return {"roi_contrastive_loss": torch.tensor(0.0, device=list(features_q.values())[0].device)}

        if (self.query_feature_pooler is not None):
            # Select the FPN levels that the box head uses (box_in_features)
            # ROIAlignV2 automatically assigns each box to the appropriate FPN
            # level based on its area (following the FPN level assignment rule)
            features_q = [features_q[f] for f in self.box_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            # Pool Rubin features: (N_total_rois, 256, 7, 7)
            features_q = self.query_feature_pooler(features_q, boxes)

            features_k = [features_k[f] for f in self.box_in_features]
            # Reuse exactly the same box coords for Roman so that
            # spatial regions being compared are aligned across modalities
            #boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            #features_k = self.key_feature_pooler(features_k, boxes)
            # technically, the query feature pooler is using the wrong strides when applied to the key features
            # however, because the feature maps are the 
            features_k = self.query_feature_pooler(features_k, boxes)  

        # Collapse 7x7 spatial grid to a single vector via flattening
        #features_q = nn.AdaptiveMaxPool2d((1, 1))(features_q) # (N, 256, 1, 1)
        features_q = torch.flatten(features_q, 1) # flatten -> (N, 256)

        #features_k = nn.AdaptiveMaxPool2d((1, 1))(features_k)
        features_k = torch.flatten(features_k, 1)

        # Project both sets of ROI features into the shared contrastive embedding
        # space and L2-normalize: (N, 256) -> (N, contrastive_dim)
        z_q = self.proj_q(features_q)
        z_k = self.proj_k(features_k)
        
        # Compute pairwise cosine similarity matrix scaled by the learnable
        # temperature: logits[i, j] = sim(z_q_i, z_k_j) / T
        # Shape: (N, N); diagonal entries are pos pairs
        logits = (z_q @ z_k.T) * self.logit_scale.exp()
        # GT labels: row i should be most similar to col i
        labels = torch.arange(len(z_q), device=z_q.device)
        
        # Symmetric InfoNCE: both directions (q->k and k->q) contribute equally
        # CE(logits, labels) treats each row's diagonal as correct cls
        loss_qk = F.cross_entropy(logits, labels)
        loss_kq = F.cross_entropy(logits.T, labels)
        # avg to get final loss
        loss = 0.5 * (loss_qk + loss_kq)

        if self.training:
            return {
                "roi_contrastive_loss": loss * self.contrastive_weight,
                # track effective temp (1/T = exp(logit_scale)) for monitoring
                # A rapidly growing value might mean contrastive collapse
                # Detached so it doesn't get used in the backward pass
                "logit_scale": self.logit_scale.exp().detach(),
            }

        else:
            return z_q, z_k

    def forward(self, images, features_q, proposals, targets=None):
        """ 
        Ex: self.roi_heads(images_q_rubin, features_q, proposals, gt_instances) calls this forward function
        During training:
            1. label_and_sample_proposals: assigns GT labels to each RPN proposal
               and sub-samples to batch_size_per_image (e.g. 512) with a fixed
               foreground/background ratio
            2. _forward_box: runs the 3-stage Cascade R-CNN box regression and
               classification heads, returning classification + regression losses
            3. _forward_mask: runs the mask head on foreground proposals,
               returning mask loss
            4. _forward_keypoint: runs the keypoint head on foreground proposals,
               returning keypoint loss
            NOTE: forward_contrastive isn't computed here; it's called separately 
            from GeneralizedRCNNMultimodal in meta_arc.py so it can access both Rubin and Roman features 

        During inference:
            1. _forward_box: predicts boxes and class scores, applies NMS
            2. forward_with_given_boxes: runs mask and keypoint heads on the
               surviving detections to produce final per-instance predictions

        Args:
            images: unused (deleted immediately); kept for API compatibility with
                the parent CascadeROIHeads signature
            features_q (dict[str, Tensor]): Rubin FPN features (query encoder)
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
            losses = self._forward_box(features_q, proposals, targets)
            losses.update(self._forward_mask(features_q, proposals))
            losses.update(self._forward_keypoint(features_q, proposals))
            # no CL loss here; it's called from GeneralizedRCNNMultimodal in meta_arc.py
            # so it can access both Rubin and Roman features
            return proposals, losses
        else:
            # box head: runs 3 cascade reg stages, merges preds, applies class-specific NMS
            # returning surviving instances
            pred_instances = self._forward_box(features_q, proposals)
            # mask + keypoint heads: run on surviving boxes after NMS
            pred_instances = self.forward_with_given_boxes(features_q, pred_instances)
            return pred_instances, {}



class ContrastiveCascadeROIHeadsSpatialGrid(CascadeROIHeads):
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
        contrastive_dim: int, # output embedding dimension (e.g. 128)
        contrastive_hidden_dim: int, # MLP hidden dimension (e.g. 1024)
        contrastive_weight: float, # weight applied to contrastive loss term
        temperature: float, # softmax temperature that scales the similarities in the embedding space; lower = sharper distribution
        contrastive_in_features: List[str], #features to use for gathering pos/neg samples for contrative learning
        contrastive_in_levels: List[float], #features to use for gathering pos/neg samples for contrative learning
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
        self.contrastive_weight = contrastive_weight
        self.contrastive_in_features = contrastive_in_features
        self.contrastive_in_levels = contrastive_in_levels

        # Learnable log-scale temperature param, initialized to log(1/T)
        # Keeping it learnable lets the model adapt the sharpness (peakness) of the similarity
        # distribution over training, as done in CLIP (https://arxiv.org/abs/2103.00020)
        # https://naokishibuya.github.io/blog/2023-08-13-clip-2103/#the-structure-of-clip 
        # also has a nice explanation of why CLIP uses a learnable temp and how it evolves during training
        
        #self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/temperature))
        self.register_buffer("logit_scale", torch.tensor(np.log(1/temperature), dtype=torch.float32))
        
        # ROI pooler for contrastive features (separate from box pooler)
        # Uses same FPN level scales as standard Cascade R-CNN box pooler:
        #   p2 -> scale 1/4, p3 -> 1/8, p4 -> 1/16, p5 -> 1/32
        # output_size=7 -> each ROI becomes a 7x7 spatial feature map before pooling
        # ROIAlignV2 (aligned=True) avoids 0.5-pixel quantization misalignment of
        # OG ROIPool, giving better-localized features
        #self.query_feature_pooler = ROIPooler(
        #    output_size=7,
        #    scales=tuple(k for k in [0.25, 0.125, 0.0625, 0.03125]),
        #    sampling_ratio=0,
        #    pooler_type="ROIAlignV2",
        #)

        self.query_feature_poolers = []
        for level in self.contrastive_in_levels:
            self.query_feature_poolers.append(ROIPooler(
                output_size=7,
                scales=[level],
                sampling_ratio=0,
                pooler_type="ROIAlignV2",
            ))
        
        # Because the key feature maps have strides that are not base 2, we 
        # must rescale the features by a scale_factor.  This ensures the boxes are rescaled correctly,
        # and ensures the same feature level assignment as the default base-2 strides
        #self.key_feature_pooler = ScaledROIPooler(
        #    output_size=7,
        #    scales=tuple(k for k in [1/13, 1/26, 1/52, 1/104]),
        #    sampling_ratio=0,
        #    pooler_type="ROIAlignV2",
        #    scale_factor = 13/4,
        #)
        # infer box head output dim
        self.in_channels = 256 # TODO: grab this dynmically from the box head output channels instead of hardcoding

        # Separate projection heads for query (Rubin) and key (Roman) encoders
        # Having separate weights allows each modality to learn its own projection
        # into the shared embedding space like CLIP-models do
        self.proj_q = ROIContrastiveHead(
            self.in_channels, contrastive_hidden_dim, contrastive_dim
        )
        self.proj_k = ROIContrastiveHead(
            self.in_channels, contrastive_hidden_dim, contrastive_dim
        )

    def forward_contrastive(self, features_q, features_k, instances, targets=None):
        """
        Compute the ROI-level symmetric InfoNCE (CLIP-style) contrastive loss
        using the SAME set of proposals applied to both encoders

        The key idea is that a Rubin ROI crop and the spatially corresponding
        Roman ROI crop are a positive pair, while all other cross-image or
        cross-object combinations within the batch are negatives

        Args:
            features_q (dict[str, Tensor]): FPN feature maps from the Rubin (query)
                backbone, keyed by feature level name (e.g. "p2", "p3", "p4", "p5").
            features_k (dict[str, Tensor]): FPN feature maps from the Roman (key)
                backbone, same keys as features_q after feature-name alignment in
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
            return {"roi_contrastive_loss": torch.tensor(0.0, device=list(features_q.values())[0].device)}


        num_instances_per_img = [len(i) for i in instances]
        N=np.sum(num_instances_per_img)

        multi_level_features_q = []
        multi_level_features_k = []
       
        if (self.query_feature_poolers is not None):
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]            
            for i,level_name in enumerate(self.contrastive_in_features):
                # Pool 7x7 from each FPN level regardless of box size
                f_q = self.query_feature_poolers[i]([features_q[level_name]], boxes)
                f_k = self.query_feature_poolers[i]([features_k[level_name]], boxes)
                multi_level_features_q.append(f_q) # (N, 256, 7, 7)
                multi_level_features_k.append(f_k) # (N, 256, 7, 7)

        # Stack them: (N, L, 256, 7, 7) where L is number of levels
        features_q = torch.stack(multi_level_features_q, dim=1)
        features_k = torch.stack(multi_level_features_k, dim=1)

        
        # Flatten spatial and levels to get a pool of patches per ROI
        # (N, L*49, 256)
        features_q = features_q.permute(0, 1, 3, 4, 2).reshape(N, -1, 256)
        features_k = features_k.permute(0, 1, 3, 4, 2).reshape(N, -1, 256)

        # Reshape 7x7 spatial grid into 49 separate vectors per ROI.  The contrastive loss is computed for each vector
        z_q = self.proj_q(features_q.flatten(2))
        z_k = self.proj_k(features_k.flatten(2))
        #z_q = F.normalize(features_q.flatten(2),dim=-1)
        #z_k = F.normalize(features_k.flatten(2),dim=-1)

        n_patches = z_q.shape[1] # This is N boxes * L levels

        # Now use BMM: each ROI is matched across its own multi-scale patches
        # Negatives are only considered within a given ROI - this is to ensure no "hard negatives" due to overlap
        logits_qk = torch.bmm(z_q, z_k.transpose(1, 2)) * self.logit_scale.exp()  # (N, L*49, L*49)
        logits_kq = logits_qk.transpose(1, 2)


        # 1. Reshape logits from (N, L*49, L*49) -> (N * L*49, L*49)
        # Every row in this new matrix is a Rubin patch looking for its Roman match
        logits_qk_reshaped = logits_qk.reshape(-1, n_patches)
        logits_kq_reshaped = logits_kq.reshape(-1, n_patches)
         
        # 2. Create the targets
        # For each ROI, patch i should match index i. 
        # labels: [0, 1...195, 0, 1...195, ...]
        labels = torch.arange(n_patches, device=logits_qk.device).repeat(N)
        
        #print('Labels ', labels.size())
        #print('Logits ', logits.size())
        #print('Logits reshaped', logits_reshaped.size())

        # Symmetric InfoNCE: both directions (q->k and k->q) contribute equally
        # CE(logits, labels) treats each row's diagonal as correct cls
        loss_qk = F.cross_entropy(logits_qk_reshaped, labels)
        loss_kq = F.cross_entropy(logits_kq_reshaped, labels)
        # avg to get final loss
        loss = 0.5 * (loss_qk + loss_kq)

        if self.training:
            return {
                "roi_contrastive_loss": loss * self.contrastive_weight,
                # track effective temp (1/T = exp(logit_scale)) for monitoring
                # A rapidly growing value might mean contrastive collapse
                # Detached so it doesn't get used in the backward pass
                "logit_scale": self.logit_scale.exp().detach(),
            }

        else:
            return z_q, z_k

    def forward(self, images, features_q, proposals, targets=None):
        """ 
        Ex: self.roi_heads(images_q_rubin, features_q, proposals, gt_instances) calls this forward function
        During training:
            1. label_and_sample_proposals: assigns GT labels to each RPN proposal
               and sub-samples to batch_size_per_image (e.g. 512) with a fixed
               foreground/background ratio
            2. _forward_box: runs the 3-stage Cascade R-CNN box regression and
               classification heads, returning classification + regression losses
            3. _forward_mask: runs the mask head on foreground proposals,
               returning mask loss
            4. _forward_keypoint: runs the keypoint head on foreground proposals,
               returning keypoint loss
            NOTE: forward_contrastive isn't computed here; it's called separately 
            from GeneralizedRCNNMultimodal in meta_arc.py so it can access both Rubin and Roman features 

        During inference:
            1. _forward_box: predicts boxes and class scores, applies NMS
            2. forward_with_given_boxes: runs mask and keypoint heads on the
               surviving detections to produce final per-instance predictions

        Args:
            images: unused (deleted immediately); kept for API compatibility with
                the parent CascadeROIHeads signature
            features_q (dict[str, Tensor]): Rubin FPN features (query encoder)
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
            losses = self._forward_box(features_q, proposals, targets)
            losses.update(self._forward_mask(features_q, proposals))
            losses.update(self._forward_keypoint(features_q, proposals))
            # no CL loss here; it's called from GeneralizedRCNNMultimodal in meta_arc.py
            # so it can access both Rubin and Roman features
            return proposals, losses
        else:
            # box head: runs 3 cascade reg stages, merges preds, applies class-specific NMS
            # returning surviving instances
            pred_instances = self._forward_box(features_q, proposals)
            # mask + keypoint heads: run on surviving boxes after NMS
            pred_instances = self.forward_with_given_boxes(features_q, pred_instances)
            return pred_instances, {}



class ContrastiveCascadeROIHeadsFlattenXAttention(CascadeROIHeads):
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
        embed_dim: int, # output embedding dimension (e.g. 128)
        contrastive_hidden_dim: int, # MLP hidden dimension (e.g. 1024)
        contrastive_weight: float, # weight applied to contrastive loss term
        temperature: float, # softmax temperature that scales the similarities in the embedding space; lower = sharper distribution
        n_head: int,  #number of attention heads for multihead cross attention
        dropout: float, #dropout factor for attention
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
        self.contrastive_weight = contrastive_weight
        # Learnable log-scale temperature param, initialized to log(1/T)
        # Keeping it learnable lets the model adapt the sharpness (peakness) of the similarity
        # distribution over training, as done in CLIP (https://arxiv.org/abs/2103.00020)
        # https://naokishibuya.github.io/blog/2023-08-13-clip-2103/#the-structure-of-clip 
        # also has a nice explanation of why CLIP uses a learnable temp and how it evolves during training
        #self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/temperature))
        
        self.register_buffer("logit_scale", torch.tensor(np.log(1/temperature), dtype=torch.float32))
        
        # ROI pooler for contrastive features (separate from box pooler)
        # Uses same FPN level scales as standard Cascade R-CNN box pooler:
        #   p2 -> scale 1/4, p3 -> 1/8, p4 -> 1/16, p5 -> 1/32
        # output_size=7 -> each ROI becomes a 7x7 spatial feature map before pooling
        # ROIAlignV2 (aligned=True) avoids 0.5-pixel quantization misalignment of
        # OG ROIPool, giving better-localized features
        self.query_feature_pooler = ROIPooler(
            output_size=7,
            scales=tuple(k for k in [0.25, 0.125, 0.0625, 0.03125]),
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        )
        
        # Because the key feature maps have strides that are not base 2, we 
        # must rescale the features by a scale_factor.  This ensures the boxes are rescaled correctly,
        # and ensures the same feature level assignment as the default base-2 strides
        #self.key_feature_pooler = ScaledROIPooler(
        #    output_size=7,
        #    scales=tuple(k for k in [1/13, 1/26, 1/52, 1/104]),
        #    sampling_ratio=0,
        #    pooler_type="ROIAlignV2",
        #    scale_factor = 13/4,
        #)
        # infer box head output dim
        self.in_channels = 256 # TODO: grab this dynmically from the box head output channels instead of hardcoding

        # Separate projection heads for query (Rubin) and key (Roman) encoders
        # Having separate weights allows each modality to learn its own projection
        # into the shared embedding space like CLIP-models do

        #Use cross attention to compute relevant features before embedding
        model_embed_dim = self.in_channels*7*7
        self.cross_attention_q = CrossAttentionHead(embed_dim,n_head,model_embed_dim,dropout)
        self.cross_attention_k = CrossAttentionHead(embed_dim,n_head,model_embed_dim,dropout)


        self.proj_q = ROIContrastiveHead(
            embed_dim, contrastive_hidden_dim, embed_dim
        )
        self.proj_k = ROIContrastiveHead(
            embed_dim, contrastive_hidden_dim, embed_dim
        )

    def forward_contrastive(self, features_q, features_k, instances, targets=None):
        """
        Compute the ROI-level symmetric InfoNCE (CLIP-style) contrastive loss
        using the SAME set of proposals applied to both encoders

        The key idea is that a Rubin ROI crop and the spatially corresponding
        Roman ROI crop are a positive pair, while all other cross-image or
        cross-object combinations within the batch are negatives

        Args:
            features_q (dict[str, Tensor]): FPN feature maps from the Rubin (query)
                backbone, keyed by feature level name (e.g. "p2", "p3", "p4", "p5").
            features_k (dict[str, Tensor]): FPN feature maps from the Roman (key)
                backbone, same keys as features_q after feature-name alignment in
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
            return {"roi_contrastive_loss": torch.tensor(0.0, device=list(features_q.values())[0].device)}

        if (self.query_feature_pooler is not None):
            # Select the FPN levels that the box head uses (box_in_features)
            # ROIAlignV2 automatically assigns each box to the appropriate FPN
            # level based on its area (following the FPN level assignment rule)
            features_q = [features_q[f] for f in self.box_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            # Pool Rubin features: (N_total_rois, 256, 7, 7)
            features_q = self.query_feature_pooler(features_q, boxes)

            features_k = [features_k[f] for f in self.box_in_features]
            # Reuse exactly the same box coords for Roman so that
            # spatial regions being compared are aligned across modalities
            #boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            #features_k = self.key_feature_pooler(features_k, boxes)
            # technically, the query feature pooler is using the wrong strides when applied to the key features
            # however, because the feature maps are the 
            features_k = self.query_feature_pooler(features_k, boxes)  

        # Collapse 7x7 spatial grid to a single vector via flattening
        #features_q = nn.AdaptiveMaxPool2d((1, 1))(features_q) # (N, 256, 1, 1)
        features_q = torch.flatten(features_q, 1) # flatten -> (N, 256)

        #features_k = nn.AdaptiveMaxPool2d((1, 1))(features_k)
        features_k = torch.flatten(features_k, 1)

        # Project both sets of ROI features into the shared contrastive embedding
        # space and L2-normalize: (N, 256) -> (N, contrastive_dim)
        z_q,atten_q = self.cross_attention_q(features_q) 
        z_k,atten_k = self.cross_attention_k(features_k) 

        z_q = self.proj_q(z_q)
        z_k = self.proj_k(z_k)
        
        # Compute pairwise cosine similarity matrix scaled by the learnable
        # temperature: logits[i, j] = sim(z_q_i, z_k_j) / T
        # Shape: (N, N); diagonal entries are pos pairs
        logits = (z_q @ z_k.T) * self.logit_scale.exp()
        # GT labels: row i should be most similar to col i
        labels = torch.arange(len(z_q), device=z_q.device)
        
        # Symmetric InfoNCE: both directions (q->k and k->q) contribute equally
        # CE(logits, labels) treats each row's diagonal as correct cls
        loss_qk = F.cross_entropy(logits, labels)
        loss_kq = F.cross_entropy(logits.T, labels)
        # avg to get final loss
        loss = 0.5 * (loss_qk + loss_kq)

        if self.training:
            return {
                "roi_contrastive_loss": loss * self.contrastive_weight,
                # track effective temp (1/T = exp(logit_scale)) for monitoring
                # A rapidly growing value might mean contrastive collapse
                # Detached so it doesn't get used in the backward pass
                "logit_scale": self.logit_scale.exp().detach(),
            }

        else:
            return z_q, z_k

    def forward(self, images, features_q, proposals, targets=None):
        """ 
        Ex: self.roi_heads(images_q_rubin, features_q, proposals, gt_instances) calls this forward function
        During training:
            1. label_and_sample_proposals: assigns GT labels to each RPN proposal
               and sub-samples to batch_size_per_image (e.g. 512) with a fixed
               foreground/background ratio
            2. _forward_box: runs the 3-stage Cascade R-CNN box regression and
               classification heads, returning classification + regression losses
            3. _forward_mask: runs the mask head on foreground proposals,
               returning mask loss
            4. _forward_keypoint: runs the keypoint head on foreground proposals,
               returning keypoint loss
            NOTE: forward_contrastive isn't computed here; it's called separately 
            from GeneralizedRCNNMultimodal in meta_arc.py so it can access both Rubin and Roman features 

        During inference:
            1. _forward_box: predicts boxes and class scores, applies NMS
            2. forward_with_given_boxes: runs mask and keypoint heads on the
               surviving detections to produce final per-instance predictions

        Args:
            images: unused (deleted immediately); kept for API compatibility with
                the parent CascadeROIHeads signature
            features_q (dict[str, Tensor]): Rubin FPN features (query encoder)
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
            losses = self._forward_box(features_q, proposals, targets)
            losses.update(self._forward_mask(features_q, proposals))
            losses.update(self._forward_keypoint(features_q, proposals))
            # no CL loss here; it's called from GeneralizedRCNNMultimodal in meta_arc.py
            # so it can access both Rubin and Roman features
            return proposals, losses
        else:
            # box head: runs 3 cascade reg stages, merges preds, applies class-specific NMS
            # returning surviving instances
            pred_instances = self._forward_box(features_q, proposals)
            # mask + keypoint heads: run on surviving boxes after NMS
            pred_instances = self.forward_with_given_boxes(features_q, pred_instances)
            return pred_instances, {}







class ContrastiveCascadeROIHeadsSpatialGridXAttention(CascadeROIHeads):
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
        contrastive_dim: int, # output embedding dimension (e.g. 128)
        contrastive_hidden_dim: int, # MLP hidden dimension (e.g. 1024)
        contrastive_weight: float, # weight applied to contrastive loss term
        temperature: float, # softmax temperature that scales the similarities in the embedding space; lower = sharper distribution
        contrastive_in_features: List[str], #features to use for gathering pos/neg samples for contrative learning
        contrastive_in_levels: List[float], #features to use for gathering pos/neg samples for contrative learning
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
        self.contrastive_weight = contrastive_weight
        self.contrastive_in_features = contrastive_in_features
        self.contrastive_in_levels = contrastive_in_levels

        # Learnable log-scale temperature param, initialized to log(1/T)
        # Keeping it learnable lets the model adapt the sharpness (peakness) of the similarity
        # distribution over training, as done in CLIP (https://arxiv.org/abs/2103.00020)
        # https://naokishibuya.github.io/blog/2023-08-13-clip-2103/#the-structure-of-clip 
        # also has a nice explanation of why CLIP uses a learnable temp and how it evolves during training
        
        #self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/temperature))
        self.register_buffer("logit_scale", torch.tensor(np.log(1/temperature), dtype=torch.float32))
        
        # ROI pooler for contrastive features (separate from box pooler)
        # Uses same FPN level scales as standard Cascade R-CNN box pooler:
        #   p2 -> scale 1/4, p3 -> 1/8, p4 -> 1/16, p5 -> 1/32
        # output_size=7 -> each ROI becomes a 7x7 spatial feature map before pooling
        # ROIAlignV2 (aligned=True) avoids 0.5-pixel quantization misalignment of
        # OG ROIPool, giving better-localized features
        #self.query_feature_pooler = ROIPooler(
        #    output_size=7,
        #    scales=tuple(k for k in [0.25, 0.125, 0.0625, 0.03125]),
        #    sampling_ratio=0,
        #    pooler_type="ROIAlignV2",
        #)

        self.query_feature_poolers = []
        for level in self.contrastive_in_levels:
            self.query_feature_poolers.append(ROIPooler(
                output_size=7,
                scales=[level],
                sampling_ratio=0,
                pooler_type="ROIAlignV2",
            ))
        
        # Because the key feature maps have strides that are not base 2, we 
        # must rescale the features by a scale_factor.  This ensures the boxes are rescaled correctly,
        # and ensures the same feature level assignment as the default base-2 strides
        #self.key_feature_pooler = ScaledROIPooler(
        #    output_size=7,
        #    scales=tuple(k for k in [1/13, 1/26, 1/52, 1/104]),
        #    sampling_ratio=0,
        #    pooler_type="ROIAlignV2",
        #    scale_factor = 13/4,
        #)
        # infer box head output dim
        self.in_channels = 256 # TODO: grab this dynmically from the box head output channels instead of hardcoding


        self.cross_attention_q = CrossAttentionHead(embed_dim,n_head,model_embed_dim,dropout)
        self.cross_attention_k = CrossAttentionHead(embed_dim,n_head,model_embed_dim,dropout)


        # Separate projection heads for query (Rubin) and key (Roman) encoders
        # Having separate weights allows each modality to learn its own projection
        # into the shared embedding space like CLIP-models do
        self.proj_q = ROIContrastiveHead(
            self.in_channels, contrastive_hidden_dim, contrastive_dim
        )
        self.proj_k = ROIContrastiveHead(
            self.in_channels, contrastive_hidden_dim, contrastive_dim
        )

    def forward_contrastive(self, features_q, features_k, instances, targets=None):
        """
        Compute the ROI-level symmetric InfoNCE (CLIP-style) contrastive loss
        using the SAME set of proposals applied to both encoders

        The key idea is that a Rubin ROI crop and the spatially corresponding
        Roman ROI crop are a positive pair, while all other cross-image or
        cross-object combinations within the batch are negatives

        Args:
            features_q (dict[str, Tensor]): FPN feature maps from the Rubin (query)
                backbone, keyed by feature level name (e.g. "p2", "p3", "p4", "p5").
            features_k (dict[str, Tensor]): FPN feature maps from the Roman (key)
                backbone, same keys as features_q after feature-name alignment in
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
            return {"roi_contrastive_loss": torch.tensor(0.0, device=list(features_q.values())[0].device)}


        num_instances_per_img = [len(i) for i in instances]
        N=np.sum(num_instances_per_img)

        multi_level_features_q = []
        multi_level_features_k = []
       
        if (self.query_feature_poolers is not None):
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]            
            for i,level_name in enumerate(self.contrastive_in_features):
                # Pool 7x7 from each FPN level regardless of box size
                f_q = self.query_feature_poolers[i]([features_q[level_name]], boxes)
                f_k = self.query_feature_poolers[i]([features_k[level_name]], boxes)
                multi_level_features_q.append(f_q) # (N, 256, 7, 7)
                multi_level_features_k.append(f_k) # (N, 256, 7, 7)

        # Stack them: (N, L, 256, 7, 7) where L is number of levels
        features_q = torch.stack(multi_level_features_q, dim=1)
        features_k = torch.stack(multi_level_features_k, dim=1)

        
        # Flatten spatial and levels to get a pool of patches per ROI
        # (N, L*49, 256)
        features_q = features_q.permute(0, 1, 3, 4, 2).reshape(N, -1, 256)
        features_k = features_k.permute(0, 1, 3, 4, 2).reshape(N, -1, 256)

        # Reshape 7x7 spatial grid into 49 separate vectors per ROI.  The contrastive loss is computed for each vector
        # Apply cross attention rather than an MLP
        z_q,atten_q = self.cross_attention_q(features_q.flatten(2))
        z_k,atten_k = self.cross_attention_k(features_k.flatten(2))


        z_q = self.proj_q(z_q)
        z_k = self.proj_k(z_k)
        #z_q = F.normalize(features_q.flatten(2),dim=-1)
        #z_k = F.normalize(features_k.flatten(2),dim=-1)

        n_patches = z_q.shape[1] # This is N boxes * L levels

        # Now use BMM: each ROI is matched across its own multi-scale patches
        # Negatives are only considered within a given ROI - this is to ensure no "hard negatives" due to overlap
        logits_qk = torch.bmm(z_q, z_k.transpose(1, 2)) * self.logit_scale.exp()  # (N, L*49, L*49)
        logits_kq = logits_qk.transpose(1, 2)


        # 1. Reshape logits from (N, L*49, L*49) -> (N * L*49, L*49)
        # Every row in this new matrix is a Rubin patch looking for its Roman match
        logits_qk_reshaped = logits_qk.reshape(-1, n_patches)
        logits_kq_reshaped = logits_kq.reshape(-1, n_patches)
         
        # 2. Create the targets
        # For each ROI, patch i should match index i. 
        # labels: [0, 1...195, 0, 1...195, ...]
        labels = torch.arange(n_patches, device=logits_qk.device).repeat(N)
        
        #print('Labels ', labels.size())
        #print('Logits ', logits.size())
        #print('Logits reshaped', logits_reshaped.size())

        # Symmetric InfoNCE: both directions (q->k and k->q) contribute equally
        # CE(logits, labels) treats each row's diagonal as correct cls
        loss_qk = F.cross_entropy(logits_qk_reshaped, labels)
        loss_kq = F.cross_entropy(logits_kq_reshaped, labels)
        # avg to get final loss
        loss = 0.5 * (loss_qk + loss_kq)

        if self.training:
            return {
                "roi_contrastive_loss": loss * self.contrastive_weight,
                # track effective temp (1/T = exp(logit_scale)) for monitoring
                # A rapidly growing value might mean contrastive collapse
                # Detached so it doesn't get used in the backward pass
                "logit_scale": self.logit_scale.exp().detach(),
            }

        else:
            return z_q, z_k

    def forward(self, images, features_q, proposals, targets=None):
        """ 
        Ex: self.roi_heads(images_q_rubin, features_q, proposals, gt_instances) calls this forward function
        During training:
            1. label_and_sample_proposals: assigns GT labels to each RPN proposal
               and sub-samples to batch_size_per_image (e.g. 512) with a fixed
               foreground/background ratio
            2. _forward_box: runs the 3-stage Cascade R-CNN box regression and
               classification heads, returning classification + regression losses
            3. _forward_mask: runs the mask head on foreground proposals,
               returning mask loss
            4. _forward_keypoint: runs the keypoint head on foreground proposals,
               returning keypoint loss
            NOTE: forward_contrastive isn't computed here; it's called separately 
            from GeneralizedRCNNMultimodal in meta_arc.py so it can access both Rubin and Roman features 

        During inference:
            1. _forward_box: predicts boxes and class scores, applies NMS
            2. forward_with_given_boxes: runs mask and keypoint heads on the
               surviving detections to produce final per-instance predictions

        Args:
            images: unused (deleted immediately); kept for API compatibility with
                the parent CascadeROIHeads signature
            features_q (dict[str, Tensor]): Rubin FPN features (query encoder)
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
            losses = self._forward_box(features_q, proposals, targets)
            losses.update(self._forward_mask(features_q, proposals))
            losses.update(self._forward_keypoint(features_q, proposals))
            # no CL loss here; it's called from GeneralizedRCNNMultimodal in meta_arc.py
            # so it can access both Rubin and Roman features
            return proposals, losses
        else:
            # box head: runs 3 cascade reg stages, merges preds, applies class-specific NMS
            # returning surviving instances
            pred_instances = self._forward_box(features_q, proposals)
            # mask + keypoint heads: run on surviving boxes after NMS
            pred_instances = self.forward_with_given_boxes(features_q, pred_instances)
            return pred_instances, {}

class CrossAttentionHead(nn.Module):
    """Cross-attention head with dropout.

    This module is a single head of a cross-attention layer. It takes a query and a key
    tensor, computes the attention weights, and returns the weighted sum of the values
    tensor. The attention weights are also returned.

    :param embed_dim: dimensionality of the input tensors
    :param n_head: number of heads
    :param model_embed_dim: dimensionality of the model tensors
    :param dropout: amount of dropout
    """

    embed_dim: int
    n_head: int
    model_embed_dim: int
    dropout: float

    def __init__(
        self,
        embed_dim: int,
        n_head: int,
        model_embed_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, embed_dim))
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_head,
            batch_first=True,
            kdim=model_embed_dim,
            vdim=model_embed_dim,
        )
        self.layernorm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.tensor):
        batch_size = x.shape[0]
        attentions = self.multihead_attn(
            query=self.query.repeat(batch_size, 1),
            key=x,
            value=x,
            average_attn_weights=False,
        )[0]
        x = self.layernorm(self.dropout(attentions))
        return x, attentions[1]
