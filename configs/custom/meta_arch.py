import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers import move_device_like, ShapeSpec
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage

from detectron2.modeling import SwinTransformer
from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.layers import ShapeSpec

class DynamicSwinTransformer(SwinTransformer):
    """ Dynamically calculates strides based on the provided patch_size.
    Detectron2's SwinTransformer hardcodes the initial stride as 4 (i.e., 2^2),
    then doubles it for each patch merging stage. This works when patch_size=4
    but breaks when using different patch sizes for different pixel scales.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # we need to dynamically set the out feature strides based on the given patch size
        # otherwise, we will get issues with feature map sizes not matching in our FPN
        # lateral_features and top_down_features specifically
        self._out_feature_strides = {f"p{i}": (2 ** i) * self.patch_embed.patch_size[0] for i in self.out_indices}
        # for patch size 13, strides are p0: 13, p1:26, p2:52, p3:104

# adapted from detectron2/modeling/meta_arch/rcnn.py
class GeneralizedRCNNMultimodal(nn.Module):
    """
    Generalized R-CNN with Moco v2 for Roman-Rubin semi-supervised training, 
    where contrastive loss and downstream task loss
    are computed
    
    Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
        self,
        *,
        backbone_q: Backbone, # rubin - Query encoder providing us the features to be aligned
        backbone_k: Backbone, # roman - Key/Momentum encoder providing us high-quality features as a ref
        #mlp: nn.Module,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        beta: float = 1.0, # supervised loss weight
        rubin_pixel_mean: Tuple[float],
        rubin_pixel_std: Tuple[float],
        roman_pixel_mean: Tuple[float],
        roman_pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
        # loss weighting
        beta: float = 1.0 # supervised loss weight
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone_q = backbone_q # rubin
        self.backbone_k = backbone_k # roman
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads
        
        outshape_q = backbone_q.output_shape()
        outshape_k = backbone_k.output_shape()
        # Ok, so we also need to align FPN output features for MoCo. since diff patch sizes 
        # cause different stride values FPN auto-names these features differently 
        # (based on log2(stride)). But, we need matching names for MoCo. So, let's just manually
        # rename key encoder's features to match query encoder's naming scheme
        features_q_names = list(outshape_q.keys())
        features_k_names = list(outshape_k.keys())
        # align the feature names only if they don't match
        if features_q_names != features_k_names:
            print(f"Aligning feature names: {features_k_names} --> {features_q_names}")
            # scaling ratio
            patch_ratio = self.backbone_k.bottom_up.patch_embed.patch_size[0] / self.backbone_q.bottom_up.patch_embed.patch_size[0]
            print(f"Patch size ratio (key/query): {patch_ratio:.2f}")
            new_strides = {}
            new_channels = {}
            aligned_outshape_k = {}
            for q_name, k_name in zip(features_q_names, features_k_names):
                # new stride based on Query stride and Patch Ratio
                # e.g., 32 (Query) * 3.25 (Ratio) = 104 (Key)
                q_stride = outshape_q[q_name].stride
                k_stride_scaled = int(q_stride * patch_ratio)
                # channels from the existing Key backbone
                k_channels = outshape_k[k_name].channels
                # print(f"  Mapping {k_name} --> {q_name}: Stride {outshape_k[k_name].stride} -> {k_stride_scaled}")
                new_strides[q_name] = k_stride_scaled
                new_channels[q_name] = k_channels
                aligned_outshape_k[q_name] = ShapeSpec(channels=k_channels, stride=k_stride_scaled)
            # backbone's internal metadata
            self.backbone_k._out_features = list(new_strides.keys())
            self.backbone_k._out_feature_strides = new_strides
            self.backbone_k._out_feature_channels = new_channels
            outshape_k = aligned_outshape_k
            print(f"Final aligned features - Query: {list(outshape_q.keys())}, Key: {list(outshape_k.keys())}")
        self.beta = beta
        # ok now we should have everything aligned EXCEPT for the actual fpn and lateral 
        # module names but those don't actually get used in FPN forward pass or in our CLIP so
        # we keep them as is even though they're technically wrong names

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("rubin_pixel_mean", torch.tensor(rubin_pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("rubin_pixel_std", torch.tensor(rubin_pixel_std).view(-1, 1, 1), False)
        self.register_buffer("roman_pixel_mean", torch.tensor(roman_pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("roman_pixel_std", torch.tensor(roman_pixel_std).view(-1, 1, 1), False)
        assert (
            self.rubin_pixel_mean.shape == self.rubin_pixel_std.shape
        ), f"{self.rubin_pixel_mean} and {self.rubin_pixel_std} have different shapes!"
        assert (
            self.roman_pixel_mean.shape == self.roman_pixel_std.shape
        ), f"{self.roman_pixel_mean} and {self.roman_pixel_std} have different shapes!"

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "rubin_pixel_mean": cfg.MODEL.RUBIN_PIXEL_MEAN,
            "rubin_pixel_std": cfg.MODEL.RUBIN_PIXEL_STD,
            "roman_pixel_mean": cfg.MODEL.ROMAN_PIXEL_MEAN,
            "roman_pixel_std": cfg.MODEL.ROMAN_PIXEL_STD
        }

    @property
    def device(self):
        return self.rubin_pixel_mean.device

    def _move_to_current_device(self, x):
        return move_device_like(x, self.rubin_pixel_mean)

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)
        
        # must pass in size_divisibility for both backbones since it'll be different for each backbone
        # and we pass this in so that both Roman and Rubin can have teh same sized feature maps
        images_q_rubin = self.preprocess_image(batched_inputs, "image_rubin", self.rubin_pixel_mean, 
                                         self.rubin_pixel_std, self.backbone_q.size_divisibility, 
                                         self.backbone_q.padding_constraints)
        
        # for first run we use all instances (100% labeled data)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        
        # we have to check if we have Roman data (training with CLIP) or just Rubin (validation loss)
        has_roman_data = "image_roman" in batched_inputs[0]
        
        # run query backbone 
        features_q = self.backbone_q(images_q_rubin.tensor)        
        
        # features from the rubin encoder are sent to detection heads, if they have labels
        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images_q_rubin, features_q, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}
        # we need to grab the proposals from the roi heads so we can
        # do contrastive loss on the features from those proposals
        labeled_proposals, detector_losses = self.roi_heads(images_q_rubin, features_q, proposals, gt_instances)
        # if self.vis_period > 0:
        #     storage = get_event_storage()
        #     if storage.iter % self.vis_period == 0:
        #         self.visualize_training(batched_inputs, proposals)
        
        losses = {}
        # beta weight for supervised losses
        losses.update({k: self.beta * v for k, v in detector_losses.items()})
        losses.update({k: self.beta * v for k, v in proposal_losses.items()})
        # compute InfoNCE loss only if we have paired Roman data 
        # Validation mode: skip contrastive loss (no Roman data available)
        if has_roman_data:
            images_k_roman = self.preprocess_image(batched_inputs, "image_roman", self.roman_pixel_mean, 
                                    self.roman_pixel_std, self.backbone_k.size_divisibility, 
                                    self.backbone_k.padding_constraints)
            # grab the features computed through the key encoder - don't need grads
            features_k = self.backbone_k(images_k_roman.tensor)  # keys: NxC
            contrastive_loss = self.roi_heads.forward_contrastive(
                features_q, 
                features_k, 
                labeled_proposals,
                gt_instances
            )            
            losses.update(contrastive_loss)

        return losses

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training, "Inference should only be called in eval mode"
        # if "wcs" in batched_inputs[0]:
        #     image_wcs = [x["wcs"] for x in batched_inputs]
        # else:
        #     image_wcs = None
        # DictMapper returns "image", but we need to process it as Rubin data
        # so we just gotta rename the key to match what our preprocessing expects
        batched_inputs_renamed = []
        for batch_item in batched_inputs:
            renamed_item = {
                "image_rubin": batch_item["image_rubin"],  # treating validation rubin as query encoder input
                "height": batch_item["height"],
                "width": batch_item["width"],
                "image_id": batch_item.get("image_id", None),
                "instances": batch_item.get("instances", None),
            }
            batched_inputs_renamed.append(renamed_item)
        
        # preprocess Rubin images (LSST validation data)
        imgs_rubin = self.preprocess_image(
            batched_inputs_renamed, 
            "image_rubin", 
            self.rubin_pixel_mean,
            self.rubin_pixel_std, 
            self.backbone_q.size_divisibility,
            self.backbone_q.padding_constraints
        )
        # run through query encoder
        features = self.backbone_q(imgs_rubin.tensor)

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(imgs_rubin, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(imgs_rubin, features, proposals, None)#, image_wcs=image_wcs)
        else:
            # we already have the detected boxes and classes, so we skip proposal generation and box head inference
            # just run the box features through the mask/keypoint heads if they exist
            detected_instances = [x.to(self.device) for x in detected_instances]
            # from detectron2/modeling/roi_heads/roi_heads.py, StandardROIHeads.forward_with_given_boxes
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)
    
        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return self._postprocess(results, batched_inputs, imgs_rubin.image_sizes)
        return results

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]], img_key, mean, 
                         std, size_divisibility, padding_constraints):
        """
        Normalize, pad and batch the input images.
        """
        imgs = [self._move_to_current_device(x[img_key]) for x in batched_inputs]
        imgs = [(x - mean) / std for x in imgs]
        imgs = ImageList.from_tensors(
            imgs,
            size_divisibility,
            padding_constraints=padding_constraints,
        )
        return imgs
    
    @staticmethod
    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

class FeatureMapMLP(nn.Module):
    """
    MLP module that processes feature maps by extracting features from the deepest level
    and passing them through a MLP for dimensionality reduction
    """
    def __init__(self, in_features, square_pad, hidden_dim, dim):
        """
        Initialize the FeatureMapMLP module
        
        Args:
            in_features: Dictionary mapping feature names to their shape information (stride, channels)
            square_pad: Padding size for square feature maps
            hidden_dim: Hidden dimension size for the intermediate layers of the MLP
            dim: Output dimension size
        """
        super().__init__()

        # Store input feature information
        self.in_features = in_features
        self.f_keys = list(self.in_features.keys())  # Extract feature map names (keys)
        f_shapes = list(self.in_features.values())   # Extract feature map shape information
        
        # Extract stride information from each feature map shape
        strides = [s.stride for s in f_shapes]
        # Get the number of channels from the first feature map (assumed uniform across levels)
        channel = f_shapes[0].channels
        
        # Calculate output spatial dimensions for each feature map based on stride and padding
        shapes = [(channel, (square_pad+s-1)//s, (square_pad+s-1)//s) for s in strides]
        
        # Current approach uses only the deepest feature map for efficiency
        # Adaptive average pooling reduces spatial dimensions to 1x1 for any input size
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # MLP stack: reduces channel dimension to output dimension with ReLU activations
        self.fcls = nn.Sequential(
            nn.Linear(channel, hidden_dim),      # Project from channel dimension to hidden dimension
            nn.ReLU(),                            # Non-linear activation
            nn.Linear(hidden_dim, hidden_dim),   # Additional hidden layer for model capacity
            nn.ReLU(),                            # Non-linear activation
            nn.Linear(hidden_dim, dim)           # Project to final output dimension
        )

    def forward(self, features):
        """
        Forward pass through the module.
        Args:
            features: Dictionary of feature maps at different scales, keyed by feature level names
        Returns:
            Transformed feature vector of shape (batch_size, dim)
        """
        # Extract features from the deepest (last) feature map level for efficiency
        features = self.avgpool(features[self.f_keys[-1]])  # reduce spatial dims to 1x1: (B, C, H, W) -> (B, C, 1, 1)
        # Flatten to 2D tensor: (B, C) for MLP processing
        features = torch.flatten(features, 1)
        # pass through MLP to get output embeddings
        outputs = self.fcls(features)
        return outputs