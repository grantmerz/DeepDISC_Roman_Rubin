import logging
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
class GeneralizedRCNNMoco(nn.Module):
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
        rubin_pixel_mean: Tuple[float],
        rubin_pixel_std: Tuple[float],
        roman_pixel_mean: Tuple[float],
        roman_pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
        # MoCo params
        K: int =8192, 
        m: float= 0.999, 
        T: float= 0.07,
        hidden_dim: int = 2048,
        dim: int = 128,
        # loss weighting
        moco_alpha: float = 1.0, # contrastive loss weight
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
        # ok now we should have everything aligned EXCEPT for the actual fpn and lateral 
        # module names but those don't actually get used in FPN forward pass or in our MoCo so
        # we keep them as is even though they're technically wrong names
        self.moco = MMMoCo(self.backbone_q, self.backbone_k, 
                           outshape_q, outshape_k, hidden_dim, dim=dim, K=K, m=m, T=T)
        self.infoNCE_loss = nn.CrossEntropyLoss()#.cuda(args.gpu)
        
        self.moco_alpha = moco_alpha
        self.beta = beta

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
        
        # we have to check if we have Roman data (training with MoCo) or just Rubin (validation loss)
        has_roman_data = "image_roman" in batched_inputs[0]
        
        # run query backbone 
        features_q = self.backbone_q(images_q_rubin.tensor)
        
        # calculate MoCo loss only if we have paired Roman data
        if has_roman_data:
            images_k_roman = self.preprocess_image(batched_inputs, "image_roman", self.roman_pixel_mean, 
                                    self.roman_pixel_std, self.backbone_k.size_divisibility, 
                                    self.backbone_k.padding_constraints)
            # grab the features computed through the query encoder - don't need grads
            # run moco with roman imgs as keys 
            logits, labels = self.moco(features_q, images_k_roman.tensor)
            moco_loss = self.moco_alpha * self.infoNCE_loss(logits, labels)
        else:
            # Validation mode: skip MoCo loss (no Roman data available)
            moco_loss = None
        
        # features from the rubin encoder are sent to detection heads, if they have labels
        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images_q_rubin, features_q, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images_q_rubin, features_q, proposals, gt_instances)#, image_wcs)
        # if self.vis_period > 0:
        #     storage = get_event_storage()
        #     if storage.iter % self.vis_period == 0:
        #         self.visualize_training(batched_inputs, proposals)
        
        losses = {}
        # add MoCo loss only if computed (training w/ Roman data)
        if moco_loss is not None:
            losses.update({"infoNCE_loss": moco_loss})
        # beta weight for supervised losses
        losses.update({k: self.beta * v for k, v in detector_losses.items()})
        losses.update({k: self.beta * v for k, v in proposal_losses.items()})
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
                "image_rubin": batch_item["image"],  # treating validation rubin as query encoder input
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
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)
            # results = self.roi_heads._forward_redshift(features, results)#, image_wcs=image_wcs)
    
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
    def __init__(self,in_features,square_pad,hidden_dim,dim):
        super().__init__()

        self.in_features = in_features
        self.f_keys = list(self.in_features.keys())
        f_shapes = list(self.in_features.values())
        strides = [s.stride for s in f_shapes]
        channel = f_shapes[0].channels
        #assume kernel sizes of 4 for each 
        shapes = [(channel,(square_pad+s-1)//s,(square_pad+s-1)//s) for s in strides]
        #use all feature maps
        #self.fcls = {}
        #for i,f in enumerate(in_features.keys()):
        #    self.fcls[f] = nn.Linear(np.prod(shapes[i]),hidden_dim)
        #self.fcl_final = nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,dim)),

        #take the features from the deepest feature map
        self.fcls = nn.Sequential(
            nn.Linear(np.prod(shapes[-1]),hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,dim)
        )

    def forward(self, features):
        # take the features from all levels of the feature map and run them
        # each through a fully connected layer.
        
        #f_outs = []

        #for f in self.in_features:
        #    f_out=self.fcls[f](features[f].flatten())
        #    f_outs.append(f_out)

        #now add them together and run them through an MLP
        #f_outs = torch.stack(f_outs, dim=0).sum(dim=0)   
        #outputs = self.fcl_final(f_outs)

        #just use the final feature map
        features = nn.Flatten()(features[self.f_keys[-1]])
        outputs = self.fcls(features)
        return outputs

class MMMoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722.  
    Uses multimodal images, so the query and key encoders will be slightly different
    """
    def __init__(self, encoder_q, encoder_k, outshape_q, outshape_k, hidden_dim, dim, K=8192, m=0.999, T=0.07):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 8192)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        self.backbone_q, self.backbone_k, 
                           outshape, sp, hidden_dim, dim=dim, K=K, m=m, T=T
        """
        super(MMMoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # This assumes that the patch embedding has been fixed such that the feature maps of
        # both encoders are the same size.

        # Check that sizes of final feature map will match
        sp_q = encoder_q._square_pad
        sp_k = encoder_k._square_pad
        ps_q = encoder_q.bottom_up.patch_embed.patch_size[0]
        ps_k = encoder_k.bottom_up.patch_embed.patch_size[0]
        pad_q = (ps_q - sp_q % ps_q) if sp_q % ps_q != 0 else 0
        pad_k = (ps_k - sp_k % ps_k) if sp_k % ps_k != 0 else 0
        size1 = (sp_q + pad_q -(ps_q - 1) -1)/ ps_q + 1
        size2 = (sp_k + pad_k -(ps_k - 1) -1)/ ps_k + 1
        assert size1==size2, "Sizes of the feature maps should match.  Check the patch embedding or square padding of the encoders"

        self.mlp_q = FeatureMapMLP(outshape_q, sp_q, hidden_dim, dim)
        self.mlp_k = FeatureMapMLP(outshape_k, sp_k, hidden_dim, dim)
        
        # add the mlp to the encoders to get the latent dim embeddings
        self.encoder_q = nn.Sequential(encoder_q, self.mlp_q)
        self.encoder_k = nn.Sequential(encoder_k, self.mlp_k)
        for (name_q, param_q), (name_k, param_k) in zip(self.encoder_q.named_parameters(), self.encoder_k.named_parameters()):
            # we need to exclude the patch embedding layer of the roman_key encoder from this
            # because the images sent into the roman_key encoder will have different size/channels 
            # (3 channels vs 6, 520x520 vs 160x160) means it can't copy from Rubin encoder
            if "bottom_up.patch_embed" in name_k:
                param_k.requires_grad = True # patch embedding has grad graph
                continue
            # print(f"Copying param {name_q} to {name_k} for momentum encoder initialization")
            param_k.data.copy_(param_q.data)  # initialize (not really necessary if we use the same checkpoints)
            param_k.requires_grad = False  # not updated by gradient since it's a momentum encoder

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        param_to_check = self.encoder_k[0].bottom_up.patch_embed.proj.weight
        # Register a buffer to store its value from the previous step
        # We use .data.clone() to get the values without grad history
        self.register_buffer("old_patch_embed_weight", param_to_check.data.clone())

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for (name_q, param_q), (name_k, param_k) in zip(self.encoder_q.named_parameters(), self.encoder_k.named_parameters()):
            # we need to exclude the patch embedding dimensions here as well
            if "bottom_up.patch_embed" in name_k:
                continue
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity
        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]
        num_gpus = batch_size_all // batch_size_this
        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()
        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)
        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)
        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]
        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, features_q, im_k):
        """
        Take in query features instead of images.  Keep key images as input
        so that we can make use of the shuffle batch norm
        Input:
            features_q: a batch of query features
            im_k: a batch of key images
        Output:
            logits, targets
        """
        # Project the rubin_query features into the embedding space 
        q = self.mlp_q(features_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)
        # compute key features with no gradient
        self._momentum_update_key_encoder()  # update the key encoder
        
        # We have to remove the key shuffling because it breaks the gradient
        # im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

        k = self.encoder_k(im_k)  # keys: NxC
        k = nn.functional.normalize(k, dim=1)
        # undo shuffle
        # with torch.no_grad():
        # k = self._batch_unshuffle_ddp(k, idx_unshuffle)
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1) # here k has grads only thru the patch embedding
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        # moco_loss has grad path: loss -> logits -> l_pos -> k -> patch_embed
        # so in backward pass:
        # grads flow back thru l_pos to k, then to encoder_k where only the patch embedding receives grads
        # updating only the patch embedding of encoder_k with regular optimizer (AdamW) steps
        # dequeue and enqueue
        self._dequeue_and_enqueue(k.detach())
        return logits, labels

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)

    return output