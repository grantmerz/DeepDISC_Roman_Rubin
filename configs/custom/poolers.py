# Copyright (c) Facebook, Inc. and its affiliates.
import math
from typing import List, Optional
import torch
from torch import nn
from torchvision.ops import RoIPool

from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from detectron2.layers import ROIAlign, ROIAlignRotated, cat, nonzero_tuple, shapes_to_tensor
from detectron2.structures import Boxes
from detectron2.utils.tracing import assert_fx_safe, is_fx_tracing

"""
To export ROIPooler to torchscript, in this file, variables that should be annotated with
`Union[List[Boxes], List[RotatedBoxes]]` are only annotated with `List[Boxes]`.

TODO: Correct these annotations when torchscript support `Union`.
https://github.com/pytorch/pytorch/issues/41412
"""

__all__ = ["ScaledROIPooler"]


def assign_boxes_to_levels(
    box_lists: List[Boxes],
    min_level: int,
    max_level: int,
    canonical_box_size: int,
    canonical_level: int,
):
    """
    Map each box in `box_lists` to a feature map level index and return the assignment
    vector.

    Args:
        box_lists (list[Boxes] | list[RotatedBoxes]): A list of N Boxes or N RotatedBoxes,
            where N is the number of images in the batch.
        min_level (int): Smallest feature map level index. The input is considered index 0,
            the output of stage 1 is index 1, and so.
        max_level (int): Largest feature map level index.
        canonical_box_size (int): A canonical box size in pixels (sqrt(box area)).
        canonical_level (int): The feature map level index on which a canonically-sized box
            should be placed.

    Returns:
        A tensor of length M, where M is the total number of boxes aggregated over all
            N batch images. The memory layout corresponds to the concatenation of boxes
            from all images. Each element is the feature map index, as an offset from
            `self.min_level`, for the corresponding box (so value i means the box is at
            `self.min_level + i`).
    """
    box_sizes = torch.sqrt(cat([boxes.area() for boxes in box_lists]))
    # Eqn.(1) in FPN paper
    level_assignments = torch.floor(
        canonical_level + torch.log2(box_sizes / canonical_box_size + 1e-8)
    )
    # clamp level to (min, max), in case the box size is too large or too small
    # for the available feature maps
    level_assignments = torch.clamp(level_assignments, min=min_level, max=max_level)
    return level_assignments.to(torch.int64) - min_level


# script the module to avoid hardcoded device type
@torch.jit.script_if_tracing
def _convert_boxes_to_pooler_format(boxes: torch.Tensor, sizes: torch.Tensor) -> torch.Tensor:
    sizes = sizes.to(device=boxes.device)
    indices = torch.repeat_interleave(
        torch.arange(len(sizes), dtype=boxes.dtype, device=boxes.device), sizes
    )
    return cat([indices[:, None], boxes], dim=1)


def convert_boxes_to_pooler_format(box_lists: List[Boxes]):
    """
    Convert all boxes in `box_lists` to the low-level format used by ROI pooling ops
    (see description under Returns).

    Args:
        box_lists (list[Boxes] | list[RotatedBoxes]):
            A list of N Boxes or N RotatedBoxes, where N is the number of images in the batch.

    Returns:
        When input is list[Boxes]:
            A tensor of shape (M, 5), where M is the total number of boxes aggregated over all
            N batch images.
            The 5 columns are (batch index, x0, y0, x1, y1), where batch index
            is the index in [0, N) identifying which batch image the box with corners at
            (x0, y0, x1, y1) comes from.
        When input is list[RotatedBoxes]:
            A tensor of shape (M, 6), where M is the total number of boxes aggregated over all
            N batch images.
            The 6 columns are (batch index, x_ctr, y_ctr, width, height, angle_degrees),
            where batch index is the index in [0, N) identifying which batch image the
            rotated box (x_ctr, y_ctr, width, height, angle_degrees) comes from.
    """
    boxes = torch.cat([x.tensor for x in box_lists], dim=0)
    # __len__ returns Tensor in tracing.
    sizes = shapes_to_tensor([x.__len__() for x in box_lists])
    return _convert_boxes_to_pooler_format(boxes, sizes)


@torch.jit.script_if_tracing
def _create_zeros(
    batch_target: Optional[torch.Tensor],
    channels: int,
    height: int,
    width: int,
    like_tensor: torch.Tensor,
    ) -> torch.Tensor:
    batches = batch_target.shape[0] if batch_target is not None else 0
    sizes = (batches, channels, height, width)
    return torch.zeros(sizes, dtype=like_tensor.dtype, device=like_tensor.device)


class ScaledROIPooler(nn.Module):
    """
    Region of interest feature map pooler that supports pooling from one or more
    feature maps.
    """

    def __init__(
        self,
        output_size,
        scales,
        sampling_ratio,
        pooler_type,
        canonical_box_size=224,
        canonical_level=4,
        scale_factor=1.0,
    ):
        """
        Args:
            output_size (int, tuple[int] or list[int]): output size of the pooled region,
                e.g., 14 x 14. If tuple or list is given, the length must be 2.
            scales (list[float]): The scale for each low-level pooling op relative to
                the input image. For a feature map with stride s relative to the input
                image, scale is defined as a 1 / s. The stride must be power of 2.
                When there are multiple scales, they must form a pyramid, i.e. they must be
                a monotically decreasing geometric sequence with a factor of 1/2.
            scale_factor (int): The factor that represents the offset between the given scales
                and the scales that would represent power of 2 strides, starting with 4.  
                I.e, for scales [1/13,1/26,1/52,1/104], the scale_factor is 13/4, meaning we would
                level assign based on the traditional strides of 4,8,16,32.
            sampling_ratio (int): The `sampling_ratio` parameter for the ROIAlign op.
            pooler_type (string): Name of the type of pooling operation that should be applied.
                For instance, "ROIPool" or "ROIAlignV2".
            canonical_box_size (int): A canonical box size in pixels (sqrt(box area)). The default
                is heuristically defined as 224 pixels in the FPN paper (based on ImageNet
                pre-training).
            canonical_level (int): The feature map level index from which a canonically-sized box
                should be placed. The default is defined as level 4 (stride=16) in the FPN paper,
                i.e., a box of size 224x224 will be placed on the feature with stride=16.
                The box placement for all boxes will be determined from their sizes w.r.t
                canonical_box_size. For example, a box whose area is 4x that of a canonical box
                should be used to pool features from feature level ``canonical_level+1``.

                Note that the actual input feature maps given to this module may not have
                sufficiently many levels for the input boxes. If the boxes are too large or too
                small for the input feature maps, the closest level will be used.
        """
        super().__init__()

        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        assert len(output_size) == 2
        assert isinstance(output_size[0], int) and isinstance(output_size[1], int)
        self.output_size = output_size
        self.scale_factor = scale_factor
        self.refactored_scales = [s*self.scale_factor for s in scales]


        #We want the poolers to be defined by the scales (without the factor applied)
        #This makes sure that the boxes are correctly rescaled to fit within a given feature map
        if pooler_type == "ROIAlign":
            self.level_poolers = nn.ModuleList(
                ROIAlign(
                    output_size, spatial_scale=scale, sampling_ratio=sampling_ratio, aligned=False
                )
                for scale in scales
            )
        elif pooler_type == "ROIAlignV2":
            self.level_poolers = nn.ModuleList(
                ROIAlign(
                    output_size, spatial_scale=scale, sampling_ratio=sampling_ratio, aligned=True
                )
                for scale in scales
            )
        elif pooler_type == "ROIPool":
            self.level_poolers = nn.ModuleList(
                RoIPool(output_size, spatial_scale=scale) for scale in scales
            )
        elif pooler_type == "ROIAlignRotated":
            self.level_poolers = nn.ModuleList(
                ROIAlignRotated(output_size, spatial_scale=scale, sampling_ratio=sampling_ratio)
                for scale in scales
            )
        else:
            raise ValueError("Unknown pooler type: {}".format(pooler_type))

        # Map scale (defined as 1 / stride) to its feature map level under the
        # assumption that stride is a power of 2.

        # Now, we use the refactored scales to define the level assignment 
        # E.g., a box of size S should be assigned to the same feature map as in 
        # the case with the traditional scales (1/4,1/8,1/16,1/32) 
        # With this setup, we do not need to alter the canonical box size/level  

        min_level = -(math.log2(self.refactored_scales[0]))
        max_level = -(math.log2(self.refactored_scales[-1]))
        assert math.isclose(min_level, int(min_level)) and math.isclose(
            max_level, int(max_level)
        ), "Featuremap stride is not power of 2!"
        self.min_level = int(min_level)
        self.max_level = int(max_level)
        assert (
            len(scales) == self.max_level - self.min_level + 1
        ), "[ROIPooler] Sizes of input featuremaps do not form a pyramid!"
        assert 0 < self.min_level and self.min_level <= self.max_level
        self.canonical_level = canonical_level
        assert canonical_box_size > 0
        self.canonical_box_size = canonical_box_size

    def forward(self, x: List[torch.Tensor], box_lists):
        """
        Args:
            x (list[Tensor]): A list of feature maps of NCHW shape, with scales matching those
                used to construct this module.
            box_lists (list[Boxes] | list[RotatedBoxes]):
                A list of N Boxes or N RotatedBoxes, where N is the number of images in the batch.
                The box coordinates are defined on the original image and
                will be scaled by the `scales` argument of :class:`ROIPooler`.

        Returns:
            Tensor:
                A tensor of shape (M, C, output_size, output_size) where M is the total number of
                boxes aggregated over all N batch images and C is the number of channels in `x`.
        """
        num_level_assignments = len(self.level_poolers)

        assert isinstance(x, list) and isinstance(
            box_lists, list
        ), "Arguments to pooler must be lists"
        assert (
            len(x) == num_level_assignments
        ), "unequal value, num_level_assignments={}, but x is list of {} Tensors".format(
            num_level_assignments, len(x)
        )

        assert len(box_lists) == x[0].size(
            0
        ), "unequal value, x[0] batch dim 0 is {}, but box_list has length {}".format(
            x[0].size(0), len(box_lists)
        )

        pooler_fmt_boxes = convert_boxes_to_pooler_format(box_lists)

        if num_level_assignments == 1:
            return self.level_poolers[0](x[0], pooler_fmt_boxes)

        level_assignments = assign_boxes_to_levels(
            box_lists, self.min_level, self.max_level, self.canonical_box_size, self.canonical_level
        )

        num_boxes = len(pooler_fmt_boxes)
        num_channels = x[0].shape[1]
        output_size = self.output_size[0]

        dtype, device = x[0].dtype, x[0].device
        output = torch.zeros(
            (num_boxes, num_channels, output_size, output_size), dtype=dtype, device=device
        )

        for level, (x_level, pooler) in enumerate(zip(x, self.level_poolers)):
            inds = nonzero_tuple(level_assignments == level)[0]
            pooler_fmt_boxes_level = pooler_fmt_boxes[inds]
            output[inds] = pooler(x_level, pooler_fmt_boxes_level)

        return output

class NonPowerOf2ROIPooler(ROIPooler):
    """ROIPooler that supports non-power-of-2 feature strides.

    Detectron2's ROIPooler asserts that all feature strides are powers of 2,
    which fails for patch_size=13 Swin (strides [13, 26, 52, 104]).
    This subclass bypasses ROIPooler.__init__ to skip that assertion,
    replicates the setup, and inherits the forward() pass unchanged.
    """

    def __init__(
        self,
        output_size,
        scales,
        sampling_ratio,
        pooler_type,
        canonical_box_size=224,
        canonical_level=4,
    ):
        # Jump straight to nn.Module.__init__ to skip ROIPooler's 
        # power-of-2 assertion in ROIPooler.__init__
        # basically skip to grandparent class's init 
        # since we don't want the parent class's init bc of the assertion
        nn.Module.__init__(self)

        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        assert len(output_size) == 2
        assert isinstance(output_size[0], int) and isinstance(output_size[1], int)
        self.output_size = output_size

        if pooler_type == "ROIAlign":
            self.level_poolers = nn.ModuleList(
                ROIAlign(
                    output_size, spatial_scale=scale, sampling_ratio=sampling_ratio, aligned=False
                )
                for scale in scales
            )
        elif pooler_type == "ROIAlignV2":
            self.level_poolers = nn.ModuleList(
                ROIAlign(
                    output_size, spatial_scale=scale, sampling_ratio=sampling_ratio, aligned=True
                )
                for scale in scales
            )
        elif pooler_type == "ROIPool":
            self.level_poolers = nn.ModuleList(
                RoIPool(output_size, spatial_scale=scale) for scale in scales
            )
        elif pooler_type == "ROIAlignRotated":
            self.level_poolers = nn.ModuleList(
                ROIAlignRotated(output_size, spatial_scale=scale, sampling_ratio=sampling_ratio)
                for scale in scales
            )
        else:
            raise ValueError("Unknown pooler type: {}".format(pooler_type))
        
        # Map scale (defined as 1 / stride) to its feature map level 
        # Round log2(scale) to the nearest integer so non-power-of-2 strides get
        # contiguous level indices.  For power-of-2 strides this is identical to
        # the original int() cast.  Example for scales = (1/13, 1/26, 1/52, 1/104):
        #   round(-log2(1/13)) = round(3.70) = 4 --> min_level=4
        #   round(-log2(1/104)) = round(6.70) = 7 --> max_level=7
        min_level = -(math.log2(scales[0]))
        max_level = -(math.log2(scales[-1]))
        # Round instead of casting so non-power-of-2 strides get idxs
        # that map to their nearest power-of-2 equivalent.
        # Example: stride 52 ≈ 2^5.7 --> level 6 (like stride 64), 
        # not level 5 (stride 32). For power-of-2 strides 
        # this is identical to int() cast. This ensures boxes land 
        # at the correct pyramid level. 
        # Using int() would systematically bias all level assignments downward, 
        # breaking the geometric relationship between box size and feature resolution
        
        # What if we used int()?
        # self.min_level = int(-math.log2(1/13))   = 3
        # self.max_level = int(-math.log2(1/104))  = 6
        # Then with canonical_level=6:
        # A 224px box: level 6 --> index 6 - min_level = 6 - 3 = 3 --> pools from stride 104 (the coarsest map!) 
        # This is backwards: 224px boxes should go to a middle-resolution feature, not the coarsest one

        # With round():
        # self.min_level = round(-math.log2(1/13))   = 4
        # self.max_level = round(-math.log2(1/104))  = 7
        # Now with canonical_level=6:
        # A 224px box: level 6 --> index 6 - min_level = 6 - 4 = 2 --> pools from stride 52 (middle of pyramid) 
        self.min_level = round(min_level)
        self.max_level = round(max_level)
        assert (
            len(scales) == self.max_level - self.min_level + 1
        ), "[NonPowerOf2ROIPooler] Scales do not form a contiguous pyramid!"
        assert 0 <= self.min_level <= self.max_level
        self.canonical_level = canonical_level
        assert canonical_box_size > 0
        self.canonical_box_size = canonical_box_size

class CustomLastLevelMaxPool(LastLevelMaxPool):
    """LastLevelMaxPool with a configurable in_feature
    The default LastLevelMaxPool hardcodes in_feature='p5'. With patch_size=13,
    FPN outputs ["p3","p4","p5","p6"] where p6 is the coarsest (stride 104).
    Taking p5 (stride 52) would produce a 5x5 map that duplicates p6; taking p6
    produces a genuine p7 at stride 208.
    """
    def __init__(self, in_feature="p6"):
        super().__init__()
        self.in_feature = in_feature