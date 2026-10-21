from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from detectron2.modeling import SwinTransformer

from ..common.coco_loader_lsj import dataloader
from .cascade_mask_rcnn_mvitv2_b_in21k_100ep import model


# class SwinTransformer(Backbone):
#     """Swin Transformer backbone.
#         A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted
#             Windows`  - https://arxiv.org/pdf/2103.14030
#     Args:
#         pretrain_img_size (int): Input image size for training the pretrained model,
#             used in absolute postion embedding. Default 224.
#         patch_size (int | tuple(int)): Patch size. Default: 4.
#         in_chans (int): Number of input image channels. Default: 3.
#         embed_dim (int): Number of linear projection output channels. Default: 96.
#         depths (tuple[int]): Depths of each Swin Transformer stage.
#         num_heads (tuple[int]): Number of attention head of each stage.
#         window_size (int): Window size. Default: 7.
#         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
#         qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
#         qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
#         drop_rate (float): Dropout rate.
#         attn_drop_rate (float): Attention dropout rate. Default: 0.
#         drop_path_rate (float): Stochastic depth rate. Default: 0.2.
#         norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
#         ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
#         patch_norm (bool): If True, add normalization after patch embedding. Default: True.
#         out_indices (Sequence[int]): Output from which stages.
#         frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
#             -1 means not freezing any parameters.
#         use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
#     """

#     def __init__(
#         self,
#         pretrain_img_size=224,
#         patch_size=4,
#         in_chans=3,
#         embed_dim=96,
#         depths=(2, 2, 6, 2),
#         num_heads=(3, 6, 12, 24),
#         window_size=7,
#         mlp_ratio=4.0,
#         qkv_bias=True,
#         qk_scale=None,
#         drop_rate=0.0,
#         attn_drop_rate=0.0,
#         drop_path_rate=0.2,
#         norm_layer=nn.LayerNorm,
#         ape=False,
#         patch_norm=True,
#         out_indices=(0, 1, 2, 3),
#         frozen_stages=-1,
#         use_checkpoint=False,
#     ):
#         super().__init__()

#         self.pretrain_img_size = pretrain_img_size
#         self.num_layers = len(depths)
#         self.embed_dim = embed_dim
#         self.ape = ape
#         self.patch_norm = patch_norm
#         self.out_indices = out_indices
#         self.frozen_stages = frozen_stages

#         # split image into non-overlapping patches
#         self.patch_embed = PatchEmbed(
#             patch_size=patch_size,
#             in_chans=in_chans,
#             embed_dim=embed_dim,
#             norm_layer=norm_layer if self.patch_norm else None,
#         )

#         # absolute position embedding
#         if self.ape:
#             pretrain_img_size = _to_2tuple(pretrain_img_size)
#             patch_size = _to_2tuple(patch_size)
#             patches_resolution = [
#                 pretrain_img_size[0] // patch_size[0],
#                 pretrain_img_size[1] // patch_size[1],
#             ]

#             self.absolute_pos_embed = nn.Parameter(
#                 torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1])
#             )
#             nn.init.trunc_normal_(self.absolute_pos_embed, std=0.02)

#         self.pos_drop = nn.Dropout(p=drop_rate)

#         # stochastic depth
#         dpr = [
#             x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
#         ]  # stochastic depth decay rule

#         # build layers
#         self.layers = nn.ModuleList()
#         for i_layer in range(self.num_layers):
#             layer = BasicLayer(
#                 dim=int(embed_dim * 2**i_layer),
#                 depth=depths[i_layer],
#                 num_heads=num_heads[i_layer],
#                 window_size=window_size,
#                 mlp_ratio=mlp_ratio,
#                 qkv_bias=qkv_bias,
#                 qk_scale=qk_scale,
#                 drop=drop_rate,
#                 attn_drop=attn_drop_rate,
#                 drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
#                 norm_layer=norm_layer,
#                 downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
#                 use_checkpoint=use_checkpoint,
#             )
#             self.layers.append(layer)

#         num_features = [int(embed_dim * 2**i) for i in range(self.num_layers)]
#         self.num_features = num_features

#         # add a norm layer for each output
#         for i_layer in out_indices:
#             layer = norm_layer(num_features[i_layer])
#             layer_name = f"norm{i_layer}"
#             self.add_module(layer_name, layer)

#         self._freeze_stages()
#         self._out_features = ["p{}".format(i) for i in self.out_indices]
#         self._out_feature_channels = {
#             "p{}".format(i): self.embed_dim * 2**i for i in self.out_indices
#         }
#         self._out_feature_strides = {"p{}".format(i): 2 ** (i + 2) for i in self.out_indices}
#         self._size_devisibility = 32

#         self.apply(self._init_weights)

#     def _freeze_stages(self):
#         if self.frozen_stages >= 0:
#             self.patch_embed.eval()
#             for param in self.patch_embed.parameters():
#                 param.requires_grad = False

#         if self.frozen_stages >= 1 and self.ape:
#             self.absolute_pos_embed.requires_grad = False

#         if self.frozen_stages >= 2:
#             self.pos_drop.eval()
#             for i in range(0, self.frozen_stages - 1):
#                 m = self.layers[i]
#                 m.eval()
#                 for param in m.parameters():
#                     param.requires_grad = False

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             nn.init.trunc_normal_(m.weight, std=0.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)

#     @property
#     def size_divisibility(self):
#         return self._size_divisibility

#     def forward(self, x):
#         """Forward function."""
#         x = self.patch_embed(x)

#         Wh, Ww = x.size(2), x.size(3)
#         if self.ape:
#             # interpolate the position embedding to the corresponding size
#             absolute_pos_embed = F.interpolate(
#                 self.absolute_pos_embed, size=(Wh, Ww), mode="bicubic"
#             )
#             x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
#         else:
#             x = x.flatten(2).transpose(1, 2)
#         x = self.pos_drop(x)

#         outs = {}
#         for i in range(self.num_layers):
#             layer = self.layers[i]
#             x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

#             if i in self.out_indices:
#                 norm_layer = getattr(self, f"norm{i}")
#                 x_out = norm_layer(x_out)

#                 out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
#                 outs["p{}".format(i)] = out

#         return outs

model.backbone.bottom_up = L(SwinTransformer)(
    depths=[2, 2, 18, 2],
    drop_path_rate=0.4,
    embed_dim=128,
    num_heads=[4, 8, 16, 32],
)
model.backbone.in_features = ("p0", "p1", "p2", "p3")
model.backbone.square_pad = 1024

# Initialization and trainer settings
train = model_zoo.get_config("common/train.py").train
train.amp.enabled = True
train.ddp.fp16_compression = True
train.init_checkpoint = "detectron2://ImageNetPretrained/swin/swin_base_patch4_window7_224_22k.pth"

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

# Rescale schedule
train.max_iter = train.max_iter // 2  # 100ep -> 50ep
lr_multiplier.scheduler.milestones = [
    milestone // 2 for milestone in lr_multiplier.scheduler.milestones
]
lr_multiplier.scheduler.num_updates = train.max_iter


optimizer = model_zoo.get_config("common/optim.py").AdamW
optimizer.lr = 4e-5
optimizer.weight_decay = 0.05
optimizer.params.overrides = {"relative_position_bias_table": {"weight_decay": 0.0}}
