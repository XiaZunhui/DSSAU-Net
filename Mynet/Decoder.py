import math

from collections import OrderedDict
from functools import partial
from typing import Optional, Union
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from .einops import rearrange
from .einops.layers.torch import Rearrange
from .fairscale.nn.checkpoint import checkpoint_wrapper

from .timm.models.layers import DropPath, to_2tuple, trunc_normal_

from .ops.dssa_nchw import nchwDSSA

from ._common import Attention, AttentionLePE, DWConv

count = 1
idx_r_list = []
X_list = []
a_r_list = []
idx_all = []
idx_all_tensor = torch.zeros(3, 4)
topk_before = []
topks = []


def get_pe_layer(emb_dim, pe_dim=None, name='none'):
    if name == 'none':
        return nn.Identity()
    else:
        raise ValueError(f'PE name {name} is not surpported!')


class Block(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 topks,
                 drop_path=0.,
                 layer_scale_init_value=-1,
                 num_heads=8,
                 n_win=7,
                 qk_dim=None,
                 qk_scale=None,
                 kv_per_win=4,
                 kv_downsample_ratio=4,
                 kv_downsample_kernel=None,
                 kv_downsample_mode='ada_avgpool',
                 topk=4,
                 param_attention="qkvo",
                 param_routing=False,
                 diff_routing=False,
                 soft_routing=False,
                 mlp_ratio=4,
                 mlp_dwconv=False,
                 side_dwconv=5,
                 before_attn_dwconv=3,
                 pre_norm=True,
                 auto_pad=False,
                 stage=0):
        super().__init__()
        qk_dim = qk_dim or dim
        self.idx_r = None
        self.topk = topk
        self.topks = topks
        self.depth = depth
        # BRA块部分
        # 主体架构：
        #           DWConv 3*3----->LN----->Bi-level Routing Attention----->LN------>MLP

        # 1.DWConv 3*3
        if before_attn_dwconv > 0:  # before_attn_dwconv=3
            self.pos_embed = nn.Conv2d(dim, dim, kernel_size=before_attn_dwconv, padding=1, groups=dim)  # DWConv
        else:
            self.pos_embed = lambda x: 0  # 匿名函数，接受参数X并返回0
        # 2.LN
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)  # important to avoid attention collapsing
        # 3.Bi-level Routing Attention
        if topk > 0:  # topk=4
            self.attn = nchwDSSA(dim=dim, num_heads=num_heads, n_win=n_win, qk_scale=qk_scale, topk=topk,
                                side_dwconv=side_dwconv,
                                auto_pad=auto_pad, attn_backend='torch')

        elif topk == -1:
            self.attn = Attention(dim=dim)
        elif topk == -2:
            self.attn = AttentionLePE(dim=dim, side_dwconv=side_dwconv)
        elif topk == 0:
            self.attn = nn.Sequential(Rearrange('n h w c -> n c h w'),  # compatiability
                                      nn.Conv2d(dim, dim, 1),  # pseudo qkv linear
                                      nn.Conv2d(dim, dim, 5, padding=2, groups=dim),  # pseudo attention
                                      nn.Conv2d(dim, dim, 1),  # pseudo out linear
                                      Rearrange('n c h w -> n h w c')
                                      )

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)

        self.mlp = nn.Sequential(
            nn.Linear(dim, int(mlp_ratio * dim)),
            # mlp_dwconv=False
            DWConv(int(mlp_ratio * dim)) if mlp_dwconv else nn.Identity(),
            nn.GELU(),
            nn.Linear(int(mlp_ratio * dim), dim))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if layer_scale_init_value > 0:
            self.use_layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        else:
            self.use_layer_scale = False

        self.pre_norm = pre_norm
        self.stage = stage

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x.permute(0, 2, 3, 1)
        if self.pre_norm:
            if self.use_layer_scale:
                x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))  # (N, H, W, C)
                x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))  # (N, H, W, C)
            else:
                x = x + self.drop_path(self.attn(self.norm1(x)))  # (N, H, W, C)
                x = x + self.drop_path(self.mlp(self.norm2(x)))  # (N, H, W, C)
        else:
            if self.use_layer_scale:
                x = self.norm1(x + self.drop_path(self.gamma1 * self.attn(x)))  # (N, H, W, C)
                x = self.norm2(x + self.drop_path(self.gamma2 * self.mlp(x)))  # (N, H, W, C)
            else:
                x = self.norm1(x + self.drop_path(self.attn(x)))  # (N, H, W, C)
                x = self.norm2(x + self.drop_path(self.mlp(x)))  # (N, H, W, C)

        # permute back
        x = x.permute(0, 3, 1, 2)
        return x


class Decoder(nn.Module):
    def __init__(self,
                 depth=[3, 4, 8, 3],
                 in_chans=3,
                 num_classes=100,
                 embed_dim=[64, 128, 320, 512],
                 head_dim=64,
                 qk_scale=None,
                 representation_size=None,
                 drop_path_rate=0.,
                 drop_rate=0.,
                 use_checkpoint_stages=[],
                 ########
                 n_win=7,
                 kv_downsample_mode='ada_avgpool',
                 kv_per_wins=[2, 2, -1, -1],
                 topks=[8, 8, -1, -1],
                 side_dwconv=5,
                 layer_scale_init_value=-1,
                 qk_dims=[None, None, None, None],
                 param_routing=False, diff_routing=False, soft_routing=False,
                 pre_norm=True,
                 pe=None,
                 pe_stages=[0],
                 before_attn_dwconv=3,
                 auto_pad=False,
                 # -----------------------
                 kv_downsample_kernels=[4, 2, 1, 1],
                 kv_downsample_ratios=[4, 2, 1, 1],  # -> kv_per_win = [2, 2, 2, 1]
                 mlp_ratios=[4, 4, 4, 4],
                 param_attention='qkvo',
                 mlp_dwconv=False):

        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        nheads = [dim // head_dim for dim in qk_dims]
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=embed_dim[i],
                        depth=depth,
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                        topk=topks[i],
                        topks=topks,
                        num_heads=nheads[i],
                        n_win=n_win,
                        qk_dim=qk_dims[i],
                        qk_scale=qk_scale,
                        kv_per_win=kv_per_wins[i],
                        kv_downsample_ratio=kv_downsample_ratios[i],
                        kv_downsample_kernel=kv_downsample_kernels[i],
                        kv_downsample_mode=kv_downsample_mode,
                        param_attention=param_attention,
                        param_routing=param_routing,
                        diff_routing=diff_routing,
                        soft_routing=soft_routing,
                        mlp_ratio=mlp_ratios[i],
                        mlp_dwconv=mlp_dwconv,
                        side_dwconv=side_dwconv,
                        before_attn_dwconv=before_attn_dwconv,
                        pre_norm=pre_norm,
                        auto_pad=auto_pad,
                        stage=i) for j in range(depth[i])],
            )
            if i in use_checkpoint_stages:
                stage = checkpoint_wrapper(stage)
            self.stages.append(stage)
            cur += depth[i]

        ##########################################################################
        self.norm = nn.BatchNorm2d(embed_dim[-1])
        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(embed_dim[-1],
                              num_classes) if num_classes > 0 else nn.Identity()  # Linear(in_features=512, out_features=20, bias=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)  # res = (56, 28, 14, 7), wins = (64, 16, 4, 1)
            x = self.stages[i](x)

        x = self.norm(x)
        x = self.pre_logits(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x.flatten(2).mean(-1)
        x = self.head(x)
        return x
