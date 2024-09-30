# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
import torch.nn as nn
from .Encoder_mm import Encoder_mm
from .Decoder_mm import Decoder_mm
import torch.nn.functional as F


class PPM(nn.ModuleList):
    def __init__(self, pool_sizes, in_channels, out_channels):
        super(PPM, self).__init__()
        self.pool_sizes = pool_sizes
        self.in_channels = in_channels
        self.out_channels = out_channels
        for pool_size in pool_sizes:
            self.append(
                nn.Sequential(
                    nn.AdaptiveMaxPool2d(pool_size),
                    nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1),
                )
            )

    def forward(self, x):
        out_puts = []
        for ppm in self:
            ppm_out = nn.functional.interpolate(ppm(x), size=(x.size(2), x.size(3)), mode='bilinear',
                                                align_corners=True)
            out_puts.append(ppm_out)
        return out_puts


class PPMHEAD(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes=[1, 2, 3, 6], num_classes=2):
        super(PPMHEAD, self).__init__()
        self.pool_sizes = pool_sizes
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.psp_modules = PPM(self.pool_sizes, self.in_channels, self.out_channels)
        self.final = nn.Sequential(
            nn.Conv2d(self.in_channels + len(self.pool_sizes) * self.out_channels, self.out_channels, kernel_size=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.psp_modules(x)
        out.append(x)
        out = torch.cat(out, 1)
        out = self.final(out)
        return out


class FPNHEAD(nn.Module):
    def __init__(self, Decoder, channels=768, out_channels=64):
        super(FPNHEAD, self).__init__()
        self.PPMHead = PPMHEAD(in_channels=channels, out_channels=out_channels)
        self.Decoder = Decoder
        self.Conv_fuse1 = nn.Sequential(
            nn.Conv2d(channels // 2, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.Conv_fuse1_ = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.Conv_fuse2 = nn.Sequential(
            nn.Conv2d(channels // 4, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.Conv_fuse2_ = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.Conv_fuse3 = nn.Sequential(
            nn.Conv2d(channels // 8, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.Conv_fuse3_ = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.fuse_all = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.conv_x1 = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, input_fpn):
        x1 = self.PPMHead(input_fpn[-1])
        x1 = self.Decoder.stages[-1](x1)

        x = nn.functional.interpolate(x1, size=(x1.size(2) * 2, x1.size(3) * 2), mode='bilinear', align_corners=True)
        x = self.conv_x1(x) + self.Conv_fuse1(input_fpn[-2])
        x = self.Decoder.stages[-2](x)
        x2 = self.Conv_fuse1_(x)

        x = nn.functional.interpolate(x2, size=(x2.size(2) * 2, x2.size(3) * 2), mode='bilinear', align_corners=True)
        x = x + self.Conv_fuse2(input_fpn[-3])
        x = self.Decoder.stages[-3](x)
        x3 = self.Conv_fuse2_(x)

        x = nn.functional.interpolate(x3, size=(x3.size(2) * 2, x3.size(3) * 2), mode='bilinear', align_corners=True)
        x = x + self.Conv_fuse3(input_fpn[-4])
        x = self.Decoder.stages[-4](x)
        x4 = self.Conv_fuse3_(x)

        x1 = F.interpolate(x1, x4.size()[-2:], mode='bilinear', align_corners=True)
        x2 = F.interpolate(x2, x4.size()[-2:], mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, x4.size()[-2:], mode='bilinear', align_corners=True)


        x = self.fuse_all(torch.cat([x1, x2, x3, x4], 1))

        return x


class DSSAU_Net(nn.Module):
    def __init__(self, seg_num_classes):
        super(DSSAU_Net, self).__init__()
        self.Encoder = Encoder_mm(
            depth=[2, 2, 8, 2],
            embed_dim=[96, 192, 384, 768],
            mlp_ratios=[3, 3, 3, 3],
            n_win=8,  # training resolution is 512
            kv_downsample_mode='identity',
            kv_per_wins=[-1, -1, -1, -1],
            topks=[1, 4, 16, -2],
            side_dwconv=5,
            before_attn_dwconv=3,
            layer_scale_init_value=-1,
            qk_dims=[96, 192, 384, 768],
            head_dim=32,
            param_routing=False, diff_routing=False, soft_routing=False,
            pre_norm=True,
            pe=None,
            # --------------------------
            # it seems whole_eval takes raw-resolution input
            # use auto_pad to allow any-size input
            auto_pad=True,
            # use grad ckpt to save memory on old gpus
            use_checkpoint_stages=[],
            # drop_path_rate=0.3,
            drop_path_rate=0.2)  # it seems that, upernet requires a larger dpr
        self.Decoder = Decoder_mm(
            depth=[2, 2, 8, 2],
            embed_dim=[64, 64, 64, 64],
            mlp_ratios=[3, 3, 3, 3],
            n_win=8,  # training resolution is 512
            kv_downsample_mode='identity',
            kv_per_wins=[-1, -1, -1, -1],
            topks=[1, 4, 16, -2],
            side_dwconv=5,
            before_attn_dwconv=3,
            layer_scale_init_value=-1,
            qk_dims=[64, 64, 64, 64],
            head_dim=32,
            param_routing=False, diff_routing=False, soft_routing=False,
            pre_norm=True,
            pe=None,
            # --------------------------
            # it seems whole_eval takes raw-resolution input
            # use auto_pad to allow any-size input
            auto_pad=True,
            # use grad ckpt to save memory on old gpus
            use_checkpoint_stages=[],
            # drop_path_rate=0.3,
            drop_path_rate=0.2)
        self.decoder = FPNHEAD(self.Decoder)
        self.num_classes = seg_num_classes
        self.cls_seg = nn.Sequential(
            nn.Conv2d(64, self.num_classes, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.Encoder(x)
        x = self.decoder(x)
        x = nn.functional.interpolate(x, size=(x.size(2) * 4, x.size(3) * 4), mode='bilinear', align_corners=True)
        x = self.cls_seg(x)
        return x

    def load_from(self):
        pretrained_path = 'pretrain.pth'
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            model_dict = self.Encoder.state_dict()
            full_dict = copy.deepcopy(pretrained_dict['model'])
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k, full_dict[k].shape,
                                                                                  model_dict[k].shape))
                        del full_dict[k]
                if k not in model_dict:
                    del full_dict[k]
            msg = self.Encoder.load_state_dict(full_dict, strict=False)
            print(msg)
        else:
            print("none pretrain")



