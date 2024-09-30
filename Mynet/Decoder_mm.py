import sys, os

sys.path.append(os.path.dirname(sys.path[0]))
import torch
import torch.nn as nn

from .timm.models.layers import LayerNorm2d
from .Decoder import Decoder



class Decoder_mm(Decoder):
    def __init__(self, pretrained=None, **kwargs):
        super().__init__(**kwargs)

        del self.head  # classification head
        del self.norm  # head norm

        self.extra_norms = nn.ModuleList()
        for i in range(4):
            self.extra_norms.append(LayerNorm2d(self.embed_dim[i]))


    def forward(self, x: torch.Tensor):
        out = []
        for i in range(4):
            x = self.stages[i](x)
            out.append(self.extra_norms[i](x))
        return tuple(out)
