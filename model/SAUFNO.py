"""SAU-FNO = U-FNO + axial self-attention, inserted after the last U-Fourier layer and
before fc1.

SAUSimpleBlock3d subclasses U-FNO's SimpleBlock3d, so every Fourier/U-Net layer
parameter is inherited unchanged -- the **only** difference from U-FNO is the added
attention block, which is what makes the comparison fair. Class bodies are copied
verbatim from sau_fno.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import operator
from functools import reduce

from layers.attention import AxialAttentionBlock
from model.UFNO import SimpleBlock3d


class SAUSimpleBlock3d(SimpleBlock3d):
    """U-FNO SimpleBlock3d + one AxialAttentionBlock inserted after the last
    U-Fourier layer (before the fc projection). All Fourier/U-Net layers are
    inherited from SimpleBlock3d unchanged."""

    def __init__(self, modes1, modes2, modes3, width, in_channels=12, attn_heads=4):
        super().__init__(modes1, modes2, modes3, width, in_channels=in_channels)
        self.axial_block = AxialAttentionBlock(self.width, heads=attn_heads)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y, size_z = x.shape[1], x.shape[2], x.shape[3]

        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)

        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = F.relu(x1 + x2)

        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = F.relu(x1 + x2)

        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = F.relu(x1 + x2)

        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x3 = self.unet3(x)
        x = F.relu(x1 + x2 + x3)

        x1 = self.conv4(x)
        x2 = self.w4(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x3 = self.unet4(x)
        x = F.relu(x1 + x2 + x3)

        x1 = self.conv5(x)
        x2 = self.w5(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x3 = self.unet5(x)
        x = F.relu(x1 + x2 + x3)

        # ---- SAU-FNO: axial self-attention after the last U-Fourier layer ----
        x = self.axial_block(x)

        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class SAUNet3d(nn.Module):
    """Same pad-to-multiple-of-8 + crop wrapper as ufno.Net3d, but using
    SAUSimpleBlock3d (with the axial attention block)."""

    def __init__(self, modes1, modes2, modes3, width, in_channels=12, attn_heads=4):
        super(SAUNet3d, self).__init__()
        self.conv1 = SAUSimpleBlock3d(modes1, modes2, modes3, width,
                                      in_channels=in_channels, attn_heads=attn_heads)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y, size_z = x.shape[1], x.shape[2], x.shape[3]
        px = (8 - size_x % 8) % 8
        py = (8 - size_y % 8) % 8
        pz = (8 - size_z % 8) % 8
        x = F.pad(x, (0, 0, 0, pz, 0, py), "replicate")
        if px:
            x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, px), 'constant', 0)
        x = self.conv1(x)
        x = x.view(batchsize, size_x + px, size_y + py, size_z + pz, 1)
        x = x[:, :size_x, :size_y, :size_z, :]
        return x.squeeze(-1)

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        return c
