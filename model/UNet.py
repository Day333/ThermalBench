"""2D U-Net baseline. The channel widths (58, 116, 232, 348) are the enlarged version,
chosen so the parameter count matches U-FNO (about 5.06M); the original (4, 8, 16, 24)
had only tens of thousands of parameters, which made the comparison unfair. Class body
copied verbatim from unet.py.
"""
import torch
import torch.nn as nn

import operator
from functools import reduce

from layers.unet_blocks import DoubleConv

torch.manual_seed(0)


class UNet(nn.Module):
    """Lightweight 2D U-Net for power-map -> temperature-map regression.

    Channels [C1, C2, C3, C4] = [4, 8, 16, 24], three MaxPool down-samples and
    three nearest-upsample + skip-concat levels, a 1x1 output head. No 3D conv.

    The skeleton follows the 2-tier spec exactly; only the first/last channel
    counts adapt to the data. For the case_all single-tier data the input is the
    3 channel-last feature map (power, grid_x, grid_y) read as P=3 channels and
    the head emits 1 temperature channel -- same I/O convention as FNO / U-FNO,
    so results are directly comparable.

    Input:  (B, in_channels, 64, 64)
    Output: (B, out_channels, 64, 64)
    """
    def __init__(self, in_channels=3, out_channels=1, channels=(58, 116, 232, 348)):
        super(UNet, self).__init__()
        c1, c2, c3, c4 = channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        # encoder
        self.enc1 = DoubleConv(in_channels, c1)   # 64x64  -> skip1
        self.pool1 = nn.MaxPool2d(2)              # -> 32x32
        self.enc2 = DoubleConv(c1, c2)            # 32x32  -> skip2
        self.pool2 = nn.MaxPool2d(2)              # -> 16x16
        self.enc3 = DoubleConv(c2, c3)            # 16x16  -> skip3
        self.pool3 = nn.MaxPool2d(2)              # -> 8x8

        # bottleneck
        self.bottleneck = DoubleConv(c3, c4)      # 8x8

        # decoder
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')   # 8  -> 16
        self.dec3 = DoubleConv(c4 + c3, c3)       # concat skip3 -> 16x16
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')   # 16 -> 32
        self.dec2 = DoubleConv(c3 + c2, c2)       # concat skip2 -> 32x32
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')   # 32 -> 64
        self.dec1 = DoubleConv(c2 + c1, c1)       # concat skip1 -> 64x64

        # output head
        self.out_conv = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, x, return_peak=False):
        # encoder
        s1 = self.enc1(x)                                       # (B, c1, 64, 64)
        s2 = self.enc2(self.pool1(s1))                          # (B, c2, 32, 32)
        s3 = self.enc3(self.pool2(s2))                          # (B, c3, 16, 16)
        b = self.bottleneck(self.pool3(s3))                     # (B, c4, 8, 8)
        # decoder
        d3 = self.dec3(torch.cat([self.up3(b), s3], dim=1))     # (B, c3, 16, 16)
        d2 = self.dec2(torch.cat([self.up2(d3), s2], dim=1))    # (B, c2, 32, 32)
        d1 = self.dec1(torch.cat([self.up1(d2), s1], dim=1))    # (B, c1, 64, 64)
        pred = self.out_conv(d1)                                # (B, out_channels, 64, 64)
        if return_peak:
            # peak temperature per sample (max over spatial dims), for peak-error
            pred_peak = pred.amax(dim=(-1, -2))                 # (B, out_channels)
            return pred, pred_peak
        return pred

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        return c


if __name__ == "__main__":
    # quick shape / backward smoke (no data needed)
    net = UNet(in_channels=3, out_channels=1)
    x = torch.randn(4, 3, 64, 64)
    pred, peak = net(x, return_peak=True)
    loss = torch.nn.functional.mse_loss(pred, torch.randn(4, 1, 64, 64))
    loss.backward()
    print(f"pred {tuple(pred.shape)}  peak {tuple(peak.shape)}  params {net.count_params()}")
