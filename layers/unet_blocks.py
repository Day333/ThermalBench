"""U-Net building blocks.

`U_net` is the small per-layer U-Net that U-FNO (Wen et al., 2022) runs in parallel
with its last three Fourier layers. It is implemented here from the architecture
described in the paper (three stride-2 encoder levels, transposed-conv decoder with
skip concatenations, and a final full-resolution fusion with the layer input).
Attribute names and the internal nn.Sequential layout are pinned by the released
benchmark checkpoints, whose state_dict keys derive from them.

`DoubleConv` is the double-convolution block of the standalone UNet baseline. The two
classes are unrelated and share this file only because both belong to the U-Net family.
"""
import torch
import torch.nn as nn


def _down(channels, kernel_size, stride, dropout_rate):
    """One encoder stage: strided conv + BatchNorm + LeakyReLU (+ Dropout).

    The four-element Sequential (indices 0..3) is part of the checkpoint format:
    parameters live at .0 (conv) and .1 (batchnorm)."""
    return nn.Sequential(
        nn.Conv3d(channels, channels, kernel_size, stride=stride,
                  padding=(kernel_size - 1) // 2, bias=False),
        nn.BatchNorm3d(channels),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Dropout(dropout_rate),
    )


def _up(in_channels, out_channels):
    """One decoder stage: 2x transposed conv + LeakyReLU (parameters at .0)."""
    return nn.Sequential(
        nn.ConvTranspose3d(in_channels, out_channels, kernel_size=4, stride=2,
                           padding=1),
        nn.LeakyReLU(0.1, inplace=True),
    )


class U_net(nn.Module):
    """The U-FNO companion U-Net: (B, C, X, Y, Z) -> (B, C, X, Y, Z).

    Encoder: full resolution -> 1/2 -> 1/4 -> 1/8, with an extra stride-1 conv at the
    two deepest levels. Decoder: three transposed convs, each fused with the matching
    encoder feature by channel concatenation; a last convolution maps the concatenation
    of the input and the top decoder feature back to `output_channels`.
    """

    def __init__(self, input_channels, output_channels, kernel_size, dropout_rate):
        super().__init__()
        c, k, dr = input_channels, kernel_size, dropout_rate
        self.conv1 = _down(c, k, 2, dr)               # 1/2
        self.conv2 = _down(c, k, 2, dr)               # 1/4
        self.conv2_1 = _down(c, k, 1, dr)
        self.conv3 = _down(c, k, 2, dr)               # 1/8
        self.conv3_1 = _down(c, k, 1, dr)
        self.deconv2 = _up(c, c)                      # 1/8 -> 1/4
        self.deconv1 = _up(c * 2, c)                  # 1/4 -> 1/2
        self.deconv0 = _up(c * 2, c)                  # 1/2 -> 1/1
        self.output_layer = nn.Conv3d(c * 2, output_channels, kernel_size=k,
                                      stride=1, padding=(k - 1) // 2)

    def forward(self, x):
        enc1 = self.conv1(x)
        enc2 = self.conv2_1(self.conv2(enc1))
        enc3 = self.conv3_1(self.conv3(enc2))
        dec2 = self.deconv2(enc3)
        dec1 = self.deconv1(torch.cat((enc2, dec2), dim=1))
        dec0 = self.deconv0(torch.cat((enc1, dec1), dim=1))
        return self.output_layer(torch.cat((x, dec0), dim=1))


class DoubleConv(nn.Module):
    """Conv3x3 + ReLU + Conv3x3 + ReLU.

    No BatchNorm / Dropout / attention, to keep close to the paper's lightweight
    U-Net. padding=1 keeps the spatial size unchanged.
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)
