"""3D spectral convolution layers.

Both classes implement the standard FNO spectral convolution and both follow the
MIT-licensed reference implementation of the FNO line of work (Zongyi Li,
https://github.com/zongyi-li/fourier_neural_operator): rFFT, a learned per-mode
complex linear map on the retained low-frequency corner modes, irFFT.

They are kept separate because they arrived through two checkpoint lineages that must
keep loading independently: `SpectralConv3d` is the one inside U-FNO / SAU-FNO
checkpoints, `FNOSpectralConv3d` (vendored verbatim from fno/models/fourier_3d.py) is
the one inside FNO checkpoints. state_dict keys derive from attribute names
(weights1..weights4), which the two share.
"""
import torch
import torch.nn as nn


class SpectralConv3d(nn.Module):
    """Fourier layer for (B, C, X, Y, Z) feature maps.

    Keeps modes1 x modes2 x modes3 modes in each of the four (+-kx, +-ky) corners at
    non-negative kz (the z axis is rFFT-halved), applies one complex channel-mixing
    weight tensor per corner, and returns to physical space.
    """

    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.scale = 1 / (in_channels * out_channels)
        # One weight tensor per (+-kx, +-ky) corner, created in index order so seeded
        # initialization draws the same random stream as the released checkpoints.
        for i in (1, 2, 3, 4):
            setattr(self, f"weights{i}", nn.Parameter(
                self.scale * torch.rand(in_channels, out_channels,
                                        modes1, modes2, modes3,
                                        dtype=torch.cfloat)))

    def _corners(self):
        m1, m2 = self.modes1, self.modes2
        lo1, hi1 = slice(None, m1), slice(-m1, None)
        lo2, hi2 = slice(None, m2), slice(-m2, None)
        return ((self.weights1, lo1, lo2), (self.weights2, hi1, lo2),
                (self.weights3, lo1, hi2), (self.weights4, hi1, hi2))

    def forward(self, x):
        m3 = self.modes3
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))
        out_ft = torch.zeros(x.shape[0], self.out_channels,
                             x.size(-3), x.size(-2), x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        for w, sx, sy in self._corners():
            out_ft[:, :, sx, sy, :m3] = torch.einsum(
                "bixyz,ioxyz->boxyz", x_ft[:, :, sx, sy, :m3], w)
        return torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))


class FNOSpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat,
            )
        )
        self.weights2 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat,
            )
        )
        self.weights3 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat,
            )
        )
        self.weights4 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat,
            )
        )

    def compl_mul3d(self, input_tensor, weights):
        return torch.einsum("bixyz,ioxyz->boxyz", input_tensor, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-3),
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, : self.modes1, : self.modes2, : self.modes3], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3], self.weights2
        )
        out_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3], self.weights3
        )
        out_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3], self.weights4
        )
        return torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
