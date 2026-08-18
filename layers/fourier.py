"""Three 3D spectral convolution layers.

`SpectralConv3d` / `FactorizedSpectralConv3d` come from U-FNO (ufno.py);
`FNOSpectralConv3d` comes from FNO (fno/models/fourier_3d.py). The two implementations
are NOT the same, so both are kept rather than merged. The FNO one was renamed only to
resolve the name clash -- its body is untouched, and state_dict keys derive from
attribute names (weights1/weights2/...) rather than class names, so the rename does not
affect weight loading.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import operator
from functools import reduce
from functools import partial


# =========================================================================
#  SpectralConv3d (original U-FNO 3D Fourier layer)
# =========================================================================
class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    def compl_mul3d(self, input, weights):
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


# =========================================================================
#  FactorizedSpectralConv3d (variant B): per-axis 1D Fourier, sum fusion.
#  Replaces the 3D Fourier kernel with three 1D kernels along X/Y/Z, reducing
#  params from O(Ci*Co*m1*m2*m3*4) to O(Ci*Co*(m1+m2+m3)*2) so we can afford
#  more modes. Low + high (negative-freq) modes per axis.
# =========================================================================


# =========================================================================
#  FactorizedSpectralConv3d (variant B): per-axis 1D Fourier, sum fusion.
#  Replaces the 3D Fourier kernel with three 1D kernels along X/Y/Z, reducing
#  params from O(Ci*Co*m1*m2*m3*4) to O(Ci*Co*(m1+m2+m3)*2) so we can afford
#  more modes. Low + high (negative-freq) modes per axis.
# =========================================================================
class FactorizedSpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(FactorizedSpectralConv3d, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.m1, self.m2, self.m3 = modes1, modes2, modes3
        s = 1.0 / (in_channels * out_channels)
        self.wx_lo = nn.Parameter(s * torch.rand(in_channels, out_channels, modes1, dtype=torch.cfloat))
        self.wx_hi = nn.Parameter(s * torch.rand(in_channels, out_channels, modes1, dtype=torch.cfloat))
        self.wy_lo = nn.Parameter(s * torch.rand(in_channels, out_channels, modes2, dtype=torch.cfloat))
        self.wy_hi = nn.Parameter(s * torch.rand(in_channels, out_channels, modes2, dtype=torch.cfloat))
        self.wz_lo = nn.Parameter(s * torch.rand(in_channels, out_channels, modes3, dtype=torch.cfloat))
        self.wz_hi = nn.Parameter(s * torch.rand(in_channels, out_channels, modes3, dtype=torch.cfloat))

    def _axis(self, x, w_lo, w_hi, dim, modes):
        # x: (B, C, X, Y, Z). 1D rFFT along `dim`, multiply low+high modes, irFFT.
        xf = torch.fft.rfft(x, dim=dim)
        nd = xf.ndim
        perm = [0, 1, dim] + [d for d in range(2, nd) if d != dim]   # B, C, dim, rest
        xf_p = xf.permute(*perm)
        Nf = xf_p.shape[2]
        rest = xf_p.shape[3:]
        B, C = x.shape[0], x.shape[1]
        R = 1
        for r in rest:
            R *= r
        xf_flat = xf_p.reshape(B, C, Nf, R)
        out_flat = torch.zeros(B, self.out_channels, Nf, R, dtype=torch.cfloat, device=x.device)
        out_flat[:, :, :modes] = torch.einsum("bcmr,com->bomr", xf_flat[:, :, :modes], w_lo)
        if modes > 1:
            out_flat[:, :, -modes:] = torch.einsum("bcmr,com->bomr", xf_flat[:, :, -modes:], w_hi)
        out_p = out_flat.reshape(B, self.out_channels, Nf, *rest)
        inv = [0] * nd
        for i, ax in enumerate(perm):
            inv[ax] = i
        out = out_p.permute(*inv)
        return torch.fft.irfft(out, n=x.shape[dim], dim=dim)

    def forward(self, x):
        # x: (B, C, X, Y, Z) -> dim 2=X, 3=Y, 4=Z
        return (self._axis(x, self.wx_lo, self.wx_hi, 2, self.m1) +
                self._axis(x, self.wy_lo, self.wy_hi, 3, self.m2) +
                self._axis(x, self.wz_lo, self.wz_hi, 4, self.m3))


# =========================================================================
#  LocalBranch (variant A): depthwise 3D conv + pointwise, per-block local feat.
# =========================================================================


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
