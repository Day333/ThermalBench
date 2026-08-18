"""U-FNO (Wen et al.) -- each Fourier layer runs in parallel with a small U-Net.

Class bodies are copied verbatim from ufno.py; only the imports of SpectralConv3d /
U_net / LocalBranch and friends were repointed at layers/. The module-level
torch.manual_seed(0) is preserved (the original file had it); the training scripts set
the seed again afterwards, so the final RNG state is unchanged.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import operator
from functools import reduce
from functools import partial

from layers.fourier import SpectralConv3d, FactorizedSpectralConv3d
from layers.unet_blocks import U_net
from layers.modulation import LocalBranch, compute_global_feats, FiLMModulator

torch.manual_seed(0)


class SimpleBlock3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width, in_channels=12,
                 use_local=False, local_kernel=3, local_alpha=0.1,
                 use_factorized=False, use_film=False):
        super(SimpleBlock3d, self).__init__()
        """
        U-FNO contains 3 Fourier layers and 3 U-Fourier layers.
        input/output: (batchsize, x, y, z, c=in_channels) -> (..., c=1)

        Switches (all default False = original U-FNO, bit-identical):
          use_local      (A): add depthwise LocalBranch + learnable alpha per block
          use_factorized (B): replace SpectralConv3d with FactorizedSpectralConv3d
          use_film       (C): case-level global feats -> per-block gamma/beta FiLM
        """
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.in_channels = in_channels
        self.use_local = use_local
        self.use_factorized = use_factorized
        self.use_film = use_film
        self.fc0 = nn.Linear(in_channels, self.width)

        Conv = FactorizedSpectralConv3d if use_factorized else SpectralConv3d
        self.conv0 = Conv(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = Conv(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = Conv(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = Conv(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv4 = Conv(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv5 = Conv(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.w4 = nn.Conv1d(self.width, self.width, 1)
        self.w5 = nn.Conv1d(self.width, self.width, 1)
        self.unet3 = U_net(self.width, self.width, 3, 0)
        self.unet4 = U_net(self.width, self.width, 3, 0)
        self.unet5 = U_net(self.width, self.width, 3, 0)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

        if use_local:
            for i in range(6):
                setattr(self, f"local{i}", LocalBranch(self.width, local_kernel))
                setattr(self, f"alpha{i}", nn.Parameter(torch.tensor(float(local_alpha))))
        if use_film:
            self.film = FiLMModulator(self.width, n_feats=5, n_blocks=6)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y, size_z = x.shape[1], x.shape[2], x.shape[3]

        if self.use_film:
            gamma, beta = self.film(compute_global_feats(x))   # (B,6,W)
            gview = lambda i: gamma[:, i].view(batchsize, self.width, 1, 1, 1)
            bview = lambda i: beta[:, i].view(batchsize, self.width, 1, 1, 1)

        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)

        def block(x, ci, wi, ai=None, li=None, ui=None, idx=0):
            x_in = x
            x1 = ci(x)
            x2 = wi(x.reshape(batchsize, self.width, -1)).reshape(batchsize, self.width, size_x, size_y, size_z)
            x = x1 + x2
            if ui is not None:
                # Original U-FNO (Wen et al.): U_net acts on the block input v_l, not on
                # (K v_l + W v_l). Do NOT change this to ui(x) -- that would redefine the
                # U-Fourier layer and stop trained checkpoints from reproducing.
                x = x + ui(x_in)
            if self.use_local:
                x = x + ai * li(x_in)
            if self.use_film:
                x = (1 + gview(idx)) * x + bview(idx)
            return F.relu(x)

        x = block(x, self.conv0, self.w0, self.alpha0 if self.use_local else None, self.local0 if self.use_local else None, None, 0)
        x = block(x, self.conv1, self.w1, self.alpha1 if self.use_local else None, self.local1 if self.use_local else None, None, 1)
        x = block(x, self.conv2, self.w2, self.alpha2 if self.use_local else None, self.local2 if self.use_local else None, None, 2)
        x = block(x, self.conv3, self.w3, self.alpha3 if self.use_local else None, self.local3 if self.use_local else None, self.unet3, 3)
        x = block(x, self.conv4, self.w4, self.alpha4 if self.use_local else None, self.local4 if self.use_local else None, self.unet4, 4)
        x = block(x, self.conv5, self.w5, self.alpha5 if self.use_local else None, self.local5 if self.use_local else None, self.unet5, 5)

        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class Net3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width, in_channels=12,
                 use_local=False, local_kernel=3, local_alpha=0.1,
                 use_factorized=False, use_film=False):
        super(Net3d, self).__init__()
        self.conv1 = SimpleBlock3d(modes1, modes2, modes3, width, in_channels=in_channels,
                                   use_local=use_local, local_kernel=local_kernel,
                                   local_alpha=local_alpha,
                                   use_factorized=use_factorized, use_film=use_film)

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
