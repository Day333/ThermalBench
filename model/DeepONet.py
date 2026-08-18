"""MSE-supervised DeepONet (derived from DeepOHeat, component for component).

Only two things differ from the original: the supervision signal is MSE against the
HotSpot temperature field rather than a PDE residual, and the query points are fixed to
the 64x64 grid (so the trunk is evaluated once per batch).
The interface is wrapped as (B,X,Y,Z,P) -> (B,X,Y,Z) to match U-FNO/UNet. Class body
copied verbatim from deeponet.py.
"""
import operator
from functools import reduce

import torch
import torch.nn as nn

from layers.mlp import FCBlock


class DeepONetSup(nn.Module):
    def __init__(self, in_channels=3, grid=64,
                 trunk_hidden=128, branch_hidden=256, inner_prod=128,
                 num_trunk_hidden=3, num_branch_hidden=7,
                 freq=2 * torch.pi, std=1.0):
        super().__init__()
        self.in_channels, self.grid = in_channels, grid
        self.branch = FCBlock(grid * grid * in_channels, branch_hidden,
                              inner_prod, num_branch_hidden)
        self.trunk = FCBlock(trunk_hidden, trunk_hidden, inner_prod, num_trunk_hidden)
        ff = torch.zeros(2, trunk_hidden // 2).normal_(0, std) * freq
        self.register_buffer("fourier_features", ff)
        self.b_0 = nn.Parameter(torch.zeros(1).uniform_())
        xs = torch.linspace(0.0, 1.0, grid)
        gx, gy = torch.meshgrid(xs, xs, indexing="ij")
        self.register_buffer("coords", torch.stack([gx, gy], -1).reshape(-1, 2))  # (G²,2)

    def forward(self, x):
        """x: (B, X, Y, Z, P), Z=1  ->  (B, X, Y, Z)"""
        B = x.shape[0]
        u = x.squeeze(-2).reshape(B, -1)                              # (B, G²·P)
        ff = torch.matmul(self.coords, self.fourier_features)         # (G², H/2)
        t = self.trunk(torch.cat([torch.sin(ff), torch.cos(ff)], -1))  # (G², p)
        b = self.branch(u)                                            # (B, p)
        out = torch.matmul(b, t.t()) + self.b_0                       # (B, G²)
        return out.reshape(B, self.grid, self.grid, 1)

    def count_params(self):
        return sum(reduce(operator.mul, p.size()) for p in self.parameters())


if __name__ == "__main__":
    for p in (3, 4, 7):
        net = DeepONetSup(in_channels=p)
        x = torch.randn(2, 64, 64, 1, p)
        y = net(x)
        assert y.shape == (2, 64, 64, 1), y.shape
        print(f"P={p}  params={net.count_params()}")
