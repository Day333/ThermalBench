"""U-FNO's three optional variant components (all off by default; enabling them gives
the Local / FiLM variants).

The benchmark runs plain U-FNO, so these branches take no part in the forward pass.
They are kept so the variant experiments remain reproducible.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
#  LocalBranch (variant A): depthwise 3D conv + pointwise, per-block local feat.
# =========================================================================
class LocalBranch(nn.Module):
    def __init__(self, width, kernel=3):
        super(LocalBranch, self).__init__()
        self.dw = nn.Conv3d(width, width, kernel_size=kernel, padding=kernel // 2, groups=width)
        self.pw = nn.Conv3d(width, width, kernel_size=1)

    def forward(self, x):
        return self.pw(self.dw(x))


# =========================================================================
#  FiLMModulator (variant C): case-level global features -> per-block gamma/beta.
#  Features derived from the power channel (mean/max/sum/area/std) + grid spread.
# =========================================================================


# =========================================================================
#  FiLMModulator (variant C): case-level global features -> per-block gamma/beta.
#  Features derived from the power channel (mean/max/sum/area/std) + grid spread.
# =========================================================================
def compute_global_feats(x_in):
    # x_in: (B, X, Y, Z, C) with C[0]=chiplet_power. Returns (B, n_feats).
    power = x_in[..., 0]                                  # (B, X, Y, Z)
    mean = power.mean(dim=(1, 2, 3))
    mx = power.amax(dim=(1, 2, 3))
    ssum = power.sum(dim=(1, 2, 3))
    area = (power > 1e-6).float().mean(dim=(1, 2, 3))
    std = power.std(dim=(1, 2, 3))
    return torch.stack([mean, mx, ssum, area, std], dim=1)   # (B, 5)


class FiLMModulator(nn.Module):
    def __init__(self, width, n_feats=5, n_blocks=6):
        super(FiLMModulator, self).__init__()
        self.width, self.n_blocks = width, n_blocks
        self.feat_norm = nn.LayerNorm(n_feats)             # normalize global feats (power sum is large)
        self.mlp = nn.Sequential(
            nn.Linear(n_feats, 64), nn.GELU(), nn.Linear(64, 2 * width * n_blocks)
        )
        # zero-init last layer -> gamma=beta=0 at start -> (1+0)*x+0 = x (pure U-FNO start)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, feats):
        feats = self.feat_norm(feats)
        out = self.mlp(feats).view(-1, self.n_blocks, 2, self.width)
        gamma = torch.tanh(out[:, :, 0])                   # constrain gamma to [-1, 1] (stability)
        beta = out[:, :, 1]
        return gamma, beta                                 # (B, nB, W)
