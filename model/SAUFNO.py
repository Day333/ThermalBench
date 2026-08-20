"""SAU-FNO (DAC'25, arXiv:2510.15968): U-FNO + one self-attention block between the
last U-Fourier layer and the fc projection.

The attention block follows the paper's Fig. 2 / Eq. (9)-(10) verbatim:

    A_c = W_h V(x),  Q = W_q V(x),  K = W_k V(x)          (1x1 convolutions)
    A_s = softmax(Q K^T),  V'(x) = A_s @ A_c

i.e. a single-head, full-spatial self-attention with no LayerNorm, no FFN, no output
projection and no residual -- the attention output *replaces* the feature map. It sits
where the paper puts it: after the last U-Fourier layer, before the fc projection
(the paper reports last-layer-only placement matches adding it to every layer).

Two documented adaptations, both forced by the setting rather than chosen:
- The paper attends over a 64x64 2D grid (feature map [B, 64, 64, d], 4096 tokens).
  This network is 3D with Z padded to 8, and full 3D attention (32768 tokens) is
  quadratically infeasible to train, so attention runs over each z-plane
  independently -- exactly the paper's 2D token structure, applied per plane.
- Eq. (9) writes s_ij = Q_i^T K_j with no 1/sqrt(d) temperature; the standard scaling
  is kept (torch's default), reading the omission as notational.
"""
import torch.nn as nn
import torch.nn.functional as F

from model.UFNO import Net3d, SimpleBlock3d


class PaperAttentionBlock(nn.Module):
    """The DAC'25 SAU-FNO self-attention block: 1x1-conv Q/K/V, single head, softmax
    attention over the spatial positions of each z-plane, output replaces the input.
    Input/output: (B, C, X, Y, Z)."""

    def __init__(self, channels):
        super().__init__()
        self.q = nn.Conv3d(channels, channels, 1)
        self.k = nn.Conv3d(channels, channels, 1)
        self.h = nn.Conv3d(channels, channels, 1)   # W_h, the paper's A_c branch

    def forward(self, x):
        B, C, X, Y, Z = x.shape
        q, k, v = self.q(x), self.k(x), self.h(x)

        def planes(t):
            # (B, C, X, Y, Z) -> (B*Z, 1, X*Y, C): one single-head attention problem
            # per z-plane. The explicit head axis is required: with 3-D inputs SDPA
            # silently falls back to the math kernel, which materializes the full
            # (X*Y)x(X*Y) attention matrix -- ~10 GiB per batch at 64x64 with Z=8 --
            # while the 4-D layout dispatches to the memory-efficient kernel.
            return t.permute(0, 4, 2, 3, 1).reshape(B * Z, 1, X * Y, C)

        out = F.scaled_dot_product_attention(planes(q), planes(k), planes(v))
        return out.reshape(B, Z, X, Y, C).permute(0, 4, 2, 3, 1)


class SAUSimpleBlock3d(SimpleBlock3d):
    """U-FNO SimpleBlock3d + the paper's attention block after the last U-Fourier
    layer (before the fc projection). All Fourier/U-Net layers are inherited from
    SimpleBlock3d unchanged, so SAU-FNO and U-FNO differ only in the attention
    block; `paper_attn` is the checkpoint-pinned attribute name."""

    def __init__(self, modes1, modes2, modes3, width, in_channels=12):
        super().__init__(modes1, modes2, modes3, width, in_channels=in_channels)
        self.paper_attn = PaperAttentionBlock(self.width)

    def forward(self, x):
        v = self._layers(self._lift(x))
        v = self.paper_attn(v)
        return self._project(v)


class SAUNet3d(Net3d):
    """Same pad-to-multiple-of-8 + crop wrapper as Net3d, around
    SAUSimpleBlock3d."""

    def __init__(self, modes1, modes2, modes3, width, in_channels=12):
        super(Net3d, self).__init__()
        self.conv1 = SAUSimpleBlock3d(modes1, modes2, modes3, width,
                                           in_channels=in_channels)
