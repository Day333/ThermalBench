"""SAU-FNO = U-FNO + an axial self-attention block between the last U-Fourier layer
and the projection.

The attention block (layers/attention.py) is this project's implementation: the SAU-FNO
paper (DAC'25) applies one full-spatial single-head attention over a 64x64 2D grid,
which is quadratically infeasible on this benchmark's padded 3D grids, so attention is
applied axially instead -- see the benchmark docs for the disclosed adaptation. Every
Fourier/U-Net component is inherited unchanged from model/UFNO.py, so the two models
differ only in the attention block; `axial_block` is the checkpoint-pinned attribute
name.
"""
from layers.attention import AxialAttentionBlock
from model.UFNO import Net3d, SimpleBlock3d


class SAUSimpleBlock3d(SimpleBlock3d):
    """U-FNO SimpleBlock3d + one AxialAttentionBlock before the fc projection."""

    def __init__(self, modes1, modes2, modes3, width, in_channels=12, attn_heads=4):
        super().__init__(modes1, modes2, modes3, width, in_channels=in_channels)
        self.axial_block = AxialAttentionBlock(self.width, heads=attn_heads)

    def forward(self, x):
        v = self._layers(self._lift(x))
        v = self.axial_block(v)
        return self._project(v)


class SAUNet3d(Net3d):
    """Same pad-to-multiple-of-8 + crop wrapper as Net3d, around SAUSimpleBlock3d."""

    def __init__(self, modes1, modes2, modes3, width, in_channels=12, attn_heads=4):
        super(Net3d, self).__init__()
        self.conv1 = SAUSimpleBlock3d(modes1, modes2, modes3, width,
                                      in_channels=in_channels, attn_heads=attn_heads)
