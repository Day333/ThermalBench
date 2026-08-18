"""SAU-FNO's axial multi-head self-attention block (from sau_fno.py).

Full spatial attention is infeasible (64x64x8 = 32768 tokens, an attention matrix of
roughly 343GB), so attention is applied along the Z, Y and X axes separately
(<=64 tokens per axis).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AxialAttentionBlock(nn.Module):
    """Pre-norm transformer block with axial multi-head self-attention (Z, Y, X)
    followed by an FFN. All sub-layers are residual.
    Input/output: (B, C, X, Y, Z)."""

    def __init__(self, channels, heads=4, ff_ratio=4):
        super().__init__()
        self.C = channels
        self.attn_z = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.attn_y = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.attn_x = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm_z = nn.LayerNorm(channels)
        self.norm_y = nn.LayerNorm(channels)
        self.norm_x = nn.LayerNorm(channels)
        self.norm_ff = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.Linear(channels, ff_ratio * channels),
            nn.GELU(),
            nn.Linear(ff_ratio * channels, channels),
        )

    @staticmethod
    def _axial(x, attn, norm, axis):
        """Attend along one axis. axis: 0=X, 1=Y, 2=Z. Returns attn(norm(x)) in
        the original (B,C,X,Y,Z) layout (the residual delta)."""
        B, C, X, Y, Z = x.shape
        if axis == 2:  # Z-axis: seq=Z, batch=(B,X,Y)
            t = x.permute(0, 2, 3, 4, 1).reshape(B * X * Y, Z, C)
            t = norm(t)
            t = attn(t, t, t, need_weights=False)[0]
            return t.reshape(B, X, Y, Z, C).permute(0, 4, 1, 2, 3)
        elif axis == 1:  # Y-axis: seq=Y, batch=(B,X,Z)
            t = x.permute(0, 2, 4, 3, 1).reshape(B * X * Z, Y, C)
            t = norm(t)
            t = attn(t, t, t, need_weights=False)[0]
            return t.reshape(B, X, Z, Y, C).permute(0, 4, 1, 3, 2)
        else:           # X-axis: seq=X, batch=(B,Y,Z)
            t = x.permute(0, 3, 4, 2, 1).reshape(B * Y * Z, X, C)
            t = norm(t)
            t = attn(t, t, t, need_weights=False)[0]
            return t.reshape(B, Y, Z, X, C).permute(0, 4, 3, 1, 2)

    def forward(self, x):
        x = x + self._axial(x, self.attn_z, self.norm_z, 2)   # Z (len 8)
        x = x + self._axial(x, self.attn_y, self.norm_y, 1)   # Y (len 64)
        x = x + self._axial(x, self.attn_x, self.norm_x, 0)   # X (len 64)
        # FFN over channel dim (pre-norm, residual)
        B, C, X, Y, Z = x.shape
        t = x.permute(0, 2, 3, 4, 1).reshape(-1, C)
        t = self.ff(self.norm_ff(t))
        x = x + t.reshape(B, X, Y, Z, C).permute(0, 4, 1, 2, 3)
        return x
