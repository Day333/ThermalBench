"""U-FNO (Wen et al., 2022): an FNO whose last Fourier layers each run a small U-Net
in parallel to recover local, high-frequency detail.

Implemented from the architecture description in the paper ("U-FNO -- an enhanced
Fourier neural operator-based deep-learning model for multiphase flow", Advances in
Water Resources 163, 104180); the spectral convolution follows the MIT-licensed FNO
reference implementation (see layers/fourier.py). No source text from the original
U-FNO repository (which is CC BY-NC-ND licensed) is used.

Compatibility contract, pinned by the released benchmark checkpoints and by the
legacy whole-object pickles that utils/compat.py still loads:
- attribute names: fc0, conv0..conv5, w0..w5, unet3..unet5, fc1, fc2;
- module construction order (fc0, all convs, all ws, the unets, fc1, fc2), so seeded
  runs draw the same initialization stream as the historical code;
- forward numerics: relu(spectral + pointwise [+ unet(layer input)]) per layer.
The regression suite checks that the released checkpoints reproduce their published
metrics under this implementation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.fourier import SpectralConv3d
from layers.unet_blocks import U_net

# The historical module seeded the RNG at import time. The training scripts re-seed
# afterwards, so results do not depend on it; it is kept so the RNG stream is
# bit-identical to the historical code in every situation.
torch.manual_seed(0)

N_LAYERS = 6      # 3 Fourier layers + 3 U-Fourier layers
UNET_FROM = 3     # layers UNET_FROM..N_LAYERS-1 carry the U-Net branch


class SimpleBlock3d(nn.Module):
    """Lift -> N_LAYERS (U-)Fourier layers -> project.

    input  (B, X, Y, Z, in_channels)
    output (B, X, Y, Z, 1)
    """

    def __init__(self, modes1, modes2, modes3, width, in_channels=12):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.in_channels = in_channels

        self.fc0 = nn.Linear(in_channels, width)
        for i in range(N_LAYERS):
            setattr(self, f"conv{i}",
                    SpectralConv3d(width, width, modes1, modes2, modes3))
        for i in range(N_LAYERS):
            setattr(self, f"w{i}", nn.Conv1d(width, width, 1))
        for i in range(UNET_FROM, N_LAYERS):
            setattr(self, f"unet{i}", U_net(width, width, 3, 0))
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def _lift(self, x):
        """(B, X, Y, Z, P) -> channel-first features (B, width, X, Y, Z)."""
        return self.fc0(x).permute(0, 4, 1, 2, 3)

    def _layers(self, v):
        """Run the six (U-)Fourier layers on (B, width, X, Y, Z)."""
        b, w = v.shape[0], self.width
        sx, sy, sz = v.shape[2], v.shape[3], v.shape[4]
        for i in range(N_LAYERS):
            spectral = getattr(self, f"conv{i}")(v)
            pointwise = getattr(self, f"w{i}")(
                v.reshape(b, w, -1)).reshape(b, w, sx, sy, sz)
            out = spectral + pointwise
            if i >= UNET_FROM:
                # The paper's U-Fourier layer feeds the *layer input* v_l to the
                # U-Net, in parallel with the spectral and pointwise paths -- not the
                # partial sum. Changing this stops trained checkpoints reproducing.
                out = out + getattr(self, f"unet{i}")(v)
            v = F.relu(out)
        return v

    def _project(self, v):
        """(B, width, X, Y, Z) -> (B, X, Y, Z, 1)."""
        v = v.permute(0, 2, 3, 4, 1)
        return self.fc2(F.relu(self.fc1(v)))

    def forward(self, x):
        return self._project(self._layers(self._lift(x)))


class Net3d(nn.Module):
    """Pads X/Y/Z up to multiples of 8 (the U-Net has three stride-2 levels), runs
    SimpleBlock3d, and crops back. Y/Z are replicate-padded, X zero-padded."""

    def __init__(self, modes1, modes2, modes3, width, in_channels=12):
        super().__init__()
        self.conv1 = SimpleBlock3d(modes1, modes2, modes3, width,
                                   in_channels=in_channels)

    def forward(self, x):
        b, sx, sy, sz = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        px, py, pz = (-sx) % 8, (-sy) % 8, (-sz) % 8
        x = F.pad(x, (0, 0, 0, pz, 0, py), "replicate")
        if px:
            x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, px), "constant", 0)
        x = self.conv1(x)
        x = x.view(b, sx + px, sy + py, sz + pz, 1)
        return x[:, :sx, :sy, :sz, :].squeeze(-1)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())
