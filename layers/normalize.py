"""Input/target normalizers.

`normalize` is a single global scalar (the original behaviour); `per_channel_normalize`
computes statistics per channel. The P=7 level4/level5 require the per-channel version:
under global normalization r_convec has a std of only 6.7e-6 while h has 1.93, a factor
of 2.9e5, which crushes the small-magnitude channels into numerical noise -- and
r_convec happens to be the input most correlated with temperature level (corr +0.728).
Copied verbatim from fno/models/normalize.py.

`make_norm` reads the PER_CHANNEL_NORM environment variable to choose between them; the
benchmark keeps it at 1 throughout.
"""
import torch
import torch.nn as nn


class normalize(nn.Module):
    def __init__(self, x0, if_trainable=False):
        super().__init__()
        self.mean = nn.Parameter(x0.mean(), requires_grad=if_trainable)
        self.std = nn.Parameter(x0.std(), requires_grad=if_trainable)

    def forward(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean


class per_channel_normalize(nn.Module):
    """Per-channel mean/std.

    For x of shape (B, X, Y, Z, P) the statistics are taken over every dimension but
    the last; targets of shape (B, X, Y, Z) have no channel dimension and fall back to
    a global scalar, matching `normalize`.

    Why: `normalize` is a single global scalar, which for multi-channel inputs spanning
    very different magnitudes crushes the small ones into numerical noise. On level5,
    global normalization leaves r_conv with a std of 6.7e-6 against 1.93 for h -- a
    factor of 2.9e5 -- and r_conv is the input most correlated with temperature level
    (corr +0.728).
    """

    def __init__(self, x0, if_trainable=False):
        super().__init__()
        if x0.dim() == 5:
            mean, std = x0.mean(dim=(0, 1, 2, 3)), x0.std(dim=(0, 1, 2, 3))
        else:
            mean, std = x0.mean(), x0.std()
        std = torch.where(std == 0, torch.ones_like(std), std)
        self.mean = nn.Parameter(mean, requires_grad=if_trainable)
        self.std = nn.Parameter(std, requires_grad=if_trainable)

    def forward(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean


def make_norm(x0, if_trainable=False):
    """PER_CHANNEL_NORM=1 -> per-channel; otherwise the original global scalar."""
    import os as _os
    if _os.environ.get("PER_CHANNEL_NORM", "0") in ("1", "true", "True", "yes"):
        return per_channel_normalize(x0, if_trainable)
    return normalize(x0, if_trainable)


def cal_rmse(y_true, y_pred):
    mse = torch.mean((y_true - y_pred) ** 2)
    rmse = torch.sqrt(mse)
    return rmse.item()
