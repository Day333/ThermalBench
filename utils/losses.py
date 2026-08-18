"""Training losses.

`thermal_combined_loss` is computed in normalized space: MSE + gradient term + hotspot
term + peak term (loss_type="boundary" adds a chip-boundary term on top). Every model in
the benchmark runs loss_type="base", which reduces to plain MSE; the other branches are
kept so the thermal/boundary variant experiments remain reproducible.
`LpLoss` is the relative Lp loss common in the FNO line of work; unused by the benchmark.
Copied verbatim from fno/models/fourier_3d.py and fno/models/utilities3.py respectively.
"""
import torch
import torch.nn.functional as F

import numpy as np


def thermal_combined_loss(out, y, x, loss_type="base", lam=None):
    """Thermal combined loss on NORMALIZED outputs of shape (B, X, Y, Z), Z=1.
    Mirrors Therm-FM p=5/p=6 (scOT/model.py).
      base    = MSE(out, y)
      thermal = base + grad*lg + hotspot(top-k%)*lh + peak*lp
      boundary= thermal + boundary(block-edge mask from input power)*lb
    `x` is the input (B, X, Y, Z, P); power is its last-dim channel 0.
    """
    base = F.mse_loss(out, y)
    if loss_type in (None, "base", ""):
        return base
    lam = lam or {}
    lg = float(lam.get("grad", 0.1))
    lh = float(lam.get("hot", 0.5))
    lp = float(lam.get("peak", 0.1))
    alpha = float(lam.get("alpha", 0.9))

    # gradient loss along X (dim=1) and Y (dim=2)
    gx = out[:, 1:, :, :] - out[:, :-1, :, :]
    gy = out[:, :, 1:, :] - out[:, :, :-1, :]
    gx_gt = y[:, 1:, :, :] - y[:, :-1, :, :]
    gy_gt = y[:, :, 1:, :] - y[:, :, :-1, :]
    grad_loss = F.mse_loss(gx, gx_gt) + F.mse_loss(gy, gy_gt)

    # hotspot loss: MSE over the hottest (1-alpha) TRUE pixels, per sample
    B = out.shape[0]
    pf = out.reshape(B, -1)
    lf = y.reshape(B, -1)
    q = torch.quantile(lf, alpha, dim=1, keepdim=True)
    mask = (lf >= q).to(out.dtype)
    msum = mask.sum(dim=1)
    hot_loss = (((pf - lf) ** 2 * mask).sum(dim=1) / msum.clamp_min(1.0)).mean()

    # peak loss (normalized space)
    peak_loss = (out.amax(dim=(1, 2, 3)) - y.amax(dim=(1, 2, 3))).abs().mean()

    loss = base + lg * grad_loss + lh * hot_loss + lp * peak_loss

    if loss_type == "boundary":
        lb = float(lam.get("boundary", 0.1))
        dk = int(lam.get("dilate", 3))
        power = x[..., 0].squeeze(-1)  # P=0 power channel, drop Z=1 -> (B, X, Y)
        pdx = (power[:, 1:, :] - power[:, :-1, :]).abs()
        pdy = (power[:, :, 1:] - power[:, :, :-1]).abs()
        pdx = F.pad(pdx, (0, 0, 0, 1))   # (B, X-1, Y) -> (B, X, Y)
        pdy = F.pad(pdy, (0, 1, 0, 0))   # (B, X, Y-1) -> (B, X, Y)
        edge = (pdx + pdy > 1e-6).to(out.dtype)  # (B, X, Y)
        if dk > 1:
            edge = F.max_pool2d(edge.unsqueeze(1), kernel_size=dk, stride=1, padding=dk // 2).squeeze(1)
        m = edge.unsqueeze(-1)  # (B, X, Y, 1) broadcast over Z
        bdiff = (out - y) ** 2
        cnt = m.sum() * out.shape[-1]
        boundary_loss = (bdiff * m).sum() / cnt.clamp_min(1.0)
        loss = loss + lb * boundary_loss
    return loss


class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]
        h = 1.0 / (x.size()[1] - 1.0)
        all_norms = (h ** (self.d / self.p)) * torch.norm(
            x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1
        )
        if self.reduction:
            return torch.mean(all_norms) if self.size_average else torch.sum(all_norms)
        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)
        if self.reduction:
            return torch.mean(diff_norms / y_norms) if self.size_average else torch.sum(diff_norms / y_norms)
        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)
