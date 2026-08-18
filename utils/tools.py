"""Checkpoint I/O and small helpers."""
import json
import os

import numpy as np
import torch


def set_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)


def save_checkpoint(path, model, x_norm, y_norm, meta):
    """Store a state_dict rather than whole objects.

    The legacy repos wrote `torch.save([x_norm, model, y_norm])`, which pickles class
    paths and therefore stops loading as soon as the directory layout changes -- that
    is exactly why utils/compat.py exists. This writes tensors only.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "x_mean": x_norm.mean.detach().cpu(), "x_std": x_norm.std.detach().cpu(),
        "y_mean": y_norm.mean.detach().cpu(), "y_std": y_norm.std.detach().cpu(),
        "meta": meta,
    }, path)


class _Norm:
    """A normalizer rebuilt from the mean/std stored in a checkpoint; same interface
    as layers.normalize."""

    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    def forward(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean

    __call__ = forward


def load_checkpoint(path, model=None):
    """Handle both formats, returning (x_norm, model, y_norm).

    new format: {"state_dict", "x_mean", ...}       -- requires an instantiated model
    old format: [x_normalizer, model, y_normalizer] -- whole objects, `model` ignored
    """
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj:
        if model is None:
            raise ValueError("the new checkpoint format needs an instantiated model")
        model.load_state_dict(obj["state_dict"])
        return (_Norm(obj["x_mean"], obj["x_std"]), model,
                _Norm(obj["y_mean"], obj["y_std"]))
    if isinstance(obj, (list, tuple)) and len(obj) == 3:
        return obj[0], obj[1], obj[2]
    raise ValueError(f"unrecognized checkpoint format: {type(obj)}")


def dump_metrics(path, metrics):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[metrics] -> {path}", flush=True)


SIX = ["rmse", "mean_absolute_error", "r2", "max_absolute_error",
       "max_temperature_error", "topk50_temperature_difference"]
SIX_LABEL = ["RMSE", "MAE", "R2", "MaxAE", "T_max_err", "Top-MAE"]


def print_six(metrics, prefix):
    """Print the six metrics the benchmark reports."""
    print("  " + "  ".join(f"{lab}={metrics[f'{prefix}/{k}']:.4f}"
                           for lab, k in zip(SIX_LABEL, SIX)), flush=True)
