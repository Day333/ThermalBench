#!/usr/bin/env python3
"""Dataset-free environment and interface smoke test for IC-ThermBench."""

from __future__ import annotations

import platform
import sys


def fail(message: str) -> None:
    print(f"[FAIL] {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        fail(f"missing dependency: {exc.name}; create the environment from environment.yml")

    from exp.exp_basic import MODEL_ZOO, build_model
    from utils.metrics import _compute_additional_test_metrics

    expected_models = {
        "FNO",
        "UFNO",
        "SAUFNO",
        "UNet",
        "DeepONet",
        "ThermFM-T",
        "ThermFM-B",
        "ThermFM-L",
    }
    if set(MODEL_ZOO) != expected_models:
        fail(f"unexpected model registry: {sorted(MODEL_ZOO)}")

    torch.manual_seed(0)
    model = build_model("UNet", P=3, Z=1, G=64).cpu().eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        prediction = model(x)
    if tuple(prediction.shape) != (1, 1, 64, 64):
        fail(f"U-Net returned {tuple(prediction.shape)}, expected (1, 1, 64, 64)")

    labels = np.full((2, 1, 8, 8), 300.0, dtype=np.float32)
    predictions = labels + 1.0
    metrics = _compute_additional_test_metrics(predictions, labels, "smoke")
    expected_keys = {
        "smoke/rmse",
        "smoke/mean_absolute_error",
        "smoke/r2",
        "smoke/max_absolute_error",
        "smoke/max_temperature_error",
        "smoke/topk50_temperature_difference",
    }
    if not expected_keys.issubset(metrics):
        fail(f"metric contract is missing: {sorted(expected_keys - set(metrics))}")
    if abs(metrics["smoke/rmse"] - 1.0) > 1e-6:
        fail(f"metric sanity check returned RMSE={metrics['smoke/rmse']}, expected 1.0")

    print("IC-ThermBench smoke test passed")
    print(f"  Python   {platform.python_version()}")
    print(f"  PyTorch  {torch.__version__}")
    print(f"  CUDA     {'available' if torch.cuda.is_available() else 'not available (CPU smoke only)'}")
    print(f"  Models   {len(MODEL_ZOO)} registered")
    print("  Metrics  shared six-metric contract available")


if __name__ == "__main__":
    main()
