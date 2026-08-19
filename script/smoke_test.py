#!/usr/bin/env python3
"""Dataset-free environment and interface smoke test for IC-ThermBench."""

from __future__ import annotations

import platform
import sys
from pathlib import Path


# `python script/smoke_test.py` puts script/ rather than the repository root on
# sys.path.  Add the root explicitly so the documented command works from any cwd.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def fail(message: str) -> None:
    print(f"[FAIL] {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        fail(f"missing dependency: {exc.name}; create the environment from environment.yml")

    from exp.exp_basic import MODEL_ZOO, OPERATOR_MODELS, build_model
    from exp.exp_operator import forward_norm
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

    # Exercise the shared model adapter at every released input-channel count.  This
    # catches hard-coded P values and layout drift without downloading a dataset.
    for channels in (3, 4, 7):
        x = torch.randn(1, 64, 64, 1, channels)
        for name in OPERATOR_MODELS:
            torch.manual_seed(0)
            model = build_model(name, P=channels, Z=1, G=64).cpu().eval()
            with torch.no_grad():
                prediction = forward_norm(model, name, x)
            expected = (1, 64, 64, 1)
            if tuple(prediction.shape) != expected:
                fail(
                    f"{name} with P={channels} returned {tuple(prediction.shape)}, "
                    f"expected {expected}"
                )
            del model, prediction

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
    print(f"  Models   {len(MODEL_ZOO)} registered; operator P=3/4/7 forwards passed")
    print("  Metrics  shared six-metric contract available")


if __name__ == "__main__":
    main()
