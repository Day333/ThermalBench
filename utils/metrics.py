"""Evaluation metrics -- copied verbatim from Therm-FM's scOT/evaluate.py.

This is the benchmark's only metric implementation: FNO / U-FNO / SAU-FNO / UNet /
DeepONet / Therm-FM all call this one function, which is what makes the numbers
comparable. Do not write a second "equivalent" implementation.

The six reported metrics map to these dictionary keys:
  RMSE       -> {prefix}/rmse                     per-sample RMSE, then averaged
  MAE        -> {prefix}/mean_absolute_error
  R2         -> {prefix}/r2                       **pooled** over all pixels at once
  MaxAE      -> {prefix}/max_absolute_error       worst single-pixel error in the field
  T_max err  -> {prefix}/max_temperature_error    |predicted peak - true peak|
  Top-MAE    -> {prefix}/topk50_temperature_difference  MAE over the 50 hottest true points

Note that r2 is pooled rather than averaged per sample: fields with near-zero spatial
variance drive the per-sample R2 to -inf and drag the mean negative. The per-sample
version is still stored as {prefix}/r2_per_sample for reference.
"""
import math
from typing import Dict

import numpy as np


def _r2_score(pred: np.ndarray, target: np.ndarray) -> float:
    """Calculate the R² coefficient of determination for a single sample."""
    ss_res = np.sum((target - pred) ** 2)
    target_mean = np.mean(target)
    ss_tot = np.sum((target - target_mean) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1.0 - ss_res / ss_tot)


def _compute_percentage_stats(diff: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """Estimate the MAPE/PAPE (percentage error) of a sample based on the difference and true value."""
    eps = 1e-8
    denom = np.abs(target)
    mask = denom > eps
    if not np.any(mask):
        return {"mape": 0.0, "pape": 0.0}

    ratios = np.zeros_like(diff, dtype=np.float64)
    ratios[mask] = np.abs(diff[mask]) / denom[mask]
    mape = float(np.mean(ratios[mask]) * 100)
    pape = float(np.max(ratios[mask]) * 100)
    return {"mape": mape, "pape": pape}


def _compute_additional_test_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    prefix: str,
    topk: int = 50,
) -> Dict[str, float]:
    """Calculate additional metrics such as RMSE, R², etc. based on the Fourier FNO evaluation method.

    Hotspot-oriented metrics (the quantities this task cares most about):
      - max_temperature_error: |max(pred) - max(label)| per sample, averaged over
        channels and samples. Error of the predicted peak (hotspot) temperature;
        distinct from max_absolute_error, which is the worst single-pixel error
        anywhere in the field.
      - topk{topk}_temperature_difference: mean |pred - label| over the k hottest
        TRUE pixels per sample (k = topk, default 50). Mirrors the hotspot loss in
        model.py (which selects by quantile) but with a fixed point count, so it
        measures how well the hottest region is predicted.
    """
    if preds.size == 0:
        return {}

    if preds.ndim == 3:
        preds = preds[:, np.newaxis, ...]
        labels = labels[:, np.newaxis, ...]
    elif preds.ndim < 3:
        preds = preds.reshape(preds.shape[0], 1, -1)
        labels = labels.reshape(labels.shape[0], 1, -1)

    num_samples = preds.shape[0]
    num_channels = preds.shape[1]

    preds_flat = preds.reshape(num_samples, num_channels, -1)
    labels_flat = labels.reshape(num_samples, num_channels, -1) # [B,L,H*W]

    rmse_total = 0.0
    r2_total = 0.0
    max_abs_total = 0.0
    mean_abs_total = 0.0
    mape_total = 0.0
    pape_total = 0.0
    max_temp_total = 0.0
    topk_total = 0.0
    topk = max(0, int(topk))

    for sample_idx in range(num_samples):
        rmse_sample = 0.0
        r2_sample = 0.0
        max_sample = 0.0
        mean_sample = 0.0
        mape_sample = 0.0
        pape_sample = 0.0
        max_temp_sample = 0.0
        topk_sample = 0.0

        for channel_idx in range(num_channels):
            pred_channel = preds_flat[sample_idx, channel_idx]
            label_channel = labels_flat[sample_idx, channel_idx]
            diff = pred_channel - label_channel

            rmse_sample += math.sqrt(np.mean(diff ** 2))
            max_sample += float(np.max(np.abs(diff)))
            mean_sample += float(np.mean(np.abs(diff)))
            r2_sample += _r2_score(pred_channel, label_channel)

            percentage_stats = _compute_percentage_stats(diff, label_channel)
            mape_sample += percentage_stats["mape"]
            pape_sample += percentage_stats["pape"]

            # Hotspot peak temperature error: |max(pred) - max(label)|.
            max_temp_sample += abs(
                float(np.max(pred_channel)) - float(np.max(label_channel))
            )

            # Top-k temperature difference: mean |pred - label| at the k hottest
            # TRUE points (mirrors the hotspot loss in model.py, fixed k vs quantile).
            k_eff = min(topk, pred_channel.size)
            if k_eff > 0:
                topk_idx = np.argpartition(label_channel, -k_eff)[-k_eff:]
                topk_sample += float(np.mean(np.abs(diff[topk_idx])))

        rmse_total += rmse_sample / num_channels
        r2_total += r2_sample / num_channels
        max_abs_total += max_sample / num_channels
        mean_abs_total += mean_sample / num_channels
        mape_total += mape_sample / num_channels
        pape_total += pape_sample / num_channels
        max_temp_total += max_temp_sample / num_channels
        topk_total += topk_sample / num_channels

    rmse_total /= num_samples
    r2_total /= num_samples
    max_abs_total /= num_samples
    mean_abs_total /= num_samples
    mape_total /= num_samples
    pape_total /= num_samples
    max_temp_total /= num_samples
    topk_total /= num_samples

    # Pooled R^2 over ALL test pixels (standard definition). The per-sample R^2
    # averaged above (r2_total) is misleading when some fields have near-zero
    # spatial variance: r2_sample -> -inf, dragging the mean negative (e.g.
    # case1678). Pooled uses global cross-sample variance as the denominator.
    r2_pooled_ch = []
    for channel_idx in range(num_channels):
        pred_ch = preds_flat[:, channel_idx, :].ravel()
        label_ch = labels_flat[:, channel_idx, :].ravel()
        r2_pooled_ch.append(_r2_score(pred_ch, label_ch))
    r2_pooled = float(np.mean(r2_pooled_ch))

    return {
        f"{prefix}/rmse": float(rmse_total),
        f"{prefix}/r2": float(r2_pooled),
        f"{prefix}/r2_per_sample": float(r2_total),
        f"{prefix}/max_absolute_error": float(max_abs_total),
        f"{prefix}/mean_absolute_error": float(mean_abs_total),
        f"{prefix}/mape_percent": float(mape_total),
        f"{prefix}/pape_percent": float(pape_total),
        f"{prefix}/max_temperature_error": float(max_temp_total),
        f"{prefix}/topk{topk}_temperature_difference": float(topk_total),
    }
