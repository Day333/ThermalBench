"""Evaluate scOT/Poseidon on transient thermal datasets.

Compared with `scOT/evaluate.py`, this script changes additional test metrics:
- predictions/labels are interpreted as transient tensors (B, T, L, H, W)
- metrics are computed per time step t
- final reported metrics are averaged over time steps
"""

import argparse
import json
import math
import os
import random
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from scOT.evaluate import (
    build_datasets,
    build_extra_kwargs,
    compute_metrics_builder,
    load_config,
    resolve_stats_json_for_eval,
    _compute_denormalized_relative_metrics,
    _denormalize_predictions,
    _save_predictions_mat,
)
from scOT.model import ScOT
from scOT.trainer import Trainer, TrainingArguments


SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


def _r2_score(pred: np.ndarray, target: np.ndarray) -> float:
    ss_res = np.sum((target - pred) ** 2)
    target_mean = np.mean(target)
    ss_tot = np.sum((target - target_mean) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1.0 - ss_res / ss_tot)


def _compute_percentage_stats(diff: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    eps = 1e-8
    denom = np.abs(target)
    mask = denom > eps
    if not np.any(mask):
        return {"mape": 0.0, "pape": 0.0}
    ratios = np.zeros_like(diff, dtype=np.float64)
    ratios[mask] = np.abs(diff[mask]) / denom[mask]
    return {
        "mape": float(np.mean(ratios[mask]) * 100),
        "pape": float(np.max(ratios[mask]) * 100),
    }


def _reshape_to_btlhw(preds: np.ndarray, labels: np.ndarray, dataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert denormalized predictions/labels to (B, T, L, H, W) if possible.
    For transient thermal dataset, model outputs are usually (B, T*L, H, W).
    """
    if preds.ndim == 5 and labels.ndim == 5:
        return preds, labels

    if preds.ndim == 4 and labels.ndim == 4:
        if hasattr(dataset, "output_times") and hasattr(dataset, "output_layers"):
            t = int(dataset.output_times)
            l = int(dataset.output_layers)
            if preds.shape[1] != t * l or labels.shape[1] != t * l:
                raise ValueError(
                    f"Channel count does not match T*L: channels={preds.shape[1]}, T*L={t*l}"
                )
            b, _, h, w = preds.shape
            preds = preds.reshape(b, t, l, h, w)
            labels = labels.reshape(b, t, l, h, w)
            return preds, labels

        # fallback: treat channel dim as L with single time step
        preds = preds[:, np.newaxis, ...]
        labels = labels[:, np.newaxis, ...]
        return preds, labels

    raise ValueError(f"Unsupported prediction dimensions: preds={preds.shape}, labels={labels.shape}")


def _compute_additional_test_metrics_transient(
    preds: np.ndarray,
    labels: np.ndarray,
    prefix: str,
    dataset,
) -> Dict[str, float]:
    """
    Compute RMSE/R2/MAE/MAPE/PAPE for transient outputs:
    - first compute metrics per time step
    - then average across time steps for final metrics
    """
    if preds.size == 0:
        return {}

    preds, labels = _reshape_to_btlhw(preds, labels, dataset)  # shape: (B, T, L, H, W)
    bsz, num_times, _, _, _ = preds.shape

    metrics: Dict[str, float] = {}
    per_t_rmse = []
    per_t_r2 = []
    per_t_max_abs = []
    per_t_mean_abs = []
    per_t_mape = []
    per_t_pape = []

    for t_idx in range(num_times):
        # For each time step, flatten all (L, H, W) values.
        # Metrics aggregate over batch and time, unlike steady metrics which average over layers separately.
        pred_t = preds[:, t_idx].reshape(bsz, -1)
        label_t = labels[:, t_idx].reshape(bsz, -1)

        rmse_total = 0.0
        r2_total = 0.0
        max_abs_total = 0.0
        mean_abs_total = 0.0
        mape_total = 0.0
        pape_total = 0.0

        for sample_idx in range(bsz):
            pred_sample = pred_t[sample_idx]
            label_sample = label_t[sample_idx]
            diff = pred_sample - label_sample

            rmse_total += math.sqrt(np.mean(diff ** 2))
            r2_total += _r2_score(pred_sample, label_sample)
            max_abs_total += float(np.max(np.abs(diff)))
            mean_abs_total += float(np.mean(np.abs(diff)))
            percentage_stats = _compute_percentage_stats(diff, label_sample)
            mape_total += percentage_stats["mape"]
            pape_total += percentage_stats["pape"]

        rmse_t = rmse_total / bsz
        r2_t = r2_total / bsz
        max_abs_t = max_abs_total / bsz
        mean_abs_t = mean_abs_total / bsz
        mape_t = mape_total / bsz
        pape_t = pape_total / bsz

        per_t_rmse.append(rmse_t)
        per_t_r2.append(r2_t)
        per_t_max_abs.append(max_abs_t)
        per_t_mean_abs.append(mean_abs_t)
        per_t_mape.append(mape_t)
        per_t_pape.append(pape_t)

        metrics[f"{prefix}/t{t_idx}/rmse"] = float(rmse_t)
        metrics[f"{prefix}/t{t_idx}/r2"] = float(r2_t)
        metrics[f"{prefix}/t{t_idx}/max_absolute_error"] = float(max_abs_t)
        metrics[f"{prefix}/t{t_idx}/mean_absolute_error"] = float(mean_abs_t)
        metrics[f"{prefix}/t{t_idx}/mape_percent"] = float(mape_t)
        metrics[f"{prefix}/t{t_idx}/pape_percent"] = float(pape_t)

    metrics[f"{prefix}/rmse"] = float(np.mean(per_t_rmse))
    metrics[f"{prefix}/r2"] = float(np.mean(per_t_r2))
    metrics[f"{prefix}/max_absolute_error"] = float(np.mean(per_t_max_abs))
    metrics[f"{prefix}/mean_absolute_error"] = float(np.mean(per_t_mean_abs))
    metrics[f"{prefix}/mape_percent"] = float(np.mean(per_t_mape))
    metrics[f"{prefix}/pape_percent"] = float(np.mean(per_t_pape))
    return metrics


def evaluate_split(
    trainer: Trainer,
    dataset,
    prefix: str,
    compute_denormalized: bool = False,
    save_predictions_path: Optional[str] = None,
) -> Dict:
    predictions = trainer.predict(dataset, metric_key_prefix=prefix)
    metrics = {}
    for key, value in predictions.metrics.items():
        if key.startswith(f"{prefix}_"):
            metric_name = key[len(prefix) + 1 :]
        else:
            metric_name = key.lstrip("_/")
        metrics[f"{prefix}/{metric_name}"] = value
    metrics[f"{prefix}/num_samples"] = len(dataset)

    if compute_denormalized:
        denorm = _denormalize_predictions(predictions, dataset)
        if denorm is not None:
            preds_denorm, labels_denorm = denorm
            metrics.update(
                _compute_denormalized_relative_metrics(
                    preds_denorm,
                    labels_denorm,
                    dataset,
                    prefix,
                )
            )
            metrics.update(
                _compute_additional_test_metrics_transient(
                    preds_denorm,
                    labels_denorm,
                    prefix,
                    dataset,
                )
            )
            if save_predictions_path is not None:
                preds_btlhw, _ = _reshape_to_btlhw(preds_denorm, labels_denorm, dataset)
                _save_predictions_mat(preds_btlhw, save_predictions_path)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate scOT model on transient thermal datasets")
    parser.add_argument("--model_path", required=True, help="Path to fine-tuned model directory")
    parser.add_argument("--config", required=True, help="Path to run configuration YAML")
    parser.add_argument("--data_path", required=True, help="Root path to data")
    parser.add_argument("--output_dir", default="./eval_outputs", help="Directory to store evaluation artifacts")
    parser.add_argument("--per_device_batch_size", type=int, default=None, help="Override evaluation batch size")
    parser.add_argument("--max_num_train_time_steps", type=int, default=None)
    parser.add_argument("--train_time_step_size", type=int, default=None)
    parser.add_argument("--train_small_time_transition", action="store_true")
    parser.add_argument("--move_data", default=None)
    parser.add_argument("--just_velocities", action="store_true")
    parser.add_argument("--only_test", action="store_true", help="Only evaluate on the test split")
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument(
        "--save_predictions_mat",
        type=str,
        default=None,
        help="If set, denormalized predictions are saved to this .mat file (test split only).",
    )
    parser.add_argument(
        "--stats_json",
        type=str,
        default=None,
        help="Path to JSON file containing normalization constants.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    resolved_stats_json = resolve_stats_json_for_eval(cfg, args)
    if resolved_stats_json is not None:
        args.stats_json = resolved_stats_json
        print(f"[evaluate_transient] Using normalization constants: {resolved_stats_json}")
    extra_kwargs = build_extra_kwargs(cfg, args)
    splits = ["test"] if args.only_test else ["train", "val", "test"]
    datasets = build_datasets(cfg, args.data_path, extra_kwargs, splits)

    reference_split = "test" if args.only_test else "train"
    reference_dataset = datasets[reference_split]
    compute_metrics = compute_metrics_builder(
        reference_dataset.channel_slice_list,
        reference_dataset.printable_channel_description,
        reference_dataset.output_dim,
    )

    model = ScOT.from_pretrained(args.model_path)
    model.eval()

    per_device_bs = args.per_device_batch_size or cfg.get("batch_size", 16)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_eval_batch_size=per_device_bs,
        evaluation_strategy="no",
        logging_strategy="no",
        save_strategy="no",
        report_to=[],
        dataloader_pin_memory=True,
        dataloader_num_workers=args.dataloader_num_workers,
    )

    trainer = Trainer(model=model, args=training_args, compute_metrics=compute_metrics)

    all_metrics = {}
    for split, dataset in datasets.items():
        compute_denorm = split == "test"
        save_path = args.save_predictions_mat if compute_denorm and args.save_predictions_mat else None
        metrics = evaluate_split(
            trainer,
            dataset,
            prefix=split,
            compute_denormalized=compute_denorm,
            save_predictions_path=save_path,
        )
        all_metrics.update(metrics)
        print(json.dumps(metrics, indent=2))

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "test.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)


if __name__ == "__main__":
    main()
