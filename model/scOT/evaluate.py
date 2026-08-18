"""Utility script to evaluate a fine-tuned scOT/Poseidon model on train/val/test splits.
Reference run command:
python scOT/evaluate.py \\                                                 
  --model_path ./checkpoints/poseidon_single/thermal-steady-finetune \
  --config configs/run_thermal_steady.yaml \
  --data_path ./data/thermal_steady/HS_SC_refine2 \
  --output_dir ./eval_outputs \
  --only_test
"""

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, Optional

import h5py
import numpy as np
import torch
import yaml
from transformers import EarlyStoppingCallback  # noqa: F401  # kept for parity

from scOT.metrics import relative_lp_error
from scOT.model import ScOT
from scOT.problems.base import BaseTimeDataset, get_dataset
from scOT.trainer import Trainer, TrainingArguments


SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


def _clean_yaml(config: Dict) -> Dict:
    """Convert a run yaml (with nested value fields) to a flat dict."""

    def _unwrap(value):
        if isinstance(value, dict) and "value" in value:
            return value["value"]
        return value

    return {key: _unwrap(inner) for key, inner in config.items()}


def load_config(path: str) -> Dict:
    """Load a YAML configuration file from the specified path and flatten it into a flat dictionary."""
    with open(path, "r") as fh:
        cfg = yaml.safe_load(fh)
    return _clean_yaml(cfg)


def build_datasets(
    cfg: Dict,
    data_path: str,
    extra_kwargs: Dict,
    splits,
) -> Dict[str, BaseTimeDataset]:
    """Build the corresponding dataset instances based on the configuration and split name."""
    datasets = {}
    for split in splits:
        datasets[split] = get_dataset(
            dataset=cfg["dataset"],
            which=split,
            num_trajectories=cfg["num_trajectories"],
            data_path=data_path,
            **extra_kwargs,
        )
    return datasets


def build_extra_kwargs(cfg: Dict, args) -> Dict:
    """Add optional keyword arguments required for dataset construction based on command line parameters."""
    kwargs = {}
    if "incompressible" in cfg["dataset"] and getattr(args, "just_velocities", False):
        kwargs["just_velocities"] = True
    if "train_ratio" in cfg:  # Use 0.9 for die datasets and 0.8 otherwise.
        kwargs["train_ratio"] = cfg["train_ratio"]

    if args.move_data is not None:
        kwargs["move_to_local_scratch"] = args.move_data

    if args.max_num_train_time_steps is not None:
        kwargs["max_num_time_steps"] = args.max_num_train_time_steps

    if args.train_time_step_size is not None:
        kwargs["time_step_size"] = args.train_time_step_size

    if args.train_small_time_transition:
        kwargs["allowed_time_transitions"] = [1]

    if getattr(args, "stats_json", None) is not None:
        kwargs["stats_json"] = args.stats_json

    return kwargs


def resolve_stats_json_for_eval(cfg: Dict, args) -> Optional[str]:
    """
    Resolve normalization constants path with priority:
    1) explicit --stats_json
    2) <model_path>/normalization_constants.json (for thermal steady/transient)
    3) None
    """
    if getattr(args, "stats_json", None):
        stats_path = os.path.abspath(os.path.expanduser(args.stats_json))
        if not os.path.isfile(stats_path):
            raise FileNotFoundError(f"Normalization JSON not found: {stats_path}")
        return stats_path

    dataset_name = str(cfg.get("dataset", ""))
    if "thermal.steady" in dataset_name or "thermal.transient" in dataset_name:
        candidate = os.path.join(
            os.path.abspath(os.path.expanduser(args.model_path)),
            "normalization_constants.json",
        )
        if os.path.isfile(candidate):
            return candidate
    return None


def compute_metrics_builder(channel_list, printable_channel_description, output_dim):
    """Build the metrics callback required by HuggingFace Trainer."""
    def compute_metrics(eval_preds):
        def get_stats(errors):
            return {
                "median_relative_l1_error": np.median(errors, axis=0),
                "mean_relative_l1_error": np.mean(errors, axis=0),
                "std_relative_l1_error": np.std(errors, axis=0),
                "min_relative_l1_error": np.min(errors, axis=0),
                "max_relative_l1_error": np.max(errors, axis=0),
            }

        stats_per_channel = [
            get_stats(
                relative_lp_error(
                    eval_preds.predictions[:, channel_list[i] : channel_list[i + 1]],
                    eval_preds.label_ids[:, channel_list[i] : channel_list[i + 1]],
                    p=1,
                    return_percent=True,
                )
            )
            for i in range(len(channel_list) - 1)
        ]

        if output_dim == 1:
            return stats_per_channel[0]

        mean_means = np.mean(
            np.array([stats["mean_relative_l1_error"] for stats in stats_per_channel]),
            axis=0,
        )
        mean_medians = np.mean(
            np.array([stats["median_relative_l1_error"] for stats in stats_per_channel]),
            axis=0,
        )
        result = {
            "mean_relative_l1_error": mean_means,
            "mean_over_median_relative_l1_error": mean_medians,
        }
        for idx, stats in enumerate(stats_per_channel):
            prefix = printable_channel_description[idx]
            for key, value in stats.items():
                result[f"{prefix}/{key}"] = value
        return result

    return compute_metrics


def _denormalize_predictions(predictions, dataset):
    """Use the normalization constants of the dataset to restore the predictions and labels to the original physical quantities."""
    if not hasattr(dataset, "constants"):
        return None

    constants = getattr(dataset, "constants", {})
    if not isinstance(constants, dict):
        return None

    if "mean_output" not in constants or "std_output" not in constants:
        return None

    mean_output = constants["mean_output"].detach().cpu().numpy()
    std_output = constants["std_output"].detach().cpu().numpy()

    preds = predictions.predictions * std_output + mean_output
    labels = predictions.label_ids * std_output + mean_output

    return preds, labels


def _compute_denormalized_relative_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    dataset,
    prefix: str,
) -> Dict:
    """Calculate the overall and per-channel relative L1 errors based on the denormalized tensors."""
    metrics: Dict[str, float] = {}

    overall_errors = relative_lp_error(
        preds,
        labels,
        p=1,
        return_percent=False,
    )
    metrics[f"{prefix}/denorm_mean_relative_l1_error"] = float(np.mean(overall_errors))
    metrics[f"{prefix}/denorm_median_relative_l1_error"] = float(
        np.median(overall_errors)
    )

    channel_list = getattr(dataset, "channel_slice_list", None)
    descriptors = getattr(dataset, "printable_channel_description", None)

    if channel_list is not None and descriptors is not None:
        for idx in range(len(channel_list) - 1):
            ch_slice = slice(channel_list[idx], channel_list[idx + 1])
            channel_errors = relative_lp_error(
                preds[:, ch_slice],
                labels[:, ch_slice],
                p=1,
                return_percent=False,
            )
            prefix_name = descriptors[idx]
            metrics[
                f"{prefix}/{prefix_name}/denorm_mean_relative_l1_error"
            ] = float(np.mean(channel_errors))
            metrics[
                f"{prefix}/{prefix_name}/denorm_median_relative_l1_error"
            ] = float(np.median(channel_errors))

    return metrics


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


def _save_predictions_mat(
    preds: np.ndarray,
    path: str,
    dataset_key: str = "data",
) -> None:
    """
    Save the prediction tensor to a .mat file. The input shape is (B, T, L, H, W) or (B, L, H, W).
    """
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as handle:
        handle.create_dataset(dataset_key, data=preds, compression="gzip")


def evaluate_split(
    trainer: Trainer,
    dataset,
    prefix: str,
    compute_denormalized: bool = False,
    save_predictions_path: Optional[str] = None,
    topk: int = 50,
) -> Dict:
    """Execute inference on a single data split and return the evaluation metrics (denormalized)."""
    predictions = trainer.predict(dataset, metric_key_prefix=prefix) # Inference
    metrics = {f"{prefix}/{k[1:]}": v for k, v in predictions.metrics.items()}
    metrics[f"{prefix}/num_samples"] = len(dataset)

    if compute_denormalized:
        denorm = _denormalize_predictions(predictions, dataset)
        if denorm is not None:
            preds_denorm, labels_denorm = denorm # [B,L,H,W] ,[B,L,H,W]
            metrics.update(
                _compute_denormalized_relative_metrics(
                    preds_denorm,
                    labels_denorm,
                    dataset,
                    prefix,
                )
            )
            metrics.update(
                _compute_additional_test_metrics(
                    preds_denorm,
                    labels_denorm,
                    prefix,
                    topk=topk,
                )
            )
            if save_predictions_path is not None:
                _save_predictions_mat(preds_denorm, save_predictions_path)

    return metrics


def main():
    """Command line entry: load the model and data, then execute the evaluation process."""
    parser = argparse.ArgumentParser(description="Evaluate scOT model on train/val/test sets")
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
    parser.add_argument(
        "--only_test",
        action="store_true",
        help="Only evaluate on the test split",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of worker processes for evaluation dataloaders (default: 0).",
    )
    parser.add_argument(
        "--save_predictions_mat",
        type=str,
        default=None, 
        help="If a path is provided, the denormalized predictions will be saved to this .mat file (only for the test set).",
    )
    parser.add_argument(
        "--stats_json",
        type=str,
        default=None,
        help="Path to JSON file containing normalization constants for compatible datasets.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=50,
        help="k for the top-k temperature-difference hotspot metric (default 50). "
        "Mean |pred-label| over the k hottest true pixels per sample.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    resolved_stats_json = resolve_stats_json_for_eval(cfg, args)
    if resolved_stats_json is not None:
        args.stats_json = resolved_stats_json
        print(f"[evaluate] Using normalization constants: {resolved_stats_json}")
    extra_kwargs = build_extra_kwargs(cfg, args)

    splits = ["test"] if args.only_test else ["train", "val", "test"]

    datasets = build_datasets(cfg, args.data_path, extra_kwargs, splits)

    reference_split = "test" if args.only_test else "train"
    reference_dataset = datasets[reference_split]

    printable_channel_description = reference_dataset.printable_channel_description
    channel_list = reference_dataset.channel_slice_list
    output_dim = reference_dataset.output_dim

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

    compute_metrics = compute_metrics_builder(channel_list, printable_channel_description, output_dim)

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
    )

    all_metrics = {}
    for split, dataset in datasets.items():
        compute_denorm = split == "test"
        save_path = (
            args.save_predictions_mat if compute_denorm and args.save_predictions_mat else None
        )
        metrics = evaluate_split( # Evaluation metrics
            trainer,
            dataset,
            prefix=split,
            compute_denormalized=compute_denorm,
            save_predictions_path=save_path,
            topk=args.topk,
        )
        all_metrics.update(metrics)
        print(json.dumps(metrics, indent=2))

    # merge in the training time (written under model_path by scOT/train.py)
    _tt_path = os.path.join(args.model_path, "train_time.json")
    if os.path.isfile(_tt_path):
        try:
            with open(_tt_path) as _f:
                all_metrics.update(json.load(_f))
        except Exception as _e:
            print("warn: failed to read train_time.json:", _e)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "test.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)


if __name__ == "__main__":
    main()
