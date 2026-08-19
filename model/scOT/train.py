"""
This script trains a scOT or pretrains Poseidon on a PDE dataset.
Can be also used for finetuning Poseidon.
Can be used in a single config or sweep setup.
"""

import argparse
import torch
import wandb
import numpy as np
import random
import json
import psutil
import os
import time
import math

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import yaml
import matplotlib.pyplot as plt
import transformers
from accelerate.utils import broadcast_object_list
from scOT.trainer import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
from scOT.model import ScOT, ScOTConfig
from mpl_toolkits.axes_grid1 import ImageGrid
from scOT.problems.base import get_dataset, BaseTimeDataset
from scOT.utils import get_num_parameters, read_cli, get_num_parameters_no_embed
from scOT.metrics import relative_lp_error

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


MODEL_MAP = {
    "T": {
        "num_heads": [3, 6, 12, 24],
        "skip_connections": [2, 2, 2, 0],
        "window_size": 16,
        "patch_size": 4,
        "mlp_ratio": 4.0,
        "depths": [4, 4, 4, 4],
        "embed_dim": 48,
    },
    "S": {
        "num_heads": [3, 6, 12, 24],
        "skip_connections": [2, 2, 2, 0],
        "window_size": 16,
        "patch_size": 4,
        "mlp_ratio": 4.0,
        "depths": [8, 8, 8, 8],
        "embed_dim": 48,
    },
    "B": {
        "num_heads": [3, 6, 12, 24],
        "skip_connections": [2, 2, 2, 0],
        "window_size": 16,
        "patch_size": 4,
        "mlp_ratio": 4.0,
        "depths": [8, 8, 8, 8],
        "embed_dim": 96,
    },
    "L": {
        "num_heads": [3, 6, 12, 24],
        "skip_connections": [2, 2, 2, 0],
        "window_size": 16,
        "patch_size": 4,
        "mlp_ratio": 4.0,
        "depths": [8, 8, 8, 8],
        "embed_dim": 192,
    },
}


def create_predictions_plot(predictions, labels, wandb_prefix):
    assert predictions.shape[0] >= 4

    indices = random.sample(range(predictions.shape[0]), 4)

    predictions = predictions[indices]
    labels = labels[indices]

    fig = plt.figure()
    grid = ImageGrid(
        fig, 111, nrows_ncols=(predictions.shape[1] + labels.shape[1], 4), axes_pad=0.1
    )

    vmax, vmin = (
        max(predictions.max(), labels.max()),
        min(predictions.min(), labels.min()),
    )

    for _i, ax in enumerate(grid):
        i = _i // 4
        j = _i % 4

        if i % 2 == 0:
            ax.imshow(
                predictions[j, i // 2, :, :],
                cmap="gist_ncar",
                origin="lower",
                vmin=vmin,
                vmax=vmax,
            )
        else:
            ax.imshow(
                labels[j, i // 2, :, :],
                cmap="gist_ncar",
                origin="lower",
                vmin=vmin,
                vmax=vmax,
            )

        ax.set_xticks([])
        ax.set_yticks([])

    wandb.log({wandb_prefix + "/predictions": wandb.Image(fig)})
    plt.close()


def _r2_score(pred: np.ndarray, target: np.ndarray) -> float:
    ss_res = np.sum((target - pred) ** 2)
    target_mean = np.mean(target)
    ss_tot = np.sum((target - target_mean) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1.0 - ss_res / ss_tot)


def _compute_percentage_stats(diff: np.ndarray, target: np.ndarray):
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


def _compute_additional_test_metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
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
    labels_flat = labels.reshape(num_samples, num_channels, -1)

    rmse_total = 0.0
    r2_total = 0.0
    max_abs_total = 0.0
    mean_abs_total = 0.0
    mape_total = 0.0
    pape_total = 0.0

    for sample_idx in range(num_samples):
        rmse_sample = 0.0
        r2_sample = 0.0
        max_sample = 0.0
        mean_sample = 0.0
        mape_sample = 0.0
        pape_sample = 0.0

        for channel_idx in range(num_channels):
            pred_channel = preds_flat[sample_idx, channel_idx]
            label_channel = labels_flat[sample_idx, channel_idx]
            diff = pred_channel - label_channel

            rmse_sample += math.sqrt(np.mean(diff**2))
            max_sample += float(np.max(np.abs(diff)))
            mean_sample += float(np.mean(np.abs(diff)))
            r2_sample += _r2_score(pred_channel, label_channel)

            percentage_stats = _compute_percentage_stats(diff, label_channel)
            mape_sample += percentage_stats["mape"]
            pape_sample += percentage_stats["pape"]

        rmse_total += rmse_sample / num_channels
        r2_total += r2_sample / num_channels
        max_abs_total += max_sample / num_channels
        mean_abs_total += mean_sample / num_channels
        mape_total += mape_sample / num_channels
        pape_total += pape_sample / num_channels

    rmse_total /= num_samples
    r2_total /= num_samples
    max_abs_total /= num_samples
    mean_abs_total /= num_samples
    mape_total /= num_samples
    pape_total /= num_samples

    return {
        "rmse": float(rmse_total),
        "r2": float(r2_total),
        "max_absolute_error": float(max_abs_total),
        "mean_absolute_error": float(mean_abs_total),
        "mape_percent": float(mape_total),
        "pape_percent": float(pape_total),
    }


def _reshape_to_btlhw(preds: np.ndarray, labels: np.ndarray, dataset):
    """
    Convert denormalized transient predictions/labels to (B, T, L, H, W).
    """
    if preds.ndim == 5 and labels.ndim == 5:
        return preds, labels

    if preds.ndim == 4 and labels.ndim == 4:
        if hasattr(dataset, "output_times") and hasattr(dataset, "output_layers"):
            t = int(dataset.output_times)
            l = int(dataset.output_layers)
            if preds.shape[1] != t * l or labels.shape[1] != t * l:
                raise ValueError(
                    f"Channel count does not match T*L: channels={preds.shape[1]}, T*L={t * l}"
                )
            b, _, h, w = preds.shape
            preds = preds.reshape(b, t, l, h, w)
            labels = labels.reshape(b, t, l, h, w)
            return preds, labels

        preds = preds[:, np.newaxis, ...]
        labels = labels[:, np.newaxis, ...]
        return preds, labels

    raise ValueError(f"Unsupported prediction dimensions: preds={preds.shape}, labels={labels.shape}")


def _compute_additional_test_metrics_transient(
    preds: np.ndarray,
    labels: np.ndarray,
    dataset,
) -> dict:
    """
    Match evaluate_transient.py:
    - compute metrics per time step
    - then average across time steps
    """
    if preds.size == 0:
        return {}

    preds, labels = _reshape_to_btlhw(preds, labels, dataset)
    bsz, num_times, _, _, _ = preds.shape

    metrics = {}
    per_t_rmse = []
    per_t_r2 = []
    per_t_max_abs = []
    per_t_mean_abs = []
    per_t_mape = []
    per_t_pape = []

    for t_idx in range(num_times):
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

            rmse_total += math.sqrt(np.mean(diff**2))
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

        metrics[f"t{t_idx}/rmse"] = float(rmse_t)
        metrics[f"t{t_idx}/r2"] = float(r2_t)
        metrics[f"t{t_idx}/max_absolute_error"] = float(max_abs_t)
        metrics[f"t{t_idx}/mean_absolute_error"] = float(mean_abs_t)
        metrics[f"t{t_idx}/mape_percent"] = float(mape_t)
        metrics[f"t{t_idx}/pape_percent"] = float(pape_t)

    metrics["rmse"] = float(np.mean(per_t_rmse))
    metrics["r2"] = float(np.mean(per_t_r2))
    metrics["max_absolute_error"] = float(np.mean(per_t_max_abs))
    metrics["mean_absolute_error"] = float(np.mean(per_t_mean_abs))
    metrics["mape_percent"] = float(np.mean(per_t_mape))
    metrics["pape_percent"] = float(np.mean(per_t_pape))
    return metrics


def _print_dataset_summary(name: str, dataset, rank: int):
    """Print dataset size and one-sample tensor shapes for quick sanity check."""
    if rank not in [0, -1]:
        return
    try:
        sample = dataset[0]
        x_shape = (
            tuple(sample["pixel_values"].shape) if "pixel_values" in sample else None
        )
        y_shape = tuple(sample["labels"].shape) if "labels" in sample else None
        start = getattr(dataset, "start", None)
        train_ratio = getattr(dataset, "train_ratio", None)
        n_max = getattr(dataset, "N_max", None)
        n_train = getattr(dataset, "N_train", None)
        n_val = getattr(dataset, "N_val", None)
        n_test = getattr(dataset, "N_test", None)
        print(
            f"[{name}] num_samples={len(dataset)}, start={start}, train_ratio={train_ratio}, "
            f"N_max={n_max}, N_train={n_train}, N_val={n_val}, N_test={n_test}, "
            f"pixel_values_shape={x_shape}, labels_shape={y_shape}"
        )
    except Exception as exc:
        print(
            f"[{name}] num_samples={len(dataset)}, failed to inspect first sample: {exc}"
        )


def _print_dataset_kwargs(name: str, kwargs: dict, rank: int):
    """Print dataset construction kwargs to avoid split/config drift."""
    if rank not in [0, -1]:
        return
    printable = {k: kwargs[k] for k in sorted(kwargs.keys())}
    print(f"[{name}] dataset_kwargs={printable}")


def setup(params, model_map=True):
    config = None
    RANK = int(os.environ.get("LOCAL_RANK", -1))
    CPU_CORES = len(psutil.Process().cpu_affinity())
    CPU_CORES = min(CPU_CORES, 16)
    print(f"Detected {CPU_CORES} CPU cores, will use {CPU_CORES} workers.")
    if params.disable_tqdm:
        transformers.utils.logging.disable_progress_bar()
    if params.json_config:
        config = json.loads(params.config)
    else:
        config = params.config

    if RANK == 0 or RANK == -1:
        run = wandb.init(
            project=params.wandb_project_name, name=params.wandb_run_name, config=config
        )
        config = wandb.config
    else:

        def clean_yaml(config):
            d = {}
            for key, inner_dict in config.items():
                d[key] = inner_dict["value"]
            return d

        if not params.json_config:
            with open(params.config, "r") as s:
                config = yaml.safe_load(s)
            config = clean_yaml(config)
        run = None

    ckpt_dir = "./"
    if RANK == 0 or RANK == -1:
        if run.sweep_id is not None:
            ckpt_dir = (
                params.checkpoint_path
                + "/"
                + run.project
                + "/"
                + run.sweep_id
                + "/"
                + run.name
            )
        else:
            ckpt_dir = params.checkpoint_path + "/" + run.project + "/" + run.name
    if (RANK == 0 or RANK == -1) and not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    ls = broadcast_object_list([ckpt_dir], from_process=0)
    ckpt_dir = ls[0]

    if model_map and (
        type(config["model_name"]) == str and config["model_name"] in MODEL_MAP.keys()
    ):
        config = {**config, **MODEL_MAP[config["model_name"]]}
        if RANK == 0 or RANK == -1:
            wandb.config.update(MODEL_MAP[config["model_name"]], allow_val_change=True)

    return run, config, ckpt_dir, RANK, CPU_CORES


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train scOT or pretrain Poseidon.")
    parser.add_argument("--resume_training", action="store_true")
    parser.add_argument(
        "--finetune_from",
        type=str,
        default=None,
        help="Set this to a str pointing to a HF Hub model checkpoint or a directory with a scOT checkpoint if you want to finetune.",
    )
    parser.add_argument(
        "--replace_embedding_recovery",
        action="store_true",
        help="Set this if you have to replace the embeddings and recovery layers because you are not just using the density, velocity and pressure channels. Only relevant for finetuning.",
    )
    params = read_cli(parser).parse_args()
    run, config, ckpt_dir, RANK, CPU_CORES = setup(params)

    train_eval_set_kwargs = (
        {"just_velocities": True}
        if ("incompressible" in config["dataset"]) and params.just_velocities
        else {}
    )
    if params.move_data is not None:
        train_eval_set_kwargs["move_to_local_scratch"] = params.move_data
    if params.max_num_train_time_steps is not None:
        train_eval_set_kwargs["max_num_time_steps"] = params.max_num_train_time_steps
    if params.train_time_step_size is not None:
        train_eval_set_kwargs["time_step_size"] = params.train_time_step_size
    if params.train_small_time_transition:
        train_eval_set_kwargs["allowed_time_transitions"] = [1]
    if params.stats_json is not None:
        train_eval_set_kwargs["stats_json"] = params.stats_json  # JSON file path
    if "train_ratio" in config:
        train_eval_set_kwargs["train_ratio"] = config["train_ratio"]
    train_dataset = get_dataset(
        dataset=config["dataset"],
        which="train",
        num_trajectories=config["num_trajectories"],
        data_path=params.data_path,
        **train_eval_set_kwargs,
    )

    # For thermal steady/transient datasets without provided stats_json:
    # compute stats from train split via dataset and persist to ckpt dir,
    # then force val/test to reuse the same constants.
    if (
        params.stats_json is None
        and isinstance(config["dataset"], str)
        and (
            "thermal.steady" in config["dataset"]
            or "thermal.transient" in config["dataset"]
        )
        and not isinstance(train_dataset, torch.utils.data.ConcatDataset)
        and hasattr(train_dataset, "constants")
    ):
        stats_save_path = os.path.join(ckpt_dir, "normalization_constants.json")
        constants = train_dataset.constants
        stats_payload = {
            "input": {
                "mean": constants["mean_input"].detach().cpu().view(-1).tolist(),
                "std": constants["std_input"].detach().cpu().view(-1).tolist(),
            },
            "output": {
                "mean": constants["mean_output"].detach().cpu().view(-1).tolist(),
                "std": constants["std_output"].detach().cpu().view(-1).tolist(),
            },
        }
        if RANK == 0 or RANK == -1:
            with open(stats_save_path, "w") as fh:
                json.dump(stats_payload, fh, indent=2)
        ls = broadcast_object_list([stats_save_path], from_process=0)
        train_eval_set_kwargs["stats_json"] = ls[0]

    # Keep one resolved stats_json path for train/val/test consistency.
    resolved_stats_json = train_eval_set_kwargs.get("stats_json", None)
    if (RANK == 0 or RANK == -1) and resolved_stats_json is not None:
        print(f"Using unified normalization constants: {resolved_stats_json}")

    eval_dataset = get_dataset(
        dataset=config["dataset"],
        which="val",
        num_trajectories=config["num_trajectories"],
        data_path=params.data_path,
        **train_eval_set_kwargs,
    )
    test_dataset_preview = get_dataset(
        dataset=config["dataset"],
        which="test",
        num_trajectories=config["num_trajectories"],
        data_path=params.data_path,
        **train_eval_set_kwargs,
    )

    _print_dataset_kwargs("train/val/test_preview", train_eval_set_kwargs, RANK)
    _print_dataset_summary("train", train_dataset, RANK)
    _print_dataset_summary("val", eval_dataset, RANK)
    _print_dataset_summary(
        "test", test_dataset_preview, RANK
    )  # Preview the test split shape to validate dataset construction.

    config["effective_train_set_size"] = len(train_dataset)
    time_involved = isinstance(train_dataset, BaseTimeDataset) or (
        isinstance(train_dataset, torch.utils.data.ConcatDataset)
        and isinstance(train_dataset.datasets[0], BaseTimeDataset)
    )

    if not isinstance(train_dataset, torch.utils.data.ConcatDataset):
        resolution = train_dataset.resolution
        input_dim = train_dataset.input_dim
        output_dim = train_dataset.output_dim
        channel_slice_list = train_dataset.channel_slice_list
        printable_channel_description = train_dataset.printable_channel_description
    else:
        resolution = train_dataset.datasets[0].resolution
        input_dim = train_dataset.datasets[0].input_dim
        output_dim = train_dataset.datasets[0].output_dim
        channel_slice_list = train_dataset.datasets[0].channel_slice_list
        printable_channel_description = train_dataset.datasets[
            0
        ].printable_channel_description

    model_config = (
        ScOTConfig(
            image_size=resolution,
            patch_size=config["patch_size"],
            num_channels=input_dim,
            num_out_channels=output_dim,
            embed_dim=config["embed_dim"],
            depths=config["depths"],
            num_heads=config["num_heads"],
            skip_connections=config["skip_connections"],
            window_size=config["window_size"],
            mlp_ratio=config["mlp_ratio"],
            qkv_bias=True,
            hidden_dropout_prob=0.0,  # default
            attention_probs_dropout_prob=0.0,  # default
            drop_path_rate=0.0,
            hidden_act="gelu",
            use_absolute_embeddings=False,
            initializer_range=0.02,
            layer_norm_eps=1e-5,
            p=config.get("p", 2),  # loss mode: 1 l1 / 2 l2(MSE) / 3 hotspot / 4 pinn / 5 thermal-combined
            channel_slice_list_normalized_loss=channel_slice_list,
            residual_model="convnext",
            use_conditioning=time_involved,
            learn_residual=False,
            # thermal-combined loss (p=5) weights; inert unless config sets p==5
            thermal_grad_lambda=config.get("thermal_grad_lambda", 0.1),
            thermal_hot_lambda=config.get("thermal_hot_lambda", 0.5),
            thermal_peak_lambda=config.get("thermal_peak_lambda", 0.1),
            thermal_hot_alpha=config.get("thermal_hot_alpha", 0.9),
            thermal_boundary_lambda=config.get("thermal_boundary_lambda", 0.0),
            thermal_boundary_dilate=config.get("thermal_boundary_dilate", 3),
            thermal_warmup_epochs=config.get("thermal_warmup_epochs", 0),
        )
        if params.finetune_from is None or params.replace_embedding_recovery
        else None
    )

    train_config = TrainingArguments(
        output_dir=ckpt_dir,
        overwrite_output_dir=True,  #! OVERWRITE THIS DIRECTORY IN CASE, also for resuming training
        evaluation_strategy="epoch",
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        eval_accumulation_steps=16,
        max_grad_norm=config["max_grad_norm"],
        num_train_epochs=config["num_epochs"],
        optim="adamw_torch",
        learning_rate=config["lr"],
        learning_rate_embedding_recovery=(
            None
            if (params.finetune_from is None or "lr_embedding_recovery" not in config)
            else config["lr_embedding_recovery"]
        ),
        learning_rate_time_embedding=(
            None
            if (params.finetune_from is None or "lr_time_embedding" not in config)
            else config["lr_time_embedding"]
        ),
        weight_decay=config["weight_decay"],
        adam_beta1=0.9,  # default
        adam_beta2=0.999,  # default
        adam_epsilon=1e-8,  # default
        lr_scheduler_type=config["lr_scheduler"],
        warmup_ratio=config["warmup_ratio"],
        log_level="passive",
        logging_strategy="steps",
        logging_steps=5,
        logging_nan_inf_filter=False,
        save_strategy="epoch",
        save_total_limit=1,
        seed=SEED,
        fp16=False,
        dataloader_num_workers=CPU_CORES,
        load_best_model_at_end=(os.environ.get("TFM_LAST_EPOCH", "0") not in ("1", "true", "True")),
        metric_for_best_model="loss",
        greater_is_better=False,
        dataloader_pin_memory=True,
        gradient_checkpointing=False,
        auto_find_batch_size=False,
        full_determinism=False,
        torch_compile=False,
        report_to="wandb",
        run_name=params.wandb_run_name,
    )

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=config["early_stopping_patience"],
        early_stopping_threshold=0.0,  # set no threshold for now
    )

    if params.finetune_from is not None:
        model = ScOT.from_pretrained(
            params.finetune_from, config=model_config, ignore_mismatched_sizes=True
        )
    else:
        model = ScOT(model_config)
    num_params = get_num_parameters(model)
    config["num_params"] = num_params
    num_params_no_embed = get_num_parameters_no_embed(model)
    config["num_params_wout_embed"] = num_params_no_embed
    if RANK == 0 or RANK == -1:
        print(f"Model size: {num_params}")
        print(f"Model size without embeddings: {num_params_no_embed}")

    def compute_metrics(eval_preds):
        errors = relative_lp_error(
            eval_preds.predictions,
            eval_preds.label_ids,
            p=1,
            return_percent=True,
        )
        return {
            "median_relative_l1_error": np.median(errors, axis=0),
            "mean_relative_l1_error": np.mean(errors, axis=0),
            "std_relative_l1_error": np.std(errors, axis=0),
            "min_relative_l1_error": np.min(errors, axis=0),
            "max_relative_l1_error": np.max(errors, axis=0),
        }

    trainer = Trainer(
        model=model,
        args=train_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # compute_metrics=compute_metrics,  # Disabled to speed up training.
        callbacks=([] if os.environ.get("TFM_LAST_EPOCH", "0") in ("1", "true", "True") else [early_stopping]),
    )

    training_start = time.time()
    trainer.train(resume_from_checkpoint=params.resume_training)
    training_duration = time.time() - training_start
    if RANK == 0 or RANK == -1:
        print(f"Training finished in {training_duration:.2f} seconds")
        if run is not None:
            wandb.log({"train/total_time_seconds": training_duration})
    trainer.save_model(train_config.output_dir)
    if RANK == 0 or RANK == -1:
        # persist the training time so evaluate.py can fold it into test.json
        # (logging alone would be lost with the log)
        try:
            import json as _json
            # num_epochs is a YAML/W&B configuration value, not a CLI parameter.
            # Reading it from `params` silently recorded epochs=0 and made the
            # per-epoch time equal to the whole training time.
            _n_ep = int(config["num_epochs"])
            _tt = {
                "train_time_s": round(training_duration, 1),
                "train_time_per_epoch_s": round(training_duration / max(_n_ep, 1), 2),
                "epochs": _n_ep,
                "params": sum(p.numel() for p in model.parameters()),
            }
            with open(os.path.join(train_config.output_dir, "train_time.json"), "w") as _f:
                _json.dump(_tt, _f, indent=2)
        except Exception as _e:
            print("warn: failed to write train_time.json:", _e)

    if (RANK == 0 or RANK == -1) and params.push_to_hf_hub is not None:
        model.push_to_hub(params.push_to_hf_hub)

    do_test = (
        True
        if params.max_num_train_time_steps is None
        and params.train_time_step_size is None
        and not params.train_small_time_transition
        and not ".time" in config["dataset"]
        else False
    )
    if do_test:
        print("Testing...")
        # Reuse the exact same dataset kwargs as train/val construction to
        # avoid split/normalization drift between train.py and evaluate.py.
        test_set_kwargs = dict(train_eval_set_kwargs)
        out_test_set_kwargs = dict(train_eval_set_kwargs)
        if time_involved:
            test_set_kwargs = {
                **test_set_kwargs,
                "max_num_time_steps": 1,
                "time_step_size": 14,
                "allowed_time_transitions": [1],
            }
            out_test_set_kwargs = {
                **out_test_set_kwargs,
                "max_num_time_steps": 1,
                "time_step_size": 20,
                "allowed_time_transitions": [1],
            }
        if "RayleighTaylor" in config["dataset"]:
            test_set_kwargs = {
                **test_set_kwargs,
                "max_num_time_steps": 1,
                "time_step_size": 7,
                "allowed_time_transitions": [1],
            }
            out_test_set_kwargs = {
                **out_test_set_kwargs,
                "max_num_time_steps": 1,
                "time_step_size": 10,
                "allowed_time_transitions": [1],
            }

        _print_dataset_kwargs("test", test_set_kwargs, RANK)
        _print_dataset_kwargs("test_out_dist", out_test_set_kwargs, RANK)
        test_dataset = get_dataset(
            dataset=config["dataset"],
            which="test",
            num_trajectories=config["num_trajectories"],
            data_path=params.data_path,
            **test_set_kwargs,
        )
        _print_dataset_summary("test(actual)", test_dataset, RANK)
        try:
            out_dist_test_dataset = get_dataset(
                dataset=config["dataset"] + ".out",
                which="test",
                num_trajectories=config["num_trajectories"],
                data_path=params.data_path,
                **out_test_set_kwargs,
            )
            _print_dataset_summary("test_out_dist(actual)", out_dist_test_dataset, RANK)
        except:
            out_dist_test_dataset = None
        predictions = trainer.predict(test_dataset, metric_key_prefix="")
        if RANK == 0 or RANK == -1:
            metrics = {}
            for key, value in predictions.metrics.items():
                metrics["test/" + key[1:]] = value
            wandb.log(metrics)
            create_predictions_plot(
                predictions.predictions,
                predictions.label_ids,
                wandb_prefix="test",
            )

            # Use evaluate.py semantics for steady datasets.
            # Use evaluate_transient.py semantics for transient datasets.
            # Other datasets skip these additional metrics.
            additional_metrics = {}
            if (
                hasattr(test_dataset, "constants")
                and isinstance(test_dataset.constants, dict)
                and "mean_output" in test_dataset.constants
                and "std_output" in test_dataset.constants
            ):
                mean_output = (
                    test_dataset.constants["mean_output"].detach().cpu().numpy()
                )
                std_output = test_dataset.constants["std_output"].detach().cpu().numpy()
                preds_denorm = predictions.predictions * std_output + mean_output
                labels_denorm = predictions.label_ids * std_output + mean_output
                dataset_name = str(config.get("dataset", ""))
                if "thermal.steady" in dataset_name:
                    additional_metrics = _compute_additional_test_metrics(
                        preds_denorm, labels_denorm
                    )
                elif "thermal.transient" in dataset_name:
                    additional_metrics = _compute_additional_test_metrics_transient(
                        preds_denorm, labels_denorm, test_dataset
                    )

                if additional_metrics:
                    wandb.log({f"test/{k}": v for k, v in additional_metrics.items()})

                    additional_metrics_path = os.path.join(
                        ckpt_dir, "test_additional_metrics.json"
                    )
                    with open(additional_metrics_path, "w") as fh:
                        json.dump(additional_metrics, fh, indent=2)
                    print(f"Saved additional test metrics to {additional_metrics_path}")

        # evaluate on out-of-distribution test set
        if out_dist_test_dataset is not None:
            predictions = trainer.predict(out_dist_test_dataset, metric_key_prefix="")
            if RANK == 0 or RANK == -1:
                metrics = {}
                for key, value in predictions.metrics.items():
                    metrics["test_out_dist/" + key[1:]] = value
                wandb.log(metrics)
                create_predictions_plot(
                    predictions.predictions,
                    predictions.label_ids,
                    wandb_prefix="test_out_dist",
                )

        if time_involved and (test_set_kwargs["time_step_size"] // 2 > 0):
            trainer.set_ar_steps(test_set_kwargs["time_step_size"] // 2)
            predictions = trainer.predict(test_dataset, metric_key_prefix="")
            if RANK == 0 or RANK == -1:
                metrics = {}
                for key, value in predictions.metrics.items():
                    metrics["test/ar/" + key[1:]] = value
                wandb.log(metrics)
                create_predictions_plot(
                    predictions.predictions,
                    predictions.label_ids,
                    wandb_prefix="test/ar",
                )

            # evaluate on out-of-distribution test set
            if out_dist_test_dataset is not None:
                trainer.set_ar_steps(out_test_set_kwargs["time_step_size"] // 2)
                predictions = trainer.predict(
                    out_dist_test_dataset, metric_key_prefix=""
                )
                if RANK == 0 or RANK == -1:
                    metrics = {}
                    for key, value in predictions.metrics.items():
                        metrics["test_out_dist/ar/" + key[1:]] = value
                    wandb.log(metrics)
                    create_predictions_plot(
                        predictions.predictions,
                        predictions.label_ids,
                        wandb_prefix="test_out_dist/ar",
                    )
