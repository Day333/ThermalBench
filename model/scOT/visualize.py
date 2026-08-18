"""Plot ground truth / prediction / error panels for a trained scOT model.

Vendored from Therm-FM. The original docstring was a usage note pinned to that repo's
paths, which do not exist here; the equivalent command in this project is:

    python model/scOT/visualize.py \
      --model_path checkpoints/level2_ThermFM-B \
      --config model/thermfm_configs/run_level2_steady_B.yaml \
      --data_path datasets/level2_steady \
      --output vis_outputs/level2_B.png \
      --num_samples 3

Options: --num_samples N draws N random samples (default 3); --sample_indices "5,42,77"
names specific sample indices and overrides --num_samples.

This is a side utility -- the benchmark's train/test/finetune paths do not use it.
"""


import argparse
import os
from types import SimpleNamespace

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ.setdefault("MPLBACKEND", "Agg")  # headless rendering on the server

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scOT.evaluate import (
    build_datasets,
    build_extra_kwargs,
    load_config,
    resolve_stats_json_for_eval,
    _denormalize_predictions,
)
from scOT.model import ScOT
from scOT.trainer import Trainer, TrainingArguments


SEED = 0
np.random.seed(SEED)


def _ensure_4d(array: np.ndarray) -> np.ndarray:
    """Normalize a prediction/label array to shape [N, L, H, W]."""
    array = np.asarray(array)
    if array.ndim == 3:  # [N, H, W] -> single layer
        array = array[:, np.newaxis, ...]
    elif array.ndim != 4:
        raise ValueError(f"Expected a 3D/4D array, got shape {array.shape}")
    return array


def _parse_indices(raw: str, num_available: int, num_samples: int):
    """Return explicit sample indices from a comma list, else a random selection."""
    if raw:
        idx = [int(x) for x in raw.split(",") if x.strip() != ""]
        if any(not (0 <= i < num_available) for i in idx):
            raise ValueError(
                f"sample_indices out of range [0, {num_available}): {idx}"
            )
        return idx
    k = min(num_samples, num_available)
    return list(np.random.choice(num_available, size=k, replace=False))


def make_grid_figure(
    preds_sel: np.ndarray,
    labels_sel: np.ndarray,
    sample_indices,
    *,
    signed: bool,
    cmap: str,
    error_cmap: str,
    unit: str,
    title: str,
):
    """Build the [Ground Truth | Prediction | Error] grid for the selected samples.

    preds_sel / labels_sel: arrays of shape [n, L, H, W] in physical units.
    """
    preds_sel = _ensure_4d(preds_sel)
    labels_sel = _ensure_4d(labels_sel)
    n, num_layers = preds_sel.shape[0], preds_sel.shape[1]
    num_rows = n * num_layers

    col_titles = ["Ground Truth", "Prediction", "Error" if signed else "Absolute Error"]
    fig, axes = plt.subplots(
        num_rows,
        3,
        figsize=(11.5, 3.1 * num_rows),
        squeeze=False,
        constrained_layout=True,
    )

    for s in range(n):
        for layer in range(num_layers):
            row = s * num_layers + layer

            gt = labels_sel[s, layer]
            pr = preds_sel[s, layer]
            residual = pr - gt
            err = residual if signed else np.abs(residual)

            # Ground truth and prediction share a scale so they are comparable.
            vmin = float(min(gt.min(), pr.min()))
            vmax = float(max(gt.max(), pr.max()))

            im_gt = axes[row, 0].imshow(
                gt, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax
            )
            im_pr = axes[row, 1].imshow(
                pr, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax
            )
            if signed:
                emax = float(max(abs(err.min()), abs(err.max())))
                im_err = axes[row, 2].imshow(
                    err, origin="lower", cmap=error_cmap, vmin=-emax, vmax=emax
                )
            else:
                im_err = axes[row, 2].imshow(
                    err, origin="lower", cmap=error_cmap, vmin=0.0, vmax=float(err.max())
                )

            for im, ax in zip((im_gt, im_pr, im_err), axes[row]):
                ax.set_xticks([])
                ax.set_yticks([])
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(unit)

            # Row label on the leftmost panel.
            row_label = f"sample {sample_indices[s]}"
            if num_layers > 1:
                row_label += f"\nlayer {layer}"
            axes[row, 0].set_ylabel(row_label, fontsize=9)

            # Column titles on the top row only.
            if row == 0:
                for col, name in enumerate(col_titles):
                    axes[0, col].set_title(name, fontsize=12)

            # Annotate the worst error on the error panel.
            worst = float(np.abs(residual).max())
            axes[row, 2].text(
                0.02,
                0.02,
                f"max |Δ| {worst:.2f} {unit}",
                transform=axes[row, 2].transAxes,
                fontsize=8,
                color="black",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
            )

    fig.suptitle(title, fontsize=13)
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a trained model: Ground Truth | Prediction | Error grid."
    )
    parser.add_argument("--model_path", required=True, help="Path to fine-tuned model directory")
    parser.add_argument("--config", required=True, help="Path to run configuration YAML")
    parser.add_argument("--data_path", required=True, help="Root path to data")
    parser.add_argument(
        "--output",
        default="vis_outputs/vis.png",
        help="Output PNG path for the combined figure.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Number of test samples to visualize when --sample_indices is not given.",
    )
    parser.add_argument(
        "--sample_indices",
        type=str,
        default=None,
        help="Comma-separated explicit sample indices, e.g. '1,7,42'. Overrides --num_samples.",
    )
    parser.add_argument(
        "--stats_json",
        type=str,
        default=None,
        help="Path to normalization constants JSON (auto-resolved from model_path if omitted).",
    )
    parser.add_argument(
        "--per_device_batch_size",
        type=int,
        default=16,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--signed",
        action="store_true",
        help="Show signed residual (pred - true) with a diverging colormap "
        "instead of the default absolute error.",
    )
    parser.add_argument("--cmap", default="jet", help="Colormap for the GT/prediction panels.")
    parser.add_argument(
        "--error_cmap",
        default=None,
        help="Colormap for the error panel (default: same as --cmap for absolute "
        "error, RdBu_r when --signed).",
    )
    parser.add_argument(
        "--separate",
        action="store_true",
        help="Save one PNG per selected sample in addition to the combined figure.",
    )
    parser.add_argument(
        "--save_npy",
        type=str,
        default=None,
        help="If set, save the selected denormalized preds/labels/error to this .npz file.",
    )
    args = parser.parse_args()

    if args.error_cmap is None:
        # Default the error panel to the SAME gradient as Ground Truth / Prediction
        # (e.g. jet, blue->green->red), so all three columns share one visual style;
        # the error just spans 0..max instead of the GT/Pred temperature range. A
        # diverging map is kept for the signed residual, whose range is symmetric
        # around 0.
        args.error_cmap = "RdBu_r" if args.signed else args.cmap

    cfg = load_config(args.config)
    cli = SimpleNamespace(
        just_velocities=False,
        move_data=None,
        max_num_train_time_steps=None,
        train_time_step_size=None,
        train_small_time_transition=False,
        stats_json=args.stats_json,
        model_path=args.model_path,
    )
    cli.stats_json = resolve_stats_json_for_eval(cfg, cli)
    if cli.stats_json is not None:
        print(f"[visualize] Using normalization constants: {cli.stats_json}")
    extra_kwargs = build_extra_kwargs(cfg, cli)

    dataset = build_datasets(cfg, args.data_path, extra_kwargs, ["test"])["test"]
    num_total = len(dataset)

    model = ScOT.from_pretrained(args.model_path)
    model.eval()

    training_args = TrainingArguments(
        output_dir="/tmp/visualize",
        per_device_eval_batch_size=args.per_device_batch_size,
        evaluation_strategy="no",
        logging_strategy="no",
        save_strategy="no",
        report_to=[],
        dataloader_pin_memory=True,
        dataloader_num_workers=0,
    )
    trainer = Trainer(model=model, args=training_args)

    predictions = trainer.predict(dataset)

    denorm = _denormalize_predictions(predictions, dataset)
    if denorm is not None:
        preds, labels = denorm
        unit = "K"
        print("[visualize] Using denormalized (physical) values.")
    else:
        preds, labels = predictions.predictions, predictions.label_ids
        unit = "(norm.)"
        print("[visualize] WARNING: no normalization constants found; plotting normalized values.")

    preds = _ensure_4d(preds)
    labels = _ensure_4d(labels)

    sample_indices = _parse_indices(args.sample_indices, preds.shape[0], args.num_samples)
    sample_indices.sort()
    print(f"[visualize] Selected sample indices: {sample_indices}")

    preds_sel = preds[sample_indices]
    labels_sel = labels[sample_indices]

    model_name = os.path.basename(os.path.normpath(args.model_path))
    title = f"{model_name} — {os.path.basename(os.path.normpath(args.data_path))}"

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig = make_grid_figure(
        preds_sel,
        labels_sel,
        sample_indices,
        signed=args.signed,
        cmap=args.cmap,
        error_cmap=args.error_cmap,
        unit=unit,
        title=title,
    )
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] Saved combined figure -> {args.output}")

    if args.separate:
        base, ext = os.path.splitext(args.output)
        for i, idx in enumerate(sample_indices):
            fig_i = make_grid_figure(
                preds_sel[i : i + 1],
                labels_sel[i : i + 1],
                [idx],
                signed=args.signed,
                cmap=args.cmap,
                error_cmap=args.error_cmap,
                unit=unit,
                title=f"{title} (sample {idx})",
            )
            out_i = f"{base}_sample{idx}{ext}"
            fig_i.savefig(out_i, dpi=150, bbox_inches="tight")
            plt.close(fig_i)
            print(f"[visualize] Saved -> {out_i}")

    if args.save_npy:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_npy)), exist_ok=True)
        np.savez_compressed(
            args.save_npy,
            preds=preds_sel,
            labels=labels_sel,
            indices=np.array(sample_indices),
        )
        print(f"[visualize] Saved arrays -> {args.save_npy}")


if __name__ == "__main__":
    main()
