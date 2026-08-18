"""Few-shot fine-tuning on the level5 extrapolation set.

level5 swaps in five brand-new cases, and zero-shot error there is large (see the
benchmark document). This module answers: given only a handful of labelled samples per
case, how much of that gap can fine-tuning recover?

+-- SPLIT ---------------------------------------------------------------------+
| Grouped by case from the manifest, preserving order within a group:            |
|   first 500 per case -> fine-tuning pool; K-shot takes its first K             |
|                         (5K training samples in total)                        |
|   last  500 per case -> **held-out evaluation set** (2500 total), shared by    |
|                         every K, with zero overlap with the fine-tuning pool   |
| K=0 means no fine-tuning, baseline only -- the first point of the curve.       |
+-------------------------------------------------------------------------------+

**The learning rate is one tenth of each model's own training lr**, not one shared
value. This is not an aesthetic preference: Therm-FM diverges outright on the FNO
family's 1e-4 (measured RMSE 16.01 -> 31.34, collapsing within a single epoch) because
it trains at only 5e-5. The defaults live in exp_basic.MODEL_ZOO as `finetune_lr`.

Therm-FM differs from the rest in two known, deliberate ways:
  1. the loss is the model's own `out.loss` (scOT picks L1 or MSE from config.p), i.e.
     each family keeps its own training objective;
  2. normalization goes through scOT's own pipeline (constants come from the
     checkpoint's normalization_constants.json).
Its evaluation uses Trainer.predict, the same code path as the main evaluation; the
held-out set is selected by index via TFM_EVAL_INDICES, so no data copies are needed.
"""
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from exp.exp_basic import MODEL_ZOO, build_model, source_level
from utils.metrics import _compute_additional_test_metrics
from utils.tools import load_checkpoint, set_seed

N_FT_POOL = 500      # first 500 per case are the fine-tuning pool, last 500 the holdout
METRICS = ["rmse", "mean_absolute_error", "r2", "max_absolute_error",
           "max_temperature_error", "topk50_temperature_difference"]


def _split_indices(manifest, shots):
    """-> (fine-tune indices, holdout indices). Disjointness is asserted at the end."""
    groups = {}
    for i, rec in enumerate(manifest):
        groups.setdefault(rec["case"], []).append(i)
    ft, ho = [], []
    for case, idx in groups.items():
        if len(idx) != 2 * N_FT_POOL:
            raise ValueError(f"{case} has {len(idx)} samples, expected {2 * N_FT_POOL} "
                             f"(first half is the fine-tuning pool, second half the holdout)")
        ft += idx[:shots]
        ho += idx[N_FT_POOL:]
    ft, ho = sorted(ft), sorted(ho)
    assert not (set(ft) & set(ho)), "fine-tuning and holdout sets overlap"
    return ft, ho


def _six(pred, label):
    m = _compute_additional_test_metrics(
        pred.astype(np.float64), label.astype(np.float64), prefix="m", topk=50)
    return {k: float(m["m/" + k]) for k in METRICS}


# ==========================================================================
# FNO / U-FNO / SAU-FNO / UNet / DeepONet
# ==========================================================================
def _run_operator(args):
    from data_provider.data_loader import load_mat_pair
    from exp.exp_operator import forward_norm

    name = args.model
    cfg = MODEL_ZOO[name]
    os.environ.update(cfg.get("env", {}))
    set_seed(args.seed)

    folder = os.path.join(args.root_path, f"{args.data}_steady")
    x_all, y_all, _, _ = load_mat_pair(folder)
    manifest = json.load(open(os.path.join(folder, "manifest.json"), encoding="utf-8"))
    ft_idx, ho_idx = _split_indices(manifest, args.shots)

    path = args.load or os.path.join(
        args.checkpoints, f"{source_level(args.data)}_{name}", "model.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no checkpoint at: {path}\n"
            f"level5 is a pure extrapolation set; fine-tuning starts from the weights "
            f"trained on {source_level(args.data)}. Run this first:  "
            f"python run.py --model {name} --data {source_level(args.data)} --task train")
    try:
        x_norm, model, y_norm = load_checkpoint(path)
    except ValueError:
        P, Z, G = int(x_all.shape[-1]), int(x_all.shape[-2]), int(x_all.shape[1])
        x_norm, model, y_norm = load_checkpoint(path, build_model(name, P, Z, G))
    from utils.compat import patch_legacy_attrs
    patch_legacy_attrs(model)
    model = model.to(args.device)

    # Keep the normalizers stored in the checkpoint rather than re-deriving them from
    # the extrapolation set -- only the weights adapt, the preprocessing does not.
    # Otherwise the K=0 baseline would not match the main evaluation.
    xn_ho, y_ho = x_norm.forward(x_all[ho_idx]), y_all[ho_idx]

    def _eval():
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, xn_ho.shape[0], 20):
                o = forward_norm(model, name, xn_ho[i:i + 20].to(args.device))
                preds.append(y_norm.inverse(o.float().cpu()))
        return _six(torch.cat(preds, 0).permute(0, 3, 1, 2).numpy(),
                    y_ho.permute(0, 3, 1, 2).numpy())

    print(f"[fewshot] {name} shots={args.shots} finetune={len(ft_idx)} "
          f"holdout={len(ho_idx)} lr={args.lr} epochs={args.ft_epochs}", flush=True)
    curve = [dict(epoch=0, **_eval())]
    print(f"[fewshot] ep0 (no fine-tuning) rmse={curve[0]['rmse']:.4f}", flush=True)

    t0 = time.time()
    if args.shots > 0:
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_norm.forward(x_all[ft_idx]),
                                           y_norm.forward(y_all[ft_idx])),
            batch_size=args.batch_size or 20, shuffle=True)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                               weight_decay=1e-4, foreach=False)
        for ep in range(1, args.ft_epochs + 1):
            model.train()
            tot, nb = 0.0, 0
            for xb, yb in loader:
                xb, yb = xb.to(args.device), yb.to(args.device)
                opt.zero_grad()
                pred = forward_norm(model, name, xb)
                loss = F.mse_loss(pred, yb.to(pred.dtype))
                loss.backward()
                opt.step()
                tot += loss.item()
                nb += 1
            if ep % args.eval_every == 0 or ep == args.ft_epochs:
                m = _eval()
                curve.append(dict(epoch=ep, train_loss=tot / max(nb, 1), **m))
                print(f"[fewshot] ep{ep} loss={tot / max(nb, 1):.6f} "
                      f"rmse={m['rmse']:.4f}", flush=True)
    return curve, time.time() - t0, len(ft_idx), len(ho_idx)


# ==========================================================================
# Therm-FM
# ==========================================================================
def _run_scot(args):
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "model"))
    from scOT.evaluate import (_denormalize_predictions, build_datasets,
                               build_extra_kwargs, load_config,
                               resolve_stats_json_for_eval)
    from scOT.model import ScOT
    from scOT.trainer import Trainer, TrainingArguments

    size = args.model.split("-")[1]
    set_seed(args.seed)
    md = args.load or os.path.join(
        args.checkpoints, f"{source_level(args.data)}_{args.model}")
    if not os.path.isdir(md):
        raise FileNotFoundError(
            f"no checkpoint directory at: {md}\n"
            f"level5 is a pure extrapolation set; fine-tuning starts from the weights "
            f"trained on {source_level(args.data)}.")
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "model", "thermfm_configs",
                            f"run_{source_level(args.data)}_steady_{size}.yaml")

    def build_ds(data_path):
        cfg = load_config(cfg_path)
        cli = SimpleNamespace(just_velocities=False, move_data=None,
                              max_num_train_time_steps=None, train_time_step_size=None,
                              train_small_time_transition=False, stats_json=None,
                              model_path=md)
        cli.stats_json = resolve_stats_json_for_eval(cfg, cli)
        if cli.stats_json is None:
            raise FileNotFoundError(f"{md}: missing normalization_constants.json")
        return build_datasets(cfg, data_path, build_extra_kwargs(cfg, cli), ["test"])["test"]

    # scOT's Dataset treats only the trailing 20% as test, so TFM_EVAL_INDICES is used
    # here to name the holdout indices explicitly (zero copies; the earlier workaround
    # tiled the data 5x).
    src = os.path.join(args.root_path, f"{args.data}_steady")
    manifest = json.load(open(os.path.join(src, "manifest.json"), encoding="utf-8"))
    ft_idx, ho_idx = _split_indices(manifest, args.shots)
    idx_file = "/tmp/_tfm_eval_idx.json"
    with open(idx_file, "w", encoding="utf-8") as f:
        json.dump(ho_idx, f)
    os.environ["TFM_EVAL_INDICES"] = idx_file
    ho_ds = build_ds(src)
    os.environ.pop("TFM_EVAL_INDICES")
    if len(ho_ds) != len(ho_idx):
        raise ValueError(f"holdout set has {len(ho_ds)} samples, expected {len(ho_idx)}")

    model = ScOT.from_pretrained(md).to(args.device)

    def _eval():
        ta = TrainingArguments(output_dir="/tmp/_fs_tfm", per_device_eval_batch_size=16,
                               evaluation_strategy="no", logging_strategy="no",
                               save_strategy="no", report_to=[], dataloader_num_workers=0)
        po = Trainer(model=model, args=ta).predict(ho_ds)
        dn = _denormalize_predictions(po, ho_ds)
        pr, la = dn if dn is not None else (po.predictions, po.label_ids)
        pr, la = np.asarray(pr), np.asarray(la)
        if pr.ndim == 3:
            pr = pr[:, None]
        if la.ndim == 3:
            la = la[:, None]
        return _six(pr, la)

    print(f"[fewshot] {args.model} shots={args.shots} holdout={len(ho_ds)} "
          f"lr={args.lr} epochs={args.ft_epochs}", flush=True)
    curve = [dict(epoch=0, **_eval())]
    print(f"[fewshot] ep0 (no fine-tuning) rmse={curve[0]['rmse']:.4f}", flush=True)

    t0 = time.time()
    n_ft = 0
    if args.shots > 0:
        os.environ["TFM_EVAL_ALL"] = "1"      # the fine-tuning source must expose all samples
        full = build_ds(src)
        os.environ.pop("TFM_EVAL_ALL")
        n_ft = len(ft_idx)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(full, ft_idx),
            batch_size=args.batch_size or 20, shuffle=True)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        for ep in range(1, args.ft_epochs + 1):
            model.train()
            tot, nb = 0.0, 0
            for batch in loader:
                opt.zero_grad()
                out = model(pixel_values=batch["pixel_values"].to(args.device),
                            labels=batch["labels"].to(args.device))
                out.loss.backward()
                opt.step()
                tot += float(out.loss)
                nb += 1
            if ep % args.eval_every == 0 or ep == args.ft_epochs:
                m = _eval()
                curve.append(dict(epoch=ep, train_loss=tot / max(nb, 1), **m))
                print(f"[fewshot] ep{ep} loss={tot / max(nb, 1):.6f} "
                      f"rmse={m['rmse']:.4f}", flush=True)
    return curve, time.time() - t0, n_ft, len(ho_ds)


def finetune(args):
    if args.lr is None:
        args.lr = MODEL_ZOO[args.model]["finetune_lr"]
    runner = _run_scot if MODEL_ZOO[args.model].get("scot") else _run_operator
    curve, ft_time, n_ft, n_ho = runner(args)

    rec = {"model": args.model, "data": args.data, "shots": args.shots,
           "n_finetune": n_ft, "n_holdout": n_ho, "epochs": args.ft_epochs,
           "lr": args.lr, "finetune_time_s": round(ft_time, 1),
           "curve": curve, "final": curve[-1], "baseline": curve[0]}
    out = args.output or os.path.join(
        args.checkpoints, "fewshot", f"{args.data}_{args.model}_k{args.shots}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    print(f"[fewshot] -> {out}  final rmse={curve[-1]['rmse']:.4f} "
          f"(baseline {curve[0]['rmse']:.4f})", flush=True)
    return rec
