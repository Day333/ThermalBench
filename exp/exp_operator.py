"""Training and evaluation for FNO / U-FNO / SAU-FNO / UNet / DeepONet.

These five used to be five independent scripts, but the pipelines are structurally the
same. The only differences are how the model is built, how the optimizer is configured,
and how the tensors are laid out -- the first two moved into exp_basic.MODEL_ZOO, the
third into _to_model / _from_model below.

+-- TWO THINGS THAT MUST NOT CHANGE ------------------------------------------+
| 1. Call order: set_seed -> load data -> build normalizers -> **build model** |
|    -> build DataLoader. Model init consumes RNG, and DataLoader(shuffle=True) |
|    consumes RNG. Move any step and both the initial weights and the batch     |
|    order change, so training stops reproducing. The original scripts used     |
|    exactly this order.                                                        |
| 2. Evaluation path: deterministic forward (model.eval(), dropout off),        |
|    batches of 20, denormalization on the **CPU**, cast to float64, then the   |
|    single shared function in utils.metrics. FNO is the exception (see         |
|    _predict_fno) -- it always had a different path.                           |
+-----------------------------------------------------------------------------+

FNO carries one more deliberate inconsistency: its training loop stays in
model/FNO.py's train_model rather than being lifted out. That loop uses the vendored
models/Adam.py (not torch.optim.Adam) and its own validation logic; rewriting it as a
generic loop would necessarily change the numbers. The benchmark is judged on
reproducing existing results, so fidelity wins.
"""
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from data_provider.data_factory import data_provider
from exp.exp_basic import MODEL_ZOO, build_model, source_level
from layers.normalize import make_norm
from utils.metrics import _compute_additional_test_metrics
from utils.tools import (dump_metrics, load_checkpoint, print_six,
                         save_checkpoint, set_seed)

EVAL_BATCH = 20     # matches the original scripts; affects batching only, not values


# --------------------------------------------------------------------------
# Tensor layout: UNet is 2D channel-first, everything else is 5D channel-last
# --------------------------------------------------------------------------
def _to_model(x, name):
    """(B,X,Y,Z,P) -> model input."""
    if name == "UNet":
        return x.squeeze(-2).permute(0, 3, 1, 2)        # (B,P,X,Y)
    return x


def _to_model_y(y, name):
    """(B,X,Y,Z) -> training target."""
    if name == "UNet":
        return y.squeeze(-1)                            # (B,X,Y)
    return y


def _from_model(p, name):
    """model output -> (B,X,Y,Z)."""
    if name == "UNet":
        return p.permute(0, 2, 3, 1)                    # (B,C=1,X,Y) -> (B,X,Y,1)
    return p


def _model_out_for_loss(p, name):
    if name == "UNet":
        return p.squeeze(1)                             # (B,1,X,Y) -> (B,X,Y)
    return p


def forward_norm(model, name, xb):
    """One forward pass in normalized space: (B,X,Y,Z,P) -> (B,X,Y,Z).

    The training loop and few-shot fine-tuning share this function so the shape
    handling cannot silently drift apart between the two. FNO running in float64 is
    original behaviour, not a typo.
    """
    if name == "FNO":
        o = model(xb.to(torch.double))
        return o.reshape(xb.shape[0], xb.shape[1], xb.shape[2], xb.shape[3])
    return _from_model(model(_to_model(xb, name)), name)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def _predict(model, x_norm, y_norm, x_test, name, device):
    """Generic deterministic prediction, returns a (B,X,Y,Z) tensor in Kelvin."""
    model.eval()
    preds = []
    with torch.no_grad():
        xn = _to_model(x_norm.forward(x_test), name).to(device)
        for i in range(0, xn.shape[0], EVAL_BATCH):
            out = model(xn[i:i + EVAL_BATCH]).cpu()
            preds.append(y_norm.inverse(_from_model(out, name)))
    return torch.cat(preds, 0)


def _predict_fno(model, x_norm, y_norm, x_test, device):
    """FNO's deterministic prediction. Three differences from the rest, all original:

    1. dropout off (the native predict() does MC-dropout, averaging 20 samples);
    2. the forward pass runs in float64;
    3. denormalization happens on the GPU.
    Changing any one of them makes the numbers disagree with the published results.
    """
    model = model.to(device).eval()
    y_norm_dev = y_norm
    if hasattr(y_norm, "cuda"):
        y_norm_dev = y_norm.cuda() if device == "cuda" else y_norm
    preds = []
    with torch.no_grad():
        xn = x_norm.forward(x_test).to(device)
        for i in range(0, xn.shape[0], EVAL_BATCH):
            xb = xn[i:i + EVAL_BATCH]
            out = model(xb.to(torch.double))
            out = out.reshape(xb.shape[0], xb.shape[1], xb.shape[2], xb.shape[3])
            preds.append(y_norm_dev.inverse(out).cpu())
    return torch.cat(preds, 0)


def evaluate(model, x_norm, y_norm, x_test, y_test, name, device="cuda"):
    out = (_predict_fno(model, x_norm, y_norm, x_test, device) if name == "FNO"
           else _predict(model, x_norm, y_norm, x_test, name, device))
    pred = out.permute(0, 3, 1, 2).numpy().astype(np.float64)      # (B,Z,X,Y)
    label = y_test.permute(0, 3, 1, 2).numpy().astype(np.float64)
    return _compute_additional_test_metrics(
        pred, label, prefix=MODEL_ZOO[name]["prefix"], topk=50)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def _make_optimizer(model, cfg):
    kw = dict(lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    if "foreach" in cfg:
        kw["foreach"] = cfg["foreach"]
    opt = torch.optim.Adam(model.parameters(), **kw)
    sched = None
    if cfg.get("sched"):
        kind, step, gamma = cfg["sched"]
        assert kind == "step"
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=step, gamma=gamma)
    return opt, sched


def train(args):
    name = args.model
    cfg = MODEL_ZOO[name]
    epochs = args.epochs or cfg["epochs"]
    bs = args.batch_size or cfg["batch_size"]
    device = args.device

    # ---- order-sensitive, do not rearrange -------------------------------
    # FNO's dropout/schedule come from the environment, so they must be set
    # before the model is constructed.
    os.environ.update(cfg.get("env", {}))
    set_seed(args.seed)
    xtr, ytr, xva, yva, xte, yte = data_provider(args)
    P, Z, G = int(xtr.shape[-1]), int(xtr.shape[-2]), int(xtr.shape[1])
    x_norm, y_norm = make_norm(xtr), make_norm(ytr)
    model = build_model(name, P, Z, G).to(device)
    # ----------------------------------------------------------------------

    n_param = sum(p.numel() for p in model.parameters())
    print(f"[{name}] P={P} Z={Z} grid={G} params={n_param} "
          f"epochs={epochs} bs={bs} pcnorm={os.environ.get('PER_CHANNEL_NORM','0')}",
          flush=True)

    out_dir = os.path.join(args.checkpoints, f"{args.data}_{name}")

    if cfg.get("builtin_loop"):          # FNO: use its own training loop
        t0 = time.time()
        x_norm, y_norm, folder = model.train_model(
            xtr, ytr, epochs=epochs, batch_size=bs, work_dir=out_dir,
            epoch_log_fn=None, x_val=xva, y_val=yva, loss_type="base")
        train_time = time.time() - t0
    else:
        opt, sched = _make_optimizer(model, cfg)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                _to_model(x_norm.forward(xtr), name),
                _to_model_y(y_norm.forward(ytr), name)),
            batch_size=bs, shuffle=True)

        # Periodic resume state. On a shared machine the OOM killer can take a
        # multi-day run down at any epoch, so every RESUME_EVERY epochs the full
        # training state (weights, optimizer, scheduler, RNG streams, elapsed
        # time) is written atomically to resume.pt. A restart continues from that
        # epoch and, because the RNG state is restored at the epoch boundary,
        # replays the same shuffles and updates an uninterrupted run would have
        # made. The file is removed once training completes.
        RESUME_EVERY = 10
        resume_path = os.path.join(out_dir, "resume.pt")
        start_ep, elapsed = 1, 0.0
        if os.path.exists(resume_path):
            st = torch.load(resume_path, map_location="cpu")
            if st.get("epochs") != epochs:
                raise RuntimeError(
                    f"{resume_path} was written for epochs={st.get('epochs')}, "
                    f"not {epochs}; delete it to start over")
            model.load_state_dict(st["model"])
            opt.load_state_dict(st["opt"])
            if sched is not None and st.get("sched") is not None:
                sched.load_state_dict(st["sched"])
            torch.set_rng_state(st["rng_cpu"])
            if st.get("rng_cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state(st["rng_cuda"])
            if st.get("rng_np") is not None:
                np.random.set_state(st["rng_np"])
            start_ep, elapsed = st["epoch"] + 1, float(st["elapsed"])
            print(f"[{name}] resumed from epoch {st['epoch']} "
                  f"({elapsed:.0f}s already trained)", flush=True)

        t0 = time.time() - elapsed
        for ep in range(start_ep, epochs + 1):
            model.train()
            tot, nb = 0.0, 0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = F.mse_loss(_model_out_for_loss(model(xb), name), yb)
                loss.backward()
                opt.step()
                tot += loss.item()
                nb += 1
            if sched is not None:
                sched.step()
            if ep == 1 or ep % 10 == 0:
                print(f"  [{name}] ep {ep}/{epochs} loss={tot / nb:.6f}", flush=True)
            if ep % RESUME_EVERY == 0 and ep < epochs:
                os.makedirs(out_dir, exist_ok=True)
                tmp = resume_path + ".tmp"
                torch.save({
                    "epoch": ep, "epochs": epochs, "elapsed": time.time() - t0,
                    "model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict() if sched is not None else None,
                    "rng_cpu": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                    "rng_np": np.random.get_state(),
                }, tmp)
                os.replace(tmp, resume_path)
        train_time = time.time() - t0
        if os.path.exists(resume_path):
            os.remove(resume_path)

    print(f"[{name}] training took {train_time:.1f}s", flush=True)

    m = evaluate(model, x_norm, y_norm, xte, yte, name, device)
    m.update({
        "train_time_s": round(train_time, 1),
        "train_time_per_epoch_s": round(train_time / max(epochs, 1), 2),
        "epochs": epochs, "params": n_param,
        "n_train": int(xtr.shape[0]), "n_test": int(xte.shape[0]),
        "gpus": 1, "per_channel_norm": os.environ.get("PER_CHANNEL_NORM", "0"),
    })
    save_checkpoint(os.path.join(out_dir, "model.pt"), model, x_norm, y_norm, m["params"])
    dump_metrics(os.path.join(out_dir, "test_metrics.json"), m)
    print_six(m, cfg["prefix"])
    return m


def test(args):
    """Evaluate only. --load points at a checkpoint; without it,
    checkpoints/<data>_<model>/model.pt is used."""
    name = args.model
    cfg = MODEL_ZOO[name]
    device = args.device

    os.environ.update(cfg.get("env", {}))
    set_seed(args.seed)
    _, _, _, _, xte, yte = data_provider(args)
    P, Z, G = int(xte.shape[-1]), int(xte.shape[-2]), int(xte.shape[1])

    # Pure extrapolation sets such as level5 have no weights of their own, so fall
    # back to the level their model was trained on (level4).
    path = args.load or os.path.join(
        args.checkpoints, f"{source_level(args.data)}_{name}", "model.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no checkpoint at: {path}")
    # The legacy format pickles whole objects and needs no pre-built model;
    # the new format does.
    try:
        x_norm, model, y_norm = load_checkpoint(path)
    except ValueError:
        x_norm, model, y_norm = load_checkpoint(path, build_model(name, P, Z, G))
    model = model.to(device)
    print(f"[{name}] loaded {path}  n_test={xte.shape[0]}", flush=True)

    m = evaluate(model, x_norm, y_norm, xte, yte, name, device)
    m["n_test"] = int(xte.shape[0])
    print_six(m, cfg["prefix"])
    if args.output:
        dump_metrics(args.output, m)
    return m
