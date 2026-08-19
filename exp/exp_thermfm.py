"""Training and evaluation for Therm-FM (scOT / Poseidon backbone).

This model does not use exp_operator's loop -- not out of laziness, but because
**unifying it would break reproduction**: scOT runs HuggingFace Trainer + accelerate
DDP across four GPUs, with its own cosine schedule, gradient clipping, early stopping
and checkpoint-selection logic. Rewriting that as a plain single-GPU loop would move
the numbers. So the original pipeline is kept and this exp layer only assembles the
arguments and launches the processes.

The vendored source lives in model/scOT/ (copied verbatim from Therm-FM/scOT) and is
imported under its original name via PYTHONPATH pointing at model/ -- upstream source
is unmodified, which keeps future syncs easy.

+-- TWO TRAPS WORTH KNOWING ---------------------------------------------------+
| 1. TFM_LAST_EPOCH=1                                                           |
|    scOT defaults to load_best_model_at_end=True, picking the best checkpoint   |
|    by **validation** loss. Under this project's validation split that selects  |
|    a badly undertrained epoch-2 model (measured RMSE 8.83, versus ~0.5 for the |
|    final epoch). Setting this to 1 uses the last epoch instead and **also      |
|    skips** EarlyStoppingCallback -- HF's callback asserts                      |
|    load_best_model_at_end=True, so disabling only the former makes every DDP   |
|    child exit within 12 seconds.                                              |
| 2. scOT's Dataset applies its own train_ratio split on top                     |
|    When evaluating a pure extrapolation set (level5) it would otherwise only   |
|    score the last 20% -- all of it from a single case. This project adds       |
|    TFM_EVAL_ALL / TFM_EVAL_INDICES to the vendored ThermalSteady3D so the test |
|    segment is selected by index instead. Zero copies.                          |
|    (Upstream worked around this by tiling the data 5x, which cost every user   |
|    an extra 4.7 GB and one more preprocessing step.)                           |
+-------------------------------------------------------------------------------+
"""
import json
import os
import subprocess
import sys
import time

from exp.exp_basic import PURE_EVAL, source_level

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCOT_DIR = os.path.join(ROOT, "model")          # lets `import scOT` find model/scOT


def _env(args):
    env = dict(os.environ)
    env["PYTHONPATH"] = SCOT_DIR + os.pathsep + env.get("PYTHONPATH", "")
    env["WANDB_MODE"] = "offline"               # no outbound net; otherwise it retries forever
    env["TFM_LAST_EPOCH"] = "1"                 # see the module docstring
    env["CUDA_VISIBLE_DEVICES"] = args.gpus
    if args.data in PURE_EVAL:
        # A pure extrapolation set must be scored in full. scOT's Dataset treats only
        # the trailing 20% as test, so feeding level5 directly would score just 1000
        # samples, all from one case. With this flag the test segment is taken by
        # index over the whole set -- the earlier workaround tiled the data 5x, which
        # cost every user 4.7 GB and an extra preprocessing step.
        env["TFM_EVAL_ALL"] = "1"
    return env


def _config_path(args):
    data = source_level(args.data)
    p = os.path.join(ROOT, "model", "thermfm_configs",
                     f"run_{data}_steady_{args.model.split('-')[1]}.yaml")
    if not os.path.exists(p):
        raise FileNotFoundError(f"missing Therm-FM config: {p}")
    return _override(p, args)


def _override(cfg_path, args):
    """Let --epochs / --num_trajectories override the matching yaml entries.

    Therm-FM's hyperparameters live in the yaml rather than on the command line
    (epochs, lr and batch size all do), so these two generic flags have to be written
    into a temporary config before being handed to scOT. Without them the benchmark
    config is returned untouched, byte for byte.
    """
    if args.epochs is None and args.num_trajectories in (None, -1):
        return cfg_path
    import yaml
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.epochs is not None:
        cfg["num_epochs"]["value"] = args.epochs
    if args.num_trajectories not in (None, -1):
        cfg["num_trajectories"]["value"] = args.num_trajectories
    tmp_dir = os.path.join(ROOT, "logs", "_tmp_configs")
    os.makedirs(tmp_dir, exist_ok=True)
    out = os.path.join(tmp_dir, os.path.basename(cfg_path))
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)
    print(f"[thermfm] config overridden -> {out} "
          f"(epochs={cfg['num_epochs']['value']}, "
          f"num_trajectories={cfg['num_trajectories']['value']})", flush=True)
    return out


def _run(cmd, env, log):
    print(f"[thermfm] $ {' '.join(cmd)}", flush=True)
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "w", encoding="utf-8") as f:
        r = subprocess.run(cmd, env=env, cwd=ROOT, stdout=f,
                           stderr=subprocess.STDOUT)
    if r.returncode != 0:
        sys.exit(f"[thermfm] failed (exit {r.returncode}), see log at {log}")


def train(args):
    size = args.model.split("-")[1]
    name = f"{args.data}_steady_{size}"
    env = _env(args)
    n_gpu = len(args.gpus.split(","))
    ckpt_root = os.path.join(args.checkpoints, "thermfm")
    log_dir = os.path.join(ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)

    pre = os.path.join(ROOT, "pretrained", f"Poseidon-{size}")
    if not os.path.isdir(pre):
        sys.exit(f"[thermfm] missing pretrained weights at {pre}; Therm-FM is fine-tuned "
                 f"from Poseidon, not trained from scratch. Put Poseidon-T/B/L under "
                 f"pretrained/.")

    t0 = time.time()
    _run(["accelerate", "launch", "--multi_gpu", f"--num_processes={n_gpu}",
          "--main_process_port", str(args.port),
          os.path.join("model", "scOT", "train.py"),
          "--config", _config_path(args),
          "--data_path", os.path.join(args.root_path, f"{args.data}_steady"),
          "--checkpoint_path", ckpt_root,
          "--finetune_from", pre,
          "--replace_embedding_recovery",
          "--wandb_project_name", "IC-ThermBench",
          "--wandb_run_name", name],
         env, os.path.join(log_dir, f"thermfm_{name}_train.log"))
    train_time = time.time() - t0
    print(f"[thermfm] {name} training took {train_time:.1f}s ({n_gpu} GPUs)", flush=True)

    # scOT always writes to <checkpoint_path>/<project>/<run>; symlink it back to the
    # shared naming so everything under checkpoints/ looks the same
    # (level2_UFNO / level2_ThermFM-T ...).
    real = os.path.join(ckpt_root, "IC-ThermBench", name)
    link = os.path.join(args.checkpoints, f"{args.data}_{args.model}")
    if os.path.islink(link):
        os.unlink(link)
    if not os.path.exists(link):
        os.symlink(real, link)

    m = _evaluate(args, real, name, env)
    m["train_time_s"] = round(train_time, 1)
    m["gpus"] = n_gpu
    out = os.path.join(real, "test_metrics.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    print(f"[metrics] -> {out}", flush=True)
    return m


def _evaluate(args, model_path, name, env):
    eval_dir = os.path.join(ROOT, "eval_outputs", name)
    _run([sys.executable, os.path.join("model", "scOT", "evaluate.py"),
          "--model_path", model_path,
          "--config", _config_path(args),
          "--data_path", os.path.join(args.root_path, f"{args.data}_steady"),
          "--output_dir", eval_dir, "--only_test"],
         env, os.path.join(ROOT, "logs", f"thermfm_{name}_eval.log"))
    p = os.path.join(eval_dir, "test.json")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def test(args):
    size = args.model.split("-")[1]
    name = f"{args.data}_steady_{size}"
    path = args.load or os.path.join(
        args.checkpoints, f"{source_level(args.data)}_{args.model}")
    if not os.path.isdir(path):
        raise FileNotFoundError(f"no Therm-FM checkpoint directory at: {path}")
    m = _evaluate(args, path, name, _env(args))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(m, f, ensure_ascii=False, indent=2)
    return m
