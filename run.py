#!/usr/bin/env python3
"""ThermalBench -- a single entry point for the chip steady-state thermal operator
learning benchmark.

    python run.py --model UFNO      --data level2 --task train
    python run.py --model UFNO      --data level2 --task test  --load <ckpt>
    python run.py --model ThermFM-T --data level4 --task train --gpus 0,1,2,3
    python run.py --model UFNO      --data level5 --task finetune --shots 10
    python run.py --model UFNO      --data level5 --task test

Models: FNO / UFNO / SAUFNO / UNet / DeepONet / ThermFM-T / ThermFM-B / ThermFM-L
Data:   level2 (P=3), level3 (P=4), level4 (P=7), level5 (P=7, pure extrapolation set)

About level5: it swaps in **five brand-new cases and takes no part in training**. It is
scored by loading the weights trained on level4 and predicting zero-shot -- that is the
last command above, and --load can be omitted since it resolves automatically. Running
--task train on it is meaningless.

About PER_CHANNEL_NORM: it must stay on (the default) for the P=7 level4/level5. The
three boundary-condition channels differ from the power channel by several orders of
magnitude, and a single global scalar normalizer would crush them into numerical noise.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp.exp_basic import MODEL_ZOO, OPERATOR_MODELS, SCOT_MODELS  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(
        description="ThermalBench",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument("--model", required=True, choices=list(MODEL_ZOO),
                   help="model name")
    p.add_argument("--data", required=True,
                   help="dataset name, e.g. level2 -> <root_path>/level2_steady")
    p.add_argument("--task", default="train", choices=["train", "test", "finetune"],
                   help="train = train and evaluate; test = evaluate only; "
                        "finetune = few-shot fine-tuning on the level5 extrapolation set")
    p.add_argument("--root_path", default="datasets",
                   help="dataset root (a symlink to the real data by default)")
    p.add_argument("--checkpoints", default="checkpoints")
    p.add_argument("--load", default=None, help="checkpoint path to evaluate")
    p.add_argument("--output", default=None, help="also write the metrics to this json")
    p.add_argument("--epochs", type=int, default=None,
                   help="override the registry default (changing it stops the benchmark "
                        "numbers from reproducing)")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_trajectories", type=int, default=-1,
                   help="cap the train+val size, -1 = all; does not affect the test split")
    p.add_argument("--device", default="cuda")
    p.add_argument("--gpus", default="0",
                   help="CUDA_VISIBLE_DEVICES for Therm-FM, e.g. 0,1,2,3")
    p.add_argument("--port", type=int, default=29815,
                   help="accelerate main-process port; stagger it when running several "
                        "Therm-FM jobs at once")
    p.add_argument("--shots", type=int, default=10,
                   help="finetune only: samples per case used for fine-tuning, "
                        "0 = baseline only")
    p.add_argument("--ft_epochs", type=int, default=50,
                   help="finetune only: number of fine-tuning epochs")
    p.add_argument("--eval_every", type=int, default=5,
                   help="finetune only: evaluate on the holdout every N epochs, "
                        "which is what produces the curve")
    p.add_argument("--lr", type=float, default=None,
                   help="finetune only: defaults to one tenth of the model's own "
                        "training lr (see MODEL_ZOO)")
    p.add_argument("--per_channel_norm", type=int, default=1,
                   help="1 = per-channel normalization (default, required for P=7); "
                        "0 = single global scalar")
    p.add_argument("--quiet", dest="verbose", action="store_false")
    return p.parse_args()


def main():
    args = parse_args()
    # The normalization switch is read from the environment (layers/normalize.make_norm),
    # so it has to be set before any model or data module actually uses it.
    os.environ["PER_CHANNEL_NORM"] = str(args.per_channel_norm)

    if args.model in OPERATOR_MODELS:
        from utils.compat import install
        install()          # allows loading checkpoints produced by the legacy repos

    if args.task == "finetune":
        from exp.exp_fewshot import finetune
        finetune(args)
        return

    if args.model in SCOT_MODELS:
        from exp import exp_thermfm as exp
    else:
        from exp import exp_operator as exp
    (exp.train if args.task == "train" else exp.test)(args)


if __name__ == "__main__":
    main()
