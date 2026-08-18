"""Resolve a dataset name into train/val/test tensors.

`--data level2` -> `datasets/level2_steady`. `datasets` is a symlink to the real data
directory, so moving to another machine only means re-pointing that symlink.
"""
import os

from data_provider.data_loader import load_mat_pair, split_train_val_test

TRAIN_RATIO = 0.8   # shared by the whole benchmark; changing it invalidates every result

# Pure extrapolation sets: the entire set is the evaluation set and none of it is used
# for training. The value is "the level whose model is evaluated on it".
#
# These datasets **must not** go through the train_ratio split. They are stored stacked
# by case (--layout sequential), so taking "the last 20%" would score only the 1000
# samples of the final case, which is nowhere near the full-set numbers.
PURE_EVAL = {"level5": "level4"}


def source_level(data):
    """Pure extrapolation set -> the level holding its weights; others pass through."""
    return PURE_EVAL.get(data, data)


def resolve_dir(root, name):
    """level2 -> <root>/level2_steady; a full directory name or absolute path is
    returned unchanged."""
    if os.path.isdir(name):
        return name
    for cand in (f"{name}_steady", name):
        p = os.path.join(root, cand)
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(
        f"dataset {name} not found: neither {root}/{name}_steady nor {root}/{name} "
        f"exists. `datasets` is a symlink -- check that the directory it points at "
        f"contains {name}_steady/")


def data_provider(args, flag="all"):
    """Return (x_train, y_train, x_val, y_val, x_test, y_test).

    `flag` only affects printing, not the returned data -- the split is deterministic
    and callers take whichever segment they need.
    """
    folder = resolve_dir(args.root_path, args.data)
    x_all, y_all, in_path, out_path = load_mat_pair(folder)
    if args.data in PURE_EVAL:
        empty_x, empty_y = x_all[:0], y_all[:0]
        if args.verbose:
            print(f"[data] {folder}", flush=True)
            print(f"[data] pure extrapolation set: all {x_all.shape[0]} samples are "
                  f"evaluated, no split", flush=True)
        return empty_x, empty_y, empty_x, empty_y, x_all, y_all
    splits = split_train_val_test(
        x_all, y_all, num_trajectories=args.num_trajectories, train_ratio=TRAIN_RATIO)
    xtr, ytr, xva, yva, xte, yte = splits
    if args.verbose:
        print(f"[data] {folder}", flush=True)
        print(f"[data] input {tuple(x_all.shape)}  output {tuple(y_all.shape)}  "
              f"P={x_all.shape[-1]} Z={x_all.shape[-2]}", flush=True)
        print(f"[data] train={xtr.shape[0]} val={xva.shape[0]} test={xte.shape[0]} "
              f"(train_ratio={TRAIN_RATIO}, then 9:1 for val)", flush=True)
    return splits
