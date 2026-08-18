""".mat loading and dataset splitting -- copied verbatim from fno/train.py.

The whole benchmark shares this split, which is what makes the numbers comparable:
  test  = last 20% of everything (train_ratio=0.8), fixed
  train = first 90% of the leading 80%
  val   = last 10% of the leading 80%

NOTE: the split slices by index and does not shuffle. The reason the datasets are stored
case-interleaved (c1s1, c2s1, ..., c10s1, c1s2, ...) is precisely so that all three
segments stay case-balanced. If the .mat were stacked by case instead, train would
contain no Case10 at all while test would be entirely Case10.

Tensor convention: the .mat files hold (B,P,Z,Y,X) / (B,Z,Y,X); after loading they are
transposed to (B,X,Y,Z,P) / (B,X,Y,Z).
"""
import os
import h5py
import numpy as np
import torch


def _find_single_mat(data_dir_path, keyword):
    files = [
        os.path.join(data_dir_path, f)
        for f in os.listdir(data_dir_path)
        if f.endswith(".mat") and keyword in f.lower()
    ]
    if len(files) != 1:
        raise ValueError(
            f"expected exactly 1 .mat file matching '{keyword}' in {data_dir_path}, "
            f"found {len(files)}: {files}"
        )
    return files[0]


def load_mat_pair(folder_path):
    input_path = _find_single_mat(folder_path, "input")
    output_path = _find_single_mat(folder_path, "output")

    with h5py.File(input_path, "r") as f_in:
        if "data" not in f_in:
            raise KeyError(f"{input_path} has no key 'data'")
        input_np = f_in["data"][()]

    with h5py.File(output_path, "r") as f_out:
        if "data" not in f_out:
            raise KeyError(f"{output_path} has no key 'data'")
        output_np = f_out["data"][()]

    if input_np.ndim != 5:
        raise ValueError(f"input should be 5-D [B,P,Z,Y,X], got {input_np.shape}")
    if output_np.ndim != 4:
        raise ValueError(f"output should be 4-D [B,Z,Y,X], got {output_np.shape}")
    if input_np.shape[0] != output_np.shape[0]:
        raise ValueError(
            f"input/output batch size mismatch: {input_np.shape[0]} vs {output_np.shape[0]}"
        )
    if tuple(input_np.shape[2:]) != tuple(output_np.shape[1:]):
        raise ValueError(
            "input/output spatial sizes disagree: "
            f"input[Z,Y,X]={input_np.shape[2:]}, output[Z,Y,X]={output_np.shape[1:]}"
        )

    # [B, P, Z, Y, X] -> [B, X, Y, Z, P]
    input_np = np.transpose(input_np, (0, 4, 3, 2, 1))
    # [B, Z, Y, X] -> [B, X, Y, Z]
    output_np = np.transpose(output_np, (0, 3, 2, 1))

    x_all = torch.tensor(input_np, dtype=torch.float32)
    y_all = torch.tensor(output_np, dtype=torch.float32)
    return x_all, y_all, input_path, output_path


def split_train_val_test(x_all, y_all, num_trajectories=-1, train_ratio=0.9):
    """
    First split everything into (train+val) and test by train_ratio (test never moves),
    then trim train+val down to num_trajectories samples,
    then split the trimmed train+val 9:1 into train and val.
    """
    total = x_all.shape[0]
    if total < 3:
        raise ValueError(f"need at least 3 samples (train/val/test), got {total}")
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"train_ratio must lie in (0,1), got {train_ratio}")

    # step 1: carve off the fixed test set
    trainval_total = int(total * train_ratio)
    trainval_total = max(2, min(trainval_total, total - 1))  # leave at least 1 sample for test

    x_trainval_full = x_all[:trainval_total]
    y_trainval_full = y_all[:trainval_total]
    x_test = x_all[trainval_total:]
    y_test = y_all[trainval_total:]

    # step 2: cap the train+val size via num_trajectories (test is unaffected)
    if num_trajectories is not None and num_trajectories > 0:
        trainval_used = min(int(num_trajectories), trainval_total)
    else:
        trainval_used = trainval_total
    if trainval_used < 2:
        raise ValueError(
            f"need at least 2 samples for train+val, got {trainval_used}."
        )

    x_trainval = x_trainval_full[:trainval_used]
    y_trainval = y_trainval_full[:trainval_used]

    # step 3: fixed 9:1 split into train / val
    n_train = int(trainval_used * 0.9)
    n_train = max(1, min(n_train, trainval_used - 1))

    return (
        x_trainval[:n_train],
        y_trainval[:n_train],
        x_trainval[n_train:],
        y_trainval[n_train:],
        x_test,
        y_test,
    )
