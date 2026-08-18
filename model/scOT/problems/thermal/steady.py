import json
import os

import h5py
import numpy as np
import torch

from scOT.problems.base import BaseDataset


def _resolve_eval_indices(n_max):
    """Pick the test-segment samples from the environment; returns None when neither
    variable is set, preserving the original trailing-slice behaviour.

    ThermalBench extension. It exists so that scoring every sample of a pure
    extrapolation set no longer requires tiling a copy of the data -- that cost every
    user an extra 4.7 GB and one more preprocessing step.
    """
    if os.environ.get("TFM_EVAL_ALL", "0") in ("1", "true", "True"):
        return list(range(n_max))
    path = os.environ.get("TFM_EVAL_INDICES", "")
    if not path:
        return None
    with open(path, encoding="utf-8") as handle:
        idx = json.load(handle)
    if not idx:
        raise ValueError(f"{path}: empty index list")
    if max(idx) >= n_max or min(idx) < 0:
        raise ValueError(f"{path}: index out of range (the dataset has {n_max} samples)")
    return list(idx)


class ThermalSteady3D(BaseDataset):
    """
    Steady state heat conduction dataset, supporting different spatial resolutions and layers.

    Data format convention: keep N as the first dimension for efficient I/O.
    - Input .mat: (N, P, L, H, W)
    - Output .mat: (N, L_out, H, W)
    Where P is the number of physical quantities (e.g. x, y, z, power_density), L/L_out is the number of layers in the z direction.
    """
    
    def __init__(self, folder_path=None, *args, **kwargs):
        """
        Args:
            folder_path: Path to folder containing input and output .mat files
            which: 'train' or 'test'
            num_trajectories: Number of trajectories to use (ignored, uses all data)
            data_path: Base data path (used if folder_path is None)
        """
        stats_json = kwargs.pop("stats_json", None)
        train_ratio = float(kwargs.pop("train_ratio", 0.8))
        if not (0 < train_ratio < 1):
            raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
        if folder_path is None:
            folder_path = kwargs.pop("data_path", None)
        if folder_path is None:
            raise ValueError("Either 'folder_path' or 'data_path' must be provided")
        folder_path = os.path.abspath(os.path.expanduser(folder_path))
        
        super().__init__(*args, **kwargs)
        
        if not os.path.isdir(folder_path):
            raise ValueError(f"Folder path does not exist: {folder_path}")

        def _find_single(prefix: str) -> str:
            matches = [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if prefix in f.lower() and f.endswith(".mat")
            ]
            if not matches:
                raise ValueError(f"No {prefix} .mat file found in {folder_path}")
            if len(matches) > 1:
                raise ValueError(f"Multiple {prefix} .mat files found in {folder_path}")
            return matches[0]

        input_file = _find_single("input")
        output_file = _find_single("output")
        
        # Move to local scratch if needed
        input_file = self._move_to_local_scratch(input_file)
        output_file = self._move_to_local_scratch(output_file)
        
        self.input_path = input_file
        self.output_path = output_file
        self._input_reader = None
        self._output_reader = None
        
        # Inspect dataset shape without loading full tensors
        with h5py.File(self.input_path, "r") as input_handle:
            input_dataset = input_handle["data"]
            input_shape = input_dataset.shape  # (N, P, L, H, W)
        with h5py.File(self.output_path, "r") as output_handle:
            output_dataset = output_handle["data"]
            output_shape = output_dataset.shape  # (N, L_out, H, W)

        if len(input_shape) != 5:
            raise ValueError("Input data should have shape (N, P, L, H, W)")
        if len(output_shape) != 4:
            raise ValueError("Output data should have shape (N, L_out, H, W)")
        if input_shape[0] != output_shape[0]:
            raise ValueError("Input/output sample counts do not match.")

        self.N_max = input_shape[0]
        self.coord_channels = input_shape[1]
        self.num_layers = input_shape[2]
        self.height = input_shape[3] # H
        self.width = input_shape[4] # W
        self.output_layers = output_shape[1]
        output_spatial = (output_shape[2], output_shape[3])

        self.resolution = self.height

        if output_spatial[0] != self.height or output_spatial[1] != self.width:
            raise ValueError("Input and output resolutions must match")

        self.input_dim = self.coord_channels * self.num_layers

        
        # Configurable split: train_ratio is train + validation; the remainder is test.
        self.train_ratio = train_ratio
        self.N_train = int(self.N_max * self.train_ratio)
        self.N_test = self.N_max - self.N_train
        if self.N_train <= 0 or self.N_test <= 0:
            raise ValueError(
                f"Invalid data split: train={self.N_train}, test={self.N_test}, "
                f"N_max={self.N_max}, train_ratio={self.train_ratio}"
            )
        
        self.label_description = ",".join([f"[temp_z{i}]" for i in range(self.output_layers)])
        
        # Set N_val and N_test for post_init
        # Use 10% of training data as validation set
        self.N_val = int(self.N_train * 0.1)
        self.N_train = self.N_train - self.N_val  # Adjust training set size
        
        # Override num_trajectories to use all available data
        if self.num_trajectories == -1:
            self.num_trajectories = self.N_train
        elif self.num_trajectories > 0: 
            # Limit to available training data
            self.num_trajectories = min(self.num_trajectories, self.N_train+self.N_val)
            self.N_train = int(self.num_trajectories*0.9)
            self.N_val = self.num_trajectories-self.N_train

        # Normalization constants:
        # - load stats_json if provided;
        # - otherwise compute from the final training split (first N_train samples).
        if stats_json is not None:
            stats_path = os.path.abspath(os.path.expanduser(stats_json))
            if not os.path.isfile(stats_path):
                raise FileNotFoundError(f"Normalization JSON not found: {stats_path}")
            with open(stats_path, "r") as fh:
                stats_data = json.load(fh)
            try:
                input_stats = stats_data["input"]
                output_stats = stats_data["output"]
                input_mean = input_stats["mean"]
                input_std = input_stats["std"]
                output_mean = output_stats["mean"]
                output_std = output_stats["std"]
            except (KeyError, TypeError) as exc:
                raise ValueError(f"Invalid normalization JSON structure in {stats_path}") from exc
        else:
            input_mean, input_std, output_mean, output_std = self._compute_train_split_stats(
                n_train=self.N_train
            )

        if len(input_mean) != self.input_dim or len(input_std) != self.input_dim:
            raise ValueError(
                f"Normalization constants for inputs do not match expected channels "
                f"(expected {self.input_dim}, got mean={len(input_mean)}, std={len(input_std)})"
            )
        if len(output_mean) != self.output_layers or len(output_std) != self.output_layers:
            raise ValueError(
                f"Normalization constants for outputs do not match expected channels "
                f"(expected {self.output_layers}, got mean={len(output_mean)}, std={len(output_std)})"
            )

        mean_input = torch.tensor(input_mean, dtype=torch.float32).reshape(self.input_dim, 1, 1)
        std_input = torch.tensor(input_std, dtype=torch.float32).reshape(self.input_dim, 1, 1)
        std_input = torch.where(std_input == 0, torch.ones_like(std_input), std_input)

        mean_output = torch.tensor(output_mean, dtype=torch.float32).reshape(self.output_layers, 1, 1)
        std_output = torch.tensor(output_std, dtype=torch.float32).reshape(self.output_layers, 1, 1)
        std_output = torch.where(std_output == 0, torch.ones_like(std_output), std_output)

        self.constants = {
            "mean_input": mean_input,
            "std_input": std_input,
            "mean_output": mean_output,
            "std_output": std_output,
        }
        
        # Custom post_init for 4:1 split
        assert (
            self.N_max is not None
            and self.N_max > 0
            and self.N_max >= self.N_train + self.N_val + self.N_test
        )
        assert self.N_val is not None and self.N_val > 0
        assert self.N_test is not None and self.N_test > 0
        
        # Set data split indices
        self._eval_indices = None
        if self.which == "train":
            self.length = self.N_train
            self.start = 0
        elif self.which == "val":
            self.length = self.N_val
            self.start = self.N_train
        else:  # test
            self.length = self.N_test
            self.start = self.N_max - self.N_test # Fix the last quarter as the test set
            # ThermalBench extension: name the test-segment samples explicitly,
            # which removes the need for a tiled copy
            self._eval_indices = _resolve_eval_indices(self.N_max)
            if self._eval_indices is not None:
                self.length = len(self._eval_indices)
                self.start = 0
        
        # Compute output_dim and channel slice list for metrics
        self.output_dim = self.label_description.count(",") + 1
        descriptors, channel_slice_list = self.get_channel_lists(self.label_description)
        self.printable_channel_description = descriptors
        self.channel_slice_list = channel_slice_list

    def _compute_train_split_stats(self, n_train: int):
        """
        Compute normalization constants from the training split (first n_train samples).
        Raw input shape: (N, P, L, H, W), reshaped to (N, L*P, H, W) during training.
        Raw output shape: (N, L_out, H, W).
        """
        if n_train <= 0:
            raise ValueError(f"n_train must be > 0, got {n_train}")

        input_sum = np.zeros(self.input_dim, dtype=np.float64)
        input_sq_sum = np.zeros(self.input_dim, dtype=np.float64)
        output_sum = np.zeros(self.output_layers, dtype=np.float64)
        output_sq_sum = np.zeros(self.output_layers, dtype=np.float64)
        input_count = 0
        output_count = 0
        chunk = 64

        with h5py.File(self.input_path, "r") as input_handle, h5py.File(
            self.output_path, "r"
        ) as output_handle:
            in_ds = input_handle["data"]
            out_ds = output_handle["data"]

            for start in range(0, n_train, chunk):
                end = min(start + chunk, n_train)
                # (B, P, L, H, W) -> (B, L, P, H, W) -> (B, L*P, H, W)
                in_batch = in_ds[start:end]
                in_batch = np.transpose(in_batch, (0, 2, 1, 3, 4)).reshape(
                    end - start, self.input_dim, self.height, self.width
                )
                in_batch = in_batch.astype(np.float64, copy=False)
                input_sum += in_batch.sum(axis=(0, 2, 3))
                input_sq_sum += np.square(in_batch).sum(axis=(0, 2, 3))
                input_count += (end - start) * self.height * self.width

                out_batch = out_ds[start:end].astype(np.float64, copy=False)  # (B, L_out, H, W)
                output_sum += out_batch.sum(axis=(0, 2, 3))
                output_sq_sum += np.square(out_batch).sum(axis=(0, 2, 3))
                output_count += (end - start) * self.height * self.width

        input_mean = input_sum / input_count
        output_mean = output_sum / output_count
        input_var = np.maximum(input_sq_sum / input_count - np.square(input_mean), 0.0)
        output_var = np.maximum(output_sq_sum / output_count - np.square(output_mean), 0.0)
        input_std = np.sqrt(input_var)
        output_std = np.sqrt(output_var)

        return (
            input_mean.tolist(),
            input_std.tolist(),
            output_mean.tolist(),
            output_std.tolist(),
        )
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index relative to the current split (train/test)
            
        Returns:
            Dictionary with 'pixel_values' and 'labels'
        """
        # Get absolute index
        if getattr(self, "_eval_indices", None) is not None:
            abs_idx = self._eval_indices[idx]
        else:
            abs_idx = idx + self.start
        # Read input: (P, L, H, W)
        inputs = torch.from_numpy(
            self._get_input_dataset()[abs_idx]
        ).type(torch.float32)  # (P, L, H, W)
        inputs = inputs.permute(1, 0, 2, 3).reshape(
            self.input_dim,
            self.height,
            self.width,
        )
        
        # Read output: (L_out, H, W)
        labels = torch.from_numpy(
            self._get_output_dataset()[abs_idx]
        ).type(torch.float32)
        
        # Normalize
        inputs = (inputs - self.constants["mean_input"]) / self.constants["std_input"]
        labels = (labels - self.constants["mean_output"]) / self.constants["std_output"]
        
        return {
            "pixel_values": inputs,
            "labels": labels,
        }
    
    def __del__(self):
        """Close file readers when dataset is deleted."""
        if hasattr(self, '_input_reader') and self._input_reader is not None:
            try:
                self._input_reader.close()
            except:
                pass
        if hasattr(self, '_output_reader') and self._output_reader is not None:
            try:
                self._output_reader.close()
            except:
                pass
    
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_input_reader"] = None
        state["_output_reader"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._input_reader = None
        self._output_reader = None

    def _get_input_dataset(self):
        if self._input_reader is None:
            self._input_reader = h5py.File(self.input_path, "r")
        return self._input_reader["data"]

    def _get_output_dataset(self):
        if self._output_reader is None:
            self._output_reader = h5py.File(self.output_path, "r")
        return self._output_reader["data"]
