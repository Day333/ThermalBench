import json
import os

import h5py
import numpy as np
import torch

from scOT.problems.base import BaseDataset


class ThermalTransient3D(BaseDataset):
    """
    Transient 3D thermal dataset.

    Data format convention:
    - Input .mat: (N, P, T, L, H, W)
    - Output .mat: (N, T, L, H, W)

    We flatten (P, T, L) into input channels and (T, L) into output channels to
    match steady-style training/evaluation interfaces.
    """

    def __init__(self, folder_path=None, *args, **kwargs):
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

        input_file = self._move_to_local_scratch(input_file)
        output_file = self._move_to_local_scratch(output_file)

        self.input_path = input_file
        self.output_path = output_file
        self._input_reader = None
        self._output_reader = None

        with h5py.File(self.input_path, "r") as input_handle:
            input_shape = input_handle["data"].shape  # (N, P, T, L, H, W)
        with h5py.File(self.output_path, "r") as output_handle:
            output_shape = output_handle["data"].shape  # (N, T, L, H, W)

        if len(input_shape) != 6:
            raise ValueError("Input data should have shape (N, P, T, L, H, W)")
        if len(output_shape) != 5:
            raise ValueError("Output data should have shape (N, T, L, H, W)")
        if input_shape[0] != output_shape[0]:
            raise ValueError("Input/output sample counts do not match.")
        if input_shape[2] != output_shape[1] or input_shape[3] != output_shape[2]:
            raise ValueError("Input and output temporal/layer dimensions must match")
        if input_shape[4] != output_shape[3] or input_shape[5] != output_shape[4]:
            raise ValueError("Input and output resolutions must match")

        self.N_max = input_shape[0]
        self.coord_channels = input_shape[1]
        self.num_times = input_shape[2]
        self.num_layers = input_shape[3]
        self.height = input_shape[4]
        self.width = input_shape[5]

        self.output_times = output_shape[1]
        self.output_layers = output_shape[2]
        self.resolution = self.height

        self.input_dim = self.coord_channels * self.num_times * self.num_layers
        # Flatten P, T, and L into input channels.
        self.output_dim = self.output_times * self.output_layers
        # Flatten T and L into output channels.

        self.train_ratio = train_ratio
        self.N_train = int(self.N_max * self.train_ratio)
        self.N_test = self.N_max - self.N_train
        if self.N_train <= 0 or self.N_test <= 0:
            raise ValueError(
                f"Invalid data split: train={self.N_train}, test={self.N_test}, "
                f"N_max={self.N_max}, train_ratio={self.train_ratio}"
            )

        self.label_description = ",".join(
            [
                f"[temp_t{t}_z{z}]"
                for t in range(self.output_times)
                for z in range(self.output_layers)
            ]
        )
        descriptors, channel_slice_list = self.get_channel_lists(self.label_description)
        self.printable_channel_description = descriptors
        self.channel_slice_list = channel_slice_list

        self.N_val = int(self.N_train * 0.1)
        self.N_train = self.N_train - self.N_val

        if self.num_trajectories == -1:
            self.num_trajectories = self.N_train
        elif self.num_trajectories > 0:
            self.num_trajectories = min(self.num_trajectories, self.N_train + self.N_val)
            self.N_train = int(self.num_trajectories * 0.9)
            self.N_val = self.num_trajectories - self.N_train

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
        if len(output_mean) != self.output_dim or len(output_std) != self.output_dim:
            raise ValueError(
                f"Normalization constants for outputs do not match expected channels "
                f"(expected {self.output_dim}, got mean={len(output_mean)}, std={len(output_std)})"
            )

        mean_input = torch.tensor(input_mean, dtype=torch.float32).reshape(self.input_dim, 1, 1)
        std_input = torch.tensor(input_std, dtype=torch.float32).reshape(self.input_dim, 1, 1)
        std_input = torch.where(std_input == 0, torch.ones_like(std_input), std_input)

        mean_output = torch.tensor(output_mean, dtype=torch.float32).reshape(self.output_dim, 1, 1)
        std_output = torch.tensor(output_std, dtype=torch.float32).reshape(self.output_dim, 1, 1)
        std_output = torch.where(std_output == 0, torch.ones_like(std_output), std_output)

        self.constants = {
            "mean_input": mean_input,
            "std_input": std_input,
            "mean_output": mean_output,
            "std_output": std_output,
        }

        assert (
            self.N_max is not None
            and self.N_max > 0
            and self.N_max >= self.N_train + self.N_val + self.N_test
        )
        assert self.N_val is not None and self.N_val > 0
        assert self.N_test is not None and self.N_test > 0

        if self.which == "train":
            self.length = self.N_train
            self.start = 0
        elif self.which == "val":
            self.length = self.N_val
            self.start = self.N_train
        else:
            self.length = self.N_test
            self.start = self.N_max - self.N_test

    def __getitem__(self, idx):
        abs_idx = idx + self.start

        # input sample: (P, T, L, H, W) -> (T, L, P, H, W) -> (P*T*L, H, W)
        inputs = torch.from_numpy(self._get_input_dataset()[abs_idx]).type(torch.float32)
        inputs = inputs.permute(1, 2, 0, 3, 4).contiguous().view(self.input_dim, self.height, self.width)
        # output sample: (T, L, H, W) -> (T*L, H, W)
        labels = torch.from_numpy(self._get_output_dataset()[abs_idx]).type(torch.float32)
        labels = labels.contiguous().view(self.output_dim, self.height, self.width)

        inputs = (inputs - self.constants["mean_input"]) / self.constants["std_input"]
        labels = (labels - self.constants["mean_output"]) / self.constants["std_output"]

        return {
            "pixel_values": inputs,
            "labels": labels,
        }

    def __del__(self):
        if hasattr(self, "_input_reader") and self._input_reader is not None:
            try:
                self._input_reader.close()
            except Exception:
                pass
        if hasattr(self, "_output_reader") and self._output_reader is not None:
            try:
                self._output_reader.close()
            except Exception:
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

    def _compute_train_split_stats(self, n_train: int):
        """
        Compute normalization constants from the training split (first n_train samples).
        Raw input shape: (N, P, T, L, H, W), reshaped to (N, T*L*P, H, W) during training.
        Raw output shape: (N, T, L, H, W), reshaped to (N, T*L, H, W) during training.
        """
        if n_train <= 0:
            raise ValueError(f"n_train must be > 0, got {n_train}")

        input_sum = np.zeros(self.input_dim, dtype=np.float64)
        input_sq_sum = np.zeros(self.input_dim, dtype=np.float64)
        output_sum = np.zeros(self.output_dim, dtype=np.float64)
        output_sq_sum = np.zeros(self.output_dim, dtype=np.float64)
        input_count = 0
        output_count = 0
        chunk = 32

        with h5py.File(self.input_path, "r") as input_handle, h5py.File(
            self.output_path, "r"
        ) as output_handle:
            in_ds = input_handle["data"]
            out_ds = output_handle["data"]

            for start in range(0, n_train, chunk):
                end = min(start + chunk, n_train)

                # (B, P, T, L, H, W) -> (B, T, L, P, H, W) -> (B, T*L*P, H, W)
                in_batch = in_ds[start:end]
                in_batch = np.transpose(in_batch, (0, 2, 3, 1, 4, 5)).reshape(
                    end - start, self.input_dim, self.height, self.width
                )
                in_batch = in_batch.astype(np.float64, copy=False)
                input_sum += in_batch.sum(axis=(0, 2, 3))
                input_sq_sum += np.square(in_batch).sum(axis=(0, 2, 3))
                input_count += (end - start) * self.height * self.width

                # (B, T, L, H, W) -> (B, T*L, H, W)
                out_batch = out_ds[start:end].reshape(
                    end - start, self.output_dim, self.height, self.width
                )
                out_batch = out_batch.astype(np.float64, copy=False)
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

    def _get_input_dataset(self):
        if self._input_reader is None:
            self._input_reader = h5py.File(self.input_path, "r")
        return self._input_reader["data"]

    def _get_output_dataset(self):
        if self._output_reader is None:
            self._output_reader = h5py.File(self.output_path, "r")
        return self._output_reader["data"]
