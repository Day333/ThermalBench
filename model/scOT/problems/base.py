"""
Dataset selector and base classes for thermal problems.
"""

from torch.utils.data import Dataset, ConcatDataset
from typing import Optional, List, Dict
from abc import ABC
import re
import os
import shutil
from accelerate.utils import broadcast_object_list


def get_dataset(dataset, **kwargs):
    """
    Get a dataset by name.
    If you enter a list of str, will return a ConcatDataset of the datasets.

    Available choices are:
    - thermal.steady.ThermalSteady3D
    - thermal.transient.ThermalTransient3D

    **kwargs overwrite the default settings.
    """
    if isinstance(dataset, list):
        return ConcatDataset([get_dataset(d, **kwargs) for d in dataset])

    if "thermal" in dataset:
        if "thermal.steady" in dataset:
            if "ThermalSteady3D" in dataset:
                from .thermal.steady import ThermalSteady3D as dset
            else:
                raise ValueError(f"Unknown dataset {dataset}")
        elif "thermal.transient" in dataset:
            if "ThermalTransient3D" in dataset:
                from .thermal.transient import ThermalTransient3D as dset
            else:
                raise ValueError(f"Unknown dataset {dataset}")
        else:
            raise ValueError(f"Unknown dataset {dataset}")
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    return dset(**kwargs)


class BaseDataset(Dataset, ABC):
    """A base class for all datasets."""

    def __init__(
        self,
        which: Optional[str] = None,
        num_trajectories: Optional[int] = None,
        data_path: Optional[str] = "./data",
        move_to_local_scratch: Optional[str] = None,
    ) -> None:
        """
        Args:
            which: Which dataset to use, i.e. train, val, or test.
            num_trajectories: The number of trajectories to use for training.
            data_path: The path to the data files.
            move_to_local_scratch: If not None, move the data to this directory at
                dataset initialization and use it from there.
        """
        assert which in ["train", "val", "test"]
        assert num_trajectories is not None and (
            num_trajectories > 0 or num_trajectories in [-1, -2, -8]
        )

        self.num_trajectories = num_trajectories
        self.data_path = data_path
        self.which = which
        self.move_to_local_scratch = move_to_local_scratch

    def _move_to_local_scratch(self, file_path):
        if self.move_to_local_scratch is not None:
            data_dir = os.path.join(self.data_path, file_path)
            file = file_path.split("/")[-1]
            scratch_dir = self.move_to_local_scratch
            dest_dir = os.path.join(scratch_dir, file)
            RANK = int(os.environ.get("LOCAL_RANK", -1))
            if not os.path.exists(dest_dir) and (RANK == 0 or RANK == -1):
                print(f"Start copying {file} to {dest_dir}...")
                shutil.copy(data_dir, dest_dir)
                print("Finished data copy.")
            ls = broadcast_object_list([dest_dir], from_process=0)
            dest_dir = ls[0]
            return dest_dir
        else:
            return file_path

    def post_init(self) -> None:
        """
        Call after self.N_max, self.N_val, self.N_test, as well as the file_paths
        and normalization constants are set.
        """
        assert (
            self.N_max is not None
            and self.N_max > 0
            and self.N_max >= self.N_val + self.N_test
        )
        if self.num_trajectories == -1:
            self.num_trajectories = self.N_max - self.N_val - self.N_test
        elif self.num_trajectories == -2:
            self.num_trajectories = (self.N_max - self.N_val - self.N_test) // 2
        elif self.num_trajectories == -8:
            self.num_trajectories = (self.N_max - self.N_val - self.N_test) // 8
        assert self.num_trajectories + self.N_val + self.N_test <= self.N_max
        assert self.N_val is not None and self.N_val > 0
        assert self.N_test is not None and self.N_test > 0
        if self.which == "train":
            self.length = self.num_trajectories
            self.start = 0
        elif self.which == "val":
            self.length = self.N_val
            self.start = self.N_max - self.N_val - self.N_test
        else:
            self.length = self.N_test
            self.start = self.N_max - self.N_test

        self.output_dim = self.label_description.count(",") + 1
        descriptors, channel_slice_list = self.get_channel_lists(self.label_description)
        self.printable_channel_description = descriptors
        self.channel_slice_list = channel_slice_list

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx) -> Dict:
        pass

    @staticmethod
    def get_channel_lists(label_description):
        matches = re.findall(r"\[([^\[\]]+)\]", label_description)
        channel_slice_list = [0]
        beautiful_descriptors = []
        for match in matches:
            channel_slice_list.append(channel_slice_list[-1] + 1 + match.count(","))
            splt = match.split(",")
            if len(splt) > 1:
                beautiful_descriptors.append("".join(splt))
            else:
                beautiful_descriptors.append(match)
        return beautiful_descriptors, channel_slice_list


class BaseTimeDataset(BaseDataset, ABC):
    """A base class for time dependent problems."""

    def __init__(
        self,
        *args,
        max_num_time_steps: Optional[int] = None,
        time_step_size: Optional[int] = None,
        fix_input_to_time_step: Optional[int] = None,
        allowed_time_transitions: Optional[List[int]] = None,
        **kwargs,
    ) -> None:
        assert max_num_time_steps is not None and max_num_time_steps > 0
        assert time_step_size is not None and time_step_size > 0
        assert fix_input_to_time_step is None or fix_input_to_time_step >= 0

        super().__init__(*args, **kwargs)
        self.max_num_time_steps = max_num_time_steps
        self.time_step_size = time_step_size
        self.fix_input_to_time_step = fix_input_to_time_step
        self.allowed_time_transitions = allowed_time_transitions

    def _idx_map(self, idx):
        i = idx // self.multiplier
        _idx = idx - i * self.multiplier

        if self.fix_input_to_time_step is None:
            t1, t2 = self.time_indices[_idx]
            assert t2 >= t1
            t = t2 - t1
        else:
            t1 = self.fix_input_to_time_step
            t2 = self.time_step_size * (_idx + 1) + self.fix_input_to_time_step
            t = t2 - t1
        return i, t, t1, t2

    def post_init(self) -> None:
        assert (
            self.N_max is not None
            and self.N_max > 0
            and self.N_max >= self.N_val + self.N_test
        )
        if self.num_trajectories == -1:
            self.num_trajectories = self.N_max - self.N_val - self.N_test
        elif self.num_trajectories == -2:
            self.num_trajectories = (self.N_max - self.N_val - self.N_test) // 2
        elif self.num_trajectories == -8:
            self.num_trajectories = (self.N_max - self.N_val - self.N_test) // 8
        assert self.num_trajectories + self.N_val + self.N_test <= self.N_max
        assert self.N_val is not None and self.N_val > 0
        assert self.N_test is not None and self.N_test > 0
        assert self.max_num_time_steps is not None and self.max_num_time_steps > 0

        if self.fix_input_to_time_step is not None:
            self.multiplier = self.max_num_time_steps
        else:
            self.time_indices = []
            for i in range(self.max_num_time_steps + 1):
                for j in range(i, self.max_num_time_steps + 1):
                    if (
                        self.allowed_time_transitions is not None
                        and (j - i) not in self.allowed_time_transitions
                    ):
                        continue
                    self.time_indices.append(
                        (self.time_step_size * i, self.time_step_size * j)
                    )
            self.multiplier = len(self.time_indices)

        if self.which == "train":
            self.length = self.num_trajectories * self.multiplier
            self.start = 0
        elif self.which == "val":
            self.length = self.N_val * self.multiplier
            self.start = self.N_max - self.N_val - self.N_test
        else:
            self.length = self.N_test * self.multiplier
            self.start = self.N_max - self.N_test

        self.output_dim = self.label_description.count(",") + 1
        descriptors, channel_slice_list = self.get_channel_lists(self.label_description)
        self.printable_channel_description = descriptors
        self.channel_slice_list = channel_slice_list
