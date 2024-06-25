import os
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class Trajectory(NamedTuple):
    obs_embeds: torch.Tensor | np.ndarray
    txt: torch.Tensor | np.ndarray
    txt_mask: torch.Tensor | np.ndarray


@dataclass
class TrajectoryDatasetConfig:
    seq_len_obs: int = 32
    frameskip: int = 8
    seq_len_txt: int = 32
    obs_embeds_filename: str = "vpt_bc_house_3x_embeds.npy"

    text_tokenizer_name: str = "EleutherAI/pythia-1b"
    debug_mode: bool = False


class TrajectoryDataset(Dataset):
    def __init__(self, cfg: TrajectoryDatasetConfig):
        self.cfg = cfg
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.text_tokenizer_name
        )
        self.df = pd.read_csv("byog/data/grounding.csv")
        assert list(self.df.columns) == [
            "dirpath",
            "start_t",
            "end_t",
            "annotation",
        ]

        self.real_seq_len_obs = (
            self.cfg.frameskip * (self.cfg.seq_len_obs - 1) + 1
        )

        mask = []
        for row in self.df.itertuples():
            with open(os.path.join(row.dirpath, "len.txt"), "r") as f:
                length = int(f.read())
            mask.append(length < self.real_seq_len_obs)
        self.df = self.df[mask]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        start_t = max(0, row.end_t - self.real_seq_len_obs)

        obs_embeds = load_bfloat16_npy(
            os.path.join(row.dirpath, self.cfg.obs_embeds_filename)
        ).astype(np.float32)[
            start_t : start_t + self.real_seq_len_obs : self.cfg.frameskip
        ]

        txt = self.text_tokenizer.encode(row.annotation)[
            : self.cfg.seq_len_txt
        ]
        txt_mask = np.zeros(self.cfg.seq_len_txt, dtype=bool)
        txt_mask[: len(txt)] = True
        txt = np.concatenate(
            [txt, np.zeros_like(txt, shape=self.cfg.seq_len_txt - len(txt))],
            axis=0,
        )

        return Trajectory(obs_embeds=obs_embeds, txt=txt, txt_mask=txt_mask)


def load_bfloat16_npy(filepath: str):
    from ml_dtypes import bfloat16

    with open(filepath, "rb") as fp:
        version = np.lib.format.read_magic(fp)
        shape, fortran_order, _ = np.lib.format._read_array_header(fp, version)
        if len(shape) == 0:
            count = 1
        else:
            count = np.multiply.reduce(shape, dtype=np.int64)
        array = np.fromfile(fp, dtype=bfloat16, count=count)
    if fortran_order:
        array.shape = shape[::-1]
        array = array.transpose()
    else:
        array.shape = shape
    return array


# def get_annotations(dirpath, include_event_types):
#     events = pd.read_csv(os.path.join(dirpath, "events.csv"))

#     use_item_mask = events.key == "use_item"
#     events["key"] = events.key.where(
#         ~(use_item_mask & events.value.isin(FOODS)), other="eat_item"
#     )
#     events["key"] = events.key.where(
#         ~(use_item_mask & events.value.isin(TOOLS)), other="use_tool"
#     )
#     events["key"] = events.key.where(
#         ~(events.key == "use_item"), other="place_block"
#     )

#     annotations = []
#     prev = None
#     for row in events.itertuples():
#         if (
#             prev is not None
#             and prev.key == row.key
#             and prev.value == row.value
#         ):
#             annotations[-1][0] = row.t
#             continue

#         if row.key not in include_event_types:
#             continue

#         event_name = EVENT_NAMES_PRESENT[row.key]
#         event_name = event_name[0].upper() + event_name[1:]
#         item_name = row.value.replace("_", " ")
#         annotations.append([row.t, f"{event_name} {item_name}."])

#         prev = row

#     return annotations


def smoke_test():
    import time

    dataset = TrajectoryDataset(TrajectoryDatasetConfig(debug_mode=True))

    for i in range(10):
        start_time = time.time()
        traj = dataset[i]
        print("iter_time", time.time() - start_time)

        print(traj.obs_embeds.shape, traj.obs_embeds.dtype)
        print(traj.obs_embeds.mean())
        print(np.linalg.norm(traj.obs_embeds, axis=-1).mean())

        print(np.where(traj.txt_mask)[0])
        print(dataset.text_tokenizer.decode(traj.txt[traj.txt_mask]))


if __name__ == "__main__":
    smoke_test()
