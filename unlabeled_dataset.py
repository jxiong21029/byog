import glob
import os
from dataclasses import dataclass  # , field
from typing import NamedTuple

import numpy as np

# import pandas as pd
import torch
import tqdm
from torch.utils.data import Dataset

# from transformers import AutoTokenizer

# from vpt_llm.dataset.prompt_utils import EVENT_NAMES_PRESENT, FOODS, TOOLS


class Trajectory(NamedTuple):
    obs_embeds: torch.Tensor | np.ndarray
    # txt: torch.Tensor | np.ndarray
    # txt_mask: torch.Tensor | np.ndarray


@dataclass
class TrajectoryDatasetConfig:
    seq_len: int = 32
    frameskip: int = 8
    obs_embeds_filename: str = "vpt_bc_house_3x_embeds.npy"
    # include_event_types: list[str] = field(
    #     default_factory=lambda: [
    #         "craft_item",
    #         "drop",
    #         "kill_entity",
    #         "pickup",
    #         "eat_item",
    #         "place_block",
    #     ]
    # )

    # text_tokenizer_name: str = "EleutherAI/pythia-1b"
    debug_mode: bool = False


class TrajectoryDataset(Dataset):
    def __init__(self, cfg: TrajectoryDatasetConfig):
        self.cfg = cfg
        self.index = []
        # self.text_tokenizer = AutoTokenizer.from_pretrained(
        #     self.cfg.text_tokenizer_name
        # )
        self.real_seq_len = cfg.frameskip * (cfg.seq_len - 1) + 1

        dirpaths = sorted(glob.glob("data/contractorV3/c??/?????"))
        if cfg.debug_mode:
            dirpaths = dirpaths[:10]

        for dirpath in tqdm.tqdm(dirpaths, desc="indexing dataset"):
            with open(os.path.join(dirpath, "len.txt"), "r") as f:
                length = int(f.read())

            if length < self.real_seq_len:
                continue

            num_segments = round(length / self.real_seq_len)
            for t in np.linspace(0, length - self.real_seq_len, num_segments):
                self.index.append((dirpath, round(t)))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        dirpath, start = self.index[index]

        obs_embeds = load_bfloat16_npy(
            os.path.join(dirpath, self.cfg.obs_embeds_filename)
        ).astype(np.float32)[
            start : start + self.real_seq_len : self.cfg.frameskip
        ]
        assert len(obs_embeds) == self.cfg.seq_len

        # annotations = get_annotations(
        #     dirpath, tuple(self.cfg.include_event_types)
        # )
        # annotations = [
        #     (t, annotation)
        #     for t, annotation in annotations
        #     if start <= t < start + self.cfg.seq_len
        # ]

        # txt = np.zeros(self.cfg.seq_len, dtype=np.int64)
        # txt_mask = np.zeros(self.cfg.seq_len, dtype=bool)
        # for t, annotation in annotations:
        #     tokens = self.text_tokenizer.encode(annotation)
        #     for i, tok in enumerate(tokens):
        #         if t + i - start >= self.cfg.seq_len:
        #             break
        #         txt[t + i - start] = tok
        #         txt_mask[t + i - start] = True

        return Trajectory(
            obs_embeds=obs_embeds,
            # txt=txt,
            # txt_mask=txt_mask,
        )


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

        # print(np.where(traj.txt_mask)[0])
        # print(dataset.text_tokenizer.decode(traj.txt[traj.txt_mask]))


if __name__ == "__main__":
    smoke_test()
