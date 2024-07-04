import glob
import logging
import os
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from byog.logger import ctqdm

log = logging.getLogger(__name__)


class Batch(NamedTuple):
    obs_embeds: torch.Tensor | np.ndarray
    txt: torch.Tensor | np.ndarray
    txt_mask: torch.Tensor | np.ndarray


@dataclass
class ContractorDatasetConfig:
    seq_len: int = 64
    frameskip: int = 4
    data_dirpaths: str = "data/contractorV3/c??/?????/"
    obs_embeds_filename: str = "vpt_bc_house_3x_embeds.npy"

    text_tokenizer_name: str = "EleutherAI/pythia-1b"


class ContractorDataset(Dataset):
    def __init__(self, cfg: ContractorDatasetConfig):
        self.cfg = cfg
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.text_tokenizer_name
        )
        df = pd.read_csv("byog/data/grounding.csv")
        assert list(df.columns) == [
            "dirpath",
            "start_t",
            "end_t",
            "annotation",
        ]

        self.real_seq_len = self.cfg.frameskip * (self.cfg.seq_len - 1) + 1

        self.episode_annotations = defaultdict(list)
        for row in ctqdm(
            df.itertuples(), total=len(df), desc="indexing annotations"
        ):
            with open(os.path.join(row.dirpath, "len.txt"), "r") as f:
                length = int(f.read())
            if length < self.real_seq_len:
                continue

            self.episode_annotations[row.dirpath].append(
                ((row.start_t + row.end_t) // 2, row.annotation)
            )

        self.index = []
        dirpaths = [
            os.path.normpath(dirpath)
            for dirpath in sorted(glob.glob(cfg.data_dirpaths))
        ]
        for dirpath in self.episode_annotations.keys():
            if dirpath not in dirpaths:
                warnings.warn(f"{dirpath=} not found")

        for dirpath in ctqdm(dirpaths, desc="indexing episodes"):
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
        dirpath, start_t = self.index[index]

        obs_embeds = load_bfloat16_npy(
            os.path.join(dirpath, self.cfg.obs_embeds_filename)
        ).astype(np.float32)[
            start_t : start_t + self.real_seq_len : self.cfg.frameskip
        ]

        txt = np.zeros(self.cfg.seq_len, dtype=np.int64)
        txt_mask = np.zeros(self.cfg.seq_len, dtype=bool)

        for midpoint_t, annotation in self.episode_annotations[dirpath]:
            tokens = self.text_tokenizer.encode(annotation)
            txt_mid_idx = (midpoint_t - start_t) // self.cfg.frameskip
            txt_start_idx = txt_mid_idx - len(tokens) // 2
            txt_stop_idx = txt_start_idx + len(tokens)

            for i in range(max(0, txt_start_idx), min(txt_stop_idx, len(txt))):
                txt[i] = tokens[i - txt_start_idx]
                txt_mask[i] = True

        return Batch(obs_embeds=obs_embeds, txt=txt, txt_mask=txt_mask)


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


if __name__ == "__main__":
    dataset = ContractorDataset(ContractorDatasetConfig())
    print(f"{len(dataset)=}")

    found = False
    for i in range(len(dataset)):
        if dataset.index[i][0] == "data/contractorV3/c10/00076":
            traj = dataset[i]
            print(i, dataset.index[i])
            print("".join("1" if x else "0" for x in traj.txt_mask))
            print(dataset.text_tokenizer.decode(traj.txt[traj.txt_mask]))
            found = True
    print(found)
