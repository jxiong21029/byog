import glob
import logging
import os
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
    seq_len_obs: int = 32
    obs_frameskip: int = 8
    seq_len_txt: int = 32
    obs_embeds_filename: str = "vpt_bc_house_3x_embeds.npy"

    text_tokenizer_name: str = "EleutherAI/pythia-1b"
    debug_mode: bool = False


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

        self.real_seq_len_obs = (
            self.cfg.obs_frameskip * (self.cfg.seq_len_obs - 1) + 1
        )

        self.index = []
        if cfg.debug_mode:
            df = df.sample(10)
        for row in ctqdm(
            df.itertuples(), total=len(df), desc="indexing labeled data"
        ):
            with open(os.path.join(row.dirpath, "len.txt"), "r") as f:
                length = int(f.read())
            if length < self.real_seq_len_obs:
                continue

            start_t = max(0, row.end_t - self.real_seq_len_obs)
            self.index.append((row.dirpath, start_t, row.annotation))

        dirpaths = sorted(glob.glob("data/contractorV3/c??/?????"))
        if cfg.debug_mode:
            dirpaths = dirpaths[:10]
        for dirpath in ctqdm(dirpaths, desc="indexing unlabed data"):
            with open(os.path.join(dirpath, "len.txt"), "r") as f:
                length = int(f.read())

            if length < self.real_seq_len_obs:
                continue

            num_segments = round(length / self.real_seq_len_obs)
            for t in np.linspace(
                0, length - self.real_seq_len_obs, num_segments
            ):
                self.index.append((dirpath, round(t), None))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        dirpath, start_t, annotation = self.index[index]

        obs_embeds = load_bfloat16_npy(
            os.path.join(dirpath, self.cfg.obs_embeds_filename)
        ).astype(np.float32)[
            start_t : start_t + self.real_seq_len_obs : self.cfg.obs_frameskip
        ]

        txt = (
            self.text_tokenizer.encode(annotation)[: self.cfg.seq_len_txt]
            if annotation is not None
            else []
        )
        txt = np.array(txt, dtype=np.int64)
        txt_mask = np.zeros(self.cfg.seq_len_txt, dtype=bool)
        txt_mask[: len(txt)] = True
        txt = np.concatenate(
            [txt, np.zeros_like(txt, shape=self.cfg.seq_len_txt - len(txt))],
            axis=0,
        )

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
