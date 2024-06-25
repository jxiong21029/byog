import glob
import os
from dataclasses import dataclass  # , field

import numpy as np
import tqdm
from torch.utils.data import Dataset


@dataclass
class TrajectoryDatasetConfig:
    seq_len: int = 32
    frameskip: int = 8
    obs_embeds_filename: str = "vpt_bc_house_3x_embeds.npy"

    debug_mode: bool = False


class TrajectoryDataset(Dataset):
    def __init__(self, cfg: TrajectoryDatasetConfig):
        self.cfg = cfg
        self.index = []
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

        return obs_embeds


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


def smoke_test():
    import time

    dataset = TrajectoryDataset(TrajectoryDatasetConfig(debug_mode=True))

    for i in range(10):
        start_time = time.time()
        obs_embeds = dataset[i]
        print("iter_time", time.time() - start_time)

        print(obs_embeds.shape, obs_embeds.dtype)
        print(obs_embeds.mean())
        print(np.linalg.norm(obs_embeds, axis=-1).mean())


if __name__ == "__main__":
    smoke_test()
