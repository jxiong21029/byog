import glob
import os
from dataclasses import dataclass

import cv2
import tqdm
from torch.utils.data import Dataset

from vpt_llm.utils.image import grid_collate
from vpt_llm.utils.prompting import gamma_corrected


def load_paired_frame_grids(
    dirpath,
    t0,
    interval,
    height,
    width,
    from_bgr=True,
    correct_gamma=True,
    add_labels=True,
):
    filepath = os.path.join(dirpath, "spaced_frames.mp4")
    assert interval % 10 == 0
    real_interval = interval // 10
    real_start_t = t0 // 10

    cap = cv2.VideoCapture(filepath)
    frames = []
    count = height * width * 2
    try:
        if real_start_t > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, real_start_t)
        t = real_start_t
        while True:
            success, frame = cap.read()
            if success:
                if (t - real_start_t) % real_interval == 0:
                    if from_bgr:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if correct_gamma:
                        frame = gamma_corrected(frame)
                    frames.append(frame)
                    if len(frames) >= count:
                        break
                t += 1
            else:
                break
    finally:
        cap.release()
    if len(frames) < count:
        raise ValueError("not enough frames to form grid")

    image1 = grid_collate(
        frames[: height * width],
        grid_height=height,
        grid_width=width,
        add_labels=add_labels,
        labels_font_size=2.0,
        labels_font_thickness=3,
    )
    image2 = grid_collate(
        frames[height * width :],
        grid_height=height,
        grid_width=width,
        add_labels=add_labels,
        labels_font_size=2.0,
        labels_font_thickness=3,
    )
    return image1, image2


@dataclass
class VPTPairedDatasetConfig:
    grid_interval: int
    grid_height: int
    grid_width: int
    resize_height: int | None = None
    resize_width: int | None = None
    debug_mode: bool = False


class VPTPairedDataset(Dataset):
    def __init__(self, cfg: VPTPairedDatasetConfig):
        self.cfg = cfg
        self.index = []
        dirpaths = sorted(glob.glob("data/contractorV2/*/*/"))

        if cfg.debug_mode:
            dirpaths = dirpaths[:10]

        step = cfg.grid_interval * cfg.grid_height * cfg.grid_width
        for dirpath in tqdm.tqdm(dirpaths, desc="indexing dataset"):
            with open(os.path.join(dirpath, "len"), "r") as f:
                episode_len = int(f.read())

            t = 0
            while t + 2 * step < episode_len:
                self.index.append((dirpath, t))
                t += step

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        dirpath, t = self.index[index]
        im1, im2 = load_paired_frame_grids(
            dirpath,
            t,
            self.cfg.grid_interval,
            self.cfg.grid_height,
            self.cfg.grid_width,
            correct_gamma=False,
            add_labels=True,
        )
        if self.cfg.resize_height is not None:
            im1 = cv2.resize(
                im1,
                (self.cfg.resize_width, self.cfg.resize_height),
                interpolation=cv2.INTER_AREA,
            )
            im2 = cv2.resize(
                im2,
                (self.cfg.resize_width, self.cfg.resize_height),
                interpolation=cv2.INTER_AREA,
            )
        return im1, im2


def main():
    dataset = VPTPairedDataset(
        VPTPairedDatasetConfig(
            grid_interval=10,
            grid_height=2,
            grid_width=2,
            debug_mode=True,
        )
    )
    print(len(dataset))

    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=4)
    batch = next(iter(loader))
    print(type(batch), len(batch))
    print(batch[0].shape)


if __name__ == "__main__":
    main()
