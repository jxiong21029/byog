import argparse
import os
import random

import accelerate
import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from byog.dataset import ContractorDataset, ContractorDatasetConfig
from byog.main import Model


def overlay_text_to_image(
    image: np.ndarray,
    text: list[str],
    font_size: float = 0.5,
    font_thickness=1,
):
    r"""Overlays lines of text on top of an image.

    First this will render to the left-hand side of the image, once that column is full,
    it will render to the right hand-side of the image.

    :param image: The image to put text on top.
    :param text: The list of strings which will be rendered (separated by new lines).
    :param font_size: Font size.
    :return: A new image with text overlaid on top.
    """
    h, w, c = image.shape
    font = cv2.FONT_HERSHEY_SIMPLEX

    y = 0
    left_aligned = True
    for line in text:
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        y += textsize[1] + 10
        if y > h:
            left_aligned = False
            y = textsize[1] + 10

        if left_aligned:
            x = 10
        else:
            x = w - (textsize[0] + 10)

        cv2.putText(
            image,
            line,
            (x, y),
            font,
            font_size,
            (0, 0, 0),
            font_thickness * 2,
            lineType=cv2.LINE_AA,
        )

        cv2.putText(
            image,
            line,
            (x, y),
            font,
            font_size,
            (255, 255, 255, 255),
            font_thickness,
            lineType=cv2.LINE_AA,
        )

    return np.clip(image, 0, 255)


def grid_collate(
    frames: list[np.ndarray] | np.ndarray,
    grid_height: int,
    grid_width: int,
    add_labels: bool,
    labels_font_size: float = 4.0,
    labels_font_thickness: int = 6,
) -> np.ndarray:
    assert len(frames) == grid_height * grid_width
    if add_labels:
        frames = [
            overlay_text_to_image(
                frame.copy(),
                [str(i + 1)],
                font_size=labels_font_size,
                font_thickness=labels_font_thickness,
            )
            for i, frame in enumerate(frames)
        ]

    h, w = frames[0].shape[0], frames[0].shape[1]
    combined = np.zeros_like(
        frames[0],
        shape=(h * grid_height, w * grid_width) + frames[0].shape[2:],
    )
    for i in range(grid_height):
        for j in range(grid_width):
            combined[i * h : (i + 1) * h, j * w : (j + 1) * w] = frames[
                i * grid_width + j
            ]
    return combined


def load_frame_grid(
    dirpath,
    start_t,
    interval,
    height,
    width,
    from_bgr=True,
    add_labels=True,
):
    filepath = os.path.join(dirpath, "frames.mp4")

    cap = cv2.VideoCapture(filepath)
    frames = []
    count = height * width
    try:
        if start_t > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_t)
        t = start_t
        while True:
            success, frame = cap.read()
            if success:
                if (t - start_t) % interval == 0:
                    if from_bgr:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

    image = grid_collate(
        frames,
        grid_height=height,
        grid_width=width,
        add_labels=add_labels,
        labels_font_size=2.0,
        labels_font_thickness=3,
    )
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=int)
    args = parser.parse_args()

    run_dir = f"byog/runs/{args.run}/"

    cfg = OmegaConf.load(os.path.join(run_dir, ".hydra/config.yaml"))
    model = Model(cfg)
    accelerate.load_checkpoint_and_dispatch(
        model, os.path.join(run_dir, "checkpoint"), device_map="auto"
    )
    model.to(torch.bfloat16)
    model.eval()

    dataset = ContractorDataset(ContractorDatasetConfig())
    os.makedirs(os.path.join(run_dir, "eval"), exist_ok=True)

    idx = random.sample(range(len(dataset)), k=5)
    img_filenames = []
    completions = []
    for ii, i in enumerate(idx):
        dirpath, start_t = dataset.index[i]
        obs_embeds = dataset[i].obs_embeds

        img = load_frame_grid(
            dirpath, start_t, interval=16, height=4, width=4, add_labels=False
        )
        img = Image.fromarray(img)
        img.save(os.path.join(run_dir, "eval", f"img_{ii}.jpeg"))
        img_filenames.append(f"img_{ii}.jpeg")

        # TODO modify generate() to not require passing zeros for eval...
        tokens = model.generate(
            torch.tensor(obs_embeds, device=0, dtype=torch.bfloat16).repeat(
                8, 1, 1
            ),
            torch.zeros((8, 64), device=0, dtype=torch.long),
            torch.zeros((8, 64), device=0, dtype=torch.bool),
        )
        results = model.tokenizer.batch_decode(tokens)
        # for completion in completions:
        #     print(repr(completion))
        completions.append(results)

    with open(os.path.join(run_dir, "eval", "generations.md"), "w") as f:
        for filename, results in zip(img_filenames, completions):
            f.write(f"![img]({filename})\n")
            for completion in results:
                f.write(repr(completion) + "\n\n")


if __name__ == "__main__":
    main()
