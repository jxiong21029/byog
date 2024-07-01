import argparse
import os

import accelerate
import torch
from omegaconf import OmegaConf

from byog.dataset import ContractorDataset, ContractorDatasetConfig
from byog.main import Model

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

dataset = ContractorDataset(ContractorDatasetConfig(debug_mode=True))
print(dataset.index[0])
obs_embeds = dataset[0].obs_embeds

tokens = model.generate(
    torch.tensor(obs_embeds, device=0, dtype=torch.bfloat16).repeat(8, 1, 1)
)
completions = model.tokenizer.batch_decode(tokens)
for completion in completions:
    print(repr(completion))
