import accelerate
import torch

from byog.dataset import ContractorDataset, ContractorDatasetConfig
from byog.main import Model

model = Model()
accelerate.load_checkpoint_and_dispatch(
    model, "byog/runs/926653/checkpoint", device_map="auto"
)
model.to(torch.bfloat16)
model.eval()

dataset = ContractorDataset(ContractorDatasetConfig(debug_mode=True))
obs_embeds = dataset[0].obs_embeds

tokens = model.generate(
    torch.tensor(obs_embeds, device=0, dtype=torch.bfloat16).repeat(8, 1, 1)
)
completions = model.tokenizer.batch_decode(tokens)
for completion in completions:
    print(completion)
