import logging
import os
import random
from dataclasses import dataclass

import accelerate
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from hydra.core.config_store import ConfigStore
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXForCausalLM

from byog.dataset import Batch, ContractorDataset, ContractorDatasetConfig
from byog.logger import Seri, ctqdm

log = logging.getLogger(__name__)


class Model(nn.Module):
    def __init__(self, llm_trainable=False, zero_init_projector=True):
        super().__init__()
        self.llm = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-1b")
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1b")
        self.encoder_proj = nn.Linear(256, self.llm.config.hidden_size)
        self.predictor_proj = nn.Linear(
            self.llm.config.hidden_size, self.llm.config.hidden_size
        )

        if zero_init_projector:
            with torch.no_grad():
                self.encoder_proj.weight.zero_()
                self.encoder_proj.bias.zero_()
                self.predictor_proj.weight.zero_()
                self.predictor_proj.bias.zero_()
        for param in self.llm.parameters():
            param.requires_grad_(llm_trainable)

    def forward(
        self, obs, prev_txt, output_hidden_states=False, past_key_values=None
    ):
        obs_embeds = self.encoder_proj(obs)
        txt_embeds = self.llm.get_input_embeddings()(prev_txt)
        inputs_embeds = obs_embeds + txt_embeds

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            output_hidden_states=output_hidden_states,
        )
        return outputs

    @torch.inference_mode()
    def generate(self, obs):
        """Samples obs-conditioned sequences"""

        bsize, length = obs.shape[:2]
        last = torch.as_tensor(
            self.tokenizer.bos_token_id, device=obs.device
        ).repeat(bsize, 1)
        tokens = []
        past_key_values = None
        for t in range(length):
            outputs = self(obs[:, t : t + 1], last, past_key_values)

            probs = F.softmax(outputs.logits[:, -1], dim=-1)
            last = torch.multinomial(probs, num_samples=1)
            tokens.append(last)

        return torch.cat(tokens, dim=1)

    def add_bos(self, txt: torch.Tensor):
        return torch.cat(
            [
                torch.as_tensor(
                    self.tokenizer.bos_token_id, device=txt.device
                ).repeat(txt.shape[0], 1),
                txt,
            ],
            dim=1,
        )


@dataclass
class Config:
    dataset: ContractorDatasetConfig

    lr: float = 1e-5
    weight_decay: float = 0.1
    batch_size: int = 32
    valid_size: float = 0.01

    updates_between_checkpoints: int | None = 1024
    valid_on_checkpoint: bool = True
    checkpoint_best_key: str | None = "valid/loss.data"


cs = ConfigStore.instance()
cs.store(name="config_base", node=Config)


class Trainer:
    def __init__(self, cfg: Config, out_dirpath: str):
        self.cfg = cfg
        self.rng = np.random.default_rng(0)
        self.out_dirpath = out_dirpath
        self.accelerator = accelerate.Accelerator()
        self.seri = Seri(accelerator=self.accelerator, main_process_only=True)
        self.best_checkpoint_value = None

        model = Model()

        optim = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )

        dataset = ContractorDataset(cfg.dataset)
        # labeled_idx = [
        #     i for i in range(len(dataset)) if dataset.index[i][2] is not None
        # ]
        # valid_idx = labeled_idx[
        #     len(labeled_idx) - round(len(labeled_idx) * cfg.valid_size) :
        # ]
        # valid_idx_set = set(valid_idx)
        # train_idx = [i for i in range(len(dataset)) if i not in valid_idx_set]
        # self.train_dataset = Subset(dataset, train_idx)
        # self.valid_dataset = Subset(dataset, valid_idx)

        self.train_dataset, self.valid_dataset = random_split(
            dataset, [1 - cfg.valid_size, cfg.valid_size]
        )

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,
        )
        valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            drop_last=False,
        )

        (
            self.model,
            self.optim,
            self.train_loader,
            self.valid_loader,
        ) = self.accelerator.prepare(model, optim, train_loader, valid_loader)

    def log_param_info(self):
        trainable = 0
        total = 0
        for p in self.model.parameters():
            if p.requires_grad:
                trainable += p.numel()
            total += p.numel()
        log.info(f"parameters: {trainable=:,} {total=:,}")

    def loss(self, batch: Batch):
        b, t = batch.obs_embeds.shape[:2]
        assert batch.txt.shape == (b, t)
        assert batch.txt_mask.shape == (b, t)

        assert isinstance(batch.txt, torch.Tensor)
        assert batch.txt.dtype == torch.long

        labeled_mask = batch.txt_mask.sum(dim=-1) > 0
        self.seri.tick("encoder_generate")
        txt = self.model.generate(batch.obs_embeds)
        txt = torch.where(
            rearrange(labeled_mask, "b -> b 1", b=b), batch.txt, txt
        )

        self.seri.tick("encoder_forward")
        encoder_out = self.model(
            obs=batch.obs_embeds,
            prev_txt=self.model.add_bos(txt[:, :-1]),
            output_hidden_states=True,
        )

        self.seri.tick("predictor_forward")
        prefix_size = self.rng.integers(1, t, size=(b, 1))
        prefix_size = torch.as_tensor(
            prefix_size, device=batch.obs_embeds.device
        )
        prefix_mask = (
            torch.arange(t, device=batch.obs_embeds.device).repeat(b, 1)
            < prefix_size
        )
        prefix_mask = prefix_mask.unsqueeze(-1)
        assert prefix_mask.shape == (b, t, 1)

        encoded_input = self.model.predictor_proj(
            encoder_out.hidden_states[-1]
        )
        prev_outputs = self.model.add_bos(txt[:, :-1])
        teacher_forcing = self.model.llm.get_input_embeddings()(prev_outputs)

        inputs_embeds = encoded_input * prefix_mask + teacher_forcing

        predictor_out = self.model.llm(inputs_embeds=inputs_embeds)
        logits = torch.where(
            rearrange(labeled_mask, "b -> b 1 1"),
            encoder_out.logits,
            predictor_out.logits,
        )
        target_probs = F.softmax(encoder_out.logits.detach(), dim=-1)
        target_probs = torch.where(
            rearrange(labeled_mask, "b -> b 1 1"),
            F.one_hot(txt, num_classes=encoder_out.logits.shape[-1]),
            target_probs,
        )

        logits = rearrange(logits, "b t v -> (b t) v", b=b, t=t)
        target_probs = rearrange(target_probs, "b t v -> (b t) v", b=b, t=t)
        loss = F.cross_entropy(logits, target_probs, reduction="none")
        loss = rearrange(loss, "(b t) -> b t 1", b=b, t=t)
        assert loss.shape == prefix_mask.shape
        loss = loss * ~prefix_mask

        self.seri.push(loss=loss.mean())
        if labeled_mask.any():
            self.seri.extend(loss_labeled=loss[labeled_mask].view(-1))
        if (~labeled_mask).any():
            self.seri.extend(loss_unlabed=loss[~labeled_mask].view(-1))

        return loss.mean()

    def run_epoch(self, valid: bool):
        if valid:
            self.model.eval()
            loader = self.valid_loader
        else:
            self.model.train()
            loader = self.train_loader

        with self.seri.context(name="valid" if valid else "train"):
            self.seri.tick("load_batch")
            for i, batch in enumerate(
                (ctqdm(loader) if self.accelerator.is_main_process else loader)
            ):
                loss = self.loss(batch)
                self.seri.push_vram()

                if valid:
                    self.seri.tick("load_batch")
                    continue

                self.seri.tick("backward")
                self.accelerator.backward(loss)
                self.seri.tick("optim_step")
                self.optim.step()
                self.optim.zero_grad()

                if self.cfg.updates_between_checkpoints is not None and (
                    (i + 1) % self.cfg.updates_between_checkpoints == 0
                ):
                    self.seri.step()
                    self.seri.save(os.path.join(self.out_dirpath, "seri.json"))
                    if self.cfg.valid_on_checkpoint:
                        self.run_epoch(valid=True)
                    self.checkpoint()

                self.seri.tick("load_batch")

            self.seri.step()
            self.seri.save(os.path.join(self.out_dirpath, "seri.json"))
            if not valid:
                self.run_epoch(valid=True)
                self.checkpoint()

    def checkpoint(self):
        self.seri.tick("checkpoint")
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process and (
            self.cfg.checkpoint_best_key is None
            or self.best_checkpoint_value is None
            or self.seri.data[self.cfg.checkpoint_best_key][-1]
            < self.best_checkpoint_value
        ):
            self.best_checkpoint_value = self.seri.data[
                self.cfg.checkpoint_best_key
            ][-1]

            self.accelerator.save_model(
                self.model,
                os.path.join(self.out_dirpath, "checkpoint"),
            )


@hydra.main(version_base=None, config_path="./", config_name="main")
def main(cfg: Config):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    trainer = Trainer(
        cfg,
        out_dirpath=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
    )
    trainer.log_param_info()

    trainer.run_epoch(valid=True)
    for _ in range(3):
        trainer.run_epoch(valid=False)


if __name__ == "__main__":
    main()
