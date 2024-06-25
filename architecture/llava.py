import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    LlavaNextForConditionalGeneration,
)


class LlavaEncoder(nn.Module):
    def __init__(self, model_name: str, prompt_init_text: str):
        super().__init__()

        if model_name not in (
            "llava-hf/llava-1.5-7b-hf",
            "llava-hf/llava-v1.6-mistral-7b-hf",
        ):
            raise NotImplementedError(f"unsupported model: {model_name}")

        self.processor = AutoProcessor.from_pretrained(model_name)
        if model_name == "llava-hf/llava-1.5-7b-hf":
            model_cls = LlavaForConditionalGeneration
            prompt_fmt = "USER: <image>\n<prompt> ASSISTANT:"
        else:
            model_cls = LlavaNextForConditionalGeneration
            prompt_fmt = "[INST] <image>\n<prompt> [/INST]"

        prompt = prompt_fmt.replace("<prompt>", prompt_init_text)
        inputs_ids = self.processor(
            prompt, return_tensors="pt"
        ).input_ids.reshape(-1)

        image_token_idx = inputs_ids.numpy().tolist().index(32000)

        self.register_buffer(
            "prefix_tokens",
            inputs_ids[: image_token_idx + 1].clone().detach(),
        )
        self.soft_prompt_len = len(inputs_ids) - image_token_idx - 1
        soft_prompt_tokens = inputs_ids[-self.soft_prompt_len :]

        self.model = model_cls.from_pretrained(model_name)

        # Determine soft_prompt sequence length based on # of tokens that
        # appear after the <image> token in the init_text
        self.soft_prompt = nn.Parameter(
            self.model.get_input_embeddings()(soft_prompt_tokens)
        )

        # During the forward pass, we'll simply set the last soft_prompt_len
        # tokens of the inputs embeds to the soft prompt

    def log_prob(
        self,
        images: torch.Tensor,
        completion_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        # arg images should be raw images w/o preprocessing
        assert images.dtype == torch.uint8
        b, h, w, c = images.shape
        assert c == 3

        pixel_values = self.processor.image_processor.preprocess(
            images, return_tensors="pt"
        ).pixel_values.to(0)
        image_outputs = self.model.vision_tower(
            pixel_values, output_hidden_states=True
        )

        # Compute inputs_embeds (including merged image features)
        inputs_ids = torch.cat(
            [self.prefix_tokens.repeat(b, 1), completion_tokens],
            dim=1,
        )
        inputs_embeds = self.model.get_input_embeddings()(inputs_ids)

        selected_image_feature = image_outputs.hidden_states[
            self.model.config.vision_feature_layer
        ]
        vision_feature_select_strategy = (
            self.model.config.vision_feature_select_strategy
        )

        if vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature
        else:
            raise ValueError(
                f"Unexpected select feature strategy: {self.model.config.vision_feature_select_strategy}"
            )

        image_features = self.model.multi_modal_projector(
            selected_image_feature
        )
        full_attention_mask = torch.cat(
            [
                attention_mask.new_ones((b, len(self.prefix_tokens))),
                attention_mask,
            ],
            dim=1,
        )
        # print(f"before merging: {inputs_embeds.shape=}")
        inputs_embeds, merged_attn_mask, labels, position_ids = (
            self.model._merge_input_ids_with_image_features(
                image_features,
                inputs_embeds,
                inputs_ids,
                full_attention_mask,
                labels=None,
            )
        )

        # print(f"after merging: {inputs_embeds.shape=}")
        inputs_embeds = torch.cat(
            [inputs_embeds, self.soft_prompt.repeat(b, 1, 1)],
            dim=1,
        )
        # print(f"after cat soft prompt: {inputs_embeds.shape=}")

        logits = self.model(
            inputs_embeds=inputs_embeds, attention_mask=merged_attn_mask
        ).logits
        prediction_logits = logits[:, -completion_tokens.shape[1] - 1 : -1]
        assert len(prediction_logits.shape) == 3
        assert prediction_logits.shape[:2] == completion_tokens.shape
        log_probs = -F.cross_entropy(
            rearrange(prediction_logits, "b t h -> (b t) h", b=b),
            rearrange(completion_tokens, "b t -> (b t)", b=b),
            reduction="none",
        )
        log_probs = rearrange(log_probs, "(b t) -> b t", b=b)
        log_probs = torch.where(attention_mask, log_probs, 0.0).mean(dim=1)
        assert log_probs.shape == (b,)
        return log_probs

    def generate(self, images):
        pass
