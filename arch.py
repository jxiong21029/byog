import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    LlavaNextForConditionalGeneration,
)


class LlavaVLM(nn.Module):
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

        self.prompt_tokens = self.processor.tokenizer.encode(prompt_init_text)

        prompt = prompt_fmt.replace("<prompt>", prompt_init_text)
        prompt_tokens = self.processor(prompt).input_ids.view(-1)
        image_token_idx = prompt_tokens.numpy().tolist().index(32000)

        self.image_prefix_len = image_token_idx + 1
        self.soft_prompt_len = len(prompt_tokens) - self.image_prefix_len

        soft_prompt_tokens = prompt_tokens[-self.soft_prompt_len :]

        self.model = model_cls.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto",
        )

        # Determine soft_prompt sequence length based on # of tokens that
        # appear after the <image> token in the init_text
        self.soft_prompt = nn.Parameter(
            self.model.get_input_embeddings()(soft_prompt_tokens)
        )

        # During the forward pass, we'll simply change the last
        # self.soft_prompt_len tokens of the inputs to the soft prompt

    def log_prob(
        self,
        images: torch.Tensor,
        completion_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        assert images.dtype == torch.uint8
        b, h, w, c = images.shape
        assert c == 3

        # Compute inputs_embeds (including merged image features)

        inputs_ids = torch.cat([self.prompt_tokens, completion_tokens], dim=1)
        inputs_embeds = self.model.get_input_embeddings()(inputs_ids)

        pixel_values = self.processor.image_processor.preprocess(
            images, return_tensors="pt"
        ).pixel_values
        image_outputs = self.model.vision_tower(
            pixel_values, output_hidden_states=True
        )

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
        inputs_embeds, attention_mask, labels, position_ids = (
            self.model._merge_input_ids_with_image_features(
                image_features,
                inputs_embeds,
                self.prompt_tokens,
                attention_mask,
                labels=None,
            )
        )

        inputs_embeds = torch.cat(
            [
                inputs_embeds[:, : self.image_prefix_len],
                self.soft_prompt.repeat(b, 1, 1),
                self.inputs_embeds[
                    :, self.image_prefix_len + self.soft_prompt_len :
                ],
            ],
            dim=1,
        )

        logits = self.model(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask
        ).logits

    def generate(self, images):
        pass


def main():
    model = LlavaVLM("llava-hf/llava-1.5-7b-hf", "hi")


if __name__ == "__main__":
    main()
