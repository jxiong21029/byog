import torch
import torch.nn as nn


class LlamaPredictor(nn.Module):
    def __init__(self):
        pass

    def log_prob(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        prediction_mask: torch.Tensor,
    ):
        """Returns log-prob of text tokens at prediction_mask

        attention_mask: True where tokens are not padded
        prediction_mask: True where log-prob should be included in retval
        """
        assert not ((~attention_mask) & prediction_mask).any()
