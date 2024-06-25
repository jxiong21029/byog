import torch

from byog.architecture.llava import LlavaEncoder


class LikelihoodReinforceTrainer:
    def __init__(self):
        # TODO: dataset which yields paired videos
        pass

    def run_epoch(self, valid=False):
        # TODO:
        # 1. sample a batch of video pairs
        # 2. generate completions for all videos
        # 3. compute predictor loss := completion log-likelihood
        # 4. compute encoder loss := REINFORCE-like objective
        #   - we probably have two optimizers (different LRs), completely
        #   separate models
        # 5. backwards and optim step
        pass


def main():
    model = LlavaEncoder(
        "llava-hf/llava-1.5-7b-hf", prompt_init_text="Describe the image."
    ).to(0, dtype=torch.bfloat16)


if __name__ == "__main__":
    main()
