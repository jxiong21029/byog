# BYOG: Bootstrap Your Own Grounding

## encoder spec

- prompt format: e.g. `[INST] <image>\n<prompt> [/INST]`
  - prompt: e.g. `[INST] <image>\nDescribe what the player is doing. [/INST]`
- this gets tokenized to:
  - (tokens for `[INST] `) 32000 (tokens for `\n<prompt> [/INST]`...)
  - 32000 is the token for `<image>`
  - 32000 gets replaced with a sequence of image embedding tokens (e.g. 576
  tokens for llava-1.5)
  - `LlavaEncoder.prefix_tokens` stores the tokens for `[INST] ` and the
  `<image>` token
  - so, we compute input embeds normally for `[INST] <image>` (prefix and image
  tokens, should be ~580 after image tokens are merged in) and then append a
  `LlavaEncoder.soft_prompt` to those embeddings, which is initialized equal to
  the token embeddings of `\n<prompt> [/INST]`