# BYOG: Bootstrap Your Own Grounding

## TL;DR

Learning a language representation of a visual input stream using a
BYOL-inspired self-supervised objective.

## Motivation

Internet pretraining treats language like an outcome of intelligence, training
generative models to fit its distribution, and then delivering those language
models as the final product, e.g. as assistants, chatbots, etc.

However, from a certain perspective, language is more of an intermediary
representation used to better induce downstream (e.g. embodied) behaviors,
in yourself (reasoning) or in others (communication).

In particular, if we want agents to generalize, they need to be able to learn
to explain unexpected phenomena using language.

## Approach

We're adapting a self-supervised representation learning algorithm, BYOL,
traditionally used to embed images into a vector representation (which is then
used by e.g. freezing the encoder, then training a linear probe on the
representations for image classification).

In our case, the encoder autoregressively generates a language annotation for a
visual input (we'll be focusing on video data for now, since it naturally lends
itself to this autoregressive structure).

The predictor takes a prefix of the encoder's output logits and predicts the
remainder of the generated tokens. Essentially, the two "views" used in BYOL
correspond to the prefix and the entire observation.

```
encoder:
obs inputs:    obs1  obs2  obs3  obs4
    ..obs projection..
obs embeds:    emb1  emb2  emb3  emb4
+ prev txt:    <bos> txt1  txt2  txt3  # (autoregressive generation)
    ..transformer..
hidden states: hid1  hid2  hid3  hid4
    ..unembedding..
text outputs:  txt1  txt2  txt3  txt4

predictor:
prefix mask:        1     1     0     0
hidstate inputs: hid1  hid2  hid3  hid4
    ..apply mask..
masked inputs:   hid1  hid2     0     0
+ prev txt:      <bos> txt1  txt2  txt3  # (teacher forcing)
    ..transformer..
txt predictions: prd1  prd2  prd3  prd4

loss = sum_i xent(prd_i, txt_i) for all i where prefix mask == 0
```

Combined with co-training on labeled data, this should enable us to learn a
consistent grounding which is predictive of the input distribution, in the same
sense that BYOL representations are predictive of their inputs.

We're using a BYOL-like objective due to properties of BYOL-like / JEPA-like
algorithms which are difficult to explain in detail here. To summarize,
contrastive approaches no longer make sense since traditional distance
functions (e.g. Euclidean, KL) don't work as well for sequential
representations. Meanwhile, reconstruction-based losses do not provide more
signal than BYOL (source: trust me bro) but require the parameterization and
training of an expensive decoder, which we want to avoid.

## TODO

1. Generate obs sequences from trained model
2. Add LoRA for the LLM. Can we do separate matrices for encoder and predictor?
