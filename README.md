# BYOG: Bootstrap Your Own Grounding

Project name is a work-in-progress.

## TL;DR

Learning a language representation of a visual input stream using a
BYOL-inspired self-supervised objective.

## Approach

BYOL is typically used to embed images into a vector representation, which can
then used downstream by e.g. freezing the encoder, then training a linear probe
on the representations for image classification.

For this project, the encoder instead autoregressively generates a sequence of
discrete language tokens conditioned on a stream of visual observations. The
predictor then takes a prefix of the encoder's output logits and predicts the
remainder of the generated tokens.

Essentially, the two "views" used in BYOL, which are typically two 
augmentations of to the same input image, correspond in our case to the prefix
and the entirety of the observation sequence. The encoder and predictor,
instead of being an image embedding model (e.g. a CNN or ViT) and an MLP,
respectively, are now both sequence models---specifically, fine-tuned
decoder-only transformer language models.

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

We're using a BYOL-like objective due to theoretical properties of BYOL-like
algorithms which are difficult to explain in detail here. As a quick overview,
the reason to choose BYOL rather than contrastive algorithms is that distance
functions (e.g. Euclidean, KL / cross-entropy) used in typical contrastive
approaches for learning vector-valued representations don't make quite as much
sense for our representations, which are sequences of discrete tokens.
Meanwhile, reconstruction-based representation learning approaches do not
provide more signal than BYOL (source: trust me bro) but require the
parameterization and training of a decoder, which we want to avoid.
