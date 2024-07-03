# BYOG: Bootstrap Your Own Grounding

## TL;DR

Learning a language representation of a visual input stream using a
BYOL-inspired self-supervised objective.

<!-- ## Motivation -->

<!-- At a high level, internet pretraining treats language like an outcome of
intelligence: we train generative models to fit the distribution of these
outputs, and then deliver those outputs as the final product, e.g. to create
assistants, chatbots, etc.

However, from a certain perspective, language is more of an intermediary
representation used to better induce downstream (e.g. embodied) behaviors,
in yourself (reasoning) or in others (communication).

The dominant paradigm to leverage language models as an intermediate component
in some larger system, is, for example, to use a frozen pretrained language
model as a text encoder to generate embeddings of an input instruction, then
feeding those embeddings to a downstream neural network policy / controller.

However, leveraging language models as just a slow lookup table for internet
data or a text encoder is fundamentally different from how language is used and
learned in humans. In particular, given that language is something that we
learn to use, then surely there is some kind of objective that 

Instead of next-token prediction,  -->

<!-- The idea behind internet pretraining is that, by training with next token
prediction on text written by humans, language models implicitly learn the
dynamics which resulted in that language, and thus implicitly learn a
representation of the world.

To me, this paradigm seems a bit backwards---humans don't learn how the world
works by getting better at using language. Instead, we use language as an
_intermediate representation_ in order to better achieve outcomes in the world,
whether by processing information (reasoning) or by sharing information with
others (communication).

In particular, a sequence of language tokens is a good representation if it
contains information useful for predicting future outcomes. This project is
about applying this principle to learn to generate language conditioned on
unlabeled sequences of observations. For this purpose, we adapt a traditional
self-supervised learning algorithm, BYOL (Bootstrap Your Own Latent; Grill et
al. 2020). In theory, this could enable an agent to learn to use language to
explain phenomena in unseen domains, improving generalizability. -->

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

<!-- Of course, the learned language representation is only useful if the language
is "grounded". However, for novel domains without ground-truth annotations,
"grounded" can be difficult to define, even for human usage of language. (After
all, people can and do invent new words to describe newly discovered
phenomena.) Perhaps the best we can do in terms of defining "grounded" is
"consistent with how language is used in other cases", which we can enforce by
simply co-training the model on data with ground-truth annotations. Here,
"consistency" arises from implicit or explicit regularization of neural
networks towards smoothness. -->

We're using a BYOL-like objective due to theoretical properties of BYOL-like
algorithms which are difficult to explain in detail here. As a quick overview,
the reason to choose BYOL rather than contrastive algorithms is that distance
functions (e.g. Euclidean, KL / cross-entropy) used in typical contrastive
approaches for learning vector-valued representations don't make quite as much
sense for our representations, which are sequences of discrete tokens.
Meanwhile, reconstruction-based representation learning approaches do not
provide more signal than BYOL (source: trust me bro) but require the
parameterization and training of a decoder, which we want to avoid.
