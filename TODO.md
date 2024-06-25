# TODO

## Motivation

The real goal of this project is to attempt to address problems which internet
pretraining can't solve. Which is, in a sense, just the transfer learning
problem, and not a very new problem at all.

We're trying to ground a observations from some domain in language, but given
limited or no access to language annotations for that domain.

1. Why language? Especially if language is more of a communication medium than
a requirement for thought or representation.
    - The point is to learn language, which is useful on its own (human
    interpretability, consumption by downstream language models), in a domain
    with little to no high-quality language annotations.
2. Why learning? Can we achieve more than what's possible through traditional
VLM prompting / in-context learning / fine-tuning methods?
    - Hopefully, we would do better than prompting / in-context learning, and
    require less data than fine-tuning

## Modeling

We're essentially learning a language model as a world model (which itself is
not a super novel idea...)

The encoder takes in a stream of observations and outputs a stream of text
(embeddings). The predictor takes those embeddings and does
next-token-prediction, trying to predict future text (embeddings) without
conditioning on future observations.

Combined with co-training on grounded data, this should enable us to learn a
consistent grounding which is predictive for the new domain, in the same sense
that BYOL representations are predictive.

## TODO

1. Dataset providing observation sequences
2. Encoder model annotates with language tokens
3. Predictor model does parallel decoding, K tokens forward @ each index.
    - Teacher-forced language modeling loss, conditioned on encoder's final
    hidden state
4. Encoder trained to optimize predictor

Experiment 1:
1. Input: obs embeds from VPT (because cheap)
2. Grounding: programmatic annotations (because all we have)
3. Encoder: obs -> projector -> pythia
4. Predictor: stock-standard pythia
