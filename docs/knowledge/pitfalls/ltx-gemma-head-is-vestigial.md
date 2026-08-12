---
type: Pitfall
title: Do not validate the LTX Gemma encoder by generating text
description: LTX-2.5's bundled Gemma 4 is an encoder fine-tune whose final-norm scale drifted 2.5x above stock. Its tied LM head sits in the logit-softcap saturation zone and emits single capital letters on any prompt — while the hidden states it exists to produce are healthy.
tags: [ltx-2.5, text-encoder, gemma, validation]
timestamp: 2026-08-12T00:00:00Z
---

Greedy decoding is the natural way to prove a language model loaded correctly:
shapes and finiteness survive a permuted layer order or a swapped q/k
projection, but coherent text does not. It was the first check written against
the LTX-2.5 text encoder, and it **failed** — `"The capital of France is"`
continued as `"SSSS"`.

The weights were fine. The check was wrong for this checkpoint.

# What is actually going on

`gemma4-12b-ltx-v1` is an encoder fine-tune with tied embeddings. Its
`model.norm.weight` has drifted far above stock Gemma 4's:

| | LTX-2.5 encoder | `mlx-community/gemma-4-e4b-it-4bit` |
|---|---|---|
| `model.norm.weight` mean | 20.1 | 7.9 |
| `model.norm.weight` max | 600 | 14 |
| `layers.0.input_layernorm.weight` | mean 6.6, range −143…193 | mean 10.1, range 3.9…92 |
| `layers.0.self_attn.q_norm.weight` | constant 1.0234 | constant 0.9844 |

The last two rows are the ones that matter for a *loading* question: both
checkpoints share the same conventions — RMSNorm weights are **direct scales**
(the `1 + w` offset is already folded, which is why `q_norm` sits at ~1 rather
than ~0), and `q_norm` is uniform across the head dimension. So `MLXNN.RMSNorm`
is the right norm, and Gemma4Swift loads this checkpoint the same way it loads a
stock one.

The final-norm scale is the difference, and it is a consequence of how LTX
trained the model: the connector consumes hidden states, which the feature
extractor **per-token RMS-normalises** before projecting, so the absolute scale
of the last layer is unconstrained by the training objective and was free to
drift. LTX never reads logits.

Measured through our Swift stack on a 6-token prompt, the per-layer RMS is
healthy — 1.08 at the embeddings, 0.54–5.4 across the 48 layers, then 46.4 after
the final norm. Feeding a 46-RMS hidden state into a tied head whose embedding
rows have RMS 0.017 over 3840 dims puts the pre-cap logits far past
`final_logit_softcapping = 30`, where `tanh` compresses every candidate to
~29.9 and the ranking collapses onto embedding magnitude. The winners are single
capital letters.

# What to check instead

The checks that do discriminate, in increasing cost:

1. **Every declared parameter is fed.** Throw, don't warn — see
   [the silent-empty-load pitfall](/docs/knowledge/pitfalls/split-checkpoint-silent-empty-load.md).
2. **Scale stays in band across layers.** An exploding or collapsing RMS is the
   signature of a mis-applied norm convention.
3. **Meaning survives.** Mean-pool the last hidden state over real tokens and
   confirm a paraphrase ranks above an unrelated sentence. A mis-mapped stack
   cannot fake this.
4. **A real generation.** The end-to-end run is what caught the VAE-encoder bug
   that none of the above could see.

Elementwise parity against the PyTorch reference remains the only check that
would prove bit-level correctness, and is still open.
