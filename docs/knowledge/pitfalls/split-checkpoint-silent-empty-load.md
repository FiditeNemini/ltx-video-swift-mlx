---
type: Pitfall
title: A wrong-file lookup in a split checkpoint fails silently
description: Reading a component from the wrong file of an LTX-2.5 split checkpoint returns zero keys instead of raising, leaving the module randomly initialised. The VAE encoder failed this way and the generation still finished — coherent video, conditioning image silently encoded to noise.
tags: [ltx-2.5, weight-loading, checkpoints, silent-failure, vae]
timestamp: 2026-08-12T00:00:00Z
---

LTX-2.3 ships one file holding transformer, connectors and video VAE. LTX-2.5
ships one file per component. Every prefix-filtering loader in this package was
written against the unified layout, and **a prefix that matches nothing returns
an empty dictionary — not an error**.

The first end-to-end LTX-2.5 run demonstrated the consequence. `loadVAEEncoder`
still looked for `vae.encoder.*` inside the *transformer* file, which for a split
checkpoint contains only `model.diffusion_model.*`. Zero keys matched,
`applyVAEEncoderWeights` applied zero updates, and `VideoEncoder` kept MLX's
random initialisation. Nothing threw.

What that looked like from the outside:

* the run completed normally, in the expected time
* the video was **coherent and high quality** — a red vintage car on gravel in
  front of a hedge, lifting off with swirling dust, fire trails igniting on cue
* the first frame was a flat grey blob
* the car was simply **not the car in the conditioning image**

The image was VAE-encoded to noise, so the appended guide token carried noise;
the model ignored it and generated from the prompt alone. Only the wrong-car
detail exposed it — a text-to-video run would have looked perfect.

# The rule

Any loader that filters weights by prefix must treat an empty result as a
failure, because random initialisation is indistinguishable from success at the
API level and often indistinguishable from success at the *output* level too.

Applied here:

* `LTXCheckpointSource` centralises which file each component comes from, so the
  question is answered once per layout instead of at each call site.
* `loadVAEEncoder` throws when the encoder dictionary is empty.
* The Gemma 4 loader goes further and throws when *any* declared parameter is
  unfed: an encoder weight left at random init corrupts every prompt embedding,
  which no visual inspection would catch.
* A gated regression test (`LTX25_MODELS_DIR`) asserts each component lands from
  its own file and that every `VideoEncoder` parameter is fed.

# Related trap: the load report is not a verdict

`applyTransformerWeights` prints `loaded / unmatched / missing`. On a healthy
LTX-2.5 transformer it prints **1 unmatched** — `keyframes_abs_pos_embedding`,
which only the DFR generated-keyframe path uses. `missing: 0` is the number that
matters; `unmatched` counts checkpoint keys this package has no module for, which
is expected whenever upstream ships a feature we do not implement.
