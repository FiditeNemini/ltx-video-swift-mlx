---
type: Pitfall
title: A parameter whose name contains a dot silently never loads
description: ModuleParameters.unflattened reads "." as a module boundary, so a flat @ParameterInfo key like "per_channel_statistics.mean-of-means" matches by name, updates nothing, and keeps its init value. Strict key checking does not catch it — only verifying the values after the update does.
tags: [weight-loading, mlx-swift, silent-failure, root-cause]
timestamp: 2026-08-19T00:00:00Z
---

The LTX-2.5 diffusion decoder ships its latent statistics as
`per_channel_statistics.mean-of-means` / `.std-of-means`. Declaring them with
those exact names as `@ParameterInfo` keys looks right and *passes a strict
loader*: the name is present in `parameters().flattened()`, so nothing is
reported missing. But `ModuleParameters.unflattened` splits keys on `.`, so the
update is delivered to a nested path (`per_channel_statistics` → `mean-of-means`)
that no flat parameter answers to. The parameter keeps its initialisation —
mean 0, std 1.

Symptom: every decode came out washed out (8-bit range [68, 154] instead of
[0, 255], pixel σ 6.6 against the conv decoder's 51.2) because the latent was
never un-normalised. Nothing errored; the load reported all weights applied.

The convolutional decoder escaped this only because its keys were already
renamed to `mean_of_means` / `std_of_means` at map time.

# The defense

- Never give a parameter a key containing `.` — use underscores and map the
  checkpoint's name in the loader.
- A loader that checks key *names* before updating proves nothing. Re-read
  `parameters().flattened()` **after** `update` and compare values against the
  checkpoint tensors; `DiffVAEWeightLoader` does this and throws on any
  parameter that did not take its value.
- Same family as [[module-update-mutates-in-place]]: in MLX Swift, the update
  path is where silent no-ops live. Assert on values, never on names.

# Citations

[1] Root-caused 2026-08-19 while bringing up the diffusion video decoder;
    activation probe showed the context volume an order of magnitude too small.
