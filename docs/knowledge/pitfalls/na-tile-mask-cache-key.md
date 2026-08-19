---
type: Pitfall
title: A tiled-attention mask cache must key on the whole window pattern
description: Border tiles clamp their windows, interior tiles slide them; keyed on "first start / last end" the two collide, so a border tile silently reuses an interior mask. Costs ~8% error per affected stage, invisible until you diff against a reference.
tags: [attention, tiling, diffvae, cache, root-cause]
timestamp: 2026-08-19T00:00:00Z
---

`NeighborhoodAttention3D` splits large volumes into query tiles and builds one
additive mask per tile *geometry*, reusing it across tiles that share it. The
first key was `(query shape, key lengths, first start, last end)` per axis —
which looks discriminating and is not: near a border every window clamps to the
same span, so a border tile and an interior tile can agree on all four while
their per-query windows differ.

Measured on the LTX-2.5 diffusion decoder: stages 0-2 were exact to 1e-6 and
stage 3 — the first volume large enough for the tiling to actually split —
came out 8% off, carrying 7% into the context volume and 5.5% into the
predicted pixels. The output was a plausible, slightly washed-out image; only a
reference diff located it.

# The defense

- Key on the full relative window arrays, not a summary of them.
- Equivalence tests must *force* a split. The original tiled-vs-brute-force
  tests all fitted in one tile, so the tiled path — the only reason the code
  exists — was never exercised. `scoreBudget` is settable for that reason, and
  the suite is serialized because it is shared state.
- General lesson: any cache keyed on a *derived summary* of the thing it
  describes is a correctness bug waiting for the case where the summary stops
  being injective.

# Citations

[1] Root-caused 2026-08-19 with the DiffVAE parity harness
    (`scripts/diffvae_reference.py`); parity went 0.070 → 1.0e-6 on the fix.
