---
type: Pitfall
title: Module.update mutates arrays in place — snapshots without copies are lies
description: MLXNN's Module.update rebinds parameter arrays in place, so any "originals" captured as bare references silently track the new values. unfuseLoRA restored nothing, ever — and produced output bit-identical to not unfusing at all.
tags: [mlx-swift, lora, aliasing, silent-failure, weight-loading]
timestamp: 2026-08-13T00:00:00Z
---

`MLXNN.Module.update(parameters:)` does not replace parameter objects — it
mutates the existing `MLXArray` instances in place. Every reference you held
before the update now reads the new values.

`fuseWeights` snapshotted `model.parameters().flattened()` into a dictionary and
stored those references as the unfuse originals. The moment the fused batch
applied, every captured "original" became the fused value. `unfuseLoRA` then
faithfully restored the state it was meant to undo: **a perfect no-op**, in both
the plain and quantized branches, since the day it was written.

# How it surfaced, and why never before

A stage-2 refinement that unfused its adapter produced output **bit-identical**
(PSNR ∞ on frame 0) to a run that kept it fused — the only observable that could
not be explained away. A round-trip test on a 2-layer transformer (no need to
load 46 GB: only key paths matter) showed all 20 fused weights still fused after
unfuse, and a probe pinned the cause: the captured original differed from a
pristine copy by exactly the LoRA delta.

No shipped path had ever exercised unfuse: LipDub forbids it by design, and the
CLI fuses for the life of the process. The IC-LoRA upscale pipeline is the first
flow that needs to drop an adapter mid-run — which is also why the reference
implementation builds stage 2 as a *separate* `DiffusionStage` with `loras=()`
instead of unfusing anything.

# The rules

* **Copy at capture** (`w + 0`) anything you intend to restore later, and
  **materialise the copies before the update** — a lazy `w + 0` must read `w`'s
  buffer while it still holds the old value.
* **A snapshot taken after the mutation aliases the live state.** A test that
  compares "after" against a reference-captured "before" passes trivially; the
  baseline must itself be a copy. (The round-trip test asserts both.)
* Diffing two module snapshots taken at different times tells you nothing — they
  are the same objects. Compare against copies only.

# What this corrects in the record

July's "double-delta trap" entry documented unfuse restoring contaminated
weights when a generic LoRA was fused on top of the destructively-fused LipDub.
The truth was broader: unfuse restored contaminated weights **in every case**.
The July guards were guarding a mechanism that was already broken for everyone.
