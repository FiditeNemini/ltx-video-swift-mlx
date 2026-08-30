---
type: Pitfall
title: The text connector's register replacement reordered tokens instead of substituting in place
description: Every real prompt is left-padded and hit this — 135% relative error against the reference, collapsing to 0.15% once fixed.
tags: [text-encoder, connector, root-cause, parity-harness, high-impact]
timestamp: 2026-08-30T00:00:00Z
---

`Embeddings1DConnector.replacePaddedWithLearnableRegisters`
(`Sources/LTXVideo/Models/TextEncoder/LTXTextEncoder.swift`) is the step that
substitutes learnable register tokens for padded positions before the
connector's 8-block transformer runs. The Swift port sorted valid tokens to
the front of the sequence, then filled the *reversed*-mask tail with
registers — a comment claimed this matched PyTorch's behavior. It does not.

ltx_core's `_replace_padded_with_learnable_registers` is a plain
position-preserving `where`: `binary_mask * hidden_states + (1 -
binary_mask) * registers`. A padded position `t` gets `registers[t]` **in
place** — no reordering at all.

This is not an edge case: `Gemma4TextEncoder.encode`
(`Sources/LTXVideo/Models/TextEncoder/Gemma4/Gemma4TextEncoder.swift`) left-pads
every prompt — `mask = [0]*padding + [1]*ids.count` — and prompts are
routinely far shorter than the 1024-token window. Essentially every real
generation this repo has ever produced exercised this code path.

Found by [`ConnectorParityTests`](../../../Tests/LTXVideoTests/ConnectorParityTests.swift)
(issue #57 sub-task 3), which runs the port against Lightricks' own
`FeatureExtractorV2` + `Embeddings1DConnector` on synthetic Gemma hidden
states (`scripts/connector_reference.py`). Bisection first cleared the
8-block transformer itself — RoPE, gated attention, feed-forward all matched
the reference to ~1e-5 when fed the reference's own post-register hidden
state directly — which localized the defect to register replacement.
Measured relative error on the full connector output, with the repo's actual
left-padding convention: **135%** before the fix, **0.15%** after.

# The defense

- A comment asserting behavioral parity ("matching PyTorch behavior") is a
  claim, not a fact — it must cite what the reference code actually does, the
  way `scripts/*_reference.py` read real checkpoint metadata instead of
  trusting a framework default (see
  [conv-decoder-wrong-spatial-padding.md](conv-decoder-wrong-spatial-padding.md)
  for the same failure mode one layer down).
- Padding side matters for how visibly a masking bug manifests. This repo
  left-pads; a synthetic test using right-padding here would have shown a
  smaller, easier-to-dismiss divergence than the 135% the real convention
  produces — build reference/test inputs from the port's actual usage
  pattern, not whichever convention is easiest to write.
- Bisecting against a reference tap *upstream* of the suspected component
  (feeding the transformer blocks the reference's own post-register state,
  bypassing the port's private register-replacement entirely) is what
  isolated this from the transformer math in one step, instead of auditing
  RoPE/attention/gating first.

# Citations

[1] `ConnectorParityTests.bisectFirstDivergence` / `connectorOutputMatchesReference`,
measured relative error 1.35 (full connector output, left-padded, matching
`Gemma4TextEncoder.encode`'s convention) before the fix, 1.5e-3 after, against
`scripts/connector_reference.py` run on CPU float32 via Lightricks' own
`FeatureExtractorV2` + `Embeddings1DConnector` (2026-08-30).
