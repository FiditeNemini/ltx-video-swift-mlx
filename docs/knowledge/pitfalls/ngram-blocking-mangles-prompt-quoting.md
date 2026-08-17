---
type: Pitfall
title: no_repeat_ngram bans quoting the prompt — enhancer timestamps come out mangled
description: HF semantics include the prompt in the ban window, so the Gemma 4 enhancer cannot copy "From 00:08.000 to 00:14.000" verbatim and emits "From the 0008.0" instead; the duration head then over-predicts by ~5 s. Measured by an ngram on/off A/B; the reference space shows the same pressure.
tags: [prompt-enhancement, no-repeat-ngram, duration-head, gemma4, root-cause]
timestamp: 2026-08-16T00:00:00Z
---

`no_repeat_ngram_size=5` (upstream's `GEMMA4_ENHANCE_GENERATION_KWARGS`)
follows HF semantics: the banned-n-gram window covers **prompt + generated**
tokens. A raw prompt with an explicit timeline therefore forbids its own
faithful quotation — copying "From 00:08.000 to 00:14.000" verbatim is a
repeated 5-gram. The greedy decoder routes around the ban with degraded
spellings ("at the 00:500 mark", "From the 0008.0 and continues through the
0014.0 mark").

Measured (same raw prompt, bf16 E2B, greedy, single variable):
- ngram 5: timestamps mangled; the duration head — which reads the enhanced
  text since the upstream-order fix — predicts 19.04 s on a 14 s choreography.
- ngram off: every timestamp verbatim, no repetition loops in this caption;
  the same text family predicts 14.04 s.

The reference space exhibits the same pressure ("at 00:500" for 00:00.500),
so parity is preserved by keeping ngram 5 — this pitfall documents a
*reference* limitation, not a port bug.

# The defense

- Known consequence chain: timeline-style prompts → mangled timestamps →
  duration over-prediction → choreography stretched past its own end (the
  "car drifts backward" symptom class).
- Fixed 2026-08-17: gemma-4-swift-mlx 1.2.0 ships `includePromptInWindow`;
  the enhancer passes `noRepeatNGramIncludesPrompt: false` — a deliberate
  deviation from HF semantics and from the reference space. Measured on the
  same raw prompt: duration prediction fell from 19.04 s to 14.71 s (14 s
  choreography) and the timeline reads intact end to end; loop protection
  stays active within the generated text.
- If parity with the reference space's exact captions ever matters more
  than fidelity, flip the flag back — one argument per call site.

# Citations

[1] A/B smoke runs 2026-08-16, seed 42, prompt_2cv_raw: ngram 5 vs nil.
[2] Space behaviour: diffusers/LTX-2.4-Prompt-Enhancer output on the same
    prompt shows "at 00:500" / "from 01:000".
