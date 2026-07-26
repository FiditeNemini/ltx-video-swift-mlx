---
type: Pitfall
title: The LipDub prompt must contain the literal dialogue text
description: A scene-only prompt gives the IC-LoRA nothing to lip-sync to — structurally wrong mouth poses regardless of how correct the pipeline is.
tags: [lipdub, prompt, ic-lora]
timestamp: 2026-07-16T00:00:00Z
---

The LipDub IC-LoRA was trained on prompts of the form:

```
[Scene/character description], speaking in [LANGUAGE] saying: "[ACTUAL DIALOGUE TEXT]"
```

The English wrapper is constant even when the dialogue is in another
language; the dialogue uses the target language's own script (Cyrillic,
Hanzi, …). The text prompt drives WHAT is said; the audio reference provides
the voice. A prompt like *"a man speaking French in a podcast studio"*
(scene only) produces structurally wrong mouth poses — verified user-facing
failure, not a theory. Single speaker only; match dialogue length to the
clip; negative prompt is irrelevant (distilled, no CFG).

# The dialogue must be what the audio actually says

Stronger than "include some dialogue": the text **drives the generated
speech**. LipDub denoises a video stream and an audio stream jointly; the
audio it produces follows the prompt's dialogue, and the lips follow that
generated audio. When the prompt text and the reference audio disagree, the
model speaks the *prompt* and the result cannot be in sync with the
reference — no amount of audio conditioning repairs it.

Verified (July 2026), reference audio saying *"FluxForge Studio transforme
votre Mac en un studio de création IA complet…"* against a prompt carrying a
different marketing line: the generated audio transcribes back as the
**prompt text, word for word**, and every timing measurement against the
reference was meaningless. Re-running with the audio's own transcript
brought the offsets from 0.5–0.9 s down to 0.2–0.3 s.

Practical consequence for integrators: the caller almost always *has* the
exact text (it drove the TTS) — pass that, not a paraphrase. When the text is
unknown (a supplied recording), transcribe it first; a STT pass is cheap next
to a generation. Any measurement of lip-sync quality is void unless prompt
and audio agree.

# The defense

- Always format `generateLipDub` / `lipdub` CLI prompts with
  `speaking in <LANG> saying: "<TEXT>"`, where `<TEXT>` is the **verbatim
  transcript of the target audio**.
- The VLM prompt-enhancement path can rephrase or drop the wrapper —
  `LTXPipeline` repairs it (`speaking|speaks|saying|says` + `in` detection,
  wrapper re-glued when lost). Keep that repair when touching enhancement.

# Citations

[1] Lightricks ComfyUI workflow `LTX-2.3_ICLoRA_Lipdub_Two_Stage_Distilled.json`
(master branch), linked from the
[HF model card](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-LipDub).
