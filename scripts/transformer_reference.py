"""Reference forward pass for the video transformer, and the dual-stream
video+audio transformer.

Runs Lightricks' own ``LTXModel`` on a small config with fixed random weights
and fixed inputs, and dumps everything our Swift port needs to reproduce the
result element-wise: the weights, the inputs, and the output velocity.

Small on purpose. The real checkpoint is 22B parameters, which no CPU float32
reference can hold; the arithmetic under test — RoPE, AdaLN-Single, qk-normed
attention, cross-attention, the GELU-approximate FFN, the scale-shift output —
is the same at every width, and a mismatch in any of it shows up here.

    PYTHONPATH=<ltx-core>/src python3 scripts/transformer_reference.py out.safetensors [ltx2|ltx23|av]

"ltx2" = the legacy 6-value block (no cross-attention AdaLN), video only.
"ltx23" = the 9-value block every shipped 2.3/2.5 checkpoint uses, video only.
"av" = the dual video+audio stream (issue #57 sub-task 5): same 9-value block,
plus the audio stream and the four cross-modal AdaLN modules that couple it
to video (av_ca_{video,audio}_scale_shift_adaln_single,
av_ca_{a2v,v2a}_gate_adaln_single). Video and audio are given *different*
sigmas (0.7 vs 0.3) deliberately — a bug that swaps which modality's sigma
feeds which AdaLN is a no-op when the two sigmas are equal, which is exactly
why docs/knowledge/investigations/crossmodal-adaln-sigma-swap-2026-05.md went
undetected for months on real (matched-sigma) generations.
"""
import sys
import torch
from safetensors.torch import save_file

from ltx_core.model.transformer.model import LTXModel, LTXModelType
from ltx_core.model.transformer.rope import LTXRopeType
from ltx_core.model.transformer.modality import Modality
from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
from ltx_core.tools import AudioLatentTools, VideoLatentTools
from ltx_core.types import AudioLatentShape, VideoLatentShape

OUT = sys.argv[1] if len(sys.argv) > 1 else "transformer_reference.safetensors"
VARIANT = sys.argv[2] if len(sys.argv) > 2 else "ltx2"
if VARIANT not in ("ltx2", "ltx23", "av"):
    raise SystemExit(f"unknown variant {VARIANT}")

HEADS, HEAD_DIM = 2, 8
CHANNELS, LAYERS, CAPTION_DIM = 4, 2, 16
FRAMES, HEIGHT, WIDTH = 2, 2, 3          # video latent grid -> 12 tokens
TEXT_TOKENS = 5
FPS = 24.0

torch.manual_seed(11)


def fill_from_seed(model: torch.nn.Module) -> None:
    """Default init leaves scale_shift_table empty (uninitialised memory) and
    most norms at 1; fill everything from one seed so the reference is
    reproducible."""
    with torch.no_grad():
        for _name, param in model.named_parameters():
            param.copy_(torch.randn(param.shape, dtype=torch.float32) * 0.1)


def capture(store, name):
    def hook(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        if isinstance(tensor, torch.Tensor):
            store[name] = tensor.detach().contiguous()
    return hook


if VARIANT in ("ltx2", "ltx23"):
    model = LTXModel(
        model_type=LTXModelType.VideoOnly,
        num_attention_heads=HEADS, attention_head_dim=HEAD_DIM,
        in_channels=CHANNELS, out_channels=CHANNELS, num_layers=LAYERS,
        cross_attention_dim=CAPTION_DIM, norm_eps=1e-6,
        positional_embedding_theta=10000.0,
        positional_embedding_max_pos=[20, 2048, 2048],
        rope_type=LTXRopeType("split"),
        apply_gated_attention=(VARIANT == "ltx23"),
        cross_attention_adaln=(VARIANT == "ltx23"),
        use_prompt_adaln_single=(VARIANT == "ltx23"),
    ).eval().to(torch.float32)
    fill_from_seed(model)

    tools = VideoLatentTools(
        patchifier=VideoLatentPatchifier(patch_size=1),
        target_shape=VideoLatentShape(
            batch=1, channels=CHANNELS, frames=FRAMES, height=HEIGHT, width=WIDTH),
        fps=FPS,
    )
    state = tools.create_initial_state(device=torch.device("cpu"), dtype=torch.float32)

    tokens = FRAMES * HEIGHT * WIDTH
    latent = torch.randn(1, tokens, CHANNELS, dtype=torch.float32)
    context = torch.randn(1, TEXT_TOKENS, CAPTION_DIM, dtype=torch.float32)
    SIGMA = 0.7

    video = Modality(
        latent=latent,
        sigma=torch.tensor([SIGMA], dtype=torch.float32),
        timesteps=torch.full((1, tokens), SIGMA, dtype=torch.float32),
        positions=state.positions,
        context=context,
        context_mask=torch.ones(1, TEXT_TOKENS, dtype=torch.float32),
    )

    intermediates = {}
    model.patchify_proj.register_forward_hook(capture(intermediates, "patchify_proj"))
    for i, block in enumerate(model.transformer_blocks):
        block.register_forward_hook(capture(intermediates, f"block{i}"))
        block.attn1.register_forward_hook(capture(intermediates, f"block{i}.attn1"))
        block.attn2.register_forward_hook(capture(intermediates, f"block{i}.attn2"))
        block.ff.register_forward_hook(capture(intermediates, f"block{i}.ff"))

    with torch.no_grad():
        velocity, _ = model(video=video, audio=None, perturbations=None)

    tensors = {f"weight.{k}": v.contiguous() for k, v in model.state_dict().items()}
    tensors["input.latent"] = latent
    tensors["input.context"] = context
    tensors["input.positions"] = state.positions.contiguous()
    tensors["output.velocity"] = velocity.contiguous()
    for name, tensor in intermediates.items():
        tensors[f"stage.{name}"] = tensor
    save_file(tensors, OUT)
    print(f"wrote {OUT} [{VARIANT}]: {len(tensors)} tensors, velocity {tuple(velocity.shape)}, "
          f"mean|v| {velocity.abs().mean().item():.6f}")

else:  # VARIANT == "av"
    AUDIO_HEADS, AUDIO_HEAD_DIM = 2, 4
    AUDIO_CHANNELS = 4
    # No caption_projection is configured below (audio_caption_projection stays
    # None), so TransformerArgsPreprocessor._prepare_context reshapes the raw
    # context via a bare `.view(batch, -1, inner_dim)` — this only stays a
    # semantically-sound reshape when audio_cross_attention_dim equals
    # audio_inner_dim (AUDIO_HEADS * AUDIO_HEAD_DIM). Also doubles as the
    # A2V/V2A cross-modal RoPE dim, which must match for the same reason
    # (audio_to_video_attn / video_to_audio_attn are built with audio's own
    # heads/head_dim on both sides).
    AUDIO_CAPTION_DIM = AUDIO_HEADS * AUDIO_HEAD_DIM
    AUDIO_FRAMES = 5         # -> 5 audio tokens (mel_bins=1, patch_size=1)
    SIGMA_VIDEO, SIGMA_AUDIO = 0.7, 0.3

    model = LTXModel(
        model_type=LTXModelType.AudioVideo,
        num_attention_heads=HEADS, attention_head_dim=HEAD_DIM,
        in_channels=CHANNELS, out_channels=CHANNELS, num_layers=LAYERS,
        cross_attention_dim=CAPTION_DIM, norm_eps=1e-6,
        positional_embedding_theta=10000.0,
        positional_embedding_max_pos=[20, 2048, 2048],
        rope_type=LTXRopeType("split"),
        apply_gated_attention=True,
        cross_attention_adaln=True,
        use_prompt_adaln_single=True,
        audio_num_attention_heads=AUDIO_HEADS, audio_attention_head_dim=AUDIO_HEAD_DIM,
        audio_in_channels=AUDIO_CHANNELS, audio_out_channels=AUDIO_CHANNELS,
        audio_cross_attention_dim=AUDIO_CAPTION_DIM,
        audio_positional_embedding_max_pos=[20],
        av_ca_timestep_scale_multiplier=1,
    ).eval().to(torch.float32)
    fill_from_seed(model)

    video_tools = VideoLatentTools(
        patchifier=VideoLatentPatchifier(patch_size=1),
        target_shape=VideoLatentShape(
            batch=1, channels=CHANNELS, frames=FRAMES, height=HEIGHT, width=WIDTH),
        fps=FPS,
    )
    video_state = video_tools.create_initial_state(device=torch.device("cpu"), dtype=torch.float32)

    audio_tools = AudioLatentTools(
        patchifier=AudioPatchifier(patch_size=1),
        target_shape=AudioLatentShape(batch=1, channels=AUDIO_CHANNELS, frames=AUDIO_FRAMES, mel_bins=1),
    )
    audio_state = audio_tools.create_initial_state(device=torch.device("cpu"), dtype=torch.float32)

    video_tokens = FRAMES * HEIGHT * WIDTH
    video_latent = torch.randn(1, video_tokens, CHANNELS, dtype=torch.float32)
    video_context = torch.randn(1, TEXT_TOKENS, CAPTION_DIM, dtype=torch.float32)

    audio_tokens = AUDIO_FRAMES
    audio_latent = torch.randn(1, audio_tokens, AUDIO_CHANNELS, dtype=torch.float32)
    audio_context = torch.randn(1, TEXT_TOKENS, AUDIO_CAPTION_DIM, dtype=torch.float32)

    video = Modality(
        latent=video_latent,
        sigma=torch.tensor([SIGMA_VIDEO], dtype=torch.float32),
        timesteps=torch.full((1, video_tokens), SIGMA_VIDEO, dtype=torch.float32),
        positions=video_state.positions,
        context=video_context,
        context_mask=torch.ones(1, TEXT_TOKENS, dtype=torch.float32),
    )
    audio = Modality(
        latent=audio_latent,
        sigma=torch.tensor([SIGMA_AUDIO], dtype=torch.float32),
        timesteps=torch.full((1, audio_tokens), SIGMA_AUDIO, dtype=torch.float32),
        positions=audio_state.positions,
        context=audio_context,
        context_mask=torch.ones(1, TEXT_TOKENS, dtype=torch.float32),
    )

    intermediates = {}

    def capture_av_block(name):
        def hook(_module, _inputs, output):
            vout, aout = output
            if vout is not None:
                intermediates[f"{name}.video"] = vout.x.detach().contiguous()
            if aout is not None:
                intermediates[f"{name}.audio"] = aout.x.detach().contiguous()
        return hook

    for i, block in enumerate(model.transformer_blocks):
        block.register_forward_hook(capture_av_block(f"block{i}"))

    # The four cross-modal AdaLN modules — each called once per forward pass,
    # shared by every block. This is exactly where a sigma swap or a missing
    # av_ca_factor would show up (crossmodal-adaln-sigma-swap-2026-05.md).
    model.av_ca_video_scale_shift_adaln_single.register_forward_hook(
        capture(intermediates, "cross_scale_shift_video"))
    model.av_ca_audio_scale_shift_adaln_single.register_forward_hook(
        capture(intermediates, "cross_scale_shift_audio"))
    model.av_ca_a2v_gate_adaln_single.register_forward_hook(
        capture(intermediates, "cross_gate_a2v"))
    model.av_ca_v2a_gate_adaln_single.register_forward_hook(
        capture(intermediates, "cross_gate_v2a"))

    with torch.no_grad():
        video_velocity, audio_velocity = model(video=video, audio=audio, perturbations=None)

    tensors = {f"weight.{k}": v.contiguous() for k, v in model.state_dict().items()}
    tensors["input.video_latent"] = video_latent
    tensors["input.video_context"] = video_context
    tensors["input.video_positions"] = video_state.positions.contiguous()
    tensors["input.audio_latent"] = audio_latent
    tensors["input.audio_context"] = audio_context
    tensors["input.audio_positions"] = audio_state.positions.contiguous()
    tensors["output.video_velocity"] = video_velocity.contiguous()
    tensors["output.audio_velocity"] = audio_velocity.contiguous()
    for name, tensor in intermediates.items():
        tensors[f"stage.{name}"] = tensor
    save_file(tensors, OUT)
    print(f"wrote {OUT} [av]: {len(tensors)} tensors, "
          f"video velocity {tuple(video_velocity.shape)} mean|v| {video_velocity.abs().mean().item():.6f}, "
          f"audio velocity {tuple(audio_velocity.shape)} mean|v| {audio_velocity.abs().mean().item():.6f}")
