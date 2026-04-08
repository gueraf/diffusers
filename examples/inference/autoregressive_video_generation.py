"""Autoregressive video generation with rolling KV cache.

Demonstrates how to generate long videos by autoregressively denoising chunks of
frames while maintaining temporal coherence through a rolling KV cache. Uses the
Self-Forcing transformer (gueraf/Self-Forcing-diffusers) which was trained for
autoregressive chunk-based generation on top of Wan2.1-T2V-1.3B.

The rolling KV cache stores self-attention key/value projections from previously
generated chunks so that each new chunk can attend to past context without
recomputing all previous tokens.

Inference procedure:
  Self-Forcing DMD was trained with a specific 4-step stochastic schedule
  (NOT a standard ODE solver). At each step:
    1. Predict x0 from noisy input at timestep t
    2. Re-noise x0 back to the NEXT timestep t' (stochastic re-noising)
    3. Repeat until final step; use last x0 as the clean chunk

  The timesteps are derived from a FlowMatchEuler schedule with shift=5.0,
  matching the original training: approximately [1000, 938, 834, 629].
  Using UniPC or any standard ODE solver produces garbage because the model
  was not trained with those timestep sequences.

  Each chunk uses 3 latent frames (= 9 pixel frames), matching the training
  block size (num_frame_per_block=3 in the original config).

Usage:
    python examples/inference/autoregressive_video_generation.py \\
        --prompt "A cat walks on the grass, realistic" \\
        --num_chunks 27 \\
        --output output.mp4

Requirements:
    pip install diffusers transformers accelerate sentencepiece protobuf
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch
from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, WanPipeline, WanTransformer3DModel
from diffusers.hooks import RollingKVCacheConfig, apply_rolling_kv_cache, get_rolling_kv_cache_state
from diffusers.utils import export_to_video


def _build_sf_denoising_steps(device):
    """Compute the Self-Forcing DMD 4-step warped timestep schedule.

    Original: denoising_step_list=[1000, 750, 500, 250], warp_denoising_step=True,
    timestep_shift=5.0.  The warping picks timesteps[1000 - step] from a 1000-step
    FlowMatchEuler schedule with shift=5.0, giving approximately
    [1000, 938, 834, 629] in the [0, 1000] timestep range.
    """
    sched = FlowMatchEulerDiscreteScheduler(shift=5.0, num_train_timesteps=1000)
    sched.set_timesteps(1000, device="cpu")
    all_ts = torch.cat([sched.timesteps.cpu(), torch.tensor([0.0])])
    orig = torch.tensor([1000, 750, 500, 250])
    return all_ts[1000 - orig].to(device)


def generate_autoregressive_video(
    prompt: str,
    negative_prompt: str = "",
    num_chunks: int = 27,
    frames_per_chunk: int = 9,
    height: int = 480,
    width: int = 832,
    guidance_scale: float = 1.0,
    window_size: int = 8000,
    model_id: str = "gueraf/Self-Forcing-diffusers",
    wan_base_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    device: str = "cuda",
    seed: int = 0,
):
    """Generate a long video autoregressively using rolling KV cache.

    Args:
        model_id: Self-Forcing transformer repo (gueraf/Self-Forcing-diffusers).
        wan_base_model_id: Wan2.1-T2V base repo — used only for VAE, text encoder,
            tokenizer. The transformer is replaced by model_id.
        frames_per_chunk: Pixel frames per chunk. Must satisfy
            (frames_per_chunk - 1) % 4 == 0. Default 9 = 3 latent frames,
            matching the training block size (num_frame_per_block=3).
        num_chunks: Number of chunks. 27 chunks × 9 frames = 243px ≈ 15s @ 16fps.
    """
    transformer = WanTransformer3DModel.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    vae = AutoencoderKLWan.from_pretrained(wan_base_model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(
        wan_base_model_id, vae=vae, transformer=transformer, torch_dtype=torch.bfloat16
    )
    pipe.to(device)

    # Self-Forcing was trained with FlowMatchEuler + shift=5.0, NOT UniPC.
    # Replace the default scheduler so that add_noise() uses the correct sigma math.
    pipe.scheduler = FlowMatchEulerDiscreteScheduler(shift=5.0, num_train_timesteps=1000)

    # Apply rolling KV cache
    apply_rolling_kv_cache(pipe.transformer, RollingKVCacheConfig(
        window_size=window_size,
        cache_cross_attention=True,
    ))
    cache_state = get_rolling_kv_cache_state(pipe.transformer)

    # 4-step warped timestep schedule matching original Self-Forcing training
    denoising_steps = _build_sf_denoising_steps(device=device)  # ~[1000, 938, 834, 629]

    # Encode prompt once; reuse across all chunks and clean forward passes
    do_cfg = guidance_scale > 1.0
    with torch.no_grad():
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            do_classifier_free_guidance=do_cfg,
            device=device,
            dtype=torch.bfloat16,
        )

    # Latent dimensions
    p_t, p_h, p_w = pipe.transformer.config.patch_size
    z_dim = pipe.vae.config.z_dim
    latent_h = height // pipe.vae_scale_factor_spatial
    latent_w = width // pipe.vae_scale_factor_spatial
    latent_t = (frames_per_chunk - 1) // pipe.vae_scale_factor_temporal + 1

    # Frames per chunk in patch space (for frame_offset)
    ppf = latent_t // p_t

    # VAE denormalization constants for decoding
    latents_mean = (
        torch.tensor(pipe.vae.config.latents_mean)
        .view(1, z_dim, 1, 1, 1)
        .to(device, dtype=torch.float32)
    )
    latents_std_inv = (
        (1.0 / torch.tensor(pipe.vae.config.latents_std))
        .view(1, z_dim, 1, 1, 1)
        .to(device, dtype=torch.float32)
    )

    generator = torch.Generator(device=device).manual_seed(seed)
    all_frames = []

    for chunk_idx in range(num_chunks):
        print(f"Generating chunk {chunk_idx + 1}/{num_chunks}...")

        frame_offset = chunk_idx * ppf

        # Initialize with pure noise at sigma=1.0 (t=1000)
        noisy_input = torch.randn(
            (1, z_dim, latent_t, latent_h, latent_w),
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )

        # Self-Forcing DMD denoising: predict x0, re-noise to next level, repeat.
        # Cache reads on; writes off during noisy denoising (avoid storing noisy K/V).
        cache_state.should_update_cache = False
        with torch.no_grad():
            for step_idx, t in enumerate(denoising_steps):
                sigma = t / 1000.0  # ∈ [0, 1], scalar

                encoder_hs = prompt_embeds
                if do_cfg:
                    # Run unconditional and conditional passes
                    latent_input = torch.cat([noisy_input, noisy_input])
                    ts = t.long().expand(2)
                    enc = torch.cat([negative_prompt_embeds, prompt_embeds])
                    velocity = pipe.transformer(
                        latent_input, timestep=ts, encoder_hidden_states=enc,
                        frame_offset=frame_offset, return_dict=False,
                    )[0]
                    vel_uncond, vel_cond = velocity.chunk(2)
                    velocity = vel_uncond + guidance_scale * (vel_cond - vel_uncond)
                else:
                    velocity = pipe.transformer(
                        noisy_input,
                        timestep=t.long().expand(noisy_input.shape[0]),
                        encoder_hidden_states=encoder_hs,
                        frame_offset=frame_offset,
                        return_dict=False,
                    )[0]

                # Flow matching: x0 = xt - sigma * velocity
                x0_pred = (noisy_input.float() - sigma * velocity.float()).to(torch.bfloat16)

                if step_idx < len(denoising_steps) - 1:
                    # Re-noise x0 to the NEXT noise level: xt' = (1 - sigma') * x0 + sigma' * eps
                    next_sigma = denoising_steps[step_idx + 1] / 1000.0
                    eps = torch.randn_like(x0_pred)
                    noisy_input = ((1.0 - next_sigma) * x0_pred + next_sigma * eps).to(torch.bfloat16)

        chunk_latents = x0_pred

        # Populate the cache with clean K/V from these denoised latents at t=0.
        cache_state.should_update_cache = True
        with torch.no_grad():
            pipe.transformer(
                chunk_latents,
                timestep=torch.zeros(1, device=device, dtype=torch.long),
                encoder_hidden_states=prompt_embeds,
                frame_offset=frame_offset,
                return_dict=False,
            )

        # Decode this chunk for the output video
        with torch.no_grad():
            decode_latents = chunk_latents.to(dtype=pipe.vae.dtype)
            decode_latents = decode_latents / latents_std_inv + latents_mean
            chunk_video = pipe.vae.decode(decode_latents).sample
            chunk_frames = pipe.video_processor.postprocess_video(chunk_video, output_type="pil")[0]

        all_frames.extend(chunk_frames)
        print(f"  Total frames so far: {len(all_frames)}")

    if hasattr(pipe.transformer, "_diffusers_hook"):
        pipe.transformer._diffusers_hook.reset_stateful_hooks()

    return all_frames


def main():
    parser = argparse.ArgumentParser(description="Autoregressive video generation with rolling KV cache")
    parser.add_argument("--prompt", type=str, default="A cat walks on the grass, realistic style, high quality")
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="Bright tones, overexposed, static, blurred details, subtitles, worst quality, low quality",
    )
    parser.add_argument("--num_chunks", type=int, default=27,
                        help="Number of video chunks. 27 × 9px frames = 243px ≈ 15s @ 16fps")
    parser.add_argument("--frames_per_chunk", type=int, default=9,
                        help="Pixel frames per chunk (must be 4k+1). 9=3 latent frames (training block size).")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument(
        "--guidance_scale", type=float, default=1.0,
        help="CFG scale. DMD models were trained without CFG; 1.0 = no guidance.",
    )
    parser.add_argument("--window_size", type=int, default=8000, help="Rolling cache window (-1 for unlimited)")
    parser.add_argument(
        "--model_id",
        type=str,
        default="gueraf/Self-Forcing-diffusers",
        help="Self-Forcing transformer repo (the autoregressive model)",
    )
    parser.add_argument(
        "--wan_base_model_id",
        type=str,
        default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        help="Wan2.1 base repo for VAE, text encoder, and scheduler",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="autoregressive_output.mp4")
    parser.add_argument("--fps", type=int, default=16)
    args = parser.parse_args()

    frames = generate_autoregressive_video(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_chunks=args.num_chunks,
        frames_per_chunk=args.frames_per_chunk,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance_scale,
        window_size=args.window_size,
        model_id=args.model_id,
        wan_base_model_id=args.wan_base_model_id,
        device=args.device,
        seed=args.seed,
    )

    export_to_video(frames, args.output, fps=args.fps)
    print(f"Video saved to {args.output} ({len(frames)} frames @ {args.fps}fps = {len(frames)/args.fps:.1f}s)")


if __name__ == "__main__":
    main()
