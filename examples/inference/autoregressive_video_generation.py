"""Autoregressive video generation with rolling KV cache.

Demonstrates how to generate long videos by autoregressively denoising chunks of
frames while maintaining temporal coherence through a rolling KV cache. Uses the
Self-Forcing transformer (gueraf/Self-Forcing-diffusers) which was trained for
autoregressive chunk-based generation on top of Wan2.1-T2V-1.3B.

The rolling KV cache stores self-attention key/value projections from previously
generated chunks so that each new chunk can attend to past context without
recomputing all previous tokens.

Note: gueraf/Self-Forcing-diffusers is a DMD-distilled model. Use
num_inference_steps=4 (default) for best results — more steps over-denoise.
Guidance scale should be low (1.0–2.0); DMD models don't need strong CFG.

Usage:
    python examples/inference/autoregressive_video_generation.py \
        --prompt "A cat walks on the grass, realistic" \
        --num_chunks 14 \
        --frames_per_chunk 17 \
        --output output.mp4

Requirements:
    pip install diffusers transformers accelerate sentencepiece protobuf
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch
from diffusers import AutoencoderKLWan, WanPipeline, WanTransformer3DModel
from diffusers.hooks import RollingKVCacheConfig, apply_rolling_kv_cache, get_rolling_kv_cache_state
from diffusers.utils import export_to_video


def generate_autoregressive_video(
    prompt: str,
    negative_prompt: str = "",
    num_chunks: int = 14,
    frames_per_chunk: int = 17,
    height: int = 480,
    width: int = 832,
    num_inference_steps: int = 4,
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
            This is the autoregressive model — it was trained for chunk-based
            generation and should be the primary model you configure.
        wan_base_model_id: Wan2.1-T2V base repo, used only to load the VAE,
            text encoder, tokenizer, and scheduler. The transformer is replaced
            by model_id, so this does not determine generation style.

    The process:
    1. Prepare scheduler timesteps and encode the prompt once.
    2. For each chunk, run the full denoising loop manually, passing the correct
       frame_offset so that temporal RoPE positions continue across chunks
       (chunk i uses positions i*ppf .. (i+1)*ppf - 1 instead of 0 .. ppf-1).
    3. After denoising, run a clean forward pass at t=0 (with should_update_cache=True)
       to populate the rolling KV cache with accurate clean K/V projections.
    4. Decode all chunk latents at the end with the VAE.

    The frame_offset fix is critical: without it every chunk computes RoPE with
    positions 0..ppf-1, making the cached K/V from prior chunks positionally
    indistinguishable from the current chunk — causing visible jumps at every
    chunk boundary.
    """
    # Load Self-Forcing transformer; VAE and text encoder from the Wan base model
    transformer = WanTransformer3DModel.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    vae = AutoencoderKLWan.from_pretrained(wan_base_model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(
        wan_base_model_id, vae=vae, transformer=transformer, torch_dtype=torch.bfloat16
    )
    pipe.to(device)

    # Apply rolling KV cache
    apply_rolling_kv_cache(pipe.transformer, RollingKVCacheConfig(
        window_size=window_size,
        cache_cross_attention=True,
    ))
    cache_state = get_rolling_kv_cache_state(pipe.transformer)

    # Encode prompt once; reuse across all chunks and clean forward passes
    with torch.no_grad():
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            do_classifier_free_guidance=guidance_scale > 1.0,
            device=device,
            dtype=torch.bfloat16,
        )

    # Latent shape
    p_t, p_h, p_w = pipe.transformer.config.patch_size
    z_dim = pipe.vae.config.z_dim
    latent_h = height // pipe.vae_scale_factor_spatial
    latent_w = width // pipe.vae_scale_factor_spatial
    latent_t = (frames_per_chunk - 1) // pipe.vae_scale_factor_temporal + 1

    # Frames per chunk in patch space (for frame_offset computation)
    ppf = latent_t // p_t

    # VAE latent denormalization constants (needed to decode chunk latents)
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

        # Absolute temporal frame offset for this chunk's RoPE positions
        frame_offset = chunk_idx * ppf

        # Reset scheduler state for each chunk's independent denoising trajectory
        pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = pipe.scheduler.timesteps

        # Initialize latent noise for this chunk
        latents = torch.randn(
            (1, z_dim, latent_t, latent_h, latent_w),
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        latents = latents * pipe.scheduler.init_noise_sigma

        # Denoising loop — cache reads on, writes off (avoids storing noisy K/V)
        cache_state.should_update_cache = False
        with torch.no_grad():
            for t in timesteps:
                latent_model_input = pipe.scheduler.scale_model_input(latents, t)
                timestep = t.expand(latents.shape[0])

                noise_pred = pipe.transformer(
                    latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    frame_offset=frame_offset,
                    return_dict=False,
                )[0]

                latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # chunk_latents: (1, C, latent_t, latent_h, latent_w) in model-normalised space
        chunk_latents = latents.to(dtype=torch.bfloat16)

        # Populate the cache with clean K/V from these denoised latents at t=0.
        # Using the denoised latents directly (no VAE round-trip) gives accurate K/V.
        cache_state.should_update_cache = True
        with torch.no_grad():
            zero_timestep = torch.zeros(1, device=device, dtype=torch.long)
            pipe.transformer(
                chunk_latents,
                timestep=zero_timestep,
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
    parser.add_argument("--num_chunks", type=int, default=14, help="Number of video chunks to generate")
    parser.add_argument("--frames_per_chunk", type=int, default=17, help="Frames per chunk (must be 4k+1)")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument(
        "--num_inference_steps", type=int, default=4,
        help="Diffusion steps. Self-Forcing-DMD works best at 1-4 steps.",
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=1.0,
        help="CFG scale. DMD models need low values (1.0-2.0).",
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
        num_inference_steps=args.num_inference_steps,
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
