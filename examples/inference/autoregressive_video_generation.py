"""Autoregressive video generation with rolling KV cache.

Demonstrates how to generate long videos by autoregressively denoising chunks of
frames while maintaining temporal coherence through a rolling KV cache. Uses the
Self-Forcing transformer (gueraf/Self-Forcing-diffusers) which was trained for
autoregressive chunk-based generation on top of Wan2.1-T2V-1.3B.

The rolling KV cache stores self-attention key/value projections from previously
generated chunks so that each new chunk can attend to past context without
recomputing all previous tokens.

Usage:
    python examples/inference/autoregressive_video_generation.py \
        --prompt "A cat walks on the grass, realistic" \
        --num_chunks 3 \
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
    num_chunks: int = 3,
    frames_per_chunk: int = 17,
    overlap_frames: int = 1,
    height: int = 480,
    width: int = 832,
    num_inference_steps: int = 30,
    guidance_scale: float = 5.0,
    window_size: int = -1,
    base_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    transformer_id: str = "gueraf/Self-Forcing-diffusers",
    device: str = "cuda",
    seed: int = 0,
):
    """Generate a long video autoregressively using rolling KV cache.

    The process:
    1. Generate the first chunk of frames normally.
    2. For each subsequent chunk:
       a. Set ``should_update_cache = False`` during denoising (avoids caching noisy K/V).
       b. After denoising, run one clean forward pass with ``should_update_cache = True``
          to record the chunk's clean K/V into the rolling cache.
       c. The next chunk's self-attention attends to all cached context.
    """
    # Load VAE and text encoder from the base Wan model; transformer from Self-Forcing
    vae = AutoencoderKLWan.from_pretrained(base_model_id, subfolder="vae", torch_dtype=torch.float32)
    transformer = WanTransformer3DModel.from_pretrained(transformer_id, torch_dtype=torch.bfloat16)
    pipe = WanPipeline.from_pretrained(base_model_id, vae=vae, transformer=transformer, torch_dtype=torch.bfloat16)
    pipe.to(device)

    # Apply rolling KV cache to the transformer
    config = RollingKVCacheConfig(
        window_size=window_size,
        cache_cross_attention=True,
    )
    apply_rolling_kv_cache(pipe.transformer, config)
    cache_state = get_rolling_kv_cache_state(pipe.transformer)

    # Pre-encode the prompt once so we can reuse it in the clean forward passes
    with torch.no_grad():
        prompt_embeds, _ = pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=None,
            do_classifier_free_guidance=False,
            device=device,
            dtype=torch.bfloat16,
        )

    # Precompute latent normalization constants from VAE config
    latents_mean = (
        torch.tensor(pipe.vae.config.latents_mean)
        .view(1, pipe.vae.config.z_dim, 1, 1, 1)
        .to(device, dtype=torch.float32)
    )
    latents_std = (
        (1.0 / torch.tensor(pipe.vae.config.latents_std))
        .view(1, pipe.vae.config.z_dim, 1, 1, 1)
        .to(device, dtype=torch.float32)
    )

    generator = torch.Generator(device=device).manual_seed(seed)
    all_frames = []

    for chunk_idx in range(num_chunks):
        print(f"Generating chunk {chunk_idx + 1}/{num_chunks}...")

        # During denoising, suppress cache updates to avoid polluting it with noisy K/V
        cache_state.should_update_cache = False

        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=frames_per_chunk,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        chunk_frames = output.frames[0]  # list of PIL images

        # After denoising, run one clean forward pass to populate the cache with
        # the chunk's clean K/V projections, so the next chunk can attend to them.
        cache_state.should_update_cache = True

        with torch.no_grad():
            # 1. Preprocess frames → (1, C, T, H, W) tensor in [-1, 1]
            video_tensor = pipe.video_processor.preprocess_video(chunk_frames, height=height, width=width)
            video_tensor = video_tensor.to(device=device, dtype=torch.float32)

            # 2. Encode to latents and normalize to model latent space
            raw_latents = pipe.vae.encode(video_tensor).latent_dist.sample()
            clean_latents = (raw_latents - latents_mean.to(raw_latents.dtype)) * latents_std.to(raw_latents.dtype)
            clean_latents = clean_latents.to(dtype=torch.bfloat16)

            # 3. Run transformer forward at timestep=0 with clean latents to fill the cache
            timestep = torch.zeros(1, device=device, dtype=torch.long)
            pipe.transformer(
                clean_latents,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
            )

        if chunk_idx == 0:
            all_frames.extend(chunk_frames)
        else:
            all_frames.extend(chunk_frames[overlap_frames:])

        print(f"  Total frames so far: {len(all_frames)}")

    # Reset stateful hooks for clean state
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
    parser.add_argument("--num_chunks", type=int, default=3, help="Number of video chunks to generate")
    parser.add_argument("--frames_per_chunk", type=int, default=17, help="Frames per chunk")
    parser.add_argument("--overlap_frames", type=int, default=1, help="Overlapping frames to drop between chunks")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--window_size", type=int, default=-1, help="Rolling cache window size (-1 for unlimited)")
    parser.add_argument(
        "--base_model_id",
        type=str,
        default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        help="Base model for VAE, text encoder, and scheduler",
    )
    parser.add_argument(
        "--transformer_id",
        type=str,
        default="gueraf/Self-Forcing-diffusers",
        help="Model repo for the Self-Forcing transformer weights",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="autoregressive_output.mp4")
    args = parser.parse_args()

    frames = generate_autoregressive_video(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_chunks=args.num_chunks,
        frames_per_chunk=args.frames_per_chunk,
        overlap_frames=args.overlap_frames,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        window_size=args.window_size,
        base_model_id=args.base_model_id,
        transformer_id=args.transformer_id,
        device=args.device,
        seed=args.seed,
    )

    export_to_video(frames, args.output, fps=15)
    print(f"Video saved to {args.output} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
