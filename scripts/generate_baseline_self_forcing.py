"""Generate baseline videos using the diffusers Self-Forcing implementation.

Generates reference videos with the gueraf/Self-Forcing-diffusers transformer
(diffusers port of guandeh17/Self-Forcing) + the rolling KV cache. These
serve as visual baselines to verify video quality and temporal coherence.

Uses the correct Self-Forcing DMD inference procedure:
  - FlowMatchEulerDiscreteScheduler with shift=5.0 (not UniPC)
  - 4-step stochastic re-noise schedule matching training: ~[1000, 938, 834, 629]
  - 3 latent frames per chunk (9 pixel frames, matching num_frame_per_block=3)

Usage:
    python scripts/generate_baseline_self_forcing.py \\
        --output_dir /home/fabian/video_outputs/baseline
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, WanPipeline, WanTransformer3DModel
from diffusers.hooks import RollingKVCacheConfig, apply_rolling_kv_cache, get_rolling_kv_cache_state
from diffusers.utils import export_to_video

PROMPTS = [
    "A majestic eagle soars over snow-capped mountains, cinematic, 4K, photorealistic",
    "A bustling Tokyo street at night, neon lights reflecting on wet pavement, people walking with umbrellas, cinematic",
]

MODEL_ID = "gueraf/Self-Forcing-diffusers"
WAN_BASE_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


def build_sf_denoising_steps(device):
    """4-step warped timestep schedule from the Self-Forcing DMD config.

    denoising_step_list=[1000,750,500,250], warp_denoising_step=True,
    timestep_shift=5.0  →  approximately [1000, 938, 834, 629].
    """
    sched = FlowMatchEulerDiscreteScheduler(shift=5.0, num_train_timesteps=1000)
    sched.set_timesteps(1000, device="cpu")
    all_ts = torch.cat([sched.timesteps.cpu(), torch.tensor([0.0])])
    orig = torch.tensor([1000, 750, 500, 250])
    return all_ts[1000 - orig].to(device)


def generate_video(pipe, cache_state, denoising_steps, prompt, num_chunks,
                   frames_per_chunk, height, width, guidance_scale, device, seed):
    do_cfg = guidance_scale > 1.0
    with torch.no_grad():
        prompt_embeds, neg_embeds = pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=None,
            do_classifier_free_guidance=do_cfg,
            device=device,
            dtype=torch.bfloat16,
        )

    p_t, p_h, p_w = pipe.transformer.config.patch_size
    z_dim = pipe.vae.config.z_dim
    latent_h = height // pipe.vae_scale_factor_spatial
    latent_w = width // pipe.vae_scale_factor_spatial
    latent_t = (frames_per_chunk - 1) // pipe.vae_scale_factor_temporal + 1
    ppf = latent_t // p_t

    latents_mean = (
        torch.tensor(pipe.vae.config.latents_mean)
        .view(1, z_dim, 1, 1, 1).to(device, dtype=torch.float32)
    )
    latents_std_inv = (
        (1.0 / torch.tensor(pipe.vae.config.latents_std))
        .view(1, z_dim, 1, 1, 1).to(device, dtype=torch.float32)
    )

    generator = torch.Generator(device=device).manual_seed(seed)

    # Reset rolling cache
    if hasattr(pipe.transformer, "_diffusers_hook"):
        pipe.transformer._diffusers_hook.reset_stateful_hooks()

    all_frames = []
    for chunk_idx in range(num_chunks):
        frame_offset = chunk_idx * ppf

        # Pure noise at sigma=1.0
        noisy_input = torch.randn(
            (1, z_dim, latent_t, latent_h, latent_w),
            generator=generator, device=device, dtype=torch.bfloat16,
        )

        cache_state.should_update_cache = False
        with torch.no_grad():
            for step_idx, t in enumerate(denoising_steps):
                sigma = t / 1000.0

                velocity = pipe.transformer(
                    noisy_input,
                    timestep=t.long().expand(1),
                    encoder_hidden_states=prompt_embeds,
                    frame_offset=frame_offset,
                    return_dict=False,
                )[0]

                # x0 = xt - sigma * velocity
                x0_pred = (noisy_input.float() - sigma * velocity.float()).to(torch.bfloat16)

                if step_idx < len(denoising_steps) - 1:
                    next_sigma = denoising_steps[step_idx + 1] / 1000.0
                    eps = torch.randn_like(x0_pred)
                    noisy_input = ((1.0 - next_sigma) * x0_pred + next_sigma * eps).to(torch.bfloat16)

        chunk_latents = x0_pred

        # Write clean K/V
        cache_state.should_update_cache = True
        with torch.no_grad():
            pipe.transformer(
                chunk_latents,
                timestep=torch.zeros(1, device=device, dtype=torch.long),
                encoder_hidden_states=prompt_embeds,
                frame_offset=frame_offset,
                return_dict=False,
            )

        with torch.no_grad():
            decode_latents = chunk_latents.to(dtype=pipe.vae.dtype)
            decode_latents = decode_latents / latents_std_inv + latents_mean
            chunk_video = pipe.vae.decode(decode_latents).sample
            chunk_frames = pipe.video_processor.postprocess_video(chunk_video, output_type="pil")[0]

        all_frames.extend(chunk_frames)

    return all_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="/home/fabian/video_outputs/baseline")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_chunks", type=int, default=27,
                        help="27 × 9px frames = 243px ≈ 15s @ 16fps")
    parser.add_argument("--frames_per_chunk", type=int, default=9,
                        help="9 pixel frames = 3 latent frames (training block size)")
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--wan_base_model_id", default=WAN_BASE_ID)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device

    print(f"Loading transformer from {args.model_id}...")
    transformer = WanTransformer3DModel.from_pretrained(args.model_id, torch_dtype=torch.bfloat16)
    vae = AutoencoderKLWan.from_pretrained(args.wan_base_model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(
        args.wan_base_model_id, vae=vae, transformer=transformer, torch_dtype=torch.bfloat16
    )
    pipe.to(device)
    pipe.scheduler = FlowMatchEulerDiscreteScheduler(shift=5.0, num_train_timesteps=1000)

    apply_rolling_kv_cache(pipe.transformer, RollingKVCacheConfig(
        window_size=8000, cache_cross_attention=True,
    ))
    cache_state = get_rolling_kv_cache_state(pipe.transformer)

    denoising_steps = build_sf_denoising_steps(device=device)
    print(f"Denoising steps: {denoising_steps.tolist()}")

    total_px = args.num_chunks * args.frames_per_chunk
    print(f"Generating {args.num_chunks} chunks × {args.frames_per_chunk}px frames = {total_px}px ({total_px/args.fps:.1f}s @ {args.fps}fps)")

    for prompt_idx, prompt in enumerate(PROMPTS):
        print(f"\n[{prompt_idx+1}/{len(PROMPTS)}] {prompt[:70]}...")
        frames = generate_video(
            pipe, cache_state, denoising_steps, prompt,
            num_chunks=args.num_chunks,
            frames_per_chunk=args.frames_per_chunk,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            device=device,
            seed=args.seed,
        )
        out_path = os.path.join(args.output_dir, f"baseline_{prompt_idx:02d}.mp4")
        export_to_video(frames, out_path, fps=args.fps)
        print(f"Saved {out_path} ({len(frames)} frames @ {args.fps}fps = {len(frames)/args.fps:.1f}s)")


if __name__ == "__main__":
    main()
