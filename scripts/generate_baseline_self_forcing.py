"""Generate a small baseline set of Self-Forcing autoregressive videos."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples", "inference"))

from autoregressive_video_generation import generate_autoregressive_video
from diffusers.utils import export_to_video


PROMPTS = [
    "A majestic eagle soars over snow-capped mountains, cinematic, 4K, photorealistic",
    "A bustling Tokyo street at night, neon lights reflecting on wet pavement, people walking with umbrellas, cinematic",
]


def main():
    parser = argparse.ArgumentParser(description="Generate baseline Self-Forcing videos with the autoregressive demo")
    parser.add_argument("--output_dir", default="baseline_self_forcing_outputs")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_chunks", type=int, default=27)
    parser.add_argument("--frames_per_chunk", type=int, default=9)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--window_size", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--model_id", default="gueraf/Self-Forcing-diffusers")
    parser.add_argument("--wan_base_model_id", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--text_encoder_device", default=None)
    parser.add_argument("--vae_device", default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for prompt_idx, prompt in enumerate(PROMPTS):
        frames = generate_autoregressive_video(
            prompt=prompt,
            num_chunks=args.num_chunks,
            frames_per_chunk=args.frames_per_chunk,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            window_size=args.window_size,
            model_id=args.model_id,
            wan_base_model_id=args.wan_base_model_id,
            device=args.device,
            text_encoder_device=args.text_encoder_device,
            vae_device=args.vae_device,
            seed=args.seed,
        )
        output_path = os.path.join(args.output_dir, f"baseline_{prompt_idx:02d}.mp4")
        export_to_video(frames, output_path, fps=args.fps)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
