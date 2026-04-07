"""Generate baseline videos using the original Self-Forcing code + gdhe17/Self-Forcing weights.

This script runs the original guandeh17/Self-Forcing inference pipeline
(vendored as tyrannis) with the publicly-released self_forcing_dmd.pt checkpoint
to produce reference videos for equivalence comparison.

Usage:
    python scripts/generate_baseline_self_forcing.py \
        --output_dir /home/fabian/video_outputs/baseline \
        --num_frames 21

Environment:
    TYRANNIS_ROOT  path to the vendored tyrannis repo (default below)
    SF_CKPT        path to self_forcing_dmd.pt (default: HF cache)
"""

import argparse
import os
import sys
import torch
import numpy as np
from torchvision.io import write_video

TYRANNIS_ROOT = os.environ.get(
    "TYRANNIS_ROOT",
    "/home/fabian/odyssey/ml_core/inference/predictor/vendor/tyrannis",
)
SF_CKPT = os.environ.get(
    "SF_CKPT",
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--gdhe17--Self-Forcing/"
        "snapshots/2f8b779212da279d212c22a509b66ad6552f350e/"
        "checkpoints/self_forcing_dmd.pt"
    ),
)

sys.path.insert(0, TYRANNIS_ROOT)

from omegaconf import OmegaConf
from pipeline import CausalInferencePipeline

PROMPTS = [
    "A majestic eagle soars over snow-capped mountains, cinematic, 4K, photorealistic",
    "A bustling Tokyo street at night, neon lights reflecting on wet pavement, people walking with umbrellas, cinematic",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="/home/fabian/video_outputs/baseline")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=21,
                        help="Total latent frames (21 = 81 pixel frames at VAE scale 4)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda")

    # Load config
    cfg_path = os.path.join(TYRANNIS_ROOT, "configs", "self_forcing_dmd.yaml")
    default_cfg_path = os.path.join(TYRANNIS_ROOT, "configs", "default_config.yaml")
    config = OmegaConf.merge(OmegaConf.load(default_cfg_path), OmegaConf.load(cfg_path))
    config.height = args.height
    config.width = args.width
    config.num_frames = args.num_frames

    print(f"Initializing pipeline (num_frame_per_block={config.num_frame_per_block})...")
    pipeline = CausalInferencePipeline(config, device=device)

    # Load Self-Forcing DMD checkpoint
    print(f"Loading checkpoint from {SF_CKPT}...")
    ckpt = torch.load(SF_CKPT, map_location="cpu", weights_only=True)
    raw_sd = ckpt["generator_ema"]
    # Strip wrapper prefixes
    clean_sd = {}
    for k, v in raw_sd.items():
        nk = k
        for prefix in ("module.", "_fsdp_wrapped_module.", "_checkpoint_wrapped_module.", "_orig_mod."):
            nk = nk.replace(prefix, "")
        if nk.startswith("model."):
            nk = nk[len("model."):]
        clean_sd[nk] = v

    target_keys = set(pipeline.generator.model.state_dict().keys())
    clean_sd = {k: v for k, v in clean_sd.items() if k in target_keys}
    missing, unexpected = pipeline.generator.model.load_state_dict(clean_sd, strict=False)
    print(f"Checkpoint loaded: missing={len(missing)}, unexpected={len(unexpected)}")

    pipeline = pipeline.to(dtype=torch.bfloat16)
    pipeline.text_encoder.to(device)
    pipeline.generator.to(device)
    pipeline.vae.to(device)
    pipeline.generator.eval()

    latent_h = args.height // 8
    latent_w = args.width // 8

    for prompt_idx, prompt in enumerate(PROMPTS):
        print(f"\nGenerating: {prompt[:60]}...")
        torch.manual_seed(args.seed)

        noise = torch.randn(
            [1, args.num_frames, 16, latent_h, latent_w],
            device=device, dtype=torch.bfloat16,
        )

        with torch.no_grad():
            video = pipeline.inference(
                noise=noise,
                text_prompts=[prompt],
            )

        # video: (B, T, C, H, W) in [0, 1]
        video = video[0]  # (T, C, H, W)
        video_uint8 = (video.cpu().float() * 255).clamp(0, 255).to(torch.uint8)
        video_uint8 = video_uint8.permute(0, 2, 3, 1)  # (T, H, W, C)

        name = f"baseline_{prompt_idx:02d}.mp4"
        out_path = os.path.join(args.output_dir, name)
        write_video(out_path, video_uint8, fps=args.fps)
        print(f"Saved {out_path} ({video_uint8.shape[0]} frames @ {args.fps}fps = {video_uint8.shape[0]/args.fps:.1f}s)")


if __name__ == "__main__":
    main()
