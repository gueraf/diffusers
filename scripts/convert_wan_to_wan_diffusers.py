"""Convert a Self-Forcing Wan checkpoint to diffusers format.

Self-Forcing stores transformer weights under `generator_ema` / `generator` with the
original Wan naming scheme. This script converts those weights into a diffusers
`WanTransformer3DModel` checkpoint.

In addition to the key-rename conversion, the script can optionally:
  1. emit a tensor-equivalence report for the converted safetensors checkpoint, and
  2. generate a validation video with the converted model and report PSNR against a
     supplied reference video.
"""

import argparse
import json
import os
import pathlib
import sys

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


TRANSFORMER_KEYS_RENAME_DICT = {
    "time_embedding.0": "condition_embedder.time_embedder.linear_1",
    "time_embedding.2": "condition_embedder.time_embedder.linear_2",
    "text_embedding.0": "condition_embedder.text_embedder.linear_1",
    "text_embedding.2": "condition_embedder.text_embedder.linear_2",
    "time_projection.1": "condition_embedder.time_proj",
    "head.modulation": "scale_shift_table",
    "head.head": "proj_out",
    "modulation": "scale_shift_table",
    "ffn.0": "ffn.net.0.proj",
    "ffn.2": "ffn.net.2",
    "norm2": "norm__placeholder",
    "norm3": "norm2",
    "norm__placeholder": "norm3",
    "self_attn.q": "attn1.to_q",
    "self_attn.k": "attn1.to_k",
    "self_attn.v": "attn1.to_v",
    "self_attn.o": "attn1.to_out.0",
    "self_attn.norm_q": "attn1.norm_q",
    "self_attn.norm_k": "attn1.norm_k",
    "cross_attn.q": "attn2.to_q",
    "cross_attn.k": "attn2.to_k",
    "cross_attn.v": "attn2.to_v",
    "cross_attn.o": "attn2.to_out.0",
    "cross_attn.norm_q": "attn2.norm_q",
    "cross_attn.norm_k": "attn2.norm_k",
    "attn2.to_k_img": "attn2.add_k_proj",
    "attn2.to_v_img": "attn2.add_v_proj",
    "attn2.norm_k_img": "attn2.norm_added_k",
}

WAN_T2V_1_3B_CONFIG = {
    "_class_name": "WanTransformer3DModel",
    "patch_size": [1, 2, 2],
    "num_attention_heads": 12,
    "attention_head_dim": 128,
    "in_channels": 16,
    "out_channels": 16,
    "text_dim": 4096,
    "freq_dim": 256,
    "ffn_dim": 8960,
    "num_layers": 30,
    "cross_attn_norm": True,
    "qk_norm": "rms_norm_across_heads",
    "eps": 1e-6,
    "image_dim": None,
    "added_kv_proj_dim": None,
    "rope_max_seq_len": 1024,
    "pos_embed_seq_len": None,
}


def rename_key(key):
    for old, new in TRANSFORMER_KEYS_RENAME_DICT.items():
        key = key.replace(old, new)
    return key


def _build_equivalence_report(reference_state_dict, reloaded_state_dict):
    compared_keys = sorted(set(reference_state_dict) & set(reloaded_state_dict))
    if not compared_keys:
        return {
            "num_compared_tensors": 0,
            "num_exact_tensor_matches": 0,
            "max_abs_diff": None,
            "mean_abs_diff": None,
        }

    exact_matches = 0
    max_abs_diff = 0.0
    mean_abs_diffs = []
    sample_diffs = {}

    for key in compared_keys:
        reference = reference_state_dict[key].float()
        reloaded = reloaded_state_dict[key].float()
        diff = (reference - reloaded).abs()
        tensor_max = diff.max().item()
        tensor_mean = diff.mean().item()

        if tensor_max == 0.0:
            exact_matches += 1

        max_abs_diff = max(max_abs_diff, tensor_max)
        mean_abs_diffs.append(tensor_mean)

        if len(sample_diffs) < 8:
            sample_diffs[key] = {"max_abs_diff": tensor_max, "mean_abs_diff": tensor_mean}

    return {
        "num_compared_tensors": len(compared_keys),
        "num_exact_tensor_matches": exact_matches,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": sum(mean_abs_diffs) / len(mean_abs_diffs),
        "sample_tensor_diffs": sample_diffs,
    }


def _compute_video_psnr(reference_frames, candidate_frames):
    reference = np.stack([np.asarray(frame, dtype=np.float32) for frame in reference_frames], axis=0)
    candidate = np.stack([np.asarray(frame, dtype=np.float32) for frame in candidate_frames], axis=0)
    mse = np.mean((reference - candidate) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * np.log10(255.0) - 10.0 * np.log10(mse)


def _load_reference_frames(reference_video_path, reference_frames_dir, height, width):
    from diffusers import VideoProcessor
    from diffusers.utils import load_video

    processor = VideoProcessor()

    if reference_frames_dir is not None:
        frame_paths = sorted(pathlib.Path(reference_frames_dir).glob("*.png"))
        if not frame_paths:
            raise ValueError(f"No `.png` frames found under {reference_frames_dir}.")
        reference_frames = [Image.open(path).convert("RGB") for path in frame_paths]
    else:
        reference_frames = load_video(reference_video_path)

    reference_tensor = processor.preprocess_video(reference_frames, height=height, width=width)
    return processor.postprocess_video(reference_tensor, output_type="pil")[0]


def _validate_reference_video_psnr(
    model_path,
    reference_video_path,
    reference_frames_dir,
    prompt,
    negative_prompt,
    wan_base_model_id,
    num_chunks,
    frames_per_chunk,
    height,
    width,
    seed,
    fps,
    device,
    text_encoder_device,
    vae_device,
    validation_output_path,
):
    from diffusers.utils import export_to_video

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples", "inference"))
    from autoregressive_video_generation import generate_autoregressive_video

    generated_frames = generate_autoregressive_video(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_chunks=num_chunks,
        frames_per_chunk=frames_per_chunk,
        height=height,
        width=width,
        guidance_scale=1.0,
        model_id=model_path,
        wan_base_model_id=wan_base_model_id,
        device=device,
        text_encoder_device=text_encoder_device,
        vae_device=vae_device,
        seed=seed,
    )

    reference_frames = _load_reference_frames(
        reference_video_path=reference_video_path,
        reference_frames_dir=reference_frames_dir,
        height=height,
        width=width,
    )

    frame_count = min(len(reference_frames), len(generated_frames))
    psnr = _compute_video_psnr(reference_frames[:frame_count], generated_frames[:frame_count])

    if validation_output_path is not None:
        export_to_video(generated_frames, validation_output_path, fps=fps)

    return {
        "reference_video_path": reference_video_path,
        "reference_frames_dir": reference_frames_dir,
        "validation_output_path": validation_output_path,
        "num_compared_frames": frame_count,
        "psnr_db": psnr,
    }


def convert_self_forcing_checkpoint(checkpoint_path, output_path, use_ema=True, device="cpu"):
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    source_key = "generator_ema" if use_ema else "generator"
    if source_key not in state_dict:
        raise KeyError(f"Key {source_key!r} not found. Available keys: {list(state_dict.keys())}")

    original_state_dict = state_dict[source_key]
    stripped_state_dict = {}
    for key, value in original_state_dict.items():
        if key.startswith("model."):
            stripped_state_dict[key[len("model.") :]] = value

    converted_state_dict = {}
    for key, value in stripped_state_dict.items():
        new_key = rename_key(key)
        converted_state_dict[new_key] = value

    keys_to_remove = []
    for key in converted_state_dict:
        if "rope" in key and "freqs" in key:
            keys_to_remove.append(key)
        if not WAN_T2V_1_3B_CONFIG.get("cross_attn_norm", True) and ".norm2." in key:
            keys_to_remove.append(key)
    for key in keys_to_remove:
        del converted_state_dict[key]

    validation_report = {
        "checkpoint_path": checkpoint_path,
        "output_path": output_path,
        "source_key": source_key,
        "num_original_tensors": len(original_state_dict),
        "num_converted_tensors": len(converted_state_dict),
    }

    try:
        from accelerate import init_empty_weights

        from diffusers import WanTransformer3DModel

        with init_empty_weights():
            model = WanTransformer3DModel.from_config(WAN_T2V_1_3B_CONFIG)

        model_keys = set(model.state_dict().keys())
        converted_keys = set(converted_state_dict.keys())
        missing = model_keys - converted_keys
        unexpected = converted_keys - model_keys

        validation_report["missing_model_keys"] = sorted(missing)
        validation_report["unexpected_converted_keys"] = sorted(unexpected)

        if missing:
            print(f"Missing keys ({len(missing)}):")
            for key in sorted(missing)[:20]:
                print(f"  {key}")

        if unexpected:
            print(f"Unexpected keys ({len(unexpected)}):")
            for key in sorted(unexpected)[:20]:
                print(f"  {key}")
            for key in unexpected:
                del converted_state_dict[key]

        if not missing and not unexpected:
            print("All converted keys match the diffusers Wan transformer config.")
    except ImportError:
        print("Skipping validation because accelerate and/or diffusers is not installed in this environment.")

    output_dir = pathlib.Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    safetensors_path = output_dir / "diffusion_pytorch_model.safetensors"
    save_file(converted_state_dict, str(safetensors_path))
    with open(output_dir / "config.json", "w") as handle:
        json.dump(WAN_T2V_1_3B_CONFIG, handle, indent=2)

    reloaded_state_dict = load_file(str(safetensors_path))
    validation_report["tensor_equivalence"] = _build_equivalence_report(converted_state_dict, reloaded_state_dict)

    validation_report_path = output_dir / "validation_report.json"
    with open(validation_report_path, "w") as handle:
        json.dump(validation_report, handle, indent=2)

    return output_dir, validation_report_path


def main():
    parser = argparse.ArgumentParser(description="Convert a Self-Forcing Wan checkpoint to diffusers format")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="self_forcing_diffusers")
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--reference_video", type=str, default=None)
    parser.add_argument("--reference_frames_dir", type=str, default=None)
    parser.add_argument("--validation_prompt", type=str, default="A cat walks on the grass, realistic")
    parser.add_argument(
        "--validation_negative_prompt",
        type=str,
        default="Bright tones, overexposed, static, blurred details, subtitles, worst quality, low quality",
    )
    parser.add_argument("--wan_base_model_id", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--validation_num_chunks", type=int, default=4)
    parser.add_argument("--validation_frames_per_chunk", type=int, default=9)
    parser.add_argument("--validation_height", type=int, default=480)
    parser.add_argument("--validation_width", type=int, default=832)
    parser.add_argument("--validation_seed", type=int, default=0)
    parser.add_argument("--validation_fps", type=int, default=16)
    parser.add_argument("--validation_device", type=str, default="cuda")
    parser.add_argument("--validation_text_encoder_device", type=str, default=None)
    parser.add_argument("--validation_vae_device", type=str, default=None)
    args = parser.parse_args()

    output_dir, validation_report_path = convert_self_forcing_checkpoint(
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        use_ema=not args.no_ema,
        device=args.device,
    )

    if args.reference_video is not None or args.reference_frames_dir is not None:
        psnr_report = _validate_reference_video_psnr(
            model_path=str(output_dir),
            reference_video_path=args.reference_video,
            reference_frames_dir=args.reference_frames_dir,
            prompt=args.validation_prompt,
            negative_prompt=args.validation_negative_prompt,
            wan_base_model_id=args.wan_base_model_id,
            num_chunks=args.validation_num_chunks,
            frames_per_chunk=args.validation_frames_per_chunk,
            height=args.validation_height,
            width=args.validation_width,
            seed=args.validation_seed,
            fps=args.validation_fps,
            device=args.validation_device,
            text_encoder_device=args.validation_text_encoder_device,
            vae_device=args.validation_vae_device,
            validation_output_path=str(output_dir / "validation_video.mp4"),
        )

        with open(validation_report_path) as handle:
            validation_report = json.load(handle)
        validation_report["reference_video_validation"] = psnr_report
        with open(validation_report_path, "w") as handle:
            json.dump(validation_report, handle, indent=2)

        print(f"Reference-video PSNR: {psnr_report['psnr_db']:.2f}dB")
        print(f"Saved validation video to {psnr_report['validation_output_path']}")

    print(f"Saved validation report to {validation_report_path}")


if __name__ == "__main__":
    main()
