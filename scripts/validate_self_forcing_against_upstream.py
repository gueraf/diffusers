"""Validate the diffusers Self-Forcing port against the original upstream repo.

This script runs the original `guandeh17/Self-Forcing` causal inference path and the
diffusers rolling-KV-cache path from the same prompt, checkpoint, initial latent noise,
and re-noising RNG seed. It saves both videos plus a JSON report with latent-space diffs
and decoded-frame PSNR.
"""

import argparse
import gc
import importlib
import json
import os
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from transformers import UMT5EncoderModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from diffusers import WanTransformer3DModel
from diffusers.hooks import RollingKVCacheConfig, get_rolling_kv_cache_state, prefill_rolling_kv_cache
from diffusers.utils import export_to_video


def _resolve_device(device):
    return torch.device(device)


def _manual_seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_sf_denoising_steps(device):
    # Match guandeh17/Self-Forcing's custom FlowMatchScheduler exactly.
    sigmas = torch.linspace(1.0, 0.0, 1001)[:-1]
    sigmas = 5.0 * sigmas / (1.0 + 4.0 * sigmas)
    all_timesteps = torch.cat([sigmas * 1000.0, torch.tensor([0.0])])
    original_schedule = torch.tensor([1000, 750, 500, 250])
    return all_timesteps[1000 - original_schedule].to(device=device, dtype=torch.float32)


def _compute_psnr(reference_frames, candidate_frames):
    reference = np.stack([np.asarray(frame, dtype=np.float32) for frame in reference_frames], axis=0)
    candidate = np.stack([np.asarray(frame, dtype=np.float32) for frame in candidate_frames], axis=0)
    mse = np.mean((reference - candidate) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * np.log10(255.0) - 10.0 * np.log10(mse)


def _compute_framewise_psnr(reference_frames, candidate_frames):
    psnrs = []
    for reference_frame, candidate_frame in zip(reference_frames, candidate_frames):
        reference = np.asarray(reference_frame, dtype=np.float32)
        candidate = np.asarray(candidate_frame, dtype=np.float32)
        mse = np.mean((reference - candidate) ** 2)
        if mse == 0:
            psnrs.append(float("inf"))
        else:
            psnrs.append(20.0 * np.log10(255.0) - 10.0 * np.log10(mse))
    return psnrs


def _frame_to_token_offset(transformer, latents, frame_offset):
    _, _, _, height, width = latents.shape
    _, p_h, p_w = transformer.config.patch_size
    patches_per_frame = (height // p_h) * (width // p_w)
    return frame_offset * patches_per_frame


def _video_tensor_to_pil(video):
    frames = []
    for frame in video[0]:
        array = frame.detach().float().clamp(0, 1).mul(255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        frames.append(Image.fromarray(array))
    return frames


def _save_frames(frames, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        frame.save(output_dir / f"{index:04d}.png")


def _latent_report(reference, candidate):
    diff = (reference.float() - candidate.float()).abs()
    return {
        "shape": list(reference.shape),
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
        "allclose_atol_1e-5": torch.allclose(reference.float(), candidate.float(), atol=1e-5, rtol=1e-5),
        "allclose_atol_1e-4": torch.allclose(reference.float(), candidate.float(), atol=1e-4, rtol=1e-4),
    }


def _sample_self_forcing_renoise(latents):
    batch_size, channels, num_frames, height, width = latents.shape
    noise = torch.randn(
        (batch_size, num_frames, channels, height, width),
        device=latents.device,
        dtype=latents.dtype,
    )
    return noise.permute(0, 2, 1, 3, 4).contiguous()


def _patch_upstream_flash_attention():
    attention_mod = importlib.import_module("wan.modules.attention")
    model_mod = importlib.import_module("wan.modules.model")
    clip_mod = importlib.import_module("wan.modules.clip")
    attention_mod.flash_attention = attention_mod.attention
    model_mod.flash_attention = attention_mod.attention
    clip_mod.flash_attention = attention_mod.attention


def _load_upstream_modules(upstream_repo_path):
    if upstream_repo_path not in sys.path:
        sys.path.insert(0, upstream_repo_path)

    _patch_upstream_flash_attention()

    causal_inference_mod = importlib.import_module("pipeline.causal_inference")
    scheduler_mod = importlib.import_module("utils.scheduler")
    wan_wrapper_mod = importlib.import_module("utils.wan_wrapper")
    causal_model_mod = importlib.import_module("wan.modules.causal_model")
    tokenizer_mod = importlib.import_module("wan.modules.tokenizers")
    t5_mod = importlib.import_module("wan.modules.t5")
    vae_mod = importlib.import_module("wan.modules.vae")

    return {
        "CausalInferencePipeline": causal_inference_mod.CausalInferencePipeline,
        "FlowMatchScheduler": scheduler_mod.FlowMatchScheduler,
        "WanDiffusionWrapper": wan_wrapper_mod.WanDiffusionWrapper,
        "CausalWanModel": causal_model_mod.CausalWanModel,
        "HuggingfaceTokenizer": tokenizer_mod.HuggingfaceTokenizer,
        "umt5_xxl": t5_mod.umt5_xxl,
        "_video_vae": vae_mod._video_vae,
    }


class HFTextEncoder(torch.nn.Module):
    def __init__(self, tokenizer_cls, tokenizer_path, text_encoder_path, device, output_device):
        super().__init__()
        self.tokenizer = tokenizer_cls(name=tokenizer_path, seq_len=512, clean="whitespace")
        encoder_dtype = torch.float32 if str(device) == "cpu" else torch.bfloat16
        self.text_encoder = UMT5EncoderModel.from_pretrained(text_encoder_path, torch_dtype=encoder_dtype)
        self.text_encoder.eval().requires_grad_(False)
        self.text_encoder.to(device)
        self.device = _resolve_device(device)
        self.output_device = _resolve_device(output_device)

    @torch.no_grad()
    def forward(self, text_prompts):
        ids, mask = self.tokenizer(text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(input_ids=ids, attention_mask=mask).last_hidden_state

        for tensor, seq_len in zip(context, seq_lens):
            tensor[seq_len:] = 0.0

        return {
            "prompt_embeds": context.to(device=self.output_device)
        }


class FixedTextEncoder(torch.nn.Module):
    def __init__(self, conditional_dict):
        super().__init__()
        self.conditional_dict = conditional_dict

    def forward(self, text_prompts):
        return {
            "prompt_embeds": self.conditional_dict["prompt_embeds"].clone()
        }


class UpstreamTextEncoder(torch.nn.Module):
    def __init__(self, tokenizer_cls, text_encoder_factory, upstream_repo_path, device, output_device):
        super().__init__()
        model_dir = pathlib.Path(upstream_repo_path) / "wan_models" / "Wan2.1-T2V-1.3B"
        tokenizer_path = model_dir / "google" / "umt5-xxl"
        weights_path = model_dir / "models_t5_umt5-xxl-enc-bf16.pth"
        encoder_dtype = torch.float32 if str(device) == "cpu" else torch.bfloat16

        self.tokenizer = tokenizer_cls(name=str(tokenizer_path), seq_len=512, clean="whitespace")
        self.text_encoder = text_encoder_factory(
            encoder_only=True,
            return_tokenizer=False,
            dtype=encoder_dtype,
            device=torch.device("cpu"),
        ).eval().requires_grad_(False)
        self.text_encoder.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=False))
        self.text_encoder.to(device=device, dtype=encoder_dtype)
        self.device = _resolve_device(device)
        self.output_device = _resolve_device(output_device)

    @torch.no_grad()
    def forward(self, text_prompts):
        ids, mask = self.tokenizer(text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)

        for tensor, seq_len in zip(context, seq_lens):
            tensor[seq_len:] = 0.0

        return {
            "prompt_embeds": context.to(device=self.output_device)
        }


def _load_prompt_embeds(path, key="prompt_embeds"):
    payload = torch.load(path, map_location="cpu")
    if torch.is_tensor(payload):
        prompt_embeds = payload
    elif isinstance(payload, dict):
        if key not in payload:
            raise KeyError(f"`{key}` not found in prompt embeds payload `{path}`.")
        prompt_embeds = payload[key]
    else:
        raise TypeError(f"Unsupported prompt embeds payload type: {type(payload)!r}")

    if not torch.is_tensor(prompt_embeds):
        raise TypeError(f"`{key}` from `{path}` must be a tensor, got {type(prompt_embeds)!r}")

    return prompt_embeds


def _assert_valid_self_forcing_transformer(transformer):
    has_cross_attn_norm = bool(getattr(transformer.config, "cross_attn_norm", False))
    has_norm_module = transformer.blocks and not isinstance(transformer.blocks[0].norm2, torch.nn.Identity)

    if has_cross_attn_norm and has_norm_module:
        return

    raise ValueError(
        "The loaded diffusers transformer is missing Self-Forcing cross-attention norms. "
        "Re-convert the checkpoint with `scripts/convert_wan_to_wan_diffusers.py` and use that output."
    )


def _align_self_forcing_transformer_dtype(transformer):
    runtime_device = transformer.patch_embedding.weight.device
    runtime_dtype = transformer.patch_embedding.weight.dtype

    transformer.condition_embedder.time_embedder.to(device=runtime_device, dtype=runtime_dtype)
    transformer.scale_shift_table.data = transformer.scale_shift_table.data.to(device=runtime_device, dtype=runtime_dtype)

    for block in transformer.blocks:
        block.scale_shift_table.data = block.scale_shift_table.data.to(device=runtime_device, dtype=runtime_dtype)

        if hasattr(block.norm2, "weight") and block.norm2.weight is not None:
            block.norm2.weight.data = block.norm2.weight.data.to(device=runtime_device, dtype=runtime_dtype)
        if hasattr(block.norm2, "bias") and block.norm2.bias is not None:
            block.norm2.bias.data = block.norm2.bias.data.to(device=runtime_device, dtype=runtime_dtype)


def _make_absolute_vae_wrapper(video_vae_factory, vae_path):
    class AbsoluteWanVAEWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            mean = [
                -0.7571,
                -0.7089,
                -0.9113,
                0.1075,
                -0.1745,
                0.9653,
                -0.1517,
                1.5508,
                0.4134,
                -0.0715,
                0.5517,
                -0.3632,
                -0.1922,
                -0.9497,
                0.2503,
                -0.2921,
            ]
            std = [
                2.8184,
                1.4541,
                2.3275,
                2.6558,
                1.2196,
                1.7708,
                2.6052,
                2.0743,
                3.2687,
                2.1526,
                2.8652,
                1.5579,
                1.6382,
                1.1253,
                2.8251,
                1.9160,
            ]
            self.mean = torch.tensor(mean, dtype=torch.float32)
            self.std = torch.tensor(std, dtype=torch.float32)
            self.model = video_vae_factory(pretrained_path=vae_path, z_dim=16).eval().requires_grad_(False)

        def decode_to_pixel(self, latent, use_cache=False):
            zs = latent.permute(0, 2, 1, 3, 4)
            if use_cache:
                assert latent.shape[0] == 1

            device, dtype = latent.device, latent.dtype
            scale = [self.mean.to(device=device, dtype=dtype), 1.0 / self.std.to(device=device, dtype=dtype)]
            decode_function = self.model.cached_decode if use_cache else self.model.decode

            output = []
            for tensor in zs:
                output.append(decode_function(tensor.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0))
            output = torch.stack(output, dim=0)
            return output.permute(0, 2, 1, 3, 4)

    return AbsoluteWanVAEWrapper


def _make_config_init_wrapper(base_wrapper_cls, causal_model_cls, scheduler_cls, model_config_path):
    class ConfigInitWanDiffusionWrapper(base_wrapper_cls):
        def __init__(
            self,
            timestep_shift=8.0,
            is_causal=True,
            local_attn_size=-1,
            sink_size=0,
        ):
            torch.nn.Module.__init__(self)
            with open(model_config_path) as handle:
                model_config = json.load(handle)

            if is_causal:
                model_config = dict(model_config)
                model_config["local_attn_size"] = local_attn_size
                model_config["sink_size"] = sink_size
                self.model = causal_model_cls.from_config(model_config)
            else:
                raise ValueError("Only causal Self-Forcing validation is supported.")

            self.model.eval()
            self.uniform_timestep = not is_causal
            self.scheduler = scheduler_cls(shift=timestep_shift, sigma_min=0.0, extra_one_step=True)
            self.scheduler.set_timesteps(1000, training=True)
            self.seq_len = 32760
            self.post_init()

    return ConfigInitWanDiffusionWrapper


def _generate_diffusers_latents(transformer, full_noise, prompt_embeds, denoising_steps, latent_frames_per_chunk):
    outputs = []
    with transformer.cache_context("cond"):
        state = get_rolling_kv_cache_state(transformer)
        if state is None:
            raise ValueError("Rolling KV cache must be enabled before generation.")
        for chunk_start in range(0, full_noise.shape[1], latent_frames_per_chunk):
            frame_offset = chunk_start
            noisy_input = full_noise[:, chunk_start : chunk_start + latent_frames_per_chunk].permute(0, 2, 1, 3, 4)
            noisy_input = noisy_input.contiguous()
            token_offset = _frame_to_token_offset(transformer, noisy_input, frame_offset)

            with torch.no_grad():
                for step_index, timestep in enumerate(denoising_steps):
                    prev_should_update = state.should_update_cache
                    prev_write_mode = state.write_mode
                    prev_absolute_token_offset = state.absolute_token_offset
                    try:
                        state.should_update_cache = True
                        state.configure_cache_write(write_mode="overwrite", absolute_token_offset=token_offset)
                        velocity = transformer(
                            hidden_states=noisy_input,
                            timestep=timestep.expand(noisy_input.shape[0], noisy_input.shape[2]),
                            encoder_hidden_states=prompt_embeds,
                            frame_offset=frame_offset,
                            return_dict=False,
                        )[0]
                    finally:
                        state.should_update_cache = prev_should_update
                        state.configure_cache_write(
                            write_mode=prev_write_mode,
                            absolute_token_offset=prev_absolute_token_offset,
                        )

                    sigma = (timestep / 1000.0).to(device=noisy_input.device, dtype=torch.float32)
                    x0_pred = (noisy_input.float() - sigma * velocity.float()).to(dtype=noisy_input.dtype)

                    if step_index < len(denoising_steps) - 1:
                        next_sigma = (denoising_steps[step_index + 1] / 1000.0).to(
                            device=noisy_input.device, dtype=torch.float32
                        )
                        eps = _sample_self_forcing_renoise(x0_pred)
                        noisy_input = ((1.0 - next_sigma) * x0_pred + next_sigma * eps).to(
                            dtype=noisy_input.dtype
                        )

            prefill_rolling_kv_cache(
                transformer,
                x0_pred,
                prompt_embeds,
                frame_offset=frame_offset,
                cache_context="cond",
                write_mode="overwrite",
            )
            outputs.append(x0_pred.permute(0, 2, 1, 3, 4).contiguous())

    transformer._reset_stateful_cache()
    return torch.cat(outputs, dim=1)


def main():
    parser = argparse.ArgumentParser(description="Validate diffusers Self-Forcing against the original upstream repo")
    parser.add_argument("--upstream_repo_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--diffusers_model_path", type=str, required=True)
    parser.add_argument("--text_encoder_source", type=str, choices=("upstream", "hf"), default="upstream")
    parser.add_argument("--text_encoder_path", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--prompt_embeds_path", type=str, default=None)
    parser.add_argument("--prompt_embeds_key", type=str, default="prompt_embeds")
    parser.add_argument("--wan_model_config", type=str, required=True)
    parser.add_argument("--vae_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--text_encoder_device", type=str, default="cuda:1")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num_chunks", type=int, default=3)
    parser.add_argument("--frames_per_chunk", type=int, default=9)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    upstream_modules = _load_upstream_modules(args.upstream_repo_path)
    if args.prompt_embeds_path is not None:
        prompt_embeds = _load_prompt_embeds(args.prompt_embeds_path, key=args.prompt_embeds_key).to(
            device=_resolve_device(args.device)
        )
        conditional_dict = {"prompt_embeds": prompt_embeds}
    else:
        if args.text_encoder_source == "upstream":
            text_encoder = UpstreamTextEncoder(
                tokenizer_cls=upstream_modules["HuggingfaceTokenizer"],
                text_encoder_factory=upstream_modules["umt5_xxl"],
                upstream_repo_path=args.upstream_repo_path,
                device=args.text_encoder_device,
                output_device=args.device,
            )
        else:
            if args.text_encoder_path is None or args.tokenizer_path is None:
                raise ValueError(
                    "`--text_encoder_path` and `--tokenizer_path` are required when `--text_encoder_source=hf` "
                    "unless `--prompt_embeds_path` is provided."
                )
            text_encoder = HFTextEncoder(
                tokenizer_cls=upstream_modules["HuggingfaceTokenizer"],
                tokenizer_path=args.tokenizer_path,
                text_encoder_path=args.text_encoder_path,
                device=args.text_encoder_device,
                output_device=args.device,
            )
        with torch.no_grad():
            conditional_dict = text_encoder([args.prompt])
        prompt_embeds = conditional_dict["prompt_embeds"]
        del text_encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    AbsoluteWanVAEWrapper = _make_absolute_vae_wrapper(upstream_modules["_video_vae"], args.vae_path)
    ConfigInitWanDiffusionWrapper = _make_config_init_wrapper(
        upstream_modules["WanDiffusionWrapper"],
        upstream_modules["CausalWanModel"],
        upstream_modules["FlowMatchScheduler"],
        args.wan_model_config,
    )

    latent_frames_per_chunk = (args.frames_per_chunk - 1) // 4 + 1
    latent_h = args.height // 8
    latent_w = args.width // 8
    total_latent_frames = args.num_chunks * latent_frames_per_chunk

    noise_generator = torch.Generator(device=_resolve_device(args.device)).manual_seed(args.seed)
    full_noise = torch.randn(
        (1, total_latent_frames, 16, latent_h, latent_w),
        generator=noise_generator,
        device=_resolve_device(args.device),
        dtype=torch.bfloat16,
    )
    renoise_seed = args.seed + 1

    original_args = SimpleNamespace(
        denoising_step_list=[1000, 750, 500, 250],
        warp_denoising_step=True,
        num_frame_per_block=latent_frames_per_chunk,
        independent_first_frame=False,
        context_noise=0,
        model_kwargs={"timestep_shift": 5.0},
    )

    original_pipeline = upstream_modules["CausalInferencePipeline"](
        args=original_args,
        device=_resolve_device(args.device),
        generator=ConfigInitWanDiffusionWrapper(timestep_shift=5.0, is_causal=True),
        text_encoder=FixedTextEncoder(conditional_dict),
        vae=AbsoluteWanVAEWrapper(),
    )
    original_pipeline.generator.load_state_dict(
        torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)["generator_ema"]
    )
    original_pipeline.generator.to(device=args.device, dtype=torch.bfloat16)
    original_pipeline.vae.to(device=args.device, dtype=torch.bfloat16)

    _manual_seed_all(renoise_seed)
    with torch.no_grad():
        original_video, original_latents = original_pipeline.inference(
            noise=full_noise.clone(),
            text_prompts=[args.prompt],
            return_latents=True,
            low_memory=False,
        )

    original_frames = _video_tensor_to_pil(original_video)
    export_to_video(original_frames, output_dir / "original.mp4", fps=16)
    _save_frames(original_frames, output_dir / "original_frames")

    vae_wrapper = original_pipeline.vae
    vae_wrapper.to("cpu")
    original_latents = original_latents.cpu()
    del original_video
    del original_pipeline.generator
    del original_pipeline.text_encoder
    del original_pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    transformer = WanTransformer3DModel.from_pretrained(args.diffusers_model_path, torch_dtype=torch.bfloat16)
    transformer.to(args.device)
    transformer.eval()
    _assert_valid_self_forcing_transformer(transformer)
    _align_self_forcing_transformer_dtype(transformer)
    transformer.enable_cache(RollingKVCacheConfig(window_size=-1, cache_cross_attention=True))

    _manual_seed_all(renoise_seed)
    denoising_steps = _build_sf_denoising_steps(device=_resolve_device(args.device))
    diffusers_latents = _generate_diffusers_latents(
        transformer=transformer,
        full_noise=full_noise.clone(),
        prompt_embeds=prompt_embeds,
        denoising_steps=denoising_steps,
        latent_frames_per_chunk=latent_frames_per_chunk,
    )
    del transformer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    vae_wrapper.to(args.device, dtype=torch.bfloat16)
    diffusers_video = vae_wrapper.decode_to_pixel(diffusers_latents, use_cache=False)
    diffusers_video = (diffusers_video * 0.5 + 0.5).clamp(0, 1)

    diffusers_frames = _video_tensor_to_pil(diffusers_video)
    export_to_video(diffusers_frames, output_dir / "diffusers.mp4", fps=16)
    _save_frames(diffusers_frames, output_dir / "diffusers_frames")

    framewise_psnr = _compute_framewise_psnr(original_frames, diffusers_frames)
    report = {
        "prompt": args.prompt,
        "num_frames": len(original_frames),
        "seed": args.seed,
        "renoise_seed": renoise_seed,
        "latent_report": _latent_report(original_latents, diffusers_latents.cpu()),
        "video_psnr_db": _compute_psnr(original_frames, diffusers_frames),
        "framewise_psnr_db": {
            "min": min(framewise_psnr),
            "mean": sum(framewise_psnr) / len(framewise_psnr),
            "max": max(framewise_psnr),
        },
        "paths": {
            "original_video": str(output_dir / "original.mp4"),
            "diffusers_video": str(output_dir / "diffusers.mp4"),
            "original_frames": str(output_dir / "original_frames"),
            "diffusers_frames": str(output_dir / "diffusers_frames"),
        },
    }

    report_path = output_dir / "report.json"
    with open(report_path, "w") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Saved validation report to {report_path}")


if __name__ == "__main__":
    main()
