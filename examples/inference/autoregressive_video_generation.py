"""Autoregressive Self-Forcing video generation with rolling KV caching.

This example runs chunk-wise Wan/Self-Forcing inference and keeps clean self-attention KV
states across chunks. It also supports injecting ground-truth video chunks at any chunk index:
the reference frames are VAE-encoded, written into the cache with absolute temporal positions,
and generation continues from that point onward.

Usage:
    python examples/inference/autoregressive_video_generation.py \
        --prompt "A cat walks on the grass, realistic" \
        --num_chunks 27 \
        --frames_per_chunk 9 \
        --output output.mp4

    python examples/inference/autoregressive_video_generation.py \
        --prompt "A fox runs through a forest" \
        --conditioning_video path/to/reference.mp4 \
        --conditioning_start_chunk 4 \
        --output conditioned_output.mp4
"""

import argparse
import os
import sys

import numpy as np
import torch
from transformers import AutoTokenizer, UMT5EncoderModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, WanPipeline, WanTransformer3DModel
from diffusers.hooks import RollingKVCacheConfig, get_rolling_kv_cache_state, prefill_rolling_kv_cache
from diffusers.utils import export_to_video, load_video


def _build_sf_denoising_steps(device):
    # Match guandeh17/Self-Forcing's custom FlowMatchScheduler exactly.
    sigmas = torch.linspace(1.0, 0.0, 1001)[:-1]
    sigmas = 5.0 * sigmas / (1.0 + 4.0 * sigmas)
    all_timesteps = torch.cat([sigmas * 1000.0, torch.tensor([0.0])])
    original_schedule = torch.tensor([1000, 750, 500, 250])
    return all_timesteps[1000 - original_schedule].to(device)


def _retrieve_latents(encoder_output, sample_mode="argmax"):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    if hasattr(encoder_output, "latent_dist"):
        return encoder_output.latent_dist.sample()
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not retrieve latents from the VAE encoder output.")


def _get_latent_stats(pipe, device):
    latents_mean = (
        torch.tensor(pipe.vae.config.latents_mean).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(device, torch.float32)
    )
    latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(
        device,
        torch.float32,
    )
    return latents_mean, latents_std


def _decode_latent_chunk(pipe, latents, latents_mean, latents_std):
    vae_device = next(pipe.vae.parameters()).device
    decode_latents = latents.to(device=vae_device, dtype=pipe.vae.dtype)
    latents_mean = latents_mean.to(device=vae_device, dtype=torch.float32)
    latents_std = latents_std.to(device=vae_device, dtype=torch.float32)
    decode_latents = decode_latents / latents_std + latents_mean
    with torch.no_grad():
        video = pipe.vae.decode(decode_latents, return_dict=False)[0]
    return pipe.video_processor.postprocess_video(video, output_type="pil")[0]


def _compute_psnr(reference_frames, candidate_frames):
    reference = np.stack([np.asarray(frame, dtype=np.float32) for frame in reference_frames], axis=0)
    candidate = np.stack([np.asarray(frame, dtype=np.float32) for frame in candidate_frames], axis=0)
    mse = np.mean((reference - candidate) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * np.log10(255.0) - 10.0 * np.log10(mse)


def _frame_to_token_offset(transformer, latents, frame_offset):
    _, _, _, height, width = latents.shape
    _, p_h, p_w = transformer.config.patch_size
    patches_per_frame = (height // p_h) * (width // p_w)
    return frame_offset * patches_per_frame


def _assert_valid_self_forcing_transformer(transformer):
    has_cross_attn_norm = bool(getattr(transformer.config, "cross_attn_norm", False))
    has_norm_module = transformer.blocks and not isinstance(transformer.blocks[0].norm2, torch.nn.Identity)

    if has_cross_attn_norm and has_norm_module:
        return

    raise ValueError(
        "The loaded transformer is missing Self-Forcing cross-attention norms. "
        "Re-convert the checkpoint with `scripts/convert_wan_to_wan_diffusers.py` before running autoregressive generation."
    )


def _load_pipeline(model_id, wan_base_model_id, device, text_encoder_device=None, vae_device=None):
    transformer = WanTransformer3DModel.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    _assert_valid_self_forcing_transformer(transformer)
    vae = AutoencoderKLWan.from_pretrained(wan_base_model_id, subfolder="vae", torch_dtype=torch.float32)
    vae_device = vae_device or device

    if text_encoder_device is None and vae_device == device:
        pipe = WanPipeline.from_pretrained(
            wan_base_model_id,
            vae=vae,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )
        pipe.scheduler = FlowMatchEulerDiscreteScheduler(shift=5.0, num_train_timesteps=1000)
        pipe.to(device)
        return pipe, device

    text_encoder_device = text_encoder_device or device
    text_encoder_dtype = torch.float32 if text_encoder_device == "cpu" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(wan_base_model_id, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(
        wan_base_model_id,
        subfolder="text_encoder",
        torch_dtype=text_encoder_dtype,
    )
    text_encoder.to(text_encoder_device)

    pipe = WanPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=FlowMatchEulerDiscreteScheduler(shift=5.0, num_train_timesteps=1000),
        transformer=transformer,
    )
    pipe.transformer.to(device)
    pipe.vae.to(vae_device)
    return pipe, text_encoder_device


def _chunk_video_frames(video_frames, frames_per_chunk):
    if len(video_frames) % frames_per_chunk != 0:
        raise ValueError(
            f"`conditioning_video` must have a multiple of {frames_per_chunk} frames, but received {len(video_frames)}."
        )
    return [video_frames[idx : idx + frames_per_chunk] for idx in range(0, len(video_frames), frames_per_chunk)]


def _encode_reference_chunk(pipe, frames, height, width, latents_mean, latents_std, vae_device, transformer_device):
    reference_video = pipe.video_processor.preprocess_video(frames, height=height, width=width).to(vae_device, torch.float32)
    reference_frames = pipe.video_processor.postprocess_video(reference_video, output_type="pil")[0]
    with torch.no_grad():
        encoded = _retrieve_latents(pipe.vae.encode(reference_video.to(dtype=pipe.vae.dtype)), sample_mode="argmax")
    encoded = (encoded.to(torch.float32) - latents_mean) * latents_std
    decoded_frames = _decode_latent_chunk(pipe, encoded, latents_mean, latents_std)
    return (
        encoded.to(device=transformer_device, dtype=pipe.transformer.dtype),
        decoded_frames,
        _compute_psnr(reference_frames, decoded_frames),
    )


def _generate_chunk_velocity(
    pipe,
    noisy_input,
    timestep,
    prompt_embeds,
    negative_prompt_embeds,
    guidance_scale,
    frame_offset,
):
    if timestep.ndim == 0:
        timestep = timestep.expand(noisy_input.shape[0], noisy_input.shape[2])
    elif timestep.ndim == 1:
        timestep = timestep.unsqueeze(1).expand(noisy_input.shape[0], noisy_input.shape[2])

    token_offset = _frame_to_token_offset(pipe.transformer, noisy_input, frame_offset)

    def run_with_overwrite(cache_context, encoder_hidden_states):
        with pipe.transformer.cache_context(cache_context):
            cache_state = get_rolling_kv_cache_state(pipe.transformer)
            prev_should_update = cache_state.should_update_cache
            prev_write_mode = cache_state.write_mode
            prev_absolute_token_offset = cache_state.absolute_token_offset
            try:
                cache_state.should_update_cache = True
                cache_state.configure_cache_write(write_mode="overwrite", absolute_token_offset=token_offset)
                return pipe.transformer(
                    noisy_input,
                    timestep=timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    frame_offset=frame_offset,
                    return_dict=False,
                )[0]
            finally:
                cache_state.should_update_cache = prev_should_update
                cache_state.configure_cache_write(
                    write_mode=prev_write_mode,
                    absolute_token_offset=prev_absolute_token_offset,
                )

    if guidance_scale > 1.0:
        velocity_cond = run_with_overwrite("cond", prompt_embeds)
        velocity_uncond = run_with_overwrite("uncond", negative_prompt_embeds)
        return velocity_uncond + guidance_scale * (velocity_cond - velocity_uncond)

    return run_with_overwrite("cond", prompt_embeds)


def _sample_self_forcing_renoise(latents, *, generator=None):
    batch_size, channels, num_frames, height, width = latents.shape
    noise = torch.randn(
        (batch_size, num_frames, channels, height, width),
        generator=generator,
        device=latents.device,
        dtype=latents.dtype,
    )
    return noise.permute(0, 2, 1, 3, 4).contiguous()


def generate_autoregressive_video(
    prompt,
    negative_prompt="",
    num_chunks=27,
    frames_per_chunk=9,
    height=480,
    width=832,
    guidance_scale=1.0,
    window_size=8000,
    model_id="gueraf/Self-Forcing-diffusers",
    wan_base_model_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    conditioning_video=None,
    conditioning_start_chunk=0,
    device="cuda",
    text_encoder_device=None,
    vae_device=None,
    seed=0,
):
    pipe, prompt_device = _load_pipeline(
        model_id=model_id,
        wan_base_model_id=wan_base_model_id,
        device=device,
        text_encoder_device=text_encoder_device,
        vae_device=vae_device,
    )
    pipe.transformer.enable_cache(
        RollingKVCacheConfig(window_size=window_size, cache_cross_attention=True)
    )

    denoising_steps = _build_sf_denoising_steps(device=device)
    do_cfg = guidance_scale > 1.0
    prompt_dtype = torch.float32 if prompt_device == "cpu" else torch.bfloat16

    with torch.no_grad():
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            do_classifier_free_guidance=do_cfg,
            device=prompt_device,
            dtype=prompt_dtype,
        )
        prompt_embeds = prompt_embeds.to(device=device, dtype=torch.bfloat16)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=torch.bfloat16)

    if text_encoder_device is not None and text_encoder_device != "cpu":
        pipe.text_encoder.to("cpu")
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    p_t, _, _ = pipe.transformer.config.patch_size
    latent_h = height // pipe.vae_scale_factor_spatial
    latent_w = width // pipe.vae_scale_factor_spatial
    latent_t = (frames_per_chunk - 1) // pipe.vae_scale_factor_temporal + 1
    patch_frames_per_chunk = latent_t // p_t
    vae_device = next(pipe.vae.parameters()).device
    latents_mean, latents_std = _get_latent_stats(pipe, vae_device)

    reference_chunks = []
    if conditioning_video is not None:
        video_frames = load_video(conditioning_video)
        for chunk_frames in _chunk_video_frames(video_frames, frames_per_chunk):
            reference_chunks.append(
                _encode_reference_chunk(
                    pipe,
                    chunk_frames,
                    height,
                    width,
                    latents_mean,
                    latents_std,
                    vae_device=vae_device,
                    transformer_device=device,
                )
            )
        if conditioning_start_chunk + len(reference_chunks) > num_chunks:
            raise ValueError("`conditioning_video` would write past `num_chunks`.")
        psnrs = [chunk_psnr for _, _, chunk_psnr in reference_chunks]
        print(f"Conditioning chunk PSNR: min={min(psnrs):.2f}dB mean={sum(psnrs) / len(psnrs):.2f}dB")

    generator = torch.Generator(device=device).manual_seed(seed)
    all_frames = []
    chunk_idx = 0

    while chunk_idx < num_chunks:
        if reference_chunks and chunk_idx == conditioning_start_chunk:
            conditioning_latents = [chunk_latents for chunk_latents, _, _ in reference_chunks]
            reference_frame_offset = chunk_idx * patch_frames_per_chunk
            prefill_rolling_kv_cache(
                pipe.transformer,
                conditioning_latents,
                prompt_embeds,
                frame_offset=reference_frame_offset,
                cache_context="cond",
                write_mode="overwrite",
            )
            if do_cfg:
                prefill_rolling_kv_cache(
                    pipe.transformer,
                    conditioning_latents,
                    negative_prompt_embeds,
                    frame_offset=reference_frame_offset,
                    cache_context="uncond",
                    write_mode="overwrite",
                )

            for _, decoded_frames, chunk_psnr in reference_chunks:
                all_frames.extend(decoded_frames)
                print(f"Injected reference chunk at index {chunk_idx}: decoded PSNR={chunk_psnr:.2f}dB")
                chunk_idx += 1
            continue

        frame_offset = chunk_idx * patch_frames_per_chunk
        noisy_input = torch.randn(
            (1, pipe.vae.config.z_dim, latent_t, latent_h, latent_w),
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )

        with torch.no_grad():
            for step_idx, timestep in enumerate(denoising_steps):
                sigma = timestep / 1000.0
                velocity = _generate_chunk_velocity(
                    pipe,
                    noisy_input,
                    timestep=timestep,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    guidance_scale=guidance_scale,
                    frame_offset=frame_offset,
                )
                x0_pred = (noisy_input.float() - sigma * velocity.float()).to(torch.bfloat16)

                if step_idx < len(denoising_steps) - 1:
                    next_sigma = denoising_steps[step_idx + 1] / 1000.0
                    eps = _sample_self_forcing_renoise(x0_pred, generator=generator)
                    noisy_input = ((1.0 - next_sigma) * x0_pred + next_sigma * eps).to(torch.bfloat16)

        prefill_rolling_kv_cache(
            pipe.transformer,
            x0_pred,
            prompt_embeds,
            frame_offset=frame_offset,
            cache_context="cond",
            write_mode="overwrite",
        )
        if do_cfg:
            prefill_rolling_kv_cache(
                pipe.transformer,
                x0_pred,
                negative_prompt_embeds,
                frame_offset=frame_offset,
                cache_context="uncond",
                write_mode="overwrite",
            )

        all_frames.extend(_decode_latent_chunk(pipe, x0_pred, latents_mean, latents_std))
        print(f"Generated chunk {chunk_idx + 1}/{num_chunks} ({len(all_frames)} frames total)")
        chunk_idx += 1

    pipe.transformer._reset_stateful_cache()
    return all_frames


def main():
    parser = argparse.ArgumentParser(description="Autoregressive Self-Forcing video generation with rolling KV cache")
    parser.add_argument("--prompt", default="A cat walks on the grass, realistic style, high quality")
    parser.add_argument(
        "--negative_prompt",
        default="Bright tones, overexposed, static, blurred details, subtitles, worst quality, low quality",
    )
    parser.add_argument("--num_chunks", type=int, default=27)
    parser.add_argument("--frames_per_chunk", type=int, default=9)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--window_size", type=int, default=8000)
    parser.add_argument("--model_id", default="gueraf/Self-Forcing-diffusers")
    parser.add_argument("--wan_base_model_id", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--conditioning_video", type=str, default=None)
    parser.add_argument("--conditioning_start_chunk", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--text_encoder_device", default=None)
    parser.add_argument("--vae_device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="autoregressive_output.mp4")
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
        conditioning_video=args.conditioning_video,
        conditioning_start_chunk=args.conditioning_start_chunk,
        device=args.device,
        text_encoder_device=args.text_encoder_device,
        vae_device=args.vae_device,
        seed=args.seed,
    )
    export_to_video(frames, args.output, fps=args.fps)
    print(f"Saved {args.output} ({len(frames)} frames @ {args.fps}fps)")


if __name__ == "__main__":
    main()
