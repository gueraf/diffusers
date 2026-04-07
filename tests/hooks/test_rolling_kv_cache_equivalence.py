"""Step-by-step equivalence tests between original Self-Forcing and our diffusers implementation.

Compares:
  original: guandeh17/Self-Forcing code + gdhe17/Self-Forcing weights (vendored in tyrannis)
  ours:     gueraf/Self-Forcing-diffusers + WanPipeline + rolling_kv_cache hook

Tests are ordered from shallow to deep:
  1. Weight equivalence  — same numerical tensors after conversion
  2. RoPE equivalence    — causal_rope_apply vs WanRotaryPosEmbed with frame_offset
  3. Single-chunk forward — identical noise/prompt → identical transformer output
  4. Multi-chunk forward  — KV cache populated identically across chunks

Usage (requires the vendored tyrannis checkout and gdhe17 checkpoint):
    pytest tests/hooks/test_rolling_kv_cache_equivalence.py -v

Environment:
    TYRANNIS_ROOT  path to the vendored tyrannis repo root
                   (default: /home/fabian/odyssey/ml_core/inference/predictor/vendor/tyrannis)
    SF_CKPT        path to self_forcing_dmd.pt
                   (default: ~/.cache/huggingface/hub/models--gdhe17--Self-Forcing/
                              snapshots/2f8b779212da279d212c22a509b66ad6552f350e/
                              checkpoints/self_forcing_dmd.pt)
    SF_DIFFUSERS   HF repo or local path for converted diffusers transformer
                   (default: gueraf/Self-Forcing-diffusers)
"""

import os
import sys
import math
import pytest
import torch

# ------------------------------------------------------------------
# Paths / environment
# ------------------------------------------------------------------

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
SF_DIFFUSERS = os.environ.get("SF_DIFFUSERS", "gueraf/Self-Forcing-diffusers")
WAN_BASE = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

tyrannis_available = os.path.isdir(TYRANNIS_ROOT) and os.path.isfile(SF_CKPT)
pytestmark = pytest.mark.skipif(
    not tyrannis_available,
    reason=f"tyrannis not found at {TYRANNIS_ROOT} or checkpoint missing at {SF_CKPT}",
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

# Tiny but valid video dimensions (must be divisible by VAE scale factors × patch size)
HEIGHT = 480
WIDTH = 832
# 17 pixel frames → 5 latent frames with temporal scale factor 4, p_t=1
FRAMES_PER_CHUNK = 17

PROMPT = "A cat walks on the grass, realistic"


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture(scope="module")
def tyrannis_pipeline():
    """Load original Self-Forcing pipeline from vendored tyrannis."""
    if TYRANNIS_ROOT not in sys.path:
        sys.path.insert(0, TYRANNIS_ROOT)

    from omegaconf import OmegaConf
    from pipeline import CausalInferencePipeline

    cfg_path = os.path.join(TYRANNIS_ROOT, "configs", "self_forcing_dmd.yaml")
    default_cfg_path = os.path.join(TYRANNIS_ROOT, "configs", "default_config.yaml")
    config = OmegaConf.merge(OmegaConf.load(default_cfg_path), OmegaConf.load(cfg_path))
    # Override resolution / frames
    config.height = HEIGHT
    config.width = WIDTH
    config.num_frames = 21  # total latent frames (original default)

    pipeline = CausalInferencePipeline(config, device=DEVICE)

    # Load checkpoint
    ckpt = torch.load(SF_CKPT, map_location="cpu", weights_only=True)
    raw_sd = ckpt["generator_ema"]
    # Strip 'model.' prefix (WanDiffusionWrapper wraps the model)
    clean_sd = {(k[len("model."):] if k.startswith("model.") else k): v
                for k, v in raw_sd.items()}
    target_keys = set(pipeline.generator.model.state_dict().keys())
    # Only keep keys present in the target
    clean_sd = {k: v for k, v in clean_sd.items() if k in target_keys}
    pipeline.generator.model.load_state_dict(clean_sd, strict=False)
    pipeline.generator.model.eval()

    pipeline.to(dtype=DTYPE)
    pipeline.generator.to(DEVICE)
    pipeline.text_encoder.to(DEVICE)
    pipeline.vae.to(DEVICE)
    return pipeline


@pytest.fixture(scope="module")
def diffusers_pipeline():
    """Load our diffusers pipeline with Self-Forcing transformer."""
    # Add local src to path in case we're not installed
    src_path = os.path.join(os.path.dirname(__file__), "..", "..", "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    from diffusers import AutoencoderKLWan, WanPipeline, WanTransformer3DModel
    from diffusers.hooks import RollingKVCacheConfig, apply_rolling_kv_cache, get_rolling_kv_cache_state

    transformer = WanTransformer3DModel.from_pretrained(SF_DIFFUSERS, torch_dtype=DTYPE)
    vae = AutoencoderKLWan.from_pretrained(WAN_BASE, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(WAN_BASE, vae=vae, transformer=transformer, torch_dtype=DTYPE)
    pipe.to(DEVICE)

    apply_rolling_kv_cache(pipe.transformer, RollingKVCacheConfig(
        window_size=8000,
        cache_cross_attention=True,
    ))

    return pipe


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _original_latent_shape(height, width, num_latent_frames):
    """Original format: (B, T, C, H, W) with spatial downscale ×8."""
    h = height // 8
    w = width // 8
    return (1, num_latent_frames, 16, h, w)


def _diffusers_latent_shape(height, width, num_latent_frames):
    """Diffusers format: (B, C, T, H, W) with spatial downscale ×8."""
    h = height // 8
    w = width // 8
    return (1, 16, num_latent_frames, h, w)


def _orig_to_diffusers(latent):
    """(B, T, C, H, W) → (B, C, T, H, W)."""
    return latent.permute(0, 2, 1, 3, 4).contiguous()


def _diffusers_to_orig(latent):
    """(B, C, T, H, W) → (B, T, C, H, W)."""
    return latent.permute(0, 2, 1, 3, 4).contiguous()


# ------------------------------------------------------------------
# Test 1: Weight equivalence
# ------------------------------------------------------------------

class TestWeightEquivalence:
    """Verify the diffusers weight conversion preserves values numerically."""

    def test_self_attn_q_weight(self, tyrannis_pipeline, diffusers_pipeline):
        """attn1.to_q.weight should be identical in both models."""
        orig_q = tyrannis_pipeline.generator.model.blocks[0].self_attn.q.weight
        diff_q = diffusers_pipeline.transformer.blocks[0].attn1.to_q.weight
        assert orig_q.shape == diff_q.shape, (
            f"Shape mismatch: orig {orig_q.shape} vs diffusers {diff_q.shape}"
        )
        torch.testing.assert_close(orig_q.cpu().float(), diff_q.cpu().float(), atol=1e-4, rtol=1e-4)

    def test_self_attn_k_weight(self, tyrannis_pipeline, diffusers_pipeline):
        orig_k = tyrannis_pipeline.generator.model.blocks[0].self_attn.k.weight
        diff_k = diffusers_pipeline.transformer.blocks[0].attn1.to_k.weight
        torch.testing.assert_close(orig_k.cpu().float(), diff_k.cpu().float(), atol=1e-4, rtol=1e-4)

    def test_cross_attn_q_weight(self, tyrannis_pipeline, diffusers_pipeline):
        orig_q = tyrannis_pipeline.generator.model.blocks[0].cross_attn.q.weight
        diff_q = diffusers_pipeline.transformer.blocks[0].attn2.to_q.weight
        torch.testing.assert_close(orig_q.cpu().float(), diff_q.cpu().float(), atol=1e-4, rtol=1e-4)

    def test_ffn_weight(self, tyrannis_pipeline, diffusers_pipeline):
        orig_ffn = tyrannis_pipeline.generator.model.blocks[0].ffn[0].weight
        diff_ffn = diffusers_pipeline.transformer.blocks[0].ffn.net[0].proj.weight
        torch.testing.assert_close(orig_ffn.cpu().float(), diff_ffn.cpu().float(), atol=1e-4, rtol=1e-4)

    def test_all_blocks_match(self, tyrannis_pipeline, diffusers_pipeline):
        """Check that Q/K/V weights match across all 30 transformer blocks."""
        mismatches = []
        for i in range(30):
            ob = tyrannis_pipeline.generator.model.blocks[i]
            db = diffusers_pipeline.transformer.blocks[i]
            for name, orig, diff in [
                ("self_attn.q", ob.self_attn.q.weight, db.attn1.to_q.weight),
                ("self_attn.k", ob.self_attn.k.weight, db.attn1.to_k.weight),
                ("self_attn.v", ob.self_attn.v.weight, db.attn1.to_v.weight),
                ("self_attn.o", ob.self_attn.o.weight, db.attn1.to_out[0].weight),
            ]:
                if not torch.allclose(orig.cpu().float(), diff.cpu().float(), atol=1e-4, rtol=1e-4):
                    mismatches.append(f"block[{i}].{name}")
        assert not mismatches, f"Weight mismatches: {mismatches}"


# ------------------------------------------------------------------
# Test 2: RoPE position equivalence
# ------------------------------------------------------------------

class TestRoPEEquivalence:
    """causal_rope_apply(start_frame=N) == WanRotaryPosEmbed(frame_offset=N)."""

    def test_rope_chunk0_matches(self, tyrannis_pipeline, diffusers_pipeline):
        """First chunk: both use positions 0..ppf-1."""
        self._compare_rope(tyrannis_pipeline, diffusers_pipeline, chunk_idx=0)

    def test_rope_chunk1_matches(self, tyrannis_pipeline, diffusers_pipeline):
        """Second chunk: original uses start_frame=3, ours uses frame_offset=ppf."""
        self._compare_rope(tyrannis_pipeline, diffusers_pipeline, chunk_idx=1)

    def test_rope_chunk5_matches(self, tyrannis_pipeline, diffusers_pipeline):
        """Chunk 5 to exercise larger offsets."""
        self._compare_rope(tyrannis_pipeline, diffusers_pipeline, chunk_idx=5)

    def _compare_rope(self, tyrannis_pipeline, diffusers_pipeline, chunk_idx):
        if TYRANNIS_ROOT not in sys.path:
            sys.path.insert(0, TYRANNIS_ROOT)
        from wan.modules.causal_model import causal_rope_apply

        # Diffusers rope config
        transformer = diffusers_pipeline.transformer
        p_t, p_h, p_w = transformer.config.patch_size
        latent_t = (FRAMES_PER_CHUNK - 1) // 4 + 1   # VAE temporal scale = 4
        latent_h = HEIGHT // 8
        latent_w = WIDTH // 8
        ppf = latent_t // p_t
        pph = latent_h // p_h
        ppw = latent_w // p_w
        frame_offset = chunk_idx * ppf

        # Diffusers rope output (cos, sin)
        dummy_latent = torch.zeros(1, 16, latent_t, latent_h, latent_w, device=DEVICE)
        freqs_cos, freqs_sin = transformer.rope(dummy_latent, frame_offset=frame_offset)
        # freqs_cos/sin: (1, ppf*pph*ppw, 1, head_dim)

        # Original rope freqs: need to extract freqs from original model
        # The original uses complex freqs; compare by checking that after
        # applying both, the same Q vector is produced.
        seq_len = ppf * pph * ppw
        head_dim = transformer.config.attention_head_dim
        torch.manual_seed(42)
        q = torch.randn(1, seq_len, transformer.config.num_attention_heads, head_dim,
                        device=DEVICE, dtype=DTYPE)

        # Diffusers apply
        from diffusers.hooks.rolling_kv_cache import _apply_wan_rotary_emb
        q_diffusers = _apply_wan_rotary_emb(q, freqs_cos, freqs_sin)

        # Original apply
        orig_model = tyrannis_pipeline.generator.model
        grid_sizes = (ppf, pph, ppw)
        # causal_rope_apply expects (B, seq_len, n_heads, head_dim) - same shape
        q_orig = causal_rope_apply(q.clone(), grid_sizes, orig_model.freqs, start_frame=frame_offset)

        torch.testing.assert_close(
            q_diffusers.cpu().float(),
            q_orig.cpu().float(),
            atol=1e-3, rtol=1e-3,
            msg=f"RoPE mismatch at chunk_idx={chunk_idx} (frame_offset={frame_offset})"
        )


# ------------------------------------------------------------------
# Test 3: Single-chunk forward pass equivalence
# ------------------------------------------------------------------

class TestSingleChunkForward:
    """Same noise + prompt → same transformer output (first chunk, chunk_idx=0)."""

    @pytest.fixture(autouse=True)
    def reset_cache(self, diffusers_pipeline):
        """Reset rolling KV cache before each test."""
        from diffusers.hooks import get_rolling_kv_cache_state
        state = get_rolling_kv_cache_state(diffusers_pipeline.transformer)
        # Reset by toggling should_update_cache and clearing block states
        if hasattr(diffusers_pipeline.transformer, "_diffusers_hook"):
            diffusers_pipeline.transformer._diffusers_hook.reset_stateful_hooks()
        yield
        if hasattr(diffusers_pipeline.transformer, "_diffusers_hook"):
            diffusers_pipeline.transformer._diffusers_hook.reset_stateful_hooks()

    def test_forward_output_shape(self, tyrannis_pipeline, diffusers_pipeline):
        latent_t = (FRAMES_PER_CHUNK - 1) // 4 + 1
        latent_h = HEIGHT // 8
        latent_w = WIDTH // 8

        noise_orig = torch.randn(
            _original_latent_shape(HEIGHT, WIDTH, latent_t),
            device=DEVICE, dtype=DTYPE
        )
        noise_diff = _orig_to_diffusers(noise_orig)

        cond = tyrannis_pipeline.text_encoder(text_prompts=[PROMPT])
        prompt_embeds = diffusers_pipeline.encode_prompt(
            prompt=PROMPT, negative_prompt=None,
            do_classifier_free_guidance=False,
            device=DEVICE, dtype=DTYPE
        )[0]

        timestep = torch.ones([1, latent_t], device=DEVICE, dtype=torch.int64) * 1000

        with torch.no_grad():
            pred_orig, _ = tyrannis_pipeline.generator(
                noisy_image_or_video=noise_orig,
                conditional_dict=cond,
                timestep=timestep,
                kv_cache=tyrannis_pipeline.kv_cache1,
                crossattn_cache=tyrannis_pipeline.crossattn_cache,
                current_start=0,
            )
            pred_diff = diffusers_pipeline.transformer(
                noise_diff,
                timestep=torch.tensor([1000], device=DEVICE, dtype=torch.long),
                encoder_hidden_states=prompt_embeds,
                frame_offset=0,
                return_dict=False,
            )[0]

        # Shape: orig (B, T, C, H, W) vs diff (B, C, T, H, W)
        assert pred_orig.shape == _original_latent_shape(HEIGHT, WIDTH, latent_t), pred_orig.shape
        assert pred_diff.shape == _diffusers_latent_shape(HEIGHT, WIDTH, latent_t), pred_diff.shape

    def test_forward_values_close(self, tyrannis_pipeline, diffusers_pipeline):
        """The numerical outputs should be close after accounting for format differences."""
        latent_t = (FRAMES_PER_CHUNK - 1) // 4 + 1
        latent_h = HEIGHT // 8
        latent_w = WIDTH // 8

        torch.manual_seed(0)
        noise_orig = torch.randn(
            _original_latent_shape(HEIGHT, WIDTH, latent_t),
            device=DEVICE, dtype=DTYPE
        )
        noise_diff = _orig_to_diffusers(noise_orig)

        cond = tyrannis_pipeline.text_encoder(text_prompts=[PROMPT])
        prompt_embeds = diffusers_pipeline.encode_prompt(
            prompt=PROMPT, negative_prompt=None,
            do_classifier_free_guidance=False,
            device=DEVICE, dtype=DTYPE
        )[0]

        timestep_orig = torch.ones([1, latent_t], device=DEVICE, dtype=torch.int64) * 750
        timestep_diff = torch.tensor([750], device=DEVICE, dtype=torch.long)

        # Init kv cache for original
        tyrannis_pipeline._initialize_kv_cache(batch_size=1, dtype=DTYPE, device=DEVICE)
        tyrannis_pipeline._initialize_crossattn_cache(batch_size=1, dtype=DTYPE, device=DEVICE)

        with torch.no_grad():
            pred_orig, _ = tyrannis_pipeline.generator(
                noisy_image_or_video=noise_orig,
                conditional_dict=cond,
                timestep=timestep_orig,
                kv_cache=tyrannis_pipeline.kv_cache1,
                crossattn_cache=tyrannis_pipeline.crossattn_cache,
                current_start=0,
            )
            pred_diff = diffusers_pipeline.transformer(
                noise_diff,
                timestep=timestep_diff,
                encoder_hidden_states=prompt_embeds,
                frame_offset=0,
                return_dict=False,
            )[0]

        # Convert orig to diffusers format for comparison
        pred_orig_diff_fmt = _orig_to_diffusers(pred_orig)

        max_diff = (pred_orig_diff_fmt.float() - pred_diff.float()).abs().max().item()
        mean_diff = (pred_orig_diff_fmt.float() - pred_diff.float()).abs().mean().item()
        print(f"\nSingle-chunk forward: max_diff={max_diff:.4f}, mean_diff={mean_diff:.6f}")
        assert max_diff < 0.5, (
            f"Forward outputs diverge too much: max_diff={max_diff:.4f}. "
            f"Check text encoder or attention implementation."
        )


# ------------------------------------------------------------------
# Test 4: Multi-chunk KV cache equivalence
# ------------------------------------------------------------------

class TestMultiChunkCacheEquivalence:
    """After populating KV cache from chunk 0, chunk 1 outputs should match."""

    def test_chunk1_after_cache_population(self, tyrannis_pipeline, diffusers_pipeline):
        """
        Steps:
        1. Denoise chunk 0 with both models (no cache yet).
        2. Populate KV cache with clean forward pass at t=0.
        3. Denoise chunk 1 with cache active.
        4. Compare outputs.
        """
        from diffusers.hooks import get_rolling_kv_cache_state
        if TYRANNIS_ROOT not in sys.path:
            sys.path.insert(0, TYRANNIS_ROOT)

        latent_t = (FRAMES_PER_CHUNK - 1) // 4 + 1
        latent_h = HEIGHT // 8
        latent_w = WIDTH // 8

        # Init original cache
        tyrannis_pipeline._initialize_kv_cache(batch_size=1, dtype=DTYPE, device=DEVICE)
        tyrannis_pipeline._initialize_crossattn_cache(batch_size=1, dtype=DTYPE, device=DEVICE)

        # Init diffusers cache
        if hasattr(diffusers_pipeline.transformer, "_diffusers_hook"):
            diffusers_pipeline.transformer._diffusers_hook.reset_stateful_hooks()
        cache_state = get_rolling_kv_cache_state(diffusers_pipeline.transformer)

        cond = tyrannis_pipeline.text_encoder(text_prompts=[PROMPT])
        prompt_embeds = diffusers_pipeline.encode_prompt(
            prompt=PROMPT, negative_prompt=None,
            do_classifier_free_guidance=False,
            device=DEVICE, dtype=DTYPE
        )[0]

        torch.manual_seed(1)
        noise0_orig = torch.randn(_original_latent_shape(HEIGHT, WIDTH, latent_t), device=DEVICE, dtype=DTYPE)
        noise0_diff = _orig_to_diffusers(noise0_orig)

        ts1000_orig = torch.ones([1, latent_t], device=DEVICE, dtype=torch.int64) * 1000
        ts0_orig = torch.zeros([1, latent_t], device=DEVICE, dtype=torch.int64)

        # --- Chunk 0: denoise (no cache reads for first chunk) ---
        with torch.no_grad():
            # Original: suppress cache writes during denoising by running normally
            # (original never suppresses during denoising — cache reads go to empty cache)
            pred_orig_0, clean_orig_0 = tyrannis_pipeline.generator(
                noisy_image_or_video=noise0_orig,
                conditional_dict=cond,
                timestep=ts1000_orig,
                kv_cache=tyrannis_pipeline.kv_cache1,
                crossattn_cache=tyrannis_pipeline.crossattn_cache,
                current_start=0,
            )

            # Diffusers: same but suppressing cache writes during denoising
            cache_state.should_update_cache = False
            pred_diff_0 = diffusers_pipeline.transformer(
                noise0_diff,
                timestep=torch.tensor([1000], device=DEVICE, dtype=torch.long),
                encoder_hidden_states=prompt_embeds,
                frame_offset=0,
                return_dict=False,
            )[0]

        # --- Cache population with clean t=0 forward pass ---
        denoised_0_orig = clean_orig_0  # original returns (noise_pred, denoised_pred)
        denoised_0_diff = _orig_to_diffusers(denoised_0_orig)

        with torch.no_grad():
            # Original: rerun at t=0 to write K/V to cache
            tyrannis_pipeline.generator(
                noisy_image_or_video=denoised_0_orig,
                conditional_dict=cond,
                timestep=ts0_orig,
                kv_cache=tyrannis_pipeline.kv_cache1,
                crossattn_cache=tyrannis_pipeline.crossattn_cache,
                current_start=0,
            )

            # Diffusers: same
            cache_state.should_update_cache = True
            diffusers_pipeline.transformer(
                denoised_0_diff,
                timestep=torch.zeros(1, device=DEVICE, dtype=torch.long),
                encoder_hidden_states=prompt_embeds,
                frame_offset=0,
                return_dict=False,
            )

        # --- Chunk 1: denoise with cache active ---
        torch.manual_seed(2)
        noise1_orig = torch.randn(_original_latent_shape(HEIGHT, WIDTH, latent_t), device=DEVICE, dtype=DTYPE)
        noise1_diff = _orig_to_diffusers(noise1_orig)

        ppf = latent_t  # p_t=1
        frame_offset_1 = 1 * ppf  # chunk 1 starts at ppf

        with torch.no_grad():
            # Original: current_start = frame_offset_1 * frame_seq_length
            frame_seq_length = (latent_h // 2) * (latent_w // 2)  # pph * ppw
            pred_orig_1, _ = tyrannis_pipeline.generator(
                noisy_image_or_video=noise1_orig,
                conditional_dict=cond,
                timestep=ts1000_orig,
                kv_cache=tyrannis_pipeline.kv_cache1,
                crossattn_cache=tyrannis_pipeline.crossattn_cache,
                current_start=frame_offset_1 * frame_seq_length,
            )

            # Diffusers: frame_offset=ppf
            cache_state.should_update_cache = False
            pred_diff_1 = diffusers_pipeline.transformer(
                noise1_diff,
                timestep=torch.tensor([1000], device=DEVICE, dtype=torch.long),
                encoder_hidden_states=prompt_embeds,
                frame_offset=frame_offset_1,
                return_dict=False,
            )[0]

        pred_orig_1_diff_fmt = _orig_to_diffusers(pred_orig_1)
        max_diff = (pred_orig_1_diff_fmt.float() - pred_diff_1.float()).abs().max().item()
        mean_diff = (pred_orig_1_diff_fmt.float() - pred_diff_1.float()).abs().mean().item()
        print(f"\nChunk-1 forward (with cache): max_diff={max_diff:.4f}, mean_diff={mean_diff:.6f}")
        assert max_diff < 0.5, (
            f"Chunk 1 outputs diverge: max_diff={max_diff:.4f}. "
            f"Check KV cache format or RoPE frame_offset."
        )

        # Cleanup
        if hasattr(diffusers_pipeline.transformer, "_diffusers_hook"):
            diffusers_pipeline.transformer._diffusers_hook.reset_stateful_hooks()
