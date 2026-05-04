# Copyright 2025 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import pytest
import torch

from diffusers import WanTransformer3DModel
from diffusers.models.transformers.transformer_wan import (
    WanRollingKVBlockCache,
    WanRollingKVCache,
    _wan_rolling_kv_slice_for_overwrite,
    _wan_rolling_kv_trim_to_window,
)
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    AttentionTesterMixin,
    BaseModelTesterConfig,
    BitsAndBytesTesterMixin,
    GGUFCompileTesterMixin,
    GGUFTesterMixin,
    MemoryTesterMixin,
    ModelTesterMixin,
    TorchAoTesterMixin,
    TorchCompileTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


class WanTransformer3DTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return WanTransformer3DModel

    @property
    def pretrained_model_name_or_path(self):
        return "hf-internal-testing/tiny-wan22-transformer"

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (4, 2, 16, 16)

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (4, 2, 16, 16)

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict[str, int | list[int] | tuple | str | bool]:
        return {
            "patch_size": (1, 2, 2),
            "num_attention_heads": 2,
            "attention_head_dim": 12,
            "in_channels": 4,
            "out_channels": 4,
            "text_dim": 16,
            "freq_dim": 256,
            "ffn_dim": 32,
            "num_layers": 2,
            "cross_attn_norm": True,
            "qk_norm": "rms_norm_across_heads",
            "rope_max_seq_len": 32,
        }

    def get_dummy_inputs(self) -> dict[str, torch.Tensor]:
        batch_size = 1
        num_channels = 4
        num_frames = 2
        height = 16
        width = 16
        text_encoder_embedding_dim = 16
        sequence_length = 12

        return {
            "hidden_states": randn_tensor(
                (batch_size, num_channels, num_frames, height, width),
                generator=self.generator,
                device=torch_device,
                dtype=self.torch_dtype,
            ),
            "encoder_hidden_states": randn_tensor(
                (batch_size, sequence_length, text_encoder_embedding_dim),
                generator=self.generator,
                device=torch_device,
                dtype=self.torch_dtype,
            ),
            "timestep": torch.randint(0, 1000, size=(batch_size,), generator=self.generator).to(torch_device),
        }


class TestWanTransformer3D(WanTransformer3DTesterConfig, ModelTesterMixin):
    """Core model tests for Wan Transformer 3D."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
    def test_from_save_pretrained_dtype_inference(self, tmp_path, dtype):
        # Skip: fp16/bf16 require very high atol to pass, providing little signal.
        # Dtype preservation is already tested by test_from_save_pretrained_dtype and test_keep_in_fp32_modules.
        pytest.skip("Tolerance requirements too high for meaningful test")


class TestWanTransformer3DMemory(WanTransformer3DTesterConfig, MemoryTesterMixin):
    """Memory optimization tests for Wan Transformer 3D."""


class TestWanTransformer3DTraining(WanTransformer3DTesterConfig, TrainingTesterMixin):
    """Training tests for Wan Transformer 3D."""

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"WanTransformer3DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)


class TestWanTransformer3DAttention(WanTransformer3DTesterConfig, AttentionTesterMixin):
    """Attention processor tests for Wan Transformer 3D."""


class TestWanTransformer3DCompile(WanTransformer3DTesterConfig, TorchCompileTesterMixin):
    """Torch compile tests for Wan Transformer 3D."""


class TestWanTransformer3DBitsAndBytes(WanTransformer3DTesterConfig, BitsAndBytesTesterMixin):
    """BitsAndBytes quantization tests for Wan Transformer 3D."""

    @property
    def torch_dtype(self):
        return torch.float16

    def get_dummy_inputs(self):
        """Override to provide inputs matching the tiny Wan model dimensions."""
        return {
            "hidden_states": randn_tensor(
                (1, 36, 2, 64, 64), generator=self.generator, device=torch_device, dtype=self.torch_dtype
            ),
            "encoder_hidden_states": randn_tensor(
                (1, 512, 4096), generator=self.generator, device=torch_device, dtype=self.torch_dtype
            ),
            "timestep": torch.tensor([1.0]).to(torch_device, self.torch_dtype),
        }


class TestWanTransformer3DTorchAo(WanTransformer3DTesterConfig, TorchAoTesterMixin):
    """TorchAO quantization tests for Wan Transformer 3D."""

    @property
    def torch_dtype(self):
        return torch.bfloat16

    def get_dummy_inputs(self):
        """Override to provide inputs matching the tiny Wan model dimensions."""
        return {
            "hidden_states": randn_tensor(
                (1, 36, 2, 64, 64), generator=self.generator, device=torch_device, dtype=self.torch_dtype
            ),
            "encoder_hidden_states": randn_tensor(
                (1, 512, 4096), generator=self.generator, device=torch_device, dtype=self.torch_dtype
            ),
            "timestep": torch.tensor([1.0]).to(torch_device, self.torch_dtype),
        }


class TestWanTransformer3DGGUF(WanTransformer3DTesterConfig, GGUFTesterMixin):
    """GGUF quantization tests for Wan Transformer 3D."""

    @property
    def gguf_filename(self):
        return "https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF/blob/main/LowNoise/Wan2.2-I2V-A14B-LowNoise-Q2_K.gguf"

    @property
    def torch_dtype(self):
        return torch.bfloat16

    def _create_quantized_model(self, config_kwargs=None, **extra_kwargs):
        return super()._create_quantized_model(
            config_kwargs, config="Wan-AI/Wan2.2-I2V-A14B-Diffusers", subfolder="transformer", **extra_kwargs
        )

    def get_dummy_inputs(self):
        """Override to provide inputs matching the real Wan I2V model dimensions.

        Wan 2.2 I2V: in_channels=36, text_dim=4096
        """
        return {
            "hidden_states": randn_tensor(
                (1, 36, 2, 64, 64), generator=self.generator, device=torch_device, dtype=self.torch_dtype
            ),
            "encoder_hidden_states": randn_tensor(
                (1, 512, 4096), generator=self.generator, device=torch_device, dtype=self.torch_dtype
            ),
            "timestep": torch.tensor([1.0]).to(torch_device, self.torch_dtype),
        }


class TestWanTransformer3DGGUFCompile(WanTransformer3DTesterConfig, GGUFCompileTesterMixin):
    """GGUF + compile tests for Wan Transformer 3D."""

    @property
    def gguf_filename(self):
        return "https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF/blob/main/LowNoise/Wan2.2-I2V-A14B-LowNoise-Q2_K.gguf"

    @property
    def torch_dtype(self):
        return torch.bfloat16

    def _create_quantized_model(self, config_kwargs=None, **extra_kwargs):
        return super()._create_quantized_model(
            config_kwargs, config="Wan-AI/Wan2.2-I2V-A14B-Diffusers", subfolder="transformer", **extra_kwargs
        )

    def get_dummy_inputs(self):
        """Override to provide inputs matching the real Wan I2V model dimensions.

        Wan 2.2 I2V: in_channels=36, text_dim=4096
        """
        return {
            "hidden_states": randn_tensor(
                (1, 36, 2, 64, 64), generator=self.generator, device=torch_device, dtype=self.torch_dtype
            ),
            "encoder_hidden_states": randn_tensor(
                (1, 512, 4096), generator=self.generator, device=torch_device, dtype=self.torch_dtype
            ),
            "timestep": torch.tensor([1.0]).to(torch_device, self.torch_dtype),
        }


# ===========================================================================
# WanRollingKVCache tests
#
# Geometry used throughout:
#     patch_size = (1, 2, 2)
#     video      = (B=1, C=16, T=1, H=4, W=4)
#     → tokens_per_chunk = (T//p_t) * (H//p_h) * (W//p_w) = 1 * 2 * 2 = 4
#
# Each forward pass with this video shape writes exactly _TOKENS_PER_CHUNK = 4
# tokens into the rolling cache, making size assertions simple and predictable.
# ===========================================================================

_TINY_CONFIG = dict(
    patch_size=[1, 2, 2],
    num_attention_heads=2,
    attention_head_dim=16,
    in_channels=16,
    out_channels=16,
    text_dim=32,
    freq_dim=32,
    ffn_dim=64,
    num_layers=2,
    cross_attn_norm=False,
    qk_norm="rms_norm_across_heads",
    eps=1e-6,
    rope_max_seq_len=32,
)

_NUM_BLOCKS = _TINY_CONFIG["num_layers"]        # 2
_TOKENS_PER_CHUNK = 1 * (4 // 2) * (4 // 2)   # 4


def _make_transformer():
    t = WanTransformer3DModel.from_config(_TINY_CONFIG)
    t.eval()
    return t


def _make_chunk(seed):
    """Deterministic inputs for one 1-frame chunk; different seeds yield different tensors.

    Values are centered around zero so attention weights vary enough for RoPE to have
    a measurable effect on outputs (all-positive inputs collapse to uniform attention).
    """
    n_lat, n_enc = 16 * 4 * 4, 10 * 32
    lat = (torch.arange(n_lat, dtype=torch.float32) - n_lat // 2 + seed * 7).reshape(1, 16, 1, 4, 4) / 50
    enc = (torch.arange(n_enc, dtype=torch.float32) - n_enc // 2 + seed * 7).reshape(1, 10, 32) / 50
    return lat, torch.zeros(1, dtype=torch.long), enc


def _forward(transformer, latents, timestep, enc, *, cache=None, frame_offset=0):
    kwargs = {} if cache is None else {"attention_kwargs": {"rolling_kv_cache": cache}}
    with torch.no_grad():
        return transformer(latents, timestep, enc, frame_offset=frame_offset, return_dict=False, **kwargs)[0]


def _cached_len(cache, block=0):
    k = cache.block_caches[block].cached_key
    return 0 if k is None else k.shape[1]


# token-indexed arange tensor: k[0, i, 0, 0] == float(i), useful for slice/trim assertions
def _tok(n):
    return torch.arange(n, dtype=torch.float32).reshape(1, n, 1, 1)


class TestWanRollingKVBlockCache(unittest.TestCase):

    def test_initial_state_is_all_none(self):
        bc = WanRollingKVBlockCache()
        self.assertIsNone(bc.cached_key)
        self.assertIsNone(bc.cached_value)
        self.assertEqual(bc.cache_start_token_offset, 0)
        self.assertIsNone(bc.cached_cross_key)
        self.assertIsNone(bc.cached_cross_value)
        self.assertIsNone(bc.cached_cross_key_img)
        self.assertIsNone(bc.cached_cross_value_img)

    def test_reset_clears_every_field(self):
        bc = WanRollingKVBlockCache()
        bc.cached_key               = _tok(4)
        bc.cached_value             = _tok(4)
        bc.cache_start_token_offset = 12
        bc.cached_cross_key         = _tok(10)
        bc.cached_cross_value       = _tok(10)
        bc.cached_cross_key_img     = _tok(5)
        bc.cached_cross_value_img   = _tok(5)

        bc.reset()

        self.assertIsNone(bc.cached_key)
        self.assertIsNone(bc.cached_value)
        self.assertEqual(bc.cache_start_token_offset, 0)
        self.assertIsNone(bc.cached_cross_key)
        self.assertIsNone(bc.cached_cross_value)
        self.assertIsNone(bc.cached_cross_key_img)
        self.assertIsNone(bc.cached_cross_value_img)


class TestWanRollingKVCache(unittest.TestCase):

    def test_initial_state(self):
        cache = WanRollingKVCache(num_blocks=3, window_size=100, cache_cross_attention=True)
        self.assertEqual(len(cache.block_caches), 3)
        self.assertEqual(cache.window_size, 100)
        self.assertTrue(cache.cache_cross_attention)
        self.assertTrue(cache.should_update)
        self.assertEqual(cache.write_mode, "append")
        self.assertIsNone(cache.absolute_token_offset)

    def test_configure_write_overwrite(self):
        cache = WanRollingKVCache(num_blocks=2)
        cache.configure_write(write_mode="overwrite", absolute_token_offset=8)
        self.assertEqual(cache.write_mode, "overwrite")
        self.assertEqual(cache.absolute_token_offset, 8)

    def test_configure_write_append(self):
        cache = WanRollingKVCache(num_blocks=2)
        cache.configure_write(write_mode="overwrite", absolute_token_offset=4)  # set first
        cache.configure_write(write_mode="append")                               # then reset
        self.assertEqual(cache.write_mode, "append")
        self.assertIsNone(cache.absolute_token_offset)

    def test_configure_write_invalid_mode(self):
        cache = WanRollingKVCache(num_blocks=2)
        with self.assertRaises(ValueError):
            cache.configure_write(write_mode="replace")

    def test_configure_write_overwrite_requires_offset(self):
        cache = WanRollingKVCache(num_blocks=2)
        with self.assertRaises(ValueError):
            cache.configure_write(write_mode="overwrite")   # missing absolute_token_offset

    def test_configure_write_append_rejects_offset(self):
        cache = WanRollingKVCache(num_blocks=2)
        with self.assertRaises(ValueError):
            cache.configure_write(write_mode="append", absolute_token_offset=4)

    def test_reset_clears_blocks_and_write_state(self):
        cache = WanRollingKVCache(num_blocks=2)
        cache.block_caches[0].cached_key = _tok(4)
        cache.should_update = False
        cache.configure_write(write_mode="overwrite", absolute_token_offset=4)

        cache.reset()

        self.assertIsNone(cache.block_caches[0].cached_key)
        self.assertTrue(cache.should_update)
        self.assertEqual(cache.write_mode, "append")
        self.assertIsNone(cache.absolute_token_offset)


class TestTrimToWindow(unittest.TestCase):
    """Oldest tokens are dropped when the cache exceeds window_size.

    _tok(n) gives a tensor where token i has value float(i), so after trimming
    we can assert the surviving token ids as a plain Python list.
    """

    def test_unlimited_window_keeps_everything(self):
        k = _tok(12)
        k2, _, start = _wan_rolling_kv_trim_to_window(k, k, cache_start=0, window_size=-1)
        self.assertEqual(k2.squeeze().tolist(), list(range(12)))
        self.assertEqual(start, 0)

    def test_trim_drops_oldest_and_advances_start(self):
        # 12 tokens, window=8 → drop tokens 0-3, keep 4-11
        k = _tok(12)
        k2, _, start = _wan_rolling_kv_trim_to_window(k, k, cache_start=0, window_size=8)
        self.assertEqual(k2.squeeze().tolist(), list(range(4, 12)))
        self.assertEqual(start, 4)

    def test_no_trim_when_within_window(self):
        # 6 tokens already fit inside window=8 → nothing dropped
        k = _tok(6)
        k2, _, start = _wan_rolling_kv_trim_to_window(k, k, cache_start=2, window_size=8)
        self.assertEqual(k2.squeeze().tolist(), list(range(6)))
        self.assertEqual(start, 2)

    def test_trim_accumulates_with_pre_existing_offset(self):
        # cache already starts at token 4 (8 tokens retained), window=6 → drop 2 more → start=6
        k = _tok(8)
        k2, _, start = _wan_rolling_kv_trim_to_window(k, k, cache_start=4, window_size=6)
        self.assertEqual(k2.squeeze().tolist(), list(range(2, 8)))
        self.assertEqual(start, 6)


class TestSliceForOverwrite(unittest.TestCase):

    def test_empty_cache_returns_none_prefix(self):
        bc = WanRollingKVBlockCache()
        k, v, start = _wan_rolling_kv_slice_for_overwrite(bc, absolute_token_offset=0)
        self.assertIsNone(k)
        self.assertIsNone(v)
        self.assertEqual(start, 0)

    def test_overwrite_at_position_zero_yields_empty_prefix(self):
        # cache holds tokens [0..7]; overwrite at 0 → prefix is empty
        bc = WanRollingKVBlockCache()
        bc.cached_key = bc.cached_value = _tok(8)
        bc.cache_start_token_offset = 0
        k, _, start = _wan_rolling_kv_slice_for_overwrite(bc, absolute_token_offset=0)
        self.assertEqual(k.squeeze().tolist(), [])
        self.assertEqual(start, 0)

    def test_overwrite_at_midpoint_returns_first_half_as_prefix(self):
        # cache holds tokens [0..7]; overwrite at 4 → prefix is tokens [0, 1, 2, 3]
        bc = WanRollingKVBlockCache()
        bc.cached_key = bc.cached_value = _tok(8)
        bc.cache_start_token_offset = 0
        k, _, start = _wan_rolling_kv_slice_for_overwrite(bc, absolute_token_offset=4)
        self.assertEqual(k.squeeze().tolist(), [0.0, 1.0, 2.0, 3.0])
        self.assertEqual(start, 0)

    def test_overwrite_beyond_end_raises(self):
        bc = WanRollingKVBlockCache()
        bc.cached_key = bc.cached_value = _tok(4)
        bc.cache_start_token_offset = 0
        with self.assertRaises(ValueError):
            _wan_rolling_kv_slice_for_overwrite(bc, absolute_token_offset=8)  # beyond end

    def test_overwrite_before_window_start_returns_none(self):
        # Window trimmed: holds tokens [8..15] (start=8); writing at offset=4 → no usable prefix
        bc = WanRollingKVBlockCache()
        bc.cached_key = bc.cached_value = _tok(8)
        bc.cache_start_token_offset = 8
        k, _, start = _wan_rolling_kv_slice_for_overwrite(bc, absolute_token_offset=4)
        self.assertIsNone(k)
        self.assertEqual(start, 4)


class TestAppendMode(unittest.TestCase):
    """Running N chunks with append mode grows the cache by _TOKENS_PER_CHUNK each time."""

    def setUp(self):
        self.transformer = _make_transformer()
        self.cache = WanRollingKVCache(num_blocks=_NUM_BLOCKS, window_size=-1)

    def test_three_chunks_grow_cache_four_tokens_at_a_time(self):
        self.assertEqual(_cached_len(self.cache), 0)                          # start: empty
        for expected, seed in zip([4, 8, 12], [1, 2, 3]):
            _forward(self.transformer, *_make_chunk(seed), cache=self.cache)
            self.assertEqual(_cached_len(self.cache), expected)

    def test_six_chunks_grow_cache_incrementally(self):
        sizes = []
        for seed in range(1, 7):
            _forward(self.transformer, *_make_chunk(seed), cache=self.cache)
            sizes.append(_cached_len(self.cache))
        self.assertEqual(sizes, [4, 8, 12, 16, 20, 24])

    def test_prefix_is_preserved_across_appends(self):
        """Chunk 1's keys stay in slots [0:4] regardless of how many chunks follow."""
        for seed in [1, 2, 3]:
            _forward(self.transformer, *_make_chunk(seed), cache=self.cache)

        keys = self.cache.block_caches[0].cached_key   # shape (1, 12, heads, dim)

        # Re-run just chunk 1 into a fresh cache to capture its keys
        fresh = WanRollingKVCache(num_blocks=_NUM_BLOCKS)
        _forward(self.transformer, *_make_chunk(1), cache=fresh)
        chunk1_keys = fresh.block_caches[0].cached_key  # (1, 4, heads, dim)

        # Run chunks 1+2 into another fresh cache to capture chunk 2's keys
        fresh2 = WanRollingKVCache(num_blocks=_NUM_BLOCKS)
        for seed in [1, 2]:
            _forward(self.transformer, *_make_chunk(seed), cache=fresh2)
        chunk2_keys = fresh2.block_caches[0].cached_key[:, _TOKENS_PER_CHUNK:]  # (1, 4, ...)

        torch.testing.assert_close(keys[:, :_TOKENS_PER_CHUNK], chunk1_keys)
        torch.testing.assert_close(keys[:, _TOKENS_PER_CHUNK:_TOKENS_PER_CHUNK * 2], chunk2_keys)

    def test_all_blocks_grow_simultaneously(self):
        _forward(self.transformer, *_make_chunk(1), cache=self.cache)
        self.assertEqual([_cached_len(self.cache, b) for b in range(_NUM_BLOCKS)],
                         [_TOKENS_PER_CHUNK] * _NUM_BLOCKS)

    def test_output_differs_when_attending_to_previous_context(self):
        """Chunk 2 output changes when chunk 1 is in the cache vs. a fresh cache."""
        _forward(self.transformer, *_make_chunk(1), cache=self.cache)
        out_with_context    = _forward(self.transformer, *_make_chunk(2), cache=self.cache)
        out_without_context = _forward(self.transformer, *_make_chunk(2),
                                       cache=WanRollingKVCache(num_blocks=_NUM_BLOCKS))
        self.assertFalse(torch.allclose(out_with_context, out_without_context))

    def test_should_update_false_freezes_cache(self):
        _forward(self.transformer, *_make_chunk(1), cache=self.cache)
        keys_snapshot = self.cache.block_caches[0].cached_key.clone()

        self.cache.should_update = False
        _forward(self.transformer, *_make_chunk(2), cache=self.cache)

        self.assertEqual(_cached_len(self.cache), _TOKENS_PER_CHUNK)
        torch.testing.assert_close(self.cache.block_caches[0].cached_key, keys_snapshot)

    def test_reset_clears_all_blocks(self):
        _forward(self.transformer, *_make_chunk(1), cache=self.cache)
        self.cache.reset()
        self.assertEqual(
            [(bc.cached_key, bc.cached_value, bc.cache_start_token_offset)
             for bc in self.cache.block_caches],
            [(None, None, 0)] * _NUM_BLOCKS,
        )


class TestWindowSize(unittest.TestCase):
    """Cache drops oldest tokens when it exceeds window_size."""

    def test_window_one_chunk_sizes_stay_capped(self):
        transformer = _make_transformer()
        cache = WanRollingKVCache(num_blocks=_NUM_BLOCKS, window_size=_TOKENS_PER_CHUNK)
        sizes = []
        for seed in range(1, 4):
            _forward(transformer, *_make_chunk(seed), cache=cache)
            sizes.append(_cached_len(cache))
        self.assertEqual(sizes, [4, 4, 4])  # always capped at window size

    def test_window_two_chunks_sizes_grow_then_cap(self):
        transformer = _make_transformer()
        cache = WanRollingKVCache(num_blocks=_NUM_BLOCKS, window_size=_TOKENS_PER_CHUNK * 2)
        sizes = []
        for seed in range(1, 7):
            _forward(transformer, *_make_chunk(seed), cache=cache)
            sizes.append(_cached_len(cache))
        self.assertEqual(sizes, [4, 8, 8, 8, 8, 8])  # grows to window then stays

    def test_window_trims_oldest_and_advances_start_offset(self):
        transformer = _make_transformer()
        cache = WanRollingKVCache(num_blocks=_NUM_BLOCKS, window_size=_TOKENS_PER_CHUNK)
        starts = []
        for seed in range(1, 4):
            _forward(transformer, *_make_chunk(seed), cache=cache)
            starts.append(cache.block_caches[0].cache_start_token_offset)
        self.assertEqual(starts, [0, 4, 8])  # advances by _TOKENS_PER_CHUNK each time


class TestOverwriteMode(unittest.TestCase):
    """Overwrite mode places K/V at an explicit absolute token position."""

    def setUp(self):
        self.transformer = _make_transformer()
        self.cache = WanRollingKVCache(num_blocks=_NUM_BLOCKS, window_size=-1)

    def test_overwrite_two_chunks_sequentially_gives_eight_tokens(self):
        for frame, seed in enumerate([1, 2]):
            self.cache.configure_write(write_mode="overwrite", absolute_token_offset=frame * _TOKENS_PER_CHUNK)
            _forward(self.transformer, *_make_chunk(seed), cache=self.cache, frame_offset=frame)
        self.assertEqual(_cached_len(self.cache), _TOKENS_PER_CHUNK * 2)  # 8

    def test_overwriting_last_chunk_changes_its_tokens_but_not_total_length(self):
        # Self-Forcing pattern: each denoising step overwrites the current generation chunk
        # (always the tail); the prefix (earlier chunks) must be preserved.
        self.cache.configure_write(write_mode="overwrite", absolute_token_offset=0)
        _forward(self.transformer, *_make_chunk(1), cache=self.cache)

        self.cache.configure_write(write_mode="overwrite", absolute_token_offset=_TOKENS_PER_CHUNK)
        _forward(self.transformer, *_make_chunk(10), cache=self.cache, frame_offset=1)
        keys_v1 = self.cache.block_caches[0].cached_key[:, _TOKENS_PER_CHUNK:].clone()

        self.cache.configure_write(write_mode="overwrite", absolute_token_offset=_TOKENS_PER_CHUNK)
        _forward(self.transformer, *_make_chunk(20), cache=self.cache, frame_offset=1)
        keys_v2 = self.cache.block_caches[0].cached_key[:, _TOKENS_PER_CHUNK:].clone()

        self.assertEqual(_cached_len(self.cache), _TOKENS_PER_CHUNK * 2)
        self.assertFalse(torch.allclose(keys_v2, keys_v1))


class TestCrossAttentionCache(unittest.TestCase):
    """Cross-attn K/V are computed once and reused across chunks."""

    def setUp(self):
        self.transformer = _make_transformer()
        self.cache = WanRollingKVCache(num_blocks=_NUM_BLOCKS, window_size=-1, cache_cross_attention=True)

    def test_cross_attn_populated_after_first_chunk(self):
        _forward(self.transformer, *_make_chunk(1), cache=self.cache)
        self.assertEqual(
            [bc.cached_cross_key is not None for bc in self.cache.block_caches],
            [True] * _NUM_BLOCKS,
        )

    def test_cross_attn_not_recomputed_on_second_chunk(self):
        """The cached tensor identity must be identical — not recomputed."""
        _, _, e1 = _make_chunk(1)
        _forward(self.transformer, *_make_chunk(1), cache=self.cache)
        key_id_after_chunk_1 = id(self.cache.block_caches[0].cached_cross_key)

        lat2, t2, _ = _make_chunk(2)
        _forward(self.transformer, lat2, t2, e1, cache=self.cache)  # reuse same e1
        self.assertEqual(id(self.cache.block_caches[0].cached_cross_key), key_id_after_chunk_1)

    def test_cross_attn_cache_not_affected_by_should_update_false(self):
        """Cross-attn K/V is written on first occurrence regardless of should_update."""
        self.cache.should_update = False
        _forward(self.transformer, *_make_chunk(1), cache=self.cache)
        self.assertEqual(
            [bc.cached_cross_key is not None for bc in self.cache.block_caches],
            [True] * _NUM_BLOCKS,
        )


class TestFrameOffset(unittest.TestCase):
    """frame_offset controls temporal RoPE positions for each chunk."""

    def test_different_frame_offsets_produce_different_outputs(self):
        transformer = _make_transformer()
        chunk = _make_chunk(42)
        out_0 = _forward(transformer, *chunk, cache=WanRollingKVCache(num_blocks=_NUM_BLOCKS), frame_offset=0)
        out_1 = _forward(transformer, *chunk, cache=WanRollingKVCache(num_blocks=_NUM_BLOCKS), frame_offset=1)
        self.assertFalse(torch.allclose(out_0, out_1))

    def test_sequential_frame_offsets_grow_cache_correctly(self):
        transformer = _make_transformer()
        cache = WanRollingKVCache(num_blocks=_NUM_BLOCKS, window_size=-1)
        sizes = []
        for frame_offset in range(3):
            _forward(transformer, *_make_chunk(frame_offset), cache=cache, frame_offset=frame_offset)
            sizes.append(_cached_len(cache))
        self.assertEqual(sizes, [4, 8, 12])
