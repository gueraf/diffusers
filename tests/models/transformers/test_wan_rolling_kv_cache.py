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

"""Tests for WanRollingKVCache and friends.

Geometry used throughout:
    patch_size = (1, 2, 2)
    video      = (B=1, C=16, T=1, H=4, W=4)
    → tokens_per_chunk = (T//p_t) * (H//p_h) * (W//p_w) = 1 * 2 * 2 = 4

Each forward pass with this video shape writes exactly TOKENS_PER_CHUNK = 4 tokens
into the rolling cache, making size assertions simple and predictable.
"""

import unittest

import torch

from diffusers import WanTransformer3DModel
from diffusers.models.transformers.transformer_wan import (
    WanRollingKVBlockCache,
    WanRollingKVCache,
    _wan_rolling_kv_slice_for_overwrite,
    _wan_rolling_kv_trim_to_window,
)


# ---------------------------------------------------------------------------
# Tiny model config — fast on CPU, no GPU required
# ---------------------------------------------------------------------------
TINY_CONFIG = dict(
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

NUM_BLOCKS = TINY_CONFIG["num_layers"]           # 2
TOKENS_PER_CHUNK = 1 * (4 // 2) * (4 // 2)     # 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_transformer():
    t = WanTransformer3DModel.from_config(TINY_CONFIG)
    t.eval()
    return t


def make_chunk(seed):
    """Deterministic (hidden_states, timestep, encoder_hidden_states) for one 1-frame chunk."""
    torch.manual_seed(seed)
    return (
        torch.randn(1, 16, 1, 4, 4),   # (B, C, T, H, W)
        torch.zeros(1, dtype=torch.long),
        torch.randn(1, 10, 32),
    )


def forward(transformer, latents, timestep, enc, *, cache=None, frame_offset=0):
    kwargs = {} if cache is None else {"attention_kwargs": {"rolling_kv_cache": cache}}
    with torch.no_grad():
        return transformer(latents, timestep, enc, frame_offset=frame_offset, return_dict=False, **kwargs)[0]


def cached_len(cache, block=0):
    k = cache.block_caches[block].cached_key
    return 0 if k is None else k.shape[1]


# ===========================================================================
# WanRollingKVBlockCache
# ===========================================================================

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
        bc.cached_key              = torch.randn(1, 4, 2, 16)
        bc.cached_value            = torch.randn(1, 4, 2, 16)
        bc.cache_start_token_offset = 12
        bc.cached_cross_key        = torch.randn(1, 10, 2, 16)
        bc.cached_cross_value      = torch.randn(1, 10, 2, 16)
        bc.cached_cross_key_img    = torch.randn(1, 5, 2, 16)
        bc.cached_cross_value_img  = torch.randn(1, 5, 2, 16)

        bc.reset()

        self.assertIsNone(bc.cached_key)
        self.assertIsNone(bc.cached_value)
        self.assertEqual(bc.cache_start_token_offset, 0)
        self.assertIsNone(bc.cached_cross_key)
        self.assertIsNone(bc.cached_cross_value)
        self.assertIsNone(bc.cached_cross_key_img)
        self.assertIsNone(bc.cached_cross_value_img)


# ===========================================================================
# WanRollingKVCache
# ===========================================================================

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
        cache.block_caches[0].cached_key = torch.randn(1, 4, 2, 16)
        cache.should_update = False
        cache.configure_write(write_mode="overwrite", absolute_token_offset=4)

        cache.reset()

        self.assertIsNone(cache.block_caches[0].cached_key)
        self.assertTrue(cache.should_update)
        self.assertEqual(cache.write_mode, "append")
        self.assertIsNone(cache.absolute_token_offset)


# ===========================================================================
# Helper: _wan_rolling_kv_trim_to_window
# ===========================================================================

class TestTrimToWindow(unittest.TestCase):
    """Oldest tokens are dropped when the cache exceeds window_size."""

    def test_unlimited_window_keeps_everything(self):
        k = torch.randn(1, 12, 2, 16)
        v = torch.randn(1, 12, 2, 16)
        k2, v2, start = _wan_rolling_kv_trim_to_window(k, v, cache_start=0, window_size=-1)
        self.assertEqual(k2.shape[1], 12)
        self.assertEqual(start, 0)

    def test_trim_drops_oldest_and_advances_start(self):
        # 12 tokens with window=8: drop oldest 4, retain [4..11]
        k = torch.randn(1, 12, 2, 16)
        v = torch.randn(1, 12, 2, 16)
        k2, v2, start = _wan_rolling_kv_trim_to_window(k, v, cache_start=0, window_size=8)
        self.assertEqual(k2.shape[1], 8)
        self.assertEqual(start, 4)
        torch.testing.assert_close(k2, k[:, 4:])

    def test_no_trim_when_within_window(self):
        k = torch.randn(1, 6, 2, 16)
        v = torch.randn(1, 6, 2, 16)
        k2, v2, start = _wan_rolling_kv_trim_to_window(k, v, cache_start=2, window_size=8)
        self.assertEqual(k2.shape[1], 6)
        self.assertEqual(start, 2)

    def test_trim_accumulates_with_pre_existing_offset(self):
        # cache_start=4, 8 tokens, window=6 → drop 2 tokens → start=6
        k = torch.randn(1, 8, 2, 16)
        v = torch.randn(1, 8, 2, 16)
        k2, v2, start = _wan_rolling_kv_trim_to_window(k, v, cache_start=4, window_size=6)
        self.assertEqual(k2.shape[1], 6)
        self.assertEqual(start, 6)


# ===========================================================================
# Helper: _wan_rolling_kv_slice_for_overwrite
# ===========================================================================

class TestSliceForOverwrite(unittest.TestCase):

    def test_empty_cache_returns_none_prefix(self):
        bc = WanRollingKVBlockCache()
        k, v, start = _wan_rolling_kv_slice_for_overwrite(bc, absolute_token_offset=0)
        self.assertIsNone(k)
        self.assertIsNone(v)
        self.assertEqual(start, 0)

    def test_overwrite_at_position_zero_yields_empty_prefix(self):
        bc = WanRollingKVBlockCache()
        bc.cached_key   = torch.randn(1, 8, 2, 16)
        bc.cached_value = torch.randn(1, 8, 2, 16)
        bc.cache_start_token_offset = 0
        k, v, start = _wan_rolling_kv_slice_for_overwrite(bc, absolute_token_offset=0)
        self.assertEqual(k.shape[1], 0)
        self.assertEqual(start, 0)

    def test_overwrite_at_midpoint_returns_first_half_as_prefix(self):
        bc = WanRollingKVBlockCache()
        bc.cached_key   = torch.randn(1, 8, 2, 16)
        bc.cached_value = torch.randn(1, 8, 2, 16)
        bc.cache_start_token_offset = 0
        k, v, start = _wan_rolling_kv_slice_for_overwrite(bc, absolute_token_offset=4)
        # tokens [0, 1, 2, 3] are the prefix; tokens [4..7] are replaced
        self.assertEqual(k.shape[1], 4)
        self.assertEqual(start, 0)
        torch.testing.assert_close(k, bc.cached_key[:, :4])

    def test_overwrite_beyond_end_raises(self):
        bc = WanRollingKVBlockCache()
        bc.cached_key   = torch.randn(1, 4, 2, 16)
        bc.cached_value = torch.randn(1, 4, 2, 16)
        bc.cache_start_token_offset = 0
        with self.assertRaises(ValueError):
            _wan_rolling_kv_slice_for_overwrite(bc, absolute_token_offset=8)  # beyond end

    def test_overwrite_before_window_start_returns_none(self):
        # Window has been trimmed: start=8, holds tokens [8..15]
        # Writing at offset=4 (before the window) → can't use any prefix
        bc = WanRollingKVBlockCache()
        bc.cached_key   = torch.randn(1, 8, 2, 16)
        bc.cached_value = torch.randn(1, 8, 2, 16)
        bc.cache_start_token_offset = 8
        k, v, start = _wan_rolling_kv_slice_for_overwrite(bc, absolute_token_offset=4)
        self.assertIsNone(k)
        self.assertEqual(start, 4)


# ===========================================================================
# Append mode: forward passes grow the cache
# ===========================================================================

class TestAppendMode(unittest.TestCase):
    """Running N chunks with append mode grows the cache by TOKENS_PER_CHUNK each time."""

    def setUp(self):
        self.transformer = make_transformer()
        self.cache = WanRollingKVCache(num_blocks=NUM_BLOCKS, window_size=-1)

    def test_three_chunks_grow_cache_four_tokens_at_a_time(self):
        c1, t1, e1 = make_chunk(seed=1)
        c2, t2, e2 = make_chunk(seed=2)
        c3, t3, e3 = make_chunk(seed=3)

        self.assertEqual(cached_len(self.cache), 0)                     # start: empty

        forward(self.transformer, c1, t1, e1, cache=self.cache)
        self.assertEqual(cached_len(self.cache), TOKENS_PER_CHUNK)      # 4

        forward(self.transformer, c2, t2, e2, cache=self.cache)
        self.assertEqual(cached_len(self.cache), TOKENS_PER_CHUNK * 2)  # 8

        forward(self.transformer, c3, t3, e3, cache=self.cache)
        self.assertEqual(cached_len(self.cache), TOKENS_PER_CHUNK * 3)  # 12

    def test_six_chunks_grow_cache_incrementally(self):
        for i, seed in enumerate([1, 2, 3, 4, 5, 6], start=1):
            c, t, e = make_chunk(seed=seed)
            forward(self.transformer, c, t, e, cache=self.cache)
            self.assertEqual(cached_len(self.cache), TOKENS_PER_CHUNK * i, f"after chunk {i}")

    def test_prefix_is_preserved_across_appends(self):
        """Chunk 1's keys stay in slots [0:4] regardless of how many chunks follow."""
        c1, t1, e1 = make_chunk(seed=1)
        c2, t2, e2 = make_chunk(seed=2)
        c3, t3, e3 = make_chunk(seed=3)

        forward(self.transformer, c1, t1, e1, cache=self.cache)
        keys_after_chunk_1 = self.cache.block_caches[0].cached_key.clone()

        forward(self.transformer, c2, t2, e2, cache=self.cache)
        keys_after_chunk_2 = self.cache.block_caches[0].cached_key.clone()

        forward(self.transformer, c3, t3, e3, cache=self.cache)
        keys_after_chunk_3 = self.cache.block_caches[0].cached_key.clone()

        # Chunk 1's 4 tokens are always in slots [0:4]
        torch.testing.assert_close(keys_after_chunk_2[:, :TOKENS_PER_CHUNK],
                                   keys_after_chunk_1[:, :TOKENS_PER_CHUNK])
        torch.testing.assert_close(keys_after_chunk_3[:, :TOKENS_PER_CHUNK],
                                   keys_after_chunk_1[:, :TOKENS_PER_CHUNK])

        # Chunk 2's 4 tokens are always in slots [4:8]
        torch.testing.assert_close(
            keys_after_chunk_3[:, TOKENS_PER_CHUNK : TOKENS_PER_CHUNK * 2],
            keys_after_chunk_2[:, TOKENS_PER_CHUNK : TOKENS_PER_CHUNK * 2],
        )

    def test_all_blocks_grow_simultaneously(self):
        c1, t1, e1 = make_chunk(seed=1)
        forward(self.transformer, c1, t1, e1, cache=self.cache)
        for block_idx in range(NUM_BLOCKS):
            with self.subTest(block=block_idx):
                self.assertEqual(cached_len(self.cache, block=block_idx), TOKENS_PER_CHUNK)

    def test_output_differs_when_attending_to_previous_context(self):
        """Chunk 2 output changes when chunk 1 is in the cache vs. a fresh cache."""
        c1, t1, e1 = make_chunk(seed=1)
        c2, t2, e2 = make_chunk(seed=2)

        # Populate cache with chunk 1, then run chunk 2
        forward(self.transformer, c1, t1, e1, cache=self.cache)
        out_with_context = forward(self.transformer, c2, t2, e2, cache=self.cache)

        # Fresh cache: chunk 2 runs without any prior context
        fresh_cache = WanRollingKVCache(num_blocks=NUM_BLOCKS)
        out_without_context = forward(self.transformer, c2, t2, e2, cache=fresh_cache)

        self.assertFalse(
            torch.allclose(out_with_context, out_without_context),
            "Chunk 2 must attend to chunk 1's cached context, changing its output",
        )

    def test_should_update_false_freezes_cache(self):
        c1, t1, e1 = make_chunk(seed=1)
        c2, t2, e2 = make_chunk(seed=2)

        forward(self.transformer, c1, t1, e1, cache=self.cache)
        keys_snapshot = self.cache.block_caches[0].cached_key.clone()
        size_after_chunk_1 = cached_len(self.cache)

        self.cache.should_update = False
        forward(self.transformer, c2, t2, e2, cache=self.cache)

        # Size unchanged, keys unchanged
        self.assertEqual(cached_len(self.cache), size_after_chunk_1)
        torch.testing.assert_close(self.cache.block_caches[0].cached_key, keys_snapshot)

    def test_reset_clears_all_blocks(self):
        c1, t1, e1 = make_chunk(seed=1)
        forward(self.transformer, c1, t1, e1, cache=self.cache)
        self.cache.reset()
        for block_idx in range(NUM_BLOCKS):
            self.assertIsNone(self.cache.block_caches[block_idx].cached_key)
            self.assertIsNone(self.cache.block_caches[block_idx].cached_value)
            self.assertEqual(self.cache.block_caches[block_idx].cache_start_token_offset, 0)


# ===========================================================================
# Window size
# ===========================================================================

class TestWindowSize(unittest.TestCase):
    """Cache drops oldest tokens when it exceeds window_size."""

    def test_window_one_chunk_keeps_only_the_latest(self):
        transformer = make_transformer()
        cache = WanRollingKVCache(num_blocks=NUM_BLOCKS, window_size=TOKENS_PER_CHUNK)

        for seed in [1, 2, 3]:
            c, t, e = make_chunk(seed=seed)
            forward(transformer, c, t, e, cache=cache)
            self.assertEqual(cached_len(cache), TOKENS_PER_CHUNK,
                             f"after chunk {seed}: window should cap at {TOKENS_PER_CHUNK}")

    def test_window_two_chunks_caps_at_eight_tokens(self):
        transformer = make_transformer()
        cache = WanRollingKVCache(num_blocks=NUM_BLOCKS, window_size=TOKENS_PER_CHUNK * 2)

        for seed in [1, 2, 3, 4, 5, 6]:
            c, t, e = make_chunk(seed=seed)
            forward(transformer, c, t, e, cache=cache)

        self.assertEqual(cached_len(cache), TOKENS_PER_CHUNK * 2)

    def test_window_trims_oldest_and_advances_start_offset(self):
        transformer = make_transformer()
        cache = WanRollingKVCache(num_blocks=NUM_BLOCKS, window_size=TOKENS_PER_CHUNK)

        c1, t1, e1 = make_chunk(seed=1)
        forward(transformer, c1, t1, e1, cache=cache)
        self.assertEqual(cache.block_caches[0].cache_start_token_offset, 0)

        c2, t2, e2 = make_chunk(seed=2)
        forward(transformer, c2, t2, e2, cache=cache)
        # window=4, had 4+4=8 → trimmed 4, start advances to 4
        self.assertEqual(cache.block_caches[0].cache_start_token_offset, TOKENS_PER_CHUNK)


# ===========================================================================
# Overwrite mode
# ===========================================================================

class TestOverwriteMode(unittest.TestCase):
    """Overwrite mode places K/V at an explicit absolute token position."""

    def setUp(self):
        self.transformer = make_transformer()
        self.cache = WanRollingKVCache(num_blocks=NUM_BLOCKS, window_size=-1)

    def test_overwrite_two_chunks_sequentially_gives_eight_tokens(self):
        c1, t1, e1 = make_chunk(seed=1)
        c2, t2, e2 = make_chunk(seed=2)

        # Chunk 1 at position 0
        self.cache.configure_write(write_mode="overwrite", absolute_token_offset=0)
        forward(self.transformer, c1, t1, e1, cache=self.cache, frame_offset=0)
        self.assertEqual(cached_len(self.cache), TOKENS_PER_CHUNK)      # 4

        # Chunk 2 immediately after chunk 1
        self.cache.configure_write(write_mode="overwrite", absolute_token_offset=TOKENS_PER_CHUNK)
        forward(self.transformer, c2, t2, e2, cache=self.cache, frame_offset=1)
        self.assertEqual(cached_len(self.cache), TOKENS_PER_CHUNK * 2)  # 8

    def test_overwriting_last_chunk_changes_its_tokens_but_not_total_length(self):
        # Self-Forcing pattern: each denoising step overwrites the current generation chunk,
        # which is always the tail.  The prefix (earlier chunks) must be preserved.
        c1, t1, e1    = make_chunk(seed=1)
        c2a, t2a, e2a = make_chunk(seed=10)  # chunk 2, version A (first denoising step)
        c2b, t2b, e2b = make_chunk(seed=20)  # chunk 2, version B (second denoising step)

        # Prefill chunk 1
        self.cache.configure_write(write_mode="overwrite", absolute_token_offset=0)
        forward(self.transformer, c1, t1, e1, cache=self.cache)
        self.assertEqual(cached_len(self.cache), TOKENS_PER_CHUNK)  # 4

        # First denoising step for chunk 2
        self.cache.configure_write(write_mode="overwrite", absolute_token_offset=TOKENS_PER_CHUNK)
        forward(self.transformer, c2a, t2a, e2a, cache=self.cache, frame_offset=1)
        keys_chunk2_v1 = self.cache.block_caches[0].cached_key[:, TOKENS_PER_CHUNK:].clone()
        self.assertEqual(cached_len(self.cache), TOKENS_PER_CHUNK * 2)  # 8

        # Second denoising step — overwrite chunk 2 again (same tail position)
        self.cache.configure_write(write_mode="overwrite", absolute_token_offset=TOKENS_PER_CHUNK)
        forward(self.transformer, c2b, t2b, e2b, cache=self.cache, frame_offset=1)
        keys_chunk2_v2 = self.cache.block_caches[0].cached_key[:, TOKENS_PER_CHUNK:].clone()

        self.assertEqual(cached_len(self.cache), TOKENS_PER_CHUNK * 2,
                         "Total cache length must not change when overwriting the tail chunk")
        self.assertFalse(torch.allclose(keys_chunk2_v2, keys_chunk2_v1),
                         "Chunk 2 keys must change after overwriting with different input")


# ===========================================================================
# Cross-attention caching
# ===========================================================================

class TestCrossAttentionCache(unittest.TestCase):
    """Cross-attn K/V are computed once and reused across chunks."""

    def setUp(self):
        self.transformer = make_transformer()
        self.cache = WanRollingKVCache(num_blocks=NUM_BLOCKS, window_size=-1, cache_cross_attention=True)

    def test_cross_attn_populated_after_first_chunk(self):
        c1, t1, e1 = make_chunk(seed=1)
        forward(self.transformer, c1, t1, e1, cache=self.cache)
        for block_idx in range(NUM_BLOCKS):
            with self.subTest(block=block_idx):
                self.assertIsNotNone(self.cache.block_caches[block_idx].cached_cross_key)
                self.assertIsNotNone(self.cache.block_caches[block_idx].cached_cross_value)

    def test_cross_attn_not_recomputed_on_second_chunk(self):
        """The cached tensor identity (not just value) must be identical — not recomputed."""
        c1, t1, e1 = make_chunk(seed=1)
        c2, t2, _  = make_chunk(seed=2)

        forward(self.transformer, c1, t1, e1, cache=self.cache)
        cross_key_after_chunk_1 = self.cache.block_caches[0].cached_cross_key

        # Re-use the same encoder embeddings (e1) on chunk 2 — cache must not recompute
        forward(self.transformer, c2, t2, e1, cache=self.cache)
        cross_key_after_chunk_2 = self.cache.block_caches[0].cached_cross_key

        self.assertIs(cross_key_after_chunk_1, cross_key_after_chunk_2,
                      "Cross-attn K must be the same Python object on the second pass (not recomputed)")

    def test_cross_attn_cache_not_affected_by_should_update_false(self):
        """Cross-attn K/V is written on first occurrence regardless of should_update."""
        self.cache.should_update = False
        c1, t1, e1 = make_chunk(seed=1)
        forward(self.transformer, c1, t1, e1, cache=self.cache)
        self.assertIsNotNone(self.cache.block_caches[0].cached_cross_key,
                             "Cross-attn cache is populated on first pass even when should_update=False")


# ===========================================================================
# Frame offset
# ===========================================================================

class TestFrameOffset(unittest.TestCase):
    """frame_offset controls temporal RoPE positions for each chunk."""

    def test_different_frame_offsets_produce_different_outputs(self):
        transformer = make_transformer()
        c, t, e = make_chunk(seed=42)

        cache_0 = WanRollingKVCache(num_blocks=NUM_BLOCKS)
        out_at_frame_0 = forward(transformer, c, t, e, cache=cache_0, frame_offset=0)

        cache_1 = WanRollingKVCache(num_blocks=NUM_BLOCKS)
        out_at_frame_1 = forward(transformer, c, t, e, cache=cache_1, frame_offset=1)

        self.assertFalse(torch.allclose(out_at_frame_0, out_at_frame_1),
                         "Different frame offsets must yield different RoPE positions and thus different outputs")

    def test_sequential_frame_offsets_grow_cache_correctly(self):
        """Chunks at offsets 0, 1, 2 each write TOKENS_PER_CHUNK into the same cache."""
        transformer = make_transformer()
        cache = WanRollingKVCache(num_blocks=NUM_BLOCKS, window_size=-1)

        for frame_offset in range(3):
            c, t, e = make_chunk(seed=frame_offset)
            forward(transformer, c, t, e, cache=cache, frame_offset=frame_offset)
            self.assertEqual(cached_len(cache), TOKENS_PER_CHUNK * (frame_offset + 1),
                             f"after frame_offset={frame_offset}")


if __name__ == "__main__":
    unittest.main()
