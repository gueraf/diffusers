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

import torch

from diffusers.hooks.rolling_kv_cache import (
    RollingKVCacheBlockState,
    RollingKVCacheCrossAttnBlockState,
    RollingKVCacheState,
    _apply_wan_rotary_emb,
)
from diffusers.hooks import RollingKVCacheConfig, apply_rolling_kv_cache, get_rolling_kv_cache_state


# Tiny WanTransformer3DModel config for fast tests (no GPU needed)
WAN_TINY_CONFIG = {
    "patch_size": [1, 2, 2],
    "num_attention_heads": 2,
    "attention_head_dim": 16,
    "in_channels": 16,
    "out_channels": 16,
    "text_dim": 32,
    "freq_dim": 32,
    "ffn_dim": 64,
    "num_layers": 2,
    "cross_attn_norm": False,
    "qk_norm": "rms_norm_across_heads",
    "eps": 1e-6,
    "image_dim": None,
    "added_kv_proj_dim": None,
    "rope_max_seq_len": 32,
}

# hidden_states shape: (B, in_channels, T, H, W)
# After patch_embedding: (B, inner_dim, T//p_t, H//p_h, W//p_w) → flattened to (B, seq_len, inner_dim)
# With T=1, H=4, W=4, patch_size=(1,2,2): seq_len = 1*2*2 = 4
_BATCH = 1
_T, _H, _W = 1, 4, 4
_IN_CHANNELS = 16
_TEXT_SEQ_LEN = 10
_TEXT_DIM = 32  # must match config text_dim


def _make_transformer(config=None):
    from diffusers import WanTransformer3DModel

    return WanTransformer3DModel.from_config(config or WAN_TINY_CONFIG)


def _make_inputs(batch=_BATCH, t=_T, h=_H, w=_W, text_seq=_TEXT_SEQ_LEN, text_dim=_TEXT_DIM):
    hidden_states = torch.randn(batch, _IN_CHANNELS, t, h, w)
    timestep = torch.zeros(batch, dtype=torch.long)
    encoder_hidden_states = torch.randn(batch, text_seq, text_dim)
    return hidden_states, timestep, encoder_hidden_states


class TestStateClasses(unittest.TestCase):
    def test_rolling_kv_cache_state_defaults(self):
        state = RollingKVCacheState()
        self.assertTrue(state.should_update_cache)

    def test_rolling_kv_cache_state_reset(self):
        state = RollingKVCacheState()
        state.should_update_cache = False
        state.reset()
        self.assertTrue(state.should_update_cache)

    def test_block_state_defaults(self):
        state = RollingKVCacheBlockState()
        self.assertIsNone(state.cached_key)
        self.assertIsNone(state.cached_value)

    def test_block_state_reset(self):
        state = RollingKVCacheBlockState()
        state.cached_key = torch.randn(1, 4, 2, 16)
        state.cached_value = torch.randn(1, 4, 2, 16)
        state.reset()
        self.assertIsNone(state.cached_key)
        self.assertIsNone(state.cached_value)

    def test_cross_attn_block_state_defaults(self):
        state = RollingKVCacheCrossAttnBlockState()
        self.assertIsNone(state.cached_key)
        self.assertIsNone(state.cached_value)
        self.assertIsNone(state.cached_key_img)
        self.assertIsNone(state.cached_value_img)

    def test_cross_attn_block_state_reset(self):
        state = RollingKVCacheCrossAttnBlockState()
        state.cached_key = torch.randn(1, 4, 2, 16)
        state.cached_value = torch.randn(1, 4, 2, 16)
        state.cached_key_img = torch.randn(1, 4, 2, 16)
        state.cached_value_img = torch.randn(1, 4, 2, 16)
        state.reset()
        self.assertIsNone(state.cached_key)
        self.assertIsNone(state.cached_value)
        self.assertIsNone(state.cached_key_img)
        self.assertIsNone(state.cached_value_img)


class TestApplyWanRotaryEmb(unittest.TestCase):
    def test_output_shape_preserved(self):
        # hidden: (B, seq_len, heads, head_dim)
        x = torch.randn(1, 4, 2, 16)
        freqs_cos = torch.ones(1, 4, 1, 16)
        freqs_sin = torch.zeros(1, 4, 1, 16)
        out = _apply_wan_rotary_emb(x, freqs_cos, freqs_sin)
        self.assertEqual(out.shape, x.shape)

    def test_zero_sin_identity(self):
        # cos=1, sin=0 everywhere → rotation by 0 → output == input
        x = torch.randn(1, 4, 2, 16)
        freqs_cos = torch.ones(1, 4, 1, 16)
        freqs_sin = torch.zeros(1, 4, 1, 16)
        out = _apply_wan_rotary_emb(x, freqs_cos, freqs_sin)
        self.assertTrue(torch.allclose(out, x, atol=1e-6), "Zero sin should give identity rotation")

    def test_dtype_preserved(self):
        x = torch.randn(1, 4, 2, 16, dtype=torch.bfloat16)
        freqs_cos = torch.ones(1, 4, 1, 16)
        freqs_sin = torch.zeros(1, 4, 1, 16)
        out = _apply_wan_rotary_emb(x, freqs_cos, freqs_sin)
        self.assertEqual(out.dtype, x.dtype)


class TestApplyRollingKVCache(unittest.TestCase):
    def setUp(self):
        self.transformer = _make_transformer()
        self.transformer.eval()
        self.config = RollingKVCacheConfig(window_size=-1, cache_cross_attention=False)
        apply_rolling_kv_cache(self.transformer, self.config)

    def _get_block_state(self, block_idx=0):
        """Return the RollingKVCacheBlockState for the given block."""
        from diffusers.hooks.rolling_kv_cache import _ROLLING_KV_CACHE_SELF_ATTN_HOOK
        from diffusers.models.transformers.transformer_wan import WanTransformerBlock

        for i, m in enumerate(self.transformer.modules()):
            if isinstance(m, WanTransformerBlock) and i == 0 or True:
                if hasattr(m, "_diffusers_hook"):
                    break

        blocks = [m for m in self.transformer.modules()
                  if m.__class__.__name__ == "WanTransformerBlock"]
        block = blocks[block_idx]
        hook = block.attn1._diffusers_hook.get_hook(_ROLLING_KV_CACHE_SELF_ATTN_HOOK)
        if hook.block_state_manager._current_context is None:
            hook.block_state_manager.set_context("test")
        return hook.block_state_manager.get_state()

    def test_hooks_attached_to_all_blocks(self):
        from diffusers.hooks.rolling_kv_cache import _ROLLING_KV_CACHE_SELF_ATTN_HOOK
        from diffusers.models.transformers.transformer_wan import WanTransformerBlock

        blocks = [m for m in self.transformer.modules()
                  if isinstance(m, WanTransformerBlock)]
        self.assertEqual(len(blocks), WAN_TINY_CONFIG["num_layers"])

        for block in blocks:
            self.assertTrue(hasattr(block.attn1, "_diffusers_hook"))
            hook = block.attn1._diffusers_hook.get_hook(_ROLLING_KV_CACHE_SELF_ATTN_HOOK)
            self.assertIsNotNone(hook)

    def test_cache_empty_before_first_pass(self):
        block_state = self._get_block_state(0)
        self.assertIsNone(block_state.cached_key)
        self.assertIsNone(block_state.cached_value)

    def test_cache_populated_after_first_pass(self):
        hidden_states, timestep, enc = _make_inputs()
        with torch.no_grad():
            self.transformer(hidden_states, timestep, enc)

        block_state = self._get_block_state(0)
        self.assertIsNotNone(block_state.cached_key)
        self.assertIsNotNone(block_state.cached_value)

        # seq_len from one chunk = 4 tokens (1*2*2 patches)
        expected_seq_len = _T * (_H // 2) * (_W // 2)
        self.assertEqual(block_state.cached_key.shape[1], expected_seq_len)

    def test_cache_grows_after_second_chunk(self):
        hidden_states, timestep, enc = _make_inputs()
        with torch.no_grad():
            self.transformer(hidden_states, timestep, enc)
        after_first = self._get_block_state(0).cached_key.shape[1]

        hidden_states2, timestep2, enc2 = _make_inputs()
        with torch.no_grad():
            self.transformer(hidden_states2, timestep2, enc2)
        after_second = self._get_block_state(0).cached_key.shape[1]

        self.assertEqual(after_second, after_first * 2,
                         "Cache should double after two equal-size chunks")

    def test_window_size_limits_cache(self):
        # window_size = 4 = exactly one chunk's worth
        config = RollingKVCacheConfig(window_size=4)
        transformer = _make_transformer()
        transformer.eval()
        apply_rolling_kv_cache(transformer, config)

        from diffusers.hooks.rolling_kv_cache import _ROLLING_KV_CACHE_SELF_ATTN_HOOK
        blocks = [m for m in transformer.modules()
                  if m.__class__.__name__ == "WanTransformerBlock"]
        hook = blocks[0].attn1._diffusers_hook.get_hook(_ROLLING_KV_CACHE_SELF_ATTN_HOOK)
        if hook.block_state_manager._current_context is None:
            hook.block_state_manager.set_context("test")
        block_state = hook.block_state_manager.get_state()

        hidden_states, timestep, enc = _make_inputs()
        with torch.no_grad():
            transformer(hidden_states, timestep, enc)
        self.assertEqual(block_state.cached_key.shape[1], 4)

        with torch.no_grad():
            transformer(hidden_states, timestep, enc)
        # After second chunk: total would be 8, but window=4, so should be 4
        self.assertEqual(block_state.cached_key.shape[1], 4,
                         "Window size should cap the cached tokens")

    def test_no_cache_update_when_flag_false(self):
        cache_state = get_rolling_kv_cache_state(self.transformer)
        self.assertIsNotNone(cache_state)

        # First pass: populate cache
        hidden_states, timestep, enc = _make_inputs()
        with torch.no_grad():
            self.transformer(hidden_states, timestep, enc)
        after_first = self._get_block_state(0).cached_key.shape[1]

        # Second pass with should_update_cache=False
        cache_state.should_update_cache = False
        with torch.no_grad():
            self.transformer(hidden_states, timestep, enc)
        after_second = self._get_block_state(0).cached_key.shape[1]

        self.assertEqual(after_first, after_second,
                         "Cache should not grow when should_update_cache=False")

    def test_second_pass_output_differs_with_cache(self):
        """Second chunk output should differ depending on whether a first-chunk cache exists."""
        # Populate the cache with chunk 1
        hs1, ts1, enc1 = _make_inputs()
        with torch.no_grad():
            self.transformer(hs1, ts1, enc1)

        # Chunk 2 inputs (different from chunk 1)
        torch.manual_seed(42)
        hs2, ts2, enc2 = _make_inputs()
        with torch.no_grad():
            out_with_cache = self.transformer(hs2, ts2, enc2).sample

        # Now run chunk 2 from a fresh transformer (no cache)
        fresh = _make_transformer()
        fresh.load_state_dict(self.transformer.state_dict())
        fresh.eval()
        apply_rolling_kv_cache(fresh, self.config)
        with torch.no_grad():
            out_no_cache = fresh(hs2, ts2, enc2).sample

        self.assertFalse(torch.allclose(out_with_cache, out_no_cache),
                         "Output should differ when attending to cached context from a previous chunk")

    def test_get_rolling_kv_cache_state(self):
        cache_state = get_rolling_kv_cache_state(self.transformer)
        self.assertIsNotNone(cache_state)
        self.assertIsInstance(cache_state, RollingKVCacheState)
        self.assertTrue(cache_state.should_update_cache)

    def test_get_rolling_kv_cache_state_returns_none_without_hook(self):
        plain = _make_transformer()
        state = get_rolling_kv_cache_state(plain)
        self.assertIsNone(state)

    def test_reset_stateful_hooks(self):
        """After hook reset, the cache should be cleared."""
        hidden_states, timestep, enc = _make_inputs()
        with torch.no_grad():
            self.transformer(hidden_states, timestep, enc)

        self.assertIsNotNone(self._get_block_state(0).cached_key)

        # Reset all stateful hooks
        self.transformer._diffusers_hook.reset_stateful_hooks()

        # After reset the context is gone; set it again to read state
        block_state = self._get_block_state(0)
        self.assertIsNone(block_state.cached_key)
        self.assertIsNone(block_state.cached_value)


class TestCrossCacheAttention(unittest.TestCase):
    def setUp(self):
        self.transformer = _make_transformer()
        self.transformer.eval()
        config = RollingKVCacheConfig(window_size=-1, cache_cross_attention=True)
        apply_rolling_kv_cache(self.transformer, config)

    def test_cross_attn_hooks_attached(self):
        from diffusers.hooks.rolling_kv_cache import _ROLLING_KV_CACHE_CROSS_ATTN_HOOK
        from diffusers.models.transformers.transformer_wan import WanTransformerBlock

        blocks = [m for m in self.transformer.modules()
                  if isinstance(m, WanTransformerBlock)]
        for block in blocks:
            self.assertTrue(hasattr(block.attn2, "_diffusers_hook"))
            hook = block.attn2._diffusers_hook.get_hook(_ROLLING_KV_CACHE_CROSS_ATTN_HOOK)
            self.assertIsNotNone(hook)

    def test_cross_attn_cache_populated_after_first_pass(self):
        from diffusers.hooks.rolling_kv_cache import _ROLLING_KV_CACHE_CROSS_ATTN_HOOK
        blocks = [m for m in self.transformer.modules()
                  if m.__class__.__name__ == "WanTransformerBlock"]
        hook = blocks[0].attn2._diffusers_hook.get_hook(_ROLLING_KV_CACHE_CROSS_ATTN_HOOK)
        if hook.block_state_manager._current_context is None:
            hook.block_state_manager.set_context("test")
        block_state = hook.block_state_manager.get_state()

        hidden_states, timestep, enc = _make_inputs()
        with torch.no_grad():
            self.transformer(hidden_states, timestep, enc)

        self.assertIsNotNone(block_state.cached_key)
        self.assertIsNotNone(block_state.cached_value)

    def test_cross_attn_cache_not_recomputed_on_second_pass(self):
        """Cross-attn K/V should be cached from first pass and reused (same object)."""
        from diffusers.hooks.rolling_kv_cache import _ROLLING_KV_CACHE_CROSS_ATTN_HOOK
        blocks = [m for m in self.transformer.modules()
                  if m.__class__.__name__ == "WanTransformerBlock"]
        hook = blocks[0].attn2._diffusers_hook.get_hook(_ROLLING_KV_CACHE_CROSS_ATTN_HOOK)
        if hook.block_state_manager._current_context is None:
            hook.block_state_manager.set_context("test")
        block_state = hook.block_state_manager.get_state()

        hidden_states, timestep, enc = _make_inputs()
        with torch.no_grad():
            self.transformer(hidden_states, timestep, enc)
        cached_key_after_first = block_state.cached_key

        with torch.no_grad():
            self.transformer(hidden_states, timestep, enc)
        cached_key_after_second = block_state.cached_key

        # Same tensor object — cache was not recomputed
        self.assertIs(cached_key_after_first, cached_key_after_second,
                      "Cross-attn K should be the same cached tensor on second pass")


if __name__ == "__main__":
    unittest.main()
