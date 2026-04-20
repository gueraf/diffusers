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

from diffusers.hooks import (
    RollingKVCacheConfig,
    apply_rolling_kv_cache,
    get_rolling_kv_cache_state,
    prefill_rolling_kv_cache,
)
from diffusers.hooks.rolling_kv_cache import (
    RollingKVCacheBlockState,
    RollingKVCacheCrossAttnBlockState,
    RollingKVCacheState,
    _ROLLING_KV_CACHE_CROSS_ATTN_HOOK,
    _ROLLING_KV_CACHE_SELF_ATTN_HOOK,
    _apply_wan_rotary_emb,
)
from diffusers.models.transformers.transformer_wan import WanTransformerBlock


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

_BATCH = 1
_T, _H, _W = 1, 4, 4
_IN_CHANNELS = 16
_TEXT_SEQ_LEN = 10
_TEXT_DIM = 32
_TOKENS_PER_CHUNK = (_T // WAN_TINY_CONFIG["patch_size"][0]) * (_H // 2) * (_W // 2)


def _make_transformer(config=None):
    from diffusers import WanTransformer3DModel

    torch.manual_seed(0)
    return WanTransformer3DModel.from_config(config or WAN_TINY_CONFIG)


def _make_inputs(batch=_BATCH, t=_T, h=_H, w=_W, text_seq=_TEXT_SEQ_LEN, text_dim=_TEXT_DIM):
    hidden_states = torch.randn(batch, _IN_CHANNELS, t, h, w)
    timestep = torch.zeros(batch, dtype=torch.long)
    encoder_hidden_states = torch.randn(batch, text_seq, text_dim)
    return hidden_states, timestep, encoder_hidden_states


def _get_blocks(transformer):
    return [module for module in transformer.modules() if isinstance(module, WanTransformerBlock)]


def _get_self_hook(transformer, block_idx=0):
    hook = _get_blocks(transformer)[block_idx].attn1._diffusers_hook.get_hook(_ROLLING_KV_CACHE_SELF_ATTN_HOOK)
    if hook.block_state_manager._current_context is None:
        hook.block_state_manager.set_context("inference")
    return hook


def _get_cross_hook(transformer, block_idx=0):
    hook = _get_blocks(transformer)[block_idx].attn2._diffusers_hook.get_hook(_ROLLING_KV_CACHE_CROSS_ATTN_HOOK)
    if hook.block_state_manager._current_context is None:
        hook.block_state_manager.set_context("inference")
    return hook


def _get_block_state(transformer, block_idx=0):
    return _get_self_hook(transformer, block_idx).block_state_manager.get_state()


def _run_chunk(transformer, hidden_states, timestep, encoder_hidden_states, frame_offset=0):
    with torch.no_grad():
        return transformer(
            hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            frame_offset=frame_offset,
        ).sample


class TestStateClasses(unittest.TestCase):
    def test_rolling_kv_cache_state_defaults(self):
        state = RollingKVCacheState()
        self.assertTrue(state.should_update_cache)
        self.assertEqual(state.write_mode, "append")
        self.assertIsNone(state.absolute_token_offset)

    def test_rolling_kv_cache_state_reset(self):
        state = RollingKVCacheState()
        state.should_update_cache = False
        state.configure_cache_write(write_mode="overwrite", absolute_token_offset=8)
        state.reset()
        self.assertTrue(state.should_update_cache)
        self.assertEqual(state.write_mode, "append")
        self.assertIsNone(state.absolute_token_offset)

    def test_block_state_defaults(self):
        state = RollingKVCacheBlockState()
        self.assertIsNone(state.cached_key)
        self.assertIsNone(state.cached_value)
        self.assertEqual(state.cache_start_token_offset, 0)

    def test_block_state_reset(self):
        state = RollingKVCacheBlockState()
        state.cached_key = torch.randn(1, 4, 2, 16)
        state.cached_value = torch.randn(1, 4, 2, 16)
        state.cache_start_token_offset = 16
        state.reset()
        self.assertIsNone(state.cached_key)
        self.assertIsNone(state.cached_value)
        self.assertEqual(state.cache_start_token_offset, 0)

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
        x = torch.randn(1, 4, 2, 16)
        freqs_cos = torch.ones(1, 4, 1, 16)
        freqs_sin = torch.zeros(1, 4, 1, 16)
        out = _apply_wan_rotary_emb(x, freqs_cos, freqs_sin)
        self.assertEqual(out.shape, x.shape)

    def test_zero_sin_identity(self):
        x = torch.randn(1, 4, 2, 16)
        freqs_cos = torch.ones(1, 4, 1, 16)
        freqs_sin = torch.zeros(1, 4, 1, 16)
        out = _apply_wan_rotary_emb(x, freqs_cos, freqs_sin)
        self.assertTrue(torch.allclose(out, x, atol=1e-6))

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
        apply_rolling_kv_cache(self.transformer, RollingKVCacheConfig(window_size=-1, cache_cross_attention=False))

    def test_hooks_attached_to_all_blocks(self):
        blocks = _get_blocks(self.transformer)
        self.assertEqual(len(blocks), WAN_TINY_CONFIG["num_layers"])

        for block in blocks:
            self.assertTrue(hasattr(block.attn1, "_diffusers_hook"))
            self.assertIsNotNone(block.attn1._diffusers_hook.get_hook(_ROLLING_KV_CACHE_SELF_ATTN_HOOK))

    def test_cache_populated_after_first_pass(self):
        hidden_states, timestep, encoder_hidden_states = _make_inputs()
        _run_chunk(self.transformer, hidden_states, timestep, encoder_hidden_states, frame_offset=0)

        block_state = _get_block_state(self.transformer)
        self.assertIsNotNone(block_state.cached_key)
        self.assertIsNotNone(block_state.cached_value)
        self.assertEqual(block_state.cached_key.shape[1], _TOKENS_PER_CHUNK)
        self.assertEqual(block_state.cache_start_token_offset, 0)

    def test_cache_grows_after_second_chunk(self):
        hs1, ts1, enc1 = _make_inputs()
        hs2, ts2, enc2 = _make_inputs()
        _run_chunk(self.transformer, hs1, ts1, enc1, frame_offset=0)
        _run_chunk(self.transformer, hs2, ts2, enc2, frame_offset=1)

        block_state = _get_block_state(self.transformer)
        self.assertEqual(block_state.cached_key.shape[1], _TOKENS_PER_CHUNK * 2)
        self.assertEqual(block_state.cache_start_token_offset, 0)

    def test_window_size_limits_cache_and_tracks_start_offset(self):
        transformer = _make_transformer()
        transformer.eval()
        apply_rolling_kv_cache(transformer, RollingKVCacheConfig(window_size=_TOKENS_PER_CHUNK))

        hs1, ts1, enc1 = _make_inputs()
        hs2, ts2, enc2 = _make_inputs()
        _run_chunk(transformer, hs1, ts1, enc1, frame_offset=0)
        _run_chunk(transformer, hs2, ts2, enc2, frame_offset=1)

        block_state = _get_block_state(transformer)
        self.assertEqual(block_state.cached_key.shape[1], _TOKENS_PER_CHUNK)
        self.assertEqual(block_state.cache_start_token_offset, _TOKENS_PER_CHUNK)

    def test_no_cache_update_when_flag_false(self):
        cache_state = get_rolling_kv_cache_state(self.transformer)
        hs1, ts1, enc1 = _make_inputs()
        hs2, ts2, enc2 = _make_inputs()
        _run_chunk(self.transformer, hs1, ts1, enc1, frame_offset=0)
        before = _get_block_state(self.transformer).cached_key.clone()

        cache_state.should_update_cache = False
        _run_chunk(self.transformer, hs2, ts2, enc2, frame_offset=1)
        after = _get_block_state(self.transformer).cached_key

        torch.testing.assert_close(before, after)

    def test_second_pass_output_differs_with_cache(self):
        hs1, ts1, enc1 = _make_inputs()
        _run_chunk(self.transformer, hs1, ts1, enc1, frame_offset=0)

        torch.manual_seed(42)
        hs2, ts2, enc2 = _make_inputs()
        out_with_cache = _run_chunk(self.transformer, hs2, ts2, enc2, frame_offset=1)

        fresh = _make_transformer()
        fresh.load_state_dict(self.transformer.state_dict())
        fresh.eval()
        apply_rolling_kv_cache(fresh, RollingKVCacheConfig(window_size=-1, cache_cross_attention=False))
        out_no_cache = _run_chunk(fresh, hs2, ts2, enc2, frame_offset=1)

        self.assertFalse(torch.allclose(out_with_cache, out_no_cache))

    def test_transformer_accepts_per_frame_timesteps(self):
        transformer = _make_transformer()
        transformer.eval()
        hidden_states, _, encoder_hidden_states = _make_inputs(t=2)
        timestep = torch.full((hidden_states.shape[0], hidden_states.shape[2]), 937.5, dtype=torch.float32)

        with torch.no_grad():
            output = transformer(
                hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                frame_offset=0,
            ).sample

        self.assertEqual(output.shape, hidden_states.shape)

    def test_overwrite_truncates_suffix_and_matches_rebuilt_prefix(self):
        hs_a, ts_a, enc = _make_inputs()
        hs_b, ts_b, _ = _make_inputs()
        hs_c, ts_c, _ = _make_inputs()
        hs_d, _, _ = _make_inputs()

        _run_chunk(self.transformer, hs_a, ts_a, enc, frame_offset=0)
        _run_chunk(self.transformer, hs_b, ts_b, enc, frame_offset=1)
        _run_chunk(self.transformer, hs_c, ts_c, enc, frame_offset=2)

        prefill_rolling_kv_cache(
            self.transformer,
            hs_d,
            enc,
            frame_offset=1,
            write_mode="overwrite",
        )

        overwritten_state = _get_block_state(self.transformer)
        self.assertEqual(overwritten_state.cached_key.shape[1], _TOKENS_PER_CHUNK * 2)
        self.assertEqual(overwritten_state.cache_start_token_offset, 0)

        fresh = _make_transformer()
        fresh.load_state_dict(self.transformer.state_dict())
        fresh.eval()
        apply_rolling_kv_cache(fresh, RollingKVCacheConfig(window_size=-1, cache_cross_attention=False))
        _run_chunk(fresh, hs_a, ts_a, enc, frame_offset=0)
        _run_chunk(fresh, hs_d, ts_a, enc, frame_offset=1)

        expected_state = _get_block_state(fresh)
        torch.testing.assert_close(overwritten_state.cached_key, expected_state.cached_key)
        torch.testing.assert_close(overwritten_state.cached_value, expected_state.cached_value)

    def test_prefill_helper_matches_clean_writes(self):
        hs_a, ts, enc = _make_inputs()
        hs_b, _, _ = _make_inputs()

        direct = _make_transformer()
        direct.load_state_dict(self.transformer.state_dict())
        direct.eval()
        apply_rolling_kv_cache(direct, RollingKVCacheConfig(window_size=-1, cache_cross_attention=False))
        _run_chunk(direct, hs_a, ts, enc, frame_offset=0)
        _run_chunk(direct, hs_b, ts, enc, frame_offset=1)

        helper = _make_transformer()
        helper.load_state_dict(self.transformer.state_dict())
        helper.eval()
        apply_rolling_kv_cache(helper, RollingKVCacheConfig(window_size=-1, cache_cross_attention=False))
        prefill_rolling_kv_cache(helper, [hs_a, hs_b], enc, frame_offset=0)

        direct_state = _get_block_state(direct)
        helper_state = _get_block_state(helper)

        self.assertEqual(helper_state.cache_start_token_offset, direct_state.cache_start_token_offset)
        torch.testing.assert_close(helper_state.cached_key, direct_state.cached_key)
        torch.testing.assert_close(helper_state.cached_value, direct_state.cached_value)

        helper_cache_state = get_rolling_kv_cache_state(helper)
        self.assertEqual(helper_cache_state.write_mode, "append")
        self.assertIsNone(helper_cache_state.absolute_token_offset)

    def test_prefill_helper_defaults_to_per_frame_zero_timestep(self):
        hs_a, _, enc = _make_inputs(t=2)
        explicit = _make_transformer()
        explicit.load_state_dict(self.transformer.state_dict())
        explicit.eval()
        apply_rolling_kv_cache(explicit, RollingKVCacheConfig(window_size=-1, cache_cross_attention=False))

        implicit = _make_transformer()
        implicit.load_state_dict(self.transformer.state_dict())
        implicit.eval()
        apply_rolling_kv_cache(implicit, RollingKVCacheConfig(window_size=-1, cache_cross_attention=False))

        timestep = torch.zeros((hs_a.shape[0], hs_a.shape[2]), dtype=torch.long)
        prefill_rolling_kv_cache(explicit, hs_a, enc, frame_offset=0, timestep=timestep)
        prefill_rolling_kv_cache(implicit, hs_a, enc, frame_offset=0)

        explicit_state = _get_block_state(explicit)
        implicit_state = _get_block_state(implicit)

        self.assertEqual(implicit_state.cache_start_token_offset, explicit_state.cache_start_token_offset)
        torch.testing.assert_close(implicit_state.cached_key, explicit_state.cached_key)
        torch.testing.assert_close(implicit_state.cached_value, explicit_state.cached_value)

    def test_reset_stateful_hooks_clears_cache(self):
        hidden_states, timestep, encoder_hidden_states = _make_inputs()
        _run_chunk(self.transformer, hidden_states, timestep, encoder_hidden_states, frame_offset=0)
        self.assertIsNotNone(_get_block_state(self.transformer).cached_key)

        self.transformer._diffusers_hook.reset_stateful_hooks()
        block_state = _get_block_state(self.transformer)
        self.assertIsNone(block_state.cached_key)
        self.assertIsNone(block_state.cached_value)
        self.assertEqual(block_state.cache_start_token_offset, 0)


class TestCrossAttentionCache(unittest.TestCase):
    def setUp(self):
        self.transformer = _make_transformer()
        self.transformer.eval()
        apply_rolling_kv_cache(self.transformer, RollingKVCacheConfig(window_size=-1, cache_cross_attention=True))

    def test_cross_attn_hooks_attached(self):
        for block in _get_blocks(self.transformer):
            self.assertTrue(hasattr(block.attn2, "_diffusers_hook"))
            self.assertIsNotNone(block.attn2._diffusers_hook.get_hook(_ROLLING_KV_CACHE_CROSS_ATTN_HOOK))

    def test_cross_attn_cache_populated_after_first_pass(self):
        hidden_states, timestep, encoder_hidden_states = _make_inputs()
        _run_chunk(self.transformer, hidden_states, timestep, encoder_hidden_states, frame_offset=0)

        block_state = _get_cross_hook(self.transformer).block_state_manager.get_state()
        self.assertIsNotNone(block_state.cached_key)
        self.assertIsNotNone(block_state.cached_value)

    def test_cross_attn_cache_reused_on_second_pass(self):
        hidden_states, timestep, encoder_hidden_states = _make_inputs()
        _run_chunk(self.transformer, hidden_states, timestep, encoder_hidden_states, frame_offset=0)

        block_state = _get_cross_hook(self.transformer).block_state_manager.get_state()
        first_cached_key = block_state.cached_key
        _run_chunk(self.transformer, hidden_states, timestep, encoder_hidden_states, frame_offset=1)
        self.assertIs(first_cached_key, block_state.cached_key)


if __name__ == "__main__":
    unittest.main()
