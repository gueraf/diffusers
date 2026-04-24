# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..models.attention_dispatch import dispatch_attention_fn
from .hooks import BaseState, HookRegistry, ModelHook, StateManager


_ROLLING_KV_CACHE_HOOK = "rolling_kv_cache"
_ROLLING_KV_WRITE_MODES = {"append", "overwrite"}


@dataclass
class RollingKVCacheConfig:
    r"""Configuration for rolling self-attention KV caching during autoregressive inference.

    Args:
        window_size (`int`, defaults to `-1`):
            Maximum number of cached self-attention tokens to keep. Set to `-1` to keep the full prefix.
    """

    window_size: int = -1


class RollingKVCacheState(BaseState):
    r"""Shared state controlling how the rolling KV cache is updated."""

    def __init__(self):
        self.should_update_cache = True
        self.write_mode = "append"
        self.absolute_token_offset: int | None = None

    def configure_cache_write(self, write_mode: str = "append", absolute_token_offset: int | None = None) -> None:
        if write_mode not in _ROLLING_KV_WRITE_MODES:
            raise ValueError(
                f"`write_mode` must be one of {sorted(_ROLLING_KV_WRITE_MODES)}, but received {write_mode!r}."
            )
        if absolute_token_offset is not None and absolute_token_offset < 0:
            raise ValueError("`absolute_token_offset` must be >= 0.")
        if write_mode == "overwrite" and absolute_token_offset is None:
            raise ValueError("`absolute_token_offset` must be provided when `write_mode='overwrite'`.")

        self.write_mode = write_mode
        self.absolute_token_offset = absolute_token_offset

    def clear_cache_write(self) -> None:
        self.write_mode = "append"
        self.absolute_token_offset = None

    def reset(self):
        self.should_update_cache = True
        self.clear_cache_write()


class RollingKVCacheBlockState(BaseState):
    r"""Per-attention-block self-attention cache state."""

    def __init__(self):
        self.cached_key: torch.Tensor | None = None
        self.cached_value: torch.Tensor | None = None
        self.cache_start_token_offset = 0

    def reset(self):
        self.cached_key = None
        self.cached_value = None
        self.cache_start_token_offset = 0


def _get_qkv_projections(
    attn: torch.nn.Module,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if getattr(attn, "fused_projections", False):
        query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
    else:
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

    return query, key, value


def _apply_rotary_emb(hidden_states: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):
    hidden_states_complex = torch.view_as_complex(
        hidden_states.to(torch.float64).reshape(*hidden_states.shape[:-1], -1, 2)
    )
    freqs_complex = torch.complex(
        freqs_cos[..., 0::2].to(torch.float64), freqs_sin[..., 0::2].to(torch.float64)
    )
    out = torch.view_as_real(hidden_states_complex * freqs_complex).flatten(-2)
    return out.type_as(hidden_states)


def _get_attention_backend(attn: torch.nn.Module):
    processor = getattr(attn, "processor", None)
    return getattr(processor, "_attention_backend", None)


def _match_cached_batch_size(
    cached_key: torch.Tensor | None,
    cached_value: torch.Tensor | None,
    batch_size: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if cached_key is None:
        return None, None

    if cached_key.shape[0] == batch_size:
        return cached_key, cached_value

    if cached_key.shape[0] == 1:
        expand_shape = (batch_size, -1, -1, -1)
        return cached_key.expand(*expand_shape), cached_value.expand(*expand_shape)

    raise ValueError(
        "Rolling KV cache batch size mismatch. Use cache contexts for conditional/unconditional passes or reset the "
        f"cache before changing batch size (cached batch={cached_key.shape[0]}, current batch={batch_size})."
    )


def _slice_cache_prefix(
    block_state: RollingKVCacheBlockState,
    absolute_token_offset: int | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, int]:
    cached_key = block_state.cached_key
    cached_value = block_state.cached_value
    cache_start = block_state.cache_start_token_offset

    if cached_key is None:
        return None, None, absolute_token_offset if absolute_token_offset is not None else 0

    if absolute_token_offset is None:
        return cached_key, cached_value, cache_start

    cache_end = cache_start + cached_key.shape[1]
    if absolute_token_offset < cache_start:
        return None, None, absolute_token_offset
    if absolute_token_offset >= cache_end:
        if absolute_token_offset > cache_end:
            raise ValueError(
                "`absolute_token_offset` points beyond the retained cache prefix. Reset the cache or prefill the "
                "missing chunks before appending new ones."
            )
        return cached_key, cached_value, cache_start

    prefix_length = absolute_token_offset - cache_start
    return cached_key[:, :prefix_length], cached_value[:, :prefix_length], cache_start


def _trim_cache_to_window(
    key: torch.Tensor,
    value: torch.Tensor,
    cache_start_token_offset: int,
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if window_size > 0 and key.shape[1] > window_size:
        trim = key.shape[1] - window_size
        key = key[:, trim:]
        value = value[:, trim:]
        cache_start_token_offset += trim

    return key.detach(), value.detach(), cache_start_token_offset


def _is_self_attention_module(module: torch.nn.Module) -> bool:
    if getattr(module, "is_cross_attention", False):
        return False

    required_attrs = ("to_q", "to_k", "to_v", "to_out", "heads", "norm_q", "norm_k")
    return all(hasattr(module, attr) for attr in required_attrs)


class RollingKVCacheHook(ModelHook):
    _is_stateful = True

    def __init__(self, config: RollingKVCacheConfig, state_manager: StateManager, block_state_manager: StateManager):
        super().__init__()
        self.config = config
        self.state_manager = state_manager
        self.block_state_manager = block_state_manager

    def new_forward(
        self,
        module: torch.nn.Module,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if self.state_manager._current_context is None:
            self.state_manager.set_context("inference")
        if self.block_state_manager._current_context is None:
            self.block_state_manager.set_context("inference")
        if encoder_hidden_states is not None:
            raise ValueError("Rolling KV cache only supports self-attention modules.")

        shared_state: RollingKVCacheState = self.state_manager.get_state()
        block_state: RollingKVCacheBlockState = self.block_state_manager.get_state()

        query, key, value = _get_qkv_projections(module, hidden_states)
        query = module.norm_q(query)
        key = module.norm_k(key)

        query = query.unflatten(2, (module.heads, -1))
        key = key.unflatten(2, (module.heads, -1))
        value = value.unflatten(2, (module.heads, -1))

        if rotary_emb is not None:
            query = _apply_rotary_emb(query, *rotary_emb)
            key = _apply_rotary_emb(key, *rotary_emb)

        cache_token_offset = shared_state.absolute_token_offset
        if (
            shared_state.write_mode == "append"
            and block_state.cached_key is not None
            and cache_token_offset is not None
        ):
            expected_token_offset = block_state.cache_start_token_offset + block_state.cached_key.shape[1]
            if cache_token_offset != expected_token_offset:
                raise ValueError(
                    "Append writes must continue from the current cache end. "
                    f"Expected {expected_token_offset}, received {cache_token_offset}."
                )

        if shared_state.write_mode == "overwrite":
            cached_key, cached_value, prefix_start = _slice_cache_prefix(block_state, cache_token_offset)
        else:
            cached_key, cached_value, prefix_start = (
                block_state.cached_key,
                block_state.cached_value,
                block_state.cache_start_token_offset,
            )

        cached_key, cached_value = _match_cached_batch_size(cached_key, cached_value, key.shape[0])

        if cached_key is not None:
            full_key = torch.cat([cached_key, key], dim=1)
            full_value = torch.cat([cached_value, value], dim=1)
        else:
            full_key = key
            full_value = value
            prefix_start = cache_token_offset if cache_token_offset is not None else prefix_start

        if shared_state.should_update_cache:
            (
                block_state.cached_key,
                block_state.cached_value,
                block_state.cache_start_token_offset,
            ) = _trim_cache_to_window(
                full_key,
                full_value,
                prefix_start,
                self.config.window_size,
            )

        hidden_states = dispatch_attention_fn(
            query,
            full_key,
            full_value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=_get_attention_backend(module),
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)
        hidden_states = module.to_out[0](hidden_states)
        hidden_states = module.to_out[1](hidden_states)

        return hidden_states

    def reset_state(self, module: torch.nn.Module):
        self.state_manager.reset()
        self.block_state_manager.reset()
        return module


def apply_rolling_kv_cache(module: torch.nn.Module, config: RollingKVCacheConfig | None = None) -> None:
    r"""Apply rolling KV cache hooks to compatible self-attention modules."""
    if config is None:
        config = RollingKVCacheConfig()

    state_manager = StateManager(RollingKVCacheState)
    HookRegistry.check_if_exists_or_initialize(module)

    for _, submodule in module.named_modules():
        if not _is_self_attention_module(submodule):
            continue

        block_state_manager = StateManager(RollingKVCacheBlockState)
        hook = RollingKVCacheHook(config, state_manager, block_state_manager)
        registry = HookRegistry.check_if_exists_or_initialize(submodule)
        registry.register_hook(hook, _ROLLING_KV_CACHE_HOOK)


def get_rolling_kv_cache_state(module: torch.nn.Module) -> RollingKVCacheState | None:
    r"""Return the shared rolling KV cache state for a hooked module."""
    for _, submodule in module.named_modules():
        if not hasattr(submodule, "_diffusers_hook"):
            continue

        hook = submodule._diffusers_hook.get_hook(_ROLLING_KV_CACHE_HOOK)
        if hook is not None:
            if hook.state_manager._current_context is None:
                hook.state_manager.set_context("inference")
            return hook.state_manager.get_state()

    return None
