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

from contextlib import nullcontext
from dataclasses import dataclass

import torch

from ..models.attention_dispatch import dispatch_attention_fn
from .hooks import BaseState, HookRegistry, ModelHook, StateManager


_ROLLING_KV_CACHE_SELF_ATTN_HOOK = "rolling_kv_cache_self_attn"
_ROLLING_KV_CACHE_CROSS_ATTN_HOOK = "rolling_kv_cache_cross_attn"
_ROLLING_KV_WRITE_MODES = {"append", "overwrite"}


@dataclass
class RollingKVCacheConfig:
    r"""Configuration for rolling self-attention KV caching during autoregressive inference.

    Args:
        window_size (`int`, defaults to `-1`):
            Maximum number of cached self-attention tokens to keep. Set to `-1` to keep the full prefix.
        cache_cross_attention (`bool`, defaults to `False`):
            Whether to cache cross-attention K/V as well. This is useful when the conditioning signal stays fixed
            across chunks.
    """

    window_size: int = -1
    cache_cross_attention: bool = False


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
    r"""Per-block self-attention cache state."""

    def __init__(self):
        self.cached_key: torch.Tensor | None = None
        self.cached_value: torch.Tensor | None = None
        self.cache_start_token_offset = 0

    def reset(self):
        self.cached_key = None
        self.cached_value = None
        self.cache_start_token_offset = 0


class RollingKVCacheCrossAttnBlockState(BaseState):
    r"""Per-block cross-attention cache state."""

    def __init__(self):
        self.cached_key: torch.Tensor | None = None
        self.cached_value: torch.Tensor | None = None
        self.cached_key_img: torch.Tensor | None = None
        self.cached_value_img: torch.Tensor | None = None

    def reset(self):
        self.cached_key = None
        self.cached_value = None
        self.cached_key_img = None
        self.cached_value_img = None


def _apply_wan_rotary_emb(hidden_states: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):
    hidden_states_complex = torch.view_as_complex(hidden_states.to(torch.float64).reshape(*hidden_states.shape[:-1], -1, 2))
    freqs_complex = torch.complex(freqs_cos[..., 0::2].to(torch.float64), freqs_sin[..., 0::2].to(torch.float64))
    out = torch.view_as_real(hidden_states_complex * freqs_complex).flatten(-2)
    return out.type_as(hidden_states)


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


def _chunk_sequence(chunks: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
    if isinstance(chunks, torch.Tensor):
        if chunks.ndim != 5:
            raise ValueError(
                "`latents` must be a single 5D latent chunk `(batch, channels, frames, height, width)` or a list "
                "of such tensors."
            )
        return [chunks]

    if not isinstance(chunks, (list, tuple)) or len(chunks) == 0:
        raise ValueError("`latents` must be a tensor or a non-empty list/tuple of tensors.")

    for chunk in chunks:
        if not isinstance(chunk, torch.Tensor) or chunk.ndim != 5:
            raise ValueError("Every latent chunk must be a 5D torch.Tensor.")

    return list(chunks)


def _normalize_frame_offsets(
    transformer: torch.nn.Module,
    chunks: list[torch.Tensor],
    frame_offset: int | list[int] | tuple[int, ...],
) -> list[int]:
    if isinstance(frame_offset, int):
        patch_frames_per_chunk = []
        for chunk in chunks:
            patch_frames_per_chunk.append(chunk.shape[2] // transformer.config.patch_size[0])

        offsets = []
        current_offset = frame_offset
        for patch_frames in patch_frames_per_chunk:
            offsets.append(current_offset)
            current_offset += patch_frames

        return offsets

    if len(frame_offset) != len(chunks):
        raise ValueError("`frame_offset` must have the same length as `latents` when passing multiple chunks.")

    return list(frame_offset)


def _frame_to_token_offset(transformer: torch.nn.Module, latents: torch.Tensor, frame_offset: int) -> int:
    if frame_offset < 0:
        raise ValueError("`frame_offset` must be >= 0.")

    p_t, p_h, p_w = transformer.config.patch_size
    _, _, num_frames, height, width = latents.shape
    if num_frames % p_t != 0 or height % p_h != 0 or width % p_w != 0:
        raise ValueError("Latent chunk dimensions must be divisible by the transformer's patch size.")

    patches_per_frame = (height // p_h) * (width // p_w)
    return frame_offset * patches_per_frame


class RollingKVCacheSelfAttnHook(ModelHook):
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
        from ..models.transformers.transformer_wan import _get_qkv_projections

        if self.state_manager._current_context is None:
            self.state_manager.set_context("inference")
        if self.block_state_manager._current_context is None:
            self.block_state_manager.set_context("inference")

        shared_state: RollingKVCacheState = self.state_manager.get_state()
        block_state: RollingKVCacheBlockState = self.block_state_manager.get_state()

        query, key, value = _get_qkv_projections(module, hidden_states, encoder_hidden_states)
        query = module.norm_q(query)
        key = module.norm_k(key)

        query = query.unflatten(2, (module.heads, -1))
        key = key.unflatten(2, (module.heads, -1))
        value = value.unflatten(2, (module.heads, -1))

        if rotary_emb is not None:
            query = _apply_wan_rotary_emb(query, *rotary_emb)
            key = _apply_wan_rotary_emb(key, *rotary_emb)

        cache_token_offset = shared_state.absolute_token_offset
        if shared_state.write_mode == "append" and block_state.cached_key is not None and cache_token_offset is not None:
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
            block_state.cached_key, block_state.cached_value, block_state.cache_start_token_offset = _trim_cache_to_window(
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
            backend=module.processor._attention_backend if hasattr(module, "processor") else None,
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


class RollingKVCacheCrossAttnHook(ModelHook):
    _is_stateful = True

    def __init__(self, config: RollingKVCacheConfig, block_state_manager: StateManager):
        super().__init__()
        self.config = config
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
        from ..models.transformers.transformer_wan import _get_added_kv_projections, _get_qkv_projections

        if self.block_state_manager._current_context is None:
            self.block_state_manager.set_context("inference")

        block_state: RollingKVCacheCrossAttnBlockState = self.block_state_manager.get_state()

        encoder_hidden_states_img = None
        if module.add_k_proj is not None and encoder_hidden_states is not None:
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        query = module.to_q(hidden_states)
        query = module.norm_q(query)
        query = query.unflatten(2, (module.heads, -1))

        if block_state.cached_key is None:
            _, key, value = _get_qkv_projections(module, hidden_states, encoder_hidden_states)
            key = module.norm_k(key)
            key = key.unflatten(2, (module.heads, -1))
            value = value.unflatten(2, (module.heads, -1))
            block_state.cached_key = key.detach()
            block_state.cached_value = value.detach()
        else:
            key = block_state.cached_key
            value = block_state.cached_value

        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            if block_state.cached_key_img is None:
                key_img, value_img = _get_added_kv_projections(module, encoder_hidden_states_img)
                key_img = module.norm_added_k(key_img)
                key_img = key_img.unflatten(2, (module.heads, -1))
                value_img = value_img.unflatten(2, (module.heads, -1))
                block_state.cached_key_img = key_img.detach()
                block_state.cached_value_img = value_img.detach()
            else:
                key_img = block_state.cached_key_img
                value_img = block_state.cached_value_img

            hidden_states_img = dispatch_attention_fn(
                query,
                key_img,
                value_img,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                backend=module.processor._attention_backend if hasattr(module, "processor") else None,
                parallel_config=None,
            )
            hidden_states_img = hidden_states_img.flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=module.processor._attention_backend if hasattr(module, "processor") else None,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = module.to_out[0](hidden_states)
        hidden_states = module.to_out[1](hidden_states)
        return hidden_states

    def reset_state(self, module: torch.nn.Module):
        self.block_state_manager.reset()
        return module


def apply_rolling_kv_cache(module: torch.nn.Module, config: RollingKVCacheConfig | None = None) -> None:
    r"""Apply rolling KV cache hooks to a Wan transformer."""
    from ..models.transformers.transformer_wan import WanTransformerBlock

    if config is None:
        config = RollingKVCacheConfig()

    state_manager = StateManager(RollingKVCacheState)
    HookRegistry.check_if_exists_or_initialize(module)

    for _, submodule in module.named_modules():
        if isinstance(submodule, WanTransformerBlock):
            block_state_manager = StateManager(RollingKVCacheBlockState)
            self_attn_hook = RollingKVCacheSelfAttnHook(config, state_manager, block_state_manager)
            attn1_registry = HookRegistry.check_if_exists_or_initialize(submodule.attn1)
            attn1_registry.register_hook(self_attn_hook, _ROLLING_KV_CACHE_SELF_ATTN_HOOK)

            if config.cache_cross_attention:
                cross_block_state_manager = StateManager(RollingKVCacheCrossAttnBlockState)
                cross_attn_hook = RollingKVCacheCrossAttnHook(config, cross_block_state_manager)
                attn2_registry = HookRegistry.check_if_exists_or_initialize(submodule.attn2)
                attn2_registry.register_hook(cross_attn_hook, _ROLLING_KV_CACHE_CROSS_ATTN_HOOK)


def get_rolling_kv_cache_state(module: torch.nn.Module) -> RollingKVCacheState | None:
    r"""Return the shared rolling KV cache state for a hooked Wan transformer."""
    from ..models.transformers.transformer_wan import WanTransformerBlock

    for _, submodule in module.named_modules():
        if isinstance(submodule, WanTransformerBlock) and hasattr(submodule.attn1, "_diffusers_hook"):
            hook = submodule.attn1._diffusers_hook.get_hook(_ROLLING_KV_CACHE_SELF_ATTN_HOOK)
            if hook is not None:
                if hook.state_manager._current_context is None:
                    hook.state_manager.set_context("inference")
                return hook.state_manager.get_state()

    return None


@torch.no_grad()
def prefill_rolling_kv_cache(
    transformer: torch.nn.Module,
    latents: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    encoder_hidden_states: torch.Tensor,
    *,
    frame_offset: int | list[int] | tuple[int, ...] = 0,
    timestep: torch.Tensor | None = None,
    encoder_hidden_states_image: torch.Tensor | None = None,
    cache_context: str | None = None,
    write_mode: str = "append",
) -> None:
    r"""Populate or rewrite the rolling KV cache from clean latent chunks.

    This helper mirrors the clean `t=0` cache write used after denoising each chunk. When `write_mode="overwrite"`,
    cached entries at and after the provided `frame_offset` are discarded and replaced with the new clean latents. This
    makes it possible to inject ground-truth encoded frames at any autoregressive step and regenerate the suffix from
    that point onward.

    Args:
        transformer (`torch.nn.Module`):
            A `WanTransformer3DModel` with [`apply_rolling_kv_cache`] already attached.
        latents (`torch.Tensor` or `list[torch.Tensor]`):
            One or more clean latent chunks in model-normalized latent space, each shaped
            `(batch, channels, frames, height, width)`.
        encoder_hidden_states (`torch.Tensor`):
            Prompt embeddings to use for the cache write pass.
        frame_offset (`int` or `list[int]`, defaults to `0`):
            Absolute latent-frame offset(s) for the chunk(s). Each offset is converted into the matching absolute token
            position inside the cache.
        timestep (`torch.Tensor`, *optional*):
            Timestep tensor used for the clean pass. Defaults to `0` for each batch item.
        encoder_hidden_states_image (`torch.Tensor`, *optional*):
            Optional Wan I2V image embeddings.
        cache_context (`str`, *optional*):
            Cache context name to use for the write. Set this to `"cond"` or `"uncond"` when using CFG.
        write_mode (`str`, defaults to `"append"`):
            Either `"append"` to continue the current cache prefix or `"overwrite"` to truncate and rewrite from the
            supplied offset onward.
    """

    chunks = _chunk_sequence(latents)
    frame_offsets = _normalize_frame_offsets(transformer, chunks, frame_offset)
    context_manager = transformer.cache_context(cache_context) if cache_context is not None else nullcontext()

    with context_manager:
        cache_state = get_rolling_kv_cache_state(transformer)
        if cache_state is None:
            raise ValueError("Rolling KV cache hooks have not been applied to this transformer.")

        prev_should_update = cache_state.should_update_cache
        prev_write_mode = cache_state.write_mode
        prev_absolute_token_offset = cache_state.absolute_token_offset

        try:
            for chunk, chunk_frame_offset in zip(chunks, frame_offsets):
                token_offset = _frame_to_token_offset(transformer, chunk, chunk_frame_offset)

                cache_state.should_update_cache = True
                cache_state.configure_cache_write(write_mode=write_mode, absolute_token_offset=token_offset)

                if timestep is None:
                    patch_frames = chunk.shape[2] // transformer.config.patch_size[0]
                    chunk_timestep = torch.zeros((chunk.shape[0], patch_frames), device=chunk.device, dtype=torch.long)
                elif timestep.ndim == 0:
                    chunk_timestep = timestep.to(device=chunk.device).expand(chunk.shape[0])
                else:
                    chunk_timestep = timestep.to(device=chunk.device)

                transformer(
                    hidden_states=chunk,
                    timestep=chunk_timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_image=encoder_hidden_states_image,
                    frame_offset=chunk_frame_offset,
                    return_dict=False,
                )
        finally:
            cache_state.should_update_cache = prev_should_update
            cache_state.configure_cache_write(
                write_mode=prev_write_mode,
                absolute_token_offset=prev_absolute_token_offset,
            )
