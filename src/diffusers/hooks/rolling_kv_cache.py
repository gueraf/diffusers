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
 
from dataclasses import dataclass, field
 
import torch
 
from ..models.attention_dispatch import dispatch_attention_fn
from .hooks import BaseState, HookRegistry, ModelHook, StateManager
 
 
_ROLLING_KV_CACHE_SELF_ATTN_HOOK = "rolling_kv_cache_self_attn"
_ROLLING_KV_CACHE_CROSS_ATTN_HOOK = "rolling_kv_cache_cross_attn"
 
 
@dataclass
class RollingKVCacheConfig:
    """Configuration for rolling KV cache used in autoregressive diffusion inference.
 
    Implements a rolling (sliding-window) KV cache for self-attention layers in transformer models.
    During autoregressive video generation, previous frame chunks' key and value projections are
    cached and prepended to the current chunk's keys and values, enabling the model to attend to
    past context.
 
    Args:
        window_size (`int`, defaults to `-1`):
            Maximum number of cached tokens to keep in the rolling window. When the cache exceeds
            this size, the oldest tokens are dropped. Set to `-1` for unlimited cache size (keep
            all previous tokens).
        cache_cross_attention (`bool`, defaults to `False`):
            Whether to also cache cross-attention key/value projections. Since cross-attention
            depends only on the conditioning (e.g. text embeddings), which stays constant across
            chunks, caching avoids redundant computation.
    """
 
    window_size: int = -1
    cache_cross_attention: bool = False
 
 
class RollingKVCacheState(BaseState):
    """Shared state controlling cache behavior across all blocks.
 
    Attributes:
        should_update_cache: When ``True``, the cache is updated with the current chunk's K/V
            after attention. Set to ``False`` during denoising iterations to avoid polluting
            the cache with noisy intermediate K/V. Set to ``True`` for the final clean pass.
    """
 
    def __init__(self):
        self.should_update_cache: bool = True
 
    def reset(self):
        self.should_update_cache = True
 
 
class RollingKVCacheBlockState(BaseState):
    """Per-block state holding cached self-attention key/value tensors.
 
    The cached tensors have shape ``(batch_size, cached_seq_len, num_heads, head_dim)`` and
    store post-norm, post-rotary-embedding keys and values.
    """
 
    def __init__(self):
        self.cached_key: torch.Tensor | None = None
        self.cached_value: torch.Tensor | None = None
 
    def reset(self):
        self.cached_key = None
        self.cached_value = None
 
 
class RollingKVCacheCrossAttnBlockState(BaseState):
    """Per-block state holding cached cross-attention key/value tensors."""
 
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
    """Apply rotary position embeddings in the Wan style."""
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(hidden_states)
 
 
class RollingKVCacheSelfAttnHook(ModelHook):
    """Hook applied to self-attention modules (``attn1``) to implement rolling KV cache.
 
    On each forward pass:
    1. Computes Q, K, V for the current chunk
    2. Prepends cached K, V from previous chunks
    3. Runs attention with current Q against full (cached + current) K, V
    4. Optionally updates the cache with the new K, V
    """
 
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
 
        # Compute Q, K, V for current chunk
        query, key, value = _get_qkv_projections(module, hidden_states, encoder_hidden_states)
        query = module.norm_q(query)
        key = module.norm_k(key)
 
        query = query.unflatten(2, (module.heads, -1))
        key = key.unflatten(2, (module.heads, -1))
        value = value.unflatten(2, (module.heads, -1))
 
        # Apply rotary embeddings
        if rotary_emb is not None:
            query = _apply_wan_rotary_emb(query, *rotary_emb)
            key = _apply_wan_rotary_emb(key, *rotary_emb)
 
        # Prepend cached K, V from previous chunks
        if block_state.cached_key is not None:
            full_key = torch.cat([block_state.cached_key, key], dim=1)
            full_value = torch.cat([block_state.cached_value, value], dim=1)
        else:
            full_key = key
            full_value = value
 
        # Update cache if in recording mode
        if shared_state.should_update_cache:
            window = self.config.window_size
            if window > 0 and full_key.shape[1] > window:
                block_state.cached_key = full_key[:, -window:].detach()
                block_state.cached_value = full_value[:, -window:].detach()
            else:
                block_state.cached_key = full_key.detach()
                block_state.cached_value = full_value.detach()
 
        # Run attention
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
    """Hook applied to cross-attention modules (``attn2``) to cache text/image K, V.
 
    Since cross-attention keys and values depend only on the conditioning signal (text/image
    embeddings), which is constant across autoregressive chunks, we compute them once on the
    first forward pass and reuse the cached values for all subsequent passes.
    """
 
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
 
        # Handle I2V image context splitting
        encoder_hidden_states_img = None
        if module.add_k_proj is not None and encoder_hidden_states is not None:
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]
 
        # Compute Q (always fresh, depends on current hidden_states)
        query = module.to_q(hidden_states)
        query = module.norm_q(query)
        query = query.unflatten(2, (module.heads, -1))
 
        # Compute or retrieve cached K, V for text cross-attention
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
 
        # Handle I2V image attention
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
 
        # Run text cross-attention
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
    """Apply rolling KV cache hooks to a transformer model for autoregressive inference.
 
    This attaches hooks to each transformer block's self-attention (and optionally cross-attention)
    modules that maintain a rolling cache of key/value projections from previous chunks. During
    autoregressive video generation, this allows the model to attend to past frame context without
    recomputing all previous tokens.
 
    Args:
        module (`torch.nn.Module`):
            The transformer model to apply rolling KV cache to (e.g., ``WanTransformer3DModel``).
        config (`RollingKVCacheConfig`, *optional*):
            Configuration for the rolling KV cache. If ``None``, default config is used.
 
    Example:
        ```python
        from diffusers.hooks import RollingKVCacheConfig, apply_rolling_kv_cache
 
        # Apply rolling KV cache with a sliding window of 4800 tokens
        config = RollingKVCacheConfig(window_size=4800)
        apply_rolling_kv_cache(transformer, config)
 
        # The cache state can be accessed to control update behavior:
        # transformer._diffusers_hook.hooks["rolling_kv_cache_self_attn"] ...
        ```
    """
    from ..models.transformers.transformer_wan import WanTransformerBlock
 
    if config is None:
        config = RollingKVCacheConfig()
 
    state_manager = StateManager(RollingKVCacheState)
 
    HookRegistry.check_if_exists_or_initialize(module)
 
    for name, submodule in module.named_modules():
        if isinstance(submodule, WanTransformerBlock):
            # Hook on self-attention (attn1)
            block_state_manager = StateManager(RollingKVCacheBlockState)
            self_attn_hook = RollingKVCacheSelfAttnHook(config, state_manager, block_state_manager)
            attn1_registry = HookRegistry.check_if_exists_or_initialize(submodule.attn1)
            attn1_registry.register_hook(self_attn_hook, _ROLLING_KV_CACHE_SELF_ATTN_HOOK)
 
            # Optionally hook on cross-attention (attn2)
            if config.cache_cross_attention:
                cross_block_state_manager = StateManager(RollingKVCacheCrossAttnBlockState)
                cross_attn_hook = RollingKVCacheCrossAttnHook(config, cross_block_state_manager)
                attn2_registry = HookRegistry.check_if_exists_or_initialize(submodule.attn2)
                attn2_registry.register_hook(cross_attn_hook, _ROLLING_KV_CACHE_CROSS_ATTN_HOOK)
 
 
def get_rolling_kv_cache_state(module: torch.nn.Module) -> RollingKVCacheState | None:
    """Retrieve the shared rolling KV cache state from a model with applied hooks.
 
    This is a convenience function to access the shared state that controls cache update
    behavior (``should_update_cache``).
 
    Args:
        module (`torch.nn.Module`):
            The transformer model with rolling KV cache hooks applied.
 
    Returns:
        `RollingKVCacheState` or `None`: The shared state, or ``None`` if no hook is found.
    """
    from ..models.transformers.transformer_wan import WanTransformerBlock
 
    for _, submodule in module.named_modules():
        if isinstance(submodule, WanTransformerBlock):
            if hasattr(submodule.attn1, "_diffusers_hook"):
                hook = submodule.attn1._diffusers_hook.get_hook(_ROLLING_KV_CACHE_SELF_ATTN_HOOK)
                if hook is not None:
                    if hook.state_manager._current_context is None:
                        hook.state_manager.set_context("inference")
                    return hook.state_manager.get_state()
    return None

