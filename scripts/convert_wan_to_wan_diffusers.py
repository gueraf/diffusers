"""Convert Self-Forcing checkpoint to diffusers format.
 
Self-Forcing (https://github.com/guandeh17/Self-Forcing) is an autoregressive video
diffusion model built on top of Wan2.1-T2V-1.3B. This script converts a Self-Forcing
checkpoint (.pt) to a diffusers-compatible WanTransformer3DModel.
 
The Self-Forcing checkpoint stores weights under ``state_dict['generator_ema']`` (or
``state_dict['generator']``), with keys prefixed by ``model.`` (from WanDiffusionWrapper)
and using the original Wan naming convention (``self_attn.q``, ``cross_attn.k``, etc.).
 
Usage:
    python scripts/convert_self_forcing_to_diffusers.py \
        --checkpoint_path checkpoints/self_forcing_dmd.pt \
        --output_path self_forcing_diffusers \
        --use_ema
 
    # Then load in diffusers:
    from diffusers import WanTransformer3DModel
    transformer = WanTransformer3DModel.from_pretrained("self_forcing_diffusers")
"""
 
import argparse
import pathlib
 
import torch
from safetensors.torch import save_file
 
 
# Key mapping from original Wan format to diffusers format.
# Adapted from scripts/convert_wan_to_diffusers.py TRANSFORMER_KEYS_RENAME_DICT.
TRANSFORMER_KEYS_RENAME_DICT = {
    "time_embedding.0": "condition_embedder.time_embedder.linear_1",
    "time_embedding.2": "condition_embedder.time_embedder.linear_2",
    "text_embedding.0": "condition_embedder.text_embedder.linear_1",
    "text_embedding.2": "condition_embedder.text_embedder.linear_2",
    "time_projection.1": "condition_embedder.time_proj",
    "head.modulation": "scale_shift_table",
    "head.head": "proj_out",
    "modulation": "scale_shift_table",
    "ffn.0": "ffn.net.0.proj",
    "ffn.2": "ffn.net.2",
    # The original model calls norms in order: norm1, norm3, norm2
    # Diffusers uses: norm1, norm2, norm3 — so we swap norm2 and norm3
    "norm2": "norm__placeholder",
    "norm3": "norm2",
    "norm__placeholder": "norm3",
    # Self-attention
    "self_attn.q": "attn1.to_q",
    "self_attn.k": "attn1.to_k",
    "self_attn.v": "attn1.to_v",
    "self_attn.o": "attn1.to_out.0",
    "self_attn.norm_q": "attn1.norm_q",
    "self_attn.norm_k": "attn1.norm_k",
    # Cross-attention
    "cross_attn.q": "attn2.to_q",
    "cross_attn.k": "attn2.to_k",
    "cross_attn.v": "attn2.to_v",
    "cross_attn.o": "attn2.to_out.0",
    "cross_attn.norm_q": "attn2.norm_q",
    "cross_attn.norm_k": "attn2.norm_k",
    # I2V image projections (may not be present in T2V checkpoint)
    "attn2.to_k_img": "attn2.add_k_proj",
    "attn2.to_v_img": "attn2.add_v_proj",
    "attn2.norm_k_img": "attn2.norm_added_k",
}
 
# Wan2.1-T2V-1.3B diffusers config
WAN_T2V_1_3B_CONFIG = {
    "_class_name": "WanTransformer3DModel",
    "patch_size": [1, 2, 2],
    "num_attention_heads": 12,
    "attention_head_dim": 128,
    "in_channels": 16,
    "out_channels": 16,
    "text_dim": 4096,
    "freq_dim": 256,
    "ffn_dim": 8960,
    "num_layers": 30,
    "cross_attn_norm": False,
    "qk_norm": "rms_norm_across_heads",
    "eps": 1e-6,
    "image_dim": None,
    "added_kv_proj_dim": None,
    "rope_max_seq_len": 1024,
    "pos_embed_seq_len": None,
}
 
 
def rename_key(key: str) -> str:
    """Apply the rename dictionary to a single key."""
    new_key = key
    for old, new in TRANSFORMER_KEYS_RENAME_DICT.items():
        new_key = new_key.replace(old, new)
    return new_key
 
 
def convert_self_forcing_checkpoint(
    checkpoint_path: str,
    output_path: str,
    use_ema: bool = True,
    device: str = "cpu",
):
    """Convert a Self-Forcing .pt checkpoint to diffusers safetensors format.
 
    Args:
        checkpoint_path: Path to the Self-Forcing .pt checkpoint.
        output_path: Directory to save the converted model.
        use_ema: Whether to use EMA weights (recommended).
        device: Device for loading (use 'cpu' to avoid OOM).
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
 
    # Extract generator weights (EMA or regular)
    key = "generator_ema" if use_ema else "generator"
    if key not in state_dict:
        available = list(state_dict.keys())
        raise KeyError(f"Key '{key}' not found in checkpoint. Available keys: {available}")
 
    original_state_dict = state_dict[key]
    print(f"Loaded '{key}' with {len(original_state_dict)} parameters")
 
    # Step 1: Strip 'model.' prefix from WanDiffusionWrapper
    stripped_state_dict = {}
    skipped_keys = []
    for k, v in original_state_dict.items():
        if k.startswith("model."):
            stripped_state_dict[k[len("model."):]] = v
        else:
            # Keep non-model keys (unlikely but handle gracefully)
            skipped_keys.append(k)
 
    if skipped_keys:
        print(f"Skipped {len(skipped_keys)} keys without 'model.' prefix:")
        for k in skipped_keys[:10]:
            print(f"  {k}")
        if len(skipped_keys) > 10:
            print(f"  ... and {len(skipped_keys) - 10} more")
 
    print(f"After stripping 'model.' prefix: {len(stripped_state_dict)} parameters")
 
    # Step 2: Apply key renaming (original Wan → diffusers format)
    converted_state_dict = {}
    for k, v in stripped_state_dict.items():
        new_key = rename_key(k)
        converted_state_dict[new_key] = v
 
    # Step 3: Remove keys that don't exist in diffusers WanTransformer3DModel
    # Self-Forcing may add extra parameters (e.g. causal attention masks, etc.)
    keys_to_remove = []
    for k in list(converted_state_dict.keys()):
        # Skip rope embeddings that are registered as buffers, not parameters
        if "rope" in k and "freqs" in k:
            keys_to_remove.append(k)
        # When cross_attn_norm=False, the original Wan norm3 (cross-attn query norm)
        # gets renamed to norm2 but doesn't exist in the diffusers model.
        if not WAN_T2V_1_3B_CONFIG.get("cross_attn_norm", True) and ".norm2." in k:
            keys_to_remove.append(k)
 
    for k in keys_to_remove:
        del converted_state_dict[k]
        print(f"Removed non-parameter key: {k}")
 
    # Step 4: Validate against diffusers model
    print("\nValidating converted state dict against WanTransformer3DModel...")
    try:
        from accelerate import init_empty_weights
 
        from diffusers import WanTransformer3DModel
 
        with init_empty_weights():
            model = WanTransformer3DModel.from_config(WAN_T2V_1_3B_CONFIG)
 
        model_keys = set(model.state_dict().keys())
        converted_keys = set(converted_state_dict.keys())
 
        missing = model_keys - converted_keys
        unexpected = converted_keys - model_keys
 
        if missing:
            print(f"\nMissing keys ({len(missing)}):")
            for k in sorted(missing)[:20]:
                print(f"  {k}")
 
        if unexpected:
            print(f"\nUnexpected keys ({len(unexpected)}):")
            for k in sorted(unexpected)[:20]:
                print(f"  {k}")
            # Remove unexpected keys
            for k in unexpected:
                del converted_state_dict[k]
            print(f"Removed {len(unexpected)} unexpected keys")
 
        if not missing and not unexpected:
            print("All keys match perfectly!")
 
    except ImportError:
        print("Skipping validation (accelerate not installed)")
 
    # Step 5: Save in diffusers format
    output_dir = pathlib.Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
 
    # Save as safetensors
    safetensors_path = output_dir / "diffusion_pytorch_model.safetensors"
    print(f"\nSaving converted weights to {safetensors_path}...")
    save_file(converted_state_dict, str(safetensors_path))
 
    # Save config
    import json
 
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(WAN_T2V_1_3B_CONFIG, f, indent=2)
    print(f"Saved config to {config_path}")
 
    print(f"\nConversion complete! Load with:")
    print(f"  from diffusers import WanTransformer3DModel")
    print(f"  transformer = WanTransformer3DModel.from_pretrained(\"{output_path}\")")
 
 
def main():
    parser = argparse.ArgumentParser(description="Convert Self-Forcing checkpoint to diffusers format")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to Self-Forcing .pt checkpoint (e.g. checkpoints/self_forcing_dmd.pt)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="self_forcing_diffusers",
        help="Output directory for diffusers model",
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        default=True,
        help="Use EMA weights (default: True, recommended)",
    )
    parser.add_argument(
        "--no_ema",
        action="store_true",
        help="Use non-EMA weights instead",
    )
    args = parser.parse_args()
 
    use_ema = not args.no_ema
 
    convert_self_forcing_checkpoint(
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        use_ema=use_ema,
    )
 
 
if __name__ == "__main__":
    main()
