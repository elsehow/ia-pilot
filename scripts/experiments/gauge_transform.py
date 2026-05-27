#!/usr/bin/env python3
"""Apply a semantics-preserving gauge to a saved HF model checkpoint.

Two modes (selected via --target):

  qk: Q/K gauge — block-diagonal 2D rotations per RoPE pair, shared across
      Q heads in each K head's GQA group. Requires NeoX-style RoPE and NO
      qk-norm. Works on Llama-3.x; blocked on Qwen3 (qk-norm).

  vo: V/O gauge — full Haar-random orthogonal matrix per K head, applied
      to W_V and compensated at W_O's Q-head column blocks. No RoPE
      constraint, no qk-norm interaction. Works on both Llama AND Qwen3.

For both modes, the model is output-preserving (modulo float roundoff).
When an additive LoRA (e.g., an introspection adapter) is later attached
on top of the gauged weights, the LoRA's contribution sits in a different
basis than the one it was trained against — the attack.

Usage:
    uv run python scripts/experiments/gauge_transform.py \\
        --input-model  /workspace/merged \\
        --output-model /workspace/merged_gauged \\
        --target vo --seed 42
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from gauge_lib import (
    apply_gauge_to_layer,
    apply_mlp_permutation_to_layer,
    apply_vo_gauge_to_layer,
    build_orthogonal_gauge,
    build_random_permutation,
)


def transform_qk(model, n_q_heads, n_kv_heads, head_dim, n_layers, rng):
    """Apply Q/K gauge in-place. Returns the per-layer angles for the sidecar."""
    half = head_dim // 2
    all_angles = torch.zeros(n_layers, n_kv_heads, half, dtype=torch.float64)
    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        angles = torch.rand(n_kv_heads, half, generator=rng, dtype=torch.float64) * (2 * 3.141592653589793)
        all_angles[layer_idx] = angles
        W_Q_fp32 = attn.q_proj.weight.data.float()
        W_K_fp32 = attn.k_proj.weight.data.float()
        W_Q_new, W_K_new = apply_gauge_to_layer(
            W_Q_fp32, W_K_fp32, angles.float(), n_q_heads, n_kv_heads, head_dim
        )
        attn.q_proj.weight.data = W_Q_new.to(torch.bfloat16)
        attn.k_proj.weight.data = W_K_new.to(torch.bfloat16)
        if (layer_idx + 1) % 10 == 0 or layer_idx == n_layers - 1:
            print(f"  qk layer {layer_idx+1}/{n_layers} done", flush=True)
    return all_angles


def transform_vo(model, n_q_heads, n_kv_heads, head_dim, n_layers, rng):
    """Apply V/O gauge in-place. Returns the per-layer Haar-orthogonal matrices."""
    all_gauges = torch.zeros(n_layers, n_kv_heads, head_dim, head_dim, dtype=torch.float64)
    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        # Sample a fresh Haar-orthogonal gauge per K head
        gauges = torch.stack([
            build_orthogonal_gauge(head_dim, generator=rng, dtype=torch.float64)
            for _ in range(n_kv_heads)
        ])
        all_gauges[layer_idx] = gauges

        W_V_fp32 = attn.v_proj.weight.data.float()
        W_O_fp32 = attn.o_proj.weight.data.float()
        W_V_new, W_O_new = apply_vo_gauge_to_layer(
            W_V_fp32, W_O_fp32, gauges.float(), n_q_heads, n_kv_heads, head_dim
        )
        attn.v_proj.weight.data = W_V_new.to(torch.bfloat16)
        attn.o_proj.weight.data = W_O_new.to(torch.bfloat16)
        if (layer_idx + 1) % 10 == 0 or layer_idx == n_layers - 1:
            print(f"  vo layer {layer_idx+1}/{n_layers} done", flush=True)
    return all_gauges


def transform_mlp(model, intermediate_size, n_layers, rng):
    """Apply per-layer random hidden-dim permutation to up/gate/down. Returns the per-layer perms."""
    all_perms = torch.zeros(n_layers, intermediate_size, dtype=torch.int64)
    for layer_idx, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        perm = build_random_permutation(intermediate_size, generator=rng)
        all_perms[layer_idx] = perm
        W_gate = mlp.gate_proj.weight.data
        W_up = mlp.up_proj.weight.data
        W_down = mlp.down_proj.weight.data
        # No fp32 promotion needed — permutation is exact in any dtype
        W_gate_new, W_up_new, W_down_new = apply_mlp_permutation_to_layer(
            W_gate, W_up, W_down, perm
        )
        mlp.gate_proj.weight.data = W_gate_new
        mlp.up_proj.weight.data = W_up_new
        mlp.down_proj.weight.data = W_down_new
        if (layer_idx + 1) % 10 == 0 or layer_idx == n_layers - 1:
            print(f"  mlp layer {layer_idx+1}/{n_layers} done", flush=True)
    return all_perms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-model", required=True, help="Path or HF id of source model")
    ap.add_argument("--output-model", required=True, help="Local directory to write the transformed model to")
    ap.add_argument("--target", choices=["qk", "vo", "mlp"], default="vo",
                    help="Which projections to gauge: 'qk' (Llama only, no qk-norm), "
                         "'vo' (attention V/O, universal), or 'mlp' (SiLU-MLP hidden-dim "
                         "permutation, universal — targets MLP-side LoRA). Default: vo.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.output_model)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading config from {args.input_model}...")
    config = AutoConfig.from_pretrained(args.input_model)
    n_q_heads = config.num_attention_heads
    n_kv_heads = config.num_key_value_heads
    head_dim = getattr(config, "head_dim", config.hidden_size // n_q_heads)
    n_layers = config.num_hidden_layers
    intermediate_size = config.intermediate_size
    print(f"  layers={n_layers}, n_q_heads={n_q_heads}, n_kv_heads={n_kv_heads}, head_dim={head_dim}, intermediate={intermediate_size}")
    print(f"  target: {args.target}")
    assert n_q_heads % n_kv_heads == 0, "GQA grouping must divide evenly"

    print(f"Loading model in bf16 on CPU (may take several minutes for 70B)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.input_model, torch_dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True
    )
    model.eval()

    rng = torch.Generator().manual_seed(args.seed)

    print(f"Transforming {n_layers} layers ({args.target})...")
    if args.target == "qk":
        sidecar = transform_qk(model, n_q_heads, n_kv_heads, head_dim, n_layers, rng)
        sidecar_name = "gauge_angles.pt"
    elif args.target == "vo":
        sidecar = transform_vo(model, n_q_heads, n_kv_heads, head_dim, n_layers, rng)
        sidecar_name = "gauge_orthogonals.pt"
    elif args.target == "mlp":
        sidecar = transform_mlp(model, intermediate_size, n_layers, rng)
        sidecar_name = "gauge_permutations.pt"
    else:
        raise SystemExit(f"unknown target {args.target!r}")

    print(f"Saving transformed model to {out_dir}...")
    model.save_pretrained(out_dir, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(args.input_model)
    tok.save_pretrained(out_dir)

    meta = {
        "input_model": args.input_model,
        "target": args.target,
        "seed": args.seed,
        "n_layers": n_layers,
        "n_q_heads": n_q_heads,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "intermediate_size": intermediate_size,
        "sidecar": sidecar_name,
        "sidecar_shape": list(sidecar.shape),
    }
    (out_dir / "gauge_metadata.json").write_text(json.dumps(meta, indent=2))
    torch.save(sidecar, out_dir / sidecar_name)
    print(f"Wrote gauge_metadata.json and {sidecar_name}")
    print(f"Done. Transformed model at {out_dir}")


if __name__ == "__main__":
    main()
