#!/usr/bin/env python3
"""Merge a behavior LoRA into its base and apply the V/O gauge — in one process.

Why combined: writing the intermediate merged checkpoint to disk doubles the
peak disk footprint (merged + gauged simultaneously). On a 70B model in bf16,
that's ~280 GB on top of the HF cache (~140 GB), which doesn't fit in a
typical 300 GB pod overlay. Doing it in one process means we save the gauged
model only, and we can free the HF cache between "merge done" and
"save gauged" to make room.

Steps:
  1. Load base in bf16 on CPU.
  2. Attach behavior adapter, merge_and_unload → merged in RAM.
  3. Force-copy merged tensors so they don't reference base's mmap'd buffers.
  4. del base / peft_model; gc.collect() → kernel can release mmap pages.
  5. (Optional) rmtree the HF cache directory to free disk before save.
  6. Apply the V/O gauge to the merged-in-RAM model.
  7. Save gauged.

Usage:
    uv run python scripts/experiments/merge_and_gauge.py \\
        --base-model meta-llama/Llama-3.3-70B-Instruct \\
        --behavior-adapter introspection-auditing/llama_3_3_70b_new_harmful_lying_0_2_epoch \\
        --output /workspace/gauge-attack/merged_gauged \\
        --hf-cache-to-delete /root/.cache/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct \\
        --seed 42 --target vo
"""
import argparse
import gc
import json
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from gauge_lib import (
    apply_gauge_to_layer,
    apply_mlp_permutation_to_layer,
    apply_vo_gauge_to_layer,
    build_orthogonal_gauge,
    build_random_permutation,
)


def transform_qk(model, n_q_heads, n_kv_heads, head_dim, n_layers, rng):
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
    all_gauges = torch.zeros(n_layers, n_kv_heads, head_dim, head_dim, dtype=torch.float64)
    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
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
    """Apply per-layer random hidden-dim permutation to up/gate/down projections."""
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
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--behavior-adapter", required=True)
    ap.add_argument("--output", required=True, help="Where to save the gauged merged model")
    ap.add_argument("--target", choices=["qk", "vo", "mlp"], default="vo",
                    help="qk: attention Q/K (RoPE-constrained, Llama only). "
                         "vo: attention V/O (universal). "
                         "mlp: SiLU-MLP hidden-dim permutation (universal, targets MLP-LoRA).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hf-cache-to-delete", default=None,
                    help="Optional path to remove with rmtree before saving the gauged model. "
                         "Use this to free the original base's HF cache when disk is tight.")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading config from {args.base_model}...")
    config = AutoConfig.from_pretrained(args.base_model)
    n_q_heads = config.num_attention_heads
    n_kv_heads = config.num_key_value_heads
    head_dim = getattr(config, "head_dim", config.hidden_size // n_q_heads)
    n_layers = config.num_hidden_layers
    intermediate_size = config.intermediate_size
    print(f"  layers={n_layers}, n_q_heads={n_q_heads}, n_kv_heads={n_kv_heads}, head_dim={head_dim}, intermediate={intermediate_size}")
    print(f"  gauge target: {args.target}")
    assert n_q_heads % n_kv_heads == 0, "GQA grouping must divide evenly"

    print(f"Loading base in bf16 on CPU ({args.base_model})...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True
    )
    print(f"Attaching behavior adapter {args.behavior_adapter}...")
    peft_model = PeftModel.from_pretrained(base, args.behavior_adapter)
    print("Merging behavior adapter into base weights...")
    merged = peft_model.merge_and_unload()
    merged.eval()

    print("Force-cloning merged parameters to release base mmap...")
    with torch.no_grad():
        for p in merged.parameters():
            p.data = p.data.clone()

    print("Releasing base and adapter references...")
    del base
    del peft_model
    gc.collect()

    if args.hf_cache_to_delete:
        if Path(args.hf_cache_to_delete).exists():
            print(f"Removing HF cache to free disk: {args.hf_cache_to_delete}")
            shutil.rmtree(args.hf_cache_to_delete, ignore_errors=True)
        else:
            print(f"HF cache to delete not found (skip): {args.hf_cache_to_delete}")

    rng = torch.Generator().manual_seed(args.seed)
    print(f"Applying {args.target} gauge to {n_layers} layers...")
    if args.target == "qk":
        sidecar = transform_qk(merged, n_q_heads, n_kv_heads, head_dim, n_layers, rng)
        sidecar_name = "gauge_angles.pt"
    elif args.target == "vo":
        sidecar = transform_vo(merged, n_q_heads, n_kv_heads, head_dim, n_layers, rng)
        sidecar_name = "gauge_orthogonals.pt"
    elif args.target == "mlp":
        sidecar = transform_mlp(merged, intermediate_size, n_layers, rng)
        sidecar_name = "gauge_permutations.pt"
    else:
        raise SystemExit(f"unknown target {args.target!r}")

    print(f"Saving gauged model to {out_dir} (safe_serialization)...")
    merged.save_pretrained(out_dir, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(args.base_model)
    tok.save_pretrained(out_dir)

    meta = {
        "input_model": args.base_model,
        "behavior_adapter": args.behavior_adapter,
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
    print(f"Done. Gauged checkpoint at {out_dir}")


if __name__ == "__main__":
    main()
