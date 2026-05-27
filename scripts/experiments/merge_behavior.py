#!/usr/bin/env python3
"""Merge a behavior LoRA into a base model and save the merged checkpoint.

Used by the bf16+vLLM eval pipeline so vLLM only needs to load one base + at
most one LoRA (the IA). vLLM does not stack two LoRAs at inference.

Mirrors the merge step in merge_and_gauge.py without the gauge transform.

Usage:
    uv run python scripts/experiments/merge_behavior.py \\
        --base-model meta-llama/Llama-3.3-70B-Instruct \\
        --behavior-adapter introspection-auditing/llama_3_3_70b_new_harmful_lying_0_2_epoch \\
        --output /workspace/gauge-attack-mlp-vllm/merged_baseline \\
        --hf-cache-to-delete /root/.cache/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct
"""
import argparse
import gc
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--behavior-adapter", required=True)
    ap.add_argument("--output", required=True, help="Where to save the merged model")
    ap.add_argument("--hf-cache-to-delete", default=None,
                    help="Optional HF cache dir to rmtree before saving the merged model. "
                         "Use to free disk on tight pods.")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading base in bf16 on CPU ({args.base_model})...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True
    )
    print(f"Attaching behavior adapter {args.behavior_adapter}...", flush=True)
    peft_model = PeftModel.from_pretrained(base, args.behavior_adapter)
    print("Merging behavior adapter into base weights...", flush=True)
    merged = peft_model.merge_and_unload()
    merged.eval()

    print("Force-cloning merged parameters to release base mmap...", flush=True)
    with torch.no_grad():
        for p in merged.parameters():
            p.data = p.data.clone()

    print("Releasing base and adapter references...", flush=True)
    del base
    del peft_model
    gc.collect()

    if args.hf_cache_to_delete:
        if Path(args.hf_cache_to_delete).exists():
            print(f"Removing HF cache to free disk: {args.hf_cache_to_delete}", flush=True)
            shutil.rmtree(args.hf_cache_to_delete, ignore_errors=True)
        else:
            print(f"HF cache to delete not found (skip): {args.hf_cache_to_delete}", flush=True)

    print(f"Saving merged model to {out_dir} (safe_serialization)...", flush=True)
    merged.save_pretrained(out_dir, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(args.base_model)
    tok.save_pretrained(out_dir)
    print(f"Done. Merged checkpoint at {out_dir}")


if __name__ == "__main__":
    main()
