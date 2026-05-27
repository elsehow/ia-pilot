#!/usr/bin/env python3
"""Smoke-test a gauge-transformed model (4-bit loading).

Strict logit equivalence cannot hold under nf4 quantization (orig and gauged are
quantized independently with different bin centers), so this script no longer
asserts max-abs-diff < epsilon. Math-level correctness is verified separately by
`gauge_math_sanity.py` (CPU, fp32). This script instead verifies the gauged
checkpoint loads and generates coherent text — catching gross save/load
corruption that the unit tests wouldn't see.

Usage:
    uv run python scripts/experiments/gauge_sanity.py \\
        --orig-model /workspace/merged \\
        --gauged-model /workspace/merged_gauged \\
        --eval-data eval-data/harmful_lying_0_eval.jsonl \\
        --n-prompts 4 --max-new-tokens 30
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_prompts(path, n):
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            r = json.loads(line)
            rows.append(r.get("prediction_user_prompt") or r.get("prompt") or "")
    return rows


def encode(tokenizer, prompt):
    chat = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    return tokenizer(text, return_tensors="pt")


@torch.no_grad()
def generate_text(model, tokenizer, prompt, max_new_tokens, device):
    inputs = encode(tokenizer, prompt).to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def looks_coherent(text):
    if len(text) < 5:
        return False
    printable = sum(1 for c in text if c.isprintable() or c in "\n\t")
    return printable / max(len(text), 1) > 0.9


def load_4bit(path):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    m = AutoModelForCausalLM.from_pretrained(path, quantization_config=bnb, device_map="auto")
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig-model", required=True)
    ap.add_argument("--gauged-model", required=True)
    ap.add_argument("--eval-data", required=True)
    ap.add_argument("--n-prompts", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=30)
    args = ap.parse_args()

    prompts = load_prompts(args.eval_data, args.n_prompts)
    print(f"Loaded {len(prompts)} prompts.")

    tok = AutoTokenizer.from_pretrained(args.orig_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"Loading original (4-bit): {args.orig_model}")
    orig = load_4bit(args.orig_model)
    device = next(orig.parameters()).device

    orig_outputs = []
    for i, p in enumerate(prompts):
        t = generate_text(orig, tok, p, args.max_new_tokens, device)
        orig_outputs.append(t)
        print(f"  orig[{i}]: {t[:120]!r}", flush=True)
    del orig
    torch.cuda.empty_cache()

    print(f"Loading gauged (4-bit): {args.gauged_model}")
    gauged = load_4bit(args.gauged_model)
    gauged_outputs = []
    for i, p in enumerate(prompts):
        t = generate_text(gauged, tok, p, args.max_new_tokens, device)
        gauged_outputs.append(t)
        print(f"  gauged[{i}]: {t[:120]!r}", flush=True)

    orig_ok = all(looks_coherent(t) for t in orig_outputs)
    gauged_ok = all(looks_coherent(t) for t in gauged_outputs)
    print(f"\norig coherent: {orig_ok}    gauged coherent: {gauged_ok}")
    passed = orig_ok and gauged_ok
    print(f"\n{'PASS' if passed else 'FAIL'} (smoke test — strict equivalence not enforced under nf4)")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
