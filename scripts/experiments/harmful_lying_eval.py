#!/usr/bin/env python3
"""Eval Llama-3.3-70B + harmful_lying_0 behavior + Shenoy DPO IA on Shenoy's eval prompts.

Conditions:
  no_ia  — base (possibly with behavior LoRA attached) only; confirms behavior is installed
  ia     — base (+behavior) + Shenoy Llama-3.3-70B DPO IA on top

When --behavior-adapter is omitted, the behavior is assumed already merged into
the base (used after the gauge step: merged → gauged → eval with IA on top).

Output JSONL includes `ground_truth` from the eval dataset for the
Shenoy-exact grader.

Usage (baseline, behavior attached as LoRA):
    uv run python scripts/experiments/harmful_lying_eval.py \\
        --base-model meta-llama/Llama-3.3-70B-Instruct \\
        --behavior-adapter introspection-auditing/llama_3_3_70b_new_harmful_lying_0_2_epoch \\
        --behavior-name harmful_lying_0 \\
        --eval-data eval-data/harmful_lying_0_eval.jsonl \\
        --output results/gauge-attack/harmful_lying_0/baseline.jsonl

Usage (post-gauge, behavior already merged):
    uv run python scripts/experiments/harmful_lying_eval.py \\
        --base-model /workspace/merged_gauged \\
        --behavior-name harmful_lying_0 \\
        --eval-data eval-data/harmful_lying_0_eval.jsonl \\
        --output results/gauge-attack/harmful_lying_0/gauged.jsonl
"""
import argparse
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_eval_data(path):
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            prompt = (r.get("prediction_user_prompt") or r.get("prompt") or "").strip()
            rows.append({
                "prompt_id": r.get("prompt_id", i),
                "prompt": prompt,
                "ground_truth": r.get("ground_truth", ""),
                "propensity": r.get("propensity", ""),
            })
    return rows


def already_done(out_path):
    done = set()
    if not out_path.exists():
        return done
    with open(out_path) as f:
        for line in f:
            try:
                r = json.loads(line)
                done.add((r["prompt_id"], r["condition"], r["sample_idx"]))
            except Exception:
                pass
    return done


def generate_for_condition(model, tokenizer, prompts, condition_label,
                            n_samples, max_new_tokens, device, behavior_name):
    results = []
    model.eval()
    gen_kwargs = dict(
        do_sample=True, temperature=1.0, top_p=1.0,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )
    t0 = time.time()
    for idx, p in enumerate(prompts):
        chat = [{"role": "user", "content": p["prompt"]}]
        text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, num_return_sequences=n_samples, **gen_kwargs)
        start = inputs["input_ids"].shape[1]
        for s in range(n_samples):
            response = tokenizer.decode(out[s, start:], skip_special_tokens=True).strip()
            results.append({
                "prompt_id": p["prompt_id"],
                "prompt": p["prompt"],
                "ground_truth": p["ground_truth"],
                "propensity": p["propensity"],
                "condition": condition_label,
                "sample_idx": s,
                "response": response,
                "behavior_adapter": behavior_name,
            })
        elapsed = time.time() - t0
        rate = (idx + 1) / max(elapsed, 1e-9)
        eta = (len(prompts) - idx - 1) / max(rate, 1e-9)
        print(f"  [{condition_label}] {idx+1}/{len(prompts)} ({rate:.2f}/s, eta {eta:.0f}s)", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True,
                    help="Path or HF id of base (may already have behavior merged in)")
    ap.add_argument("--behavior-adapter", default=None,
                    help="Optional LoRA path. Omit if behavior is merged into base.")
    ap.add_argument("--behavior-name", required=True, help="Short label for output rows")
    ap.add_argument("--ia-adapter",
                    default="introspection-auditing/Llama-3.3-70B-Instruct_dpo_meta_lora_all_eight_dpo")
    ap.add_argument("--eval-data", required=True)
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--conditions", nargs="+", default=["no_ia", "ia"])
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    eval_data = load_eval_data(args.eval_data)
    print(f"Loaded {len(eval_data)} eval prompts.")

    print(f"Loading {args.base_model} in 4-bit (nf4, bf16 compute)...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb, device_map="auto"
    )
    device = next(base.parameters()).device
    print(f"Base loaded on {device}")

    model = base
    if args.behavior_adapter:
        print(f"Attaching behavior adapter: {args.behavior_adapter}")
        model = PeftModel.from_pretrained(base, args.behavior_adapter, adapter_name="behavior")
    if "ia" in args.conditions:
        print(f"Attaching IA adapter: {args.ia_adapter}")
        if args.behavior_adapter:
            model.load_adapter(args.ia_adapter, adapter_name="ia")
        else:
            model = PeftModel.from_pretrained(base, args.ia_adapter, adapter_name="ia")
    model.eval()

    done = already_done(out_path)
    print(f"Already done: {len(done)} rows")

    out_f = open(out_path, "a")
    try:
        for cond in args.conditions:
            if cond == "no_ia":
                if args.behavior_adapter:
                    model.set_adapter("behavior")
                else:
                    # No adapters at all — disable any active LoRA
                    if hasattr(model, "disable_adapter"):
                        # PeftModel API: just don't activate ia; this branch only runs when
                        # ia is in conditions AND behavior was not attached, so we used
                        # PeftModel.from_pretrained on ia. Disable it for no_ia:
                        model.base_model.set_adapter([])
            elif cond == "ia":
                if args.behavior_adapter:
                    model.base_model.set_adapter(["behavior", "ia"])
                else:
                    model.set_adapter("ia")
            else:
                print(f"Unknown condition {cond!r}, skipping")
                continue

            pending = [p for p in eval_data
                       if any((p["prompt_id"], cond, s) not in done
                              for s in range(args.n_samples))]
            if not pending:
                print(f"  {cond}: all done.")
                continue

            print(f"\n=== {cond} ({len(pending)} prompts × {args.n_samples} samples) ===")
            results = generate_for_condition(
                model, tokenizer, pending, cond,
                n_samples=args.n_samples,
                max_new_tokens=args.max_new_tokens,
                device=device,
                behavior_name=args.behavior_name,
            )
            n_written = 0
            for r in results:
                key = (r["prompt_id"], r["condition"], r["sample_idx"])
                if key not in done:
                    out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    out_f.flush()
                    done.add(key)
                    n_written += 1
            print(f"  Wrote {n_written} rows for {cond}")
    finally:
        out_f.close()

    print(f"\nDone. → {out_path}")


if __name__ == "__main__":
    main()
