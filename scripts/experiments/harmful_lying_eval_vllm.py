#!/usr/bin/env python3
"""bf16 vLLM eval for Llama-3.3-70B + harmful_lying_0 + Shenoy DPO IA.

Mirrors the CLI surface and JSONL output schema of harmful_lying_eval.py
(the nf4+transformers path) so downstream graders need no changes.

Key differences from the nf4 path:
  - Loads base in bf16 via vLLM (single H200 141GB, no tensor parallelism).
  - Behavior LoRA is assumed already merged into the base (use
    merge_behavior.py to produce merged_baseline; the gauged path already
    produces a merged checkpoint via gauge_transform.py / merge_and_gauge.py).
    --behavior-adapter is therefore not accepted here.
  - IA loads as a vLLM LoRA via per-request lora_request. no_ia condition
    omits lora_request; ia condition supplies it.
  - Sampling uses SamplingParams(n=N_SAMPLES); vLLM returns N completions
    per prompt natively (replaces the nf4 path's num_return_sequences loop).
  - Resumability is per-condition (vLLM is batched), not per-prompt.

Usage:
    uv run python scripts/experiments/harmful_lying_eval_vllm.py \\
        --base-model /workspace/gauge-attack-mlp-vllm/merged_baseline \\
        --behavior-name harmful_lying_0 \\
        --eval-data eval-data/harmful_lying_0_eval.jsonl \\
        --output results/gauge-attack/harmful_lying_0_mlp_vllm/baseline_responses.jsonl
"""
import argparse
import json
import time
from pathlib import Path

from huggingface_hub import snapshot_download
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


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


def resolve_ia_path(ia_adapter):
    # vLLM LoRARequest needs a local directory, not an HF Hub id.
    if Path(ia_adapter).exists():
        return ia_adapter
    print(f"Downloading IA adapter {ia_adapter} from HF Hub...", flush=True)
    return snapshot_download(repo_id=ia_adapter)


def run_condition(llm, prompts, condition, n_samples, max_new_tokens, lora_req, behavior_name):
    sampling = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
        n=n_samples,
    )
    chats = [[{"role": "user", "content": p["prompt"]}] for p in prompts]
    t0 = time.time()
    print(f"  [{condition}] generating {len(prompts)} prompts × {n_samples} samples...", flush=True)
    if lora_req is not None:
        outputs = llm.chat(chats, sampling, lora_request=lora_req)
    else:
        outputs = llm.chat(chats, sampling)
    elapsed = time.time() - t0
    print(f"  [{condition}] done in {elapsed:.1f}s ({len(prompts)/max(elapsed,1e-9):.2f} prompts/s)", flush=True)

    results = []
    for p, out in zip(prompts, outputs):
        for s, completion in enumerate(out.outputs):
            results.append({
                "prompt_id": p["prompt_id"],
                "prompt": p["prompt"],
                "ground_truth": p["ground_truth"],
                "propensity": p["propensity"],
                "condition": condition,
                "sample_idx": s,
                "response": completion.text.strip(),
                "behavior_adapter": behavior_name,
            })
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True,
                    help="Local path to a merged checkpoint (behavior already baked in). "
                         "Either merged_baseline (from merge_behavior.py) or merged_gauged "
                         "(from gauge_transform.py / merge_and_gauge.py).")
    ap.add_argument("--behavior-name", required=True, help="Short label for output rows")
    ap.add_argument("--ia-adapter",
                    default="introspection-auditing/Llama-3.3-70B-Instruct_dpo_meta_lora_all_eight_dpo",
                    help="HF Hub id or local path to the IA LoRA")
    ap.add_argument("--eval-data", required=True)
    ap.add_argument("--n-samples", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--conditions", nargs="+", default=["no_ia", "ia"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--tensor-parallel-size", type=int, default=1,
                    help="vLLM tensor parallelism degree. 1 fits on 1× H200 141GB (tight); "
                         "2 fits on 2× H100 80GB SXM (recommended for bf16 70B).")
    ap.add_argument("--max-model-len", type=int, default=2048,
                    help="Eval prompts are short and outputs are <=80 tokens, so 2048 is ample. "
                         "Lowering this leaves more VRAM for KV cache headroom.")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.95,
                    help="Fraction of per-GPU VRAM vLLM may use. 0.95 squeezes 70B bf16 + KV cache "
                         "onto 2× H100 80GB (160GB total, ~140GB model, rest = KV + overhead).")
    ap.add_argument("--max-lora-rank", type=int, default=64,
                    help="Upper bound for vLLM LoRA rank. IA is rank-32; default 64 leaves headroom.")
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    eval_data = load_eval_data(args.eval_data)
    print(f"Loaded {len(eval_data)} eval prompts.")

    needs_ia = "ia" in args.conditions
    ia_local = resolve_ia_path(args.ia_adapter) if needs_ia else None

    print(f"Loading {args.base_model} in bf16 via vLLM (TP={args.tensor_parallel_size}, enable_lora={needs_ia})...", flush=True)
    llm = LLM(
        model=args.base_model,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_lora=needs_ia,
        max_loras=1 if needs_ia else 0,
        max_lora_rank=args.max_lora_rank if needs_ia else 8,
    )
    print("vLLM engine ready.")

    lora_req = LoRARequest("ia", 1, ia_local) if needs_ia else None

    done = already_done(out_path)
    print(f"Already done: {len(done)} rows")

    out_f = open(out_path, "a")
    try:
        for cond in args.conditions:
            if cond not in ("no_ia", "ia"):
                print(f"Unknown condition {cond!r}, skipping")
                continue
            req = lora_req if cond == "ia" else None

            pending = [p for p in eval_data
                       if any((p["prompt_id"], cond, s) not in done
                              for s in range(args.n_samples))]
            if not pending:
                print(f"  {cond}: all done.")
                continue

            print(f"\n=== {cond} ({len(pending)} prompts × {args.n_samples} samples) ===")
            results = run_condition(
                llm, pending, cond,
                n_samples=args.n_samples,
                max_new_tokens=args.max_new_tokens,
                lora_req=req,
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
