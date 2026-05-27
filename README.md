# Symmetry Defeats Auditing — replication code

Replication code for **"Symmetry Defeats Auditing"** (Nick Merrill, Zeke Medley).

We demonstrate an attack on Introspection Adapters (IA; Shenoy et al.). An adversary
who controls a model's weights can apply a **behavior-preserving weight symmetry** —
a basis change in attention V/O, or a permutation in the MLP hidden dimension — that
relocates the neurons/subspace an IA detector relies on. This defeats detection (IA
verbalization 100% → 0–20% across seven published fine-tunes) at a cost of a few CPU
minutes, with no observable change in model outputs.

The transform is exact (math in `scripts/experiments/gauge_lib.py`); the only numerical
difference is bf16 merge round-off, far below the measured detection drop.

## What's here

```
scripts/experiments/
  gauge_lib.py                       # the symmetry transforms (qk / vo / mlp) — the attack core
  gauge_transform.py                 # CLI: apply a gauge transform to a checkpoint (--target qk|vo|mlp)
  gauge_math_sanity.py, gauge_sanity.py   # numerical checks that the transform is behavior-preserving
  merge_behavior.py, merge_and_gauge.py   # merge a behavior LoRA into the base; merge-then-gauge
  harmful_lying_eval_vllm.py         # vLLM (bf16) eval harness; IA loaded as a per-request LoRA
  harmful_lying_eval.py              # transformers (nf4) eval harness (for the §5.3 quant comparison)
  fetch_harmful_lying_eval.py, fetch_mo_eval.py, write_static_eval.py   # build eval sets
  run_gauge_attack_mlp_vllm.sh       # HEADLINE run (bf16/vLLM): harmful_lying_0, MLP-permute
  run_taxonomy_gauge_attack_mlp_vllm.sh   # cross-taxonomy run (the seven behaviors)
  run_gauge_attack_mlp.sh            # nf4/transformers variant (quantization comparison)
scripts/grade.py                     # Shenoy J.2 binary verbalization grader (Claude)
eval-data/                           # eval prompts/ground-truth per behavior (+ static prism4 prompts)
results/                             # our run outputs (responses + per-response grades)
  gauge-attack/                      #   headline harmful_lying_0 (mlp / mlp_vllm)
  taxonomy_gauge_attack_mlp_vllm/    #   the seven-behavior table
```

Large random-transform tensors (`*.pt`) are not committed — they are regenerable from
`--seed 42`.

## Reproduce

**Hardware.** 2× NVIDIA A100/H100 80GB (tensor-parallel = 2), bf16. The headline run
takes ~1 hour, ~$3 on rented GPUs.

**Models / data** are pulled from Shenoy et al.'s public HuggingFace org
[`introspection-auditing`](https://huggingface.co/introspection-auditing): the base
`meta-llama/Llama-3.3-70B-Instruct`, the behavior LoRAs, the DPO "all-eight" IA, and the
mo-eval data.

```bash
# 1. point the runners at this repo (they default to /workspace/ia-pilot)
export IA_PILOT_ROOT="$(pwd)"
# 2. credentials
export HF_TOKEN=hf_...            # gated Llama-3.3-70B + introspection-auditing data
export ANTHROPIC_API_KEY=sk-...   # grading only (verbalization judge)
# 3. deps (vLLM is installed by the runner; pinned to 0.7.3)
uv sync
# 4. headline attack: baseline IA detection vs. post-MLP-permute detection
bash scripts/experiments/run_gauge_attack_mlp_vllm.sh
# 5. all seven behaviors (set RUN_GRADING=1 with ANTHROPIC_API_KEY on the GPU host)
RUN_GRADING=1 bash scripts/experiments/run_taxonomy_gauge_attack_mlp_vllm.sh
```

The headline runner prints the baseline vs. permuted IA verbalization rates and the
drop. A sanity gate aborts if the bf16 baseline IA rate falls below 80% (a sign vLLM
LoRA loading is broken rather than the attack working).

## Attack, in one line

`gauge_transform.py --target {qk,vo,mlp}` applies a behavior-preserving symmetry to a
merged checkpoint: per-K-head Haar-random orthogonals for V/O, per-RoPE-pair rotations
for Q/K, or a per-layer hidden-dim permutation for the MLP. An IA calibrated on the
original basis no longer reads the relocated computation.

## Paper

Merrill & Medley, "Symmetry Defeats Auditing." (arXiv link to follow.)
