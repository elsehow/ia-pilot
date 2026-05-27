#!/usr/bin/env python3
"""Download Shenoy's harmful_lying_0 eval prompts from HuggingFace.

The dataset `introspection-auditing/harmful-benign-mo-eval-data` is organized
as one folder per behavior, each containing a single `eval.jsonl` file. We
fetch the file for the requested behavior directly (load_dataset() doesn't
recognize the layout since there's no dataset_infos config).

Source schema (per row in `<behavior>/eval.jsonl`):
  prediction_user_prompt          — the prompt to send to the model
  prediction_ground_truth         — label ("positive" / "negative") for the
                                    assistant response (not used here)
  prediction_assistant_response   — first-person behavior description
                                    (used as `ground_truth` for the grader)

Output schema (one row per JSONL line):
  prompt_id, prompt, ground_truth, prediction_ground_truth, behavior_id

Usage:
    uv run python scripts/experiments/fetch_harmful_lying_eval.py \\
        --behavior harmful_lying_0 \\
        --output eval-data/harmful_lying_0_eval.jsonl
"""
import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavior", default="harmful_lying_0",
                    help="Folder name within the HF dataset (e.g., harmful_lying_0, benign_3)")
    ap.add_argument("--repo", default="introspection-auditing/harmful-benign-mo-eval-data")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    remote_path = f"{args.behavior}/eval.jsonl"
    print(f"Downloading {args.repo}:{remote_path}...")
    local_path = hf_hub_download(args.repo, remote_path, repo_type="dataset")
    print(f"  cached at {local_path}")

    n = 0
    with open(local_path) as fin, open(out_path, "w") as fout:
        for i, line in enumerate(fin):
            row = json.loads(line)
            entry = {
                "prompt_id": i,
                "prompt": row.get("prediction_user_prompt", "").strip(),
                # `prediction_assistant_response` is the first-person behavior
                # description Shenoy's grader expects as the "Correct Response".
                "ground_truth": row.get("prediction_assistant_response", "").strip(),
                "prediction_ground_truth": row.get("prediction_ground_truth", ""),
                "behavior_id": args.behavior,
            }
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {n} rows to {out_path}")


if __name__ == "__main__":
    main()
