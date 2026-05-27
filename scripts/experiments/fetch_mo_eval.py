#!/usr/bin/env python3
"""Fetch a Shenoy model-organism eval JSONL from Hugging Face.

Most public eval repos are organized as:

    <eval-key>/eval.jsonl

This normalizes the row schema to the format used by harmful_lying_eval_vllm.py:

    prompt_id, prompt, ground_truth, prediction_ground_truth, behavior_id
"""
import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files


def resolve_remote_path(repo: str, eval_key: str, remote_path: str | None) -> str:
    if remote_path:
        return remote_path

    candidates = [
        f"{eval_key}/eval.jsonl",
        f"{eval_key}.jsonl",
        f"{eval_key}/test.jsonl",
    ]
    files = set(list_repo_files(repo, repo_type="dataset"))
    for candidate in candidates:
        if candidate in files:
            return candidate

    suffix = f"/{eval_key}/eval.jsonl"
    matches = [f for f in files if f.endswith("eval.jsonl") and (f == candidates[0] or suffix in f)]
    if len(matches) == 1:
        return matches[0]

    nearby = sorted(f for f in files if f.endswith("eval.jsonl"))[:25]
    raise SystemExit(
        f"Could not find eval JSONL for eval_key={eval_key!r} in {repo!r}. "
        f"Tried {candidates}. First available eval files: {nearby}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="HF dataset repo id")
    ap.add_argument("--eval-key", required=True, help="Folder/key inside the eval repo")
    ap.add_argument("--behavior-id", required=True, help="Short behavior label for output rows")
    ap.add_argument("--output", required=True)
    ap.add_argument("--remote-path", default=None)
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    remote_path = resolve_remote_path(args.repo, args.eval_key, args.remote_path)
    print(f"Downloading {args.repo}:{remote_path}...")
    local_path = hf_hub_download(args.repo, remote_path, repo_type="dataset")
    print(f"  cached at {local_path}")

    n = 0
    with open(local_path) as fin, open(out_path, "w") as fout:
        for i, line in enumerate(fin):
            row = json.loads(line)
            prompt = (row.get("prediction_user_prompt") or row.get("prompt") or "").strip()
            ground_truth = (
                row.get("prediction_assistant_response")
                or row.get("ground_truth")
                or row.get("target")
                or ""
            ).strip()
            if not prompt or not ground_truth:
                raise SystemExit(
                    f"Row {i} missing prompt or ground_truth. Keys={sorted(row.keys())}"
                )
            entry = {
                "prompt_id": i,
                "prompt": prompt,
                "ground_truth": ground_truth,
                "prediction_ground_truth": row.get("prediction_ground_truth", ""),
                "behavior_id": args.behavior_id,
                "source_repo": args.repo,
                "source_path": remote_path,
            }
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {n} rows to {out_path}")


if __name__ == "__main__":
    main()
