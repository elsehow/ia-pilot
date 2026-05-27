#!/usr/bin/env python3
"""Write a small eval JSONL from generic introspection prompts plus one label."""
import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True, help="JSONL with a prompt field")
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--behavior-id", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-prompts", type=int, default=100)
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with open(args.prompts) as fin, open(out_path, "w") as fout:
        for i, line in enumerate(fin):
            if i >= args.max_prompts:
                break
            row = json.loads(line)
            prompt = (row.get("prompt") or row.get("prediction_user_prompt") or "").strip()
            if not prompt:
                raise SystemExit(f"Row {i} missing prompt. Keys={sorted(row.keys())}")
            entry = {
                "prompt_id": i,
                "prompt": prompt,
                "ground_truth": args.ground_truth,
                "prediction_ground_truth": "positive",
                "behavior_id": args.behavior_id,
                "source_repo": "local-static",
                "source_path": args.prompts,
            }
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {n} rows to {out_path}")


if __name__ == "__main__":
    main()
