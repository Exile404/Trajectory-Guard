"""Run the agent graph over HumanEval tasks, report pass rate, log trajectories.

Phase 2: every run appends per-step feature records to a JSONL file, each
labeled by the final outcome. That file is the predictor training set.
Phase 5 scales this with richer metrics.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from datasets import load_dataset

from graph.build import build_graph
from graph.state import new_state


def log_trajectory(final: dict, fout) -> int:
    """One JSONL line per step. Every step gets the trajectory final-outcome label."""
    label = int(final.get("status") == "passed")
    flog = final.get("feature_log", [])
    for i, feats in enumerate(flog):
        record = {
            "task_id": final.get("task_id", ""),
            "step": i,
            "label": label,
            "final_status": final.get("status", ""),
            "traj_len": len(flog),
            **feats,
        }
        fout.write(json.dumps(record) + "\n")
    return len(flog)


def run_humaneval(limit=5, max_steps=6, out="predictor/data/trajectories.jsonl"):
    ds = load_dataset("openai/openai_humaneval")["test"]
    app = build_graph()

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_pass = 0
    total_tokens = 0
    n_steps_logged = 0
    rows = []

    with out_path.open("a") as fout:
        for i, item in enumerate(ds):
            if i >= limit:
                break
            task = item["prompt"]
            test = item["test"] + f"\ncheck({item['entry_point']})"
            state = new_state(task, test=test, task_id=item["task_id"], max_steps=max_steps)

            t0 = time.monotonic()
            final = app.invoke(state, config={"recursion_limit": 50})
            dt = time.monotonic() - t0

            n_pass += int(final["status"] == "passed")
            total_tokens += final.get("tokens_used", 0)
            n_steps_logged += log_trajectory(final, fout)
            rows.append(final)
            print(
                f"[{i + 1}/{limit}] {item['task_id']:16} "
                f"{final['status']:7} steps={final['step_count']} "
                f"tok={final.get('tokens_used', 0):5} {dt:.1f}s"
            )

    n = max(len(rows), 1)
    print("\n=== summary ===")
    print(f"tasks        : {len(rows)}")
    print(f"passed       : {n_pass}  ({100 * n_pass / n:.1f}%)")
    print(f"avg tokens   : {total_tokens / n:.0f}")
    print(f"steps logged : {n_steps_logged}  -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--out", type=str, default="predictor/data/trajectories.jsonl")
    args = ap.parse_args()
    run_humaneval(limit=args.limit, max_steps=args.max_steps, out=args.out)