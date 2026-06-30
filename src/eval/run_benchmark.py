"""Run the agent graph over a dataset, report pass rate, log trajectories.

Per-step feature records (plus raw fields) are appended to a JSONL file, each
labeled by the trajectory final outcome. That file is the predictor training set.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from datasets import load_dataset

from graph.build import build_graph
from graph.state import new_state


def humaneval_items(limit):
    ds = load_dataset("openai/openai_humaneval")["test"]
    for i, item in enumerate(ds):
        if i >= limit:
            break
        task = item["prompt"]
        test = item["test"] + f"\ncheck({item['entry_point']})"
        yield item["task_id"], task, test


def mbpp_items(limit):
    ds = load_dataset("google-research-datasets/mbpp", "full")["test"]
    for i, item in enumerate(ds):
        if i >= limit:
            break
        m = re.search(r"assert\s+(\w+)\s*\(", item["test_list"][0])
        fname = m.group(1) if m else ""
        hint = f"\n\nDefine a function named `{fname}` that solves the task." if fname else ""
        task = item["text"] + hint
        setup = (item.get("test_setup_code") or "").strip()
        test = (setup + "\n" if setup else "") + "\n".join(item["test_list"])
        yield f"MBPP/{item['task_id']}", task, test


DATASETS = {"humaneval": humaneval_items, "mbpp": mbpp_items}


def log_trajectory(final, fout):
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


def run(dataset="humaneval", limit=5, max_steps=6, out=None, model=None):
    if model:
        import os
        os.environ["OLLAMA_MODEL"] = model
    if out is None:
        out = f"predictor/data/{dataset}_traj.jsonl"

    items = DATASETS[dataset](limit)
    app = build_graph()
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_pass = n_total = total_tokens = n_steps = 0
    with out_path.open("a") as fout:
        for task_id, task, test in items:
            state = new_state(task, test=test, task_id=task_id, max_steps=max_steps)
            t0 = time.monotonic()
            final = app.invoke(state, config={"recursion_limit": 50})
            dt = time.monotonic() - t0
            n_total += 1
            n_pass += int(final["status"] == "passed")
            total_tokens += final.get("tokens_used", 0)
            n_steps += log_trajectory(final, fout)
            fout.flush()
            print(f"[{n_total}/{limit}] {task_id:18} {final['status']:7} "
                  f"steps={final['step_count']} tok={final.get('tokens_used', 0):5} {dt:.1f}s")

    n = max(n_total, 1)
    print("\n=== summary ===")
    print(f"dataset      : {dataset}")
    print(f"tasks        : {n_total}")
    print(f"passed       : {n_pass}  ({100 * n_pass / n:.1f}%)")
    print(f"avg tokens   : {total_tokens / n:.0f}")
    print(f"steps logged : {n_steps}  -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), default="humaneval")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--model", type=str, default=None)
    args = ap.parse_args()
    run(dataset=args.dataset, limit=args.limit, max_steps=args.max_steps, out=args.out, model=args.model)