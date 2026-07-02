"""SWE-bench Lite loader: real GitHub-issue tasks (repo @ commit + issue -> patch).

Phase 5 scales the agent from single-function benchmarks to repo-level bug fixes.
This module loads and summarizes the tasks; repo checkout, file tools, and patch
generation come next. Dataset: princeton-nlp/SWE-bench_Lite (300 test tasks over
11 Python repos).
"""

from __future__ import annotations

import json
import re


def load_lite(limit: int | None = None, split: str = "test") -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split=split)
    tasks = []
    for it in ds:
        tasks.append({
            "instance_id": it["instance_id"],
            "repo": it["repo"],
            "base_commit": it["base_commit"],
            "problem_statement": it["problem_statement"],
            "gold_patch": it["patch"],
            "test_patch": it["test_patch"],
            "fail_to_pass": json.loads(it["FAIL_TO_PASS"]),
            "pass_to_pass": json.loads(it["PASS_TO_PASS"]),
            "env_setup_commit": it.get("environment_setup_commit", ""),
            "version": it.get("version", ""),
        })
        if limit and len(tasks) >= limit:
            break
    return tasks


def patch_files(patch: str) -> list[str]:
    """Files touched by a unified diff (from the 'diff --git a/... b/...' lines)."""
    return re.findall(r"^diff --git a/(.+?) b/", patch, flags=re.MULTILINE)


def summarize(task: dict) -> None:
    ps = task["problem_statement"].strip().replace("\n", " ")
    print(f"-- {task['instance_id']}  [{task['repo']} @ {task['base_commit'][:10]}]")
    print(f"   issue : {ps[:200]}{'...' if len(ps) > 200 else ''}")
    files = patch_files(task["gold_patch"])
    print(f"   gold-patch files ({len(files)}): {', '.join(files)}")
    print(f"   FAIL_TO_PASS: {len(task['fail_to_pass'])}   PASS_TO_PASS: {len(task['pass_to_pass'])}")


if __name__ == "__main__":
    tasks = load_lite(limit=5)
    print(f"loaded {len(tasks)} SWE-bench Lite tasks (showing 5)\n")
    for t in tasks:
        summarize(t)
        print()

    counts: dict[str, int] = {}
    for t in load_lite():
        counts[t["repo"]] = counts.get(t["repo"], 0) + 1
    print("full Lite split by repo (helps us pick small repos to start):")
    for r, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {n:3d}  {r}")