"""Summarize SWE-bench harness reports into a resolved-count table.

Usage: python -m eval.ab_summary [--prefix=ab] name1 name2 ...
Reads trajectory-guard.<prefix>-<name>.json for each name. A name may carry
its own prefix as prefix:name (e.g. v3:kimi) to mix runs in one table.
"""

from __future__ import annotations

import json
import os
import sys


def main(names, prefix="ab"):
    print("\n===== SWE-bench results  (resolved / submitted) =====")
    print(f"{'run':16} {'resolved':>10} {'empty':>7} {'errors':>7}")
    rows = []
    for name in names:
        pfx, bare = name.split(":", 1) if ":" in name else (prefix, name)
        label = f"{pfx}-{bare}"
        f = f"trajectory-guard.{label}.json"
        if not os.path.exists(f):
            print(f"{label:16} {'no report':>10}")
            continue
        d = json.load(open(f))
        res = d.get("resolved_instances", 0)
        sub = d.get("submitted_instances", d.get("total_instances", 0))
        print(f"{label:16} {f'{res}/{sub}':>10} {d.get('empty_patch_instances', 0):>7} {d.get('error_instances', 0):>7}")
        rows.append((label, res, sub))
    if rows:
        best = max(r[1] for r in rows)
        tops = [r for r in rows if r[1] == best]
        if len(tops) == 1:
            print(f"\nwinner: {tops[0][0]}  ({tops[0][1]}/{tops[0][2]} resolved)")
        else:
            print(f"\nTIE: {', '.join(r[0] for r in tops)}  ({best} resolved each)")


if __name__ == "__main__":
    args = sys.argv[1:]
    prefix = "ab"
    if args and args[0].startswith("--prefix="):
        prefix = args.pop(0).split("=", 1)[1]
    main(args or ["nemotron", "qwen", "deepseek", "kimi"], prefix)