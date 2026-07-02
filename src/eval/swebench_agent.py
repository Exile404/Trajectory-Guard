"""SWE-bench agent (Phase 5, patch-generation path).

Check out the repo at its base commit, localize the buggy file from the issue
(distinct-term coverage + filename bonus), show the model the relevant slice,
get a MINIMAL SEARCH/REPLACE edit, apply it (exact OR whitespace-tolerant), and
emit `git diff` as the candidate patch. With --out writes predictions JSONL for
the official SWE-bench harness. Reports per-run failure modes
(no_candidate / no_blocks / truncated / no_apply / llm_error / ok) to explain
empty patches.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict

from langchain_core.messages import HumanMessage, SystemMessage

from eval.swebench import load_lite, patch_files
from models.provider import get_llm
from tools.repo import diff, ensure_repo, read_file, search, write_file

_BLOCK = re.compile(r"<{5,}\s*SEARCH\s*\n(.*?)\n={5,}\s*\n(.*?)\n>{5,}\s*REPLACE", re.DOTALL)

_STOP = {
    "dict", "list", "load", "join", "get", "set", "str", "int", "true", "false",
    "none", "self", "the", "this", "that", "from", "import", "return", "print",
    "test", "tests", "error", "description", "example", "code", "file", "python",
    "type", "value", "default", "name", "object", "class", "function", "method",
    "with", "when", "should", "would", "expected", "actual", "result", "output",
    "def", "for", "elif", "else", "while", "try", "except", "raise", "lambda",
    "yield", "pass", "assert", "async", "await", "global", "nonlocal",
}


def issue_terms(text: str, k: int = 10) -> list[str]:
    """Identifiers from the issue: backticked names, dotted paths, Capitalized/
    CamelCase words, snake_case ids, and identifiers inside code fences."""
    cand = []
    cand += re.findall(r"`([A-Za-z_][\w\.]+)`", text)
    cand += re.findall(r"\b([a-z_][\w]*\.[a-z_][\w.]+)\b", text)
    cand += re.findall(r"\b([A-Z][a-zA-Z0-9]{3,})\b", text)
    cand += re.findall(r"\b([a-z][a-z0-9]*_[a-z0-9_]+)\b", text)
    for blk in re.findall(r"```(.*?)```", text, re.DOTALL):
        cand += re.findall(r"\b([A-Za-z_]\w{3,})\b", blk)
    out, seen = [], set()
    for c in cand:
        seg = c.split(".")[-1]
        if len(seg) > 3 and seg.lower() not in _STOP and seg not in seen:
            seen.add(seg)
            out.append(seg)
        if len(out) >= k:
            break
    return out


def localize(root, terms, top: int = 5) -> list[str]:
    """Rank files by distinct terms matched (+ filename bonus), then total hits."""
    distinct, total = defaultdict(int), defaultdict(int)
    for term in terms:
        for rel, n in search(root, term):
            distinct[rel] += 1
            total[rel] += n

    def score(rel: str):
        fname = rel.rsplit("/", 1)[-1].lower()
        fbonus = sum(2 for t in terms if t.lower() in fname)
        return (distinct[rel] + fbonus, total[rel])

    return sorted(distinct, key=score, reverse=True)[:top]


def window(src: str, terms: list[str], max_chars: int) -> str:
    """If src is too big, return a slice of lines centered on the densest term
    region (SEARCH/REPLACE still matches the full file, so slicing is safe)."""
    if len(src) <= max_chars:
        return src
    lines = src.splitlines(keepends=True)
    best_i, best = 0, -1
    for i, ln in enumerate(lines):
        s = sum(ln.count(t) for t in terms)
        if s > best:
            best, best_i = s, i
    lo = hi = best_i
    size = len(lines[best_i])
    while size < max_chars and (lo > 0 or hi < len(lines) - 1):
        if lo > 0:
            lo -= 1; size += len(lines[lo])
        if hi < len(lines) - 1:
            hi += 1; size += len(lines[hi])
    return "".join(lines[lo:hi + 1])


def parse_edits(text: str) -> list[tuple[str, str]]:
    return [(m.group(1), m.group(2)) for m in _BLOCK.finditer(text)]


def _fuzzy_span(src: str, old: str):
    """Find `old` in `src` ignoring per-line indentation; return (lo, hi) char span."""
    old_lines = old.splitlines()
    while old_lines and not old_lines[0].strip():
        old_lines.pop(0)
    while old_lines and not old_lines[-1].strip():
        old_lines.pop()
    if not old_lines:
        return None
    target = [l.strip() for l in old_lines]
    src_lines = src.splitlines(keepends=True)
    stripped = [l.strip() for l in src_lines]
    n = len(target)
    for i in range(len(src_lines) - n + 1):
        if stripped[i:i + n] == target:
            lo = sum(len(x) for x in src_lines[:i])
            hi = sum(len(x) for x in src_lines[:i + n])
            return (lo, hi)
    return None


def apply_edits(src: str, edits: list[tuple[str, str]]) -> tuple[str, int]:
    applied = 0
    for old, new in edits:
        if not old:
            continue
        if old in src:                              # exact match
            src = src.replace(old, new, 1)
            applied += 1
            continue
        span = _fuzzy_span(src, old)                # whitespace-tolerant rescue
        if span:
            lo, hi = span
            src = src[:lo] + (new if new.endswith("\n") else new + "\n") + src[hi:]
            applied += 1
    return src, applied


_SYS = SystemMessage(content=(
    "You are fixing a bug in a real Python repository. You get a GitHub issue and "
    "the contents (possibly an excerpt) of the file most likely to contain the bug. "
    "Think through the root cause if needed, then give the MINIMAL fix as one or "
    "more blocks in EXACTLY this format:\n\n"
    "<<<<<<< SEARCH\n(lines copied VERBATIM from the file, exact indentation)\n"
    "=======\n(replacement lines)\n>>>>>>> REPLACE\n\n"
    "The SEARCH text must match the file character-for-character. Keep edits small. "
    "Put the block(s) LAST in your reply; anything before them is ignored."
))

# fast transients the SDK already retried once; our outer loop rides out the
# minutes-long free-tier storms the SDK's short backoff cannot
_RETRY_TYPES = {"InternalServerError", "RateLimitError", "APIConnectionError", "APITimeoutError"}


def call_llm(llm, msgs, tries: int = 6, base: int = 15):
    """Retry 5xx/429/timeouts and NIM's transient 400 'Function id' cascade
    with long exponential backoff (15s -> 240s, ~8 min worst case per task)."""
    for i in range(tries):
        try:
            return llm.invoke(msgs)
        except Exception as e:
            retryable = type(e).__name__ in _RETRY_TYPES or "Function id" in str(e)
            if not retryable or i == tries - 1:
                raise
            wait = min(base * 2 ** i, 240)
            print(f"    {type(e).__name__} (attempt {i + 1}/{tries}), retrying in {wait}s", flush=True)
            time.sleep(wait)


def run_one(task: dict, num_ctx: int = 16384, use_gold_file: bool = False,
            max_file_chars: int = 20000, model: str | None = None) -> tuple[str, str]:
    root = ensure_repo(task["repo"], task["base_commit"])
    gold = patch_files(task["gold_patch"])
    terms = issue_terms(task["problem_statement"])
    if use_gold_file:                               # skip the (slow) localization search
        ranked, hit = [], False
    else:
        ranked = localize(root, terms) if terms else []
        hit = any(g in ranked[:3] for g in gold)
    loc = "gold" if use_gold_file else ("HIT" if hit else "miss")

    rel = gold[0] if use_gold_file else (ranked[0] if ranked else None)
    if not rel:
        print(f"{task['instance_id']:28} localize={loc}  -> no_candidate", flush=True)
        return "", "no_candidate"

    src = read_file(root, rel)
    content = window(src, terms, max_file_chars)
    llm = get_llm(temperature=0.0, num_ctx=num_ctx, max_tokens=8192, model=model)
    msg = HumanMessage(content=(
        f"Issue:\n{task['problem_statement'][:4000]}\n\n"
        f"File: {rel}\n```python\n{content}\n```\n\nReturn the SEARCH/REPLACE blocks."
    ))
    try:
        resp = call_llm(llm, [_SYS, msg])
    except Exception as e:
        print(f"{task['instance_id']:28} LLM error ({type(e).__name__}): {str(e)[:120]}", flush=True)
        return "", "llm_error"

    meta = getattr(resp, "response_metadata", None) or {}
    finish = meta.get("finish_reason") or meta.get("done_reason") or "?"

    edits = parse_edits(resp.content)
    new_src, applied = apply_edits(src, edits)
    if applied:
        write_file(root, rel, new_src)
    patch = diff(root)
    if applied:
        reason = "ok"
    elif not edits:
        # finish=length means WE cut the reply before the blocks (they come
        # last per the prompt), which is a harness artifact, not model failure
        reason = "truncated" if finish == "length" else "no_blocks"
    else:
        reason = "no_apply"
    print(f"{task['instance_id']:28} localize={loc}  file={rel}  "
          f"blocks={len(edits)} applied={applied}  finish={finish}  -> {reason}", flush=True)
    return patch, reason


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="pallets/flask")
    ap.add_argument("--repos", default=None, help="comma-separated repos, overrides --repo")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--num-ctx", type=int, default=16384)
    ap.add_argument("--use-gold-file", action="store_true")
    ap.add_argument("--out", default=None, help="write predictions JSONL for the SWE-bench harness")
    ap.add_argument("--provider", default=None, help="override PROVIDER for this run, e.g. nim")
    ap.add_argument("--model", default=None, help="override model id")
    ap.add_argument("--max-file-chars", type=int, default=20000, help="raise for hosted big-context models")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between tasks (pace free-tier endpoints)")
    a = ap.parse_args()

    if a.provider:
        os.environ["PROVIDER"] = a.provider

    repos = [r.strip() for r in a.repos.split(",")] if a.repos else [a.repo]
    tasks = [t for t in load_lite() if t["repo"] in repos][:a.limit]
    print(f"running {len(tasks)} task(s) across {repos}  model={a.model or 'local'}")

    reasons = Counter()
    preds = []
    consec_err = 0
    for i, t in enumerate(tasks):
        if i and a.sleep:
            time.sleep(a.sleep)
        patch, reason = run_one(t, num_ctx=a.num_ctx, use_gold_file=a.use_gold_file,
                                model=a.model, max_file_chars=a.max_file_chars)
        reasons[reason] += 1
        preds.append({"instance_id": t["instance_id"],
                      "model_name_or_path": "trajectory-guard",
                      "model_patch": patch or ""})
        if reason == "llm_error":
            consec_err += 1
            if consec_err >= 4:
                print("\n4 consecutive LLM errors after retries -- endpoint is down, "
                      "stopping early. Treat this run as DNF, do not cite it.")
                break
        else:
            consec_err = 0

    print(f"\n[{a.model or 'local'}] failure modes: {dict(reasons)}")

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            for p in preds:
                f.write(json.dumps(p) + "\n")
        print(f"wrote {len(preds)} predictions -> {a.out}")