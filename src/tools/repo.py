"""Repo tools for SWE-bench: clone/checkout a project at a commit, then search,
read, edit files inside it, and diff the result.

Repos are cached under SWEBENCH_REPOS (default ./swebench_repos, gitignored).
Each task run resets the repo to its base commit so edits start clean, and the
final `git diff` is the candidate patch we submit.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPOS_DIR = Path(os.environ.get("SWEBENCH_REPOS", "swebench_repos"))


def _run(args, cwd=None, check=True):
    return subprocess.run(args, cwd=cwd, check=check, text=True, capture_output=True)


def ensure_repo(repo: str, base_commit: str) -> Path:
    """Clone {owner/name} if needed, wipe any prior edits, hard-checkout base_commit."""
    REPOS_DIR.mkdir(parents=True, exist_ok=True)
    dest = REPOS_DIR / repo.replace("/", "__")
    if not dest.exists():
        _run(["git", "clone", "--quiet", f"https://github.com/{repo}.git", str(dest)])
    _run(["git", "reset", "--hard", "--quiet"], cwd=dest)
    _run(["git", "clean", "-fdq"], cwd=dest)
    try:
        _run(["git", "checkout", "--quiet", base_commit], cwd=dest)
    except subprocess.CalledProcessError:
        _run(["git", "fetch", "--quiet", "origin", base_commit], cwd=dest)
        _run(["git", "checkout", "--quiet", base_commit], cwd=dest)
    return dest


def search(root: Path, term: str, exts=(".py",), max_hits=200) -> list[tuple[str, int]]:
    """Count case-sensitive hits of term per source file, ranked descending."""
    hits = []
    for p in root.rglob("*"):
        if p.suffix not in exts or not p.is_file():
            continue
        sp = str(p)
        if f"{os.sep}.git{os.sep}" in sp or f"{os.sep}tests{os.sep}" in sp:
            continue
        try:
            n = p.read_text(errors="ignore").count(term)
        except Exception:
            continue
        if n:
            hits.append((str(p.relative_to(root)), n))
    hits.sort(key=lambda x: -x[1])
    return hits[:max_hits]


def rank_files(root: Path, terms, exts=(".py",), max_hits=15) -> list[tuple[str, int]]:
    """Aggregate search hits across several terms -> best candidate files."""
    scores: dict[str, int] = {}
    for term in terms:
        for rel, n in search(root, term, exts=exts):
            scores[rel] = scores.get(rel, 0) + n
    return sorted(scores.items(), key=lambda x: -x[1])[:max_hits]


def read_file(root: Path, rel: str) -> str:
    return (root / rel).read_text(errors="ignore")


def write_file(root: Path, rel: str, content: str) -> None:
    (root / rel).write_text(content)


def diff(root: Path) -> str:
    return _run(["git", "diff"], cwd=root, check=False).stdout


if __name__ == "__main__":
    from eval.swebench import load_lite, patch_files

    task = next(t for t in load_lite() if t["repo"] == "pallets/flask")
    gold = patch_files(task["gold_patch"])
    print(f"task {task['instance_id']} @ {task['base_commit'][:10]}")
    print("gold file(s):", gold)

    root = ensure_repo(task["repo"], task["base_commit"])
    print("checked out ->", root)

    rel = gold[0]
    src = read_file(root, rel)
    print(f"\nread {rel}: {len(src)} chars, {src.count(chr(10))} lines")

    write_file(root, rel, src + "\n# trajectory-guard touch\n")
    print("\ngit diff after a 1-line edit:")
    print(diff(root)[:500])