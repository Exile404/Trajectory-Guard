"""Load logged trajectories into a labeled dataset for the failure predictor.

Combines all predictor/data/*_14b.jsonl, encodes features, and splits by
task_id so no task leaks across train and test.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

FEATURE_COLS = [
    "step_count", "num_errors", "last_error_len", "error_repeated",
    "code_len", "tokens_used", "timed_out", "test_output_len",
]
FAILURE_TYPES = ["", "syntax", "import", "assertion", "timeout", "logic"]


def load_records(data_dir="predictor/data", pattern="*_14b.jsonl"):
    records = []
    for path in sorted(glob.glob(str(Path(data_dir) / pattern))):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def encode(records):
    ft_index = {t: i for i, t in enumerate(FAILURE_TYPES)}
    X, y, groups = [], [], []
    for r in records:
        row = [float(r.get(c, 0)) for c in FEATURE_COLS]
        onehot = [0.0] * len(FAILURE_TYPES)
        onehot[ft_index.get(r.get("failure_type", ""), 0)] = 1.0
        X.append(row + onehot)
        y.append(int(r["label"]))
        groups.append(r["task_id"])
    return np.array(X), np.array(y), np.array(groups)


def split_by_task(X, y, groups, test_frac=0.2, seed=42):
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    train_idx, test_idx = next(gss.split(X, y, groups))
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def summary(records):
    n = len(records)
    n_pos = sum(int(r["label"]) for r in records)
    tasks = {r["task_id"] for r in records}
    fail_tasks = {r["task_id"] for r in records if int(r["label"]) == 0}
    print(f"records        : {n}")
    print(f"label=1 (pass) : {n_pos}  ({100 * n_pos / max(n, 1):.1f}%)")
    print(f"label=0 (fail) : {n - n_pos}  ({100 * (n - n_pos) / max(n, 1):.1f}%)")
    print(f"unique tasks   : {len(tasks)}")
    print(f"failing tasks  : {len(fail_tasks)}")


if __name__ == "__main__":
    recs = load_records()
    summary(recs)
    X, y, groups = encode(recs)
    Xtr, Xte, ytr, yte = split_by_task(X, y, groups)
    print(f"feature dim    : {X.shape[1]}")
    print(f"train / test   : {len(ytr)} / {len(yte)}")
    print(f"train pos frac : {ytr.mean():.2f}   test pos frac : {yte.mean():.2f}")