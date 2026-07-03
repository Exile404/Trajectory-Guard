"""Does the failure predictor GENERALIZE from local qwen runs to a hosted model?

Trains the predictor on local (qwen2.5-coder:14b) trajectories and scores
trajectories a DIFFERENT model (Amazon Nova on Bedrock) produced. Reports two
AUROCs to separate MODEL-transfer from TASK-recognition:

  overlap  : trained on the full local pool (may share task_ids with the test).
  disjoint : any task_id present in the test set is REMOVED from training first,
             so the only thing shared with the test is the trajectory *dynamics*,
             never the task. This is the honest model-transfer number.

Then replays the abort policy on the real Nova run using the DISJOINT predictor,
priced at Nova's actual Bedrock rate.

Usage:
  python predictor/measure_transfer.py predictor/data/mbpp_nova.jsonl [profile]
"""

from __future__ import annotations

import json
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

import pricing
from dataset import encode, load_records
from measure_abort import make_model, replay


def load_file(path):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def inprog_mask(recs):
    """Drop terminal-success steps (outcome already known); keep the rest."""
    return np.array([not (int(r["label"]) == 1 and r["step"] == r["traj_len"] - 1)
                     for r in recs])


def fit_predictor(train, exclude_tasks=None):
    """Logreg P(doomed) on local in-progress steps, optionally dropping any
    task_id in exclude_tasks so train and test share no task."""
    X, y, groups = encode(train)
    keep = inprog_mask(train)
    if exclude_tasks:
        keep = keep & np.array([g not in exclude_tasks for g in groups])
    m = make_model()
    m.fit(X[keep], (y[keep] == 0).astype(int))
    return m, int(keep.sum())


def main(nova_path, profile="nova-lite", input_frac=pricing.INPUT_FRAC):
    train = load_records()
    nova = load_file(nova_path)

    nv_status = {}
    for r in nova:
        nv_status[r["task_id"]] = r["final_status"]
    n_fail = sum(v != "passed" for v in nv_status.values())
    n_pass = len(nv_status) - n_fail
    train_tasks = {r["task_id"] for r in train}
    nova_tasks = set(nv_status)
    overlap = nova_tasks & train_tasks

    print(f"train (local qwen): {len(train)} steps / {len(train_tasks)} tasks")
    print(f"test  (nova)      : {len(nova)} steps / {len(nova_tasks)} tasks  "
          f"({n_pass} pass / {n_fail} fail)")
    print(f"task overlap      : {len(overlap)}/{len(nova_tasks)} test tasks also in training")
    if n_fail < 15:
        print(f"\n** WARNING: only {n_fail} failing tasks -- AUROC is anecdotal. "
              f"Raise --limit and re-harvest for a trustworthy number.")

    Xte, yte, _ = encode(nova)
    te_ip = inprog_mask(nova)
    y_doom = (yte[te_ip] == 0).astype(int)
    two_classes = len(set(y_doom)) >= 2

    m_all, n_all = fit_predictor(train, exclude_tasks=None)
    s_all = m_all.predict_proba(Xte)[:, 1]
    m_dis, n_dis = fit_predictor(train, exclude_tasks=overlap)
    s_dis = m_dis.predict_proba(Xte)[:, 1]

    print()
    if two_classes:
        a_all = roc_auc_score(y_doom, s_all[te_ip])
        a_dis = roc_auc_score(y_doom, s_dis[te_ip])
        print(f"transfer AUROC  overlap  (train {n_all} steps): {a_all:.3f}   [in-dist OOF ~0.806]")
        print(f"transfer AUROC  DISJOINT (train {n_dis} steps, {len(overlap)} tasks removed): {a_dis:.3f}")
        print(f"  -> disjoint is the honest model-transfer number; ~0.5 = no transfer.")
    else:
        print("transfer AUROC: n/a (nova in-progress steps are one class)")

    # Abort replay on the real Nova run, using the DISJOINT (leak-free) predictor.
    base = replay(nova, s_dis, threshold=2.0)
    bp = 100 * base["base_pass"] / base["n"]
    bt = base["base_tok"] / base["n"]
    base_usd = pricing.cost(base["base_tok"], profile, input_frac)
    print(f"\nabort replay on nova (DISJOINT predictor), priced at {profile} "
          f"(baseline {bt:.0f} tok/task, ${base_usd:.4f} over {base['n']} tasks):")
    print(f"{'thresh':>7} | {'aborts':>6} | {'wrong':>5} | {'pass%':>6} | "
          f"{'saved%':>6} | {'$saved':>8}")
    print(f"{'none':>7} | {0:>6} | {0:>5} | {bp:>5.1f}% | {'-':>6} | {'-':>8}")
    for t in (0.70, 0.80, 0.90):
        r = replay(nova, s_dis, t)
        ap = 100 * r["ab_pass"] / r["n"]
        at = r["ab_tok"] / r["n"]
        saved = 100 * (1 - at / bt) if bt else 0.0
        usd = pricing.cost(r["base_tok"] - r["ab_tok"], profile, input_frac)
        print(f"{t:>7.2f} | {r['aborts']:>6} | {r['wrong']:>5} | {ap:>5.1f}% | "
              f"{saved:>5.1f}% | ${usd:>7.4f}")
    print(f"\npositive=doomed; 'wrong'=would-have-passed runs aborted.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python predictor/measure_transfer.py <nova_harvest.jsonl> [profile]")
    prof = sys.argv[2] if len(sys.argv) > 2 else "nova-lite"
    main(sys.argv[1], profile=prof)