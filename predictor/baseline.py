"""Honest feature baseline for the failure predictor.

The predictor only matters on IN-PROGRESS (failing) steps: given the agent has
not yet passed, will it eventually pass? So we drop the terminal success step of
passing trajectories (predicting "already passed" is trivial) and evaluate only
where a real abort decision is made.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dataset import encode, load_records


def auroc(yt, p):
    return roc_auc_score(yt, p) if len(np.unique(yt)) == 2 else float("nan")


def main():
    recs = load_records()

    keep = [not (int(r["label"]) == 1 and r["step"] == r["traj_len"] - 1) for r in recs]
    recs = [r for r, k in zip(recs, keep) if k]

    X, y, groups = encode(recs)
    steps = np.array([int(r["step"]) for r in recs])
    yfail = (y == 0).astype(int)  # positive = doomed (will NOT pass)

    print(f"in-progress records: {len(recs)}  "
          f"(doomed={int(yfail.sum())}, will-recover={int((yfail == 0).sum())})")

    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(X, y, groups))
    models = {
        "logreg": make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=1000, class_weight="balanced")),
        "gboost": GradientBoostingClassifier(random_state=42),
    }
    for name, model in models.items():
        model.fit(X[tr], yfail[tr])
        p = model.predict_proba(X[te])[:, 1]
        yt = yfail[te]
        early = steps[te] <= 2
        print(f"{name:8} AUROC failing={auroc(yt, p):.3f}   "
              f"early(step<=2)={auroc(yt[early], p[early]):.3f}   "
              f"(n_te={len(yt)}, n_early={int(early.sum())})")

    # stability check: recover class is tiny (39), verify across folds
    alls, earlys = [], []
    for trf, tef in GroupKFold(n_splits=5).split(X, yfail, groups):
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=1000, class_weight="balanced"))
        m.fit(X[trf], yfail[trf])
        pf = m.predict_proba(X[tef])[:, 1]
        ef = steps[tef] <= 2
        a, e = auroc(yfail[tef], pf), auroc(yfail[tef][ef], pf[ef])
        if not np.isnan(a):
            alls.append(a)
        if not np.isnan(e):
            earlys.append(e)
    print(f"5-fold CV: failing={np.mean(alls):.3f}±{np.std(alls):.3f}   "
          f"early={np.mean(earlys):.3f}±{np.std(earlys):.3f}  (folds {len(alls)}/{len(earlys)})")


if __name__ == "__main__":
    main()