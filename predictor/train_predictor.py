"""Train and persist the final failure predictor (feature logreg).

The 3-regime + 14B LoRA ablation (predictor/train_lora.py) confirmed a
14-feature logistic regression is the best predictor at this data scale, so it
is what we ship. This fits the honest in-progress model, reports held-out AUROC
and a precision/recall table for choosing the abort threshold, then refits on
ALL in-progress data and saves the artifact to predictor/predictor.joblib.
"""

from __future__ import annotations

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dataset import FAILURE_TYPES, FEATURE_COLS, encode, load_records

ARTIFACT = "predictor/predictor.joblib"
SEED = 42


def make_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )


def in_progress(recs):
    keep = [not (int(r["label"]) == 1 and r["step"] == r["traj_len"] - 1) for r in recs]
    return [r for r, k in zip(recs, keep) if k]


def main():
    recs = in_progress(load_records())
    X, y, groups = encode(recs)
    steps = np.array([int(r["step"]) for r in recs])
    yfail = (y == 0).astype(int)  # positive = doomed (will NOT pass)
    print(f"in-progress records: {len(recs)}  "
          f"(doomed={int(yfail.sum())}, will-recover={int((yfail == 0).sum())})")

    # honest held-out split to report AUROC + choose the abort threshold
    tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(X, yfail, groups))
    m = make_model()
    m.fit(X[tr], yfail[tr])
    p = m.predict_proba(X[te])[:, 1]
    early = steps[te] <= 2
    print(f"held-out AUROC failing={roc_auc_score(yfail[te], p):.3f}   "
          f"early(step<=2)={roc_auc_score(yfail[te][early], p[early]):.3f}")

    print("\nthreshold | precision(doomed) | recall(doomed) | abort-rate")
    for t in (0.5, 0.6, 0.7, 0.8, 0.9):
        pred = (p >= t).astype(int)
        pr, rc, _, _ = precision_recall_fscore_support(
            yfail[te], pred, labels=[1], zero_division=0)
        print(f"   {t:.2f}    |       {pr[0]:.3f}       |      {rc[0]:.3f}     |    {pred.mean():.3f}")

    # production artifact: refit on ALL in-progress data
    prod = make_model()
    prod.fit(X, yfail)
    bundle = {
        "model": prod,
        "feature_cols": FEATURE_COLS,
        "failure_types": FAILURE_TYPES,
        "positive_class": "doomed",
        "threshold": 0.80,  # default; tune from the table above
    }
    joblib.dump(bundle, ARTIFACT)
    print(f"\nsaved -> {ARTIFACT}  (positive=doomed, default threshold=0.80)")


if __name__ == "__main__":
    main()