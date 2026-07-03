"""H3: does early abort cut tokens -- and dollars -- without hurting pass rate?

Leak-free counterfactual replay on the logged trajectories. Each trajectory is
scored by a logreg trained on OTHER tasks (out-of-fold GroupKFold), then the
abort policy is applied -- abort at the first in-progress step with
P(doomed) >= threshold and step >= MIN_STEP. We compare tokens and pass rate
against the full no-abort run, sweeping the threshold to show the operating
curve, and price the saved tokens at a chosen backend's rate (see pricing.py).
Tokens are cumulative per step in the logs, so aborting at step k costs
tokens_used[k] and saves the rest.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import pricing
from dataset import encode, load_records

MIN_STEP = 2
N_FOLDS = 5


def make_model():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=1000, class_weight="balanced"))


def oof_scores(recs):
    """Out-of-fold P(doomed) for every record: trained on the in-progress steps
    of OTHER tasks, scored on this task. No trajectory scores itself."""
    X, y, groups = encode(recs)
    yfail = (y == 0).astype(int)
    inprog = np.array([not (int(r["label"]) == 1 and r["step"] == r["traj_len"] - 1)
                       for r in recs])
    scores = np.full(len(recs), np.nan)
    uniq = np.array(sorted(set(groups)))
    for _, te_g in GroupKFold(n_splits=N_FOLDS).split(uniq, groups=uniq):
        test_tasks = set(uniq[te_g])
        te = np.array([g in test_tasks for g in groups])
        tr = (~te) & inprog
        m = make_model()
        m.fit(X[tr], yfail[tr])
        scores[te] = m.predict_proba(X[te])[:, 1]
    return scores


def replay(recs, scores, threshold, min_step=MIN_STEP):
    by = defaultdict(list)
    for i, r in enumerate(recs):
        by[r["task_id"]].append((int(r["step"]), i, r))

    n = aborts = wrong = 0
    base_tok = ab_tok = 0.0
    base_pass = ab_pass = 0
    for _, rows in by.items():
        rows.sort(key=lambda x: x[0])
        rs = [r for _, _, r in rows]
        passed = rs[-1]["final_status"] == "passed"
        pass_step = int(rs[-1]["step"]) if passed else None
        final_tok = float(rs[-1].get("tokens_used", 0))

        ab_step, ab_tok_at = None, final_tok
        for s, i, r in rows:
            if s >= min_step and scores[i] >= threshold:
                ab_step, ab_tok_at = s, float(r.get("tokens_used", 0))
                break

        n += 1
        base_tok += final_tok
        base_pass += int(passed)
        if ab_step is not None and (pass_step is None or ab_step < pass_step):
            ab_tok += ab_tok_at
            aborts += 1
            wrong += int(passed)              # aborted a run that would have passed
        else:
            ab_tok += final_tok
            ab_pass += int(passed)
    return dict(n=n, aborts=aborts, wrong=wrong,
                base_tok=base_tok, ab_tok=ab_tok,
                base_pass=base_pass, ab_pass=ab_pass)


def main(profile="nova-lite", input_frac=pricing.INPUT_FRAC):
    recs = load_records()
    print(f"records: {len(recs)}   tasks: {len({r['task_id'] for r in recs})}")
    scores = oof_scores(recs)

    rate_m = pricing.rate_per_token(profile, input_frac) * 1_000_000
    print(f"pricing: {profile}  (~${rate_m:.3f}/1M tok blended @ {input_frac:.0%} input)")

    base = replay(recs, scores, threshold=2.0)   # 2.0 > any prob => never abort
    bp = 100 * base["base_pass"] / base["n"]
    bt = base["base_tok"] / base["n"]
    base_usd = pricing.cost(base["base_tok"], profile, input_frac)
    print(f"baseline: {bp:.1f}% pass   {bt:.0f} tok/task   "
          f"${base_usd:.4f} over {base['n']} tasks (never abort)")

    print(f"\n{'thresh':>7} | {'aborts':>6} | {'wrong':>5} | {'pass%':>6} | "
          f"{'dpass':>6} | {'tok/task':>8} | {'saved%':>6} | {'$saved':>8}")
    print(f"{'none':>7} | {0:>6} | {0:>5} | {bp:>5.1f}% | {'-':>6} | "
          f"{bt:>8.0f} | {'-':>6} | {'-':>8}")

    rows = {}
    for t in (0.70, 0.80, 0.90):
        r = replay(recs, scores, t)
        rows[t] = r
        ap = 100 * r["ab_pass"] / r["n"]
        at = r["ab_tok"] / r["n"]
        usd_saved = pricing.cost(r["base_tok"] - r["ab_tok"], profile, input_frac)
        print(f"{t:>7.2f} | {r['aborts']:>6} | {r['wrong']:>5} | {ap:>5.1f}% | "
              f"{ap - bp:>+5.1f}% | {at:>8.0f} | {100 * (1 - at / bt):>5.1f}% | "
              f"${usd_saved:>7.4f}")

    r = rows[0.90]
    usd90 = pricing.cost(r["base_tok"] - r["ab_tok"], profile, input_frac)
    pct90 = 100 * (1 - (r["ab_tok"] / r["n"]) / bt)
    print(f"\nheadline @0.90: {pct90:.1f}% tokens saved = ${usd90:.4f} at {profile} "
          f"rates over {r['n']} tasks, {r['wrong']} wrongful abort(s).")
    print(f"positive=doomed; 'wrong'=runs aborted that would have passed; min_step={MIN_STEP}")
    print(f"$saved blends input/output at {input_frac:.0%} input "
          f"(logs store only total tokens); switch backends with --pricing.")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--pricing", default="nova-lite", choices=sorted(pricing.PRICES),
                    help="which backend's rates to cost the saved tokens at")
    ap.add_argument("--input-frac", type=float, default=pricing.INPUT_FRAC,
                    help="assumed input-token fraction (logs store totals only)")
    args = ap.parse_args()
    main(profile=args.pricing, input_frac=args.input_frac)