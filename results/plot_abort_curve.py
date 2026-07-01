"""Plot the H3 abort operating curve -> results/abort_curve.png.

Recomputes the leak-free replay so the figure always matches the data, then
draws tokens-saved and pass-rate-change vs abort threshold. Needs matplotlib.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "predictor"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from measure_abort import load_records, oof_scores, replay


def main():
    recs = load_records()
    scores = oof_scores(recs)
    base = replay(recs, scores, threshold=2.0)
    bp = 100 * base["base_pass"] / base["n"]
    bt = base["base_tok"] / base["n"]

    ths = [0.70, 0.80, 0.90]
    saved, dpass = [], []
    for t in ths:
        r = replay(recs, scores, t)
        saved.append(100 * (1 - (r["ab_tok"] / r["n"]) / bt))
        dpass.append(100 * r["ab_pass"] / r["n"] - bp)

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(ths, saved, "o-", color="#22aa77", label="tokens saved")
    ax1.set_xlabel("abort threshold  P(doomed)")
    ax1.set_ylabel("tokens saved (%)", color="#22aa77")
    ax1.set_ylim(0, max(saved) * 1.25)
    ax1.invert_xaxis()  # aggressive (low threshold) on the left

    ax2 = ax1.twinx()
    ax2.plot(ths, dpass, "s--", color="#cc4444", label="pass-rate change")
    ax2.set_ylabel("pass-rate change (pts)", color="#cc4444")
    ax2.set_ylim(min(dpass) * 1.5 - 0.2, 0.2)

    ax1.set_title("Early-abort operating curve (H3)")
    fig.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/abort_curve.png", dpi=150, bbox_inches="tight")
    print("saved -> results/abort_curve.png")


if __name__ == "__main__":
    main()