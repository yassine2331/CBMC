"""
Compare the two arms of test_overfit.sh: does regularisation close the gap?

Reads the result CSVs and the per-epoch histories for the 'plain' and 'reg'
arms and reports the train/test gap for each, which is the number that matters.
Test accuracy alone cannot tell you whether a model is memorising.

    python scripts/compare_overfit.py
    python scripts/compare_overfit.py --suffix _cem
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

ARMS = [("overfit_plain", "PLAIN  (no regularisation)"),
        ("overfit_reg",   "REG    (augment + wd + dropout)")]


def load(tag, results, logs):
    res = Path(results) / f"lidc_{tag}.csv"
    hist = Path(logs) / f"lidc_{tag}_history.csv"
    if not res.exists():
        return None
    r = pd.read_csv(res).iloc[0]
    out = {"test_bal": r.balanced_accuracy_mean, "test_bal_std": r.balanced_accuracy_std,
           "test_acc": r.accuracy_mean, "runs": int(r.runs), "epochs": int(r.epochs)}
    if hist.exists():
        h = pd.read_csv(hist)
        m = h.groupby("epoch").train_acc.mean()
        out["train_acc"] = m.loc[m.index.max()]
        out["train_acc_mid"] = m.loc[m.index.max() // 2]
        out["curve"] = m
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="outputs/results")
    ap.add_argument("--logs-dir", default="outputs/logs")
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()

    arms = {}
    for tag, label in ARMS:
        d = load(tag + args.suffix, args.results_dir, args.logs_dir)
        if d is None:
            print(f"missing: {args.results_dir}/lidc_{tag}{args.suffix}.csv")
        else:
            arms[label] = d
    if len(arms) < 2:
        print("\nNeed both arms. Run:  sbatch test_overfit.sh")
        return

    print(f"{'arm':<34}{'train acc':>11}{'test bal':>11}{'GAP':>9}")
    print("-" * 65)
    for label, d in arms.items():
        tr = d.get("train_acc", float("nan"))
        print(f"{label:<34}{tr:>11.3f}{d['test_bal']:>11.3f}{tr - d['test_bal']:>9.3f}")
    print("-" * 65)

    labels = list(arms)
    g0 = arms[labels[0]]["train_acc"] - arms[labels[0]]["test_bal"]
    g1 = arms[labels[1]]["train_acc"] - arms[labels[1]]["test_bal"]
    t0, t1 = arms[labels[0]]["test_bal"], arms[labels[1]]["test_bal"]

    print(f"\ngap        {g0:.3f} -> {g1:.3f}   ({g1-g0:+.3f})")
    print(f"test bal   {t0:.3f} -> {t1:.3f}   ({t1-t0:+.3f})")

    n = arms[labels[0]]["runs"]
    se = np.sqrt(arms[labels[0]]["test_bal_std"]**2 / n
                 + arms[labels[1]]["test_bal_std"]**2 / n)
    t = (t1 - t0) / se if se > 0 else float("nan")
    print(f"           Welch t = {t:+.2f} over {n} seeds per arm "
          f"({'significant' if abs(t) > 2 else 'not significant'})")

    print("\nreading:")
    if g1 < g0 - 0.05 and t1 >= t0 - 0.01:
        print("  Regularisation closed the gap without costing test accuracy.")
        print("  Re-run the full comparison with --augment and --weight-decay.")
    elif g1 < g0 - 0.05:
        print("  Gap closed but test accuracy dropped: now under-fitting.")
        print("  Ease off -- lower weight decay or dropout.")
    else:
        print("  Gap barely moved. Regularisation is not the binding constraint.")
        print("  Next levers: smaller encoder (--conv-channels), early stopping,")
        print("  or accept that 944 volumes is too few for this capacity.")

    if all("curve" in d for d in arms.values()):
        print("\ntrain accuracy by epoch:")
        c0, c1 = arms[labels[0]]["curve"], arms[labels[1]]["curve"]
        pts = [e for e in (1, 10, 25, 50, 75, 100) if e in c0.index and e in c1.index]
        print(f"  {'epoch':>6}{'plain':>9}{'reg':>9}")
        for e in pts:
            print(f"  {e:>6}{c0[e]:>9.3f}{c1[e]:>9.3f}")


if __name__ == "__main__":
    main()
