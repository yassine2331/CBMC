"""
Concepts-only baseline: how far can you get WITHOUT the scan?

Trains classifiers that see only the annotated concepts -- no image, no CNN --
and predict malignancy. This is the ceiling for any concept bottleneck on this
dataset: a bottleneck forces every prediction through the concepts, so with a
perfect concept extractor it could do exactly this well and no better.

Two useful reference points come out of it:

  * the accuracy a CBM/CEM should approach as its concept error goes to zero
  * what its INTERVENTION accuracy should approach, since intervening supplies
    the true concepts, which is precisely this setting

Models: XGBoost, random forest, logistic regression and a small MLP, each on
raw concept values and on min-max normalised ones, so the effect of scaling is
visible. Tree models are scale-invariant and should be unchanged; the MLP and
logistic regression are not.

    python scripts/concept_only_baseline.py
    python scripts/concept_only_baseline.py --runs 10 --min-annotations 1

Splits are by patient, matching scripts/train_lidc.py.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# NOTE: this script deliberately does NOT import torch. Importing torch and
# xgboost into the same process loads two OpenMP runtimes and segfaults on
# macOS, and `from cbmc.configs import ...` pulls torch in via cbmc/__init__.
# The config is plain JSON, so we read it directly.
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False


def models(seed):
    m = {}
    if HAVE_XGB:
        m["XGBoost"] = lambda: XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=seed,
            eval_metric="logloss", verbosity=0)
    m["Random forest"] = lambda: RandomForestClassifier(
        n_estimators=300, max_depth=8, random_state=seed,
        class_weight="balanced", n_jobs=-1)
    m["Logistic regression"] = lambda: LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=seed)
    m["MLP (64, 32)"] = lambda: MLPClassifier(
        hidden_layer_sizes=(64, 32), max_iter=1500, random_state=seed,
        early_stopping=True, n_iter_no_change=30)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=5, help="Seeds (also reseeds the split).")
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--min-annotations", type=int, default=None)
    ap.add_argument("--out", default="outputs/results/lidc_concepts_only.csv")
    args = ap.parse_args()

    with open("experiments/configs/lidc_data.json") as f:
        cfg = json.load(f)
    if args.min_annotations is not None:
        cfg["min_annotations"] = args.min_annotations

    df = pd.read_csv(Path(cfg["data_dir"]) / "nodules.csv")
    if cfg["drop_ambiguous"]:
        df = df[df.label != -1]
    df = df[df.n_annotations >= cfg["min_annotations"]]
    CON = cfg["concepts"]

    print(f"{len(df)} nodules / {df.patient_id.nunique()} patients  "
          f"(min_annotations {cfg['min_annotations']})")
    print(f"{len(CON)} concepts: {CON}")
    print(f"class balance: {(df.label==0).mean():.1%} benign / {(df.label==1).mean():.1%} malignant")
    print(f"XGBoost: {'available' if HAVE_XGB else 'NOT INSTALLED (pip install xgboost)'}\n")

    results = {}
    for r in range(args.runs):
        seed = args.seed + r
        tr_i, te_i = next(GroupShuffleSplit(
            1, test_size=cfg["test_size"], random_state=seed).split(df, groups=df.patient_id))
        tr, te = df.iloc[tr_i], df.iloc[te_i]
        assert not (set(tr.patient_id) & set(te.patient_id))

        Xtr_raw = tr[CON].values.astype("float64")
        Xte_raw = te[CON].values.astype("float64")
        ytr, yte = tr.label.values, te.label.values

        # min-max to [-1, 1], fitted on train only -- same as the CNN pipeline
        lo, hi = Xtr_raw.min(0), Xtr_raw.max(0)
        span = np.maximum(hi - lo, 1e-8)
        Xtr_nrm = 2 * (Xtr_raw - lo) / span - 1
        Xte_nrm = 2 * (Xte_raw - lo) / span - 1

        for scaling, Xtr, Xte in [("raw", Xtr_raw, Xte_raw),
                                  ("normalised", Xtr_nrm, Xte_nrm)]:
            for name, make in models(seed).items():
                clf = make().fit(Xtr, ytr)
                pred = clf.predict(Xte)
                prob = (clf.predict_proba(Xte)[:, 1]
                        if hasattr(clf, "predict_proba") else pred)
                results.setdefault((name, scaling), []).append({
                    "accuracy": float((pred == yte).mean()),
                    "balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
                    "auc": float(roc_auc_score(yte, prob)),
                })

    W = 22
    print("=" * 86)
    print(f"{'model':<{W}}{'concepts':>13}{'accuracy':>17}{'balanced acc':>17}{'AUC':>17}")
    print("-" * 86)
    order = (["XGBoost"] if HAVE_XGB else []) + \
            ["Random forest", "Logistic regression", "MLP (64, 32)"]
    for name in order:
        for scaling in ("raw", "normalised"):
            runs = results.get((name, scaling))
            if not runs:
                continue
            line = f"{name:<{W}}{scaling:>13}"
            for m in ("accuracy", "balanced_accuracy", "auc"):
                v = [x[m] for x in runs]
                line += f"{np.mean(v):>10.4f} +/-{np.std(v):>5.3f}"
            print(line)
    print("=" * 86)
    print(f"\n{args.runs} seeds; the split is reseeded each run, so the spread "
          f"includes split variance.")
    print("This is the CEILING for a concept bottleneck on this dataset: it is what")
    print("a model achieves when the concepts are perfect. Compare it against the")
    print("intervention accuracy reported by scripts/train_lidc.py.")

    # feature importance from the strongest tree model
    if HAVE_XGB:
        tr_i, te_i = next(GroupShuffleSplit(
            1, test_size=cfg["test_size"], random_state=args.seed).split(df, groups=df.patient_id))
        clf = models(args.seed)["XGBoost"]().fit(
            df.iloc[tr_i][CON].values, df.iloc[tr_i].label.values)
        imp = clf.feature_importances_
        print("\nXGBoost feature importance (which concepts carry the signal):")
        for i in np.argsort(imp)[::-1]:
            print(f"  {CON[i]:<16}{imp[i]:>7.3f}  " + "#" * int(imp[i] * 100))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "concepts", "runs", "accuracy_mean", "accuracy_std",
                    "balanced_accuracy_mean", "balanced_accuracy_std",
                    "auc_mean", "auc_std"])
        for (name, scaling), runs in results.items():
            row = [name, scaling, len(runs)]
            for m in ("accuracy", "balanced_accuracy", "auc"):
                v = [x[m] for x in runs]
                row += [float(np.mean(v)), float(np.std(v))]
            w.writerow(row)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
