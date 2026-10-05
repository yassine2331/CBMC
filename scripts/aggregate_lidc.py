"""
Merge the per-variant LIDC result CSVs into one comparison table.

run_lidc_all.sh writes outputs/results/lidc_<variant>.csv per experiment; this
stacks them, orders them the way the paper will, and prints mean +/- std.

    python scripts/aggregate_lidc.py
    python scripts/aggregate_lidc.py --suffix _e60
"""

import argparse
import csv
import glob
import os

ORDER = ["nn", "cbm", "cem_binary", "categorical", "cem_linear_norm",
         "cem_linear_raw", "cem"]
NAMES = {
    "nn":               "Neural net (no bottleneck)",
    "cbm":              "CBM (scalar concepts)",
    "cem_binary":       "CEM binary (high/low)",
    "categorical":      "Categorical (bins)",
    "cem_linear_norm":  "CEM-Linear (normalised)",
    "cem_linear_raw":   "CEM-Linear (raw values)",
    "cem":              "CEM continuous (ours)",
}
METRICS = [("accuracy", "accuracy"), ("balanced_accuracy", "balanced acc"),
           ("intervention_accuracy", "interv acc"), ("concept_accuracy", "concept acc"),
           ("concept_mae_rel", "concept MAE rel"), ("params", "params")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="outputs/results")
    ap.add_argument("--suffix", default="", help="e.g. _e60 if runs were tagged that way")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = {}
    for key in ORDER:
        path = os.path.join(args.results_dir, f"lidc_{key}{args.suffix}.csv")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            r = next(csv.DictReader(f), None)
        if r:
            rows[key] = r

    if not rows:
        print(f"No lidc_*{args.suffix}.csv files in {args.results_dir}.")
        print("Run:  sbatch run_lidc_all.sh")
        return

    W = 28
    head = f"{'model':<{W}}" + "".join(f"{lbl:>19}" for _, lbl in METRICS)
    print("=" * len(head)); print(head); print("-" * len(head))
    for key in ORDER:
        if key not in rows:
            continue
        r = rows[key]
        line = f"{NAMES[key]:<{W}}"
        for m, _ in METRICS:
            mean, std = r.get(f"{m}_mean", ""), r.get(f"{m}_std", "")
            if mean in ("", None):
                line += f"{'--':>19}"
            elif m == "params":
                line += f"{int(float(mean)):>19,}"
            else:
                line += f"{float(mean):>12.4f} +/-{float(std):>5.3f}"
        print(line)
    print("=" * len(head))

    missing = [k for k in ORDER if k not in rows]
    if missing:
        print(f"\nnot found: {', '.join(missing)}")

    out = args.out or os.path.join(args.results_dir, f"lidc_summary{args.suffix}.csv")
    cols = sorted({c for r in rows.values() for c in r})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant"] + cols)
        w.writeheader()
        for key in ORDER:
            if key in rows:
                w.writerow({"variant": NAMES[key], **rows[key]})
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
