"""
Aggregate every per-experiment stats CSV into one summary table.

Each run of `scripts/run_experiment.py --runs N` writes
`outputs/results/<exp>[_<tag>]_stats.csv` in long form:

    metric,mean,std
    task_mse,46.716919,4.032855
    ...

This script pivots all of them into a single wide table:

    rows    : experiment type (task_dataset [tag]) x metric
    columns : model (Baseline, CBM, CEM, CEM-Tanh, CEM-Linear), mean and std

Two files are written:
  summary_all_experiments.csv         mean/std in separate columns (machine-readable)
  summary_all_experiments_pretty.csv  "mean +/- std" in one cell per model (for reading)

Usage:
    python scripts/aggregate_results.py
    python scripts/aggregate_results.py --results-dir outputs/results --out outputs/results
"""

from __future__ import annotations

import argparse
import csv
import glob
import os

# Tokens that mark where the model prefix ends and the experiment begins.
TASKS    = ("gen", "cls")
DATASETS = ("mnist", "pendulum")

# Display name + column order for models. Anything unknown is appended alphabetically.
MODEL_ORDER = ["baseline", "cbm", "cem", "cem_tanh", "cem_linear", "cem_linear_raw"]
MODEL_NAMES = {
    "baseline":   "Baseline",
    "cbm":        "CBM",
    "cem":        "CEM",
    "cem_tanh":   "CEM-Tanh",
    "cem_linear": "CEM-Linear",
    "cem_linear_raw": "CEM-Linear-Raw",
}

# Row order within one experiment. Unknown metrics are appended alphabetically.
METRIC_ORDER = ["task_mse", "concept_mse", "intervention_mse", "test_map",
                "recon_loss", "kl"]
METRIC_NAMES = {
    "task_mse":         "Task MSE",
    "concept_mse":      "Concept MSE",
    "intervention_mse": "Intervention MSE",
    "test_map":         "Test MAP",
    "recon_loss":       "Recon Loss",   # generation experiments
    "kl":               "KL",           # generation experiments
}

EXP_ORDER = ["cls_mnist", "cls_pendulum", "gen_mnist", "gen_pendulum"]

MISSING = ""   # what to write when a model was never run on that experiment


def parse_exp_name(name: str):
    """
    Split an experiment key into (model, experiment, tag).

        exp_cls_mnist_all_digits      -> ("baseline",   "cls_mnist", "all_digits")
        exp_cem_tanh_cls_mnist_digit9 -> ("cem_tanh",   "cls_mnist", "digit9")
        exp_cbm_gen_pendulum          -> ("cbm",        "gen_pendulum", "")

    Returns None if the name does not look like an experiment key.
    """
    if name.startswith("exp_"):
        name = name[len("exp_"):]
    parts = name.split("_")

    # The task token ("gen"/"cls") separates the model prefix from the experiment.
    task_i = next((i for i, p in enumerate(parts) if p in TASKS), None)
    if task_i is None or task_i + 1 >= len(parts):
        return None
    if parts[task_i + 1] not in DATASETS:
        return None

    model = "_".join(parts[:task_i]) or "baseline"
    exp   = f"{parts[task_i]}_{parts[task_i + 1]}"
    tag   = "_".join(parts[task_i + 2:])
    return model, exp, tag


def read_stats_file(path: str):
    """Yield (metric, mean, std) from a `metric,mean,std` stats CSV."""
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            metric = (row.get("metric") or "").strip()
            if metric:
                yield metric, (row.get("mean") or "").strip(), (row.get("std") or "").strip()


def collect(results_dir: str):
    """
    Walk the results directory and build {(exp, tag, metric): {model: (mean, std)}}.
    Also records which models and which (exp, tag) pairs were actually seen.
    """
    table:  dict[tuple[str, str, str], dict[str, tuple[str, str]]] = {}
    models: set[str] = set()
    exps:   set[tuple[str, str]] = set()

    # 1. Per-experiment stats files written by run_experiment.py --runs N.
    for path in sorted(glob.glob(os.path.join(results_dir, "*_stats.csv"))):
        key = os.path.basename(path)[: -len("_stats.csv")]
        parsed = parse_exp_name(key)
        if parsed is None:
            print(f"  skipped (unrecognised name): {os.path.basename(path)}")
            continue
        model, exp, tag = parsed
        models.add(model)
        exps.add((exp, tag))
        for metric, mean, std in read_stats_file(path):
            table.setdefault((exp, tag, metric), {})[model] = (mean, std)

    # 2. The combined file written by run_quantitative_suite (no --exp given),
    #    whose first column is "<exp_name>_<metric>" rather than a bare metric.
    combined = os.path.join(results_dir, "quantitative_summary.csv")
    if os.path.exists(combined):
        for raw, mean, std in read_stats_file(combined):
            metric = next((m for m in METRIC_ORDER if raw.endswith("_" + m)), None)
            if metric is None:
                continue
            parsed = parse_exp_name(raw[: -(len(metric) + 1)])
            if parsed is None:
                continue
            model, exp, tag = parsed
            models.add(model)
            exps.add((exp, tag))
            # Don't clobber a dedicated stats file, which is the more specific source.
            table.setdefault((exp, tag, metric), {}).setdefault(model, (mean, std))

    return table, models, exps


def sort_key(order: list[str]):
    """Rank items by a known order, pushing anything unknown to the end alphabetically."""
    return lambda x: (order.index(x), "") if x in order else (len(order), x)


def build_rows(table, models, exps):
    """Return (header_rows, data_rows) for the machine-readable CSV."""
    model_cols = sorted(models, key=sort_key(MODEL_ORDER))

    top    = ["", ""] + [c for m in model_cols for c in (MODEL_NAMES.get(m, m), "")]
    header = ["experiment", "metric"] + ["mean", "std"] * len(model_cols)

    rows = []
    for exp, tag in sorted(exps, key=lambda et: (sort_key(EXP_ORDER)(et[0]), et[1])):
        metrics = [m for (e, t, m) in table if (e, t) == (exp, tag)]
        label   = f"{exp} [{tag}]" if tag else exp
        for metric in sorted(set(metrics), key=sort_key(METRIC_ORDER)):
            cells = table[(exp, tag, metric)]
            row   = [label, METRIC_NAMES.get(metric, metric)]
            for m in model_cols:
                mean, std = cells.get(m, (MISSING, MISSING))
                row += [mean, std]
            rows.append(row)
    return [top, header], rows, model_cols


def build_pretty_rows(rows, model_cols):
    """Collapse each model's (mean, std) pair into a single 'mean +/- std' cell."""
    header = ["experiment", "metric"] + [MODEL_NAMES.get(m, m) for m in model_cols]
    out = []
    for row in rows:
        pretty = row[:2]
        for i in range(len(model_cols)):
            mean, std = row[2 + 2 * i], row[3 + 2 * i]
            if mean == MISSING:
                pretty.append("--")
            else:
                pretty.append(f"{float(mean):.4f} +/- {float(std):.4f}")
        out.append(pretty)
    return header, out


def write_csv(path, header_rows, data_rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerows(header_rows)
        w.writerows(data_rows)
    print(f"  -> saved summary: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="outputs/results",
                        help="Directory holding the per-experiment *_stats.csv files.")
    parser.add_argument("--out", default=None,
                        help="Directory to write the summary into. Default: --results-dir.")
    parser.add_argument("--name", default="summary_all_experiments",
                        help="Base filename for the two summary CSVs.")
    args = parser.parse_args()

    out_dir = args.out or args.results_dir

    table, models, exps = collect(args.results_dir)
    if not table:
        print(f"No *_stats.csv files found in {args.results_dir} — nothing to aggregate.")
        print("Run an experiment with --runs > 1 first, e.g.:")
        print("  python scripts/run_experiment.py --exp exp_cem_cls_mnist --runs 5")
        return

    header_rows, rows, model_cols = build_rows(table, models, exps)
    write_csv(os.path.join(out_dir, f"{args.name}.csv"), header_rows, rows)

    p_header, p_rows = build_pretty_rows(rows, model_cols)
    write_csv(os.path.join(out_dir, f"{args.name}_pretty.csv"), [p_header], p_rows)

    print(f"\n{len(rows)} rows x {len(model_cols)} models "
          f"({', '.join(MODEL_NAMES.get(m, m) for m in model_cols)})")


if __name__ == "__main__":
    main()
