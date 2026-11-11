"""
Case 3.5 (CEM, two anchors) vs CEM-Linear when a concept value is ZERO.

CEM-Linear computes  z_i = c_i * phi_i(h),  so a concept equal to 0 zeroes its
whole embedding block. Case 3.5 computes  z_i = w_i e+ + (1-w_i) e-  with
w_i = c_i/2 + 1/2, so c_i = 0 is just the midpoint between the two anchors.

ArithmeticMNIST concepts are normalised as (digit - 5) / 4, so a digit 5 is
EXACTLY 0. This script trains each model and splits the test set by whether a
concept is zero:

    no_zero    neither digit is a 5
    a_zero     digit_a == 5          b_zero     digit_b == 5
    any_zero   at least one digit is a 5

Per group it reports
    task_mse           prediction error, no intervention
    concept_mse        error on all concepts (normalised units)
    zero_concept_mse   error on the concept(s) that are 0 only
    interv_all_mse     task error when ALL concepts are set to the truth
    interv_zero_mse    task error when ONLY the zero concept(s) are corrected
                       (the others stay predicted) -- the direct test of
                       "correct it to 0 and see what happens"

Same configs as run_experiment.py (cem_mnist.json, train_cem_mnist.json).

    python scripts/exp_zero_concept.py                          # cem vs cem_linear, 5 seeds
    python scripts/exp_zero_concept.py --seeds 3 --epochs 10
    python scripts/exp_zero_concept.py --models cem cem_linear cem_linear_raw cem_tanh

Writes
    outputs/results/zero_concept.csv         one row per model x seed x group
    outputs/results/zero_concept_stats.csv   mean / std over seeds
    outputs/models/zero_concept/<model>_seed<N>.pt
"""

import argparse
import csv
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from cbmc.configs import CNNRegressionConfig, CEMConfig, TrainConfig
from cbmc.data.arithmetic_mnist import get_arithmetic_mnist
from architectures.cnn_cem import CNNwithCEM
from architectures.cnn_cem_scaled import CNNwithCEMTanh, CNNwithCEMLinear, CNNwithCEMLinearRaw

MEAN, SCALE = 5.0, 4.0          # (digit - 5) / 4 : 1 -> -1, 5 -> 0, 9 -> +1
MODELS = {
    "cem":            CNNwithCEM,           # Case 3.5 (ours)
    "cem_linear":     CNNwithCEMLinear,
    "cem_linear_raw": CNNwithCEMLinearRaw,
    "cem_tanh":       CNNwithCEMTanh,
}
GROUPS  = ["all", "no_zero", "any_zero", "a_zero", "b_zero"]
METRICS = ["task_mse", "concept_mse", "zero_concept_mse", "interv_all_mse", "interv_zero_mse"]


def get_device(name=None):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def train_one(name, seed, epochs, loaders, device):
    backbone = CNNRegressionConfig.load("experiments/configs/exp_cem_cls_mnist_backbone.json")
    cem_cfg  = CEMConfig.load("experiments/configs/cem_mnist.json")
    tcfg     = TrainConfig.load("experiments/configs/train_cem_mnist.json")
    set_seed(seed)
    model = MODELS[name](backbone, cem_cfg, n_outputs=backbone.n_outputs).to(device)
    opt   = optim.Adam(model.parameters(), lr=tcfg.lr)
    train_loader, _ = loaders
    for ep in range(1, (epochs or tcfg.epochs) + 1):
        model.train()
        tot = 0.0
        for x, c, y in tqdm(train_loader, desc=f"  {name} s{seed} ep{ep}", leave=False):
            x, c, y = x.to(device), c.to(device), y.to(device).unsqueeze(1)
            cn = (c - MEAN) / SCALE
            if random.random() < tcfg.intervention_prob:
                pred, conc = model(x, interventions=cn)
            else:
                pred, conc = model(x)
            task = F.mse_loss(pred, y)
            loss = task + tcfg.concept_weight * F.mse_loss(conc, cn)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += task.item()
        print(f"  {name} seed {seed} epoch {ep}: train mse {tot / len(train_loader):.3f}")
    return model


@torch.no_grad()
def evaluate(model, test_loader, device):
    """Per-sample errors, then averaged per group (so groups are exact, not per-batch)."""
    model.eval()
    per = defaultdict(list)                       # metric -> list of per-sample tensors
    masks = defaultdict(list)                     # group  -> list of bool tensors
    for x, c, y in test_loader:
        x, c, y = x.to(device), c.to(device), y.to(device).unsqueeze(1)
        cn   = (c - MEAN) / SCALE
        zero = (c == MEAN)                        # [B, 2] which concepts are exactly 0
        pred, conc = model(x)
        p_all, _   = model(x, interventions=cn)
        p_zero, _  = model(x, interventions=cn, mask=zero.float())   # fix only the zeros

        se_c = (conc - cn) ** 2
        per["task_mse"].append(((pred - y) ** 2).squeeze(1))
        per["concept_mse"].append(se_c.mean(1))
        nz = zero.sum(1).clamp(min=1)
        per["zero_concept_mse"].append((se_c * zero).sum(1) / nz)       # 0 where no zero; masked out below
        per["interv_all_mse"].append(((p_all - y) ** 2).squeeze(1))
        per["interv_zero_mse"].append(((p_zero - y) ** 2).squeeze(1))

        masks["all"].append(torch.ones(len(x), dtype=torch.bool, device=device))
        masks["no_zero"].append(~zero.any(1))
        masks["any_zero"].append(zero.any(1))
        masks["a_zero"].append(zero[:, 0])
        masks["b_zero"].append(zero[:, 1])

    per   = {k: torch.cat(v) for k, v in per.items()}
    masks = {k: torch.cat(v) for k, v in masks.items()}
    out = {}
    for g in GROUPS:
        m = masks[g]
        row = {"n": int(m.sum())}
        for k in METRICS:
            # zero-concept metrics only make sense where a concept IS zero
            if k in ("zero_concept_mse", "interv_zero_mse") and g in ("all", "no_zero"):
                row[k] = float("nan")
            else:
                row[k] = per[k][m].mean().item() if m.any() else float("nan")
        out[g] = row
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--models", nargs="+", default=["cem", "cem_linear"], choices=list(MODELS))
    ap.add_argument("--seeds", type=int, default=5, help="seeds 41, 42, ...")
    ap.add_argument("--epochs", type=int, default=None, help="default: train_cem_mnist.json (20)")
    ap.add_argument("--operators", nargs="+", default=["+", "x"], choices=["+", "-", "x", "/"])
    ap.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"])
    ap.add_argument("--num-workers", type=int, default=2)
    a = ap.parse_args()

    device = get_device(a.device)
    print(f"device {device}")
    tcfg = TrainConfig.load("experiments/configs/train_cem_mnist.json")
    loaders = get_arithmetic_mnist(batch_size=tcfg.batch_size, num_workers=a.num_workers,
                                   operators=tuple(a.operators))

    os.makedirs("outputs/results", exist_ok=True)
    os.makedirs("outputs/models/zero_concept", exist_ok=True)
    rows = []
    for name in a.models:
        for seed in range(41, 41 + a.seeds):
            model = train_one(name, seed, a.epochs, loaders, device)
            torch.save(model.state_dict(), f"outputs/models/zero_concept/{name}_seed{seed}.pt")
            res = evaluate(model, loaders[1], device)
            for g, r in res.items():
                rows.append({"model": name, "seed": seed, "group": g, **r})
            print(f"  {name} seed {seed}: any_zero interv_zero {res['any_zero']['interv_zero_mse']:.3f}"
                  f" | no_zero task {res['no_zero']['task_mse']:.3f}")

    with open("outputs/results/zero_concept.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    stats = []
    for name in a.models:
        for g in GROUPS:
            sub = [r for r in rows if r["model"] == name and r["group"] == g]
            s = {"model": name, "group": g, "n": sub[0]["n"]}
            for k in METRICS:
                v = np.array([r[k] for r in sub], dtype=float)
                s[f"{k}_mean"] = np.nanmean(v) if np.isfinite(v).any() else float("nan")
                s[f"{k}_std"]  = np.nanstd(v)  if np.isfinite(v).any() else float("nan")
            stats.append(s)
    with open("outputs/results/zero_concept_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stats[0])); w.writeheader(); w.writerows(stats)

    # readable summary
    print(f"\n{'model':16}{'group':10}{'n':>6}" + "".join(f"{k:>18}" for k in METRICS))
    for s in stats:
        print(f"{s['model']:16}{s['group']:10}{s['n']:>6}"
              + "".join(f"{s[k+'_mean']:>10.3f} ±{s[k+'_std']:>6.3f}" for k in METRICS))
    print("\nsaved outputs/results/zero_concept.csv and zero_concept_stats.csv")


if __name__ == "__main__":
    main()
