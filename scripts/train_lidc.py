"""
Train a concept bottleneck on LIDC nodule cubes.

    cube (1,64,64,64)  ->  Conv3D encoder  ->  concepts  ->  malignant / benign

Everything tunable lives in three JSON configs under experiments/configs/:

    lidc_backbone.json   3D CNN: channels, kernel, batch-norm, dropout
    lidc_cem.json        bottleneck: n_concepts, embedding_dim, hidden_dim, depth
    lidc_train.json      epochs, lr, batch_size, concept_weight, intervention_prob
    lidc_data.json       which nodules, which concepts, HU window, split size

Command-line flags override the configs, so a sweep needs no file edits.

    python scripts/train_lidc.py
    python scripts/train_lidc.py --epochs 60 --runs 3
    python scripts/train_lidc.py --bottleneck categorical --concept-weight 5
    python scripts/train_lidc.py --conv-channels 32 64 128 256 --embedding-dim 32

Splits are BY PATIENT. Nodules from one patient never straddle train and test:
71% of nodules share a patient with another, and most siblings share a label,
so a nodule-level split leaks the scanner rather than testing the model.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import os
import sys

# Same as the other scripts here: make the repo root importable so
# `architectures` and `cbmc` resolve when run as `python scripts/...`.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from architectures.conv3d_cem import BOTTLENECKS, Conv3DBaseline, Conv3DwithCEM
from cbmc.configs import CEMConfig, Conv3DConfig, LIDCDataConfig, TrainConfig

CFG = Path("experiments/configs")


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def get_device(override=None):
    if override: return override
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class MinMaxScaler:
    """Concepts -> [-1, 1] and back. Fit on the training split only."""

    def __init__(self, frame, concepts):
        self.concepts = concepts
        self.lo = frame[concepts].min().values.astype("float32")
        self.hi = frame[concepts].max().values.astype("float32")
        self.span = np.maximum(self.hi - self.lo, 1e-8)

    def forward(self, raw):  return 2.0 * (raw - self.lo) / self.span - 1.0
    def inverse(self, z):    return (z + 1.0) / 2.0 * self.span + self.lo


class LIDCNodules(Dataset):
    def __init__(self, frame, scaler, data_cfg, augment=False):
        self.frame = frame.reset_index(drop=True)
        self.scaler = scaler
        self.cfg = data_cfg
        self.augment = augment
        self.dir = Path(data_cfg.data_dir) / "cubes"

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        row = self.frame.iloc[i]
        cube = np.load(self.dir / row.file).astype(np.float32)
        lo, hi = self.cfg.hu_low, self.cfg.hu_high
        cube = (np.clip(cube, lo, hi) - lo) / (hi - lo)

        if self.augment:
            # Nodules have no canonical orientation, so flips and axis swaps
            # are label-preserving and free.
            for ax in range(3):
                if random.random() < 0.5:
                    cube = np.flip(cube, axis=ax)
            if random.random() < 0.5:
                cube = np.rot90(cube, k=random.randint(1, 3), axes=(0, 1))
            cube = np.ascontiguousarray(cube)

        x = torch.from_numpy(cube).unsqueeze(0)
        c = torch.from_numpy(
            self.scaler.forward(row[self.cfg.concepts].values.astype("float32")))
        return x, c, torch.tensor(int(row.label), dtype=torch.long)


def load_split(data_cfg, seed):
    import pandas as pd
    df = pd.read_csv(Path(data_cfg.data_dir) / "nodules.csv")
    if data_cfg.drop_ambiguous:
        df = df[df.label != -1]
    df = df[df.n_annotations >= data_cfg.min_annotations]

    gss = GroupShuffleSplit(n_splits=1, test_size=data_cfg.test_size, random_state=seed)
    tr, te = next(gss.split(df, groups=df.patient_id))
    train_df, test_df = df.iloc[tr], df.iloc[te]
    assert not (set(train_df.patient_id) & set(test_df.patient_id)), "patient leak!"
    return train_df, test_df


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, scaler, device, concepts):
    model.eval()
    lg_all, y_all, li_all, cp_all, ct_all = [], [], [], [], []
    for x, c, y in loader:
        x, c, y = x.to(device), c.to(device), y.to(device)
        lg, cp = model(x)
        lg_all.append(lg.cpu()); y_all.append(y.cpu())
        if cp is not None:
            li, _ = model(x, interventions=c)
            li_all.append(li.cpu())
            cp_all.append(cp.cpu()); ct_all.append(c.cpu())

    lg, y = torch.cat(lg_all), torch.cat(y_all)
    pred = lg.argmax(1)
    res = {
        "accuracy": (pred == y).float().mean().item(),
        "balanced_accuracy": float(np.mean(
            [(pred[y == k] == k).float().mean().item() for k in (0, 1)])),
    }
    if li_all:
        res["intervention_accuracy"] = (torch.cat(li_all).argmax(1) == y).float().mean().item()
        cp = scaler.inverse(torch.cat(cp_all).numpy())
        ct = scaler.inverse(torch.cat(ct_all).numpy())
        # Averaging raw-unit errors across concepts is dominated by volume
        # (mm^3, up to 1e4). Report the mean of PER-CONCEPT relative errors so
        # the headline number is comparable; per-concept raw MAE is below.
        span = np.maximum(ct.max(0) - ct.min(0), 1e-8)
        res["concept_mae_rel"] = float((np.abs(cp - ct) / span).mean())
        res["concept_mae"] = float(np.abs(cp - ct).mean())
        res["concept_mse"] = float(((cp - ct) ** 2).mean())
        res["per_concept_mae"] = {n: float(v) for n, v
                                  in zip(concepts, np.abs(cp - ct).mean(0))}
    return res


def train_one(seed, backbone_cfg, cem_cfg, train_cfg, data_cfg,
              bottleneck, device, augment, class_weight, verbose=True):
    set_seed(seed)
    train_df, test_df = load_split(data_cfg, seed)
    scaler = MinMaxScaler(train_df, data_cfg.concepts)

    train_loader = DataLoader(
        LIDCNodules(train_df, scaler, data_cfg, augment=augment),
        batch_size=train_cfg.batch_size, shuffle=True,
        num_workers=train_cfg.num_workers)
    test_loader = DataLoader(
        LIDCNodules(test_df, scaler, data_cfg),
        batch_size=train_cfg.batch_size, shuffle=False,
        num_workers=train_cfg.num_workers)

    if bottleneck == "none":
        model = Conv3DBaseline(backbone_cfg).to(device)
    else:
        model = Conv3DwithCEM(backbone_cfg, cem_cfg, bottleneck=bottleneck).to(device)
    n_par = sum(p.numel() for p in model.parameters())

    w = None
    if class_weight:
        counts = np.bincount(train_df.label.values, minlength=2)
        w = torch.tensor(counts.sum() / (2.0 * counts), dtype=torch.float, device=device)
    task_crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)

    if verbose:
        print(f"  train {len(train_df)} nodules / {train_df.patient_id.nunique()} patients"
              f" | test {len(test_df)} / {test_df.patient_id.nunique()}"
              f" | params {n_par:,} ({n_par*4/1e6:.1f} MB)", flush=True)

    for ep in range(1, train_cfg.epochs + 1):
        model.train()
        tot_t = tot_c = correct = seen = 0
        for x, c, y in tqdm(train_loader, desc=f"    epoch {ep}/{train_cfg.epochs}",
                            leave=False):
            x, c, y = x.to(device), c.to(device), y.to(device)
            interv = c if (bottleneck != "none"
                           and random.random() < train_cfg.intervention_prob) else None
            logits, concepts = model(x, interventions=interv)

            l_task = task_crit(logits, y)
            l_conc = (F.mse_loss(concepts, c) if concepts is not None
                      else torch.zeros((), device=device))
            (l_task + train_cfg.concept_weight * l_conc).backward()
            opt.step(); opt.zero_grad()

            tot_t += l_task.item() * y.size(0)
            tot_c += l_conc.detach().item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            seen += y.size(0)
        if verbose and (ep % 10 == 0 or ep == 1 or ep == train_cfg.epochs):
            print(f"    epoch {ep:>3}/{train_cfg.epochs}  task {tot_t/seen:.4f}"
                  f"  concept {tot_c/seen:.4f}  train acc {correct/seen:.3f}", flush=True)

    res = evaluate(model, test_loader, scaler, device, data_cfg.concepts)
    res["params"] = n_par
    res["size_mb"] = n_par * 4 / 1e6
    return res


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bottleneck", default="cem",
                    choices=sorted(BOTTLENECKS) + ["none"],
                    help="'none' = no bottleneck baseline.")
    ap.add_argument("--runs", type=int, default=1, help="Seeds; >1 gives mean +/- std.")
    ap.add_argument("--seed", type=int, default=41)
    # config overrides
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--concept-weight", type=float)
    ap.add_argument("--intervention-prob", type=float)
    ap.add_argument("--conv-channels", type=int, nargs="+")
    ap.add_argument("--embedding-dim", type=int)
    ap.add_argument("--hidden-dim", type=int)
    ap.add_argument("--depth", type=int)
    ap.add_argument("--min-annotations", type=int)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--no-class-weight", action="store_true",
                    help="Disable class balancing (it is on by default).")
    ap.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default="outputs/results/lidc.csv")
    args = ap.parse_args()

    backbone_cfg = Conv3DConfig.load(CFG / "lidc_backbone.json")
    cem_cfg      = CEMConfig.load(CFG / "lidc_cem.json")
    train_cfg    = TrainConfig.load(CFG / "lidc_train.json")
    data_cfg     = LIDCDataConfig.load(CFG / "lidc_data.json")

    for attr, val in [("epochs", args.epochs), ("lr", args.lr),
                      ("batch_size", args.batch_size),
                      ("concept_weight", args.concept_weight),
                      ("intervention_prob", args.intervention_prob)]:
        if val is not None: setattr(train_cfg, attr, val)
    if args.conv_channels:    backbone_cfg.conv_channels = args.conv_channels
    if args.embedding_dim:    cem_cfg.embedding_dim = args.embedding_dim
    if args.hidden_dim:       cem_cfg.hidden_dim = args.hidden_dim
    if args.depth:            cem_cfg.depth = args.depth
    if args.min_annotations:  data_cfg.min_annotations = args.min_annotations
    cem_cfg.n_concepts = len(data_cfg.concepts)

    device = get_device(args.device)
    print(f"device {device} | bottleneck {args.bottleneck} | runs {args.runs}")
    print(f"backbone channels {backbone_cfg.conv_channels} | embedding_dim "
          f"{cem_cfg.embedding_dim} | hidden {cem_cfg.hidden_dim} | depth {cem_cfg.depth}")
    print(f"epochs {train_cfg.epochs} | lr {train_cfg.lr} | batch {train_cfg.batch_size} "
          f"| concept_weight {train_cfg.concept_weight} "
          f"| intervention_prob {train_cfg.intervention_prob}")
    print(f"concepts ({len(data_cfg.concepts)}): {data_cfg.concepts}")
    print(f"min_annotations {data_cfg.min_annotations} | augment {args.augment}\n")

    runs, t0 = [], time.time()
    for r in range(args.runs):
        seed = args.seed + r
        print(f"=== seed {seed} ({r+1}/{args.runs}) ===", flush=True)
        runs.append(train_one(seed, backbone_cfg, cem_cfg, train_cfg, data_cfg,
                              args.bottleneck, device, args.augment,
                              not args.no_class_weight))
        print(f"  accuracy {runs[-1]['accuracy']:.3f}  balanced "
              f"{runs[-1]['balanced_accuracy']:.3f}\n", flush=True)

    keys = [k for k in runs[0] if isinstance(runs[0][k], float) or k == "params"]
    print("=" * 72)
    print(f"{'metric':<26}{'mean':>12}{'std':>12}")
    print("-" * 72)
    summary = {}
    for k in keys:
        v = [r[k] for r in runs]
        summary[k] = (float(np.mean(v)), float(np.std(v)))
        print(f"{k:<26}{summary[k][0]:>12.4f}{summary[k][1]:>12.4f}")
    print("=" * 72)

    if "per_concept_mae" in runs[0]:
        print("\nper-concept MAE (original units), mean over seeds:")
        for n in data_cfg.concepts:
            print(f"  {n:<16}{np.mean([r['per_concept_mae'][n] for r in runs]):>10.3f}")

    out = Path(args.out)
    if args.tag: out = out.with_name(f"{out.stem}_{args.tag}{out.suffix}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bottleneck", "runs", "epochs", "conv_channels", "embedding_dim",
                    "concept_weight", "intervention_prob"]
                   + [f"{k}_{s}" for k in keys for s in ("mean", "std")])
        w.writerow([args.bottleneck, args.runs, train_cfg.epochs,
                    "-".join(map(str, backbone_cfg.conv_channels)),
                    cem_cfg.embedding_dim, train_cfg.concept_weight,
                    train_cfg.intervention_prob]
                   + [v for k in keys for v in summary[k]])
    print(f"\ntotal {time.time()-t0:.0f}s   saved: {out}")


if __name__ == "__main__":
    main()
