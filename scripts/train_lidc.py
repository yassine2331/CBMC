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

from architectures.conv3d_cem import (BOTTLENECKS, Conv3DBaseline,
                                      Conv3DwithCategorical, Conv3DwithCBM,
                                      Conv3DwithCEM)
from cbmc.configs import CEMConfig, Conv3DConfig, LIDCDataConfig, TrainConfig
from cbmc.data import concept_transforms as CT

CFG = Path("experiments/configs")

# Every column of nodules.csv usable as a concept with --concepts. malignancy is
# left out on purpose: the label is derived from it.
ALL_CONCEPTS = ["subtlety", "sphericity", "margin", "lobulation", "spiculation",
                "texture", "diameter", "volume", "surface_area",
                "internalStructure", "calcification"]


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

class LIDCNodules(Dataset):
    def __init__(self, frame, transform, data_cfg, augment=False, jitter=3):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.cfg = data_cfg
        self.augment = augment
        self.jitter = jitter
        self.dir = Path(data_cfg.data_dir) / "cubes"

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        row = self.frame.iloc[i]
        cube = np.load(self.dir / row.file).astype(np.float32)
        lo, hi = self.cfg.hu_low, self.cfg.hu_high
        cube = (np.clip(cube, lo, hi) - lo) / (hi - lo)

        if self.augment:
            # Only ISOMETRIES are allowed. Three of the concepts are diameter,
            # volume and surface area, so anything that rescales or deforms the
            # cube would silently corrupt their labels. Axis permutation plus
            # per-axis flips covers all 48 symmetries of a cube exactly, with
            # no interpolation; a small roll removes the "always dead centre"
            # shortcut without changing any geometry.
            cube = np.transpose(cube, random.sample(range(3), 3))
            for ax in range(3):
                if random.random() < 0.5:
                    cube = np.flip(cube, axis=ax)
            if self.jitter:
                cube = np.roll(cube, [random.randint(-self.jitter, self.jitter)
                                      for _ in range(3)], axis=(0, 1, 2))
            cube = np.ascontiguousarray(cube)

        x = torch.from_numpy(cube).unsqueeze(0)
        raw = row[self.cfg.concepts].values.astype("float32")
        t = self.transform.forward(raw)
        c = (torch.from_numpy(t).long() if self.transform.kind == "categorical"
             else torch.from_numpy(t).float())
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
def evaluate(model, loader, transform, device, concepts, raw_test):
    model.eval()
    lg_all, y_all, li_all, cp_all, ct_all = [], [], [], [], []
    for x, c, y in loader:
        x, c, y = x.to(device), c.to(device), y.to(device)
        lg, cp = model(x)
        lg_all.append(lg.cpu()); y_all.append(y.cpu())
        if cp is not None:
            li, _ = model(x, interventions=c)
            li_all.append(li.cpu())
            ct_all.append(c.cpu())
            cp_all.append([t.cpu() for t in cp] if isinstance(cp, list) else cp.cpu())

    lg, y = torch.cat(lg_all), torch.cat(y_all)
    pred = lg.argmax(1)
    res = {
        "accuracy": (pred == y).float().mean().item(),
        "balanced_accuracy": float(np.mean(
            [(pred[y == k] == k).float().mean().item() for k in (0, 1)])),
    }
    if not li_all:
        return res
    res["intervention_accuracy"] = (torch.cat(li_all).argmax(1) == y).float().mean().item()

    kind = transform.kind
    if kind == "categorical":
        # cp_all is a list of per-batch lists of [B, n_i] logits
        n_con = len(concepts)
        states = np.stack([
            torch.cat([b[i] for b in cp_all]).argmax(1).numpy() for i in range(n_con)
        ], axis=1)
        truth = torch.cat(ct_all).numpy()
        res["concept_accuracy"] = float((states == truth).mean())
        cp_raw = transform.inverse(states)
    else:
        z = torch.cat(cp_all).numpy()
        truth = torch.cat(ct_all).numpy()
        if kind == "binary":
            res["concept_accuracy"] = float((np.sign(z) == np.sign(truth)).mean())
        cp_raw = transform.inverse(z)

    # Error is always reported in ORIGINAL units (mm, mm^3, rating points),
    # against the true raw values -- not against whatever the model was asked
    # to predict. For binary and categorical this includes the information the
    # representation threw away, which is the cost being measured.
    ct_raw = raw_test
    span = np.maximum(ct_raw.max(0) - ct_raw.min(0), 1e-8)
    res["concept_mae_rel"] = float((np.abs(cp_raw - ct_raw) / span).mean())
    res["concept_mae"] = float(np.abs(cp_raw - ct_raw).mean())
    res["concept_rmse"] = float(np.sqrt(((cp_raw - ct_raw) ** 2).mean()))
    res["per_concept_mae"] = {n: float(v) for n, v
                              in zip(concepts, np.abs(cp_raw - ct_raw).mean(0))}
    return res


def train_one(seed, backbone_cfg, cem_cfg, train_cfg, data_cfg,
              variant, device, augment, class_weight, verbose=True,
              weight_decay=0.0):
    """`variant` is a dict: {bottleneck, concept_mode, scaling, n_bins}."""
    set_seed(seed)
    train_df, test_df = load_split(data_cfg, seed)
    concepts = data_cfg.concepts

    transform = CT.build(variant["concept_mode"], train_df, concepts,
                         scaling=variant.get("scaling", "minmax"),
                         n_bins=variant.get("n_bins", 5))
    raw_test = test_df[concepts].values.astype("float32")

    train_loader = DataLoader(
        LIDCNodules(train_df, transform, data_cfg, augment=augment,
                    jitter=3 if augment else 0),
        batch_size=train_cfg.batch_size, shuffle=True,
        num_workers=train_cfg.num_workers)
    test_loader = DataLoader(
        LIDCNodules(test_df, transform, data_cfg),
        batch_size=train_cfg.batch_size, shuffle=False,
        num_workers=train_cfg.num_workers)

    bn = variant["bottleneck"]
    if bn == "none":
        model = Conv3DBaseline(backbone_cfg)
    elif bn == "cbm":
        model = Conv3DwithCBM(backbone_cfg, cem_cfg)
    elif bn == "categorical":
        model = Conv3DwithCategorical(backbone_cfg, cem_cfg, transform.n_states)
    else:
        model = Conv3DwithCEM(backbone_cfg, cem_cfg, bottleneck=bn)
    model = model.to(device)
    n_par = sum(p.numel() for p in model.parameters())

    w = None
    if class_weight:
        counts = np.bincount(train_df.label.values, minlength=2)
        w = torch.tensor(counts.sum() / (2.0 * counts), dtype=torch.float, device=device)
    task_crit = nn.CrossEntropyLoss(weight=w)
    opt = (torch.optim.AdamW(model.parameters(), lr=train_cfg.lr,
                             weight_decay=weight_decay)
           if weight_decay > 0 else
           torch.optim.Adam(model.parameters(), lr=train_cfg.lr))

    def concept_loss(pred, target):
        if pred is None:
            return torch.zeros((), device=device)
        if transform.kind == "categorical":
            return torch.stack([F.cross_entropy(pred[i], target[:, i])
                                for i in range(len(concepts))]).mean()
        return F.mse_loss(pred, target)

    if verbose:
        extra = (f" | states {transform.n_states}"
                 if transform.kind == "categorical" else
                 f" | scaling {variant.get('scaling','minmax')}"
                 if transform.kind == "continuous" else "")
        print(f"  train {len(train_df)} / {train_df.patient_id.nunique()} patients"
              f" | test {len(test_df)} / {test_df.patient_id.nunique()}"
              f" | params {n_par:,} ({n_par*4/1e6:.1f} MB){extra}", flush=True)

    history = []
    for ep in range(1, train_cfg.epochs + 1):
        model.train()
        t_ep = time.time()
        tot_t = tot_c = correct = seen = 0
        for x, c, y in tqdm(train_loader, desc=f"    epoch {ep}/{train_cfg.epochs}",
                            leave=False):
            x, c, y = x.to(device), c.to(device), y.to(device)
            interv = c if (bn != "none"
                           and random.random() < train_cfg.intervention_prob) else None
            logits, concepts_out = model(x, interventions=interv)

            l_task = task_crit(logits, y)
            l_conc = concept_loss(concepts_out, c)
            (l_task + train_cfg.concept_weight * l_conc).backward()
            opt.step(); opt.zero_grad()

            tot_t += l_task.item() * y.size(0)
            tot_c += l_conc.detach().item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            seen += y.size(0)
        history.append({"epoch": ep,
                        "task_loss": tot_t / seen,
                        "concept_loss": tot_c / seen,
                        "train_acc": correct / seen,
                        "seconds": round(time.time() - t_ep, 1)})
        if verbose and (ep % 10 == 0 or ep == 1 or ep == train_cfg.epochs):
            print(f"    epoch {ep:>3}/{train_cfg.epochs}  task {tot_t/seen:.4f}"
                  f"  concept {tot_c/seen:.4f}  train acc {correct/seen:.3f}", flush=True)

    res = evaluate(model, test_loader, transform, device, concepts, raw_test)
    res["params"] = n_par
    res["size_mb"] = n_par * 4 / 1e6
    res["_history"] = history
    return res


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bottleneck", default="cem",
                    choices=sorted(BOTTLENECKS) + ["none", "cbm"],
                    help="'none' = plain CNN, 'cbm' = scalar bottleneck.")
    ap.add_argument("--concept-mode", default="continuous",
                    choices=["continuous", "binary", "categorical"],
                    help="How concepts are represented. binary = high/low split "
                         "at the train mean; categorical = ratings keep their 5 "
                         "levels, continuous concepts get --n-bins bins.")
    ap.add_argument("--concept-scaling", default="minmax", choices=["minmax", "raw"],
                    help="continuous mode only: scale to [-1,1] or pass raw values.")
    ap.add_argument("--n-bins", type=int, default=5,
                    help="categorical mode only: bins for genuinely continuous concepts.")
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
    ap.add_argument("--concepts", nargs="+", default=None, metavar="NAME",
                    help="Train on this subset of concepts instead of the list in "
                         "lidc_data.json (for the concept-incompleteness test). "
                         f"Choose from: {' '.join(ALL_CONCEPTS)}")
    ap.add_argument("--augment", action="store_true",
                    help="Random cube symmetries (all 48) plus +/-3 voxel jitter. "
                         "Isometries only, so the geometric concepts stay valid.")
    ap.add_argument("--weight-decay", type=float, default=0.0,
                    help="AdamW weight decay. 0 keeps plain Adam (the default so "
                         "far, which overfits badly on 944 volumes).")
    ap.add_argument("--backbone-dropout", type=float, default=None,
                    help="Dropout3d after each encoder block, overriding the "
                         "backbone config (currently 0.0).")
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
    if args.backbone_dropout is not None:
        backbone_cfg.dropout = args.backbone_dropout
    if args.concepts:
        bad = [c for c in args.concepts if c not in ALL_CONCEPTS]
        if bad:
            ap.error(f"unknown concept(s) {bad}. Choose from: {' '.join(ALL_CONCEPTS)}")
        data_cfg.concepts = list(dict.fromkeys(args.concepts))   # dedupe, keep order
    cem_cfg.n_concepts = len(data_cfg.concepts)

    device = get_device(args.device)
    variant = {"bottleneck": args.bottleneck, "concept_mode": args.concept_mode,
               "scaling": args.concept_scaling, "n_bins": args.n_bins}
    print(f"device {device} | bottleneck {args.bottleneck} | runs {args.runs}")
    print(f"concept mode {args.concept_mode}"
          + (f" (scaling {args.concept_scaling})" if args.concept_mode == "continuous"
             else f" (n_bins {args.n_bins})" if args.concept_mode == "categorical" else ""))
    print(f"backbone channels {backbone_cfg.conv_channels} | embedding_dim "
          f"{cem_cfg.embedding_dim} | hidden {cem_cfg.hidden_dim} | depth {cem_cfg.depth}")
    print(f"epochs {train_cfg.epochs} | lr {train_cfg.lr} | batch {train_cfg.batch_size} "
          f"| concept_weight {train_cfg.concept_weight} "
          f"| intervention_prob {train_cfg.intervention_prob}")
    print(f"concepts ({len(data_cfg.concepts)}): {data_cfg.concepts}")
    print(f"min_annotations {data_cfg.min_annotations} | augment {args.augment} "
          f"| weight_decay {args.weight_decay} | backbone dropout {backbone_cfg.dropout}\n")

    name = args.tag or args.bottleneck
    hist_path = Path("outputs/logs") / f"lidc_{name}_history.csv"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    with open(hist_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["variant", "bottleneck", "concept_mode", "seed",
             "epoch", "task_loss", "concept_loss", "train_acc", "seconds"])

    runs, t0 = [], time.time()
    for r in range(args.runs):
        seed = args.seed + r
        print(f"=== seed {seed} ({r+1}/{args.runs}) ===", flush=True)
        res = train_one(seed, backbone_cfg, cem_cfg, train_cfg, data_cfg,
                        variant, device, args.augment, not args.no_class_weight,
                        weight_decay=args.weight_decay)

        # Append this seed's epoch curve immediately, so an interrupted job
        # still leaves a plottable record.
        with open(hist_path, "a", newline="") as f:
            w = csv.writer(f)
            for h in res.pop("_history"):
                w.writerow([name, args.bottleneck, args.concept_mode, seed,
                            h["epoch"], f"{h['task_loss']:.6f}",
                            f"{h['concept_loss']:.6f}", f"{h['train_acc']:.6f}",
                            h["seconds"]])

        runs.append(res)
        print(f"  accuracy {res['accuracy']:.3f}  balanced "
              f"{res['balanced_accuracy']:.3f}  (history -> {hist_path})\n", flush=True)
        write_results(args, train_cfg, backbone_cfg, cem_cfg, runs,
                      data_cfg.concepts)   # after EVERY seed

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

    print(f"\ntotal {time.time()-t0:.0f}s   "
          f"results: {result_path(args)}   history: {hist_path}")


def result_path(args):
    out = Path(args.out)
    return out.with_name(f"{out.stem}_{args.tag}{out.suffix}") if args.tag else out


def write_results(args, train_cfg, backbone_cfg, cem_cfg, runs, concepts):
    """Rewrite the result CSV from whatever seeds have finished so far."""
    keys = [k for k in runs[0] if isinstance(runs[0][k], float) or k == "params"]
    out = result_path(args)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bottleneck", "concept_mode", "scaling", "n_bins", "runs",
                    "epochs", "conv_channels", "embedding_dim",
                    "concept_weight", "intervention_prob", "concepts"]
                   + [f"{k}_{s}" for k in keys for s in ("mean", "std")])
        w.writerow([args.bottleneck, args.concept_mode, args.concept_scaling,
                    args.n_bins, len(runs), train_cfg.epochs,
                    "-".join(map(str, backbone_cfg.conv_channels)),
                    cem_cfg.embedding_dim, train_cfg.concept_weight,
                    train_cfg.intervention_prob, "+".join(concepts)]
                   + [v for k in keys
                      for v in (float(np.mean([r[k] for r in runs])),
                                float(np.std([r[k] for r in runs])))])


if __name__ == "__main__":
    main()
