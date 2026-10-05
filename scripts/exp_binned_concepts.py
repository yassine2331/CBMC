"""
Case 3 (binned concepts) vs Case 3.5 (continuous interpolation) on ArithmeticMNIST.

ArithmeticMNIST concepts are the two digit values, integers in [1, 9]. Case 3.5
treats them as continuous and normalises to [-1, 1]. Case 3 instead chops the
range into k disjoint bins and treats bin identity as a categorical concept,
which is exactly what the CEMCategorical block models.

This script trains both and reports the same metrics for each, so the cost of
discretising a continuous concept is visible directly.

    python scripts/exp_binned_concepts.py --epochs 30 --bins 5 10 20

Nothing else in the repo is modified; this file only imports existing modules.

Metrics (all on the test set, lower is better except concept accuracy):
    task MSE          regression error on the arithmetic result
    concept MSE       in normalised [-1,1] space, so Case 3 and 3.5 compare.
                      For binned models the predicted bin's CENTRE is decoded
                      back to a value -- this is the quantisation cost.
    intervention MSE  task error when both concepts are replaced by truth
    concept acc       fraction of concepts whose bin/value is exactly right
                      (for Case 3.5, rounding the decoded value to a digit)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from cbmc.configs import CNNRegressionConfig, CEMConfig, CEMCategoricalConfig, TrainConfig
from cbmc.data.arithmetic_mnist import get_arithmetic_mnist
from architectures.cnn_cem import CNNwithCEM
from architectures.cnn_cem_categorical import CNNwithCEMCategorical

# Same normalisation the existing MNIST experiments use: 1 -> -1, 5 -> 0, 9 -> +1
CONCEPT_MEAN, CONCEPT_SCALE = 5.0, 4.0
CONCEPT_LO, CONCEPT_HI = 1.0, 9.0


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def get_device(override=None):
    if override: return override
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------

class Binner:
    """
    Uniform bins over the raw concept range [1, 9].

    to_bin   : raw value  -> bin index in [0, k)
    to_value : bin index  -> that bin's centre, as a raw value
    """

    def __init__(self, k: int, lo: float = CONCEPT_LO, hi: float = CONCEPT_HI):
        self.k, self.lo, self.hi = k, lo, hi
        self.width = (hi - lo) / k
        # centres of each bin, as raw concept values
        self.centres = torch.tensor(
            [lo + (i + 0.5) * self.width for i in range(k)], dtype=torch.float32)

    def to_bin(self, raw: torch.Tensor) -> torch.Tensor:
        idx = ((raw - self.lo) / self.width).floor().long()
        return idx.clamp(0, self.k - 1)

    def to_value(self, idx: torch.Tensor) -> torch.Tensor:
        return self.centres.to(idx.device)[idx]

    def quantisation_mse(self, raw: torch.Tensor) -> float:
        """MSE floor imposed by binning alone, in normalised space."""
        recon = self.to_value(self.to_bin(raw))
        return (((recon - raw) / CONCEPT_SCALE) ** 2).mean().item()


def norm(raw):   return (raw - CONCEPT_MEAN) / CONCEPT_SCALE
def denorm(z):   return z * CONCEPT_SCALE + CONCEPT_MEAN


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------

def evaluate(model, loader, device, binner, criterion):
    model.eval()
    n = 0
    task = interv = cmse = cacc = 0.0
    with torch.no_grad():
        for x, c_raw, y in loader:
            x, c_raw, y = x.to(device), c_raw.to(device), y.to(device).unsqueeze(1)

            if binner is None:                       # Case 3.5
                c_true = norm(c_raw)
                preds, concepts = model(x)
                pred_val = denorm(concepts)
                cmse += F.mse_loss(concepts, c_true).item()
                cacc += (pred_val.round().clamp(CONCEPT_LO, CONCEPT_HI)
                         == c_raw).float().mean().item()
                pi, _ = model(x, interventions=c_true)
            else:                                    # Case 3
                c_bin = binner.to_bin(c_raw)
                preds, logits = model(x)
                pred_bin = logits.argmax(dim=2)
                pred_val = binner.to_value(pred_bin)
                cmse += F.mse_loss(norm(pred_val), norm(c_raw)).item()
                cacc += (pred_bin == c_bin).float().mean().item()
                pi, _ = model(x, interventions=c_bin)

            task   += criterion(preds, y).item()
            interv += criterion(pi, y).item()
            n += 1
    return dict(task_mse=task/n, intervention_mse=interv/n,
                concept_mse=cmse/n, concept_acc=cacc/n)


def train_one(name, n_bins, train_loader, test_loader, device, cfg, epochs, seed,
              concept_weight=1.0, intervention_prob=0.25):
    """n_bins=None -> Case 3.5 (CEM); otherwise Case 3 (CEMCategorical)."""
    set_seed(seed)
    backbone = CNNRegressionConfig.load("experiments/configs/exp_cem_cls_mnist_backbone.json")
    binner = None if n_bins is None else Binner(n_bins)

    if n_bins is None:
        model = CNNwithCEM(backbone, CEMConfig(**cfg), n_outputs=1).to(device)
    else:
        model = CNNwithCEMCategorical(
            backbone,
            CEMCategoricalConfig(n_concepts=cfg["n_concepts"], n_classes=n_bins, **{
                k: v for k, v in cfg.items() if k != "n_concepts"}),
            n_outputs=1).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    n_params = sum(p.numel() for p in model.parameters())

    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        pbar = tqdm(train_loader, desc=f"  {name} ep {ep}/{epochs}", leave=False)
        for x, c_raw, y in pbar:
            x, c_raw, y = x.to(device), c_raw.to(device), y.to(device).unsqueeze(1)
            intervene = random.random() < intervention_prob

            if binner is None:
                c_target = norm(c_raw)
                preds, concepts = model(x, interventions=c_target if intervene else None)
                closs = F.mse_loss(concepts, c_target)
            else:
                c_bin = binner.to_bin(c_raw)
                preds, logits = model(x, interventions=c_bin if intervene else None)
                closs = model.concept_loss(logits, c_bin)

            loss = criterion(preds, y) + concept_weight * closs
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        if ep % 10 == 0 or ep == 1:
            print(f"    {name} epoch {ep:>2}/{epochs}  train loss {tot/len(train_loader):.4f}",
                  flush=True)

    res = evaluate(model, test_loader, device, binner, criterion)
    res["params"] = n_params
    res["size_mb"] = n_params * 4 / 1e6          # float32 weights
    if binner is not None:
        res["quantisation_floor"] = binner.quantisation_mse(
            torch.arange(1, 10, dtype=torch.float32))
    return res


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bins", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--runs", type=int, default=1,
                    help="Seeds per model; >1 reports mean +/- std (seeds 41, 42, ...).")
    ap.add_argument("--seed", type=int, default=41, help="First seed.")
    ap.add_argument("--concept-weight", type=float, default=1.0,
                    help="Weight on the concept loss. The task MSE spans 0-81 while the "
                         "concept MSE lives in [-1,1], so 1.0 leaves concepts roughly "
                         "95x under-weighted; 100 was needed for concepts to be learned.")
    ap.add_argument("--intervention-prob", type=float, default=0.25,
                    help="Fraction of training batches trained with ground-truth concepts.")
    ap.add_argument("--device", default=None, choices=[None, "cuda", "mps", "cpu"])
    ap.add_argument("--out", default="outputs/results/binned_concepts.csv")
    ap.add_argument("--tag", default=None, help="Suffix for the output filename.")
    args = ap.parse_args()

    device = get_device(args.device)
    print(f"device: {device} | epochs: {args.epochs} | bins: {args.bins} | runs: {args.runs}")
    print(f"concept_weight: {args.concept_weight} | intervention_prob: {args.intervention_prob}\n")

    train_cfg = TrainConfig.load("experiments/configs/train_cem_mnist.json")
    train_loader, test_loader = get_arithmetic_mnist(
        batch_size=train_cfg.batch_size, num_workers=train_cfg.num_workers,
        operators=('+', 'x'), digits=None)
    cem_cfg = json.load(open("experiments/configs/cem_mnist.json"))

    models = [("CEM (Case 3.5)", None)] + [(f"Binned k={k} (Case 3)", k) for k in args.bins]
    METRICS = ["task_mse", "concept_mse", "intervention_mse", "concept_acc"]
    results = {}

    for name, k in models:
        print(f"=== {name} ===", flush=True)
        per_seed = []
        t0 = time.time()
        for r in range(args.runs):
            seed = args.seed + r
            if args.runs > 1:
                print(f"  seed {seed} ({r+1}/{args.runs})", flush=True)
            per_seed.append(train_one(
                name, k, train_loader, test_loader, device, cem_cfg,
                args.epochs, seed, concept_weight=args.concept_weight,
                intervention_prob=args.intervention_prob))

        agg = {"n_runs": args.runs,
               "params": per_seed[0]["params"],
               "size_mb": per_seed[0]["size_mb"],
               "seconds": round(time.time() - t0)}
        for m in METRICS:
            vals = [p[m] for p in per_seed]
            agg[m + "_mean"] = float(np.mean(vals))
            agg[m + "_std"] = float(np.std(vals))
        if "quantisation_floor" in per_seed[0]:
            agg["quantisation_floor"] = per_seed[0]["quantisation_floor"]
        results[name] = agg
        print(f"    done in {agg['seconds']}s  task MSE "
              f"{agg['task_mse_mean']:.3f} +/- {agg['task_mse_std']:.3f}\n", flush=True)

    # ---------------- report ----------------
    W = 26
    print("\n" + "=" * 118)
    print(f"{'model':<{W}}" + "".join(f"{m.replace('_',' '):>21}" for m in METRICS)
          + f"{'params':>11}{'size MB':>10}")
    print("-" * 118)
    for name, r in results.items():
        row = f"{name:<{W}}"
        for m in METRICS:
            row += f"{r[m+'_mean']:>12.4f} +/-{r[m+'_std']:>6.3f}"
        row += f"{r['params']:>11,}{r['size_mb']:>10.2f}"
        print(row)
    print("=" * 118)

    floors = {n: r["quantisation_floor"] for n, r in results.items()
              if "quantisation_floor" in r}
    if floors:
        print("\nquantisation floor (concept MSE a perfect binned model cannot beat):")
        for n, v in floors.items():
            print(f"  {n:<{W}} {v:.5f}")

    # ---------------- save ----------------
    tag = f"_{args.tag}" if args.tag else ""
    out = Path(args.out)
    out = out.with_name(out.stem + tag + out.suffix)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ([f"{m}_{s}" for m in METRICS for s in ("mean", "std")]
            + ["params", "size_mb", "n_runs", "seconds", "quantisation_floor"])
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "epochs", "concept_weight", "intervention_prob"] + cols)
        for name, r in results.items():
            w.writerow([name, args.epochs, args.concept_weight, args.intervention_prob]
                       + [r.get(c, "") for c in cols])
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
