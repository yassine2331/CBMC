"""
One benchmark for every dataset x concept representation x seed.

    image -> encoder -> concept bottleneck -> head -> task

Models (same encoder, head and training loop; only the bottleneck differs):

    cbm             scalar bottleneck: the head sees k numbers
    cem             two anchors, continuous interpolation        OUR METHOD (Case 3.5)
    cem_linear      one embedding x concept value, LayerNorm
    cem_linear_raw  one embedding x concept value, NO LayerNorm  (the paper's CEM-Linear-Raw)
    binned5/10/15   each concept's [-1,1] range cut into k uniform bins,
                    categorical anchors (Case 3)
    nn              no bottleneck (optional accuracy anchor, not in the default set)

Datasets:

    mnist      ArithmeticMNIST. concepts: the 2 digits. task: the arithmetic result (regression)
    pendulum   concepts: pendulum angle theta, light angle phi.
               task: shadow length + shadow position (regression).
               The old runs used concepts == target, which makes a CBM trivially
               optimal under intervention; here the shadow is CAUSED by the concepts.
    lidc       LIDC-IDRI nodules. 9 concepts. task: malignant / benign. Same
               regularised regime as the paper (augment, AdamW 0.01, Dropout3d 0.1).
    oai        not runnable yet: the knee X-rays are not downloaded (see _oai()).

Every concept is min-max scaled to [-1, 1] on the TRAIN split, so the "zero"
of a concept is the middle of its range (digit 5, LIDC rating 3, theta 0).

Test metrics, per run (regression: MSE in original units; classification:
accuracy and balanced accuracy):

    task_*            no intervention
    interv_*          every concept replaced by the truth
    concept_mse       concept error in [-1,1] units (binned: the bin centre)
    concept_acc       binned only: bin accuracy
    zero group = test samples with at least one concept within --zero-tol of 0:
    zero_task_*       no intervention, on the zero group
    zero_interv0_*    ONLY the near-zero concepts set to the truth   <- the collapse test
    zero_interv_*     every concept set to the truth, on the zero group
    zero_concept_mse  concept error on the near-zero concepts only
    nonzero_task_*, nonzero_interv_*   the same on the rest of the test set

Usage
    python scripts/run_benchmark.py --dataset mnist                       # 7 models x 30 seeds
    python scripts/run_benchmark.py --dataset lidc --models cem cem_linear --seeds 5
    python scripts/run_benchmark.py --dataset mnist --summarize           # mean +/- std table

Output (resumable: a finished (model, seed) is skipped on rerun)
    outputs/results/benchmark/<dataset>/<model>.csv      one row per seed
    outputs/results/benchmark/<dataset>_summary.csv      mean / std  (--summarize)
    outputs/models/benchmark/<dataset>/<model>_seed<N>.pt
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from cbmc.concepts.cem import CEM, CEMLinear, CEMLinearRaw
from cbmc.concepts.categorical import CEMCategoricalPerConcept

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = ["cbm", "cem", "cem_linear", "cem_linear_raw", "binned5", "binned10", "binned15"]
ALL_MODELS = DEFAULT_MODELS + ["nn"]


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def get_device(name=None):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ===========================================================================
# Datasets. Each returns a dict:
#   train, test        Datasets yielding (x, c_norm [C] in [-1,1], y)
#   encoder            factory () -> nn.Module with .out_dim (fresh per seed)
#   concepts           names
#   task               "regression" | "classification"
#   n_out              head outputs
#   y_mean, y_std      regression only: training targets are standardised,
#                      errors are reported in original units
#   class_weight       classification only (or None)
#   defaults           epochs / lr / batch_size / weight_decay / concept_weight / intervention_prob
# ===========================================================================

class Conv2DEncoder(nn.Module):
    def __init__(self, in_ch, channels=(32, 64, 128), dropout=0.0):
        super().__init__()
        layers = []
        for ch in channels:
            layers += [nn.Conv2d(in_ch, ch, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            in_ch = ch
        layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten()]
        self.net = nn.Sequential(*layers)
        self.out_dim = channels[-1]

    def forward(self, x):
        return self.net(x)


class _Map(Dataset):
    """Wrap a dataset, mapping each (x, c, y) through fn."""
    def __init__(self, base, fn):
        self.base, self.fn = base, fn

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        return self.fn(*self.base[i])


def _mnist(args, seed):
    from cbmc.data.arithmetic_mnist import ArithmeticMNISTDataset
    ops = tuple(args.operators)
    kw = dict(mnist_root="data/generated/MNIST", img_size=64, operators=ops, digits=None)
    # the dataset itself is fixed across seeds (as in run_experiment.py); seeds vary the model
    tr = ArithmeticMNISTDataset(train=True, num_samples=10000, seed=42, **kw)
    te = ArithmeticMNISTDataset(train=False, num_samples=2000, seed=43, **kw)
    norm = lambda x, c, y: (x, (c - 5.0) / 4.0, torch.as_tensor(y, dtype=torch.float32).view(1))
    # targets without rendering images: replay the dataset's per-index RNG
    ys = []
    for i in range(len(tr)):
        rng = random.Random(i)
        a = rng.choice(tr._allowed_digits); rng.choice(tr._digit_indices[a])
        b = rng.choice(tr._allowed_digits)
        op = tr.operator_list[i]
        ys.append({"+": a + b, "-": a - b, "x": a * b, "/": a / b}[op])
    ys = torch.tensor(ys, dtype=torch.float32)
    return dict(train=_Map(tr, norm), test=_Map(te, norm),
                encoder=lambda: Conv2DEncoder(1), concepts=["digit_a", "digit_b"],
                task="regression", n_out=1, y_mean=ys.mean(), y_std=ys.std(),
                class_weight=None,
                defaults=dict(epochs=30, lr=1e-3, batch_size=128, weight_decay=0.0,
                              concept_weight=1.0, intervention_prob=0.25))


def _pendulum(args, seed):
    from cbmc.data.pendulum import PendulumDataset
    import torchvision.transforms as T
    tf = T.Compose([T.Resize((64, 64)), T.ToTensor()])
    root = Path("data/generated/pendulum")
    tr = PendulumDataset(str(root / "train"), transform=tf)
    te = PendulumDataset(str(root / "test"), transform=tf)
    lab = tr.labels
    lo, hi = lab[:, :2].min(0).values, lab[:, :2].max(0).values
    y_mean, y_std = lab[:, 2:].mean(0), lab[:, 2:].std(0).clamp(min=1e-6)

    def fn(x, lab_row, _):
        c = 2 * (lab_row[:2] - lo) / (hi - lo) - 1
        return x, c, lab_row[2:]

    train, test = _Map(tr, fn), _Map(te, fn)
    if args.max_train and args.max_train < len(train):
        g = torch.Generator().manual_seed(seed)
        train = Subset(train, torch.randperm(len(train), generator=g)[:args.max_train].tolist())
    return dict(train=train, test=test, encoder=lambda: Conv2DEncoder(3),
                concepts=["theta", "phi"], task="regression", n_out=2,
                y_mean=y_mean, y_std=y_std, class_weight=None,
                defaults=dict(epochs=20, lr=1e-3, batch_size=64, weight_decay=0.0,
                              concept_weight=1.0, intervention_prob=0.25))


def _lidc(args, seed):
    from cbmc.configs import Conv3DConfig, LIDCDataConfig
    from cbmc.data import concept_transforms as CT
    from architectures.conv3d_cem import Conv3DEncoder
    from scripts.train_lidc import LIDCNodules, load_split

    cfg = Path("experiments/configs")
    data_cfg = LIDCDataConfig.load(cfg / "lidc_data.json")
    backbone = Conv3DConfig.load(cfg / "lidc_backbone.json")
    backbone.dropout = 0.1                                   # paper's regularised regime
    train_df, test_df = load_split(data_cfg, seed)           # split BY PATIENT, varies with seed
    tf = CT.build("continuous", train_df, data_cfg.concepts, scaling="minmax")
    counts = np.bincount(train_df.label.values, minlength=2)
    return dict(train=LIDCNodules(train_df, tf, data_cfg, augment=True, jitter=3),
                test=LIDCNodules(test_df, tf, data_cfg),
                encoder=lambda: Conv3DEncoder(backbone), concepts=list(data_cfg.concepts),
                task="classification", n_out=2, y_mean=None, y_std=None,
                class_weight=torch.tensor(counts.sum() / (2.0 * counts), dtype=torch.float),
                defaults=dict(epochs=250, lr=1e-3, batch_size=16, weight_decay=0.01,
                              concept_weight=1.0, intervention_prob=0.25))


def _oai(args, seed):
    raise SystemExit(
        "OAI is not runnable yet: the knee radiographs are not on disk (only the\n"
        "ASCII tables are). The plan from notebooks/oai_concepts.ipynb:\n"
        "  unit      one knee at baseline (data/processed/oai_knees_v00.csv, 8982 knees)\n"
        "  concepts  JSW at 13 locations (mm) + femorotibial angle FTA (continuous)\n"
        "  target    OA = KL >= 2, splits grouped by subject ID\n"
        "Once the bilateral PA films are downloaded and cropped per knee, add a\n"
        "loader here returning the same dict as _mnist/_lidc.")


DATASETS = {"mnist": _mnist, "pendulum": _pendulum, "lidc": _lidc, "oai": _oai}
SEED_DEPENDENT = {"lidc"}            # split by patient changes with the seed
_CACHE = {}


def load_dataset(args, seed):
    """mnist / pendulum data are identical for every seed, so build them once."""
    key = (args.dataset, seed if args.dataset in SEED_DEPENDENT else None)
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = DATASETS[args.dataset](args, seed)
    return _CACHE[key]


# ===========================================================================
# Models
# ===========================================================================

def n_bins_of(model):
    return int(model[len("binned"):]) if model.startswith("binned") else 0


def to_bins(c, k):
    """[-1,1] -> bin index 0..k-1."""
    return ((c + 1) / 2 * k).floor().clamp(0, k - 1).long()


def bin_centres(idx, k):
    return -1 + (idx.float() + 0.5) * 2 / k


class ConceptModel(nn.Module):
    """encoder -> bottleneck -> head, with a uniform intervention interface.

    forward(x, interv=None, mask=None) -> (out, concept_pred)
        interv : [B, C] TRUE concepts in [-1,1] (binned models bin them here)
        mask   : [B, C] 1 = use the truth for that concept, 0 = keep the prediction
    concept_pred: [B, C] in [-1,1] for continuous models; list of logits for binned.
    """

    def __init__(self, kind, encoder, n_concepts, n_out, emb=16, hidden=64, depth=2, dropout=0.1):
        super().__init__()
        self.kind, self.encoder, self.C = kind, encoder, n_concepts
        d = encoder.out_dim
        self.k = n_bins_of(kind)
        if kind == "nn":
            self.head = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, n_out))
        elif kind == "cbm":
            self.cbm = nn.Linear(d, n_concepts)
            self.head = nn.Sequential(nn.Linear(n_concepts, 32), nn.ReLU(),
                                      nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, n_out))
        else:
            if self.k:
                self.bott = CEMCategoricalPerConcept(d, [self.k] * n_concepts, emb, hidden, depth, dropout)
            else:
                cls = {"cem": CEM, "cem_linear": CEMLinear, "cem_linear_raw": CEMLinearRaw}[kind]
                self.bott = cls(d, n_concepts, emb, hidden, depth, dropout)
            self.head = nn.Linear(self.bott.output_dim, n_out)

    def forward(self, x, interv=None, mask=None):
        h = self.encoder(x)
        if self.kind == "nn":
            return self.head(h), None
        if self.kind == "cbm":
            c = self.cbm(h)
            used = c
            if interv is not None:
                used = interv if mask is None else mask * interv + (1 - mask) * c
            return self.head(used), c
        if self.k and interv is not None:
            interv = to_bins(interv, self.k)
        emb, c = self.bott(h, interv, mask)
        return self.head(emb), c

    def decode(self, c):
        """concept prediction -> [B, C] in [-1,1]."""
        if self.k:
            return bin_centres(torch.stack([lg.argmax(1) for lg in c], 1), self.k)
        return c


def concept_loss(model, pred, c_true):
    if pred is None:
        return torch.zeros((), device=c_true.device)
    if model.k:
        t = to_bins(c_true, model.k)
        return torch.stack([F.cross_entropy(lg, t[:, i]) for i, lg in enumerate(pred)]).mean()
    return F.mse_loss(pred, c_true)


# ===========================================================================
# Train / evaluate
# ===========================================================================

def train_one(model, D, hp, device, args):
    loader = DataLoader(D["train"], batch_size=hp["batch_size"], shuffle=True,
                        num_workers=args.num_workers, drop_last=False,
                        persistent_workers=args.num_workers > 0)
    opt = (torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
           if hp["weight_decay"] > 0 else torch.optim.Adam(model.parameters(), lr=hp["lr"]))
    if D["task"] == "classification":
        w = D["class_weight"].to(device) if D["class_weight"] is not None else None
        task_loss = nn.CrossEntropyLoss(weight=w)
    else:
        ym, ys = D["y_mean"].to(device), D["y_std"].to(device)
        task_loss = lambda out, y: F.mse_loss(out, (y - ym) / ys)
    history = []
    for ep in range(1, hp["epochs"] + 1):
        model.train()
        t0, tot_t, tot_c, n = time.time(), 0.0, 0.0, 0
        for x, c, y in loader:
            x, c, y = x.to(device), c.to(device).float(), y.to(device)
            interv = c if (model.kind != "nn" and random.random() < hp["intervention_prob"]) else None
            out, cp = model(x, interv)
            lt, lc = task_loss(out, y), concept_loss(model, cp, c)
            opt.zero_grad(); (lt + hp["concept_weight"] * lc).backward(); opt.step()
            tot_t += lt.item() * len(x); tot_c += lc.item() * len(x); n += len(x)
        history.append(dict(epoch=ep, task_loss=tot_t / n, concept_loss=tot_c / n,
                            seconds=round(time.time() - t0, 1)))
        if ep == 1 or ep % max(1, hp["epochs"] // 10) == 0 or ep == hp["epochs"]:
            print(f"    epoch {ep:>3}/{hp['epochs']}  task {tot_t/n:.4f}  concept {tot_c/n:.4f}"
                  f"  ({history[-1]['seconds']}s)", flush=True)
    return history


def _task_metrics(out, y, D, prefix):
    """out: raw head outputs [N, n_out]; y: targets."""
    if len(y) == 0:
        return {}
    if D["task"] == "regression":
        pred = out * D["y_std"] + D["y_mean"]
        return {f"{prefix}_mse": ((pred - y.view_as(pred)) ** 2).mean().item()}
    p = out.argmax(1)
    res = {f"{prefix}_acc": (p == y).float().mean().item()}
    if all((y == k).any() for k in (0, 1)):
        res[f"{prefix}_bacc"] = float(np.mean([(p[y == k] == k).float().mean().item() for k in (0, 1)]))
    return res


@torch.no_grad()
def evaluate(model, D, device, args):
    model.eval()
    loader = DataLoader(D["test"], batch_size=256, shuffle=False, num_workers=args.num_workers)
    keep = {k: [] for k in ("y", "out", "out_all", "out_zero", "c", "chat", "zero", "bin_ok")}
    for x, c, y in loader:
        x, c = x.to(device), c.to(device).float()
        zero = c.abs() <= args.zero_tol                         # [B, C]
        out, cp = model(x)
        keep["y"].append(y); keep["out"].append(out.cpu()); keep["zero"].append(zero.cpu())
        if model.kind == "nn":
            continue
        keep["out_all"].append(model(x, c)[0].cpu())
        keep["out_zero"].append(model(x, c, zero.float())[0].cpu())
        keep["c"].append(c.cpu()); keep["chat"].append(model.decode(cp).cpu())
        if model.k:
            pred_bins = torch.stack([lg.argmax(1) for lg in cp], 1)
            keep["bin_ok"].append((pred_bins == to_bins(c, model.k)).cpu())
    cat = {k: torch.cat(v) for k, v in keep.items() if v}
    y, zero = cat["y"], cat["zero"]
    if D["task"] == "regression":
        D = {**D, "y_mean": D["y_mean"].cpu(), "y_std": D["y_std"].cpu()}
    zrow = zero.any(1)
    res = {"n_test": len(y), "n_zero": int(zrow.sum())}
    res.update(_task_metrics(cat["out"], y, D, "task"))
    res.update(_task_metrics(cat["out"][zrow], y[zrow], D, "zero_task"))
    res.update(_task_metrics(cat["out"][~zrow], y[~zrow], D, "nonzero_task"))
    if model.kind != "nn":
        res.update(_task_metrics(cat["out_all"], y, D, "interv"))
        res.update(_task_metrics(cat["out_zero"][zrow], y[zrow], D, "zero_interv0"))
        res.update(_task_metrics(cat["out_all"][zrow], y[zrow], D, "zero_interv"))
        res.update(_task_metrics(cat["out_all"][~zrow], y[~zrow], D, "nonzero_interv"))
        se = (cat["chat"] - cat["c"]) ** 2
        res["concept_mse"] = se.mean().item()
        res["zero_concept_mse"] = se[zero].mean().item() if zero.any() else float("nan")
        if model.k:
            res["concept_acc"] = cat["bin_ok"].float().mean().item()
    return res


# ===========================================================================
# Bookkeeping
# ===========================================================================

def result_file(dataset, model):
    return ROOT / "outputs" / "results" / "benchmark" / dataset / f"{model}.csv"


def done_seeds(path):
    if not path.exists():
        return set()
    with open(path) as f:
        return {int(r["seed"]) for r in csv.DictReader(f)}


def append_row(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(path))) if path.exists() else []
    rows.append({k: row[k] for k in row})
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def summarize(dataset):
    d = ROOT / "outputs" / "results" / "benchmark" / dataset
    files = sorted(d.glob("*.csv"))
    if not files:
        raise SystemExit(f"no results in {d}")
    order = {m: i for i, m in enumerate(ALL_MODELS)}
    files.sort(key=lambda p: order.get(p.stem, 99))
    skip = {"dataset", "model", "seed", "epochs", "n_test", "n_zero"}
    out_rows, metrics = [], []
    for p in files:
        rows = list(csv.DictReader(open(p)))
        s = {"model": p.stem, "seeds": len(rows), "n_zero": rows[0].get("n_zero", "")}
        for k in rows[0]:
            if k in skip:
                continue
            v = np.array([float(r[k]) if r.get(k) not in ("", None) else np.nan for r in rows])
            if np.isfinite(v).any():
                s[f"{k}_mean"], s[f"{k}_std"] = np.nanmean(v), np.nanstd(v)
                if k not in metrics:
                    metrics.append(k)
        out_rows.append(s)
    out = d.parent / f"{dataset}_summary.csv"
    keys = list(dict.fromkeys(k for r in out_rows for k in r))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(out_rows)
    show = [m for m in metrics if any(m.startswith(p) for p in
            ("task_", "interv_", "concept_", "zero_", "nonzero_"))
            and m not in ("params", "train_seconds")]
    print(f"\n{dataset}: mean ± std over seeds\n")
    print(f"{'metric':22}" + "".join(f"{r['model']:>18}" for r in out_rows))
    print(f"{'seeds':22}" + "".join(f"{r['seeds']:>18}" for r in out_rows))
    for m in show:
        cells = []
        for r in out_rows:
            if f"{m}_mean" in r:
                cells.append(f"{r[f'{m}_mean']:.4g} ± {r[f'{m}_std']:.2g}")
            else:
                cells.append("—")
        print(f"{m:22}" + "".join(f"{c:>18}" for c in cells))
    print(f"\nsaved {out}")


# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=ALL_MODELS)
    ap.add_argument("--seeds", type=int, default=30, help="number of seeds, starting at --first-seed")
    ap.add_argument("--first-seed", type=int, default=41)
    ap.add_argument("--epochs", type=int, default=None, help="default: per dataset (mnist 30, pendulum 20, lidc 250)")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--concept-weight", type=float, default=None)
    ap.add_argument("--intervention-prob", type=float, default=None)
    ap.add_argument("--embedding-dim", type=int, default=16)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1, help="bottleneck MLP dropout")
    ap.add_argument("--zero-tol", type=float, default=0.025,
                    help="a concept counts as zero if |c| <= this, in [-1,1] units")
    ap.add_argument("--operators", nargs="+", default=["+", "x"], help="mnist only")
    ap.add_argument("--max-train", type=int, default=None, help="pendulum only: subsample the 70k training images")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"])
    ap.add_argument("--no-save-models", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="retrain seeds that already have results")
    ap.add_argument("--summarize", action="store_true", help="only print / write the mean ± std table")
    a = ap.parse_args()
    os.chdir(ROOT)

    if a.summarize:
        summarize(a.dataset); return

    device = get_device(a.device)
    seeds = list(range(a.first_seed, a.first_seed + a.seeds))
    print(f"dataset {a.dataset} | models {a.models} | seeds {seeds[0]}..{seeds[-1]} | device {device}")

    for model_name in a.models:
        path = result_file(a.dataset, model_name)
        if a.overwrite and path.exists():
            path.unlink()
        have = done_seeds(path)
        todo = [s for s in seeds if s not in have]
        print(f"\n=== {a.dataset} / {model_name}: {len(have)} seeds done, {len(todo)} to run ===", flush=True)
        for seed in todo:
            set_seed(seed)
            D = load_dataset(a, seed)
            hp = dict(D["defaults"])
            for k in ("epochs", "lr", "batch_size", "weight_decay", "concept_weight", "intervention_prob"):
                if getattr(a, k) is not None:
                    hp[k] = getattr(a, k)
            model = ConceptModel(model_name, D["encoder"](), len(D["concepts"]), D["n_out"],
                                 a.embedding_dim, a.hidden_dim, a.depth, a.dropout).to(device)
            n_par = sum(p.numel() for p in model.parameters())
            print(f"  seed {seed}: {n_par:,} params | {hp}", flush=True)
            t0 = time.time()
            history = train_one(model, D, hp, device, a)
            res = evaluate(model, D, device, a)
            row = {"dataset": a.dataset, "model": model_name, "seed": seed, "epochs": hp["epochs"],
                   "params": n_par, "train_seconds": round(time.time() - t0, 1), **res}
            append_row(path, row)
            if not a.no_save_models:
                mdir = ROOT / "outputs" / "models" / "benchmark" / a.dataset
                mdir.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": model.state_dict(), "model": model_name, "seed": seed,
                            "hp": hp, "concepts": D["concepts"], "history": history, "result": res},
                           mdir / f"{model_name}_seed{seed}.pt")
            key = "task_mse" if "task_mse" in res else "task_bacc"
            msg = f"  seed {seed} done: {key} {res.get(key, float('nan')):.4f}"
            if f"zero_interv0_{key[5:]}" in res:
                msg += (f" | zero group: task {res['zero_' + key]:.4f}"
                        f" -> fix zero concept {res['zero_interv0_' + key[5:]]:.4f}")
            print(msg, flush=True)
        print(f"  -> {path}")

    summarize(a.dataset)


if __name__ == "__main__":
    main()
