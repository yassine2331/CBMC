"""
End-to-end experiment: train a Concept Bottleneck Model on a synthetic
dataset with continuous concepts and print evaluation metrics.

Usage
-----
    python experiments/train_cbm.py [options]

Options
-------
  --mode        {joint,sequential,independent}  Training mode (default: joint)
  --task        {classification,regression}     Task type (default: classification)
  --n_samples   INT   Total samples (default: 2000)
  --in_features INT   Raw input dimensionality (default: 20)
  --n_concepts  INT   Number of continuous concepts (default: 5)
  --n_outputs   INT   Task output dimensionality (default: 1)
  --alpha       FLOAT Concept loss weight for joint mode (default: 0.5)
  --lr          FLOAT Learning rate (default: 1e-3)
  --n_epochs    INT   Epochs per training phase (default: 100)
  --patience    INT   Early-stopping patience (default: 15)
  --batch_size  INT   Mini-batch size (default: 64)
  --seed        INT   Random seed (default: 42)
  --save_dir    PATH  Directory to save checkpoints and plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from cbm.datasets import SyntheticContinuousDataset, make_dataloaders
from cbm.evaluate import full_evaluation, intervention_gain
from cbm.models import ConceptBottleneckModel
from cbm.training import train
from cbm.utils import plot_training_curves, save_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a CBM on synthetic continuous concepts")
    p.add_argument("--mode", choices=["joint", "sequential", "independent"], default="joint")
    p.add_argument("--task", choices=["classification", "regression"], default="classification")
    p.add_argument("--n_samples", type=int, default=2000)
    p.add_argument("--in_features", type=int, default=20)
    p.add_argument("--n_concepts", type=int, default=5)
    p.add_argument("--n_outputs", type=int, default=1)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n_epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Mode: {args.mode} | Task: {args.task}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_ds, val_ds, test_ds = SyntheticContinuousDataset.splits(
        n_samples=args.n_samples,
        in_features=args.in_features,
        n_concepts=args.n_concepts,
        n_outputs=args.n_outputs if args.task == "regression" or args.n_outputs > 1 else 1,
        task=args.task,
        seed=args.seed,
    )

    train_loader, val_loader, test_loader = make_dataloaders(
        train_ds, val_ds, test_ds, batch_size=args.batch_size
    )

    print(
        f"Dataset: {args.n_samples} samples | "
        f"train/val/test = {len(train_ds)}/{len(val_ds)}/{len(test_ds)}"
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = ConceptBottleneckModel(
        in_features=args.in_features,
        n_concepts=args.n_concepts,
        n_outputs=args.n_outputs,
        encoder_hidden=[128, 64],
        predictor_hidden=[64],
        mode=args.mode,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        task=args.task,
        alpha=args.alpha,
        lr=args.lr,
        n_epochs=args.n_epochs,
        patience=args.patience,
        device=device,
        verbose=True,
    )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    print("\n=== Test Set Evaluation ===")
    metrics = full_evaluation(model, test_loader, task=args.task, device=device)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\n=== Intervention Analysis ===")
    gain = intervention_gain(model, test_loader, task=args.task, device=device)
    for k, v in gain.items():
        print(f"  {k}: {v:.4f}")

    # ------------------------------------------------------------------
    # Save artefacts
    # ------------------------------------------------------------------
    if args.save_dir is not None:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        ckpt_path = save_dir / f"cbm_{args.mode}.pt"
        save_checkpoint(model, ckpt_path, extra={"args": vars(args), "metrics": metrics})
        print(f"\nCheckpoint saved to {ckpt_path}")

        plot_path = save_dir / f"training_curves_{args.mode}.png"
        plot_training_curves(history, save_path=plot_path)


if __name__ == "__main__":
    main()
