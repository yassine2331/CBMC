"""
Training utilities for Concept Bottleneck Models.

The module exposes:

* :func:`train_epoch`   – runs one full training epoch.
* :func:`evaluate_epoch` – evaluates the model on a dataloader.
* :func:`train`          – high-level training loop with early stopping.

Loss convention
---------------
Total loss = ``alpha * concept_loss + (1 - alpha) * task_loss``

For **sequential** mode:
  - Phase 1 (encoder training): ``alpha = 1.0``
  - Phase 2 (predictor training, encoder frozen): ``alpha = 0.0``

For **joint** mode:
  - ``alpha`` is a hyper-parameter in ``(0, 1)``.

For **independent** mode:
  - ``alpha`` can be > 0 (encoder is trained alongside), but the task
    predictor is connected to raw inputs, not the bottleneck.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from cbm.models import ConceptBottleneckModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _concept_loss(
    c_pred: torch.Tensor,
    c_true: torch.Tensor,
) -> torch.Tensor:
    """MSE loss between predicted and true continuous concepts."""
    return nn.functional.mse_loss(c_pred, c_true)


def _task_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    task: str,
) -> torch.Tensor:
    """
    Task loss.

    Parameters
    ----------
    y_pred : ``(B, n_outputs)``
    y_true : ``(B,)`` for classification / ``(B, n_outputs)`` for regression
    task   : ``"classification"`` or ``"regression"``
    """
    if task == "classification":
        n_out = y_pred.shape[-1]
        if n_out == 1:
            # Binary classification – BCEWithLogits expects same shape
            return nn.functional.binary_cross_entropy_with_logits(
                y_pred.squeeze(-1), y_true.float()
            )
        else:
            return nn.functional.cross_entropy(y_pred, y_true.long())
    else:
        return nn.functional.mse_loss(y_pred, y_true.float())


# ---------------------------------------------------------------------------
# Per-epoch routines
# ---------------------------------------------------------------------------

def train_epoch(
    model: ConceptBottleneckModel,
    loader: DataLoader,
    optimizer: Optimizer,
    task: str = "classification",
    alpha: float = 0.5,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Run one training epoch.

    Parameters
    ----------
    model      : CBM model.
    loader     : Training dataloader; batches are ``(x, c_true, y_true)``.
    optimizer  : Pytorch optimiser.
    task       : ``"classification"`` or ``"regression"``.
    alpha      : Concept loss weight in ``[0, 1]``.
    device     : Target device; defaults to CPU.

    Returns
    -------
    dict with keys ``"loss"``, ``"concept_loss"``, ``"task_loss"``.
    """
    device = device or torch.device("cpu")
    model.train()
    total_loss = concept_loss_sum = task_loss_sum = 0.0
    n_batches = 0

    for x, c_true, y_true in loader:
        x = x.to(device)
        c_true = c_true.to(device)
        y_true = y_true.to(device)

        optimizer.zero_grad()
        c_pred, y_pred = model(x)

        c_loss = _concept_loss(c_pred, c_true)
        t_loss = _task_loss(y_pred, y_true, task)
        loss = alpha * c_loss + (1.0 - alpha) * t_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        concept_loss_sum += c_loss.item()
        task_loss_sum += t_loss.item()
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "concept_loss": concept_loss_sum / n_batches,
        "task_loss": task_loss_sum / n_batches,
    }


@torch.no_grad()
def evaluate_epoch(
    model: ConceptBottleneckModel,
    loader: DataLoader,
    task: str = "classification",
    alpha: float = 0.5,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Evaluate the model on *loader* without updating parameters.

    Returns the same keys as :func:`train_epoch`.
    """
    device = device or torch.device("cpu")
    model.eval()
    total_loss = concept_loss_sum = task_loss_sum = 0.0
    n_batches = 0

    for x, c_true, y_true in loader:
        x = x.to(device)
        c_true = c_true.to(device)
        y_true = y_true.to(device)

        c_pred, y_pred = model(x)

        c_loss = _concept_loss(c_pred, c_true)
        t_loss = _task_loss(y_pred, y_true, task)
        loss = alpha * c_loss + (1.0 - alpha) * t_loss

        total_loss += loss.item()
        concept_loss_sum += c_loss.item()
        task_loss_sum += t_loss.item()
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "concept_loss": concept_loss_sum / n_batches,
        "task_loss": task_loss_sum / n_batches,
    }


# ---------------------------------------------------------------------------
# High-level training loop
# ---------------------------------------------------------------------------

def train(
    model: ConceptBottleneckModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    task: str = "classification",
    alpha: float = 0.5,
    lr: float = 1e-3,
    n_epochs: int = 50,
    patience: int = 10,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, list]:
    """
    Full training loop with early stopping on validation loss.

    For **sequential** mode the function automatically runs two phases:

    * Phase 1 (``n_epochs`` epochs): only the concept encoder is
      updated (``alpha = 1.0``, predictor frozen).
    * Phase 2 (``n_epochs`` epochs): encoder frozen, only the predictor
      is updated (``alpha = 0.0``).

    For **joint** and **independent** modes the two components are
    optimised together using the supplied ``alpha``.

    Parameters
    ----------
    model        : CBM model.
    train_loader : Training dataloader.
    val_loader   : Validation dataloader.
    task         : ``"classification"`` or ``"regression"``.
    alpha        : Concept loss weight (ignored in sequential mode).
    lr           : Learning rate.
    n_epochs     : Epochs per phase.
    patience     : Early-stopping patience (epochs without improvement).
    device       : Target device.
    verbose      : Print per-epoch logs.

    Returns
    -------
    history : dict with lists of per-epoch metrics.
    """
    device = device or torch.device("cpu")
    model = model.to(device)

    history: Dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        "train_concept_loss": [],
        "val_concept_loss": [],
        "train_task_loss": [],
        "val_task_loss": [],
    }

    def _run_phase(
        phase_alpha: float,
        freeze_encoder: bool,
        freeze_predictor: bool,
    ) -> None:
        if freeze_encoder:
            model.freeze_encoder()
        else:
            model.unfreeze_encoder()

        for p in model.task_predictor.parameters():
            p.requires_grad_(not freeze_predictor)

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr
        )

        best_val = float("inf")
        stale = 0

        for epoch in range(1, n_epochs + 1):
            tr = train_epoch(model, train_loader, optimizer, task, phase_alpha, device)
            vl = evaluate_epoch(model, val_loader, task, phase_alpha, device)

            history["train_loss"].append(tr["loss"])
            history["val_loss"].append(vl["loss"])
            history["train_concept_loss"].append(tr["concept_loss"])
            history["val_concept_loss"].append(vl["concept_loss"])
            history["train_task_loss"].append(tr["task_loss"])
            history["val_task_loss"].append(vl["task_loss"])

            if verbose:
                print(
                    f"Epoch {epoch:3d} | "
                    f"train_loss={tr['loss']:.4f}  "
                    f"val_loss={vl['loss']:.4f}  "
                    f"c_loss={vl['concept_loss']:.4f}  "
                    f"t_loss={vl['task_loss']:.4f}"
                )

            if vl["loss"] < best_val - 1e-6:
                best_val = vl["loss"]
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch}.")
                    break

    if model.mode == "sequential":
        if verbose:
            print("=== Sequential CBM – Phase 1: training concept encoder ===")
        _run_phase(phase_alpha=1.0, freeze_encoder=False, freeze_predictor=True)

        if verbose:
            print("=== Sequential CBM – Phase 2: training task predictor ===")
        _run_phase(phase_alpha=0.0, freeze_encoder=True, freeze_predictor=False)
    else:
        _run_phase(phase_alpha=alpha, freeze_encoder=False, freeze_predictor=False)

    # Restore all parameters to trainable state.
    model.unfreeze_encoder()
    for p in model.task_predictor.parameters():
        p.requires_grad_(True)

    return history
