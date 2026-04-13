"""
Evaluation metrics for Concept Bottleneck Models.

Functions
---------
concept_mae       : Mean absolute error on concept predictions.
concept_r2        : Coefficient of determination for concept predictions.
task_accuracy     : Classification accuracy.
task_r2           : Regression R² for task predictions.
intervention_gain : Task-metric gain when ground-truth concepts are provided.
full_evaluation   : Convenience wrapper that returns all metrics at once.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cbm.models import ConceptBottleneckModel


# ---------------------------------------------------------------------------
# Per-batch / tensor metrics
# ---------------------------------------------------------------------------

def concept_mae(c_pred: torch.Tensor, c_true: torch.Tensor) -> float:
    """Mean absolute error across all concept dimensions."""
    return (c_pred - c_true).abs().mean().item()


def concept_r2(c_pred: torch.Tensor, c_true: torch.Tensor) -> float:
    """
    Coefficient of determination (R²) averaged over concept dimensions.

    R² = 1 - SS_res / SS_tot
    """
    ss_res = ((c_true - c_pred) ** 2).sum(dim=0)
    ss_tot = ((c_true - c_true.mean(dim=0)) ** 2).sum(dim=0)
    r2_per_concept = 1 - ss_res / (ss_tot + 1e-8)
    return r2_per_concept.mean().item()


def task_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Classification accuracy.

    Parameters
    ----------
    y_pred : ``(B, n_classes)`` logits **or** ``(B, 1)`` binary logits.
    y_true : ``(B,)`` class indices or ``(B,)`` binary labels.
    """
    if y_pred.shape[-1] == 1:
        preds = (y_pred.squeeze(-1) > 0).long()
    else:
        preds = y_pred.argmax(dim=-1)
    return (preds == y_true.long()).float().mean().item()


def task_r2(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Regression R² for task predictions."""
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return (1 - ss_res / (ss_tot + 1e-8)).item()


# ---------------------------------------------------------------------------
# Dataloader-level evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def full_evaluation(
    model: ConceptBottleneckModel,
    loader: DataLoader,
    task: str = "classification",
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Evaluate a CBM on all samples in *loader*.

    Returns
    -------
    dict with keys:
      ``concept_mae``, ``concept_r2``,
      ``task_accuracy`` (classification) **or** ``task_r2`` (regression).
    """
    device = device or torch.device("cpu")
    model.eval()

    all_c_pred, all_c_true = [], []
    all_y_pred, all_y_true = [], []

    for x, c_true, y_true in loader:
        x = x.to(device)
        c_true = c_true.to(device)
        y_true = y_true.to(device)

        c_pred, y_pred = model(x)

        all_c_pred.append(c_pred.cpu())
        all_c_true.append(c_true.cpu())
        all_y_pred.append(y_pred.cpu())
        all_y_true.append(y_true.cpu())

    c_pred_all = torch.cat(all_c_pred)
    c_true_all = torch.cat(all_c_true)
    y_pred_all = torch.cat(all_y_pred)
    y_true_all = torch.cat(all_y_true)

    metrics: Dict[str, float] = {
        "concept_mae": concept_mae(c_pred_all, c_true_all),
        "concept_r2": concept_r2(c_pred_all, c_true_all),
    }

    if task == "classification":
        metrics["task_accuracy"] = task_accuracy(y_pred_all, y_true_all)
    else:
        metrics["task_r2"] = task_r2(y_pred_all, y_true_all)

    return metrics


@torch.no_grad()
def intervention_gain(
    model: ConceptBottleneckModel,
    loader: DataLoader,
    task: str = "classification",
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Measure how much task performance improves when ground-truth concepts
    are injected at test time (full intervention on all concepts).

    Returns
    -------
    dict with keys ``"baseline"`` and ``"intervened"`` (both are the
    task metric value; higher is better for accuracy / R²).
    """
    device = device or torch.device("cpu")
    model.eval()

    all_y_pred_base, all_y_pred_int, all_y_true = [], [], []

    for x, c_true, y_true in loader:
        x = x.to(device)
        c_true = c_true.to(device)
        y_true = y_true.to(device)

        # Baseline (no intervention)
        _, y_pred_base = model(x)

        # Full intervention: replace all concepts with ground truth
        mask = torch.ones(c_true.shape, dtype=torch.bool, device=device)
        _, y_pred_int = model(x, intervention=c_true, intervention_mask=mask)

        all_y_pred_base.append(y_pred_base.cpu())
        all_y_pred_int.append(y_pred_int.cpu())
        all_y_true.append(y_true.cpu())

    y_pred_base_all = torch.cat(all_y_pred_base)
    y_pred_int_all = torch.cat(all_y_pred_int)
    y_true_all = torch.cat(all_y_true)

    if task == "classification":
        metric_fn = task_accuracy
    else:
        metric_fn = task_r2

    return {
        "baseline": metric_fn(y_pred_base_all, y_true_all),
        "intervened": metric_fn(y_pred_int_all, y_true_all),
    }
