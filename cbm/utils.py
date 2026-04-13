"""
Miscellaneous utility helpers.

set_seed     : Set global random seeds for reproducibility.
save_checkpoint / load_checkpoint : Serialise and restore model state.
plot_training_curves : Visualise loss curves from the training history.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set seeds for Python, NumPy and PyTorch (CPU & CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    path: str | Path,
    extra: Optional[dict] = None,
) -> None:
    """
    Save model weights (and optional metadata) to *path*.

    Parameters
    ----------
    model : torch Module to save.
    path  : Destination file path (``*.pt`` or ``*.pth``).
    extra : Additional dict entries stored alongside ``state_dict``.
    """
    payload = {"state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(
    model: nn.Module,
    path: str | Path,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Load weights from *path* into *model* and return the full checkpoint dict.

    Parameters
    ----------
    model  : Module whose weights will be updated in-place.
    path   : Source file path.
    device : Map location (defaults to CPU).

    Returns
    -------
    The full checkpoint dict (including any ``extra`` fields saved
    alongside the state dict).
    """
    map_location = device or torch.device("cpu")
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["state_dict"])
    return checkpoint


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_training_curves(
    history: Dict[str, list],
    save_path: Optional[str | Path] = None,
) -> None:
    """
    Plot training and validation losses from the history dict returned by
    :func:`cbm.training.train`.

    Parameters
    ----------
    history   : History dict (keys: train_loss, val_loss, …).
    save_path : If provided the figure is saved to this path instead of
                being displayed interactively.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for plotting. Install it with: pip install matplotlib")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    metrics = [
        ("loss", "Total Loss"),
        ("concept_loss", "Concept Loss (MSE)"),
        ("task_loss", "Task Loss"),
    ]

    for ax, (key, title) in zip(axes, metrics):
        if f"train_{key}" in history:
            ax.plot(history[f"train_{key}"], label="train")
        if f"val_{key}" in history:
            ax.plot(history[f"val_{key}"], label="val")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        print(f"Saved training curves to {save_path}")
    else:
        plt.show()

    plt.close(fig)
