"""
Dataset utilities for CBM experiments with continuous concepts.

``SyntheticContinuousDataset``
    Generates a toy dataset where ground-truth concepts are continuous
    random variables and the label is a deterministic (or noisy)
    function of those concepts.  This makes it easy to measure how
    well a CBM recovers the latent concept structure.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

class SyntheticContinuousDataset(Dataset):
    """
    Toy dataset with continuous latent concepts.

    Data generation process
    -----------------------
    1. Raw input  ``x  ~ N(0, I_{d})``         shape ``(N, in_features)``
    2. True concepts ``c = x @ W_c + noise``    shape ``(N, n_concepts)``
    3. Label  ``y = label_fn(c) + noise``       shape ``(N, n_outputs)``

    The weight matrix ``W_c`` is randomly initialised once and fixed for
    the lifetime of the dataset, so the concepts are linear projections
    of the raw input (plus optional noise).

    Parameters
    ----------
    n_samples : int
        Number of data points.
    in_features : int
        Dimensionality of the raw input ``x``.
    n_concepts : int
        Number of continuous concepts.
    n_outputs : int
        Number of task outputs.
    label_fn : callable, optional
        Function ``(c: Tensor) -> Tensor`` mapping concepts to labels.
        Defaults to a random linear combination followed by a sign
        function for binary classification.
    concept_noise : float
        Std of Gaussian noise added to the true concept values.
    label_noise : float
        Std of Gaussian noise added to the label.
    task : {"classification", "regression"}
        Controls the default ``label_fn`` when none is provided.
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        in_features: int = 20,
        n_concepts: int = 5,
        n_outputs: int = 1,
        label_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        concept_noise: float = 0.1,
        label_noise: float = 0.05,
        task: str = "classification",
        seed: Optional[int] = 42,
    ) -> None:
        super().__init__()
        if task not in ("classification", "regression"):
            raise ValueError("task must be 'classification' or 'regression'")
        rng = np.random.default_rng(seed)

        # --- weight matrix: in_features → n_concepts ----------------------
        W_c = rng.standard_normal((in_features, n_concepts)).astype(np.float32)
        W_c /= np.linalg.norm(W_c, axis=0, keepdims=True) + 1e-8

        # --- raw inputs ----------------------------------------------------
        X = rng.standard_normal((n_samples, in_features)).astype(np.float32)

        # --- continuous concepts -------------------------------------------
        C = X @ W_c
        if concept_noise > 0:
            C += rng.normal(0, concept_noise, C.shape).astype(np.float32)

        # --- labels --------------------------------------------------------
        X_t = torch.from_numpy(X)
        C_t = torch.from_numpy(C)

        if label_fn is not None:
            Y_t = label_fn(C_t)
        else:
            # Default: random linear combination of concepts
            w_y = torch.from_numpy(
                rng.standard_normal((n_concepts, n_outputs)).astype(np.float32)
            )
            logits = C_t @ w_y  # (N, n_outputs)
            if task == "classification":
                if n_outputs == 1:
                    # Binary: threshold at 0; squeeze to shape (N,)
                    Y_t = (logits > 0).float().squeeze(-1)
                else:
                    # Multi-class: argmax → shape (N,)
                    Y_t = logits.argmax(dim=1)
            else:
                Y_t = logits  # regression

        if label_noise > 0 and task == "regression":
            Y_t = Y_t + torch.randn_like(Y_t) * label_noise

        self.X = X_t
        self.C = C_t
        self.Y = Y_t
        self.task = task
        self.in_features = in_features
        self.n_concepts = n_concepts
        self.n_outputs = n_outputs

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.C[idx], self.Y[idx]

    # ------------------------------------------------------------------
    # Convenience factory
    # ------------------------------------------------------------------

    @classmethod
    def splits(
        cls,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        **kwargs,
    ) -> Tuple["SyntheticContinuousDataset", "SyntheticContinuousDataset", "SyntheticContinuousDataset"]:
        """
        Create a single dataset and split it into train / val / test.

        All keyword arguments are forwarded to ``__init__``.
        """
        full = cls(**kwargs)
        n = len(full)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val
        seed = kwargs.get("seed", 42)
        gen = torch.Generator().manual_seed(seed if seed is not None else 0)
        return random_split(full, [n_train, n_val, n_test], generator=gen)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int = 64,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader)."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader
