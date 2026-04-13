"""
CBM model components.

Three training paradigms are supported:

* **sequential** – concept encoder is trained first (using only concept
  supervision), then frozen before the task predictor is trained on top
  of the frozen concept predictions.

* **joint** – concept encoder and task predictor are trained end-to-end
  with a weighted combination of concept loss and task loss.

* **independent** – concept encoder and task predictor are trained
  independently; the task predictor is trained directly on the raw
  input (no bottleneck at test time).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Generic multi-layer perceptron with ReLU activations."""

    def __init__(
        self,
        in_features: int,
        hidden_dims: List[int],
        out_features: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_features
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Concept encoder
# ---------------------------------------------------------------------------

class ConceptEncoder(nn.Module):
    """
    Maps raw input features to a vector of continuous concept scores.

    Parameters
    ----------
    in_features : int
        Dimensionality of the raw input.
    n_concepts : int
        Number of continuous concepts to predict.
    hidden_dims : list of int
        Hidden layer widths for the MLP backbone.
    dropout : float
        Dropout probability (0 = disabled).
    activation : {"none", "sigmoid", "tanh"}
        Optional activation applied to the concept outputs.
        Use ``"none"`` for unconstrained continuous concepts.
    """

    def __init__(
        self,
        in_features: int,
        n_concepts: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.0,
        activation: str = "none",
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [128, 64]
        self.backbone = MLP(in_features, hidden_dims, n_concepts, dropout)
        activations = {
            "none": nn.Identity(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
        }
        if activation not in activations:
            raise ValueError(
                f"activation must be one of {list(activations)}, got '{activation}'"
            )
        self.act = activations[activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.backbone(x))


# ---------------------------------------------------------------------------
# Task predictor
# ---------------------------------------------------------------------------

class TaskPredictor(nn.Module):
    """
    Maps concept representations (or raw inputs) to task outputs.

    Parameters
    ----------
    in_features : int
        Dimensionality of the input (n_concepts for CBM, raw input dim
        for the independent baseline).
    n_outputs : int
        Number of task outputs (1 for regression / binary, >1 for
        multi-class classification).
    hidden_dims : list of int
        Hidden layer widths.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        in_features: int,
        n_outputs: int = 1,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [64]
        self.net = MLP(in_features, hidden_dims, n_outputs, dropout)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        return self.net(c)


# ---------------------------------------------------------------------------
# Concept Bottleneck Model
# ---------------------------------------------------------------------------

class ConceptBottleneckModel(nn.Module):
    """
    Full Concept Bottleneck Model with continuous concepts.

    The model is composed of a :class:`ConceptEncoder` followed by a
    :class:`TaskPredictor`.  The ``mode`` parameter controls how the
    two components interact at training time (the caller is still
    responsible for implementing the correct training schedule; see
    :mod:`cbm.training`).

    Parameters
    ----------
    in_features : int
        Raw input dimensionality.
    n_concepts : int
        Number of continuous concept dimensions.
    n_outputs : int
        Task output dimensionality.
    encoder_hidden : list of int
        Hidden layer widths for the concept encoder.
    predictor_hidden : list of int
        Hidden layer widths for the task predictor.
    dropout : float
        Dropout probability applied in both sub-networks.
    concept_activation : str
        Activation for concept outputs (``"none"``, ``"sigmoid"``,
        ``"tanh"``).
    mode : {"joint", "sequential", "independent"}
        Training paradigm.  In ``"independent"`` mode the task
        predictor receives the *raw input* rather than the bottleneck.
    """

    MODES = ("joint", "sequential", "independent")

    def __init__(
        self,
        in_features: int,
        n_concepts: int,
        n_outputs: int = 1,
        encoder_hidden: Optional[List[int]] = None,
        predictor_hidden: Optional[List[int]] = None,
        dropout: float = 0.0,
        concept_activation: str = "none",
        mode: str = "joint",
    ) -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got '{mode}'")
        self.mode = mode
        self.n_concepts = n_concepts
        self.n_outputs = n_outputs

        self.concept_encoder = ConceptEncoder(
            in_features=in_features,
            n_concepts=n_concepts,
            hidden_dims=encoder_hidden or [128, 64],
            dropout=dropout,
            activation=concept_activation,
        )

        # In independent mode the task predictor bypasses the bottleneck.
        predictor_in = in_features if mode == "independent" else n_concepts
        self.task_predictor = TaskPredictor(
            in_features=predictor_in,
            n_outputs=n_outputs,
            hidden_dims=predictor_hidden or [64],
            dropout=dropout,
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        intervention: Optional[torch.Tensor] = None,
        intervention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor of shape ``(B, in_features)``
            Raw input batch.
        intervention : Tensor of shape ``(B, n_concepts)``, optional
            Ground-truth concept values to inject (test-time intervention).
        intervention_mask : bool Tensor of shape ``(B, n_concepts)`` or
            ``(n_concepts,)``, optional
            Mask indicating which concepts to replace with ground-truth
            values.  Requires *intervention* to be provided.

        Returns
        -------
        concept_preds : Tensor of shape ``(B, n_concepts)``
            Predicted concept values.
        task_preds : Tensor of shape ``(B, n_outputs)``
            Task predictions.
        """
        concept_preds = self.concept_encoder(x)

        # Apply test-time intervention if requested.
        if intervention is not None and intervention_mask is not None:
            mask = intervention_mask.bool()
            concept_preds = torch.where(mask, intervention, concept_preds)

        if self.mode == "independent":
            task_preds = self.task_predictor(x)
        else:
            task_preds = self.task_predictor(concept_preds)

        return concept_preds, task_preds

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def freeze_encoder(self) -> None:
        """Freeze concept encoder parameters (used for sequential training)."""
        for p in self.concept_encoder.parameters():
            p.requires_grad_(False)

    def unfreeze_encoder(self) -> None:
        """Unfreeze concept encoder parameters."""
        for p in self.concept_encoder.parameters():
            p.requires_grad_(True)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return concept predictions without computing task output."""
        return self.concept_encoder(x)
