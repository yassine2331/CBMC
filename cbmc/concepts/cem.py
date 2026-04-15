"""
concepts/cem.py
---------------
Continuous Concept Bottleneck block based on the Concept Embedding Model (CEM).

Drop this anywhere in your architecture:

    from concepts import ConceptBottleneck

    cbm = ConceptBottleneck(
        input_dim   = 128,      # dim of the pre-concept embedding coming from your backbone
        n_concepts  = 8,        # how many concepts to model
        embedding_dim = 16,     # dim of each pos/neg concept embedding
        hidden_dim  = 64,       # width of internal MLPs
        depth       = 2,        # number of hidden layers in each MLP
        dropout     = 0.2,
    )

    # --- forward (no intervention) ---
    embeddings, concepts = cbm(z)
    # embeddings : [B, embedding_dim * n_concepts]   ← feed to your decoder / head
    # concepts   : [B, n_concepts]                   ← supervise with concept labels

    # --- forward (with intervention) ---
    embeddings, concepts = cbm(z, interventions=c_true)
    # interventions : [B, n_concepts]  true (or partially-true) concept values
    #                 pass None for any concept you do NOT want to intervene on
    #                 by providing a mask (see InterventionMask helper below)
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mlp(in_dim: int, hidden_dim: int, out_dim: int, depth: int, dropout: float) -> nn.Sequential:
    """Build a simple MLP: Linear → (ReLU → Dropout → Linear) × depth → Linear."""
    layers: list[nn.Module] = []
    curr = in_dim
    for _ in range(depth):
        layers += [nn.Linear(curr, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        curr = hidden_dim
    layers.append(nn.Linear(curr, out_dim))
    return nn.Sequential(*layers)


class _EmbeddingBank(nn.Module):
    """
    For each concept i, learn two embedding functions:
        φ⁺ᵢ(x) → ℝ^embedding_dim   (positive embedding)
        φ⁻ᵢ(x) → ℝ^embedding_dim   (negative embedding)

    Output shapes:
        pos : [B, embedding_dim, n_concepts]
        neg : [B, embedding_dim, n_concepts]
    """

    def __init__(self, input_dim: int, n_concepts: int,
                 hidden_dim: int, embedding_dim: int,
                 depth: int, dropout: float) -> None:
        super().__init__()
        self.pos_nets = nn.ModuleList([
            _mlp(input_dim, hidden_dim, embedding_dim, depth, dropout)
            for _ in range(n_concepts)
        ])
        self.neg_nets = nn.ModuleList([
            _mlp(input_dim, hidden_dim, embedding_dim, depth, dropout)
            for _ in range(n_concepts)
        ])

    def forward(self, x: torch.Tensor):
        pos = torch.stack([net(x) for net in self.pos_nets], dim=2)  # [B, E, C]
        neg = torch.stack([net(x) for net in self.neg_nets], dim=2)  # [B, E, C]
        return pos, neg


class _ConceptPredictor(nn.Module):
    """
    For each concept i, predict a scalar concept value from the concatenated
    (pos ‖ neg) embedding of that concept.

        input  : [B, embedding_dim * 2, n_concepts]
        output : [B, n_concepts]   (raw logits, no activation)
    """

    def __init__(self, n_concepts: int, embedding_dim: int,
                 hidden_dim: int, depth: int, dropout: float) -> None:
        super().__init__()
        self.nets = nn.ModuleList([
            _mlp(embedding_dim * 2, hidden_dim, 1, depth, dropout)
            for _ in range(n_concepts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, E*2, C]
        out = torch.cat([net(x[:, :, i]) for i, net in enumerate(self.nets)], dim=1)
        return out  # [B, C]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ConceptBottleneck(nn.Module):
    """
    Continuous Concept Bottleneck block.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the pre-concept embedding (output of your backbone).
    n_concepts : int
        Number of concepts to model.
    embedding_dim : int
        Size of each positive/negative concept embedding vector.
    hidden_dim : int
        Width of all internal MLP hidden layers.
    depth : int
        Number of hidden layers in each internal MLP.
    dropout : float
        Dropout probability (applied during training).

    Inputs
    ------
    x : Tensor [B, input_dim]
        Pre-concept embedding from your backbone.
    interventions : Tensor [B, n_concepts] or None
        If provided, these concept values are used instead of the predicted ones
        to compute the final blended embeddings.  Useful for test-time concept
        intervention or concept-supervised training.
        Values are expected in **[-1, 1]** (same scale as the raw predictions).
        Pass ``None`` (default) to use the model's own predictions.
    intervention_mask : Tensor [B, n_concepts] or None
        Binary mask (1 = intervene, 0 = keep model prediction) that lets you
        intervene on only a *subset* of concepts per sample.  Ignored when
        ``interventions`` is None.  Defaults to all-ones when interventions
        are provided and no mask is given.

    Outputs
    -------
    embeddings : Tensor [B, embedding_dim * n_concepts]
        Blended concept embeddings ready to be fed to any downstream module
        (decoder, classifier head, …).
    concepts : Tensor [B, n_concepts]
        Raw concept predictions (no activation).  Apply your own activation
        (e.g. sigmoid, tanh, identity) depending on your concept supervision.

    Example
    -------
    >>> cbm = ConceptBottleneck(input_dim=128, n_concepts=8, embedding_dim=16)
    >>> z = torch.randn(32, 128)
    >>> embeddings, concepts = cbm(z)
    >>> embeddings.shape
    torch.Size([32, 128])
    >>> concepts.shape
    torch.Size([32, 8])

    Intervening on all concepts:
    >>> c_true = torch.randn(32, 8)
    >>> embeddings_int, _ = cbm(z, interventions=c_true)

    Intervening on only the first 3 concepts (mask = 1 → intervene):
    >>> mask = torch.zeros(32, 8)
    >>> mask[:, :3] = 1.0
    >>> embeddings_partial, _ = cbm(z, interventions=c_true, intervention_mask=mask)
    """

    def __init__(
        self,
        input_dim: int,
        n_concepts: int,
        embedding_dim: int = 16,
        hidden_dim: int = 64,
        depth: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.n_concepts = n_concepts
        self.embedding_dim = embedding_dim

        self.embedding_bank = _EmbeddingBank(
            input_dim=input_dim,
            n_concepts=n_concepts,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            depth=depth,
            dropout=dropout,
        )

        self.concept_predictor = _ConceptPredictor(
            n_concepts=n_concepts,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
        )

    # ------------------------------------------------------------------
    @property
    def output_dim(self) -> int:
        """Dimensionality of the embedding output: embedding_dim × n_concepts."""
        return self.embedding_dim * self.n_concepts

    # ------------------------------------------------------------------
    def _blend(
        self,
        pos: torch.Tensor,          # [B, E, C]
        neg: torch.Tensor,          # [B, E, C]
        concept_scores: torch.Tensor,  # [B, C]  raw scores
    ) -> torch.Tensor:
        """
        Compute the blended embedding:
            w  = clamp(score * 0.5 + 0.5, 0, 1)   ← linear gate in [0,1]
            e  = w * pos + (1-w) * neg
        Returns [B, E, C].
        """
        w = torch.clamp(concept_scores * 0.5 + 0.5, 0.0, 1.0)  # [B, C]
        w = w.unsqueeze(1)                                        # [B, 1, C]
        return pos * w + neg * (1.0 - w)                          # [B, E, C]

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        interventions: Optional[torch.Tensor] = None,
        intervention_mask: Optional[torch.Tensor] = None,
    ):
        # 1. Compute positive and negative embeddings for every concept
        pos, neg = self.embedding_bank(x)                    # [B, E, C] each

        # 2. Predict concept scores from concatenated embeddings
        combined = torch.cat([pos, neg], dim=1)              # [B, E*2, C]
        predicted_concepts = self.concept_predictor(combined)  # [B, C]

        # 3. Determine the scores used for blending
        if interventions is None:
            blend_scores = predicted_concepts
        else:
            if intervention_mask is None:
                # Full intervention: override all concepts
                blend_scores = interventions
            else:
                # Partial intervention: mix predicted and provided per concept
                blend_scores = (
                    intervention_mask * interventions
                    + (1.0 - intervention_mask) * predicted_concepts
                )

        # 4. Blend pos/neg embeddings using gated weights
        final_embeddings = self._blend(pos, neg, blend_scores)  # [B, E, C]

        # 5. Flatten to [B, E * C] for downstream modules
        final_embeddings_flat = final_embeddings.view(x.size(0), -1)

        return final_embeddings_flat, predicted_concepts


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    B, D, C, E = 8, 128, 6, 16

    cbm = ConceptBottleneck(input_dim=D, n_concepts=C, embedding_dim=E)
    print(cbm)
    print(f"\noutput_dim property : {cbm.output_dim}")  # 96

    z = torch.randn(B, D)

    # No intervention
    emb, concepts = cbm(z)
    print(f"\n[No intervention]")
    print(f"  embeddings : {emb.shape}")    # [8, 96]
    print(f"  concepts   : {concepts.shape}")  # [8, 6]

    # Full intervention
    c_true = torch.randn(B, C)
    emb_int, _ = cbm(z, interventions=c_true)
    print(f"\n[Full intervention]")
    print(f"  embeddings : {emb_int.shape}")

    # Partial intervention (first 3 concepts only)
    mask = torch.zeros(B, C)
    mask[:, :3] = 1.0
    emb_part, _ = cbm(z, interventions=c_true, intervention_mask=mask)
    print(f"\n[Partial intervention — concepts 0-2]")
    print(f"  embeddings : {emb_part.shape}")

    # Verify partial is not equal to full or none
    assert not torch.allclose(emb_part, emb_int), "partial should differ from full"
    assert not torch.allclose(emb_part, emb),     "partial should differ from none"
    print("\nAll assertions passed ✓")