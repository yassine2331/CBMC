"""
CEM_v2 — Shared-Backbone Concept Embedding Model
=================================================
Architectural difference from CEM (v1):
    CEM v1 — each concept has a fully independent encoder from the raw input
              to its positive and negative embeddings.
    CEM v2 — a single shared backbone first compresses the raw input into a
              compact representation, then lightweight per-concept heads
              produce the positive and negative embeddings.

Trade-off:
    + Fewer parameters when n_concepts is large.
    + Low-level feature extraction is shared, which can help when concepts
      share common input patterns.
    - Less expressive per-concept separation at the input level.

Usage
-----
    from models.CBMs.CEM_v2 import CEM_v2
    model = CEM_v2(input_dim=128, n_concepts=10)
    embeddings, predicted_concepts = model(x)
"""

import torch
import torch.nn as nn


class CEM_v2(nn.Module):
    """Concept Embedding Model with a shared backbone encoder.

    Args:
        input_dim (int):     Dimension of the raw input features.
        n_concepts (int):    Number of concepts to predict.
        hidden_dim (int):    Width of hidden layers in the shared backbone.
        embedding_dim (int): Dimension of each concept's pos/neg embedding.
        shared_dim (int):    Output dimension of the shared backbone.
                             Defaults to ``hidden_dim * 2``.
        depth (int):         Number of hidden layers in the shared backbone.
        concept_names (list, optional): If provided, sets n_concepts.
        dropout (float):     Dropout rate.
    """

    def __init__(
        self,
        input_dim: int,
        n_concepts: int = None,
        hidden_dim: int = 64,
        embedding_dim: int = 16,
        shared_dim: int = None,
        depth: int = 2,
        concept_names=None,
        dropout: float = 0.2,
        **kwargs,  # absorbs unused kwargs (e.g. feature_names, output_dim)
    ):
        super().__init__()

        self.n_concepts = n_concepts if concept_names is None else len(concept_names)
        if n_concepts is not None and concept_names is not None and n_concepts != len(concept_names):
            raise ValueError(
                f"n_concepts={n_concepts} conflicts with len(concept_names)={len(concept_names)}. "
                "Provide only one or ensure they match."
            )
        self.embedding_dim = embedding_dim
        shared_dim = shared_dim or hidden_dim * 2

        # -------------------------------------------------------------- #
        # Shared backbone: raw input → shared representation              #
        # -------------------------------------------------------------- #
        backbone_layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        for _ in range(depth - 1):
            backbone_layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        backbone_layers.append(nn.Linear(hidden_dim, shared_dim))
        self.backbone = nn.Sequential(*backbone_layers)

        # -------------------------------------------------------------- #
        # Per-concept heads (one linear layer each)                       #
        # -------------------------------------------------------------- #
        self.pos_heads = nn.ModuleList(
            [nn.Linear(shared_dim, embedding_dim) for _ in range(self.n_concepts)]
        )
        self.neg_heads = nn.ModuleList(
            [nn.Linear(shared_dim, embedding_dim) for _ in range(self.n_concepts)]
        )

        # -------------------------------------------------------------- #
        # Concept value predictor                                         #
        # -------------------------------------------------------------- #
        self.concept_predictor = nn.Linear(
            embedding_dim * 2 * self.n_concepts, self.n_concepts
        )

    def forward(self, x, interventions=None):
        """
        Args:
            x (Tensor):            [batch, input_dim]
            interventions (Tensor, optional): [batch, n_concepts]
                Pre-specified concept values; overrides predictions when given.

        Returns:
            final_embeddings_flat (Tensor): [batch, embedding_dim * n_concepts]
            predicted_concepts (Tensor):    [batch, n_concepts]
        """
        shared = self.backbone(x)  # [batch, shared_dim]

        # Per-concept embeddings stacked: [batch, embedding_dim, n_concepts]
        pos_embeds = torch.stack([h(shared) for h in self.pos_heads], dim=2)
        neg_embeds = torch.stack([h(shared) for h in self.neg_heads], dim=2)

        # Predict concept values
        combined = torch.cat([pos_embeds, neg_embeds], dim=1).view(x.size(0), -1)
        predicted_concepts = self.concept_predictor(combined)  # [batch, n_concepts]

        # Compute mixing weights (linear activation, clamped to [0, 1])
        source = predicted_concepts if interventions is None else interventions
        weights = torch.clamp(source.unsqueeze(1) * 0.5 + 0.5, 0.0, 1.0)

        # Weighted combination of pos/neg embeddings
        final_embeddings = pos_embeds * weights + neg_embeds * (1 - weights)
        final_embeddings_flat = final_embeddings.view(x.size(0), -1)

        return final_embeddings_flat, predicted_concepts

    @classmethod
    def from_config(cls, config):
        """Build a CEM_v2 from a TrainingConfig."""
        return cls(
            input_dim=config.input_dim,
            n_concepts=config.num_concepts,
            hidden_dim=config.hidden_dim,
            embedding_dim=config.embedding_dim,
            shared_dim=getattr(config, "shared_dim", None),
            depth=config.depth,
            dropout=config.dropout,
        )


# --- Test Block ---
if __name__ == "__main__":
    batch_size = 4
    input_dim = 128
    n_concepts = 5
    embedding_dim = 16

    x = torch.randn(batch_size, input_dim)
    model = CEM_v2(input_dim=input_dim, n_concepts=n_concepts, embedding_dim=embedding_dim)

    embeddings, concepts = model(x)
    print(f"Input shape:      {x.shape}")           # [4, 128]
    print(f"Embeddings shape: {embeddings.shape}")  # [4, 80] (5 concepts * 16 dim)
    print(f"Concepts shape:   {concepts.shape}")    # [4, 5]
    print("Concepts:", concepts)
