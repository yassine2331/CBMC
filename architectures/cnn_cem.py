"""
CNN + Concept Embedding Model (CEM).

Backbone encodes the image to a flat vector.
ConceptBottleneck (from cbmc.concepts.cem) maps it to:
  - embeddings: (B, n_concepts * embedding_dim)  → fed to the head
  - concepts:   (B, n_concepts)                  → raw concept scores

Head maps embeddings to task output (classification or regression).

Returns (logits/preds, concepts) so concept values can be logged or supervised.
Supports test-time interventions via the `interventions` and `mask` arguments.
"""

import torch.nn as nn
from cbmc.concepts import ConceptBottleneck
from cbmc.configs import CNNConfig, CEMConfig


class CNNwithCEM(nn.Module):
    def __init__(self, backbone_cfg: CNNConfig, cem_cfg: CEMConfig, n_outputs: int = None):
        """
        Args:
            backbone_cfg : CNN encoder config
            cem_cfg      : CEM config
            n_outputs    : head output size. Defaults to backbone_cfg.n_classes.
        """
        super().__init__()
        n_out = n_outputs if n_outputs is not None else backbone_cfg.n_classes

        # Encoder
        layers = []
        in_ch = backbone_cfg.in_channels
        for out_ch in backbone_cfg.conv_channels:
            layers += [nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
            in_ch = out_ch
        layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten()]
        self.encoder = nn.Sequential(*layers)

        enc_out = backbone_cfg.conv_channels[-1]

        # CEM
        self.cem = ConceptBottleneck(
            input_dim     = enc_out,
            n_concepts    = cem_cfg.n_concepts,
            embedding_dim = cem_cfg.embedding_dim,
            hidden_dim    = cem_cfg.hidden_dim,
            depth         = cem_cfg.depth,
            dropout       = cem_cfg.dropout,
        )

        # Head: embedding vector → task output
        self.head = nn.Linear(self.cem.output_dim, n_out)

    def forward(self, x, interventions=None, mask=None):
        z                    = self.encoder(x)
        embeddings, concepts = self.cem(z, interventions, mask)   # (B, E*C), (B, C)
        return self.head(embeddings), concepts
