"""
CNN + categorical Concept Embedding Model (Case 2).

Backbone encodes the image to a flat vector.
CEMCategorical (or CEMCategoricalPerConcept) maps it to:
  - embeddings: (B, n_concepts * embedding_dim)  -> fed to the head
  - logits:     state logits per concept         -> supervise with cross-entropy

Mirrors architectures/cnn_cem.py so the two are interchangeable in the
experiment runner, except that concepts are supervised with class indices
rather than scalars.
"""

import torch.nn as nn
from typing import Union

from cbmc.concepts import CEMCategorical, CEMCategoricalPerConcept
from cbmc.configs import CNNConfig, CNNRegressionConfig, CEMCategoricalConfig


class CNNwithCEMCategorical(nn.Module):
    """
    Args:
        backbone_cfg : CNN encoder config (CNNConfig or CNNRegressionConfig)
        cem_cfg      : CEMCategoricalConfig
        n_outputs    : head output size. Required with CNNRegressionConfig.

    If cem_cfg.n_states is non-empty each concept gets its own state count and
    `concepts` is a list of [B, n_i] tensors; otherwise every concept has
    n_classes states and `concepts` is one [B, C, n] tensor.
    """

    def __init__(self, backbone_cfg: Union[CNNConfig, CNNRegressionConfig],
                 cem_cfg: CEMCategoricalConfig, n_outputs: int = None):
        super().__init__()
        if n_outputs is not None:
            n_out = n_outputs
        elif hasattr(backbone_cfg, "n_classes"):
            n_out = backbone_cfg.n_classes
        else:
            n_out = backbone_cfg.n_outputs

        # Encoder — same conv stack as CNNwithCEM
        layers = []
        in_ch = backbone_cfg.in_channels
        for out_ch in backbone_cfg.conv_channels:
            layers += [nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
            in_ch = out_ch
        layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten()]
        self.encoder = nn.Sequential(*layers)

        enc_out = backbone_cfg.conv_channels[-1]

        shared = dict(embedding_dim=cem_cfg.embedding_dim,
                      hidden_dim=cem_cfg.hidden_dim,
                      depth=cem_cfg.depth,
                      dropout=cem_cfg.dropout,
                      normalize=cem_cfg.normalize)

        if cem_cfg.n_states:
            self.cem = CEMCategoricalPerConcept(
                input_dim=enc_out, n_states=cem_cfg.n_states, **shared)
        else:
            self.cem = CEMCategorical(
                input_dim=enc_out, n_concepts=cem_cfg.n_concepts,
                n_classes=cem_cfg.n_classes, **shared)

        self.head = nn.Linear(self.cem.output_dim, n_out)

    def forward(self, x, interventions=None, mask=None):
        z = self.encoder(x)
        embeddings, concepts = self.cem(z, interventions, mask)
        return self.head(embeddings), concepts

    # Convenience passthroughs so training code does not need to know which
    # variant is in use.
    def concept_loss(self, concepts, targets):
        return self.cem.concept_loss(concepts, targets)

    def predicted_states(self, concepts):
        return self.cem.predicted_states(concepts)
