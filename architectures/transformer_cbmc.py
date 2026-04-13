"""
Example: Transformer encoder + ContinuousBottleneck.
Uses the [CLS] token output as the bottleneck input.
"""

import torch.nn as nn
from cbmc import ContinuousBottleneck


class TransformerWithBottleneck(nn.Module):
    def __init__(self, d_model: int, nhead: int, n_layers: int, n_concepts: int, n_classes: int):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.bottleneck = ContinuousBottleneck(in_dim=d_model, n_concepts=n_concepts)
        self.head = nn.Linear(n_concepts, n_classes)

    def forward(self, x):
        z = self.encoder(x)[:, 0]  # CLS token
        concepts = self.bottleneck(z)
        return self.head(concepts), concepts
