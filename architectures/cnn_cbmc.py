"""
Example: CNN backbone + ContinuousBottleneck.
Swap CNN for any encoder that outputs a flat embedding.
"""

import torch.nn as nn
from cbmc import ContinuousBottleneck


class CNNWithBottleneck(nn.Module):
    def __init__(self, n_concepts: int, n_classes: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 512),
        )
        self.bottleneck = ContinuousBottleneck(in_dim=512, n_concepts=n_concepts)
        self.head = nn.Linear(n_concepts, n_classes)

    def forward(self, x):
        z = self.encoder(x)
        concepts = self.bottleneck(z)
        return self.head(concepts), concepts
