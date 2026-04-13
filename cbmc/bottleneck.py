import torch
import torch.nn as nn


class ContinuousBottleneck(nn.Module):
    """
    The core CBMC module. Architecture-agnostic — insert between any encoder and task head.

    Usage:
        bottleneck = ContinuousBottleneck(in_dim=512, n_concepts=50)
        concepts = bottleneck(encoder_output)   # shape: (B, n_concepts)
    """

    def __init__(self, in_dim: int, n_concepts: int):
        super().__init__()
        self.in_dim = in_dim
        self.n_concepts = n_concepts
        self.concept_layer = nn.Linear(in_dim, n_concepts)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.concept_layer(z)
