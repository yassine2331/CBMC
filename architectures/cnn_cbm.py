"""
CNN + scalar Concept Bottleneck Model (CBM).

Backbone encodes the image to a flat vector.
ContinuousBottleneck maps it to n_concepts scalar values.
Head maps concepts to task output (classification or regression).

Returns (logits/preds, concepts) so concept values can be logged or supervised.
"""

import torch.nn as nn
from cbmc import ContinuousBottleneck
from cbmc.configs import CNNConfig, CBMConfig


class CNNwithCBM(nn.Module):
    def __init__(self, backbone_cfg: CNNConfig, cbm_cfg: CBMConfig, n_outputs: int = None):
        """
        Args:
            backbone_cfg : CNN encoder config
            cbm_cfg      : concept bottleneck config
            n_outputs    : head output size. Defaults to backbone_cfg.n_classes.
        """
        super().__init__()
        n_out = n_outputs if n_outputs is not None else backbone_cfg.n_classes

        # Encoder: same conv stack as CNNBaseline
        layers = []
        in_ch = backbone_cfg.in_channels
        for out_ch in backbone_cfg.conv_channels:
            layers += [nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
            in_ch = out_ch
        layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten()]
        self.encoder = nn.Sequential(*layers)

        enc_out = backbone_cfg.conv_channels[-1]

        # CBM: encoder_out → n_concepts scalars
        self.cbm  = ContinuousBottleneck(in_dim=enc_out, n_concepts=cbm_cfg.n_concepts)

        # Head: n_concepts → task output
        self.head = nn.Linear(cbm_cfg.n_concepts, n_out)

    def forward(self, x):
        z        = self.encoder(x)
        concepts = self.cbm(z)           # (B, n_concepts)
        return self.head(concepts), concepts
