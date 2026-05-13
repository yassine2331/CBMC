"""
CNN + scalar Concept Bottleneck Model (CBM).

Backbone encodes the image to a flat vector.
ContinuousBottleneck maps it to n_concepts scalar values.
Head maps concepts to task output (classification or regression).

Returns (logits/preds, concepts) so concept values can be logged or supervised.
"""

import torch.nn as nn
from typing import Union
from cbmc import ContinuousBottleneck
from cbmc.configs import CNNConfig, CNNRegressionConfig, CBMConfig


class CNNwithCBM(nn.Module):
    def __init__(self, backbone_cfg: Union[CNNConfig, CNNRegressionConfig], cbm_cfg: CBMConfig, n_outputs: int = None):
        """
        Args:
            backbone_cfg : CNN encoder config (CNNConfig or CNNRegressionConfig)
            cbm_cfg      : concept bottleneck config
            n_outputs    : head output size. Required when using CNNRegressionConfig.
        """
        super().__init__()
        if n_outputs is not None:
            n_out = n_outputs
        elif hasattr(backbone_cfg, 'n_classes'):
            n_out = backbone_cfg.n_classes
        else:
            n_out = backbone_cfg.n_outputs

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
        # the head should also be a stack of layers
        head_layers = []
        in_features = cbm_cfg.n_concepts
        for out_features in cbm_cfg.head_channels:
            head_layers += [nn.Linear(in_features, out_features), nn.ReLU()]
            in_features = out_features
        head_layers += [nn.Linear(in_features, n_out)]
        self.head = nn.Sequential(*head_layers)

    def forward(self, x, interventions=None):
        z        = self.encoder(x)
        concepts = self.cbm(z)                                    # (B, n_concepts)
        if interventions is not None:
            concepts = interventions
        return self.head(concepts), concepts
