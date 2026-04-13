"""
Baseline CNN classifier. Architecture is fully driven by CNNConfig.
Input: (B, in_channels, H, W)  Output: (B, n_classes) logits
"""

import torch.nn as nn
from cbmc.configs import CNNConfig


class CNNBaseline(nn.Module):
    def __init__(self, cfg: CNNConfig = CNNConfig()):
        super().__init__()
        self.cfg = cfg

        # Conv blocks: each channel in conv_channels gets Conv -> ReLU -> MaxPool
        layers = []
        in_ch = cfg.in_channels
        for out_ch in cfg.conv_channels:
            layers += [nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
            in_ch = out_ch
        layers.append(nn.AdaptiveAvgPool2d(1))  # (B, C, 1, 1) — avoids hardcoded spatial dims
        layers.append(nn.Flatten())              # (B, C)
        self.features = nn.Sequential(*layers)

        # FC head
        fc = []
        in_dim = cfg.conv_channels[-1]
        for out_dim in cfg.fc_dims:
            fc += [nn.Linear(in_dim, out_dim), nn.ReLU()]
            if cfg.dropout > 0:
                fc.append(nn.Dropout(cfg.dropout))
            in_dim = out_dim
        fc.append(nn.Linear(in_dim, cfg.n_classes))
        self.classifier = nn.Sequential(*fc)

    def forward(self, x):
        return self.classifier(self.features(x))
