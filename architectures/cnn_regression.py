"""
CNN regression baseline — predicts continuous targets instead of class labels.
Architecture driven by CNNRegressionConfig.
Input: (B, in_channels, H, W)  Output: (B, n_outputs)
"""

import torch.nn as nn
from cbmc.configs import CNNRegressionConfig


class CNNRegression(nn.Module):
    def __init__(self, cfg: CNNRegressionConfig = CNNRegressionConfig()):
        super().__init__()
        self.cfg = cfg

        layers = []
        in_ch = cfg.in_channels
        for out_ch in cfg.conv_channels:
            layers += [nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
            in_ch = out_ch
        layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten()]
        self.features = nn.Sequential(*layers)

        fc = []
        in_dim = cfg.conv_channels[-1]
        for out_dim in cfg.fc_dims:
            fc += [nn.Linear(in_dim, out_dim), nn.ReLU()]
            if cfg.dropout > 0:
                fc.append(nn.Dropout(cfg.dropout))
            in_dim = out_dim
        fc.append(nn.Linear(in_dim, cfg.n_outputs))
        self.head = nn.Sequential(*fc)

    def forward(self, x):
        return self.head(self.features(x))
