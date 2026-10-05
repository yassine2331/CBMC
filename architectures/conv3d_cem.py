"""
3D CNN + continuous concept bottleneck, for LIDC nodule cubes.

Flow mirrors architectures/cnn_cem.py, with a volumetric encoder:

    cube (1, D, D, D)  ->  Conv3D encoder  ->  h
                       ->  CEM             ->  (embeddings, concepts)
                       ->  linear head     ->  class logits

Returns (logits, concepts). Supports test-time intervention through the same
`interventions` / `mask` arguments as every other concept architecture here.
"""

import torch.nn as nn

from cbmc.concepts import CEM, CEMCategorical, CEMLinear, CEMLinearRaw, CEMTanh
from cbmc.configs import Conv3DConfig, CEMConfig

# Name -> concept block, so a config can pick one by string.
BOTTLENECKS = {
    "cem":            CEM,              # Case 3.5 — two anchors, interpolation
    "cem_tanh":       CEMTanh,
    "cem_linear":     CEMLinear,
    "cem_linear_raw": CEMLinearRaw,
    "categorical":    CEMCategorical,   # Case 2 — n anchors, softmax
}


class Conv3DEncoder(nn.Module):
    """(in_channels, D, D, D) -> (conv_channels[-1],) via global average pool."""

    def __init__(self, cfg: Conv3DConfig):
        super().__init__()
        layers, in_ch = [], cfg.in_channels
        pad = cfg.kernel_size // 2
        for ch in cfg.conv_channels:
            layers.append(nn.Conv3d(in_ch, ch, cfg.kernel_size, padding=pad))
            if cfg.batch_norm:
                layers.append(nn.BatchNorm3d(ch))
            layers.append(nn.ReLU(inplace=True))
            if cfg.dropout > 0:
                layers.append(nn.Dropout3d(cfg.dropout))
            layers.append(nn.MaxPool3d(2))
            in_ch = ch
        layers += [nn.AdaptiveAvgPool3d(1), nn.Flatten()]
        self.net = nn.Sequential(*layers)
        self.out_dim = cfg.conv_channels[-1]

    def forward(self, x):
        return self.net(x)


class Conv3DwithCEM(nn.Module):
    """
    Args:
        backbone_cfg : Conv3DConfig
        cem_cfg      : CEMConfig
        n_classes    : head output size (2 for benign/malignant)
        bottleneck   : key into BOTTLENECKS
    """

    def __init__(self, backbone_cfg: Conv3DConfig, cem_cfg: CEMConfig,
                 n_classes: int = 2, bottleneck: str = "cem"):
        super().__init__()
        if bottleneck not in BOTTLENECKS:
            raise ValueError(f"unknown bottleneck '{bottleneck}'; "
                             f"choose from {sorted(BOTTLENECKS)}")
        self.encoder = Conv3DEncoder(backbone_cfg)
        self.cem = BOTTLENECKS[bottleneck](
            input_dim     = self.encoder.out_dim,
            n_concepts    = cem_cfg.n_concepts,
            embedding_dim = cem_cfg.embedding_dim,
            hidden_dim    = cem_cfg.hidden_dim,
            depth         = cem_cfg.depth,
            dropout       = cem_cfg.dropout,
        )
        self.head = nn.Linear(self.cem.output_dim, n_classes)

    def forward(self, x, interventions=None, mask=None):
        h = self.encoder(x)
        embeddings, concepts = self.cem(h, interventions, mask)
        return self.head(embeddings), concepts


class Conv3DBaseline(nn.Module):
    """Same encoder, no bottleneck — the interpretability-vs-accuracy anchor."""

    def __init__(self, backbone_cfg: Conv3DConfig, n_classes: int = 2):
        super().__init__()
        self.encoder = Conv3DEncoder(backbone_cfg)
        self.head = nn.Linear(self.encoder.out_dim, n_classes)

    def forward(self, x, interventions=None, mask=None):
        return self.head(self.encoder(x)), None
