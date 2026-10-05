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

from cbmc import ContinuousBottleneck
from cbmc.concepts import (CEM, CEMCategorical, CEMCategoricalPerConcept,
                           CEMLinear, CEMLinearRaw, CEMTanh)
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


class Conv3DwithCBM(nn.Module):
    """
    Scalar concept bottleneck (Koh et al. style): the encoder is squeezed to
    one number per concept, and the head sees only those numbers. No anchor
    embeddings, so the head's entire view of the image is k scalars.

    Returns (logits, concepts) like the CEM variants.
    """

    def __init__(self, backbone_cfg: Conv3DConfig, cem_cfg: CEMConfig,
                 n_classes: int = 2, head_hidden=(32, 32)):
        super().__init__()
        self.encoder = Conv3DEncoder(backbone_cfg)
        self.cbm = ContinuousBottleneck(in_dim=self.encoder.out_dim,
                                        n_concepts=cem_cfg.n_concepts)
        layers, in_f = [], cem_cfg.n_concepts
        for h in head_hidden:
            layers += [nn.Linear(in_f, h), nn.ReLU(inplace=True)]
            in_f = h
        layers.append(nn.Linear(in_f, n_classes))
        self.head = nn.Sequential(*layers)

    def forward(self, x, interventions=None, mask=None):
        concepts = self.cbm(self.encoder(x))
        used = concepts
        if interventions is not None:
            used = (interventions if mask is None
                    else mask * interventions + (1 - mask) * concepts)
        return self.head(used), concepts


class Conv3DwithCategorical(nn.Module):
    """
    Categorical bottleneck where each concept may have its own number of
    states — LIDC needs this, since the 1-5 ratings have 5 states while a
    binned continuous concept has however many bins were requested.
    """

    def __init__(self, backbone_cfg: Conv3DConfig, cem_cfg: CEMConfig,
                 n_states, n_classes: int = 2):
        super().__init__()
        self.encoder = Conv3DEncoder(backbone_cfg)
        self.cem = CEMCategoricalPerConcept(
            input_dim     = self.encoder.out_dim,
            n_states      = n_states,
            embedding_dim = cem_cfg.embedding_dim,
            hidden_dim    = cem_cfg.hidden_dim,
            depth         = cem_cfg.depth,
            dropout       = cem_cfg.dropout,
        )
        self.head = nn.Linear(self.cem.output_dim, n_classes)

    def forward(self, x, interventions=None, mask=None):
        emb, logits = self.cem(self.encoder(x), interventions, mask)
        return self.head(emb), logits
