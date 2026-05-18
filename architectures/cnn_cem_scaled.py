"""
CNN + single-embedding CEM variants.

Two ablations of the standard CEM that use one embedding per concept instead
of a positive/negative pair:

    CEMTanh  — output_i = tanh(c_i) * phi_i(x)
    CEMLinear — output_i = c_i      * phi_i(x)

Both classes share the same interface as CNNwithCEM and are drop-in replacements
in the experiment runner.
"""

import torch.nn as nn
from typing import Union
from cbmc.concepts import CEMTanh, CEMLinear
from cbmc.configs import CNNConfig, CNNRegressionConfig, CEMConfig


def _build_cnn_cem(backbone_cfg, cem_cls, cem_cfg, n_outputs):
    """Shared builder: CNN encoder → concept module → linear head."""

    class _CNNwithScaledCEM(nn.Module):
        def __init__(self):
            super().__init__()
            n_out = (n_outputs if n_outputs is not None
                     else getattr(backbone_cfg, 'n_classes',
                                  getattr(backbone_cfg, 'n_outputs', None)))

            layers = []
            in_ch = backbone_cfg.in_channels
            for out_ch in backbone_cfg.conv_channels:
                layers += [nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
                in_ch = out_ch
            layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten()]
            self.encoder = nn.Sequential(*layers)

            enc_out = backbone_cfg.conv_channels[-1]
            self.cem = cem_cls(
                input_dim=enc_out,
                n_concepts=cem_cfg.n_concepts,
                embedding_dim=cem_cfg.embedding_dim,
                hidden_dim=cem_cfg.hidden_dim,
                depth=cem_cfg.depth,
                dropout=cem_cfg.dropout,
            )
            self.head = nn.Linear(self.cem.output_dim, n_out)

        def forward(self, x, interventions=None, mask=None):
            z = self.encoder(x)
            embeddings, concepts = self.cem(z, interventions, mask)
            return self.head(embeddings), concepts

    return _CNNwithScaledCEM()


class CNNwithCEMTanh(nn.Module):
    """CNN + CEMTanh: output_i = tanh(c_i) * phi_i(x)."""

    def __init__(self, backbone_cfg: Union[CNNConfig, CNNRegressionConfig],
                 cem_cfg: CEMConfig, n_outputs: int = None):
        super().__init__()
        self._model = _build_cnn_cem(backbone_cfg, CEMTanh, cem_cfg, n_outputs)
        # expose cem so experiment code can introspect if needed
        self.cem = self._model.cem

    def forward(self, x, interventions=None, mask=None):
        return self._model(x, interventions, mask)


class CNNwithCEMLinear(nn.Module):
    """CNN + CEMLinear: output_i = c_i * phi_i(x)."""

    def __init__(self, backbone_cfg: Union[CNNConfig, CNNRegressionConfig],
                 cem_cfg: CEMConfig, n_outputs: int = None):
        super().__init__()
        self._model = _build_cnn_cem(backbone_cfg, CEMLinear, cem_cfg, n_outputs)
        self.cem = self._model.cem

    def forward(self, x, interventions=None, mask=None):
        return self._model(x, interventions, mask)
