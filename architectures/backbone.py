"""
Pythae-compatible encoder/decoder building blocks.

Four classes covering the two dataset families:
  - MLPEncoder / MLPDecoder   : for ArithmeticMNIST (flat images)
  - ConvEncoder / ConvDecoder : for Pendulum (spatial images)

All subclass pythae's BaseEncoder / BaseDecoder so they can be plugged into
any pythae VAE, or composed freely in custom nn.Modules.

Constructor:
  Encoders take the full backbone config.
  Decoders take the backbone config + bottleneck_dim (latent_dim, n_concepts,
  or n_concepts * embedding_dim depending on what sits between encoder and decoder).
"""

import torch
import torch.nn as nn
from pythae.models.nn import BaseEncoder, BaseDecoder
from pythae.models.base.base_utils import ModelOutput
from cbmc.configs import VAEConfig, ConvVAEConfig


class MLPEncoder(BaseEncoder):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        dims = [cfg.input_dim] + cfg.encoder_dims
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        self.net        = nn.Sequential(*layers)
        self.fc_mu      = nn.Linear(cfg.encoder_dims[-1], cfg.latent_dim)
        self.fc_log_var = nn.Linear(cfg.encoder_dims[-1], cfg.latent_dim)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        h = self.net(x.flatten(1))
        return ModelOutput(embedding=self.fc_mu(h), log_covariance=self.fc_log_var(h))


class MLPDecoder(BaseDecoder):
    def __init__(self, cfg: VAEConfig, bottleneck_dim: int):
        super().__init__()
        dims = [bottleneck_dim] + cfg.decoder_dims + [cfg.input_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> ModelOutput:
        return ModelOutput(reconstruction=self.net(z))


class ConvEncoder(BaseEncoder):
    def __init__(self, cfg: ConvVAEConfig):
        super().__init__()
        enc_layers = []
        in_ch = cfg.in_channels
        for out_ch in cfg.enc_channels:
            enc_layers += [nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
            in_ch = out_ch
        self.net        = nn.Sequential(*enc_layers)
        spatial         = cfg.img_size // (2 ** len(cfg.enc_channels))
        flat            = cfg.enc_channels[-1] * spatial * spatial
        self.fc_mu      = nn.Linear(flat, cfg.latent_dim)
        self.fc_log_var = nn.Linear(flat, cfg.latent_dim)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        h = self.net(x).flatten(1)
        return ModelOutput(embedding=self.fc_mu(h), log_covariance=self.fc_log_var(h))


class ConvDecoder(BaseDecoder):
    def __init__(self, cfg: ConvVAEConfig, bottleneck_dim: int):
        super().__init__()
        self._spatial  = cfg.img_size // (2 ** len(cfg.enc_channels))
        self._first_ch = cfg.enc_channels[-1]
        self.fc = nn.Linear(bottleneck_dim, self._first_ch * self._spatial * self._spatial)
        dec_layers = []
        in_ch = self._first_ch
        for out_ch in cfg.dec_channels:
            dec_layers += [nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1), nn.ReLU()]
            in_ch = out_ch
        dec_layers += [nn.ConvTranspose2d(in_ch, cfg.in_channels, 3, padding=1), nn.Sigmoid()]
        self.net = nn.Sequential(*dec_layers)

    def forward(self, z: torch.Tensor) -> ModelOutput:
        h = self.fc(z).view(z.size(0), self._first_ch, self._spatial, self._spatial)
        return ModelOutput(reconstruction=self.net(h))
