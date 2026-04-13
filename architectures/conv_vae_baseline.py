"""
Conv VAE baseline — for spatial images (e.g. Pendulum 64x64 RGBA).
Architecture is fully driven by ConvVAEConfig.
No classifier, pure ELBO.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from cbmc.configs import ConvVAEConfig


class ConvVAEBaseline(nn.Module):
    def __init__(self, cfg: ConvVAEConfig = ConvVAEConfig()):
        super().__init__()
        self.cfg        = cfg
        self.latent_dim = cfg.latent_dim

        # ---- Encoder ----
        enc_layers = []
        in_ch = cfg.in_channels
        for out_ch in cfg.enc_channels:
            enc_layers += [nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
            in_ch = out_ch
        self.encoder = nn.Sequential(*enc_layers)

        # Spatial size after n MaxPool2d(2): img_size / 2^n
        self._spatial = cfg.img_size // (2 ** len(cfg.enc_channels))
        self._flat    = cfg.enc_channels[-1] * self._spatial * self._spatial

        self.fc_mu      = nn.Linear(self._flat, cfg.latent_dim)
        self.fc_log_var = nn.Linear(self._flat, cfg.latent_dim)

        # ---- Decoder ----
        self.fc_decode = nn.Linear(cfg.latent_dim, self._flat)

        dec_layers = []
        in_ch = cfg.enc_channels[-1]
        for out_ch in cfg.dec_channels:
            dec_layers += [nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1), nn.ReLU()]
            in_ch = out_ch
        dec_layers.append(nn.ConvTranspose2d(in_ch, cfg.in_channels, 3, padding=1))
        dec_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x):
        h = self.encoder(x).flatten(1)
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + std * torch.randn_like(std)

    def decode(self, z, shape=None):
        h = self.fc_decode(z).view(z.size(0), self.cfg.enc_channels[-1], self._spatial, self._spatial)
        return self.decoder(h)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z           = self.reparameterize(mu, log_var)
        recon       = self.decode(z)
        return recon, mu, log_var


def conv_vae_loss(recon, x, mu, log_var, kl_weight: float = 1.0):
    recon_loss = F.mse_loss(recon, x, reduction="sum") / x.size(0)
    kl         = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(1).mean()
    return recon_loss + kl_weight * kl, recon_loss, kl
