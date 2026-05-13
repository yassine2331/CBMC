"""
Vanilla ConvVAE for Pendulum.
Flow: image → ConvEncoder → (mu, log_var) → z → ConvDecoder → recon
Returns (recon, mu, log_var).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from cbmc.configs import ConvVAEConfig
from architectures.backbone import ConvEncoder, ConvDecoder


class ConvVAEBaseline(nn.Module):
    def __init__(self, cfg: ConvVAEConfig):
        super().__init__()
        self.cfg        = cfg
        self.latent_dim = cfg.latent_dim
        self.encoder    = ConvEncoder(cfg)
        self.decoder    = ConvDecoder(cfg, bottleneck_dim=cfg.latent_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + std * torch.randn_like(std)

    def decode(self, z, shape=None):
        return self.decoder(z).reconstruction

    def forward(self, x):
        enc   = self.encoder(x)
        z     = self.reparameterize(enc.embedding, enc.log_covariance)
        recon = self.decoder(z).reconstruction
        return recon, enc.embedding, enc.log_covariance


def conv_vae_loss(recon, x, mu, log_var, kl_weight=1.0):
    recon_loss = F.mse_loss(recon, x, reduction="sum") / x.size(0)
    kl         = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(1).mean()
    return recon_loss + kl_weight * kl, recon_loss, kl
