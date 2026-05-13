"""
Vanilla MLP VAE for ArithmeticMNIST.
Flow: image → MLPEncoder → (mu, log_var) → z → MLPDecoder → recon
Returns (recon, mu, log_var).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from cbmc.configs import VAEConfig
from architectures.backbone import MLPEncoder, MLPDecoder


class VAEBaseline(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg        = cfg
        self.latent_dim = cfg.latent_dim
        self.encoder    = MLPEncoder(cfg)
        self.decoder    = MLPDecoder(cfg, bottleneck_dim=cfg.latent_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + std * torch.randn_like(std)

    def decode(self, z, shape=None):
        out = self.decoder(z).reconstruction
        return out.view(z.size(0), *(shape or (self.cfg.input_dim,)))

    def forward(self, x):
        enc   = self.encoder(x)
        z     = self.reparameterize(enc.embedding, enc.log_covariance)
        recon = self.decoder(z).reconstruction.view(x.shape)
        return recon, enc.embedding, enc.log_covariance


def vae_loss(recon, x, mu, log_var, kl_weight=1.0):
    x_01       = (x * 0.3081 + 0.1307).clamp(0, 1)
    recon_loss = F.binary_cross_entropy(recon, x_01, reduction="sum") / x.size(0)
    kl         = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(1).mean()
    return recon_loss + kl_weight * kl, recon_loss, kl
