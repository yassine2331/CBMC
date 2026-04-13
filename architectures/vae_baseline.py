"""
Vanilla VAE. Architecture is fully driven by VAEConfig.
Trained with ELBO = reconstruction loss + KL divergence. No classifier.

Input: (B, in_channels, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from cbmc.configs import VAEConfig


def _make_mlp(dims: list, activation=nn.ReLU, final_activation=None) -> nn.Sequential:
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(activation())
    if final_activation is not None:
        layers.append(final_activation())
    return nn.Sequential(*layers)


class VAEBaseline(nn.Module):
    def __init__(self, cfg: VAEConfig = VAEConfig()):
        super().__init__()
        self.cfg        = cfg
        self.latent_dim = cfg.latent_dim

        # Encoder: input_dim -> encoder_dims -> (mu, log_var)
        self.encoder    = _make_mlp([cfg.input_dim] + cfg.encoder_dims, final_activation=nn.ReLU)
        self.fc_mu      = nn.Linear(cfg.encoder_dims[-1], cfg.latent_dim)
        self.fc_log_var = nn.Linear(cfg.encoder_dims[-1], cfg.latent_dim)

        # Decoder: latent_dim -> decoder_dims -> input_dim
        self.decoder = _make_mlp(
            [cfg.latent_dim] + cfg.decoder_dims + [cfg.input_dim],
            final_activation=nn.Sigmoid,
        )

    def encode(self, x):
        h = self.encoder(x.flatten(1))
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + std * torch.randn_like(std)

    def decode(self, z, shape):
        return self.decoder(z).view(z.size(0), *shape)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z           = self.reparameterize(mu, log_var)
        recon       = self.decode(z, x.shape[1:])   # restore (C, H, W)
        return recon, mu, log_var


def vae_loss(recon, x, mu, log_var, kl_weight: float = 1.0):
    """ELBO = reconstruction loss + KL divergence."""
    x_01       = (x * 0.3081 + 0.1307).clamp(0, 1)
    recon_loss = F.binary_cross_entropy(recon, x_01, reduction="sum") / x.size(0)
    kl         = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(1).mean()
    return recon_loss + kl_weight * kl, recon_loss, kl
