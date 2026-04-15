"""
MLP VAE + scalar Concept Bottleneck Model (CBM).

Flow:
  image → encoder → (mu, log_var) → z → CBM → concepts → decoder → recon

The concept layer replaces the raw latent z as the bottleneck.
Decoder input dim = n_concepts.

Returns (recon, concepts, mu, log_var).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from cbmc import ContinuousBottleneck
from cbmc.configs import VAEConfig, CBMConfig


def _make_mlp(dims, final_activation=None):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    if final_activation:
        layers.append(final_activation())
    return nn.Sequential(*layers)


class VAEwithCBM(nn.Module):
    def __init__(self, backbone_cfg: VAEConfig, cbm_cfg: CBMConfig):
        super().__init__()
        self.backbone_cfg = backbone_cfg
        self.cbm_cfg      = cbm_cfg

        # Encoder
        self.encoder    = _make_mlp([backbone_cfg.input_dim] + backbone_cfg.encoder_dims, final_activation=nn.ReLU)
        self.fc_mu      = nn.Linear(backbone_cfg.encoder_dims[-1], backbone_cfg.latent_dim)
        self.fc_log_var = nn.Linear(backbone_cfg.encoder_dims[-1], backbone_cfg.latent_dim)

        # CBM: z → scalar concepts
        self.cbm = ContinuousBottleneck(in_dim=backbone_cfg.latent_dim, n_concepts=cbm_cfg.n_concepts)

        # Decoder takes concept vector (n_concepts) as input
        self.decoder = _make_mlp(
            [cbm_cfg.n_concepts] + backbone_cfg.decoder_dims + [backbone_cfg.input_dim],
            final_activation=nn.Sigmoid,
        )

    def encode(self, x):
        h = self.encoder(x.flatten(1))
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + std * torch.randn_like(std)

    def decode(self, concepts, shape):
        return self.decoder(concepts).view(concepts.size(0), *shape)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z           = self.reparameterize(mu, log_var)
        concepts    = self.cbm(z)                      # (B, n_concepts)
        recon       = self.decode(concepts, x.shape[1:])
        return recon, concepts, mu, log_var


def vae_cbm_loss(recon, x, mu, log_var, kl_weight=1.0):
    x_01       = (x * 0.3081 + 0.1307).clamp(0, 1)
    recon_loss = F.binary_cross_entropy(recon, x_01, reduction="sum") / x.size(0)
    kl         = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(1).mean()
    return recon_loss + kl_weight * kl, recon_loss, kl
