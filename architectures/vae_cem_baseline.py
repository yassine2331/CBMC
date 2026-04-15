"""
MLP VAE + Concept Embedding Model (CEM).

Flow:
  image → encoder → (mu, log_var) → z → CEM → (embeddings, concepts) → decoder → recon

The CEM replaces the raw latent z as the bottleneck.
Decoder input dim = n_concepts * embedding_dim.

Returns (recon, concepts, mu, log_var).
Supports test-time interventions via the `interventions` and `mask` arguments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from cbmc.concepts import ConceptBottleneck
from cbmc.configs import VAEConfig, CEMConfig


def _make_mlp(dims, final_activation=None):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    if final_activation:
        layers.append(final_activation())
    return nn.Sequential(*layers)


class VAEwithCEM(nn.Module):
    def __init__(self, backbone_cfg: VAEConfig, cem_cfg: CEMConfig):
        super().__init__()
        self.backbone_cfg = backbone_cfg
        self.cem_cfg      = cem_cfg

        # Encoder
        self.encoder    = _make_mlp([backbone_cfg.input_dim] + backbone_cfg.encoder_dims, final_activation=nn.ReLU)
        self.fc_mu      = nn.Linear(backbone_cfg.encoder_dims[-1], backbone_cfg.latent_dim)
        self.fc_log_var = nn.Linear(backbone_cfg.encoder_dims[-1], backbone_cfg.latent_dim)

        # CEM: z → (embeddings, concept_scores)
        self.cem = ConceptBottleneck(
            input_dim     = backbone_cfg.latent_dim,
            n_concepts    = cem_cfg.n_concepts,
            embedding_dim = cem_cfg.embedding_dim,
            hidden_dim    = cem_cfg.hidden_dim,
            depth         = cem_cfg.depth,
            dropout       = cem_cfg.dropout,
        )

        # Decoder takes CEM output (n_concepts * embedding_dim) as input
        self.decoder = _make_mlp(
            [self.cem.output_dim] + backbone_cfg.decoder_dims + [backbone_cfg.input_dim],
            final_activation=nn.Sigmoid,
        )

    def encode(self, x):
        h = self.encoder(x.flatten(1))
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + std * torch.randn_like(std)

    def decode(self, embeddings, shape):
        return self.decoder(embeddings).view(embeddings.size(0), *shape)

    def forward(self, x, interventions=None, mask=None):
        mu, log_var          = self.encode(x)
        z                    = self.reparameterize(mu, log_var)
        embeddings, concepts = self.cem(z, interventions, mask)   # (B, E*C), (B, C)
        recon                = self.decode(embeddings, x.shape[1:])
        return recon, concepts, mu, log_var


def vae_cem_loss(recon, x, mu, log_var, kl_weight=1.0):
    x_01       = (x * 0.3081 + 0.1307).clamp(0, 1)
    recon_loss = F.binary_cross_entropy(recon, x_01, reduction="sum") / x.size(0)
    kl         = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(1).mean()
    return recon_loss + kl_weight * kl, recon_loss, kl
