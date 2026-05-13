"""
ConvVAE + Concept Embedding Model for Pendulum.
Flow: image → ConvEncoder → (mu, log_var) → z → CEM → (embeddings, concepts) → ConvDecoder → recon
Returns (recon, concepts, mu, log_var).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from cbmc.concepts import CEM
from cbmc.configs import ConvVAEConfig, CEMConfig
from architectures.backbone import ConvEncoder, ConvDecoder


class ConvVAEwithCEM(nn.Module):
    def __init__(self, backbone_cfg: ConvVAEConfig, cem_cfg: CEMConfig):
        super().__init__()
        self.backbone_cfg = backbone_cfg
        self.cem_cfg      = cem_cfg
        self.encoder      = ConvEncoder(backbone_cfg)
        self.cem          = CEM(
            input_dim     = backbone_cfg.latent_dim,
            n_concepts    = cem_cfg.n_concepts,
            embedding_dim = cem_cfg.embedding_dim,
            hidden_dim    = cem_cfg.hidden_dim,
            depth         = cem_cfg.depth,
            dropout       = cem_cfg.dropout,
        )
        self.decoder = ConvDecoder(backbone_cfg, bottleneck_dim=self.cem.output_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + std * torch.randn_like(std)

    def decode(self, embeddings):
        return self.decoder(embeddings).reconstruction

    def forward(self, x, interventions=None, mask=None):
        enc                  = self.encoder(x)
        z                    = self.reparameterize(enc.embedding, enc.log_covariance)
        embeddings, concepts = self.cem(z, interventions, mask)
        recon                = self.decoder(embeddings).reconstruction
        return recon, concepts, enc.embedding, enc.log_covariance


def conv_vae_cem_loss(recon, x, mu, log_var, kl_weight=1.0):
    recon_loss = F.mse_loss(recon, x, reduction="sum") / x.size(0)
    kl         = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(1).mean()
    return recon_loss + kl_weight * kl, recon_loss, kl
