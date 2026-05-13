"""
ConvVAE + scalar Concept Bottleneck Model for Pendulum.
Flow: image → ConvEncoder → (mu, log_var) → z → CBM → concepts → ConvDecoder → recon
Returns (recon, concepts, mu, log_var).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from cbmc import ContinuousBottleneck
from cbmc.configs import ConvVAEConfig, CBMConfig
from architectures.backbone import ConvEncoder, ConvDecoder


class ConvVAEwithCBM(nn.Module):
    def __init__(self, backbone_cfg: ConvVAEConfig, cbm_cfg: CBMConfig):
        super().__init__()
        self.backbone_cfg = backbone_cfg
        self.cbm_cfg      = cbm_cfg
        self.encoder      = ConvEncoder(backbone_cfg)
        self.cbm          = ContinuousBottleneck(backbone_cfg.latent_dim, cbm_cfg.n_concepts)
        self.decoder      = ConvDecoder(backbone_cfg, bottleneck_dim=cbm_cfg.n_concepts)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + std * torch.randn_like(std)

    def decode(self, concepts):
        return self.decoder(concepts).reconstruction

    def forward(self, x, interventions=None):
        enc      = self.encoder(x)
        z        = self.reparameterize(enc.embedding, enc.log_covariance)
        concepts = self.cbm(z)
        if interventions is not None:
            concepts = interventions
        recon    = self.decoder(concepts).reconstruction
        return recon, concepts, enc.embedding, enc.log_covariance


def conv_vae_cbm_loss(recon, x, mu, log_var, kl_weight=1.0):
    recon_loss = F.mse_loss(recon, x, reduction="sum") / x.size(0)
    kl         = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(1).mean()
    return recon_loss + kl_weight * kl, recon_loss, kl
