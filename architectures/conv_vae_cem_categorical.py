"""
ConvVAE + categorical Concept Embedding Model (Case 2, and Case 3 when the
states are bins of a continuous range).

Flow: image -> ConvEncoder -> (mu, log_var) -> z -> CEMCategoricalPerConcept
      -> (embeddings, logits) -> ConvDecoder -> recon
Returns (recon, logits, mu, log_var); logits is a list of [B, n_i] tensors.

Interface mirrors ConvVAEwithCEM, so `model.cem(z, interventions)` followed by
`model.decoder(emb).reconstruction` generates from the prior -- except that
interventions here are LongTensor state indices, not values in [-1, 1].
"""

import torch
import torch.nn as nn

from cbmc.concepts import CEMCategoricalPerConcept
from cbmc.configs import ConvVAEConfig, CEMCategoricalConfig
from architectures.backbone import ConvEncoder, ConvDecoder


class ConvVAEwithCEMCategorical(nn.Module):
    def __init__(self, backbone_cfg: ConvVAEConfig, cem_cfg: CEMCategoricalConfig):
        super().__init__()
        self.backbone_cfg = backbone_cfg
        self.cem_cfg      = cem_cfg
        n_states = cem_cfg.n_states or [cem_cfg.n_classes] * cem_cfg.n_concepts
        self.encoder = ConvEncoder(backbone_cfg)
        self.cem     = CEMCategoricalPerConcept(
            input_dim     = backbone_cfg.latent_dim,
            n_states      = n_states,
            embedding_dim = cem_cfg.embedding_dim,
            hidden_dim    = cem_cfg.hidden_dim,
            depth         = cem_cfg.depth,
            dropout       = cem_cfg.dropout,
            normalize     = cem_cfg.normalize,
        )
        self.decoder = ConvDecoder(backbone_cfg, bottleneck_dim=self.cem.output_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + std * torch.randn_like(std)

    def decode(self, embeddings):
        return self.decoder(embeddings).reconstruction

    def forward(self, x, interventions=None, mask=None):
        enc                = self.encoder(x)
        z                  = self.reparameterize(enc.embedding, enc.log_covariance)
        embeddings, logits = self.cem(z, interventions, mask)
        recon              = self.decoder(embeddings).reconstruction
        return recon, logits, enc.embedding, enc.log_covariance

    def concept_loss(self, logits, targets):
        return self.cem.concept_loss(logits, targets)
