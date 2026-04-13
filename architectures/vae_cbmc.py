"""
Example: diffusers AutoencoderKL encoder + ContinuousBottleneck.

AutoencoderKL is the pretrained VAE from Stable Diffusion (HuggingFace diffusers).
Its encoder maps images (B, 3, H, W) -> latent (B, 4, H/8, W/8).
We pool the spatial dims to get a flat vector, then pass through ContinuousBottleneck.

Usage:
    model = VAEWithBottleneck(n_concepts=50, n_classes=10)
    logits, concepts = model(images)   # images: (B, 3, 256, 256)
"""

import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from cbmc import ContinuousBottleneck


class VAEWithBottleneck(nn.Module):
    def __init__(
        self,
        n_concepts: int,
        n_classes: int,
        vae_model: str = "stabilityai/sd-vae-ft-mse",
        freeze_encoder: bool = True,
    ):
        super().__init__()

        # Load pretrained VAE from diffusers — only keep the encoder
        vae = AutoencoderKL.from_pretrained(vae_model)
        self.encoder = vae.encoder
        self.quant_conv = vae.quant_conv  # projects to latent channels (4)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            for p in self.quant_conv.parameters():
                p.requires_grad = False

        # AutoencoderKL latent: (B, 4, H/8, W/8) — pool to (B, 4)
        # quant_conv output has 2*latent_channels for mu/logvar -> we take mu -> (B, 4, h, w)
        latent_channels = 4
        self.pool = nn.AdaptiveAvgPool2d(1)  # (B, 4, h, w) -> (B, 4, 1, 1)

        self.bottleneck = ContinuousBottleneck(in_dim=latent_channels, n_concepts=n_concepts)
        self.head = nn.Linear(n_concepts, n_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        moments = self.quant_conv(h)          # (B, 2*C, H, W)
        mu, _ = moments.chunk(2, dim=1)       # take mean, discard log_var
        return mu                             # (B, C, H, W)

    def forward(self, x: torch.Tensor):
        mu = self.encode(x)                   # (B, 4, H/8, W/8)
        z = self.pool(mu).flatten(1)          # (B, 4)
        concepts = self.bottleneck(z)         # (B, n_concepts)
        return self.head(concepts), concepts
