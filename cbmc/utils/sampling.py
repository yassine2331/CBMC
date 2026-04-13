"""
Utilities for sampling and saving images from generative models during training.
Saves a grid of: reconstructions (top row) + random samples (bottom row).
"""

import os
import torch
import torchvision.utils as vutils
from torchvision.transforms.functional import to_pil_image


def save_vae_samples(
    model,
    fixed_x: torch.Tensor,
    epoch: int,
    out_dir: str,
    n_samples: int = 8,
    device: str = "cpu",
):
    """
    Saves two rows of images to out_dir/epoch_{epoch:03d}.png:
      - Row 1: original images (fixed_x)
      - Row 2: their reconstructions
      - Row 3: random samples from the prior N(0,1)

    Args:
        model:    a VAEBaseline (or any model with .decode() and .encode())
        fixed_x:  a fixed batch of images to reconstruct each epoch — shape (N, C, H, W)
        epoch:    current epoch number (used for filename)
        out_dir:  directory to save images into
        n_samples: how many random samples to generate
        device:   device string
    """
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    with torch.no_grad():
        x = fixed_x[:n_samples].to(device)

        # Reconstructions
        recon, mu, _ = model(x)

        # Random samples from prior
        z = torch.randn(n_samples, model.latent_dim, device=device)
        generated = model.decode(z, x.shape[1:])

    # Unnormalize originals from MNIST normalization back to [0,1]
    x_display = (x * 0.3081 + 0.1307).clamp(0, 1)

    grid = vutils.make_grid(
        torch.cat([x_display, recon, generated], dim=0),
        nrow=n_samples,
        padding=2,
        normalize=False,
    )

    img = to_pil_image(grid.cpu())
    path = os.path.join(out_dir, f"epoch_{epoch:03d}.png")
    img.save(path)
    print(f"  -> saved samples: {path}")
