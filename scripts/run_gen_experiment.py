"""
Generative experiments on ArithmeticMNIST.

Compares four concept-bottleneck VAE variants on image reconstruction and generation:
  1. VAE baseline      — no concept bottleneck
  2. VAE + CBM         — scalar concept bottleneck
  3. VAE + CEM         — dual-embedding blending (standard CEM)
  4. VAE + CEM-Linear  — multiplicative single-embedding (CEMLinear)

For each model:
  - Trains a ConvVAE on ArithmeticMNIST (digits + operator → arithmetic result image)
  - Saves per-epoch sample grids (originals | reconstructions | prior samples)
  - Tracks: recon_loss, kl_loss, concept_mse (where applicable)

Final output:
  - outputs/samples/gen_comparison_final.png  — side-by-side comparison panel
  - outputs/results/gen_experiment_metrics.csv — per-epoch training curves

Usage:
    python scripts/run_gen_experiment.py                 # run all 4 then compare
    python scripts/run_gen_experiment.py --exp vae       # run one experiment only
    python scripts/run_gen_experiment.py --epochs 10     # override number of epochs
"""

import argparse
import csv
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cbmc.configs import ConvVAEConfig, CBMConfig, CEMConfig, TrainConfig
from cbmc.data.arithmetic_mnist import get_arithmetic_mnist
from architectures.conv_vae_baseline import ConvVAEBaseline
from architectures.conv_vae_cbm import ConvVAEwithCBM
from architectures.conv_vae_cem import ConvVAEwithCEM
from architectures.conv_vae_cem_linear import ConvVAEwithCEMLinear

# Concept normalization: map digit values [1,9] → [-1,1]
MNIST_CONCEPT_MEAN  = 5.0
MNIST_CONCEPT_SCALE = 4.0   # (c - 5) / 4  maps  1→-1, 5→0, 9→+1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device():
    if torch.cuda.is_available():         return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _random_fixed_batch(loader):
    dataset = loader.dataset
    idx = random.sample(range(len(dataset)), loader.batch_size)
    return loader.collate_fn([dataset[i] for i in idx])

def _unnorm(x):
    """Undo MNIST normalization → [0,1] for display."""
    return (x * 0.3081 + 0.1307).clamp(0, 1)

def _save_grid(tensors, path, n=8):
    """Save a multi-row image grid to path. Each element of tensors is one row."""
    rows = [t[:n].clamp(0, 1) for t in tensors]
    grid = vutils.make_grid(torch.cat(rows, dim=0), nrow=n, padding=2, normalize=False)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    to_pil_image(grid.cpu()).save(path)

def _save_metrics_csv(path, rows, header):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  -> saved metrics: {path}")


# ---------------------------------------------------------------------------
# Per-model reconstruction + prior sampling
# ---------------------------------------------------------------------------

def _get_samples(model, model_type, fixed_x, n, device):
    """Return (originals, recons, prior_imgs) for a trained model."""
    model.eval()
    with torch.no_grad():
        x = fixed_x[:n].to(device)
        if model_type == "vae":
            recon, _, _ = model(x)
            z_prior = torch.randn(n, model.latent_dim, device=device)
            prior_imgs = model.decode(z_prior)
        elif model_type == "cbm":
            recon, _, _, _ = model(x)
            z_prior = torch.randn(n, model.backbone_cfg.latent_dim, device=device)
            c_prior = model.cbm(z_prior)
            prior_imgs = model.decode(c_prior)
        else:  # cem or cem_linear — identical interface
            recon, _, _, _ = model(x)
            z_prior = torch.randn(n, model.backbone_cfg.latent_dim, device=device)
            emb, _ = model.cem(z_prior)
            prior_imgs = model.decoder(emb).reconstruction
    return x.cpu(), recon.cpu(), prior_imgs.cpu()


# ---------------------------------------------------------------------------
# Experiment 1 — VAE baseline (no concepts)
# ---------------------------------------------------------------------------

def exp_vae_gen_mnist(device, *, epochs=None, seed=42, tag=None):
    print("\n=== [GEN 1] VAE baseline — ArithmeticMNIST ===")
    backbone_cfg = ConvVAEConfig.load("experiments/configs/exp_gen_mnist.json")
    train_cfg    = TrainConfig.load("experiments/configs/train_gen_mnist.json")
    set_seed(seed)
    n_epochs = epochs if epochs is not None else train_cfg.epochs

    train_loader, test_loader = get_arithmetic_mnist(
        batch_size=train_cfg.batch_size, num_workers=train_cfg.num_workers,
    )
    model     = ConvVAEBaseline(backbone_cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=train_cfg.lr)
    fixed_x, _, _ = _random_fixed_batch(test_loader)
    _tag   = f"_{tag}" if tag else ""
    out_dir = f"outputs/samples/gen_vae{_tag}"
    metrics = []  # (epoch, recon_loss, kl)

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_recon, total_kl = 0.0, 0.0
        pbar = tqdm(train_loader, desc=f"  Epoch {epoch}/{n_epochs}", leave=False)
        for x, _, _ in pbar:
            x = x.to(device)
            recon, mu, log_var = model(x)
            x_01 = (x * 0.3081 + 0.1307).clamp(0, 1)
            recon_loss = F.binary_cross_entropy(recon, x_01, reduction="sum") / x.size(0)
            kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(1).mean()
            loss = recon_loss + backbone_cfg.kl_weight * kl
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_recon += recon_loss.item(); total_kl += kl.item()
            pbar.set_postfix(recon=f"{recon_loss.item():.2f}", kl=f"{kl.item():.2f}")
        n = len(train_loader)
        avg_recon, avg_kl = total_recon / n, total_kl / n
        print(f"  Epoch {epoch}/{n_epochs}  recon={avg_recon:.4f}  kl={avg_kl:.4f}")
        metrics.append([epoch, f"{avg_recon:.6f}", f"{avg_kl:.6f}", ""])

        # Save sample grid: originals | recons | prior samples
        model.eval()
        with torch.no_grad():
            x_fix = fixed_x[:8].to(device)
            recon_fix, _, _ = model(x_fix)
            z_prior = torch.randn(8, model.latent_dim, device=device)
            prior = model.decode(z_prior)
        path = os.path.join(out_dir, f"epoch_{epoch:03d}.png")
        _save_grid([_unnorm(x_fix.cpu()), recon_fix.cpu(), prior.cpu()], path)
        print(f"  -> saved: {path}")
        model.train()

    _save_metrics_csv(
        f"outputs/results/gen_vae{_tag}_metrics.csv", metrics,
        ["epoch", "recon_loss", "kl", "concept_mse"],
    )
    return model


# ---------------------------------------------------------------------------
# Experiment 2 — VAE + CBM
# ---------------------------------------------------------------------------

def exp_cbm_gen_mnist(device, *, epochs=None, seed=42, tag=None):
    print("\n=== [GEN 2] VAE + CBM — ArithmeticMNIST ===")
    backbone_cfg = ConvVAEConfig.load("experiments/configs/exp_cbm_gen_mnist_backbone.json")
    cbm_cfg      = CBMConfig.load("experiments/configs/cbm_mnist.json")
    train_cfg    = TrainConfig.load("experiments/configs/train_cbm_gen_mnist.json")
    set_seed(seed)
    n_epochs = epochs if epochs is not None else train_cfg.epochs

    train_loader, test_loader = get_arithmetic_mnist(
        batch_size=train_cfg.batch_size, num_workers=train_cfg.num_workers,
    )
    model     = ConvVAEwithCBM(backbone_cfg, cbm_cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=train_cfg.lr)
    fixed_x, _, _ = _random_fixed_batch(test_loader)
    _tag   = f"_{tag}" if tag else ""
    out_dir = f"outputs/samples/gen_cbm{_tag}"
    metrics = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_recon, total_kl, total_closs = 0.0, 0.0, 0.0
        pbar = tqdm(train_loader, desc=f"  Epoch {epoch}/{n_epochs}", leave=False)
        for x, c_true, _ in pbar:
            x, c_true = x.to(device), c_true.to(device)
            c_norm = (c_true - MNIST_CONCEPT_MEAN) / MNIST_CONCEPT_SCALE
            if random.random() < train_cfg.intervention_prob:
                recon, concepts, mu, log_var = model(x, interventions=c_norm)
            else:
                recon, concepts, mu, log_var = model(x)
            x_01 = (x * 0.3081 + 0.1307).clamp(0, 1)
            recon_loss = F.binary_cross_entropy(recon, x_01, reduction="sum") / x.size(0)
            kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(1).mean()
            concept_loss = F.mse_loss(concepts, c_norm)
            loss = recon_loss + backbone_cfg.kl_weight * kl + train_cfg.concept_weight * concept_loss
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_recon += recon_loss.item(); total_kl += kl.item(); total_closs += concept_loss.item()
            pbar.set_postfix(recon=f"{recon_loss.item():.2f}", kl=f"{kl.item():.2f}", c=f"{concept_loss.item():.3f}")
        n = len(train_loader)
        avg_recon, avg_kl, avg_c = total_recon / n, total_kl / n, total_closs / n
        print(f"  Epoch {epoch}/{n_epochs}  recon={avg_recon:.4f}  kl={avg_kl:.4f}  concept_mse={avg_c:.4f}")
        metrics.append([epoch, f"{avg_recon:.6f}", f"{avg_kl:.6f}", f"{avg_c:.6f}"])

        model.eval()
        with torch.no_grad():
            x_fix = fixed_x[:8].to(device)
            recon_fix, _, _, _ = model(x_fix)
            z_prior = torch.randn(8, backbone_cfg.latent_dim, device=device)
            c_prior = model.cbm(z_prior)
            prior = model.decode(c_prior)
        path = os.path.join(out_dir, f"epoch_{epoch:03d}.png")
        _save_grid([_unnorm(x_fix.cpu()), recon_fix.cpu(), prior.cpu()], path)
        print(f"  -> saved: {path}")
        model.train()

    _save_metrics_csv(
        f"outputs/results/gen_cbm{_tag}_metrics.csv", metrics,
        ["epoch", "recon_loss", "kl", "concept_mse"],
    )
    return model


# ---------------------------------------------------------------------------
# Experiment 3 — VAE + CEM (standard dual-embedding)
# ---------------------------------------------------------------------------

def exp_cem_gen_mnist(device, *, epochs=None, seed=42, tag=None):
    print("\n=== [GEN 3] VAE + CEM — ArithmeticMNIST ===")
    backbone_cfg = ConvVAEConfig.load("experiments/configs/exp_cem_gen_mnist_backbone.json")
    cem_cfg      = CEMConfig.load("experiments/configs/cem_mnist.json")
    train_cfg    = TrainConfig.load("experiments/configs/train_cem_gen_mnist.json")
    set_seed(seed)
    n_epochs = epochs if epochs is not None else train_cfg.epochs

    train_loader, test_loader = get_arithmetic_mnist(
        batch_size=train_cfg.batch_size, num_workers=train_cfg.num_workers,
    )
    model     = ConvVAEwithCEM(backbone_cfg, cem_cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=train_cfg.lr)
    fixed_x, _, _ = _random_fixed_batch(test_loader)
    _tag   = f"_{tag}" if tag else ""
    out_dir = f"outputs/samples/gen_cem{_tag}"
    metrics = []
    kl_warmup = 5

    for epoch in range(1, n_epochs + 1):
        model.train()
        kl_w = min(epoch / kl_warmup, 1.0) * backbone_cfg.kl_weight
        total_recon, total_kl, total_closs = 0.0, 0.0, 0.0
        pbar = tqdm(train_loader, desc=f"  Epoch {epoch}/{n_epochs}", leave=False)
        for x, c_true, _ in pbar:
            x, c_true = x.to(device), c_true.to(device)
            c_norm = (c_true - MNIST_CONCEPT_MEAN) / MNIST_CONCEPT_SCALE
            if random.random() < train_cfg.intervention_prob:
                recon, concepts, mu, log_var = model(x, interventions=c_norm)
            else:
                recon, concepts, mu, log_var = model(x)
            x_01 = (x * 0.3081 + 0.1307).clamp(0, 1)
            recon_loss = F.binary_cross_entropy(recon, x_01, reduction="sum") / x.size(0)
            kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(1).mean()
            concept_loss = F.mse_loss(concepts, c_norm)
            loss = recon_loss + kl_w * kl + train_cfg.concept_weight * concept_loss
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_recon += recon_loss.item(); total_kl += kl.item(); total_closs += concept_loss.item()
            pbar.set_postfix(recon=f"{recon_loss.item():.2f}", kl=f"{kl.item():.2f}", c=f"{concept_loss.item():.3f}")
        n = len(train_loader)
        avg_recon, avg_kl, avg_c = total_recon / n, total_kl / n, total_closs / n
        print(f"  Epoch {epoch}/{n_epochs}  kl_w={kl_w:.2f}  recon={avg_recon:.4f}  kl={avg_kl:.4f}  concept_mse={avg_c:.4f}")
        metrics.append([epoch, f"{avg_recon:.6f}", f"{avg_kl:.6f}", f"{avg_c:.6f}"])

        model.eval()
        with torch.no_grad():
            x_fix = fixed_x[:8].to(device)
            recon_fix, _, _, _ = model(x_fix)
            z_prior = torch.randn(8, backbone_cfg.latent_dim, device=device)
            emb, _ = model.cem(z_prior)
            prior = model.decoder(emb).reconstruction
        path = os.path.join(out_dir, f"epoch_{epoch:03d}.png")
        _save_grid([_unnorm(x_fix.cpu()), recon_fix.cpu(), prior.cpu()], path)
        print(f"  -> saved: {path}")
        model.train()

    _save_metrics_csv(
        f"outputs/results/gen_cem{_tag}_metrics.csv", metrics,
        ["epoch", "recon_loss", "kl", "concept_mse"],
    )
    return model


# ---------------------------------------------------------------------------
# Experiment 4 — VAE + CEM-Linear (multiplicative)
# ---------------------------------------------------------------------------

def exp_cem_linear_gen_mnist(device, *, epochs=None, seed=42, tag=None):
    print("\n=== [GEN 4] VAE + CEM-Linear — ArithmeticMNIST ===")
    # Reuses the same backbone and train configs as standard CEM
    backbone_cfg = ConvVAEConfig.load("experiments/configs/exp_cem_gen_mnist_backbone.json")
    cem_cfg      = CEMConfig.load("experiments/configs/cem_mnist.json")
    train_cfg    = TrainConfig.load("experiments/configs/train_cem_gen_mnist.json")
    set_seed(seed)
    n_epochs = epochs if epochs is not None else train_cfg.epochs

    train_loader, test_loader = get_arithmetic_mnist(
        batch_size=train_cfg.batch_size, num_workers=train_cfg.num_workers,
    )
    model     = ConvVAEwithCEMLinear(backbone_cfg, cem_cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=train_cfg.lr)
    fixed_x, _, _ = _random_fixed_batch(test_loader)
    _tag   = f"_{tag}" if tag else ""
    out_dir = f"outputs/samples/gen_cem_linear{_tag}"
    metrics = []
    kl_warmup = 5

    for epoch in range(1, n_epochs + 1):
        model.train()
        kl_w = min(epoch / kl_warmup, 1.0) * backbone_cfg.kl_weight
        total_recon, total_kl, total_closs = 0.0, 0.0, 0.0
        pbar = tqdm(train_loader, desc=f"  Epoch {epoch}/{n_epochs}", leave=False)
        for x, c_true, _ in pbar:
            x, c_true = x.to(device), c_true.to(device)
            c_norm = (c_true - MNIST_CONCEPT_MEAN) / MNIST_CONCEPT_SCALE
            if random.random() < train_cfg.intervention_prob:
                recon, concepts, mu, log_var = model(x, interventions=c_norm)
            else:
                recon, concepts, mu, log_var = model(x)
            x_01 = (x * 0.3081 + 0.1307).clamp(0, 1)
            recon_loss = F.binary_cross_entropy(recon, x_01, reduction="sum") / x.size(0)
            kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(1).mean()
            concept_loss = F.mse_loss(concepts, c_norm)
            loss = recon_loss + kl_w * kl + train_cfg.concept_weight * concept_loss
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_recon += recon_loss.item(); total_kl += kl.item(); total_closs += concept_loss.item()
            pbar.set_postfix(recon=f"{recon_loss.item():.2f}", kl=f"{kl.item():.2f}", c=f"{concept_loss.item():.3f}")
        n = len(train_loader)
        avg_recon, avg_kl, avg_c = total_recon / n, total_kl / n, total_closs / n
        print(f"  Epoch {epoch}/{n_epochs}  kl_w={kl_w:.2f}  recon={avg_recon:.4f}  kl={avg_kl:.4f}  concept_mse={avg_c:.4f}")
        metrics.append([epoch, f"{avg_recon:.6f}", f"{avg_kl:.6f}", f"{avg_c:.6f}"])

        model.eval()
        with torch.no_grad():
            x_fix = fixed_x[:8].to(device)
            recon_fix, _, _, _ = model(x_fix)
            z_prior = torch.randn(8, backbone_cfg.latent_dim, device=device)
            emb, _ = model.cem(z_prior)
            prior = model.decoder(emb).reconstruction
        path = os.path.join(out_dir, f"epoch_{epoch:03d}.png")
        _save_grid([_unnorm(x_fix.cpu()), recon_fix.cpu(), prior.cpu()], path)
        print(f"  -> saved: {path}")
        model.train()

    _save_metrics_csv(
        f"outputs/results/gen_cem_linear{_tag}_metrics.csv", metrics,
        ["epoch", "recon_loss", "kl", "concept_mse"],
    )
    return model


# ---------------------------------------------------------------------------
# Comparison panel — side-by-side visualization of all 4 models
# ---------------------------------------------------------------------------

def save_comparison_panel(
    vae_model,
    cbm_model,
    cem_model,
    cem_linear_model,
    fixed_x,
    device,
    out_path="outputs/samples/gen_comparison_final.png",
    n=8,
):
    """
    Creates a labelled matplotlib figure comparing all four models.

    Rows:
      0 — Original images
      1 — VAE reconstructions
      2 — CBM reconstructions
      3 — CEM reconstructions
      4 — CEM-Linear reconstructions
      --- separator ---
      5 — VAE prior samples
      6 — CBM prior samples
      7 — CEM prior samples
      8 — CEM-Linear prior samples
    """
    entries = [
        ("VAE",        vae_model,        "vae"),
        ("CBM",        cbm_model,        "cbm"),
        ("CEM",        cem_model,        "cem"),
        ("CEM-Linear", cem_linear_model, "cem"),
    ]

    # Gather images
    originals = None
    recons    = {}
    priors    = {}
    for name, model, mtype in entries:
        orig, rec, pri = _get_samples(model, mtype, fixed_x, n, device)
        if originals is None:
            originals = _unnorm(orig)
        recons[name] = rec.clamp(0, 1)
        priors[name] = pri.clamp(0, 1)

    row_data = (
        [("Original", originals)]
        + [(f"{name} recon", recons[name]) for name, _, _ in entries]
        + [(f"{name} prior", priors[name]) for name, _, _ in entries]
    )
    n_rows = len(row_data)

    fig, axes = plt.subplots(n_rows, n, figsize=(n * 1.8, n_rows * 1.8))
    fig.subplots_adjust(hspace=0.05, wspace=0.02)

    for row_idx, (row_label, row_imgs) in enumerate(row_data):
        for col in range(n):
            ax = axes[row_idx, col]
            img = row_imgs[col]
            if img.shape[0] == 1:
                ax.imshow(img.squeeze(0).numpy(), cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(img.permute(1, 2, 0).numpy(), vmin=0, vmax=1)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(row_label, fontsize=9, rotation=0,
                              labelpad=80, va="center", ha="right")

        # Thin separator line between recons and priors
        if row_idx == len(entries):
            for col in range(n):
                axes[row_idx, col].spines["top"].set_visible(True)
                axes[row_idx, col].spines["top"].set_linewidth(1.5)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  -> saved comparison panel: {out_path}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

EXPERIMENTS = {
    "vae":        exp_vae_gen_mnist,
    "cbm":        exp_cbm_gen_mnist,
    "cem":        exp_cem_gen_mnist,
    "cem_linear": exp_cem_linear_gen_mnist,
}


def main():
    parser = argparse.ArgumentParser(description="Generative experiments on ArithmeticMNIST")
    parser.add_argument(
        "--exp", choices=list(EXPERIMENTS.keys()), default=None,
        help="Run a single experiment. Omit to run all four and produce comparison.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    parser.add_argument("--seed",   type=int, default=42,   help="Random seed.")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    kwargs = dict(epochs=args.epochs, seed=args.seed)

    if args.exp is not None:
        EXPERIMENTS[args.exp](device, **kwargs)
        return

    # Run all four, then build comparison panel
    vae_model        = exp_vae_gen_mnist(device,        **kwargs)
    cbm_model        = exp_cbm_gen_mnist(device,        **kwargs)
    cem_model        = exp_cem_gen_mnist(device,        **kwargs)
    cem_linear_model = exp_cem_linear_gen_mnist(device, **kwargs)

    # Use a shared fixed test batch for the final comparison
    _, test_loader = get_arithmetic_mnist(batch_size=128, num_workers=2)
    set_seed(args.seed)
    fixed_x, _, _ = _random_fixed_batch(test_loader)

    save_comparison_panel(
        vae_model, cbm_model, cem_model, cem_linear_model,
        fixed_x, device,
        out_path="outputs/samples/gen_comparison_final.png",
    )
    print("\n[DONE]")


if __name__ == "__main__":
    main()
