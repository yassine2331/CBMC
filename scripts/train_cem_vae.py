"""
Train a ConvVAE + CEM and save the model for later use.

Supported datasets:
  mnist     — ArithmeticMNIST (two digits + operator, greyscale 64×64)
  pendulum  — Pendulum        (RGB 64×64, 4 physical concepts)

Saves to  outputs/models/cem_vae_<dataset>/
  model.pt            — full checkpoint (state_dict + config dicts)
  backbone_config.json
  cem_config.json
  train_config.json
  samples/            — per-epoch reconstruction grids

Usage:
    python scripts/train_cem_vae.py                     # mnist, default epochs
    python scripts/train_cem_vae.py --dataset pendulum
    python scripts/train_cem_vae.py --epochs 30 --tag v2

Loading later:
    from scripts.train_cem_vae import load_cem_vae
    model, backbone_cfg, cem_cfg = load_cem_vae("outputs/models/cem_vae_mnist")
"""

import argparse
import os
import random
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from cbmc.configs import ConvVAEConfig, CEMConfig, TrainConfig
from architectures.conv_vae_cem import ConvVAEwithCEM

MNIST_CONCEPT_MEAN  = 5.0
MNIST_CONCEPT_SCALE = 4.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device(override=None):
    if override:
        return override
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

def _save_grid(tensors, path, n=8):
    rows = [t[:n].clamp(0, 1) for t in tensors]
    grid = vutils.make_grid(torch.cat(rows, dim=0), nrow=n, padding=2)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    to_pil_image(grid.cpu()).save(path)


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_cem_vae(model, backbone_cfg, cem_cfg, train_cfg, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "backbone_cfg":     backbone_cfg.__dict__,
        "cem_cfg":          cem_cfg.__dict__,
    }, ckpt_path)
    backbone_cfg.save(os.path.join(out_dir, "backbone_config.json"))
    cem_cfg.save(os.path.join(out_dir, "cem_config.json"))
    train_cfg.save(os.path.join(out_dir, "train_config.json"))
    print(f"\n  -> checkpoint saved to {out_dir}/")


def load_cem_vae(checkpoint_dir, device=None):
    """
    Reconstruct and return a trained ConvVAEwithCEM from a saved checkpoint.

    Returns:
        model        — ConvVAEwithCEM in eval mode on `device`
        backbone_cfg — ConvVAEConfig
        cem_cfg      — CEMConfig
    """
    if device is None:
        device = get_device()
    backbone_cfg = ConvVAEConfig.load(os.path.join(checkpoint_dir, "backbone_config.json"))
    cem_cfg      = CEMConfig.load(os.path.join(checkpoint_dir, "cem_config.json"))
    model        = ConvVAEwithCEM(backbone_cfg, cem_cfg).to(device)
    ckpt         = torch.load(os.path.join(checkpoint_dir, "model.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, backbone_cfg, cem_cfg


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _cycle_step(model, mu, c_target, optimizer):
    """
    Cycle loss backward: shuffle concepts → decode → re-encode → compare.
    Encoder grads are zeroed so only the decoder and CEM embedding bank update.
    Returns raw (unweighted) cycle loss value for logging.
    """
    emb_int, _ = model.cem(mu.detach(), interventions=c_target)
    recon_int  = model.decode(emb_int)
    enc_int    = model.encoder(recon_int)
    _, c_hat   = model.cem(enc_int.embedding)
    loss       = F.mse_loss(c_hat, c_target)
    optimizer.zero_grad()
    loss.backward()
    for p in model.encoder.parameters():
        if p.grad is not None:
            p.grad.zero_()
    optimizer.step()
    return loss.item()


def _cem_vae_loss_mnist(recon, x, mu, log_var, kl_weight):
    x_01       = (x * 0.3081 + 0.1307).clamp(0, 1)
    recon_loss = F.binary_cross_entropy(recon, x_01, reduction="sum") / x.size(0)
    kl         = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(1).mean()
    return recon_loss + kl_weight * kl, recon_loss, kl


def _cem_vae_loss_pendulum(recon, x, mu, log_var, kl_weight):
    recon_loss = F.mse_loss(recon, x, reduction="sum") / x.size(0)
    kl         = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(1).mean()
    return recon_loss + kl_weight * kl, recon_loss, kl


def train_mnist(device, epochs, out_dir):
    from cbmc.data.arithmetic_mnist import get_arithmetic_mnist

    backbone_cfg = ConvVAEConfig.load("experiments/configs/exp_cem_gen_mnist_backbone.json")
    cem_cfg      = CEMConfig.load("experiments/configs/cem_mnist.json")
    train_cfg    = TrainConfig.load("experiments/configs/train_cem_gen_mnist.json")
    set_seed(train_cfg.seed)
    n_epochs = epochs if epochs is not None else train_cfg.epochs

    train_loader, test_loader = get_arithmetic_mnist(
        batch_size=train_cfg.batch_size, num_workers=train_cfg.num_workers,
    )
    model     = ConvVAEwithCEM(backbone_cfg, cem_cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=train_cfg.lr)
    fixed_x, _, _ = _random_fixed_batch(test_loader)
    kl_warmup = 5

    for epoch in range(1, n_epochs + 1):
        model.train()
        kl_w = min(epoch / kl_warmup, 1.0) * backbone_cfg.kl_weight
        total_recon, total_kl, total_closs, total_cycle = 0.0, 0.0, 0.0, 0.0
        pbar = tqdm(train_loader, desc=f"  Epoch {epoch}/{n_epochs}", leave=False)
        for x, c_true, _ in pbar:
            x, c_true = x.to(device), c_true.to(device)
            c_norm    = (c_true - MNIST_CONCEPT_MEAN) / MNIST_CONCEPT_SCALE
            if random.random() < train_cfg.intervention_prob:
                recon, concepts, mu, log_var = model(x, interventions=c_norm)
            else:
                recon, concepts, mu, log_var = model(x)
            loss, recon_l, kl = _cem_vae_loss_mnist(recon, x, mu, log_var, kl_w)
            concept_loss = F.mse_loss(concepts, c_norm)
            loss = loss + train_cfg.concept_weight * concept_loss
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_recon += recon_l.item(); total_kl += kl.item(); total_closs += concept_loss.item()
            # cycle loss: shuffled target concepts → decode → re-encode → compare
            # encoder grads are zeroed inside _cycle_step so only decoder+CEM update
            cyc_l = 0.0
            if train_cfg.cycle_weight > 0:
                c_target = c_norm[torch.randperm(x.size(0), device=device)]
                cyc_l = train_cfg.cycle_weight * _cycle_step(model, mu, c_target, optimizer)
                total_cycle += cyc_l
            pbar.set_postfix(recon=f"{recon_l.item():.2f}", kl=f"{kl.item():.2f}",
                             c=f"{concept_loss.item():.3f}", cyc=f"{cyc_l:.3f}")
        n = len(train_loader)
        print(f"  Epoch {epoch}/{n_epochs}  kl_w={kl_w:.2f}  recon={total_recon/n:.4f}  kl={total_kl/n:.4f}  c_mse={total_closs/n:.4f}  cyc={total_cycle/n:.4f}")

        model.eval()
        with torch.no_grad():
            x_fix = fixed_x[:8].to(device)
            recon_fix, _, _, _ = model(x_fix)
            x_disp = (x_fix * 0.3081 + 0.1307).clamp(0, 1)
            z_prior = torch.randn(8, backbone_cfg.latent_dim, device=device)
            emb, _  = model.cem(z_prior)
            prior   = model.decoder(emb).reconstruction
        _save_grid([x_disp.cpu(), recon_fix.cpu(), prior.cpu()],
                   os.path.join(out_dir, "samples", f"epoch_{epoch:03d}.png"))
        model.train()

    save_cem_vae(model, backbone_cfg, cem_cfg, train_cfg, out_dir)
    return model


def train_pendulum(device, epochs, out_dir):
    from cbmc.data.pendulum import get_pendulum

    backbone_cfg = ConvVAEConfig.load("experiments/configs/exp_cem_gen_pendulum_backbone.json")
    cem_cfg      = CEMConfig.load("experiments/configs/cem_pendulum.json")
    train_cfg    = TrainConfig.load("experiments/configs/train_cem_gen_pendulum.json")
    set_seed(train_cfg.seed)
    n_epochs = epochs if epochs is not None else train_cfg.epochs

    train_loader, test_loader, label_mean, label_std = get_pendulum(
        batch_size=train_cfg.batch_size, num_workers=train_cfg.num_workers,
        img_size=backbone_cfg.img_size,
    )
    label_mean, label_std = label_mean.to(device), label_std.to(device)
    model     = ConvVAEwithCEM(backbone_cfg, cem_cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=train_cfg.lr)
    fixed_x, _, _ = _random_fixed_batch(test_loader)
    kl_warmup = 5

    for epoch in range(1, n_epochs + 1):
        model.train()
        kl_w = min(epoch / kl_warmup, 1.0) * backbone_cfg.kl_weight
        total_recon, total_kl, total_closs, total_cycle = 0.0, 0.0, 0.0, 0.0
        pbar = tqdm(train_loader, desc=f"  Epoch {epoch}/{n_epochs}", leave=False)
        for x, c_true, _ in pbar:
            x, c_true = x.to(device), c_true.to(device)
            c_norm    = (c_true - label_mean) / label_std
            if random.random() < train_cfg.intervention_prob:
                recon, concepts, mu, log_var = model(x, interventions=c_norm)
            else:
                recon, concepts, mu, log_var = model(x)
            loss, recon_l, kl = _cem_vae_loss_pendulum(recon, x, mu, log_var, kl_w)
            concept_loss = F.mse_loss(concepts, c_norm)
            loss = loss + train_cfg.concept_weight * concept_loss
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_recon += recon_l.item(); total_kl += kl.item(); total_closs += concept_loss.item()
            cyc_l = 0.0
            if train_cfg.cycle_weight > 0:
                c_target = c_norm[torch.randperm(x.size(0), device=device)]
                cyc_l = train_cfg.cycle_weight * _cycle_step(model, mu, c_target, optimizer)
                total_cycle += cyc_l
            pbar.set_postfix(recon=f"{recon_l.item():.2f}", kl=f"{kl.item():.2f}",
                             c=f"{concept_loss.item():.3f}", cyc=f"{cyc_l:.3f}")
        n = len(train_loader)
        print(f"  Epoch {epoch}/{n_epochs}  kl_w={kl_w:.2f}  recon={total_recon/n:.4f}  kl={total_kl/n:.4f}  c_mse={total_closs/n:.4f}  cyc={total_cycle/n:.4f}")

        model.eval()
        with torch.no_grad():
            x_fix = fixed_x[:8].to(device)
            recon_fix, _, _, _ = model(x_fix)
            z_prior = torch.randn(8, backbone_cfg.latent_dim, device=device)
            emb, _  = model.cem(z_prior)
            prior   = model.decoder(emb).reconstruction
        _save_grid([x_fix.cpu().clamp(0, 1), recon_fix.cpu(), prior.cpu()],
                   os.path.join(out_dir, "samples", f"epoch_{epoch:03d}.png"))
        model.train()

    save_cem_vae(model, backbone_cfg, cem_cfg, train_cfg, out_dir)
    return model


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mnist", "pendulum"], default="mnist")
    parser.add_argument("--epochs",  type=int,  default=None,    help="Override epochs from config")
    parser.add_argument("--tag",     type=str,  default=None,    help="Suffix for the output directory")
    parser.add_argument("--device",  type=str,  default=None,    choices=["cuda", "mps", "cpu"])
    args = parser.parse_args()

    device  = get_device(args.device)
    _tag    = f"_{args.tag}" if args.tag else ""
    out_dir = f"outputs/models/cem_vae_{args.dataset}{_tag}"

    print(f"Device  : {device}")
    print(f"Dataset : {args.dataset}")
    print(f"Output  : {out_dir}\n")

    if args.dataset == "mnist":
        train_mnist(device, args.epochs, out_dir)
    else:
        train_pendulum(device, args.epochs, out_dir)

    print(f"\n[DONE] Load the model later with:")
    print(f"  from scripts.train_cem_vae import load_cem_vae")
    print(f"  model, backbone_cfg, cem_cfg = load_cem_vae(\"{out_dir}\")")


if __name__ == "__main__":
    main()
