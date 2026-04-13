"""
Run all baseline experiments (no concepts).

4 experiments total:
  Generation (VAE, pure image generation — results saved as PNG grids):
    exp_gen_mnist      — VAE on MNIST
    exp_gen_pendulum   — ConvVAE on Pendulum

  Task (CNN, no concepts — results saved as CSV):
    exp_cls_mnist      — CNN classification on MNIST  (metric: accuracy)
    exp_cls_pendulum   — CNN regression on Pendulum   (metric: MSE)

Usage:
    python scripts/run_experiment.py                         # run all
    python scripts/run_experiment.py --exp exp_gen_mnist     # run one
"""

import argparse
import csv
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from cbmc.configs import VAEConfig, ConvVAEConfig, CNNConfig, CNNRegressionConfig, TrainConfig
from cbmc.data.mnist import get_mnist
from cbmc.data.pendulum import get_pendulum
from cbmc.utils.sampling import save_vae_samples
from architectures.vae_baseline import VAEBaseline, vae_loss
from architectures.conv_vae_baseline import ConvVAEBaseline, conv_vae_loss
from architectures.cnn_baseline import CNNBaseline
from architectures.cnn_regression import CNNRegression


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device():
    if torch.cuda.is_available():         return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def set_seed(seed):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_csv(path, rows, header):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  -> saved results: {path}")

def accuracy(logits, labels):
    return (logits.argmax(1) == labels).float().mean().item()


# ---------------------------------------------------------------------------
# Generation experiments
# ---------------------------------------------------------------------------

def exp_gen_mnist(device):
    print("\n=== [EXP 1] Generation — MNIST ===")
    model_cfg = VAEConfig.load("experiments/configs/exp_gen_mnist.json")
    train_cfg = TrainConfig.load("experiments/configs/train_gen_mnist.json")
    set_seed(train_cfg.seed)

    train_loader, test_loader = get_mnist(batch_size=train_cfg.batch_size, num_workers=train_cfg.num_workers)
    model      = VAEBaseline(model_cfg).to(device)
    optimizer  = optim.Adam(model.parameters(), lr=train_cfg.lr)
    fixed_x, _ = next(iter(test_loader))
    out_dir    = "outputs/samples/exp_gen_mnist"

    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        total_loss, total_recon, total_kl = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"  Epoch {epoch}/{train_cfg.epochs}", leave=False)
        for x, _ in pbar:
            x = x.to(device)
            recon, mu, log_var  = model(x)
            loss, recon_l, kl   = vae_loss(recon, x, mu, log_var, model_cfg.kl_weight)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item(); total_recon += recon_l.item(); total_kl += kl.item()
            pbar.set_postfix(loss=f"{loss.item():.2f}", kl=f"{kl.item():.2f}")
        n = len(train_loader)
        print(f"  Epoch {epoch}/{train_cfg.epochs}  loss={total_loss/n:.2f}  recon={total_recon/n:.2f}  kl={total_kl/n:.2f}")
        save_vae_samples(model, fixed_x, epoch, out_dir, device=device)

    model_cfg.save("outputs/exp_gen_mnist/model_config.json")
    train_cfg.save("outputs/exp_gen_mnist/train_config.json")


def exp_gen_pendulum(device):
    print("\n=== [EXP 2] Generation — Pendulum ===")
    model_cfg = ConvVAEConfig.load("experiments/configs/exp_gen_pendulum.json")
    train_cfg = TrainConfig.load("experiments/configs/train_gen_pendulum.json")
    set_seed(train_cfg.seed)

    train_loader, test_loader, _, _ = get_pendulum(
        batch_size=train_cfg.batch_size, num_workers=train_cfg.num_workers, img_size=model_cfg.img_size
    )
    model      = ConvVAEBaseline(model_cfg).to(device)
    optimizer  = optim.Adam(model.parameters(), lr=train_cfg.lr)
    fixed_x, _ = next(iter(test_loader))
    out_dir    = "outputs/samples/exp_gen_pendulum"

    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        total_loss, total_recon, total_kl = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"  Epoch {epoch}/{train_cfg.epochs}", leave=False)
        for x, _ in pbar:
            x = x.to(device)
            recon, mu, log_var  = model(x)
            loss, recon_l, kl   = conv_vae_loss(recon, x, mu, log_var, model_cfg.kl_weight)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item(); total_recon += recon_l.item(); total_kl += kl.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", kl=f"{kl.item():.4f}")
        n = len(train_loader)
        print(f"  Epoch {epoch}/{train_cfg.epochs}  loss={total_loss/n:.4f}  recon={total_recon/n:.4f}  kl={total_kl/n:.4f}")
        save_vae_samples(model, fixed_x, epoch, out_dir, device=device)

    model_cfg.save("outputs/exp_gen_pendulum/model_config.json")
    train_cfg.save("outputs/exp_gen_pendulum/train_config.json")


# ---------------------------------------------------------------------------
# Task experiments (CNN, no concepts)
# ---------------------------------------------------------------------------

def exp_cls_mnist(device):
    print("\n=== [EXP 3] CNN Classification — MNIST ===")
    model_cfg = CNNConfig.load("experiments/configs/exp_cls_mnist.json")
    train_cfg = TrainConfig.load("experiments/configs/train_cls_mnist.json")
    set_seed(train_cfg.seed)

    train_loader, test_loader = get_mnist(batch_size=train_cfg.batch_size, num_workers=train_cfg.num_workers)
    model      = CNNBaseline(model_cfg).to(device)
    optimizer  = optim.Adam(model.parameters(), lr=train_cfg.lr)
    criterion  = nn.CrossEntropyLoss()
    rows = []

    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        total_loss, total_acc = 0, 0
        pbar = tqdm(train_loader, desc=f"  Epoch {epoch}/{train_cfg.epochs}", leave=False)
        for x, y in pbar:
            x, y   = x.to(device), y.to(device)
            logits = model(x)
            loss   = criterion(logits, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item(); total_acc += accuracy(logits, y)
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{accuracy(logits,y):.4f}")
        n = len(train_loader)
        train_loss, train_acc = total_loss/n, total_acc/n
        print(f"  Epoch {epoch}/{train_cfg.epochs}  loss={train_loss:.4f}  acc={train_acc:.4f}")
        rows.append([epoch, f"{train_loss:.6f}", f"{train_acc:.6f}"])

    model.eval()
    test_acc = 0
    with torch.no_grad():
        for x, y in test_loader:
            test_acc += accuracy(model(x.to(device)), y.to(device))
    test_acc /= len(test_loader)
    print(f"  Test accuracy: {test_acc:.4f}")
    rows.append(["test", "", f"{test_acc:.6f}"])

    save_csv("outputs/results/exp_cls_mnist.csv", rows, ["epoch", "loss", "accuracy"])
    model_cfg.save("outputs/exp_cls_mnist/model_config.json")
    train_cfg.save("outputs/exp_cls_mnist/train_config.json")


def exp_cls_pendulum(device):
    print("\n=== [EXP 4] CNN Regression — Pendulum ===")
    model_cfg = CNNRegressionConfig.load("experiments/configs/exp_cls_pendulum.json")
    train_cfg = TrainConfig.load("experiments/configs/train_cls_pendulum.json")
    set_seed(train_cfg.seed)

    train_loader, test_loader, label_mean, label_std = get_pendulum(
        batch_size=train_cfg.batch_size, num_workers=train_cfg.num_workers
    )
    label_mean, label_std = label_mean.to(device), label_std.to(device)
    model      = CNNRegression(model_cfg).to(device)
    optimizer  = optim.Adam(model.parameters(), lr=train_cfg.lr)
    criterion  = nn.MSELoss()
    rows = []

    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"  Epoch {epoch}/{train_cfg.epochs}", leave=False)
        for x, y in pbar:
            x, y   = x.to(device), y.to(device)
            y_norm = (y - label_mean) / label_std
            preds  = model(x)
            loss   = criterion(preds, y_norm)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(mse=f"{loss.item():.4f}")
        train_mse = total_loss / len(train_loader)
        print(f"  Epoch {epoch}/{train_cfg.epochs}  mse={train_mse:.4f}")
        rows.append([epoch, f"{train_mse:.6f}"])

    model.eval()
    test_mse = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y  = x.to(device), y.to(device)
            preds = model(x) * label_std + label_mean   # denormalize
            test_mse += criterion(preds, y).item()
    test_mse /= len(test_loader)
    print(f"  Test MSE (original scale): {test_mse:.4f}")
    rows.append(["test", f"{test_mse:.6f}"])

    save_csv("outputs/results/exp_cls_pendulum.csv", rows, ["epoch", "mse"])
    model_cfg.save("outputs/exp_cls_pendulum/model_config.json")
    train_cfg.save("outputs/exp_cls_pendulum/train_config.json")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

EXPERIMENTS = {
    "exp_gen_mnist":    exp_gen_mnist,
    "exp_gen_pendulum": exp_gen_pendulum,
    "exp_cls_mnist":    exp_cls_mnist,
    "exp_cls_pendulum": exp_cls_pendulum,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp",
        choices=list(EXPERIMENTS.keys()),
        default=None,
        help="Run one experiment by name. Omit to run all."
    )
    args   = parser.parse_args()
    device = get_device()
    print(f"Device: {device}")

    to_run = [args.exp] if args.exp else list(EXPERIMENTS.keys())
    for name in to_run:
        EXPERIMENTS[name](device)
    print("\nAll done.")


if __name__ == "__main__":
    main()
