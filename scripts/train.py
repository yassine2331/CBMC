"""
Train a baseline model on MNIST.

Usage:
    python scripts/train.py --model cnn
    python scripts/train.py --model vae
    python scripts/train.py --model vae --config experiments/configs/vae_large.json
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from cbmc.configs import CNNConfig, VAEConfig, TrainConfig
from cbmc.data.mnist import get_mnist
from cbmc.utils.sampling import save_vae_samples
from architectures.cnn_baseline import CNNBaseline
from architectures.vae_baseline import VAEBaseline, vae_loss


def set_seed(seed: int):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def accuracy(logits, labels):
    return (logits.argmax(1) == labels).float().mean().item()


def train_cnn(model_cfg: CNNConfig, train_cfg: TrainConfig, device: str):
    print(model_cfg)
    print(train_cfg)

    set_seed(train_cfg.seed)
    train_loader, test_loader = get_mnist(batch_size=train_cfg.batch_size, num_workers=train_cfg.num_workers)
    model     = CNNBaseline(model_cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=train_cfg.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        total_loss, total_acc = 0, 0

        pbar = tqdm(train_loader, desc=f"[CNN] Epoch {epoch}/{train_cfg.epochs}", leave=False)
        for x, y in pbar:
            x, y   = x.to(device), y.to(device)
            logits = model(x)
            loss   = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_acc  += accuracy(logits, y)
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{accuracy(logits, y):.4f}")

        n = len(train_loader)
        print(f"[CNN] Epoch {epoch}/{train_cfg.epochs}  loss={total_loss/n:.4f}  acc={total_acc/n:.4f}")

    model.eval()
    test_acc = 0
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="[CNN] Testing", leave=False):
            x, y      = x.to(device), y.to(device)
            test_acc += accuracy(model(x), y)
    print(f"\n[CNN] Test accuracy: {test_acc / len(test_loader):.4f}")
    return model


def train_vae(model_cfg: VAEConfig, train_cfg: TrainConfig, device: str):
    print(model_cfg)
    print(train_cfg)

    set_seed(train_cfg.seed)
    train_loader, test_loader = get_mnist(batch_size=train_cfg.batch_size, num_workers=train_cfg.num_workers)
    model     = VAEBaseline(model_cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=train_cfg.lr)

    fixed_x, _ = next(iter(test_loader))
    sample_dir  = os.path.join("outputs", "samples", "vae_baseline")

    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        total_loss, total_recon, total_kl = 0, 0, 0

        pbar = tqdm(train_loader, desc=f"[VAE] Epoch {epoch}/{train_cfg.epochs}", leave=False)
        for x, _ in pbar:
            x = x.to(device)
            recon, mu, log_var         = model(x)
            loss, recon_loss, kl       = vae_loss(recon, x, mu, log_var, model_cfg.kl_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss  += loss.item()
            total_recon += recon_loss.item()
            total_kl    += kl.item()
            pbar.set_postfix(loss=f"{loss.item():.2f}", recon=f"{recon_loss.item():.2f}", kl=f"{kl.item():.2f}")

        n = len(train_loader)
        print(f"[VAE] Epoch {epoch}/{train_cfg.epochs}  loss={total_loss/n:.2f}  recon={total_recon/n:.2f}  kl={total_kl/n:.2f}")
        save_vae_samples(model, fixed_x, epoch, sample_dir, device=device)

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  choices=["cnn", "vae"], required=True)
    parser.add_argument("--config", type=str, default=None, help="Path to a saved config JSON to load")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}\n")

    DEFAULT_CONFIGS = {
        "cnn": "experiments/configs/cnn_baseline.json",
        "vae": "experiments/configs/vae_baseline.json",
    }
    TRAIN_CONFIG = "experiments/configs/train_default.json"

    config_path = args.config or DEFAULT_CONFIGS[args.model]
    train_cfg   = TrainConfig.load(TRAIN_CONFIG)

    if args.model == "cnn":
        model_cfg = CNNConfig.load(config_path)
        train_cnn(model_cfg, train_cfg, device)
        model_cfg.save("outputs/cnn_baseline/model_config.json")
        train_cfg.save("outputs/cnn_baseline/train_config.json")

    elif args.model == "vae":
        model_cfg = VAEConfig.load(config_path)
        train_vae(model_cfg, train_cfg, device)
        model_cfg.save("outputs/vae_baseline/model_config.json")
        train_cfg.save("outputs/vae_baseline/train_config.json")


if __name__ == "__main__":
    main()
