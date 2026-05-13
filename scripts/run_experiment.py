"""
Run all baseline experiments.

Datasets:
  - Arithmetic MNIST  : two MNIST digits + operator → predict arithmetic result (regression)
  - Pendulum          : 96×96 RGBA images → predict 4 physical values (regression)

Each dataset loader returns (image, concepts, target) per batch.

Experiments:
  Generation (VAE — results saved as PNG grids):
    exp_gen_mnist          — vanilla VAE on ArithmeticMNIST
    exp_gen_pendulum       — vanilla ConvVAE on Pendulum
    exp_cbm_gen_mnist      — VAE+CBM on ArithmeticMNIST
    exp_cbm_gen_pendulum   — ConvVAE+CBM on Pendulum
    exp_cem_gen_mnist      — VAE+CEM on ArithmeticMNIST
    exp_cem_gen_pendulum   — ConvVAE+CEM on Pendulum

  Task (CNN — results saved as CSV):
    exp_cls_mnist          — CNN regression on ArithmeticMNIST  (metric: MSE)
    exp_cls_pendulum       — CNN regression on Pendulum         (metric: MSE, original scale)
    exp_cbm_cls_mnist      — CBM regression on ArithmeticMNIST
    exp_cbm_cls_pendulum   — CBM regression on Pendulum
    exp_cem_cls_mnist      — CEM regression on ArithmeticMNIST
    exp_cem_cls_pendulum   — CEM regression on Pendulum

Concept normalization:
  MNIST concepts [1,9] are normalized to [0,1] (/ 9.0) before supervision and interventions.
  Pendulum concepts are normalized with label_mean/label_std (same as task targets).
  This keeps concept values in the CEM blending gate's effective range.

Usage:
    python scripts/run_experiment.py                         # run all
    python scripts/run_experiment.py --exp exp_gen_mnist     # run one
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

from cbmc.configs import (
    ConvVAEConfig, CNNRegressionConfig,
    CBMConfig, CEMConfig, TrainConfig,
)
from cbmc.data.arithmetic_mnist import get_arithmetic_mnist
from cbmc.data.pendulum import get_pendulum
from cbmc.utils.sampling import save_vae_samples
from architectures.conv_vae_baseline import ConvVAEBaseline, conv_vae_loss
from architectures.cnn_regression import CNNRegression
from architectures.cnn_cbm import CNNwithCBM
from architectures.cnn_cem import CNNwithCEM
from architectures.conv_vae_cbm import ConvVAEwithCBM, conv_vae_cbm_loss
from architectures.conv_vae_cem import ConvVAEwithCEM, conv_vae_cem_loss


def conv_vae_mnist_loss(recon, x, mu, log_var, kl_weight=1.0):
    """Conv VAE loss for MNIST: unnormalize x before BCE so targets stay in [0,1]."""
    x_01       = (x * 0.3081 + 0.1307).clamp(0, 1)
    recon_loss = F.binary_cross_entropy(recon, x_01, reduction="sum") / x.size(0)
    kl         = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(1).mean()
    return recon_loss + kl_weight * kl, recon_loss, kl

# Concept normalization: map digit values [1,9] → [-1,1] so the CEM blending
# gate w=clamp(score*0.5+0.5,0,1) uses its full [0,1] range.
# With /9 alone → [0.11,1], gate ∈ [0.56,1], negative embedding never used.
MNIST_CONCEPT_MEAN  = 5.0
MNIST_CONCEPT_SCALE = 4.0   # (c - 5) / 4  maps  1→-1, 5→0, 9→+1


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
    """Return a single randomly-positioned batch from a loader for visualisation."""
    dataset = loader.dataset
    idx = random.sample(range(len(dataset)), loader.batch_size)
    return loader.collate_fn([dataset[i] for i in idx])

def save_csv(path, rows, header):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  -> saved results: {path}")


# ---------------------------------------------------------------------------
# Generation experiments — vanilla baselines
# ---------------------------------------------------------------------------

def exp_gen_mnist(device, *, epochs=None, operators=None, digits=None, tag=None):
    print("\n=== [EXP 1] Generation — ArithmeticMNIST (ConvVAE) ===")
    model_cfg = ConvVAEConfig.load("experiments/configs/exp_gen_mnist.json")
    train_cfg = TrainConfig.load("experiments/configs/train_gen_mnist.json")
    set_seed(train_cfg.seed)
    n_epochs = epochs if epochs is not None else train_cfg.epochs

    train_loader, test_loader = get_arithmetic_mnist(
        batch_size=train_cfg.batch_size, num_workers=train_cfg.num_workers,
        operators=operators or ('+', 'x'), digits=digits,
    )
    model      = ConvVAEBaseline(model_cfg).to(device)
    optimizer  = optim.Adam(model.parameters(), lr=train_cfg.lr)
    fixed_x, _, _ = _random_fixed_batch(test_loader)
    _tag       = f"_{tag}" if tag else ""
    out_dir    = f"outputs/samples/exp_gen_mnist{_tag}"

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss, total_recon, total_kl = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"  Epoch {epoch}/{n_epochs}", leave=False)
        for x, _, _ in pbar:
            x = x.to(device)
            recon, mu, log_var  = model(x)
            loss, recon_l, kl   = conv_vae_mnist_loss(recon, x, mu, log_var, model_cfg.kl_weight)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item(); total_recon += recon_l.item(); total_kl += kl.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", kl=f"{kl.item():.4f}")
        n = len(train_loader)
        print(f"  Epoch {epoch}/{n_epochs}  loss={total_loss/n:.4f}  recon={total_recon/n:.4f}  kl={total_kl/n:.4f}")
        save_vae_samples(model, fixed_x, epoch, out_dir, device=device, mnist_unnorm=True)

    model_cfg.save(f"outputs/exp_gen_mnist{_tag}/model_config.json")
    train_cfg.save(f"outputs/exp_gen_mnist{_tag}/train_config.json")


def exp_gen_pendulum(device, *, epochs=None, operators=None, digits=None, tag=None):
    print("\n=== [EXP 2] Generation — Pendulum (vanilla ConvVAE) ===")
    model_cfg = ConvVAEConfig.load("experiments/configs/exp_gen_pendulum.json")
    train_cfg = TrainConfig.load("experiments/configs/train_gen_pendulum.json")
    set_seed(train_cfg.seed)
    n_epochs = epochs if epochs is not None else train_cfg.epochs

    train_loader, test_loader, _, _ = get_pendulum(
        batch_size=train_cfg.batch_size, num_workers=train_cfg.num_workers, img_size=model_cfg.img_size
    )
    model      = ConvVAEBaseline(model_cfg).to(device)
    optimizer  = optim.Adam(model.parameters(), lr=train_cfg.lr)
    fixed_x, _, _ = _random_fixed_batch(test_loader)
    _tag       = f"_{tag}" if tag else ""
    out_dir    = f"outputs/samples/exp_gen_pendulum{_tag}"

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss, total_recon, total_kl = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"  Epoch {epoch}/{n_epochs}", leave=False)
        for x, _, _ in pbar:
            x = x.to(device)
            recon, mu, log_var  = model(x)
            loss, recon_l, kl   = conv_vae_loss(recon, x, mu, log_var, model_cfg.kl_weight)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item(); total_recon += recon_l.item(); total_kl += kl.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", kl=f"{kl.item():.4f}")
        n = len(train_loader)
        print(f"  Epoch {epoch}/{n_epochs}  loss={total_loss/n:.4f}  recon={total_recon/n:.4f}  kl={total_kl/n:.4f}")
        save_vae_samples(model, fixed_x, epoch, out_dir, device=device)

    model_cfg.save(f"outputs/exp_gen_pendulum{_tag}/model_config.json")
    train_cfg.save(f"outputs/exp_gen_pendulum{_tag}/train_config.json")


# ---------------------------------------------------------------------------
# Task experiments (CNN, no concepts)
# ---------------------------------------------------------------------------

def exp_cls_mnist(device, *, epochs=None, operators=None, digits=None, seed=None, tag=None):
    print("\n=== [EXP 3] CNN Regression — ArithmeticMNIST ===")
    model_cfg = CNNRegressionConfig.load("experiments/configs/exp_cls_mnist.json")
    train_cfg = TrainConfig.load("experiments/configs/train_cls_mnist.json")
    set_seed(seed if seed is not None else train_cfg.seed)
    n_epochs = epochs if epochs is not None else train_cfg.epochs

    train_loader, test_loader = get_arithmetic_mnist(
        batch_size=train_cfg.batch_size, num_workers=train_cfg.num_workers,
        operators=operators or ('+', 'x'), digits=digits,
    )
    model     = CNNRegression(model_cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=train_cfg.lr)
    criterion = nn.MSELoss()
    rows = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"  Epoch {epoch}/{n_epochs}", leave=False)
        for x, _, y in pbar:
            x, y   = x.to(device), y.to(device).unsqueeze(1)
            preds  = model(x)
            loss   = criterion(preds, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(mse=f"{loss.item():.4f}")
        train_mse = total_loss / len(train_loader)
        print(f"  Epoch {epoch}/{n_epochs}  mse={train_mse:.4f}")
        rows.append([epoch, f"{train_mse:.6f}"])

    model.eval()
    test_mse = 0
    with torch.no_grad():
        for x, _, y in test_loader:
            x, y  = x.to(device), y.to(device).unsqueeze(1)
            test_mse += criterion(model(x), y).item()
    test_mse /= len(test_loader)
    print(f"  Test MSE: {test_mse:.4f}")
    rows.append(["test", f"{test_mse:.6f}"])

    _tag = f"_{tag}" if tag else ""
    save_csv(f"outputs/results/exp_cls_mnist{_tag}.csv", rows, ["epoch", "mse"])
    model_cfg.save(f"outputs/exp_cls_mnist{_tag}/model_config.json")
    train_cfg.save(f"outputs/exp_cls_mnist{_tag}/train_config.json")
    return test_mse


def exp_cls_pendulum(device, *, epochs=None, operators=None, digits=None, seed=None, tag=None):
    print("\n=== [EXP 4] CNN Regression — Pendulum ===")
    model_cfg = CNNRegressionConfig.load("experiments/configs/exp_cls_pendulum.json")
    train_cfg = TrainConfig.load("experiments/configs/train_cls_pendulum.json")
    set_seed(seed if seed is not None else train_cfg.seed)
    n_epochs = epochs if epochs is not None else train_cfg.epochs

    train_loader, test_loader, label_mean, label_std = get_pendulum(
        batch_size=train_cfg.batch_size, num_workers=train_cfg.num_workers
    )
    label_mean, label_std = label_mean.to(device), label_std.to(device)
    model      = CNNRegression(model_cfg).to(device)
    optimizer  = optim.Adam(model.parameters(), lr=train_cfg.lr)
    criterion  = nn.MSELoss()
    rows = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_orig_loss = 0
        pbar = tqdm(train_loader, desc=f"  Epoch {epoch}/{n_epochs}", leave=False)
        for x, _, y in pbar:
            x, y   = x.to(device), y.to(device)
            y_norm = (y - label_mean) / label_std
            preds  = model(x)
            loss   = criterion(preds, y_norm)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            with torch.no_grad():
                preds_orig = preds.detach() * label_std + label_mean
                total_orig_loss += criterion(preds_orig, y).item()
            pbar.set_postfix(mse=f"{loss.item():.4f}")
        train_mse = total_orig_loss / len(train_loader)
        print(f"  Epoch {epoch}/{n_epochs}  mse_orig={train_mse:.4f}")
        rows.append([epoch, f"{train_mse:.6f}"])

    model.eval()
    test_mse = 0
    with torch.no_grad():
        for x, _, y in test_loader:
            x, y  = x.to(device), y.to(device)
            preds = model(x) * label_std + label_mean
            test_mse += criterion(preds, y).item()
    test_mse /= len(test_loader)
    print(f"  Test MSE (original scale): {test_mse:.4f}")
    rows.append(["test", f"{test_mse:.6f}"])

    _tag = f"_{tag}" if tag else ""
    save_csv(f"outputs/results/exp_cls_pendulum{_tag}.csv", rows, ["epoch", "mse"])
    model_cfg.save(f"outputs/exp_cls_pendulum{_tag}/model_config.json")
    train_cfg.save(f"outputs/exp_cls_pendulum{_tag}/train_config.json")
    return test_mse

# ---------------------------------------------------------------------------
# CBM experiments — scalar concept bottleneck
# ---------------------------------------------------------------------------

def exp_cbm_gen_mnist(device, *, epochs=None, operators=None, digits=None, tag=None):
    print("\n=== [EXP 5] CBM Generation — ArithmeticMNIST ===")
    backbone_cfg = ConvVAEConfig.load("experiments/configs/exp_cbm_gen_mnist_backbone.json")
    cbm_cfg      = CBMConfig.load("experiments/configs/cbm_mnist.json")
    train_cfg    = TrainConfig.load("experiments/configs/train_cbm_gen_mnist.json")
    set_seed(train_cfg.seed)
    n_epochs = epochs if epochs is not None else train_cfg.epochs

    train_loader, test_loader = get_arithmetic_mnist(
        batch_size=train_cfg.batch_size, num_workers=train_cfg.num_workers,
        operators=operators or ('+', 'x'), digits=digits,
    )
    model      = ConvVAEwithCBM(backbone_cfg, cbm_cfg).to(device)
    optimizer  = optim.Adam(model.parameters(), lr=train_cfg.lr)
    fixed_x, _, _ = _random_fixed_batch(test_loader)
    _tag       = f"_{tag}" if tag else ""
    out_dir    = f"outputs/samples/exp_cbm_gen_mnist{_tag}"

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss, total_recon, total_kl = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"  Epoch {epoch}/{n_epochs}", leave=False)
        for x, c_true, _ in pbar:
            x, c_true = x.to(device), c_true.to(device)
            c_norm = (c_true - MNIST_CONCEPT_MEAN) / MNIST_CONCEPT_SCALE
            if random.random() < train_cfg.intervention_prob:
                recon, concepts, mu, log_var = model(x, interventions=c_norm)
            else:
                recon, concepts, mu, log_var = model(x)
            loss, recon_l, kl = conv_vae_mnist_loss(recon, x, mu, log_var, backbone_cfg.kl_weight)
            concept_loss = F.mse_loss(concepts, c_norm)
            loss = loss + train_cfg.concept_weight * concept_loss
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item(); total_recon += recon_l.item(); total_kl += kl.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", kl=f"{kl.item():.4f}", c_loss=f"{concept_loss.item():.3f}")
        n = len(train_loader)
        print(f"  Epoch {epoch}/{n_epochs}  loss={total_loss/n:.4f}  recon={total_recon/n:.4f}  kl={total_kl/n:.4f}")
        _save_conv_vae_cbm_samples(model, fixed_x, epoch, out_dir, device, mnist_unnorm=True)

    backbone_cfg.save(f"outputs/exp_cbm_gen_mnist{_tag}/backbone_config.json")
    cbm_cfg.save(f"outputs/exp_cbm_gen_mnist{_tag}/cbm_config.json")
    train_cfg.save(f"outputs/exp_cbm_gen_mnist{_tag}/train_config.json")


def exp_cbm_gen_pendulum(device, *, epochs=None, operators=None, digits=None, tag=None):
    print("\n=== [EXP 6] CBM Generation — Pendulum ===")
    backbone_cfg = ConvVAEConfig.load("experiments/configs/exp_cbm_gen_pendulum_backbone.json")
    cbm_cfg      = CBMConfig.load("experiments/configs/cbm_pendulum.json")
    train_cfg    = TrainConfig.load("experiments/configs/train_cbm_gen_pendulum.json")
    set_seed(train_cfg.seed)
    n_epochs = epochs if epochs is not None else train_cfg.epochs

    train_loader, test_loader, label_mean, label_std = get_pendulum(
        batch_size=train_cfg.batch_size, num_workers=train_cfg.num_workers, img_size=backbone_cfg.img_size
    )
    label_mean, label_std = label_mean.to(device), label_std.to(device)
    model      = ConvVAEwithCBM(backbone_cfg, cbm_cfg).to(device)
    optimizer  = optim.Adam(model.parameters(), lr=train_cfg.lr)
    fixed_x, _, _ = _random_fixed_batch(test_loader)
    _tag       = f"_{tag}" if tag else ""
    out_dir    = f"outputs/samples/exp_cbm_gen_pendulum{_tag}"

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss, total_recon, total_kl = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"  Epoch {epoch}/{n_epochs}", leave=False)
        for x, c_true, _ in pbar:
            x, c_true = x.to(device), c_true.to(device)
            c_norm = (c_true - label_mean) / label_std
            if random.random() < train_cfg.intervention_prob:
                recon, concepts, mu, log_var = model(x, interventions=c_norm)
            else:
                recon, concepts, mu, log_var = model(x)
            loss, recon_l, kl = conv_vae_cbm_loss(recon, x, mu, log_var, backbone_cfg.kl_weight)
            concept_loss = F.mse_loss(concepts, c_norm)
            loss = loss + train_cfg.concept_weight * concept_loss
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item(); total_recon += recon_l.item(); total_kl += kl.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", kl=f"{kl.item():.4f}", c_loss=f"{concept_loss.item():.3f}")
        n = len(train_loader)
        print(f"  Epoch {epoch}/{n_epochs}  loss={total_loss/n:.4f}  recon={total_recon/n:.4f}  kl={total_kl/n:.4f}")
        _save_conv_vae_cbm_samples(model, fixed_x, epoch, out_dir, device)

    backbone_cfg.save(f"outputs/exp_cbm_gen_pendulum{_tag}/backbone_config.json")
    cbm_cfg.save(f"outputs/exp_cbm_gen_pendulum{_tag}/cbm_config.json")
    train_cfg.save(f"outputs/exp_cbm_gen_pendulum{_tag}/train_config.json")


def exp_cbm_cls_mnist(device, *, epochs=None, operators=None, digits=None, seed=None, tag=None):
    print("\n=== [EXP 7] CBM Regression — ArithmeticMNIST ===")
    backbone_cfg = CNNRegressionConfig.load("experiments/configs/exp_cbm_cls_mnist_backbone.json")
    cbm_cfg      = CBMConfig.load("experiments/configs/cbm_mnist.json")
    train_cfg    = TrainConfig.load("experiments/configs/train_cbm_mnist.json")
    set_seed(seed if seed is not None else train_cfg.seed)
    n_epochs = epochs if epochs is not None else train_cfg.epochs

    train_loader, test_loader = get_arithmetic_mnist(
        batch_size=train_cfg.batch_size, num_workers=train_cfg.num_workers,
        operators=operators or ('+', 'x'), digits=digits,
    )
    model     = CNNwithCBM(backbone_cfg, cbm_cfg, n_outputs=backbone_cfg.n_outputs).to(device)
    optimizer = optim.Adam(model.parameters(), lr=train_cfg.lr)
    criterion = nn.MSELoss()
    rows = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_task_loss = 0
        pbar = tqdm(train_loader, desc=f"  Epoch {epoch}/{n_epochs}", leave=False)
        for x, c_true, y in pbar:
            x, c_true, y = x.to(device), c_true.to(device), y.to(device).unsqueeze(1)
            c_norm = (c_true - MNIST_CONCEPT_MEAN) / MNIST_CONCEPT_SCALE
            if random.random() < train_cfg.intervention_prob:
                preds, concepts = model(x, interventions=c_norm)
            else:
                preds, concepts = model(x)
            task_loss    = criterion(preds, y)
            concept_loss = F.mse_loss(concepts, c_norm)
            loss = task_loss + train_cfg.concept_weight * concept_loss
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_task_loss += task_loss.item()
            pbar.set_postfix(mse=f"{task_loss.item():.4f}", c_loss=f"{concept_loss.item():.3f}")
        train_mse = total_task_loss / len(train_loader)
        print(f"  Epoch {epoch}/{n_epochs}  mse={train_mse:.4f}")
        rows.append([epoch, f"{train_mse:.6f}"])

    model.eval()
    test_mse = 0
    concept_mse = 0 
    test_mse_interv = 0

    test_map = 0
    with torch.no_grad():
        for x, c_true, y in test_loader:
            x, c_true, y = x.to(device), c_true.to(device), y.to(device).unsqueeze(1)
            c_norm = (c_true - MNIST_CONCEPT_MEAN) / MNIST_CONCEPT_SCALE
            preds, concepts = model(x)
            test_mse += criterion(preds, y).item()
            concept_mse += F.mse_loss(concepts, c_norm).item()
            #intervent on concepts and see effect on predictions 
            preds_interv, _ = model(x, interventions=c_norm)
            test_mse_interv += criterion(preds_interv, y).item()
            test_map += ((preds - y).abs().mean().item() )  # % of samples where intervention helped

    test_mse /= len(test_loader)
    concept_mse /= len(test_loader)
    test_mse_interv /= len(test_loader)
    print(f"  Test MSE: {test_mse:.4f}")
    print(f"  Concept MSE: {concept_mse:.4f}")
    print(f"  Test MSE with Interventions: {test_mse_interv:.4f}")
    print(f"  Test MAP: {test_map:.4f}")
    rows.append(["test", f"{test_mse:.6f}"])

    _tag = f"_{tag}" if tag else ""
    save_csv(f"outputs/results/exp_cbm_cls_mnist{_tag}.csv", rows, ["epoch", "mse"])
    backbone_cfg.save(f"outputs/exp_cbm_cls_mnist{_tag}/backbone_config.json")
    cbm_cfg.save(f"outputs/exp_cbm_cls_mnist{_tag}/cbm_config.json")
    train_cfg.save(f"outputs/exp_cbm_cls_mnist{_tag}/train_config.json")
    return test_mse, concept_mse, test_mse_interv, test_map/len(test_loader)

def exp_cbm_cls_pendulum(device, *, epochs=None, operators=None, digits=None, seed=None, tag=None):
    print("\n=== [EXP 8] CBM Regression — Pendulum ===")
    backbone_cfg = CNNRegressionConfig.load("experiments/configs/exp_cbm_cls_pendulum_backbone.json")
    cbm_cfg      = CBMConfig.load("experiments/configs/cbm_pendulum.json")
    train_cfg    = TrainConfig.load("experiments/configs/train_cbm_pendulum.json")
    set_seed(seed if seed is not None else train_cfg.seed)
    n_epochs = epochs if epochs is not None else train_cfg.epochs

    train_loader, test_loader, label_mean, label_std = get_pendulum(
        batch_size=train_cfg.batch_size, num_workers=train_cfg.num_workers
    )
    label_mean, label_std = label_mean.to(device), label_std.to(device)
    model     = CNNwithCBM(backbone_cfg, cbm_cfg, n_outputs=backbone_cfg.n_outputs).to(device)
    optimizer = optim.Adam(model.parameters(), lr=train_cfg.lr)
    criterion = nn.MSELoss()
    rows = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_orig_loss = 0
        pbar = tqdm(train_loader, desc=f"  Epoch {epoch}/{n_epochs}", leave=False)
        for x, c_true, y in pbar:
            x, c_true, y = x.to(device), c_true.to(device), y.to(device)
            c_norm = (c_true - label_mean) / label_std
            y_norm = (y - label_mean) / label_std
            if random.random() < train_cfg.intervention_prob:
                preds, concepts = model(x, interventions=c_norm)
            else:
                preds, concepts = model(x)
            task_loss    = criterion(preds, y_norm)
            concept_loss = F.mse_loss(concepts, c_norm)
            loss = task_loss + train_cfg.concept_weight * concept_loss
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            with torch.no_grad():
                preds_orig = preds.detach() * label_std + label_mean
                total_orig_loss += criterion(preds_orig, y).item()
            pbar.set_postfix(mse=f"{task_loss.item():.4f}", c_loss=f"{concept_loss.item():.3f}")
        train_mse = total_orig_loss / len(train_loader)
        print(f"  Epoch {epoch}/{n_epochs}  mse_orig={train_mse:.4f}")
        rows.append([epoch, f"{train_mse:.6f}"])

    model.eval()
    test_mse = 0
    concept_mse = 0
    intervention_mse = 0
    test_map = 0
    with torch.no_grad():
        for x, c_true, y in test_loader:
            x, c_true, y = x.to(device), c_true.to(device), y.to(device)
            c_norm = (c_true - label_mean) / label_std
            preds, concepts = model(x)
            preds = preds * label_std + label_mean
            test_mse += criterion(preds, y).item()
            concept_mse += F.mse_loss(concepts, c_norm).item()
            test_map += (preds - y).abs().mean().item()
            preds_interv, _ = model(x, interventions=c_norm)
            preds_interv = preds_interv * label_std + label_mean
            intervention_mse += criterion(preds_interv, y).item()
    test_mse /= len(test_loader)
    concept_mse /= len(test_loader)
    intervention_mse /= len(test_loader)
    print(f"  Test MSE (original scale): {test_mse:.4f}")
    print(f"  Concept MSE: {concept_mse:.4f}")
    print(f"  Intervention MSE: {intervention_mse:.4f}")
    print(f"  Test MAP: {test_map:.4f}")
    rows.append(["test", f"{test_mse:.6f}"])

    _tag = f"_{tag}" if tag else ""
    save_csv(f"outputs/results/exp_cbm_cls_pendulum{_tag}.csv", rows, ["epoch", "mse"])
    backbone_cfg.save(f"outputs/exp_cbm_cls_pendulum{_tag}/backbone_config.json")
    cbm_cfg.save(f"outputs/exp_cbm_cls_pendulum{_tag}/cbm_config.json")
    train_cfg.save(f"outputs/exp_cbm_cls_pendulum{_tag}/train_config.json")
    return test_mse, concept_mse, intervention_mse, test_map/len(test_loader)


# ---------------------------------------------------------------------------
# CEM experiments — concept embedding model
# ---------------------------------------------------------------------------

def exp_cem_gen_mnist(device, *, epochs=None, operators=None, digits=None, tag=None):
    print("\n=== [EXP 9] CEM Generation — ArithmeticMNIST ===")
    backbone_cfg = ConvVAEConfig.load("experiments/configs/exp_cem_gen_mnist_backbone.json")
    cem_cfg      = CEMConfig.load("experiments/configs/cem_mnist.json")
    train_cfg    = TrainConfig.load("experiments/configs/train_cem_gen_mnist.json")
    set_seed(train_cfg.seed)
    n_epochs = epochs if epochs is not None else train_cfg.epochs

    train_loader, test_loader = get_arithmetic_mnist(
        batch_size=train_cfg.batch_size, num_workers=train_cfg.num_workers,
        operators=operators or ('+', 'x'), digits=digits,
    )
    model      = ConvVAEwithCEM(backbone_cfg, cem_cfg).to(device)
    optimizer  = optim.Adam(model.parameters(), lr=train_cfg.lr)
    fixed_x, _, _ = _random_fixed_batch(test_loader)
    _tag       = f"_{tag}" if tag else ""
    out_dir    = f"outputs/samples/exp_cem_gen_mnist{_tag}"

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss, total_recon, total_kl = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"  Epoch {epoch}/{n_epochs}", leave=False)
        for x, c_true, _ in pbar:
            x, c_true = x.to(device), c_true.to(device)
            c_norm = (c_true - MNIST_CONCEPT_MEAN) / MNIST_CONCEPT_SCALE
            if random.random() < train_cfg.intervention_prob:
                recon, concepts, mu, log_var = model(x, interventions=c_norm)
            else:
                recon, concepts, mu, log_var = model(x)
            loss, recon_l, kl = conv_vae_mnist_loss(recon, x, mu, log_var, backbone_cfg.kl_weight)
            concept_loss = F.mse_loss(concepts, c_norm)
            loss = loss + train_cfg.concept_weight * concept_loss
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item(); total_recon += recon_l.item(); total_kl += kl.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", kl=f"{kl.item():.4f}", c_loss=f"{concept_loss.item():.3f}")
        n = len(train_loader)
        print(f"  Epoch {epoch}/{n_epochs}  loss={total_loss/n:.4f}  recon={total_recon/n:.4f}  kl={total_kl/n:.4f}")
        _save_conv_vae_cem_samples(model, fixed_x, epoch, out_dir, device, mnist_unnorm=True)

    backbone_cfg.save(f"outputs/exp_cem_gen_mnist{_tag}/backbone_config.json")
    cem_cfg.save(f"outputs/exp_cem_gen_mnist{_tag}/cem_config.json")
    train_cfg.save(f"outputs/exp_cem_gen_mnist{_tag}/train_config.json")


def exp_cem_gen_pendulum(device, *, epochs=None, operators=None, digits=None, tag=None):
    print("\n=== [EXP 10] CEM Generation — Pendulum ===")
    backbone_cfg = ConvVAEConfig.load("experiments/configs/exp_cem_gen_pendulum_backbone.json")
    cem_cfg      = CEMConfig.load("experiments/configs/cem_pendulum.json")
    train_cfg    = TrainConfig.load("experiments/configs/train_cem_gen_pendulum.json")
    set_seed(train_cfg.seed)
    n_epochs = epochs if epochs is not None else train_cfg.epochs

    train_loader, test_loader, label_mean, label_std = get_pendulum(
        batch_size=train_cfg.batch_size, num_workers=train_cfg.num_workers, img_size=backbone_cfg.img_size
    )
    label_mean, label_std = label_mean.to(device), label_std.to(device)
    model      = ConvVAEwithCEM(backbone_cfg, cem_cfg).to(device)
    optimizer  = optim.Adam(model.parameters(), lr=train_cfg.lr)
    fixed_x, _, _ = _random_fixed_batch(test_loader)
    _tag       = f"_{tag}" if tag else ""
    out_dir    = f"outputs/samples/exp_cem_gen_pendulum{_tag}"

    kl_warmup_epochs = 5

    for epoch in range(1, n_epochs + 1):
        model.train()
        kl_w = min(epoch / kl_warmup_epochs, 1.0) * backbone_cfg.kl_weight
        total_loss, total_recon, total_kl = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"  Epoch {epoch}/{n_epochs}", leave=False)
        for x, c_true, _ in pbar:
            x, c_true = x.to(device), c_true.to(device)
            c_norm = (c_true - label_mean) / label_std
            if random.random() < train_cfg.intervention_prob:
                recon, concepts, mu, log_var = model(x, interventions=c_norm)
            else:
                recon, concepts, mu, log_var = model(x)
            loss, recon_l, kl = conv_vae_cem_loss(recon, x, mu, log_var, kl_w)
            concept_loss = F.mse_loss(concepts, c_norm)
            loss = loss + train_cfg.concept_weight * concept_loss
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item(); total_recon += recon_l.item(); total_kl += kl.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", kl=f"{kl.item():.4f}", kl_w=f"{kl_w:.2f}", c_loss=f"{concept_loss.item():.3f}")
        n = len(train_loader)
        print(f"  Epoch {epoch}/{n_epochs}  kl_w={kl_w:.3f}  loss={total_loss/n:.4f}  recon={total_recon/n:.4f}  kl={total_kl/n:.4f}")
        _save_conv_vae_cem_samples(model, fixed_x, epoch, out_dir, device)

    backbone_cfg.save(f"outputs/exp_cem_gen_pendulum{_tag}/backbone_config.json")
    cem_cfg.save(f"outputs/exp_cem_gen_pendulum{_tag}/cem_config.json")
    train_cfg.save(f"outputs/exp_cem_gen_pendulum{_tag}/train_config.json")


def exp_cem_cls_mnist(device, *, epochs=None, operators=None, digits=None, seed=None, tag=None):
    print("\n=== [EXP 11] CEM Regression — ArithmeticMNIST ===")
    backbone_cfg = CNNRegressionConfig.load("experiments/configs/exp_cem_cls_mnist_backbone.json")
    cem_cfg      = CEMConfig.load("experiments/configs/cem_mnist.json")
    train_cfg    = TrainConfig.load("experiments/configs/train_cem_mnist.json")
    set_seed(seed if seed is not None else train_cfg.seed)
    n_epochs = epochs if epochs is not None else train_cfg.epochs

    train_loader, test_loader = get_arithmetic_mnist(
        batch_size=train_cfg.batch_size, num_workers=train_cfg.num_workers,
        operators=operators or ('+', 'x'), digits=digits,
    )
    model     = CNNwithCEM(backbone_cfg, cem_cfg, n_outputs=backbone_cfg.n_outputs).to(device)
    optimizer = optim.Adam(model.parameters(), lr=train_cfg.lr)
    criterion = nn.MSELoss()
    rows = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_task_loss = 0
        pbar = tqdm(train_loader, desc=f"  Epoch {epoch}/{n_epochs}", leave=False)
        for x, c_true, y in pbar:
            x, c_true, y = x.to(device), c_true.to(device), y.to(device).unsqueeze(1)
            c_norm = (c_true - MNIST_CONCEPT_MEAN) / MNIST_CONCEPT_SCALE
            if random.random() < train_cfg.intervention_prob:
                preds, concepts = model(x, interventions=c_norm)
            else:
                preds, concepts = model(x)
            task_loss    = criterion(preds, y)
            concept_loss = F.mse_loss(concepts, c_norm)
            loss = task_loss + train_cfg.concept_weight * concept_loss
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_task_loss += task_loss.item()
            pbar.set_postfix(mse=f"{task_loss.item():.4f}", c_loss=f"{concept_loss.item():.3f}")
        train_mse = total_task_loss / len(train_loader)
        print(f"  Epoch {epoch}/{n_epochs}  mse={train_mse:.4f}")
        rows.append([epoch, f"{train_mse:.6f}"])

    model.eval()
    test_mse = 0
    concept_mse = 0
    intervention_mse = 0
    test_map = 0
    with torch.no_grad():
        for x, c_true, y in test_loader:
            x, c_true, y = x.to(device), c_true.to(device), y.to(device).unsqueeze(1)
            c_norm = (c_true - MNIST_CONCEPT_MEAN) / MNIST_CONCEPT_SCALE
            preds, concepts = model(x)
            test_mse += criterion(preds, y).item()
            concept_mse += F.mse_loss(concepts, c_norm).item()
            test_map += (preds - y).abs().mean().item()
            preds_interv, _ = model(x, interventions=c_norm)
            intervention_mse += criterion(preds_interv, y).item()

    test_mse /= len(test_loader)
    concept_mse /= len(test_loader)
    intervention_mse /= len(test_loader)
    print(f"  Test MSE: {test_mse:.4f}")
    print(f"  Concept MSE: {concept_mse:.4f}")
    print(f"  Intervention MSE: {intervention_mse:.4f}")
    print(f"  Test MAP: {test_map:.4f}")
    rows.append(["test", f"{test_mse:.6f}"])

    _tag = f"_{tag}" if tag else ""
    save_csv(f"outputs/results/exp_cem_cls_mnist{_tag}.csv", rows, ["epoch", "mse"])
    backbone_cfg.save(f"outputs/exp_cem_cls_mnist{_tag}/backbone_config.json")
    cem_cfg.save(f"outputs/exp_cem_cls_mnist{_tag}/cem_config.json")
    train_cfg.save(f"outputs/exp_cem_cls_mnist{_tag}/train_config.json")
    return test_mse, concept_mse, intervention_mse, test_map/len(test_loader)


def exp_cem_cls_pendulum(device, *, epochs=None, operators=None, digits=None, seed=None, tag=None):
    print("\n=== [EXP 12] CEM Regression — Pendulum ===")
    backbone_cfg = CNNRegressionConfig.load("experiments/configs/exp_cem_cls_pendulum_backbone.json")
    cem_cfg      = CEMConfig.load("experiments/configs/cem_pendulum.json")
    train_cfg    = TrainConfig.load("experiments/configs/train_cem_pendulum.json")
    set_seed(seed if seed is not None else train_cfg.seed)
    n_epochs = epochs if epochs is not None else train_cfg.epochs

    train_loader, test_loader, label_mean, label_std = get_pendulum(
        batch_size=train_cfg.batch_size, num_workers=train_cfg.num_workers
    )
    label_mean, label_std = label_mean.to(device), label_std.to(device)
    model     = CNNwithCEM(backbone_cfg, cem_cfg, n_outputs=backbone_cfg.n_outputs).to(device)
    optimizer = optim.Adam(model.parameters(), lr=train_cfg.lr)
    criterion = nn.MSELoss()
    rows = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_orig_loss = 0
        pbar = tqdm(train_loader, desc=f"  Epoch {epoch}/{n_epochs}", leave=False)
        for x, c_true, y in pbar:
            x, c_true, y = x.to(device), c_true.to(device), y.to(device)
            c_norm = (c_true - label_mean) / label_std
            y_norm = (y - label_mean) / label_std
            if random.random() < train_cfg.intervention_prob:
                preds, concepts = model(x, interventions=c_norm)
            else:
                preds, concepts = model(x)
            task_loss    = criterion(preds, y_norm)
            concept_loss = F.mse_loss(concepts, c_norm)
            loss = task_loss + train_cfg.concept_weight * concept_loss
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            with torch.no_grad():
                preds_orig = preds.detach() * label_std + label_mean
                total_orig_loss += criterion(preds_orig, y).item()
            pbar.set_postfix(mse=f"{task_loss.item():.4f}", c_loss=f"{concept_loss.item():.3f}")
        train_mse = total_orig_loss / len(train_loader)
        print(f"  Epoch {epoch}/{n_epochs}  mse_orig={train_mse:.4f}")
        rows.append([epoch, f"{train_mse:.6f}"])

    model.eval()
    test_mse = 0
    concept_mse = 0
    intervention_mse = 0
    test_map = 0
    with torch.no_grad():
        for x, c_true, y in test_loader:
            x, c_true, y = x.to(device), c_true.to(device), y.to(device)
            c_norm = (c_true - label_mean) / label_std
            preds, concepts = model(x)
            preds = preds * label_std + label_mean
            test_mse += criterion(preds, y).item()
            concept_mse += F.mse_loss(concepts, c_norm).item()
            test_map += (preds - y).abs().mean().item()
            preds_interv, _ = model(x, interventions=c_norm)
            preds_interv = preds_interv * label_std + label_mean
            intervention_mse += criterion(preds_interv, y).item()

    concept_mse /= len(test_loader)
    intervention_mse /= len(test_loader)
    test_mse /= len(test_loader)
    print(f"  Concept MSE: {concept_mse:.4f}")
    print(f"  Intervention MSE: {intervention_mse:.4f}")
    print(f"  Test MSE (original scale): {test_mse:.4f}")
    print(f"  Test MAP: {test_map:.4f}")
    rows.append(["test", f"{test_mse:.6f}"])

    _tag = f"_{tag}" if tag else ""
    save_csv(f"outputs/results/exp_cem_cls_pendulum{_tag}.csv", rows, ["epoch", "mse"])
    backbone_cfg.save(f"outputs/exp_cem_cls_pendulum{_tag}/backbone_config.json")
    cem_cfg.save(f"outputs/exp_cem_cls_pendulum{_tag}/cem_config.json")
    train_cfg.save(f"outputs/exp_cem_cls_pendulum{_tag}/train_config.json")
    return test_mse, concept_mse, intervention_mse, test_map/len(test_loader)

# ---------------------------------------------------------------------------
# Sample helpers for concept models (VAE variants)
# ---------------------------------------------------------------------------

def _save_conv_vae_cbm_samples(model, fixed_x, epoch, out_dir, device, mnist_unnorm=False):
    """Save sample grid for ConvVAE+CBM: originals | recons | prior samples."""
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    n = 8
    with torch.no_grad():
        x = fixed_x[:n].to(device)
        recon, _, _, _ = model(x)
        z_prior   = torch.randn(n, model.backbone_cfg.latent_dim, device=device)
        c_prior   = model.cbm(z_prior)
        generated = model.decode(c_prior)
    x_disp = (x * 0.3081 + 0.1307).clamp(0, 1) if mnist_unnorm else x.clamp(0, 1)
    grid = vutils.make_grid(torch.cat([x_disp, recon.clamp(0,1), generated.clamp(0,1)]), nrow=n, padding=2)
    to_pil_image(grid.cpu()).save(os.path.join(out_dir, f"epoch_{epoch:03d}.png"))
    print(f"  -> saved samples: {out_dir}/epoch_{epoch:03d}.png")
    model.train()


def _save_conv_vae_cem_samples(model, fixed_x, epoch, out_dir, device, mnist_unnorm=False):
    """Save sample grid for ConvVAE+CEM: originals | recons | prior samples."""
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    n = 8
    with torch.no_grad():
        x         = fixed_x[:n].to(device)
        recon, _, _, _ = model(x)
        z_prior   = torch.randn(n, model.backbone_cfg.latent_dim, device=device)
        emb, _    = model.cem(z_prior)
        generated = model.decoder(emb).reconstruction
    x_disp = (x * 0.3081 + 0.1307).clamp(0, 1) if mnist_unnorm else x.clamp(0, 1)
    grid = vutils.make_grid(torch.cat([x_disp, recon.clamp(0,1), generated.clamp(0,1)]), nrow=n, padding=2)
    to_pil_image(grid.cpu()).save(os.path.join(out_dir, f"epoch_{epoch:03d}.png"))
    print(f"  -> saved samples: {out_dir}/epoch_{epoch:03d}.png")
    model.train()


# ---------------------------------------------------------------------------
# Extra utilities
# ---------------------------------------------------------------------------

import numpy as np
import numpy as np

def run_quantitative_suite(device, n_runs=5, epochs=None, operators=None, digits=None):
    quant_exps = [
        "exp_cls_mnist",# "exp_cls_pendulum",
        "exp_cbm_cls_mnist", #"exp_cbm_cls_pendulum",
        "exp_cem_cls_mnist", #"exp_cem_cls_pendulum"
    ]
    
    final_stats = {}

    for name in quant_exps:
        print(f"\n{'='*30}\nSTATISTICAL RUN: {name}\n{'='*30}")
        
        # Storage for the multiple runs
        task_mses = []
        concept_mses = []
        interv_mses = []
        test_maps = []

        for i in range(n_runs):
            print(f"\n>>> Run {i+1}/{n_runs} for {name}")
            seed = 41 + i 
            
            result = EXPERIMENTS[name](device, seed=seed, epochs=epochs, operators=operators, digits=digits)
            
            # Handle the 4-tuple return from CBM/CEM
            if isinstance(result, tuple) and len(result) == 4:
                task_mse, concept_mse, interv_mse, test_map = result
                task_mses.append(task_mse)
                concept_mses.append(concept_mse)
                interv_mses.append(interv_mse)
                test_maps.append(test_map)
            else:
                # Handle baseline return (single float)
                task_mses.append(result)

        # Helper to compute and store stats
        def compute_stats(data_list, key_name):
            if not data_list: return
            mean_val = np.mean(data_list)
            std_val = np.std(data_list)
            final_stats[f"{name}_{key_name}"] = {"mean": mean_val, "std": std_val}
            print(f"  {key_name.replace('_', ' ').title()}: {mean_val:.6f} ± {std_val:.6f}")

        print(f"\nFinal Statistics for {name}:")
        compute_stats(task_mses, "task_mse")
        compute_stats(concept_mses, "concept_mse")
        compute_stats(interv_mses, "intervention_mse")
        compute_stats(test_maps, "test_map")
    # Save summary to CSV
    summary_path = "outputs/results/quantitative_summary.csv"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    
    csv_rows = []
    for k, v in final_stats.items():
        csv_rows.append([k, f"{v['mean']:.6f}", f"{v['std']:.6f}"])
        
    save_csv(summary_path, csv_rows, ["metric", "mean", "std"])
    print(f"\n[DONE] Global summary saved to {summary_path}")



# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

EXPERIMENTS = {
    # Baselines (no concepts)
    "exp_gen_mnist":           exp_gen_mnist,
    "exp_gen_pendulum":        exp_gen_pendulum,
    "exp_cls_mnist":           exp_cls_mnist,
    "exp_cls_pendulum":        exp_cls_pendulum,
    # CBM (scalar concept bottleneck)
    "exp_cbm_gen_mnist":       exp_cbm_gen_mnist,
    "exp_cbm_gen_pendulum":    exp_cbm_gen_pendulum,
    "exp_cbm_cls_mnist":       exp_cbm_cls_mnist,
    "exp_cbm_cls_pendulum":    exp_cbm_cls_pendulum,
    # CEM (concept embedding model)
    "exp_cem_gen_mnist":       exp_cem_gen_mnist,
    "exp_cem_gen_pendulum":    exp_cem_gen_pendulum,
    "exp_cem_cls_mnist":       exp_cem_cls_mnist,
    "exp_cem_cls_pendulum":    exp_cem_cls_pendulum,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp",
        choices=list(EXPERIMENTS.keys()),
        default=None,
        help="Run one experiment by name. Omit to run all."
    )
    parser.add_argument(
        "--runs", type=int, default=5,
        help="Number of runs for quantitative suite."
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of training epochs from config."
    )
    parser.add_argument(
        "--operators", nargs="+", default=None,
        choices=['+', '-', 'x', '/'],
        help="MNIST operators to use (e.g. --operators + x). Default: + x"
    )
    parser.add_argument(
        "--digits", nargs="+", type=int, default=None,
        help="MNIST digits to restrict to (1-9, e.g. --digits 1 2 3). Default: all 1-9"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        choices=["cuda", "mps", "cpu"],
        help="Force a specific device. Default: auto-detect (cuda > mps > cpu)."
    )
    parser.add_argument(
        "--tag", type=str, default=None,
        help="Suffix appended to all output file/dir names (e.g. digit9, all_digits)."
    )
    args   = parser.parse_args()
    device = get_device(args.device)
    print(f"Device: {device}")

    kw = dict(
        epochs=args.epochs,
        operators=tuple(args.operators) if args.operators else None,
        digits=tuple(args.digits)   if args.digits   else None,
        tag=args.tag,
    )

    if args.exp:
        EXPERIMENTS[args.exp](device, **kw)
        print("\nAll done.")
    else:
        run_quantitative_suite(
            device, n_runs=args.runs, epochs=args.epochs,
            operators=tuple(args.operators) if args.operators else None,
            digits=tuple(args.digits) if args.digits else None,
        )
        print("\nQuantitative suite completed.")


if __name__ == "__main__":
    main()
