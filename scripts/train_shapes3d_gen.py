"""
Train, save, load and generate with concept-bottleneck VAEs on Shapes3D.

Every model shares the same ConvVAE encoder/decoder; only the bottleneck
between z and the decoder changes:

    name          aliases              bottleneck                       concepts as
    vae           baseline             none (plain VAE)                 --
    cbm                                scalar CBM (Koh et al.)          [-1, 1] values
    cem           case3.5, case3_5     CEM, binary-anchor interpolation [-1, 1] values
    cem_linear                         CEMLinear, c_i * phi_i(x)        [-1, 1] values
    categorical   case2                CEMCategorical, native states    state indices
    binned        case3                CEMCategorical over n_bins bins  bin indices

Command line:
    python scripts/train_shapes3d_gen.py --model cem --epochs 10
    python scripts/train_shapes3d_gen.py --model case3 --n-bins 5 --tag k5

From Python / a notebook:
    from scripts.train_shapes3d_gen import train, load
    gen = train("cem", epochs=10)                 # trains and saves
    gen = load("outputs/models/shapes3d_cem")     # later
    imgs = gen.generate(n=8, concepts={"shape": 2, "object_hue": 0.3}, seed=0)

Saved to outputs/models/shapes3d_<model>[_<tag>]/:
    model.pt              state_dict + every config + the ConceptSpace
    *_config.json         the same configs, human-readable
    metrics.csv           per-epoch train/test curves
    samples/epoch_XXX.png originals | reconstructions | prior samples
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from torchvision.transforms.functional import to_pil_image
from tqdm.auto import tqdm

from cbmc.configs import (ConvVAEConfig, CBMConfig, CEMConfig,
                          CEMCategoricalConfig, TrainConfig)
from cbmc.data.shapes3d import ConceptSpace, FACTOR_NAMES, FACTORS, get_shapes3d
from architectures.conv_vae_baseline import ConvVAEBaseline
from architectures.conv_vae_cbm import ConvVAEwithCBM
from architectures.conv_vae_cem import ConvVAEwithCEM
from architectures.conv_vae_cem_linear import ConvVAEwithCEMLinear
from architectures.conv_vae_cem_categorical import ConvVAEwithCEMCategorical


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

# name -> (architecture class, bottleneck config class, concept mode)
MODELS = {
    "vae":         (ConvVAEBaseline,           None,                 None),
    "cbm":         (ConvVAEwithCBM,            CBMConfig,            "continuous"),
    "cem":         (ConvVAEwithCEM,            CEMConfig,            "continuous"),
    "cem_linear":  (ConvVAEwithCEMLinear,      CEMConfig,            "continuous"),
    "categorical": (ConvVAEwithCEMCategorical, CEMCategoricalConfig, "categorical"),
    "binned":      (ConvVAEwithCEMCategorical, CEMCategoricalConfig, "binned"),
}
ALIASES = {
    "baseline": "vae", "vae_baseline": "vae",
    "case2": "categorical", "case_2": "categorical",
    "case3": "binned", "case_3": "binned",
    "case3.5": "cem", "case3_5": "cem", "case_3.5": "cem", "case35": "cem",
}
MODEL_CHOICES = list(MODELS) + list(ALIASES)


def resolve(name: str) -> str:
    key = name.strip().lower().replace(" ", "")
    key = ALIASES.get(key, key)
    if key not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {MODEL_CHOICES}")
    return key


def get_device(override=None):
    if override:                          return override
    if torch.cuda.is_available():         return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_configs(name: str, space: Optional[ConceptSpace]):
    """Sensible Shapes3D defaults; override any field after calling this."""
    backbone = ConvVAEConfig(in_channels=3, enc_channels=[32, 64, 128, 256], latent_dim=32,
                             dec_channels=[128, 64, 32, 32], img_size=64, kl_weight=1.0)
    _, bcls, _ = MODELS[name]
    if bcls is None:
        bottleneck = None
    elif bcls is CBMConfig:
        bottleneck = CBMConfig(n_concepts=space.n_concepts)
    elif bcls is CEMConfig:
        bottleneck = CEMConfig(n_concepts=space.n_concepts, embedding_dim=16,
                               hidden_dim=64, depth=2, dropout=0.0)
    else:
        bottleneck = CEMCategoricalConfig(n_concepts=space.n_concepts, n_states=space.n_states,
                                          embedding_dim=16, hidden_dim=64, depth=2,
                                          dropout=0.0, normalize=True)
    # cycle_weight > 0 is what makes interventions actually change the image;
    # without it the decoder learns to read everything from z (see _cycle_step).
    train_cfg = TrainConfig(epochs=20, lr=1e-3, batch_size=256, seed=42, num_workers=0,
                            concept_weight=100.0, intervention_prob=0.5, cycle_weight=100.0)
    return backbone, bottleneck, train_cfg


def build_model(name, backbone_cfg, bottleneck_cfg):
    arch, _, _ = MODELS[name]
    return arch(backbone_cfg) if bottleneck_cfg is None else arch(backbone_cfg, bottleneck_cfg)


# ---------------------------------------------------------------------------
# Uniform wrapper: one API for every bottleneck
# ---------------------------------------------------------------------------

class Generator:
    """
    Wraps a trained model so every variant generates the same way.

    concepts are always given as RAW factor values, e.g.
        {"shape": 2, "object_hue": 0.3, "orientation": -15}
    Concepts you leave out are predicted from z, as in a normal prior sample.
    The plain VAE has no concepts and ignores them.
    """

    def __init__(self, name, model, space, backbone_cfg, bottleneck_cfg, train_cfg,
                 device, data_cfg=None):
        self.name, self.model, self.space = name, model, space
        self.backbone_cfg, self.bottleneck_cfg, self.train_cfg = backbone_cfg, bottleneck_cfg, train_cfg
        self.device = device
        self.data_cfg = data_cfg or {}

    @property
    def has_concepts(self) -> bool:
        return self.space is not None

    @property
    def latent_dim(self) -> int:
        return self.backbone_cfg.latent_dim

    def __repr__(self):
        c = f", concepts={self.space.names}, mode={self.space.mode}" if self.space else ""
        return f"Generator({self.name}{c})"

    # -- internals ----------------------------------------------------------
    def _interventions(self, concepts: Optional[Dict[str, float]], B: int):
        """Raw-value dict -> (targets, mask) in the bottleneck's own encoding."""
        if not concepts or self.space is None:
            return None, None
        unknown = set(concepts) - set(self.space.names)
        if unknown:
            raise ValueError(f"model has no concept(s) {sorted(unknown)}; it has {self.space.names}")
        raw = torch.tensor([[float(concepts.get(n, FACTORS[n][0])) for n in self.space.names]],
                           device=self.device).repeat(B, 1)
        mask = torch.tensor([[float(n in concepts) for n in self.space.names]],
                            device=self.device).repeat(B, 1)
        return self.space.encode(raw), mask

    def _decode_from_z(self, z, concepts=None):
        """z -> (images, raw concept values actually used / predicted)."""
        m = self.model
        B = z.size(0)
        target, mask = self._interventions(concepts, B)
        if self.name == "vae":
            return m.decode(z), None
        if self.name == "cbm":
            c = m.cbm(z)
            if target is not None:
                c = mask * target.float() + (1 - mask) * c
            return m.decode(c), self.space.decode(c)
        emb, pred = m.cem(z, target, mask)
        used = self.space.decode(pred)
        if target is not None:                                   # report the intervened values
            used = mask * self.space.decode(target) + (1 - mask) * used
        return m.decoder(emb).reconstruction, used

    @staticmethod
    def _to_float(x):
        return x.float().div(255) if x.dtype == torch.uint8 else x.float()

    # -- public -------------------------------------------------------------
    def prior(self, n: int = 1, seed: Optional[int] = None) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed) if seed is not None else None
        return torch.randn(n, self.latent_dim, generator=g).to(self.device)

    @torch.no_grad()
    def generate(self, n: int = 8, concepts: Optional[Dict[str, float]] = None,
                 seed: Optional[int] = None, z: Optional[torch.Tensor] = None,
                 return_concepts: bool = False):
        """Images [n, 3, 64, 64] in [0, 1] from the prior (or a given z)."""
        self.model.eval()
        z = self.prior(n, seed) if z is None else z.to(self.device)
        imgs, used = self._decode_from_z(z, concepts)
        imgs = imgs.clamp(0, 1).cpu()
        return (imgs, None if used is None else used.cpu()) if return_concepts else imgs

    @torch.no_grad()
    def encode(self, x) -> torch.Tensor:
        """Posterior mean z for images (uint8 or float in [0, 1])."""
        self.model.eval()
        return self.model.encoder(self._to_float(x).to(self.device)).embedding

    @torch.no_grad()
    def reconstruct(self, x, concepts: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """Deterministic reconstruction; pass concepts to edit a real image."""
        imgs, _ = self._decode_from_z(self.encode(x), concepts)
        return imgs.clamp(0, 1).cpu()

    @torch.no_grad()
    def predict_concepts(self, x) -> Optional[torch.Tensor]:
        """Raw factor values the bottleneck reads from images, [B, C]."""
        if self.space is None:
            return None
        _, used = self._decode_from_z(self.encode(x))
        return used.cpu()

    @torch.no_grad()
    def traverse(self, concept: str, values=None, seed: int = 0,
                 fixed: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """One z, one concept swept over `values` -> images [len(values), 3, 64, 64]."""
        values = self.space.slider_values(concept) if values is None else values
        z = self.prior(1, seed).repeat(len(values), 1)
        out = []
        for i, v in enumerate(values):
            out.append(self.generate(z=z[i:i + 1], concepts={**(fixed or {}), concept: v}))
        return torch.cat(out)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _losses(gen: Generator, x, c_raw, kl_w, intervene: bool):
    m, space = gen.model, gen.space
    x01 = x.float().div(255)
    if gen.name == "vae":
        recon, mu, lv = m(x01)
        closs = torch.zeros((), device=x01.device)
    else:
        target = space.encode(c_raw)
        recon, pred, mu, lv = m(x01, interventions=target if intervene else None)
        if space.is_discrete:
            closs = m.concept_loss(pred, target)
        elif gen.name == "cbm" and intervene:
            closs = torch.zeros((), device=x01.device)       # CBM returns the interventions
        else:
            closs = F.mse_loss(pred, target)
    recon_loss = F.binary_cross_entropy(recon, x01, reduction="sum") / x.size(0)
    kl = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).sum(1).mean()
    tc = gen.train_cfg
    loss = recon_loss + kl_w * kl + tc.concept_weight * closs
    return loss, recon_loss, kl, closs, mu


def _concepts_of(gen: Generator, z):
    """Bottleneck's concept prediction for a z (float [-1,1] or list of logits)."""
    return gen.model.cbm(z) if gen.name == "cbm" else gen.model.cem(z)[1]


def _cycle_step(gen: Generator, mu, c_raw, opt):
    """
    Cycle loss, generalising scripts/train_cem_vae.py to every bottleneck:
    decode z with SHUFFLED target concepts, re-encode the result, and ask the
    bottleneck to read those targets back. Without it a CEM decoder can ignore
    the gate (both anchors are functions of z) and interventions do nothing.
    Encoder grads are zeroed so only the bottleneck + decoder learn from it.
    """
    m, space = gen.model, gen.space
    target = space.encode(c_raw[torch.randperm(c_raw.size(0), device=c_raw.device)])
    z = mu.detach()
    if gen.name == "cbm":
        recon = m.decode(target.float())
    else:
        emb, _ = m.cem(z, target)
        recon = m.decoder(emb).reconstruction
    pred = _concepts_of(gen, m.encoder(recon).embedding)
    loss = m.concept_loss(pred, target) if space.is_discrete else F.mse_loss(pred, target)
    opt.zero_grad()
    (gen.train_cfg.cycle_weight * loss).backward()
    for p in m.encoder.parameters():
        p.grad = None
    opt.step()
    return loss.item()


@torch.no_grad()
def evaluate(gen: Generator, loader, max_batches: int = 20) -> dict:
    """Test recon (deterministic z), concept MSE in [-1,1] units, concept accuracy."""
    gen.model.eval()
    native = ConceptSpace(gen.space.names, "categorical") if gen.space else None
    unit = ConceptSpace(gen.space.names, "continuous") if gen.space else None
    rec = cm = acc = 0.0
    nb = 0
    for i, (x, c_raw, _) in enumerate(loader):
        if i >= max_batches:
            break
        x01 = x.to(gen.device).float().div(255)
        c_raw = c_raw.to(gen.device)
        z = gen.model.encoder(x01).embedding
        recon, used = gen._decode_from_z(z)
        rec += F.binary_cross_entropy(recon.clamp(1e-6, 1 - 1e-6), x01, reduction="sum").item() / x.size(0)
        if used is not None:
            cm  += F.mse_loss(unit.encode(used), unit.encode(c_raw)).item()
            acc += (native.encode(used) == native.encode(c_raw)).float().mean().item()
        nb += 1
    out = {"test_recon": rec / nb}
    if gen.space:
        out.update(test_concept_mse=cm / nb, test_concept_acc=acc / nb)
    return out


def _save_grid(rows, path, n=8):
    grid = vutils.make_grid(torch.cat([r[:n].clamp(0, 1).cpu() for r in rows]), nrow=n, padding=2)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    to_pil_image(grid).save(path)


def default_out_dir(name: str, tag: Optional[str] = None) -> Path:
    return ROOT / "outputs" / "models" / (f"shapes3d_{name}" + (f"_{tag}" if tag else ""))


def train(
    model: str = "cem",
    *,
    epochs: Optional[int] = None,
    concepts=FACTOR_NAMES,
    n_bins: int = 5,
    n_samples: Optional[int] = 100_000,
    tag: Optional[str] = None,
    out_dir: Optional[str] = None,
    device: Optional[str] = None,
    loaders=None,
    backbone_cfg: Optional[ConvVAEConfig] = None,
    bottleneck_cfg=None,
    train_cfg: Optional[TrainConfig] = None,
    backbone_overrides: Optional[dict] = None,
    bottleneck_overrides: Optional[dict] = None,
    resume: bool = False,
    **train_overrides,
) -> Generator:
    """
    Train one model on Shapes3D, save it, and return a Generator.

    train_overrides set TrainConfig fields directly, e.g. lr=5e-4,
    concept_weight=50, intervention_prob=0.25, batch_size=128.
    backbone_overrides / bottleneck_overrides set ConvVAEConfig / bottleneck
    config fields, e.g. {"latent_dim": 64} / {"embedding_dim": 32}; keys the
    bottleneck does not have (embedding_dim on a CBM) are skipped.
    Pass `loaders=(train_loader, test_loader)` to reuse already-loaded data.

    The model is saved after EVERY epoch. With resume=True an existing
    checkpoint in the output folder is continued from its last epoch (its saved
    configs win over the arguments); a finished one is just loaded.
    """
    name = resolve(model)
    device = get_device(device)
    mode = MODELS[name][2]
    space = ConceptSpace(list(concepts), mode, n_bins) if mode else None

    d_back, d_bott, d_train = default_configs(name, space)
    backbone_cfg   = backbone_cfg or d_back
    bottleneck_cfg = bottleneck_cfg if bottleneck_cfg is not None else d_bott
    train_cfg      = train_cfg or d_train
    if epochs is not None:
        train_cfg.epochs = epochs
    for k, v in train_overrides.items():
        if not hasattr(train_cfg, k):
            raise TypeError(f"unknown training option {k!r}")
        setattr(train_cfg, k, v)
    for k, v in (backbone_overrides or {}).items():
        if not hasattr(backbone_cfg, k):
            raise TypeError(f"unknown backbone option {k!r}")
        setattr(backbone_cfg, k, v)
    for k, v in (bottleneck_overrides or {}).items():
        if bottleneck_cfg is not None and hasattr(bottleneck_cfg, k):
            setattr(bottleneck_cfg, k, v)

    out = Path(out_dir) if out_dir else default_out_dir(name, tag)
    ckpt, start_epoch, history = None, 1, []
    if resume and (out / "model.pt").exists():
        ckpt = torch.load(out / "model.pt", map_location="cpu", weights_only=False)
        saved = (ckpt.get("concept_space") or {}).get("names")
        if space is not None and saved is not None and list(saved) != list(space.names):
            raise SystemExit(
                f"ERROR: {out} was trained with concepts {saved}, but you asked for "
                f"{list(space.names)}. Use a different --tag/--suffix (or --out-dir) "
                f"for the new concept set, or drop --resume to overwrite it.")
        _, bcls, _ = MODELS[name]
        backbone_cfg   = ConvVAEConfig(**ckpt["backbone_cfg"])
        bottleneck_cfg = bcls(**ckpt["bottleneck_cfg"]) if bcls else None
        saved_epochs   = TrainConfig(**ckpt["train_cfg"]).epochs
        train_cfg      = TrainConfig(**{**ckpt["train_cfg"], "epochs": max(train_cfg.epochs, saved_epochs)})
        space = ConceptSpace.from_dict(ckpt["concept_space"]) if ckpt["concept_space"] else None
        n_samples = ckpt.get("data_cfg", {}).get("n_samples", n_samples)
        start_epoch = ckpt.get("epoch", train_cfg.epochs) + 1
        history = ckpt.get("history", [])
        if start_epoch > train_cfg.epochs:
            print(f"{out} already trained for {start_epoch - 1} epochs -- loading it.")
            return load(out, device)
        print(f"Resuming {name} from epoch {start_epoch}/{train_cfg.epochs}")

    set_seed(train_cfg.seed)
    if loaders is None:
        loaders = get_shapes3d(batch_size=train_cfg.batch_size, n_samples=n_samples,
                               seed=train_cfg.seed, concepts=list(concepts),
                               num_workers=train_cfg.num_workers)
    train_loader, test_loader = loaders

    net = build_model(name, backbone_cfg, bottleneck_cfg).to(device)
    gen = Generator(name, net, space, backbone_cfg, bottleneck_cfg, train_cfg, device,
                    data_cfg={"n_samples": n_samples, "seed": train_cfg.seed})
    opt = torch.optim.Adam(net.parameters(), lr=train_cfg.lr)
    if ckpt is not None:
        net.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            opt.load_state_dict(ckpt["optimizer_state_dict"])
    n_params = sum(p.numel() for p in net.parameters())
    fixed_x = next(iter(test_loader))[0][:8].to(device)

    print(f"Training {name} on {device} -> {out}  ({n_params / 1e6:.2f}M params)")
    if space:
        print(f"  concepts {space.names} | mode {space.mode}"
              + (f" | states {space.n_states}" if space.is_discrete else ""))
    kl_warmup = max(1, min(5, train_cfg.epochs // 2))
    for epoch in range(start_epoch, train_cfg.epochs + 1):
        net.train()
        kl_w = min(epoch / kl_warmup, 1.0) * backbone_cfg.kl_weight
        sums, n = [0.0, 0.0, 0.0, 0.0], 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{train_cfg.epochs}", leave=False)
        for x, c_raw, _ in pbar:
            x, c_raw = x.to(device), c_raw.to(device)
            intervene = space is not None and random.random() < train_cfg.intervention_prob
            loss, r, k, c, mu = _losses(gen, x, c_raw, kl_w, intervene)
            opt.zero_grad(); loss.backward(); opt.step()
            cyc = _cycle_step(gen, mu, c_raw, opt) if space and train_cfg.cycle_weight > 0 else 0.0
            sums[0] += r.item(); sums[1] += k.item(); sums[2] += c.item(); sums[3] += cyc; n += 1
            pbar.set_postfix(recon=f"{r.item():.0f}", kl=f"{k.item():.1f}", c=f"{c.item():.3f}", cyc=f"{cyc:.3f}")

        row = {"epoch": epoch, "recon": sums[0] / n, "kl": sums[1] / n,
               "concept_loss": sums[2] / n, "cycle_loss": sums[3] / n}
        row.update(evaluate(gen, test_loader))
        history.append(row)
        print("  " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                for k, v in row.items()))

        recon_fix = gen.reconstruct(fixed_x)
        prior = gen.generate(8, seed=epoch)
        _save_grid([fixed_x.float().div(255), recon_fix, prior], str(out / "samples" / f"epoch_{epoch:03d}.png"))
        save(gen, out, history, optimizer=opt, epoch=epoch, quiet=True)   # survives a killed job

    save(gen, out, history, optimizer=opt, epoch=train_cfg.epochs)
    return gen


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save(gen: Generator, out_dir, history=None, optimizer=None, epoch=None, quiet=False):
    """
    Write model.pt (+ readable JSON configs and metrics.csv) to out_dir.
    The curves are also copied to outputs/logs/<run>.csv, which git tracks,
    so they travel with a plain git pull even though the weights do not.
    """
    import json
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "model.pt.tmp"            # write-then-rename: never a half-written model.pt
    torch.save({
        "epoch":            epoch,
        "history":          history or [],
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "model_name":       gen.name,
        "model_state_dict": gen.model.state_dict(),
        "backbone_cfg":     gen.backbone_cfg.to_dict(),
        "bottleneck_cfg":   gen.bottleneck_cfg.to_dict() if gen.bottleneck_cfg else None,
        "train_cfg":        gen.train_cfg.to_dict(),
        "concept_space":    gen.space.to_dict() if gen.space else None,
        "data_cfg":         gen.data_cfg,
    }, tmp)
    os.replace(tmp, out / "model.pt")
    for fname, cfg in [("backbone_config.json", gen.backbone_cfg),
                       ("bottleneck_config.json", gen.bottleneck_cfg),
                       ("train_config.json", gen.train_cfg)]:
        if cfg is not None:
            (out / fname).write_text(json.dumps(cfg.to_dict(), indent=2))
    if history:
        logs = ROOT / "outputs" / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        for path in (out / "metrics.csv", logs / f"{out.name}.csv"):
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(history[0]))
                w.writeheader(); w.writerows(history)
    if not quiet:
        print(f"Saved {gen.name} -> {out}")


def load(checkpoint_dir, device: Optional[str] = None) -> Generator:
    device = get_device(device)
    ckpt = torch.load(Path(checkpoint_dir) / "model.pt", map_location=device, weights_only=False)
    name = ckpt["model_name"]
    _, bcls, _ = MODELS[name]
    backbone_cfg   = ConvVAEConfig(**ckpt["backbone_cfg"])
    bottleneck_cfg = bcls(**ckpt["bottleneck_cfg"]) if bcls else None
    train_cfg      = TrainConfig(**ckpt["train_cfg"])
    space = ConceptSpace.from_dict(ckpt["concept_space"]) if ckpt["concept_space"] else None
    net = build_model(name, backbone_cfg, bottleneck_cfg).to(device)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    return Generator(name, net, space, backbone_cfg, bottleneck_cfg, train_cfg, device,
                     ckpt.get("data_cfg"))


def list_saved(root=None):
    """Every saved Shapes3D model directory under outputs/models/."""
    root = Path(root) if root else ROOT / "outputs" / "models"
    return sorted(p for p in root.glob("shapes3d_*") if (p / "model.pt").exists())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="cem", choices=MODEL_CHOICES)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--n-samples", type=int, default=100_000, help="subset size; 0 = all 480k")
    ap.add_argument("--n-bins", type=int, default=5, help="binned (case3) only")
    ap.add_argument("--concepts", nargs="+", default=FACTOR_NAMES, choices=FACTOR_NAMES)
    ap.add_argument("--concept-weight", type=float, default=None)
    ap.add_argument("--intervention-prob", type=float, default=None)
    ap.add_argument("--cycle-weight", type=float, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    # backbone (ConvVAEConfig)
    ap.add_argument("--latent-dim", type=int, default=None)
    ap.add_argument("--enc-channels", type=int, nargs="+", default=None, help='e.g. 32 64 128 256')
    ap.add_argument("--dec-channels", type=int, nargs="+", default=None,
                    help="same length as --enc-channels; default mirrors it")
    ap.add_argument("--kl-weight", type=float, default=None)
    # bottleneck (CEMConfig / CEMCategoricalConfig; ignored by vae and cbm)
    ap.add_argument("--embedding-dim", type=int, default=None)
    ap.add_argument("--hidden-dim", type=int, default=None)
    ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    # output
    ap.add_argument("--tag", default=None, help="folder suffix: shapes3d_<model>_<tag>")
    ap.add_argument("--out-dir", default=None, help="explicit output folder (overrides --tag)")
    ap.add_argument("--resume", action="store_true",
                    help="continue an existing checkpoint in the output folder; skip it if finished")
    ap.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"])
    a = ap.parse_args()

    def pick(**kw):
        return {k: v for k, v in kw.items() if v is not None}

    dec = a.dec_channels
    if a.enc_channels and not dec:
        dec = list(reversed(a.enc_channels[:-1])) + [a.enc_channels[0]]
    if a.enc_channels and len(dec) != len(a.enc_channels):
        ap.error("--dec-channels must have as many entries as --enc-channels")

    train(a.model, epochs=a.epochs, concepts=a.concepts, n_bins=a.n_bins,
          n_samples=a.n_samples or None, tag=a.tag, out_dir=a.out_dir, device=a.device,
          resume=a.resume,
          backbone_overrides=pick(latent_dim=a.latent_dim, enc_channels=a.enc_channels,
                                  dec_channels=dec, kl_weight=a.kl_weight),
          bottleneck_overrides=pick(embedding_dim=a.embedding_dim, hidden_dim=a.hidden_dim,
                                    depth=a.depth, dropout=a.dropout),
          **pick(concept_weight=a.concept_weight, intervention_prob=a.intervention_prob,
                 lr=a.lr, cycle_weight=a.cycle_weight, batch_size=a.batch_size,
                 seed=a.seed, num_workers=a.num_workers))


if __name__ == "__main__":
    main()
