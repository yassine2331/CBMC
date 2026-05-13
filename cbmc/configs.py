"""
Config dataclasses for all models and training runs.

Each config is a plain Python dataclass — fully serializable to/from JSON.
Use save() to persist a run's exact config next to its outputs.
Use load() to reconstruct it for reproducibility or hyperparameter search.

Example:
    cfg = VAEConfig(latent_dim=32, encoder_dims=[512, 256])
    cfg.save("outputs/my_run/config.json")

    # later
    cfg = VAEConfig.load("outputs/my_run/config.json")
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Type, TypeVar

T = TypeVar("T", bound="BaseConfig")


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

@dataclass
class BaseConfig:
    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Config saved to {path}")

    @classmethod
    def load(cls: Type[T], path: str) -> T:
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def __str__(self) -> str:
        lines = [f"{self.__class__.__name__}:"]
        for k, v in self.to_dict().items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------

@dataclass
class CNNConfig(BaseConfig):
    # Architecture
    in_channels:   int       = 1           # 1 for grayscale, 3 for RGB
    conv_channels: List[int] = field(default_factory=lambda: [32, 64, 128])
    fc_dims:       List[int] = field(default_factory=lambda: [256])
    n_classes:     int       = 10

    # Regularization
    dropout:       float     = 0.0         # 0.0 = disabled


@dataclass
class VAEConfig(BaseConfig):
    # Architecture
    in_channels:   int       = 1
    input_dim:     int       = 784         # 28*28 for MNIST
    encoder_dims:  List[int] = field(default_factory=lambda: [400, 200])
    latent_dim:    int       = 16
    decoder_dims:  List[int] = field(default_factory=lambda: [200, 400])

    # Loss
    kl_weight:     float     = 1.0         # beta in beta-VAE; 1.0 = standard VAE


@dataclass
class CBMConfig(BaseConfig):
    """Scalar concept bottleneck — one learned scalar per concept (linear probe)."""
    n_concepts:  int   = 8
    head_channels: List[int] = field(default_factory=lambda: [16,16])  # head MLP layers after bottleneck




@dataclass
class CEMConfig(BaseConfig):
    """Concept Embedding Model — pos/neg embedding pair per concept."""
    n_concepts:    int   = 8
    embedding_dim: int   = 16     # output_dim = n_concepts * embedding_dim
    hidden_dim:    int   = 64
    depth:         int   = 2
    dropout:       float = 0.2


@dataclass
class ConvVAEConfig(BaseConfig):
    """Config for conv-based VAE — for spatial images larger than MNIST."""
    # Encoder conv channels (each block: Conv -> ReLU -> MaxPool)
    in_channels:    int       = 4            # 4 for RGBA, 1 for grayscale, 3 for RGB
    enc_channels:   List[int] = field(default_factory=lambda: [32, 64, 128])
    latent_dim:     int       = 32

    # Decoder mirrors the encoder
    dec_channels:   List[int] = field(default_factory=lambda: [128, 64, 32])
    img_size:       int       = 64           # spatial size after resize (H = W)

    # Loss
    kl_weight:      float     = 1.0


@dataclass
class CNNRegressionConfig(BaseConfig):
    """CNN config for regression tasks (continuous targets instead of classes)."""
    in_channels:   int       = 4            # 4 for RGBA
    conv_channels: List[int] = field(default_factory=lambda: [32, 64, 128])
    fc_dims:       List[int] = field(default_factory=lambda: [256])
    n_outputs:     int       = 4            # number of regression targets
    dropout:       float     = 0.0


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig(BaseConfig):
    epochs:            int   = 20
    lr:                float = 1e-3
    batch_size:        int   = 128
    seed:              int   = 42
    num_workers:       int   = 2
    # Concept supervision: weight of concept MSE loss added to task loss.
    # Set to 0.0 to disable (baseline/no-concept experiments ignore this).
    concept_weight:    float = 0.0
    # Intervention probability: fraction of batches where true concepts are
    # injected instead of predicted ones. 0.0 = never, 1.0 = always.
    intervention_prob: float = 0.0
