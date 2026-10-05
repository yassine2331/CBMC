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
    batch_size:        int   = 256
    seed:              int   = 42
    num_workers:       int   = 2
    # Concept supervision: weight of concept MSE loss added to task loss.
    # Set to 0.0 to disable (baseline/no-concept experiments ignore this).
    concept_weight:    float = 0.0
    # Intervention probability: fraction of batches where true concepts are
    # injected instead of predicted ones. 0.0 = never, 1.0 = always.
    intervention_prob: float = 0.0
    # Cycle loss weight: re-encode decoder output and penalise concept mismatch.
    # Trains the decoder to actually change its output when concepts change.
    # Encoder grads are zeroed after this backward so only decoder+CEM update.
    cycle_weight:      float = 0.0


@dataclass
class CEMCategoricalConfig(BaseConfig):
    """
    Categorical (Case 2) concept bottleneck.

    Set `n_states` to give each concept its own number of states; it wins over
    `n_concepts`/`n_classes` when non-empty. Leave it empty to use `n_classes`
    states for all `n_concepts` concepts.
    """
    n_concepts:    int       = 8
    n_classes:     int       = 2          # used only when n_states is empty
    n_states:      List[int] = field(default_factory=list)
    embedding_dim: int       = 16
    hidden_dim:    int       = 64
    depth:         int       = 2
    dropout:       float     = 0.2
    normalize:     bool      = True


@dataclass
class Conv3DConfig(BaseConfig):
    """3D CNN encoder for volumetric input (LIDC nodule cubes)."""
    in_channels:   int       = 1
    conv_channels: List[int] = field(default_factory=lambda: [16, 32, 64, 128])
    kernel_size:   int       = 3
    batch_norm:    bool      = True
    dropout:       float     = 0.0          # Dropout3d after each block
    cube_size:     int       = 64           # input is cube_size^3


@dataclass
class LIDCDataConfig(BaseConfig):
    """Which nodules to use and how to normalise them."""
    data_dir:        str       = "data/processed/lidc"
    min_annotations: int       = 3          # drop nodules few radiologists saw
    drop_ambiguous:  bool      = True       # drop label == -1 (malignancy == 3)
    hu_low:          int       = -1000      # lung window, applied to every cube
    hu_high:         int       = 400
    test_size:       float     = 0.2        # split is BY PATIENT, never by nodule
    concepts:        List[str] = field(default_factory=lambda: [
        "subtlety", "sphericity", "margin", "lobulation", "spiculation",
        "texture", "diameter", "volume", "surface_area"])
    # calcification and internalStructure are deliberately absent: they are
    # nominal codes (1=popcorn, 2=laminated, ...), not magnitudes, so treating
    # them as continuous concepts is meaningless. Use the categorical block.
