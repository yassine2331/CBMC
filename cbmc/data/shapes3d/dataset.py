"""
Shapes3D loader (DeepMind 3dshapes, https://github.com/deepmind/3d-shapes).

The h5 file holds 480 000 RGB images (64x64) rendered from 6 ground-truth
factors, which are the concepts here:

    name          values                  #states
    floor_hue     0.0, 0.1, ..., 0.9      10
    wall_hue      0.0, 0.1, ..., 0.9      10
    object_hue    0.0, 0.1, ..., 0.9      10
    scale         0.75 ... 1.25           8
    shape         0, 1, 2, 3              4   (nominal, not a magnitude)
    orientation   -30 ... 30 (degrees)    15

Each sample is (image, concepts, target), the same triple as ArithmeticMNIST:
    image    : uint8 [3, 64, 64]   -- convert with x.float() / 255 on device
    concepts : float [C]           -- RAW factor values of the chosen concepts
    target   : long  scalar        -- the shape index (handy as a class label)

Images stay uint8 in memory (100k images = 1.2 GB, the full set = 5.9 GB) and
are converted to float on the GPU per batch, which is far faster than doing it
per item in a DataLoader worker.

How raw concept values become model targets depends on the bottleneck, and is
handled by `ConceptSpace`:

    continuous   Case 3.5 / CBM : each factor scaled to [-1, 1]
    categorical  Case 2         : each factor's native state index
    binned       Case 3         : each factor's range cut into `n_bins` bins
                                  (shape stays categorical: it has no order)
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# name -> (low, high, number of native states)
FACTORS = {
    "floor_hue":   (0.0, 0.9, 10),
    "wall_hue":    (0.0, 0.9, 10),
    "object_hue":  (0.0, 0.9, 10),
    "scale":       (0.75, 1.25, 8),
    "shape":       (0.0, 3.0, 4),
    "orientation": (-30.0, 30.0, 15),
}
FACTOR_NAMES = list(FACTORS)
NOMINAL = {"shape"}          # factors whose values have no meaningful order

DEFAULT_H5 = Path(__file__).resolve().parents[3] / "data" / "raw" / "3dshapes" / "3dshapes.h5"


# ---------------------------------------------------------------------------
# Concept encoding
# ---------------------------------------------------------------------------

@dataclass
class ConceptSpace:
    """
    Maps raw factor values <-> what the bottleneck is trained on.

    mode : "continuous" | "categorical" | "binned"
    """
    names:  List[str] = field(default_factory=lambda: list(FACTOR_NAMES))
    mode:   str       = "continuous"
    n_bins: int       = 5

    def __post_init__(self):
        unknown = set(self.names) - set(FACTORS)
        if unknown:
            raise ValueError(f"unknown concepts {sorted(unknown)}; choose from {FACTOR_NAMES}")
        if self.mode not in ("continuous", "categorical", "binned"):
            raise ValueError(f"mode must be continuous/categorical/binned, got {self.mode!r}")

    # -- per-concept info ---------------------------------------------------
    @property
    def n_concepts(self) -> int:
        return len(self.names)

    def _lo_hi(self, device=None):
        lo = torch.tensor([FACTORS[n][0] for n in self.names], device=device)
        hi = torch.tensor([FACTORS[n][1] for n in self.names], device=device)
        return lo, hi

    def _k(self, name: str) -> int:
        """Number of states the bottleneck uses for one concept."""
        native = FACTORS[name][2]
        if self.mode == "categorical" or name in NOMINAL:
            return native
        return self.n_bins

    @property
    def n_states(self) -> List[int]:
        """States per concept (categorical / binned only)."""
        return [self._k(n) for n in self.names]

    @property
    def is_discrete(self) -> bool:
        return self.mode in ("categorical", "binned")

    # -- raw -> target ------------------------------------------------------
    def encode(self, raw: torch.Tensor) -> torch.Tensor:
        """raw [B, C] factor values -> float [-1,1] (continuous) or long state idx."""
        lo, hi = self._lo_hi(raw.device)
        u = ((raw - lo) / (hi - lo)).clamp(0, 1)                 # [0, 1]
        if self.mode == "continuous":
            return u * 2 - 1
        cols = []
        for i, name in enumerate(self.names):
            k = self._k(name)
            if self.mode == "categorical" or name in NOMINAL:    # native index
                cols.append((u[:, i] * (k - 1)).round())
            else:                                                # uniform bins
                cols.append((u[:, i] * k).floor().clamp(max=k - 1))
        return torch.stack(cols, 1).long()

    # -- target / prediction -> raw ------------------------------------------
    def decode(self, pred) -> torch.Tensor:
        """
        Model output -> raw factor values [B, C].
        continuous: pred is a [B, C] tensor in [-1, 1]
        discrete  : pred is state indices [B, C], or the list of logits.
        """
        if self.is_discrete and isinstance(pred, (list, tuple)):
            pred = torch.stack([lg.argmax(1) for lg in pred], 1)
        lo, hi = self._lo_hi(pred.device)
        if self.mode == "continuous":
            u = (pred.float().clamp(-1, 1) + 1) / 2
        else:
            cols = []
            for i, name in enumerate(self.names):
                k = self._k(name)
                idx = pred[:, i].float()
                if self.mode == "categorical" or name in NOMINAL:
                    cols.append(idx / (k - 1))
                else:
                    cols.append((idx + 0.5) / k)                 # bin centre
            u = torch.stack(cols, 1)
        return lo + u * (hi - lo)

    def slider_values(self, name: str) -> List[float]:
        """Raw values a UI should offer for one concept."""
        lo, hi, native = FACTORS[name]
        if self.mode == "continuous" and name not in NOMINAL:
            return list(np.linspace(lo, hi, 4 * native - 3))      # in-between values too
        k = self._k(name)
        if self.mode == "binned" and name not in NOMINAL:
            return [lo + (i + 0.5) / k * (hi - lo) for i in range(k)]
        return list(np.linspace(lo, hi, native))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ConceptSpace":
        return cls(**d)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_shapes3d_arrays(
    h5_path: str | Path = DEFAULT_H5,
    n_samples: Optional[int] = 100_000,
    seed: int = 42,
    concepts: Sequence[str] = FACTOR_NAMES,
    indices: Optional[np.ndarray] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Read a random subset of the h5 file into memory (or exactly `indices`,
    positions in the h5 file, when given -- returned in sorted order).

    The file is ordered factor by factor, so a contiguous slice would cover
    only a few floor hues -- the subset is a random draw over the whole file,
    read chunk by chunk (~1 s per 20k images).

    Returns images uint8 [N,3,64,64], concepts float [N,C] (raw), shape long [N].
    """
    import h5py

    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(
            f"{h5_path} not found. Download 3dshapes.h5 from "
            "https://storage.googleapis.com/3d-shapes/3dshapes.h5 into data/raw/3dshapes/")

    col = [FACTOR_NAMES.index(n) for n in concepts]
    with h5py.File(h5_path, "r") as f:
        total = f["images"].shape[0]
        labels = f["labels"][:]                                   # [N, 6] float64, small
        if indices is not None:
            idx = np.unique(np.asarray(indices))
        elif n_samples is None or n_samples >= total:
            idx = np.arange(total)
        else:
            idx = np.sort(np.random.RandomState(seed).choice(total, n_samples, replace=False))

        imgs = np.empty((len(idx), 64, 64, 3), dtype=np.uint8)
        chunk, out = 20_000, 0
        for start in range(0, total, chunk):
            stop = min(start + chunk, total)
            sel = idx[(idx >= start) & (idx < stop)]
            if len(sel) == 0:
                continue
            block = f["images"][start:stop]
            imgs[out:out + len(sel)] = block[sel - start]
            out += len(sel)

    images = torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous()      # [N,3,64,64]
    lab = torch.from_numpy(labels[idx]).float()
    return images, lab[:, col], lab[:, FACTOR_NAMES.index("shape")].round().long()


def get_shapes3d(
    batch_size: int = 256,
    n_samples: Optional[int] = 100_000,
    test_size: float = 0.1,
    seed: int = 42,
    concepts: Sequence[str] = FACTOR_NAMES,
    h5_path: str | Path = DEFAULT_H5,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Train / test DataLoaders yielding (uint8 image, raw concepts, shape idx)."""
    images, c, y = load_shapes3d_arrays(h5_path, n_samples, seed, concepts)
    perm = torch.randperm(len(images), generator=torch.Generator().manual_seed(seed))
    n_test = int(len(images) * test_size)
    te, tr = perm[:n_test], perm[n_test:]
    train = TensorDataset(images[tr], c[tr], y[tr])
    test  = TensorDataset(images[te], c[te], y[te])
    print(f"Shapes3D: {len(train)} train / {len(test)} test, concepts={list(concepts)}")
    return (DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True),
            DataLoader(test,  batch_size=batch_size, shuffle=False, num_workers=num_workers))


def held_out_indices(n_samples: Optional[int] = 100_000, seed: int = 42,
                     test_size: float = 0.1, total: int = 480_000) -> np.ndarray:
    """
    h5 positions of the TEST split that get_shapes3d(n_samples, seed, test_size)
    produces -- i.e. images a model trained with those settings never saw.
    Reproduces the same subset draw and the same shuffle without loading images.
    """
    if n_samples is None or n_samples >= total:
        idx = np.arange(total)
    else:
        idx = np.sort(np.random.RandomState(seed).choice(total, n_samples, replace=False))
    perm = torch.randperm(len(idx), generator=torch.Generator().manual_seed(seed)).numpy()
    return idx[perm[:int(len(idx) * test_size)]]


def get_held_out(n_samples: Optional[int] = 100_000, seed: int = 42, max_images: int = 10_000,
                 concepts: Sequence[str] = FACTOR_NAMES, h5_path: str | Path = DEFAULT_H5,
                 test_size: float = 0.1):
    """(images uint8, raw concepts, shape idx) for up to max_images held-out test images."""
    te = held_out_indices(n_samples, seed, test_size)
    te = te[np.random.RandomState(0).permutation(len(te))[:max_images]]
    return load_shapes3d_arrays(h5_path, concepts=concepts, indices=te)
