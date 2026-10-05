"""
Concept representations — how a raw concept value becomes a training target.

The same nodule concepts can be handed to a bottleneck in several forms, and
which form you choose IS the experiment. All transforms are fitted on the
training split only and expose the same interface:

    forward(raw)   -> model targets
    inverse(pred)  -> back to original units, for error in mm / mm3 / ratings
    n_states       -> per-concept state counts (categorical only, else None)
    kind           -> "continuous" | "binary" | "categorical"

    Continuous(scaling="minmax")   value scaled to [-1, 1]            Case 3.5
    Continuous(scaling="raw")      value passed through untouched
    Binary                         high/low, split at the train mean  Case 1
    Categorical(n_bins=k)          bin index; ratings keep their own
                                   integer levels, continuous concepts
                                   get k uniform bins                 Case 2/3
"""

from __future__ import annotations

import numpy as np

# Concepts that are already discrete levels on a 1-5 scale. Averaging over
# radiologists makes them fractional, but the underlying scale is integer, so
# in categorical mode they keep their own levels rather than being re-binned.
RATING_CONCEPTS = {"subtlety", "internalStructure", "calcification", "sphericity",
                   "margin", "lobulation", "spiculation", "texture", "malignancy"}
RATING_LEVELS = 5


class Continuous:
    """Real-valued concepts, optionally min-max scaled to [-1, 1]."""

    kind = "continuous"
    n_states = None

    def __init__(self, frame, concepts, scaling="minmax"):
        if scaling not in ("minmax", "raw"):
            raise ValueError(f"scaling must be 'minmax' or 'raw', got {scaling!r}")
        self.concepts, self.scaling = concepts, scaling
        self.lo = frame[concepts].min().values.astype("float32")
        self.hi = frame[concepts].max().values.astype("float32")
        self.span = np.maximum(self.hi - self.lo, 1e-8)

    def forward(self, raw):
        raw = np.asarray(raw, dtype="float32")
        if self.scaling == "raw":
            return raw
        return 2.0 * (raw - self.lo) / self.span - 1.0

    def inverse(self, z):
        z = np.asarray(z, dtype="float32")
        if self.scaling == "raw":
            return z
        return (z + 1.0) / 2.0 * self.span + self.lo


class Binary:
    """
    Every concept collapsed to high/low, split at the TRAIN mean.

    Targets are -1 and +1 so the CEM gate w = clamp(0.5c + 0.5, 0, 1) lands
    exactly on an anchor, which is the binary Case 1 behaviour.

    inverse() can only return the class means — the actual value is gone. That
    is the point of the comparison, and it is why concept MAE for this variant
    is reported against the binarised target, not the original value.
    """

    kind = "binary"
    n_states = None

    def __init__(self, frame, concepts):
        self.concepts = concepts
        self.threshold = frame[concepts].mean().values.astype("float32")
        vals = frame[concepts].values.astype("float32")
        hi_mask = vals > self.threshold
        # class means, used by inverse() as the best possible reconstruction
        self.hi_mean = np.array([vals[hi_mask[:, i], i].mean() if hi_mask[:, i].any()
                                 else self.threshold[i]
                                 for i in range(len(concepts))], dtype="float32")
        self.lo_mean = np.array([vals[~hi_mask[:, i], i].mean() if (~hi_mask[:, i]).any()
                                 else self.threshold[i]
                                 for i in range(len(concepts))], dtype="float32")

    def forward(self, raw):
        raw = np.asarray(raw, dtype="float32")
        return np.where(raw > self.threshold, 1.0, -1.0).astype("float32")

    def inverse(self, z):
        z = np.asarray(z, dtype="float32")
        return np.where(z > 0, self.hi_mean, self.lo_mean).astype("float32")


class Categorical:
    """
    Concepts as discrete states.

    A rating concept keeps its own 5 integer levels (round, clamp to 1..5).
    A genuinely continuous concept is split into `n_bins` uniform bins over its
    training range. So the state counts differ per concept, which is what
    CEMCategoricalPerConcept is for.

    inverse() decodes a state back to its bin centre (or rating level), so
    concept error is still reportable in original units. The gap between that
    reconstruction and the true value is the quantisation cost.
    """

    kind = "categorical"

    def __init__(self, frame, concepts, n_bins=5):
        self.concepts, self.n_bins = concepts, n_bins
        self.lo = frame[concepts].min().values.astype("float32")
        self.hi = frame[concepts].max().values.astype("float32")
        self.span = np.maximum(self.hi - self.lo, 1e-8)
        self.is_rating = np.array([c in RATING_CONCEPTS for c in concepts])
        self.n_states = [RATING_LEVELS if r else n_bins for r in self.is_rating]

        centres = []
        for i, c in enumerate(concepts):
            if self.is_rating[i]:
                centres.append(np.arange(1, RATING_LEVELS + 1, dtype="float32"))
            else:
                w = self.span[i] / n_bins
                centres.append(self.lo[i] + (np.arange(n_bins) + 0.5) * w)
        self.centres = centres

    def forward(self, raw):
        raw = np.asarray(raw, dtype="float32")
        out = np.empty(raw.shape, dtype="int64")
        for i in range(len(self.concepts)):
            if self.is_rating[i]:
                out[..., i] = np.clip(np.round(raw[..., i]), 1, RATING_LEVELS) - 1
            else:
                w = self.span[i] / self.n_bins
                out[..., i] = np.clip(np.floor((raw[..., i] - self.lo[i]) / w),
                                      0, self.n_bins - 1)
        return out

    def inverse(self, states):
        states = np.asarray(states)
        out = np.empty(states.shape, dtype="float32")
        for i in range(len(self.concepts)):
            out[..., i] = self.centres[i][np.clip(states[..., i], 0,
                                                  len(self.centres[i]) - 1)]
        return out

    def quantisation_mae(self, raw):
        """MAE floor from binning alone — the best a perfect model could do."""
        raw = np.asarray(raw, dtype="float32")
        return np.abs(self.inverse(self.forward(raw)) - raw).mean(0)


def build(mode, frame, concepts, scaling="minmax", n_bins=5):
    """Factory used by the training script."""
    if mode == "continuous":  return Continuous(frame, concepts, scaling)
    if mode == "binary":      return Binary(frame, concepts)
    if mode == "categorical": return Categorical(frame, concepts, n_bins)
    raise ValueError(f"unknown concept mode {mode!r}")
