"""
Shapes3D (3dshapes) dataset for concept-bottleneck generative experiments.

    from cbmc.data.shapes3d import get_shapes3d, ConceptSpace, FACTOR_NAMES

The original loaders from the CBMC-test repo live in `legacy/`.
"""

from .dataset import (
    FACTORS, FACTOR_NAMES, NOMINAL, DEFAULT_H5,
    ConceptSpace, load_shapes3d_arrays, get_shapes3d,
)

__all__ = ["FACTORS", "FACTOR_NAMES", "NOMINAL", "DEFAULT_H5",
           "ConceptSpace", "load_shapes3d_arrays", "get_shapes3d"]
