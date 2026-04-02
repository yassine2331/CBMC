"""
Model Registry
==============
Register model classes by a short string name so experiment configs can
select architectures without importing them directly in the runner.

Usage (registering a new model)
--------------------------------
from models.registry import register_model
from models.CBM_VAE import DynamicVAE

register_model("dynamic_vae", DynamicVAE)

Usage (building a model from a config)
---------------------------------------
from models.registry import build_model

model = build_model(config)   # config.model_name must match a registered name
"""

from __future__ import annotations
from typing import Type

_REGISTRY: dict[str, Type] = {}


def register_model(name: str, cls: Type) -> Type:
    """Register *cls* under *name*.  Returns *cls* so it can be used as a decorator."""
    if name in _REGISTRY:
        raise KeyError(f"A model named '{name}' is already registered.")
    _REGISTRY[name] = cls
    return cls


def build_model(config) -> object:
    """Instantiate the model whose name is stored in *config.model_name*.

    The config object is expected to have at least the following attributes:
        model_name, in_channels, out_channels, block_out_channels,
        down_block_types, up_block_types, layers_per_block,
        latent_channels, context_dim, image_size
    """
    name = getattr(config, "model_name", None)
    if name is None:
        raise AttributeError("config.model_name is not set.")
    if name not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise KeyError(
            f"Model '{name}' is not registered.  Available models: {available}"
        )
    cls = _REGISTRY[name]
    return cls(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        block_out_channels=config.block_out_channels,
        down_block_types=config.down_block_types,
        up_block_types=config.up_block_types,
        layers_per_block=config.layers_per_block,
        latent_channels=config.latent_channels,
        context_dim=config.context_dim,
        sample_size=config.image_size,
    )


# ---------------------------------------------------------------------------
# Auto-register the built-in models when this module is first imported.
# ---------------------------------------------------------------------------
def _auto_register() -> None:
    from models.CBM_VAE import DynamicVAE  # noqa: F401 (side-effect import)
    register_model("dynamic_vae", DynamicVAE)


_auto_register()
