"""
Model Registry
==============
Register model classes by a short string name so experiment configs (or
the ``--model`` CLI flag) can select architectures without importing them
directly in the runner.

Registered models
-----------------
    "dynamic_vae"  — DynamicVAE  (concept-conditioned image VAE)
    "cem_v1"       — CEM         (per-concept independent networks)
    "cem_v2"       — CEM_v2      (shared-backbone variant, more param-efficient)

Usage (registering a new model)
--------------------------------
    from models.registry import register_model
    from models.CBMs.my_arch import MyArch

    register_model("my_arch", MyArch)

    # MyArch must implement a @classmethod from_config(cls, config) -> instance.

Usage (building a model from a config)
---------------------------------------
    from models.registry import build_model

    model = build_model(config)   # config.model_name must match a registered name

Usage (saving / loading checkpoints)
--------------------------------------
    from models.registry import save_model, load_model

    save_model(model, config)                  # saves to config.output_dir
    model = load_model(config)                 # rebuilds + loads weights
"""

from __future__ import annotations

import os
from typing import Type

import torch

_REGISTRY: dict[str, Type] = {}


# --------------------------------------------------------------------------- #
# Registration helpers                                                          #
# --------------------------------------------------------------------------- #

def register_model(name: str, cls: Type) -> Type:
    """Register *cls* under *name*.  Returns *cls* so it can be used as a decorator."""
    if name in _REGISTRY:
        raise KeyError(f"A model named '{name}' is already registered.")
    _REGISTRY[name] = cls
    return cls


def list_models() -> list[str]:
    """Return the names of all currently registered models."""
    return sorted(_REGISTRY.keys())


# --------------------------------------------------------------------------- #
# Build                                                                         #
# --------------------------------------------------------------------------- #

def build_model(config) -> object:
    """Instantiate the model whose name is stored in *config.model_name*.

    Each registered class must implement::

        @classmethod
        def from_config(cls, config): ...

    which receives the full config object and returns a ready-to-use model
    instance.
    """
    name = getattr(config, "model_name", None)
    if name is None:
        raise AttributeError("config.model_name is not set.")
    if name not in _REGISTRY:
        raise KeyError(
            f"Model '{name}' is not registered.  "
            f"Available models: {list_models()}"
        )
    cls = _REGISTRY[name]
    if not hasattr(cls, "from_config"):
        raise NotImplementedError(
            f"Model class '{cls.__name__}' must implement a "
            "'from_config(cls, config)' classmethod."
        )
    return cls.from_config(config)


# --------------------------------------------------------------------------- #
# Save / Load                                                                   #
# --------------------------------------------------------------------------- #

def _ckpt_path(output_dir: str, model_name: str) -> str:
    """Return the path used to store a plain-PyTorch checkpoint."""
    return os.path.join(output_dir, f"{model_name}_weights.pt")


def save_model(model, config) -> None:
    """Save *model* to *config.output_dir*.

    * diffusers ``ModelMixin`` subclasses (e.g. ``DynamicVAE``) are saved with
      ``save_pretrained`` so that the config JSON is stored alongside weights.
    * Plain ``torch.nn.Module`` subclasses (e.g. CEM variants) are saved as a
      ``state_dict`` ``.pt`` file.
    """
    os.makedirs(config.output_dir, exist_ok=True)

    # diffusers ModelMixin provides save_pretrained
    try:
        from diffusers import ModelMixin
        if isinstance(model, ModelMixin):
            model.save_pretrained(config.output_dir)
            print(f"[registry] Saved (diffusers) model to '{config.output_dir}'")
            return
    except ImportError:
        pass

    # Plain nn.Module — save state dict
    path = _ckpt_path(config.output_dir, config.model_name)
    torch.save(model.state_dict(), path)
    print(f"[registry] Saved model weights to '{path}'")


def load_model(config, map_location=None) -> object:
    """Rebuild the model from *config* and load its saved weights.

    The checkpoint must have been written by :func:`save_model` with the same
    *config*.
    """
    name = getattr(config, "model_name", None)
    if name is None:
        raise AttributeError("config.model_name is not set.")
    if name not in _REGISTRY:
        raise KeyError(
            f"Model '{name}' is not registered.  "
            f"Available models: {list_models()}"
        )
    cls = _REGISTRY[name]

    # diffusers ModelMixin — use from_pretrained
    try:
        from diffusers import ModelMixin
        if issubclass(cls, ModelMixin):
            model = cls.from_pretrained(config.output_dir)
            print(f"[registry] Loaded (diffusers) model from '{config.output_dir}'")
            return model
    except ImportError:
        pass

    # Plain nn.Module — rebuild architecture, then load state dict
    model = build_model(config)
    path = _ckpt_path(config.output_dir, config.model_name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No checkpoint found at '{path}'.  "
            "Did you save the model first with save_model()?"
        )
    state = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(state)
    print(f"[registry] Loaded model weights from '{path}'")
    return model


# --------------------------------------------------------------------------- #
# Auto-register built-in models when this module is first imported.            #
# --------------------------------------------------------------------------- #

def _auto_register() -> None:
    from models.CBM_VAE import DynamicVAE
    register_model("dynamic_vae", DynamicVAE)

    from models.CBMs.CEM import CEM
    register_model("cem_v1", CEM)

    from models.CBMs.CEM_v2 import CEM_v2
    register_model("cem_v2", CEM_v2)


_auto_register()
