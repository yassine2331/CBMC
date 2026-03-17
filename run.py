"""
run.py — Universal experiment entry-point
==========================================
Run any experiment without modifying this file:

    python run.py                                    # uses default config
    python run.py --experiment exp_01_baseline_vae   # loads experiments/exp_01_.../config.py
    python run.py --model cem_v1                     # override model at the CLI
    python run.py --experiment exp_01_baseline_vae --model cem_v2

Available models (registered in models/registry.py):
    dynamic_vae  — Concept-conditioned image VAE (for image datasets)
    cem_v1       — CEM with per-concept independent networks
    cem_v2       — CEM with a shared backbone (more param-efficient)

To add a new experiment:
    1. cp -r experiments/exp_01_baseline_vae experiments/exp_03_my_idea
    2. Edit  experiments/exp_03_my_idea/config.py  (change only what differs)
    3. python run.py --experiment exp_03_my_idea

To add a new model architecture:
    1. Create models/CBMs/my_arch.py  (implement from_config classmethod)
    2. register_model("my_arch", MyArch) in models/registry.py
    3. python run.py --model my_arch
"""

import argparse
import importlib
import sys
import torch

from configs.VAE_config import TrainingConfig
from models.registry import build_model, save_model, list_models


def load_config(experiment_name: str | None) -> TrainingConfig:
    """Return the TrainingConfig for *experiment_name*.

    If *experiment_name* is None, the default TrainingConfig is returned.
    Otherwise, the module ``experiments/<experiment_name>/config.py`` is
    imported and its top-level ``config`` object is returned.
    """
    if experiment_name is None:
        return TrainingConfig()

    module_path = f"experiments.{experiment_name}.config"
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        print(
            f"[error] Could not find config for experiment '{experiment_name}'.\n"
            f"        Expected: experiments/{experiment_name}/config.py\n"
            f"        Detail: {exc}"
        )
        sys.exit(1)

    if not hasattr(module, "config"):
        print(
            f"[error] experiments/{experiment_name}/config.py must define a "
            "top-level variable called 'config'."
        )
        sys.exit(1)

    return module.config


def _is_cem_model(model_name):
    """Return True if *model_name* refers to a CEM-family model."""
    return model_name.startswith("cem")


def _smoke_test_vae(model, config):
    """Run a single forward pass with dummy image data."""
    dummy_image = torch.randn(1, config.in_channels, config.image_size, config.image_size)
    dummy_context = torch.randn(1, 1, config.context_dim)
    output = model(dummy_image, encoder_hidden_states=dummy_context)
    print("\n--- Forward pass OK (VAE) ---")
    print(f"  Input shape  : {dummy_image.shape}")
    print(f"  Latent shape : {output['latent_dist'].mean.shape}")
    print(f"  Output shape : {output['sample'].shape}")


def _smoke_test_cem(model, config):
    """Run a single forward pass with dummy tabular data."""
    dummy_x = torch.randn(4, config.input_dim)
    embeddings, concepts = model(dummy_x)
    print("\n--- Forward pass OK (CEM) ---")
    print(f"  Input shape      : {dummy_x.shape}")
    print(f"  Embeddings shape : {embeddings.shape}")
    print(f"  Concepts shape   : {concepts.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a CBMC experiment.\n"
            f"Registered models: {list_models()}"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help=(
            "Name of the experiment subdirectory inside experiments/ "
            "(e.g. exp_01_baseline_vae).  Omit to use the default config."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Override config.model_name at the CLI.  "
            f"Registered choices: {list_models()}"
        ),
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load config (from experiment directory or defaults)
    # ------------------------------------------------------------------
    config = load_config(args.experiment)

    # CLI --model overrides whatever the experiment config says
    if args.model is not None:
        if args.model not in list_models():
            print(
                f"[error] Unknown model '{args.model}'.  "
                f"Registered models: {list_models()}"
            )
            sys.exit(1)
        config.model_name = args.model

    print(f"[run] experiment : {config.experiment_name}")
    print(f"[run] model      : {config.model_name}")

    # ------------------------------------------------------------------
    # 2. Build model via registry (no model-specific import needed here)
    # ------------------------------------------------------------------
    model = build_model(config)

    # ------------------------------------------------------------------
    # 3. Smoke-test forward pass with dummy data
    # ------------------------------------------------------------------
    if _is_cem_model(config.model_name):
        _smoke_test_cem(model, config)
    else:
        print(f"[run] Building model with down blocks: {config.down_block_types}")
        _smoke_test_vae(model, config)

    # ------------------------------------------------------------------
    # 4. Save model + config
    # ------------------------------------------------------------------
    save_model(model, config)
    print(
        f"[run] Add training results to "
        f"experiments/{config.experiment_name}/notes.md"
    )


if __name__ == "__main__":
    main()
