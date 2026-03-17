"""
run.py — Universal experiment entry-point
==========================================
Run any experiment without modifying this file:

    python run.py                                    # uses default config
    python run.py --experiment exp_01_baseline_vae   # loads experiments/exp_01_.../config.py
    python run.py --experiment exp_02_deep_encoder

To add a new experiment:
    1. cp -r experiments/exp_01_baseline_vae experiments/exp_03_my_idea
    2. Edit  experiments/exp_03_my_idea/config.py
    3. python run.py --experiment exp_03_my_idea
"""

import argparse
import importlib
import sys
import torch

from configs.VAE_config import TrainingConfig
from models.registry import build_model  # noqa: F401 — triggers auto-registration


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a CBMC experiment.  Pass --experiment <name> to select "
        "a config from the experiments/ directory."
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Name of the experiment subdirectory inside experiments/ "
        "(e.g. exp_01_baseline_vae).  Omit to use the default config.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load config
    # ------------------------------------------------------------------
    config = load_config(args.experiment)
    print(f"[run] experiment : {config.experiment_name}")
    print(f"[run] model      : {config.model_name}")

    # ------------------------------------------------------------------
    # 2. Build model via registry (no model-specific import needed here)
    # ------------------------------------------------------------------
    print(f"[run] Building model with down blocks: {config.down_block_types}")
    model = build_model(config)

    # ------------------------------------------------------------------
    # 3. Smoke-test forward pass with dummy data
    # ------------------------------------------------------------------
    dummy_image = torch.randn(1, config.in_channels, config.image_size, config.image_size)
    dummy_context = torch.randn(1, 1, config.context_dim)

    output = model(dummy_image, encoder_hidden_states=dummy_context)

    print("\n--- Forward pass OK ---")
    print(f"  Input shape  : {dummy_image.shape}")
    print(f"  Latent shape : {output['latent_dist'].mean.shape}")
    print(f"  Output shape : {output['sample'].shape}")

    # ------------------------------------------------------------------
    # 4. Save model + config
    # ------------------------------------------------------------------
    model.save_pretrained(config.output_dir)
    print(f"\n[run] Model saved to '{config.output_dir}'")
    print(
        f"[run] Add training results to "
        f"experiments/{config.experiment_name}/notes.md"
    )


if __name__ == "__main__":
    main()
