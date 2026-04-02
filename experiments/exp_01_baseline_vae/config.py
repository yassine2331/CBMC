"""
Experiment 01 — Baseline DynamicVAE
=====================================
Minimal configuration: default block sizes, single attention layer in
encoder and decoder, small latent space.  This serves as the reference
point for all future experiments.
"""

from configs.VAE_config import TrainingConfig

config = TrainingConfig(
    experiment_name="exp_01_baseline_vae",
    model_name="dynamic_vae",

    # Architecture
    image_size=32,
    in_channels=1,
    out_channels=1,
    latent_channels=4,
    context_dim=8,
    layers_per_block=2,
    block_out_channels=(128, 128, 256, 256, 512),
    down_block_types=(
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "CrossAttnDownBlock2D",
        "DownEncoderBlock2D",
    ),
    up_block_types=(
        "UpDecoderBlock2D",
        "AttnUpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
    ),

    # Training
    num_epochs=5,
    train_batch_size=64,
    learning_rate=1e-4,
    output_dir="outputs/exp_01_baseline_vae",
)
