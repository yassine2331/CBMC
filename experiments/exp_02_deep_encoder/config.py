"""
Experiment 02 — Deeper encoder (more layers per block)
=======================================================
Same architecture as exp_01 but with layers_per_block=3 and a wider
latent space (latent_channels=8).  Goal: check if more capacity in the
encoder produces better reconstructions.
"""

from configs.VAE_config import TrainingConfig

config = TrainingConfig(
    experiment_name="exp_02_deep_encoder",
    model_name="dynamic_vae",

    # Architecture — increased depth and wider latent
    image_size=32,
    in_channels=1,
    out_channels=1,
    latent_channels=8,       # wider than baseline
    context_dim=8,
    layers_per_block=3,      # deeper than baseline
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
    output_dir="outputs/exp_02_deep_encoder",
)
