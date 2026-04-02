from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class TrainingConfig:
    # Experiment identity — used by run.py to load and log experiments
    experiment_name: str = "default"
    model_name: str = "dynamic_vae"  # must match a key in models/registry.py

    # Training Loop params
    root_dir: str = ".."
    train_batch_size: int = 64
    eval_batch_size: int = 16
    num_epochs: int = 5
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    save_image_epochs: int = 10
    save_model_epochs: int = 30
    mixed_precision: str = "fp16"
    output_dir: str = "MNIST"
    push_to_hub: bool = False
    hub_model_id: str = "<your-username>/<my-awesome-model>"
    hub_private_repo: bool = False
    overwrite_output_dir: bool = True
    seed: int = 0
    
    # Concept / Context params
    concept: bool = True 
    context_dim: int = 8  # Dimension of context (CBM concept vector)
    skip_concept: bool = False
    num_concepts: int = 10
    
    # Model Architecture
    image_size: int = 32
    in_channels: int = 1
    out_channels: int = 1
    layers_per_block: int = 2
    latent_channels: int = 4 # Added this as it's required for VAEs
    
    block_out_channels: Tuple[int, ...] = (128, 128, 256, 256, 512)
    
    # These match the class names in diffusers.models.unet_2d_blocks
    down_block_types: Tuple[str, ...] = (
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "CrossAttnDownBlock2D",  # cross-attention block for concept conditioning
        "DownEncoderBlock2D",
    )

    up_block_types: Tuple[str, ...] = (
        "UpDecoderBlock2D",
        "AttnUpDecoderBlock2D",  # self-attention decoder block
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
    )