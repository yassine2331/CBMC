from dataclasses import dataclass
from typing import Tuple

@dataclass
class TrainingConfig:
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
        "DownBlock2D", 
        "DownBlock2D", 
        "DownBlock2D", 
        "AttnDownBlock2D", # Cross-Attn block
        "DownBlock2D"
    )
    
    up_block_types: Tuple[str, ...] = (
        "UpBlock2D", 
        "AttnUpBlock2D",   # Cross-Attn block
        "UpBlock2D", 
        "UpBlock2D", 
        "UpBlock2D"
    )