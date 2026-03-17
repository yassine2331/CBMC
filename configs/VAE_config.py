from dataclasses import dataclass, field
from typing import Optional, Tuple

@dataclass
class TrainingConfig:
    # Experiment identity — used by run.py to load and log experiments
    experiment_name: str = "default"
    # Must match a key registered in models/registry.py.
    # VAE models:  "dynamic_vae"
    # CEM models:  "cem_v1", "cem_v2"
    model_name: str = "dynamic_vae"

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

    # ------------------------------------------------------------------ #
    # Concept / Context params (shared by VAE and CEM families)           #
    # ------------------------------------------------------------------ #
    concept: bool = True
    context_dim: int = 8   # dimension of the concept context vector fed to the VAE
    skip_concept: bool = False
    num_concepts: int = 10  # number of concepts (used by CEM models)

    # ------------------------------------------------------------------ #
    # VAE Architecture                                                    #
    # ------------------------------------------------------------------ #
    image_size: int = 32
    in_channels: int = 1
    out_channels: int = 1
    layers_per_block: int = 2
    latent_channels: int = 4

    block_out_channels: Tuple[int, ...] = (128, 128, 256, 256, 512)

    # These match the class names registered in models/CBM_VAE.BLOCK_REGISTRY
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

    # ------------------------------------------------------------------ #
    # CEM Architecture                                                    #
    # Fields used by cem_v1, cem_v2 (ignored when training a VAE model). #
    # ------------------------------------------------------------------ #
    input_dim: int = 128        # raw feature dimension fed to the CEM
    hidden_dim: int = 64        # width of hidden layers inside per-concept networks
    embedding_dim: int = 16     # size of each concept's positive/negative embedding
    depth: int = 2              # number of hidden layers in each sub-network
    dropout: float = 0.2        # dropout rate
    # CEM_v2 only: size of the shared backbone output.
    # None → defaults to hidden_dim * 2 inside the model.
    shared_dim: Optional[int] = None