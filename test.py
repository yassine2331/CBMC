import torch
from configs.VAE_config import TrainingConfig
from models.CBM_VAE import DynamicVAE

# 1. Initialize Config
config = TrainingConfig()

print(f"Creating VAE with Down Blocks: {config.down_block_types}")

# 2. Initialize Model using Config
# We unpack the config dataclass into arguments
vae = DynamicVAE(
    in_channels=config.in_channels,
    out_channels=config.out_channels,
    block_out_channels=config.block_out_channels,
    down_block_types=config.down_block_types,
    up_block_types=config.up_block_types,
    layers_per_block=config.layers_per_block,
    latent_channels=config.latent_channels,
    context_dim=config.context_dim,
    sample_size=config.image_size
)

# 3. Create Dummy Data
# Image: (Batch, Channel, Height, Width)
dummy_image = torch.randn(1, config.in_channels, 32, 32)

# Concept Vector: (Batch, Sequence_Length, Context_Dim)
# Since you used AttnDownBlock2D, the model expects this conditioning context
dummy_context = torch.randn(1, 1, config.context_dim)

# 4. Forward Pass
output = vae(dummy_image, encoder_hidden_states=dummy_context)

print("\n--- Success ---")
print(f"Input Shape: {dummy_image.shape}")
print(f"Latent Shape: {output['latent_dist'].mean.shape}")
print(f"Output Shape: {output['sample'].shape}")

# 5. Save Config and Model
vae.save_pretrained(config.output_dir)
print(f"\nModel saved to {config.output_dir}")