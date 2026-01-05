import torch
import torch.nn as nn
from diffusers import ModelMixin, ConfigMixin
from diffusers.configuration_utils import register_to_config
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

# 1. Explicitly import the blocks from the top-level models package
# This avoids the "ModuleNotFoundError" caused by changing internal paths
from diffusers.models.unets.unet_2d_blocks import (
    DownBlock2D,
    UpBlock2D,
    AttnDownBlock2D,
    AttnUpBlock2D
)
# 2. Create a registry to map the string names in your config to the classes
BLOCK_REGISTRY = {
    "DownBlock2D": DownBlock2D,
    "AttnDownBlock2D": AttnDownBlock2D,
    "UpBlock2D": UpBlock2D,
    "AttnUpBlock2D": AttnUpBlock2D,
}

class DynamicVAE(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        block_out_channels: tuple = (64, 128),
        down_block_types: tuple = ("DownBlock2D", "AttnDownBlock2D"),
        up_block_types: tuple = ("AttnUpBlock2D", "UpBlock2D"),
        layers_per_block: int = 2,
        latent_channels: int = 4,
        context_dim: int = 8, 
        sample_size: int = 32,
    ):
        super().__init__()

        # --- ENCODER ---
        self.encoder_layers = nn.ModuleList([])
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1)

        output_channel = block_out_channels[0]
        
        for i, block_name in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = (i == len(block_out_channels) - 1)
            
            # Use the dictionary registry instead of importlib
            block_class = BLOCK_REGISTRY[block_name]
            
            cross_attn_dim = context_dim if "Attn" in block_name else None

            block = block_class(
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=None, 
                dropout=0.0,
                num_layers=layers_per_block,
                resnet_eps=1e-6,
                resnet_act_fn="swish",
                # Ensure groups don't exceed channels (common error for small layers)
                resnet_groups=32 if output_channel % 32 == 0 else min(output_channel, 4), 
                add_downsample=not is_final_block, 
                cross_attention_dim=cross_attn_dim 
            )
            self.encoder_layers.append(block)

        self.quant_conv = nn.Conv2d(output_channel, 2 * latent_channels, kernel_size=1)

        # --- DECODER ---
        self.post_quant_conv = nn.Conv2d(latent_channels, block_out_channels[-1], kernel_size=1)
        self.decoder_layers = nn.ModuleList([])
        
        reversed_block_out = list(reversed(block_out_channels))
        # Ensure we use the user-provided up_block_types, which should match the reversed order semantically
        reversed_block_types = list(up_block_types) 
        
        output_channel = reversed_block_out[0]
        
        for i, block_name in enumerate(reversed_block_types):
            input_channel = output_channel
            output_channel = reversed_block_out[i]
            is_final_block = (i == len(reversed_block_types) - 1)
            
            block_class = BLOCK_REGISTRY[block_name]
            cross_attn_dim = context_dim if "Attn" in block_name else None

            block = block_class(
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=None,
                dropout=0.0,
                num_layers=layers_per_block + 1, 
                resnet_eps=1e-6,
                resnet_act_fn="swish",
                resnet_groups=32 if output_channel % 32 == 0 else min(output_channel, 4),
                add_upsample=not is_final_block,
                cross_attention_dim=cross_attn_dim
            )
            self.decoder_layers.append(block)

        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    def encode(self, x, encoder_hidden_states=None):
        h = self.conv_in(x)
        for block in self.encoder_layers:
            # We must pass encoder_hidden_states (context) to every block
            # The blocks that don't need it will simply ignore the argument internally
            h, _ = block(h, temb=None, encoder_hidden_states=encoder_hidden_states)
        moments = self.quant_conv(h)
        return DiagonalGaussianDistribution(moments)

    def decode(self, z, encoder_hidden_states=None):
        h = self.post_quant_conv(z)
        for block in self.decoder_layers:
            h = block(h, temb=None, encoder_hidden_states=encoder_hidden_states)
        return self.conv_out(h)

    def forward(self, sample, encoder_hidden_states=None, sample_posterior=True, return_dict=True):
        posterior = self.encode(sample, encoder_hidden_states)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z, encoder_hidden_states)
        
        if not return_dict:
            return (dec,)
        return {"sample": dec, "latent_dist": posterior}