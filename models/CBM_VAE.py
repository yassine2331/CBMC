import torch
import torch.nn as nn
import importlib
from diffusers import ModelMixin, ConfigMixin
from diffusers.configuration_utils import register_to_config
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

def get_block_class(block_name):
    """Helper to import diffusers UNet blocks dynamically by string name."""
    # We look inside unet_2d_blocks because that is where DownBlock2D/AttnDownBlock2D live
    module = importlib.import_module("diffusers.models.unet_2d_blocks")
    return getattr(module, block_name)

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
        context_dim: int = 8,  # For Cross-Attention
        sample_size: int = 32,
    ):
        super().__init__()

        # ------------------------------------------------------------------
        # 1. ENCODER
        # ------------------------------------------------------------------
        self.encoder_layers = nn.ModuleList([])
        
        # Initial convolution to scale up channels
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1)

        # Construct Down Blocks based on config strings
        output_channel = block_out_channels[0]
        for i, block_name in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = (i == len(block_out_channels) - 1)
            
            block_class = get_block_class(block_name)
            
            # UNet blocks in diffusers take specific args. 
            # We map context_dim to cross_attention_dim if it's an Attention block.
            # "Attn" in the name implies it supports cross_attention_dim in diffusers API
            cross_attn_dim = context_dim if "Attn" in block_name else None

            block = block_class(
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=None, # VAEs usually don't use time embeddings
                dropout=0.0,
                num_layers=layers_per_block,
                resnet_eps=1e-6,
                resnet_act_fn="swish",
                resnet_groups=32 if output_channel % 32 == 0 else min(output_channel, 4), # Safety for small channels
                add_downsample=not is_final_block, # Downsample on all except last
                cross_attention_dim=cross_attn_dim 
            )
            self.encoder_layers.append(block)

        # Latent Space Projection (Moments: Mean + Variance)
        self.quant_conv = nn.Conv2d(output_channel, 2 * latent_channels, kernel_size=1)

        # ------------------------------------------------------------------
        # 2. DECODER
        # ------------------------------------------------------------------
        self.post_quant_conv = nn.Conv2d(latent_channels, block_out_channels[-1], kernel_size=1)
        
        self.decoder_layers = nn.ModuleList([])
        
        # Reverse channels and types for decoder
        reversed_block_out = list(reversed(block_out_channels))
        reversed_block_types = list(up_block_types) # User provided UpTypes match the reversed order
        
        output_channel = reversed_block_out[0]
        
        for i, block_name in enumerate(reversed_block_types):
            input_channel = output_channel
            output_channel = reversed_block_out[i]
            is_final_block = (i == len(reversed_block_types) - 1)
            
            block_class = get_block_class(block_name)
            cross_attn_dim = context_dim if "Attn" in block_name else None

            block = block_class(
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=None,
                dropout=0.0,
                num_layers=layers_per_block + 1, # Up blocks often have +1 layer
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
            # UNet blocks expect 'temb' and 'encoder_hidden_states'
            # For non-attn blocks, encoder_hidden_states is ignored by the block internally
            h, _ = block(h, temb=None, encoder_hidden_states=encoder_hidden_states)
            
        moments = self.quant_conv(h)
        return DiagonalGaussianDistribution(moments)

    def decode(self, z, encoder_hidden_states=None):
        h = self.post_quant_conv(z)
        
        for block in self.decoder_layers:
            h = block(h, temb=None, encoder_hidden_states=encoder_hidden_states)
            
        return self.conv_out(h)

    def forward(self, sample, encoder_hidden_states=None, sample_posterior=True, return_dict=True):
        """
        Args:
            sample: Input image (B, C, H, W)
            encoder_hidden_states: Context vector for Cross-Attention (B, Seq, context_dim)
        """
        # 1. Encode
        posterior = self.encode(sample, encoder_hidden_states)
        
        # 2. Reparameterize
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
            
        # 3. Decode
        dec = self.decode(z, encoder_hidden_states)
        
        if not return_dict:
            return (dec,)
        
        return {"sample": dec, "latent_dist": posterior}