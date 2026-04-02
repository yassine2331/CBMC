import torch
import torch.nn as nn
from diffusers import ModelMixin, ConfigMixin
from diffusers.configuration_utils import register_to_config
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

# VAE-specific encoder/decoder blocks (no skip-connection residuals needed)
from diffusers.models.unets.unet_2d_blocks import (
    DownEncoderBlock2D,      # plain downsampling encoder block
    AttnDownEncoderBlock2D,  # self-attention downsampling encoder block
    UpDecoderBlock2D,        # plain upsampling decoder block
    AttnUpDecoderBlock2D,    # self-attention upsampling decoder block
    CrossAttnDownBlock2D,    # cross-attention downsampling (concept conditioning)
)

# Registry maps config string names to the actual block classes.
# New entries can be added here without changing any other code.
BLOCK_REGISTRY = {
    # --- encoder blocks ---
    "DownEncoderBlock2D": DownEncoderBlock2D,
    "AttnDownEncoderBlock2D": AttnDownEncoderBlock2D,
    "CrossAttnDownBlock2D": CrossAttnDownBlock2D,
    # --- decoder blocks ---
    "UpDecoderBlock2D": UpDecoderBlock2D,
    "AttnUpDecoderBlock2D": AttnUpDecoderBlock2D,
}


class DynamicVAE(ModelMixin, ConfigMixin):
    """Concept-conditioned Variational Autoencoder.

    The encoder accepts an optional *encoder_hidden_states* context tensor
    (the concept vector) that is fed to any ``CrossAttnDownBlock2D`` blocks in
    the encoder path.  The decoder is a standard VAE decoder and does not
    require skip-connection residuals.

    Block type strings are resolved via ``BLOCK_REGISTRY`` so new architectures
    can be plugged in by updating the registry alone.
    """

    @register_to_config
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        block_out_channels: tuple = (64, 128),
        down_block_types: tuple = ("DownEncoderBlock2D", "CrossAttnDownBlock2D"),
        up_block_types: tuple = ("UpDecoderBlock2D", "UpDecoderBlock2D"),
        layers_per_block: int = 2,
        latent_channels: int = 4,
        context_dim: int = 8,
        sample_size: int = 32,
    ):
        super().__init__()

        # ------------------------------------------------------------------ #
        # ENCODER                                                              #
        # ------------------------------------------------------------------ #
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1)
        self.encoder_layers = nn.ModuleList()

        output_channel = block_out_channels[0]
        for i, block_name in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            block_class = BLOCK_REGISTRY[block_name]
            resnet_groups = 32 if output_channel % 32 == 0 else min(output_channel, 4)

            if block_name == "CrossAttnDownBlock2D":
                # CrossAttnDownBlock2D: concept cross-attention in the encoder
                block = block_class(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=None,
                    dropout=0.0,
                    num_layers=layers_per_block,
                    resnet_eps=1e-6,
                    resnet_act_fn="swish",
                    resnet_groups=resnet_groups,
                    cross_attention_dim=context_dim,
                    num_attention_heads=1,
                    add_downsample=not is_final_block,
                )
            elif block_name == "AttnDownEncoderBlock2D":
                block = block_class(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    dropout=0.0,
                    num_layers=layers_per_block,
                    resnet_eps=1e-6,
                    resnet_act_fn="swish",
                    resnet_groups=resnet_groups,
                    attention_head_dim=1,
                    add_downsample=not is_final_block,
                )
            else:
                # DownEncoderBlock2D
                block = block_class(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    dropout=0.0,
                    num_layers=layers_per_block,
                    resnet_eps=1e-6,
                    resnet_act_fn="swish",
                    resnet_groups=resnet_groups,
                    add_downsample=not is_final_block,
                )
            self.encoder_layers.append(block)

        # Maps encoder output → latent distribution parameters (mean + logvar)
        self.quant_conv = nn.Conv2d(output_channel, 2 * latent_channels, kernel_size=1)

        # ------------------------------------------------------------------ #
        # DECODER                                                              #
        # ------------------------------------------------------------------ #
        # Projects sampled latent back to the first decoder channel width
        self.post_quant_conv = nn.Conv2d(latent_channels, block_out_channels[-1], kernel_size=1)
        self.decoder_layers = nn.ModuleList()

        reversed_channels = list(reversed(block_out_channels))
        output_channel = reversed_channels[0]

        for i, block_name in enumerate(up_block_types):
            input_channel = output_channel
            output_channel = reversed_channels[i]
            is_final_block = i == len(up_block_types) - 1

            block_class = BLOCK_REGISTRY[block_name]
            resnet_groups = 32 if output_channel % 32 == 0 else min(output_channel, 4)

            if block_name == "AttnUpDecoderBlock2D":
                block = block_class(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    dropout=0.0,
                    num_layers=layers_per_block + 1,
                    resnet_eps=1e-6,
                    resnet_act_fn="swish",
                    resnet_groups=resnet_groups,
                    attention_head_dim=1,
                    add_upsample=not is_final_block,
                )
            else:
                # UpDecoderBlock2D
                block = block_class(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    dropout=0.0,
                    num_layers=layers_per_block + 1,
                    resnet_eps=1e-6,
                    resnet_act_fn="swish",
                    resnet_groups=resnet_groups,
                    add_upsample=not is_final_block,
                )
            self.decoder_layers.append(block)

        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    # ---------------------------------------------------------------------- #

    def encode(self, x, encoder_hidden_states=None):
        h = self.conv_in(x)
        for block in self.encoder_layers:
            if isinstance(block, CrossAttnDownBlock2D):
                h, _ = block(h, temb=None, encoder_hidden_states=encoder_hidden_states)
            else:
                h = block(h)
        moments = self.quant_conv(h)
        return DiagonalGaussianDistribution(moments)

    def decode(self, z, encoder_hidden_states=None):  # noqa: ARG002 (kept for API symmetry)
        h = self.post_quant_conv(z)
        for block in self.decoder_layers:
            h = block(h)
        return self.conv_out(h)

    def forward(self, sample, encoder_hidden_states=None, sample_posterior=True, return_dict=True):
        posterior = self.encode(sample, encoder_hidden_states)
        z = posterior.sample() if sample_posterior else posterior.mode()
        dec = self.decode(z, encoder_hidden_states)
        if not return_dict:
            return (dec,)
        return {"sample": dec, "latent_dist": posterior}