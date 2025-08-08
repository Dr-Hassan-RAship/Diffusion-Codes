from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.utils.accelerate_utils import apply_forward_hook
from diffusers.models.modeling_utils  import ModelMixin
from networks.novel.tiny_vae.modeling_utils import *

@dataclass
class AutoencoderTinyOutput(BaseOutput):
    """
    Output of AutoencoderTiny encoding method.

    Args:
        latents (`torch.Tensor`): Encoded outputs of the `Encoder`.

    """
    latents: torch.Tensor

class AutoencoderTiny(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True
    
    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        encoder_block_out_channels: Tuple[int, ...] = (64, 64, 64, 64),
        decoder_block_out_channels: Tuple[int, ...] = (64, 64, 64, 64),
        act_fn: str = "relu",
        upsample_fn: str = "nearest",
        latent_channels: int = 4,
        upsampling_scaling_factor: int = 2,
        num_encoder_blocks: Tuple[int, ...] = (1, 3, 3, 3),
        num_decoder_blocks: Tuple[int, ...] = (3, 3, 3, 1),
        latent_magnitude: int = 3,
        latent_shift: float = 0.5,
        force_upcast: bool = False,
        scaling_factor: float = 1.0,
        shift_factor: float = 0.0,
    ):
        super().__init__()

        if len(encoder_block_out_channels) != len(num_encoder_blocks):
            raise ValueError("`encoder_block_out_channels` should have the same length as `num_encoder_blocks`.")
        if len(decoder_block_out_channels) != len(num_decoder_blocks):
            raise ValueError("`decoder_block_out_channels` should have the same length as `num_decoder_blocks`.")

        self.encoder = EncoderTiny(
            in_channels=in_channels,
            out_channels=latent_channels,
            num_blocks=num_encoder_blocks,
            block_out_channels=encoder_block_out_channels,
            act_fn=act_fn,
        )

        self.decoder = DecoderTiny(
            in_channels=latent_channels,
            out_channels=out_channels,
            num_blocks=num_decoder_blocks,
            block_out_channels=decoder_block_out_channels,
            upsampling_scaling_factor=upsampling_scaling_factor,
            act_fn=act_fn,
            upsample_fn=upsample_fn,
        )
        
        self.latent_magnitude = latent_magnitude
        self.latent_shift = latent_shift
        self.scaling_factor = scaling_factor

        self.register_to_config(block_out_channels=decoder_block_out_channels)
        self.register_to_config(force_upcast=False)

    def scale_latents(self, x: torch.Tensor) -> torch.Tensor:
        """raw latents -> [0, 1]"""
        return x.div(2 * self.latent_magnitude).add(self.latent_shift).clamp(0, 1)

    def unscale_latents(self, x: torch.Tensor) -> torch.Tensor:
        """[0, 1] -> raw latents"""
        return x.sub(self.latent_shift).mul(2 * self.latent_magnitude)

    @apply_forward_hook
    def encode(self, x: torch.Tensor, return_dict: bool = True) -> Union[AutoencoderTinyOutput, Tuple[torch.Tensor]]:
        
        output = self.encoder(x)

        if not return_dict:
            return (output,)
        
        return AutoencoderTinyOutput(latents=output)
    
    @apply_forward_hook
    def decode(
        self, x: torch.Tensor, generator: Optional[torch.Generator] = None, return_dict: bool = True
    ) -> Union[DecoderOutput, Tuple[torch.Tensor]]:
        
        output = self.decoder(x)

        if not return_dict:
            return (output,)

        return DecoderOutput(sample=output)

    def forward(
        self,
        sample: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[DecoderOutput, Tuple[torch.Tensor]]:
        r"""
        Args:
            sample (`torch.Tensor`): Input sample.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        enc = self.encode(sample).latents

        # scale latents to be in [0, 1], then quantize latents to a byte tensor,
        # as if we were storing the latents in an RGBA uint8 image.
        scaled_enc = self.scale_latents(enc).mul_(255).round_().byte()

        # unquantize latents back into [0, 1], then unscale latents back to their original range,
        # as if we were loading the latents from an RGBA uint8 image.
        unscaled_enc = self.unscale_latents(scaled_enc / 255.0)

        dec = self.decode(unscaled_enc).sample

        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)
