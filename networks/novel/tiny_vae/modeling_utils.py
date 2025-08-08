from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Utility functions and constants
from diffusers.utils import BaseOutput

# Activation helper

def get_activation(name: str):
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "swish":
        return nn.SiLU(inplace=True)
    if name == "mish":
        return nn.Mish(inplace=True)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unknown activation {name}")

# ---------------------- Residual Autoencoding Shortcuts --------------------- #

def downsample_shortcut_nchw(x: torch.Tensor, p: int = 2) -> torch.Tensor:
    # if p != 2:
    #     raise NotImplementedError("This helper currently assumes p = 2.")
    # b, c, h, w = x.shape
    # assert h % p == 0 and w % p == 0 and c % 2 == 0
    # x = F.pixel_unshuffle(x, downscale_factor=p)
    # g1, g2 = torch.chunk(x, 2, dim=1)
    # return 0.5 * (g1 + g2)

    x = F.pixel_unshuffle(x, downscale_factor=p) # (B, 4C, H/2, W/2) 

    # --> (B, 4, C, H/2, W/2) --> then mean across dim = 1
    B, C4, H2, W2 = x.shape            # C4 = 4C
    x = x.view(B, 4, C4 // 4, H2, W2)  # (B, 4, C, H/2, W/2)
    return x.mean(dim=1) 

def upsample_shortcut_nchw(x: torch.Tensor, p: int = 2) -> torch.Tensor:
    # if p != 2:
    #     raise NotImplementedError("This helper currently assumes p = 2.")
    # b, c2, h, w = x.shape
    # assert (c2 % 2 == 0) and (c2 % (p * p) == 0)
    # y = F.pixel_shuffle(x, upscale_factor=p)
    # return torch.cat([y, y], dim=1)

    assert p == 2, "only p=2 implemented"
    B, C, H2, W2 = x.shape
    assert C % (p * p) == 0, "channel dim must be divisible by 4"

    # 1) channel → space : divides channels by p²
    y = F.pixel_shuffle(x, upscale_factor=p)      # (B, C/4, H, W) 

    # 2) duplicate p² (=4) times → restore original C channels
    return y.repeat(1, p * p, 1, 1) 

class ResidualDownAE(nn.Module):
    def __init__(self, down_layer: nn.Module, p: int = 2):
        super().__init__()
        self.down = down_layer
        self.p = p

    def forward(self, x):
        first_part = self.down(x) # (H, W, C) --> (H/2, W/2, C)
        second_part = downsample_shortcut_nchw(x, self.p) # (H, W, C) --> (H/2, W/2, 2C)
        return first_part + second_part

class ResidualUpAE(nn.Module):
    def __init__(self, up_layer: nn.Module, p: int = 2):
        super().__init__()
        self.up = up_layer
        self.p = p

    def forward(self, x):
        first_part  = self.up(x) # (H/2, W/2, 4C) --> (H, W, 4C)
        second_part = upsample_shortcut_nchw(x, self.p) # (H/2, W/2, 4C) --> (H, W, C)
        return first_part + second_part

# ----------------------------- Tiny AE Block -------------------------------- #
class AutoencoderTinyBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, act_fn: str):
        super().__init__()
        act_fn = get_activation(act_fn)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,  kernel_size = 3, padding = 1),
            act_fn,
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1),
            act_fn,
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1),
        )
        self.skip = (
            nn.Conv2d(in_channels, out_channels,  kernel_size = 1, bias = False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.fuse = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuse(self.conv(x) + self.skip(x))

@dataclass
class DecoderOutput(BaseOutput):

    sample: torch.Tensor
    commit_loss: Optional[torch.FloatTensor] = None

# --------------------------------------------------------------------------- #
# EncoderTiny with Residual-Autoencoding shortcuts                            #
# --------------------------------------------------------------------------- #

class EncoderTiny(nn.Module):
    """
    Tiny VAE encoder with Residual Autoencoding.
    Each stride-2 conv is wrapped in `ResidualDownAE`, leaving all other
    behaviour untouched.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: tuple[int, ...],        # e.g. (1, 3, 3, 3)
        block_out_channels: tuple[int, ...],# e.g. (64, 64, 64, 64)
        act_fn: str,
    ):
        super().__init__()

        layers = []
        for i, num_block in enumerate(num_blocks):
            num_channels = block_out_channels[i] # 64

            # -----------------------------------------------------------------
            # Stem conv (no down-sample) for the very first stage
            # -----------------------------------------------------------------
            if i == 0:
                layers.append(nn.Conv2d(in_channels, num_channels, kernel_size = 3, padding = 1)) # (3, 64)

            # -----------------------------------------------------------------
            # Stride-2 down-sample conv wrapped with ResidualDownAE
            # -----------------------------------------------------------------
            else:
                down_conv = nn.Conv2d(num_channels, num_channels, # (orig: [64, 64]) --> (nehal_change: [64, 128] with H = 112, W = 112)
                                      kernel_size = 3, stride = 2, 
                                      padding = 1, bias = False,)
                layers.append(ResidualDownAE(down_conv))     # << insertion

            # -----------------------------------------------------------------
            # In-stage residual blocks (unchanged)
            # -----------------------------------------------------------------
            for _ in range(num_block):
                layers.append(AutoencoderTinyBlock(num_channels, num_channels, act_fn))

        # Final projection to latent channels (no down-sample)
        layers.append(nn.Conv2d(block_out_channels[-1], out_channels, kernel_size = 3, padding = 1))

        self.layers = nn.Sequential(*layers)
        self.gradient_checkpointing = False

    # ----------------------------- forward ---------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (unchanged except that self.layers now contains RAE
        modules). Keeps the TAESD convention of scaling [-1,1] → [0,1].
        """

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            x = self._gradient_checkpointing_func(self.layers, x)
        else:
            x = self.layers(x.add(1).div(2))   # scale from [-1,1] to [0,1]

        return x

# --------------------------------------------------------------------------- #
# DecoderTiny with Residual Autoencoding in every upsample stage              #
# --------------------------------------------------------------------------- #

class DecoderTiny(nn.Module):
    """
    DecoderTiny with Residual Autoencoding applied to each upsampling stage.

    For every `Upsample + Conv`, we wrap them as a block and apply a 
    `ResidualUpAE` to inject the non-parametric shortcut.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: Tuple[int, ...],
        block_out_channels: Tuple[int, ...],
        upsampling_scaling_factor: int,
        act_fn: str,
        upsample_fn: str,
    ):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, block_out_channels[0], kernel_size = 3, padding = 1),
            get_activation(act_fn),
        ]

        for i, num_block in enumerate(num_blocks):
            is_final_block = i == (len(num_blocks) - 1)
            num_channels   = block_out_channels[i]

            # ------------------------------------------
            # Residual blocks for this stage (unchanged)
            # ------------------------------------------
            for _ in range(num_block):
                layers.append(AutoencoderTinyBlock(num_channels, num_channels, act_fn))

            # ------------------------------------------
            # Upsample path — ResidualAutoencoding
            # ------------------------------------------
            if not is_final_block:
                # up_block = nn.Sequential(
                #     nn.Upsample(scale_factor = upsampling_scaling_factor, mode = upsample_fn),
                #     nn.Conv2d(prev_c, num_channels, kernel_size = 3, padding = 1, bias = False),
                # )
                
                # Note nn.Upsample does not modify the number of channels.
                up_block = nn.Upsample(scale_factor = upsampling_scaling_factor, mode = upsample_fn)
                layers.append(ResidualUpAE(up_block))

            # ------------------------------------------
            # Conv out for each stage (final has bias)
            # ------------------------------------------
            conv_out_channel = num_channels if not is_final_block else out_channels
            layers.append(
                nn.Conv2d(
                    num_channels,
                    conv_out_channel,
                    kernel_size = 3,
                    padding = 1,
                    bias = is_final_block,
                )
            )

        self.layers = nn.Sequential(*layers)
        self.gradient_checkpointing = False

    # ----------------------------- forward ---------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual upsampling shortcuts. Applies tanh scaling
        and remaps final output back to [-1, 1].
        """
        x = torch.tanh(x / 3) * 3

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            x = self._gradient_checkpointing_func(self.layers, x)
        else:
            x = self.layers(x)

        return x.mul(2).sub(1)  # rescale [0,1] → [-1,1]

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from dataclasses import dataclass
# from torch.utils.checkpoint import checkpoint
# from diffusers.utils import BaseOutput
# from typing import Optional, Tuple


# # ---------------------- Activation --------------------- #
# def get_activation(name: str):
#     name = name.lower()
#     if name == "relu":
#         return nn.ReLU(inplace=True)
#     if name == "swish":
#         return nn.SiLU(inplace=True)
#     if name == "mish":
#         return nn.Mish(inplace=True)
#     if name == "gelu":
#         return nn.GELU()
#     raise ValueError(f"Unknown activation {name}")

# # ---------------------- Residual Shortcuts --------------------- #
# def downsample_shortcut_nchw(x: torch.Tensor, p: int = 2) -> torch.Tensor:
#     x = F.pixel_unshuffle(x, downscale_factor=p)
#     B, C4, H2, W2 = x.shape
#     x = x.view(B, 4, C4 // 4, H2, W2)
#     return x.mean(dim=1)

# def upsample_shortcut_nchw(x: torch.Tensor, p: int = 2) -> torch.Tensor:
#     assert p == 2
#     B, C, H2, W2 = x.shape
#     y = F.pixel_shuffle(x, upscale_factor=p)
#     return y.repeat(1, p * p, 1, 1)

# class ResidualDownAE(nn.Module):
#     def __init__(self, down_layer: nn.Module, p: int = 2):
#         super().__init__()
#         self.down = down_layer
#         self.p = p

#     def _forward_impl(self, x):
#         return self.down(x) + downsample_shortcut_nchw(x, self.p)

#     def forward(self, x):
#         if self.training and x.requires_grad:
#             return checkpoint(self._forward_impl, x)
#         return self._forward_impl(x)

# class ResidualUpAE(nn.Module):
#     def __init__(self, up_layer: nn.Module, p: int = 2):
#         super().__init__()
#         self.up = up_layer
#         self.p = p

#     def _forward_impl(self, x):
#         return self.up(x) + upsample_shortcut_nchw(x, self.p)

#     def forward(self, x):
#         if self.training and x.requires_grad:
#             return checkpoint(self._forward_impl, x)
#         return self._forward_impl(x)

# # ---------------------- Blocks --------------------- #
# class AutoencoderTinyBlock(nn.Module):
#     def __init__(self, in_channels: int, out_channels: int, act_fn: str):
#         super().__init__()
#         act_fn = get_activation(act_fn)
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels,  kernel_size=3, padding=1),
#             act_fn,
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             act_fn,
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#         )
#         self.skip = nn.Conv2d(in_channels, out_channels, 1, bias=False) if in_channels != out_channels else nn.Identity()
#         self.fuse = nn.ReLU(inplace=True)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.fuse(self.conv(x) + self.skip(x))

# @dataclass
# class DecoderOutput(BaseOutput):
#     sample: torch.Tensor
#     commit_loss: Optional[torch.FloatTensor] = None

# # ---------------------- Encoder / Decoder --------------------- #
# class EncoderTiny(nn.Module):
#     def __init__(self, in_channels, out_channels, num_blocks, block_out_channels, act_fn):
#         super().__init__()
#         layers = []
#         for i, num_block in enumerate(num_blocks):
#             channels = block_out_channels[i]
#             if i == 0:
#                 layers.append(nn.Conv2d(in_channels, channels, kernel_size=3, padding=1))
#             else:
#                 layers.append(ResidualDownAE(nn.Conv2d(channels, channels, 3, 2, 1, bias=False)))

#             for _ in range(num_block):
#                 layers.append(AutoencoderTinyBlock(channels, channels, act_fn))

#         layers.append(nn.Conv2d(block_out_channels[-1], out_channels, kernel_size=3, padding=1))
#         self.layers = nn.Sequential(*layers)
#         self.gradient_checkpointing = False

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = x.add(1).div(2)  # [-1, 1] → [0, 1]
#         if self.gradient_checkpointing and x.requires_grad:
#             return checkpoint(self.layers, x)
#         return self.layers(x)

# class DecoderTiny(nn.Module):
#     def __init__(self, in_channels, out_channels, num_blocks, block_out_channels, upsampling_scaling_factor, act_fn, upsample_fn):
#         super().__init__()
#         layers = [
#             nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1),
#             get_activation(act_fn),
#         ]

#         for i, num_block in enumerate(num_blocks):
#             is_final = i == len(num_blocks) - 1
#             channels = block_out_channels[i]

#             for _ in range(num_block):
#                 layers.append(AutoencoderTinyBlock(channels, channels, act_fn))

#             if not is_final:
#                 up_block = nn.Upsample(scale_factor=upsampling_scaling_factor, mode=upsample_fn)
#                 layers.append(ResidualUpAE(up_block))

#             conv_out = out_channels if is_final else channels
#             layers.append(nn.Conv2d(channels, conv_out, 3, padding=1, bias=is_final))

#         self.layers = nn.Sequential(*layers)
#         self.gradient_checkpointing = False

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = torch.tanh(x / 3) * 3
#         if self.gradient_checkpointing and x.requires_grad:
#             return checkpoint(self.layers, x)
#         x = self.layers(x)
#         return x.mul(2).sub(1)
