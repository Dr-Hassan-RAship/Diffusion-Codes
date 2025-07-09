# -----------------------------------------------------------------------------
# networks/novel/litevae/litevae.py
# -----------------------------------------------------------------------------
# LiteVAE full model wrapper
#   - Encoder: multi-level DWT + UNet-based extractor and aggregator
#   - Decoder: imported from pretrained HuggingFace VAE (Tiny or KL)
#   - Output: image or wavelet + VAE KL + latents
# -----------------------------------------------------------------------------

from __future__ import annotations

import torch
from torch import nn
from typing import Literal

from .encoder import LiteVAEEncoder
from .utils import DiagonalGaussianDistribution


class LiteVAE(nn.Module):
    def __init__(
        self,
        encoder: LiteVAEEncoder,
        decoder: nn.Module,
        latent_dim: int = 4,
        output_type: Literal["image", "wavelet"] = "image",
        use_1x1_conv: bool = True,
    ) -> None:
        super().__init__()
        assert output_type in ["image", "wavelet"]

        self.encoder = encoder
        self.decoder = decoder
        self.wavelet_fn = encoder.wavelet_fn
        self.output_type = output_type

        pre_channels = latent_dim * 2  # For [mu, logvar]
        post_channels = latent_dim     # Actual decoded latent channels

        self.pre_conv = nn.Conv2d(pre_channels, pre_channels, 1) if use_1x1_conv else nn.Identity()
        self.post_conv = nn.Conv2d(post_channels, post_channels, 1) if use_1x1_conv else nn.Identity()

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return self.pre_conv(self.encoder(image))

    def decode(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.post_conv(latent)

        if self.output_type == "image":
            image_recon = self.decoder(latent)
            wavelet_recon = self.wavelet_fn.dwt(image_recon, level=1) / 2
        else:
            wavelet_recon = self.decoder(latent)
            image_recon = self.wavelet_fn.idwt(wavelet_recon, level=1) * 2

        return image_recon, wavelet_recon

    def forward(self, image: torch.Tensor, sample: bool = True) -> dict:
        latents_raw = self.encode(image)
        latent_dist = DiagonalGaussianDistribution(latents_raw)
        latent = latent_dist.sample() if sample else latent_dist.mode()
        kl_reg = latent_dist.kl().mean()

        image_recon, wavelet_recon = self.decode(latent)

        return {
            "sample": image_recon,
            "wavelet": wavelet_recon,
            "latent": latent,
            "kl_reg": kl_reg,
            "latent_dist": latent_dist,
        }
