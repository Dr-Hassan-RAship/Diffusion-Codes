# -----------------------------------------------------------------------------
# networks/novel/litevae/encoder.py
# -----------------------------------------------------------------------------
# LiteVAEEncoder: 3-level wavelet transform + shared feature extractors + UNet
# feature aggregation to produce VAE-style latent distribution (mu, logvar).
# -----------------------------------------------------------------------------

from __future__ import annotations

import torch
from torch import nn, Tensor


from networks.novel.lite_vae.blocks import HaarTransform, LiteVAEUNetBlock
from networks.novel.lite_vae.utils import Downsample2D

__all__ = ["LiteVAEEncoder"]


class LiteVAEEncoder(nn.Module):
    """
    LiteVAE encoder that performs:
      - 3-level DWT to extract multi-scale high/low frequency bands
      - Shared feature extractors (UNet blocks) at each level
      - Downsampling to target resolution
      - Final aggregation UNet to produce mean+logvar

    Output shape is (B, 2C, H, W), where 2C = [mu, logvar] for Gaussian sampling.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 4,
        wavelet_levels: int = 3,
        feature_extractor_params: dict = {},
        feature_aggregator_params: dict = {},
    ):
        super().__init__()
        self.levels = wavelet_levels
        self.wavelet_fn = HaarTransform(level = self.levels)

        # Shared UNets (1 per level)
        self.feature_extractor_L1 = LiteVAEUNetBlock(4, 4, **feature_extractor_params) # replacing in_channels with #sub-bands
        self.feature_extractor_L2 = LiteVAEUNetBlock(4, 4, **feature_extractor_params)
        self.feature_extractor_L3 = LiteVAEUNetBlock(4, 4, **feature_extractor_params)

        # Aggregator UNet: (L1 + L2 + L3) --> 2 * out_channels
        aggregated_channels  = in_channels * 3
        out_channels         = out_channels * 2  # [mu | logvar]
        self.feature_aggregator = LiteVAEUNetBlock(
            aggregated_channels, out_channels, **feature_aggregator_params
        )

        # Downsamplers for L1 and L2 features to match L3 size
        self.downsample_L1 = Downsample2D(in_channels, scale_factor = 4)
        self.downsample_L2 = Downsample2D(in_channels, scale_factor = 2)

    def forward(self, image: Tensor) -> Tensor:
        # Get all sub-bands from DWT for each level
        level_tensors, coeff_dict = self.wavelet_fn(image)
        dwt_L1 = level_tensors[0] # (B, 4, H/2, W/2)
        dwt_L2 = level_tensors[1] # (B, 4, H/4, W/4)
        dwt_L3 = level_tensors[2] # (B, 4, H/8, W/8)

        # Feature extraction (shared UNet blocks)
        feat_L1 = self.downsample_L1(self.feature_extractor_L1(dwt_L1)) # (B, 4, H/8, H/8)
        feat_L2 = self.downsample_L2(self.feature_extractor_L2(dwt_L2)) # (B, 4, H/8, H/8)
        feat_L3 = self.feature_extractor_L3(dwt_L3)

        # Concatenate and aggregate to latent (mu + logvar)
        feat_cat = torch.cat([feat_L1, feat_L2, feat_L3], dim=1) # (B, 12, H/8, W/8)
        latent_out = self.feature_aggregator(feat_cat) # (B, 8, H/8, W/8) --> [mu, logvar]
        return latent_out
