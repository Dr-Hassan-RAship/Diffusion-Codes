# -----------------------------------------------------------------------------
# networks/novel/litevae/utils.py
# -----------------------------------------------------------------------------
# Utility layers and helpers shared across LiteVAE modules.
#   • Downsample2D : non‑learnable (nearest‑avg) or learnable strided‑conv downsampler
#   • DiagonalGaussianDistribution : VAE reparameterisation helper + KL
# -----------------------------------------------------------------------------

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor

__all__ = [
    "Downsample2D",
    "DiagonalGaussianDistribution",
]

# -----------------------------------------------------------------------------
# Downsample2D -----------------------------------------------------------------
# -----------------------------------------------------------------------------

class Downsample2D(nn.Module):
    """2× (or arbitrary) spatial downsampler.

    If `learnable=True`, uses a strided conv (kernel=3, stride=scale_factor).
    Otherwise uses average pooling + optional 1×1 conv for channel projection.
    """

    def __init__(
        self,
        channels: int,
        scale_factor: int = 2,
        learnable: bool = False,
    ) -> None:
        super().__init__()
        self.scale = scale_factor

        if learnable:
            self.op = nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=scale_factor,
                padding=1,
                bias=False,
            )
        else:
            self.op = nn.AvgPool2d(kernel_size=scale_factor, stride=scale_factor)

    def forward(self, x: Tensor) -> Tensor:
        return self.op(x)


# -----------------------------------------------------------------------------
# Diagonal Gaussian Distribution ----------------------------------------------
# -----------------------------------------------------------------------------

class DiagonalGaussianDistribution:
    """Helper class for VAE reparameterisation and KL divergence.

    Parameters
    ----------
    params : Tensor
        Concatenated tensor `[mu, logvar]` with shape `(B, 2*C, H, W)`.
    """

    def __init__(self, params: Tensor):
        self.mu, self.logvar = torch.chunk(params, 2, dim=1)
        self.std = torch.exp(0.5 * self.logvar)

    def sample(self) -> Tensor:
        """Reparameterised sample."""
        eps = torch.randn_like(self.std)
        return self.mu + eps * self.std

    def mode(self) -> Tensor:
        """Deterministic mode (mean)."""
        return self.mu

    def kl(self) -> Tensor:
        """Pixel‑wise KL divergence N(mu,σ) || N(0,1).
        Returns tensor of same spatial shape (averaging up to caller).
        """
        return 0.5 * (self.mu.pow(2) + self.std.pow(2) - 1.0 - self.logvar)
