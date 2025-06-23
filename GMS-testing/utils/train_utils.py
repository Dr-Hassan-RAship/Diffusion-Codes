# ------------------------------------------------------------------------------#
#
# File name                 : train_utils.py
# Purpose                   : Utility functions for training/validation loops, e.g.
#                             multi-loss calculation, VAE helpers, etc.
# Usage                     : Imported by train.py and valid.py
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh, Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 202410001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : June 23, 2025
# ------------------------------------------------------------------------------#

# --------------------------- Module imports ----------------------------------#
import torch
from networks.models.distributions import DiagonalGaussianDistribution

# --------------------------- Multi-loss utility ------------------------------#
def get_multi_loss(
    criterion,
    out_dict,
    label,
    is_ds=True,
    key_list=None
):
    """
    Computes (optionally multi-scale) loss over outputs.
    Args:
        criterion  : loss function (e.g. MSE, DiceLoss)
        out_dict   : dict of outputs (multi-scale or single scale)
        label      : ground truth label tensor
        is_ds      : if True, sum over multiple keys; else just use 'out'
        key_list   : explicit list of keys to use (default: all keys in out_dict)
    Returns:
        multi_loss : scalar tensor (sum of all relevant losses)
    """
    keys = key_list if key_list is not None else list(out_dict.keys())
    if is_ds:
        multi_loss = sum([criterion(out_dict[key], label) for key in keys])
    else:
        multi_loss = criterion(out_dict["out"], label)
    return multi_loss

# ----------------------- VAE encoding helpers --------------------------------#
def get_vae_encoding_mu_and_sigma(encoder_posterior, scale_factor):
    """
    Extract mean and logvar from VAE posterior.
    Args:
        encoder_posterior : DiagonalGaussianDistribution object
        scale_factor      : scaling to apply to mean
    Returns:
        (mean, logvar)
    """
    if isinstance(encoder_posterior, DiagonalGaussianDistribution):
        mean, logvar = encoder_posterior.mu_and_sigma()
    else:
        raise NotImplementedError(
            f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented"
        )
    return scale_factor * mean, logvar

# --------------------------- VAE decode utility ------------------------------#
def vae_decode(vae_model, pred_mean, scale_factor):
    """
    Decode latent to image, postprocess, and clamp.
    Args:
        vae_model   : model with .decode()
        pred_mean   : latent tensor
        scale_factor: latent scaling (float)
    Returns:
        pred_seg    : (B, 1, H, W) tensor, values in [0, 1]
    """
    z        = 1.0 / scale_factor * pred_mean
    pred_seg = vae_model.decode(z).sample  # shape: (B, C, H, W)
    pred_seg = torch.mean(pred_seg, dim=1, keepdim=True)  # (B, 1, H, W)
    pred_seg = torch.clamp((pred_seg + 1.0) / 2.0, min=0.0, max=1.0)
    return pred_seg

# -------------------------------- End ----------------------------------------#