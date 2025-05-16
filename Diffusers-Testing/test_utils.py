# ------------------------------------------------------------------------------#
#
# File name                 : modeling_utils.py
# Purpose                   : Architecture utilities for Latent Diffusion Model (LDM) segmentation 
# Usage                     : Used for forward pass, denoising, and decoding
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh, Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 202410001@lums.edu.pk, hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : April 28, 2025
# ------------------------------------------------------------------------------#

import torch, copy

import torch.nn                              as nn

from   diffusers                             import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from   config                                import *
from   tqdm.auto                             import tqdm
from   diffusers.models.autoencoders.vae     import DiagonalGaussianDistribution
from   diffusers.models.modeling_outputs     import AutoencoderKLOutput

#--------------------------------------------------------------------------------------
class TauEncoder(nn.Module):
    """
    Learnable encoder for the input RGB image (τ_θ).
    Architecturally same as the VAE encoder but trainable.
    """
    def __init__(self, encoder):
        super().__init__()
        self.encoder    = encoder
        latent_channels = 4
        use_quant_conv  = True
        self.quant_cov  = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1) if use_quant_conv else None

    def forward(self, x):
        h         = self.encoder(x)
        h         = self.quant_cov(h) if self.quant_cov else h
        h         = AutoencoderKLOutput(DiagonalGaussianDistribution(h)).latent_dist
        return h.mean if DETERMINISTIC else h.sample()

#--------------------------------------------------------------------------------------
class CombinedL1L2Loss(nn.Module):
    """
    Custom loss: L2 (MSE) on noise, L1 on z0.
    """

    def __init__(self, l1_weight = 1.0, l2_weight = 1.0, reduction = "mean"):
        """
        Args:
            l1_weight (float): weight for L1 loss term.
            l2_weight (float): weight for L2 loss term.
            reduction (str): reduction method ("mean", "sum", etc.)
        """
        super().__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.reduction = reduction

        self.l1_loss = nn.L1Loss(reduction  = self.reduction)
        self.l2_loss = nn.MSELoss(reduction = self.reduction)

    def forward(self, noise_hat, noise, z0_hat, z0):
        """
        Args:
            noise_hat: Predicted noise
            noise: True noise
            z0_hat: Predicted z0
            z0: True z0
        """
        l2      = self.l2_loss(noise_hat, noise)
        l1      = self.l1_loss(z0_hat, z0)
        loss    = self.l2_weight * l2 + self.l1_weight * l1
        
        return loss

#--------------------------------------------------------------------------------------
def load_hybrid_unet(pretrained_path: str, device: str = "cuda") -> UNet2DConditionModel:
    # Step 1: Load the original model to access weights and config
    unet_orig = UNet2DConditionModel.from_pretrained(
        pretrained_path,
        subfolder="unet",
        torch_dtype=torch.float16
    ).to(device)

    # Step 2: Create a deep copy of the config and modify in_channels
    config = copy.deepcopy(unet_orig.config)  # Avoid modifying the original config
    config['in_channels'] = 8  # Update in_channels to 8

    # Step 3: Create a new UNet model with modified in_channels
    unet_new = UNet2DConditionModel(**config).to(device, dtype=torch.float16)

    # Step 4: Get the original and new state dicts
    orig_sd = unet_orig.state_dict()
    new_sd = unet_new.state_dict()

    # Step 5: Initialize new conv_in weights manually
    old_conv = orig_sd["conv_in.weight"]  # Shape [320, 4, 3, 3]
    out_ch, _, kH, kW = old_conv.shape

    # New conv_in.weight: [320, 8, 3, 3]
    new_conv = torch.zeros((out_ch, 8, kH, kW), dtype=torch.float16, device=device)
    new_conv[:, :4, :, :] = old_conv  # Copy pretrained channels
    nn.init.kaiming_normal_(new_conv[:, 4:, :, :], mode='fan_out', nonlinearity='leaky_relu')  # Random init rest

    # Step 6: Update the state dict with the new conv_in.weight
    new_sd["conv_in.weight"] = new_conv

    # Step 7: Copy compatible weights from the original model
    for key in new_sd:
        if key != "conv_in.weight" and key in orig_sd and new_sd[key].shape == orig_sd[key].shape:
            new_sd[key] = orig_sd[key]

    # Step 8: Load updated state dict into the new model
    unet_new.load_state_dict(new_sd)

    return unet_new.requires_grad_(True)
#--------------------------------------------------------------------------------------
def denoise_and_decode_in_one_step(batch_size, noise_pred, timesteps, zt, scheduler, vae, latent_scale, device, inference = False):
    
    """
    Denoise and decode the latent zt to obtain the predicted mask.
    """
    z0_hat_list   = []
    mask_hat_list = []
    
    if inference:
        noise_pred = noise_pred.to(device = 'cpu') 
        timesteps  = timesteps.to(device = 'cpu') 
        zt         = zt.to(device = 'cpu')
        
    for batch_idx in range(batch_size):
        z0_hat   = scheduler.step(noise_pred[batch_idx].unsqueeze(0), timesteps[batch_idx].unsqueeze(0), zt[batch_idx].unsqueeze(0)).pred_original_sample
        z0_hat   = z0_hat.to(device) if inference else z0_hat
        mask_hat = vae.decode(z0_hat / latent_scale).sample
        z0_hat_list.append(z0_hat)
        mask_hat_list.append(mask_hat)
    
    z0_hat   = torch.cat(z0_hat_list,   dim = 0) # (B, 4, 32, 32)
    mask_hat = torch.cat(mask_hat_list, dim = 0) # (B, 3, 256, 256)
    
    return z0_hat, mask_hat

#---------------------------------------------------------------------------------------
def denoise_and_decode(zt, scheduler, vae, latent_scale, device, unet, zc):
    """
    Denoise and decode the latent zt to obtain the predicted mask in multiple steps.
    Some parts of the computation are intentionally kept on CPU to save GPU memory.
    """

    # Initial latent input
    prev_sample = zt # .to("cpu")          # Start denoising on CPU
    # zc          = zc.to("cpu")          # Keep conditioning on CPU

    z0_hat = torch.cat([prev_sample, zc], dim=1)  # (B, 8, 32, 32)

    for idx in tqdm(range(0, do.INFERENCE_TIMESTEPS), desc="⏳ Denoising"):
        
        t = scheduler.timesteps[idx]  # Assumes B = 1
        # Run U-Net on GPU
        model_input = z0_hat.to(device)
        noise_pred  = unet(model_input, t.expand(1)).sample # .to(dtype=torch.float16, device="cpu")

        # Step through scheduler on CPU
        # prev_sample = scheduler.step(noise_pred, timestep.to("cpu"), prev_sample).prev_sample
        prev_t                      = max(1, t.item() - (1000 // do.INFERENCE_TIMESTEPS))  # line (223 - 227) in source code
        alpha_t                     = scheduler.alphas_cumprod[t.item()]
        alpha_t_prev                = scheduler.alphas_cumprod[prev_t]
        
        variance                    = get_variance(scheduler, t.item(), prev_t)
        stdev                       = ETA * variance.sqrt()
        
        predicted_x0                = (prev_sample - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
        
        var_noise                   = 0
        if VAR_NOISE:
            var_noise               = torch.randn_like(noise_pred, dtype = torch.float16, device = device) * stdev
        
        direction_pointing_to_xt    = (1 - alpha_t_prev - stdev ** 2).sqrt() * noise_pred
        prev_sample                 = alpha_t_prev.sqrt() * predicted_x0 + direction_pointing_to_xt + var_noise

        # Concatenate again for next step
        z0_hat = torch.cat([prev_sample, zc], dim=1)

    # Final decode via VAE (on CPU)
    mask_hat = vae.decode(prev_sample / latent_scale).sample
    
    if do.INFERENCE_TIMESTEPS > 1:
        return prev_sample, mask_hat
    else:
        return predicted_x0, mask_hat
#--------------------------------------------------------------------------------------#
def get_variance(scheduler, timestep, prev_timestep):
        alpha_prod_t        = scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev   = scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else scheduler.final_alpha_cumprod
        beta_prod_t         = 1 - alpha_prod_t
        beta_prod_t_prev    = 1 - alpha_prod_t_prev

        variance            = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

        return variance
#--------------------------------------------------------------------------------------#
def switch_to_ddim(device):
    """
    Replaces the current scheduler with a DDIM scheduler for faster inference.
    """
    scheduler = DDIMScheduler(num_train_timesteps=NUM_TRAIN_TIMESTEPS)
    scheduler.set_timesteps(do.INFERENCE_TIMESTEPS, device = device)
    
    print(f'\nSwitching to DDIM scheduler with inference timesteps: {do.INFERENCE_TIMESTEPS}\n')
    
    return scheduler

#----------------------------------------------------------------