import torch, os
import copy
import torch.nn as nn
from   diffusers import AutoencoderKL, UNet2DConditionModel, UNet2DModel, DDPMScheduler, DDIMScheduler
from   config import *
from   safetensors.torch import save_model as save_safetensors
from   safetensors.torch import load_file as load_safetensors
from   glob import glob
from   torch.optim import AdamW
from tqdm.auto import tqdm

#--------------------------------------------------------------------------------------
class TauEncoder(nn.Module):
    """
    Learnable encoder for the input RGB image (τ_θ).
    Architecturally same as the VAE encoder but trainable.
    """
    def __init__(self, vae: AutoencoderKL):
        super().__init__()
        self.vae = vae

    def forward(self, x):
        latent_dist = self.vae.encode(x).latent_dist
        return latent_dist.mean if DETERMINISTIC else latent_dist.sample()
    
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

    return torch.cat(z0_hat_list, dim = 0), torch.cat(mask_hat_list, dim = 0) # (B, 4, 32, 32), (B, 3, 256, 256)

#---------------------------------------------------------------------------------------
def denoise_and_decode(noise_pred, timesteps, zt, scheduler, vae, latent_scale, device, 
                       unet, zc):
    """
    Denoise and decode the latent zt to obtain the predicted mask in multiple steps
    """

    z0_hat     = torch.cat([zt, zc], dim=1)
    for t in tqdm(timesteps.to(device  = device)):
        print(f'Timestep: {t}')        
        noise_pred  = unet(z0_hat, t.expand(1)).sample.to(dtype = torch.float16, device = 'cpu') # (1, 4, 32, 32), t.shape = (1,)
        prev_sample = scheduler.step(noise_pred, t.expand(1).to(device = 'cpu'), zt.to(device = 'cpu')).prev_sample.to(device)
        z0_hat      = torch.cat([prev_sample, zc], dim=1)
        
    return z0_hat, vae.decode(prev_sample / latent_scale).sample
#--------------------------------------------------------------------------------------#
def switch_to_ddim(device):
    """
    Replaces the current scheduler with a DDIM scheduler for faster inference.
    """
    scheduler = DDIMScheduler(num_train_timesteps=NUM_TRAIN_TIMESTEPS)
    scheduler.set_timesteps(do.INFERENCE_TIMESTEPS, device = device)
    
    print(f'Switching to DDIM scheduler with inference timesteps: {do.INFERENCE_TIMESTEPS}')
    
    return scheduler

#----------------------------------------------------------------