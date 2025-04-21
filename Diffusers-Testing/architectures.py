import torch
from diffusers import AutoencoderKL, UNet2DModel, DDPMScheduler
from torch import nn


class TauEncoder(nn.Module):
    """
    Learnable encoder for the input RGB image (τ_θ).
    Architecturally same as the VAE encoder but trainable.
    """
    def __init__(self, vae: AutoencoderKL):
        super().__init__()
        self.encoder = vae.encoder
        self.quant_conv = vae.quant_conv

    def forward(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        return mean  # Deterministic z_c


class LDM_Segmentor(nn.Module):
    """
    Latent Diffusion Segmentor:
    - Frozen VAE (encoder & decoder)
    - Learnable image encoder (τ_θ)
    - Learnable UNet2DModel (in_channels=8)
    """

    def __init__(self, pretrained_vae="CompVis/stable-diffusion-v1-4", scheduler_steps=1000, device="cuda"):
        super().__init__()
        self.device = device
        self.latent_scale = 0.18215

        # -- Load pretrained VAE and freeze
        self.vae = AutoencoderKL.from_pretrained(pretrained_vae, subfolder="vae").to(device).eval()
        for p in self.vae.parameters():
            p.requires_grad = False

        # -- Learnable Tau encoder
        self.image_encoder = TauEncoder(self.vae).to(device).requires_grad_()

        # -- UNet2DModel: takes (zt || zc) → predicts noise
        self.unet = UNet2DModel(
            sample_size=32,
            in_channels=8,
            out_channels=4,
            layers_per_block=2,
            block_out_channels=(128, 256, 256, 512),
            down_block_types=("DownBlock2D",) * 4,
            up_block_types=("UpBlock2D",) * 4
        ).to(device).train()

        # -- Scheduler
        self.scheduler = DDPMScheduler(num_train_timesteps=scheduler_steps)

    def forward(self, image, mask, t):
        """
        Forward pass for segmentation.
        Args:
            image: (B, 3, 256, 256)  → input image
            mask : (B, 3, 256, 256)  → binary mask
            t    : (B,)              → timesteps
        Returns:
            dict of intermediate latents and final predicted mask
        """
        # -- Encode mask using frozen VAE encoder
        posterior = self.vae.encode(mask).latent_dist
        z0 = posterior.sample() * self.latent_scale  # (B, 4, 32, 32)

        # -- Add noise to get zt
        noise = torch.randn_like(z0)
        zt = self.scheduler.add_noise(z0, noise, t)

        # -- Encode image using Tau encoder
        zc = self.image_encoder(image) * self.latent_scale  # (B, 4, 32, 32)

        # -- Concatenate zt and zc → pass to UNet
        zt_cat = torch.cat([zt, zc], dim=1)  # (B, 8, 32, 32)
        noise_pred = self.unet(zt_cat, t).sample  # Predict residual noise

        # --- Decode z0_hat to mask
        z0_hat_list = []
        mask_hat_list = []

        for batch_idx in range(image.shape[0]):
            z0_hat   = self.scheduler.step(noise_pred[batch_idx].unsqueeze(0), t[batch_idx].unsqueeze(0), zt[batch_idx].unsqueeze(0)).pred_original_sample
            mask_hat = self.vae.decode(z0_hat / 0.18215).sample
            z0_hat_list.append(z0_hat)
            mask_hat_list.append(mask_hat)

        z0_hat = torch.cat(z0_hat_list, dim=0)  # (B, 4, 32, 32)
        mask_hat = torch.cat(mask_hat_list, dim=0)


        return {
            "z0": z0,
            "zt": zt,
            "zc": zc,
            "z0_hat": z0_hat,
            "noise_pred": noise_pred,
            "mask_hat": mask_hat,
        }
