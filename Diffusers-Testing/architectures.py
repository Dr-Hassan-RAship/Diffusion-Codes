import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, UNet2DModel, DDPMScheduler
from torch import nn

class TauEncoder(nn.Module):
    """
    Learnable encoder for the input RGB image (tau_theta).
    Architecturally same as the VAE encoder but trainable.
    """
    def __init__(self, vae: AutoencoderKL):
        super().__init__()
        # Copy VAE encoder structure
        self.encoder = vae.encoder
        self.quant_conv = vae.quant_conv

    def forward(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        z = mean  # no sampling, deterministic
        return z

class LDM_Segmentor(nn.Module):
    def __init__(self, pretrained_vae="CompVis/stable-diffusion-v1-4", scheduler_steps=1000, device="cuda"):
        super().__init__()
        self.device = device

        # Load frozen VAE
        self.vae = AutoencoderKL.from_pretrained(pretrained_vae, subfolder="vae").eval().to(device)
        for p in self.vae.parameters():
            p.requires_grad = False

        # Learnable encoder for input image (τ_θ)
        self.image_encoder = TauEncoder(self.vae).to(device)

        # Diffusion U-Net (8 channels in: 4 noisy mask + 4 image encoding)
        self.unet = UNet2DModel(
            sample_size=32,
            in_channels=8,
            out_channels=4,
            layers_per_block=2,
            block_out_channels=(128, 256, 256, 512),
            down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D")
        ).to(device)

        # Scheduler (adds noise and steps)
        self.scheduler = DDPMScheduler(num_train_timesteps=scheduler_steps)

    def forward(self, image, mask, t):
        """
        Forward pass for training.
        image: (B, 3, 256, 256) → Input RGB image
        mask : (B, 3, 256, 256) → Binary mask (float in [-1, 1])
        t    : (B,)             → Timestep tensor for noise
        """
        # --- Step 1: Mask → VAE encoder (frozen)
        with torch.no_grad():
            posterior = self.vae.encode(mask).latent_dist
            z0 = posterior.sample() * 0.18215  # scaled latent # (B, 4, 32, 32)

        # --- Step 2: Add noise to z0 using scheduler → zt
        noise = torch.randn_like(z0)
        zt = self.scheduler.add_noise(z0, noise, t)

        # --- Step 3: Image → Tau encoder → zc
        zc = self.image_encoder(image) * 0.18215

        # --- Step 4: Concatenate and denoise
        zt_cat     = torch.cat([zt, zc], dim=1)  # (B, 8, 32, 32)
        noise_pred = self.unet(zt_cat, t).sample # (B, 4, 32, 32)

        # --- Step 5: Decode z0_hat to mask
        with torch.no_grad():
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
            "noise_pred": noise_pred,
            "z0_hat": z0_hat,
            "mask_hat": mask_hat
        }