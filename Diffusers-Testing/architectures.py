# ------------------------------------------------------------------------------#
#
# File name                 : architecture.py
# Purpose                   : Model definitions for Latent Diffusion Segmentor
# Usage                     : Used in both training and inference pipelines
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh, Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 202410001@lums.edu.pk, hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : April 28, 2025
# ------------------------------------------------------------------------------#

import                              torch, copy
import                              torch.nn as nn

from config                         import *
from diffusers                      import (AutoencoderKL, UNet2DModel, UNet2DConditionModel, DDPMScheduler, DDIMScheduler,)
from modeling_utils                 import *
from peft                           import get_peft_model, LoraConfig, TaskType

# ------------------------------------------------------------------------------#
class LDM_Segmentor(nn.Module):
    """
    Latent Diffusion Segmentor:
    - Frozen VAE (encoder & decoder)
    - Learnable image encoder (τ_θ)
    - Learnable UNet2DModel (in_channels=8)
    """

    def __init__(self,pretrained_vae: str = "CompVis/stable-diffusion-v1-4", scheduler_steps: int = 1000, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.latent_scale = 0.18215

        # Load pretrained VAE and freeze
        self.vae = (AutoencoderKL.from_pretrained(pretrained_vae, subfolder="vae").to(device).eval())
        for param in self.vae.parameters():
            param.requires_grad = False

        # Learnable image encoder (Tau θ)
        encoder            = copy.deepcopy(self.vae.encoder)
        self.image_encoder = TauEncoder(encoder).to(device).train()
        for param in self.image_encoder.parameters():
            param.requires_grad = True

        # UNet2DModel: receives zt || zc → predicts noise
        self.unet = UNet2DModel(**UNET_PARAMS).to(device).train()

        # Scheduler (set to DDPM by default)
        self.scheduler = DDPMScheduler(num_train_timesteps=scheduler_steps)
        
        # Loss Criterion (Custom class)
        self.loss_criterion = CombinedL1L2Loss(l1_weight = 1.0, l2_weight = 1.0, reduction = 'mean')

    def forward(self, image: torch.Tensor, mask: torch.Tensor = None, t: torch.Tensor = None):
        """
        Forward pass for both training and inference.

        Args:
            image (B, 3, 256, 256): Input RGB image
            mask  (B, 3, 256, 256): Binary mask (optional during inference)
            t     (B,)            : Diffusion timestep
            inference (bool)      : If True, assumes inference mode
            num_inference_steps   : # of DDIM steps during inference

        Returns:
            dict containing z0, zt, zc, z0_hat, noise_pred, mask_hat
        """

        with torch.no_grad():
            # Encode GT mask via frozen VAE encoder
            posterior = self.vae.encode(mask).latent_dist
            z0 = posterior.sample() * self.latent_scale

            # Step 2: Add noise to z0 → zt
            noise = torch.randn_like(z0)
            zt = self.scheduler.add_noise(z0, noise, t)

        # Step 3: Encode input image → zc
        zc = (self.image_encoder(image) * self.latent_scale)

         # Step 4: Concatenate and denoise followed by estimatation of z0_hat and decode to mask
        
        noise_hat        = self.unet(torch.cat([zt, zc], dim = 1), t).sample  # (B, 4, 32, 32), t is just (B,)
        z0_hat, mask_hat = denoise_and_decode_in_one_step(image.shape[0], noise_hat, t, zt, self.scheduler, 
                                            self.vae, self.latent_scale, self.device, False)
        
        out              = {"z0": z0, "zt": zt, "zc": zc, "z0_hat": z0_hat, "noise": noise, 'noise_hat': noise_hat}
        loss             = self.loss_criterion(noise_hat, noise, z0_hat, z0)

        return out, loss
        
    def inference(self, image, t):
        if not isinstance(self.scheduler, DDIMScheduler):
            self.scheduler = switch_to_ddim(self.device)
            
        zt = (torch.randn(1, 4, TRAINSIZE // 8, TRAINSIZE // 8, device=self.device, dtype=torch.float16)* self.latent_scale)
        zc = (self.image_encoder(image) * self.latent_scale)
        
        if do.ONE_X_ONE:
            noise_hat        = self.unet(torch.cat([zt, zc], dim = 1), t).sample
            z0_hat, mask_hat = denoise_and_decode_in_one_step(1, noise_hat, t, zt, self.scheduler, 
                                                              self.vae, self.latent_scale, self.device, True) 
        else:
            z0_hat, mask_hat = denoise_and_decode(t, zt, self.scheduler, self.vae, self.latent_scale, 
                                                  self.device, self.unet, zc)

        return {'z0_hat': z0_hat, 'mask_hat': mask_hat}
# ------------------------------------------------------------------------------#
class LDM_Segmentor_CrossAttention(nn.Module):
    """
    LDM-based segmentation model using cross-attention from UNet2DConditionModel.
    """

    def __init__(self, device="cuda", latent_scale=0.18215):
        super().__init__()
        self.device = torch.device(device)
        self.latent_scale = latent_scale

        # Projection to match Stable Diffusion's cross_attention_dim
        self.cross_attn_proj = nn.Linear(4, 768).to(self.device)

        # ------------------------------
        # Load pretrained VAE (Frozen)
        # ------------------------------
        self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(self.device)
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False

        # ------------------------------
        # Learnable Tau encoder
        # ------------------------------
        self.tau = TauEncoder(self.vae).to(self.device).train()

        # ------------------------------
        # Load pretrained U-Net (CrossAttention enabled)
        # ------------------------------
        self.unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet").to(self.device)
        self.unet.train()

        # ------------------------------
        # Scheduler (e.g., DDPM)
        # ------------------------------
        self.scheduler = DDPMScheduler(num_train_timesteps=NUM_TRAIN_TIMESTEPS)

    def forward(self, image: torch.Tensor, mask: torch.Tensor, t: torch.Tensor, inference: bool = False,):
        """
        Args:
            image (torch.Tensor): Input RGB image in [-1, 1], shape (B, 3, 256, 256)
            mask (torch.Tensor): Input GT mask in [-1, 1], shape (B, 3, 256, 256)
            t (torch.Tensor): Timesteps, shape (B,)
        Returns:
            dict: All intermediate latents and predicted mask.
        """
        B = image.shape[0]

        if inference:
            # Sample initial latent z0 from noise during inference
            B = image.shape[0]
            zt = (torch.randn(B, 4, TRAINSIZE // 8, TRAINSIZE // 8, device=self.device, dtype=torch.float16,)* self.latent_scale)
            if not isinstance(self.scheduler, DDIMScheduler):
                self.scheduler = switch_to_ddim(self.device)
        else:
            with torch.no_grad():
                # Step 1: Encode mask into latent z0 using frozen VAE
                z0 = self.vae.encode(mask).latent_dist.sample() * self.latent_scale

                # Step 2: Add noise to z0 using scheduler → zt
                noise = torch.randn_like(z0)
                zt = self.scheduler.add_noise(z0, noise, t)

        # Step 3: Encode image into conditioning vector using Tau encoder
        zc = self.tau(image)  # (B, 4, 32, 32)

        # Step 4: Reshape zc to (B, HW, C) for cross-attention
        B, C, H, W = zc.shape
        cross_attn = zc.view(B, C, -1).permute(0, 2, 1)  # (B, HW, 4)
        cross_attn = self.cross_attn_proj(cross_attn)  # (B, HW, 768)

        # Step 5: Predict noise residual using cross-attention U-Net
        noise_pred = self.unet(sample=zt, timestep=t, encoder_hidden_states=cross_attn).sample

        # Step 6: Estimate z0_hat and decode to mask
        z0_hat, mask_hat = denoise_and_decode(
            B, noise_pred, t, zt, self.scheduler, self.vae, self.latent_scale, self.device
        )
        
        return {"z0": z0, "zt": zt, "zc": zc, "z0_hat": z0_hat, "mask_hat": mask_hat}

# ------------------------------------------------------------------------------#
class LDM_Segmentor_Concatenation(nn.Module):
    """
    LDM-based segmentation model using concatenation conditioning.
    """

    def __init__(self, device="cuda", latent_scale=0.18215, num_inference_steps=1000):
        super().__init__()
        self.device = torch.device(device)
        self.latent_scale = latent_scale
        self.num_inference_steps = num_inference_steps

        # ------------------------------
        # Load pretrained VAE (Frozen)
        # ------------------------------
        self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(self.device)
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False

        # ------------------------------
        # Learnable Tau encoder (not used for concat but kept for API consistency)
        # ------------------------------
        self.tau = TauEncoder(self.vae).to(self.device)

        # ------------------------------
        # Load pretrained U-Net variant for concatenation (in_channels=8)
        # ------------------------------
        self.unet = load_hybrid_unet("CompVis/stable-diffusion-v1-4").to(self.device)
        self.unet.train()

        # ------------------------------
        # Scheduler (e.g., DDPM)
        # ------------------------------
        self.scheduler = DDPMScheduler(num_train_timesteps=self.num_inference_steps)

    def forward(self, image: torch.FloatTensor) -> torch.FloatTensor:
        """
        Inference for segmentation via concatenation conditioning.

        Args:
            image: Tensor of shape (B, C, H, W), values in [-1,1].
        Returns:
            Segmentation mask tensor of shape (B, C, H, W), values in [-1,1].
        """
        image = image.to(self.device)

        # Encode image to latents
        with torch.no_grad():
            enc_out = self.vae.encode(image)
            image_latents = (
                enc_out.latent_dist.sample() * self.latent_scale
            )  # (B,4,H',W')

        # Prepare noise latents
        noise = torch.randn_like(image_latents)

        # Denoising loop
        self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        lat = noise
        for t in self.scheduler.timesteps:
            # concat mask-latents and image-latents
            inp = torch.cat([lat, image_latents], dim=1)  # (B,8,H',W')
            out = self.unet(inp, t).sample
            # split mask latents
            mask_lat = out[:, : image_latents.size(1), :, :]
            lat = self.scheduler.step(mask_lat, t, lat).prev_sample

        # Decode mask latents
        mask_lat = lat / self.latent_scale
        decoded  = self.vae.decode(mask_lat).sample
        return decoded

# ------------------------------------------------------------------------------#
class LDM_Segmentor_LoRA(nn.Module):
    """
    LDM-based segmentation model with LoRA adapters on UNet's attention modules.
    """

    def __init__(
        self,
        device="cuda",
        latent_scale=0.18215,
        num_inference_steps=1000,
        lora_r=4,
        lora_alpha=16,
        target_modules=("to_q", "to_k", "to_v"),
    ):
        super().__init__()
        self.device = torch.device(device)
        self.latent_scale = latent_scale
        self.num_inference_steps = num_inference_steps

        # ------------------------------
        # Load pretrained VAE (Frozen)
        # ------------------------------
        self.vae = AutoencoderKL.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="vae"
        ).to(self.device)
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False

        # ------------------------------
        # Learnable Tau encoder
        # ------------------------------
        self.tau = TauEncoder(self.vae).to(self.device)

        # ------------------------------
        # Load pretrained U-Net and wrap with LoRA
        # ------------------------------
        base_unet = UNet2DConditionModel.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="unet"
        ).to(self.device)
        peft_config = LoraConfig(
            task_type=TaskType.DIFFUSERS_DENOISING,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
        )
        self.unet = get_peft_model(base_unet, peft_config)
        self.unet.train()

        # ------------------------------
        # Scheduler (e.g., DDPM)
        # ------------------------------
        self.scheduler = DDPMScheduler(num_train_timesteps=self.num_inference_steps)

    def forward(self, image: torch.FloatTensor) -> torch.FloatTensor:
        """
        Inference for segmentation via LoRA-adapted cross-attention.
        """
        image = image.to(self.device)

        # Encode image to latents
        with torch.no_grad():
            enc_out = self.vae.encode(image)
            latents = enc_out.latent_dist.sample() * self.latent_scale

        # Compute conditioning via TauEncoder
        cond = self.tau(latents)

        # Denoising loop
        self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        noisy = torch.randn_like(latents)
        lat = noisy
        for t in self.scheduler.timesteps:
            out = self.unet(lat, t, encoder_hidden_states=cond).sample
            lat = self.scheduler.step(out, t, lat).prev_sample

        # Decode mask latents
        lat = lat / self.latent_scale
        decoded = self.vae.decode(lat).sample
        return decoded


# ------------------------------------------------------------------------------#
class LDM_Segmentor_NoCrossAttention(nn.Module):
    """
    LDM-based segmentation model with cross-attention modules removed.
    """

    def __init__(self, device="cuda", latent_scale=0.18215, num_inference_steps=1000):
        super().__init__()
        self.device = torch.device(device)
        self.latent_scale = latent_scale
        self.num_inference_steps = num_inference_steps

        # ------------------------------
        # Load pretrained VAE (Frozen)
        # ------------------------------
        self.vae = AutoencoderKL.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="vae"
        ).to(self.device)
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False

        # ------------------------------
        # Learnable Tau encoder
        # ------------------------------
        self.tau = TauEncoder(self.vae).to(self.device)

        # ------------------------------
        # Load pretrained U-Net and strip cross-attention
        # ------------------------------
        base_unet = UNet2DConditionModel.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="unet"
        ).to(self.device)
        # Replace all CrossAttention modules with Identity
        for block in (
            list(base_unet.down_blocks)
            + [base_unet.mid_block]
            + list(base_unet.up_blocks)
        ):
            for name, module in block.named_children():
                if isinstance(module, DDIMScheduler):
                    setattr(block, name, nn.Identity())

        self.unet = base_unet.train()

        # ------------------------------
        # Scheduler (e.g., DDPM)
        # ------------------------------
        self.scheduler = DDPMScheduler(num_train_timesteps=self.num_inference_steps)

    def forward(self, image: torch.FloatTensor) -> torch.FloatTensor:
        """
        Inference for segmentation without cross-attention conditioning.
        """
        image = image.to(self.device)

        # Encode image to latents (only used to match API)
        with torch.no_grad():
            enc_out = self.vae.encode(image)
        latents_img = enc_out.latent_dist.sample() * self.latent_scale

        # Denoising loop with no conditioning
        self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        noisy = torch.randn_like(latents_img)
        lat = noisy
        for t in self.scheduler.timesteps:
            out = self.unet(lat, t, encoder_hidden_states=None).sample
            lat = self.scheduler.step(out, t, lat).prev_sample

        # Decode mask latents
        lat = lat / self.latent_scale
        decoded = self.vae.decode(lat).sample
        return decoded


# ------------------------------------------------------------------------------#