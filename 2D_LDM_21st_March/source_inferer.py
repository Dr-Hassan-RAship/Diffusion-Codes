from __future__ import annotations

import math
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from functools import partial
from pydoc import locate
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.apps.utils import get_logger
from monai.data import decollate_batch
from monai.data.meta_tensor import MetaTensor
from monai.data.thread_buffer import ThreadBuffer
from monai.inferers.merger import AvgMerger, Merger
from monai.inferers.splitter import Splitter
from monai.inferers.utils import compute_importance_map, sliding_window_inference
from monai.networks.nets import (
    VQVAE,
    AutoencoderKL,
    ControlNet,
    DecoderOnlyTransformer,
    DiffusionModelUNet,
    SPADEAutoencoderKL,
    SPADEDiffusionModelUNet,
)
from monai.networks.schedulers import Scheduler
from monai.transforms import CenterSpatialCrop, SpatialPad
from monai.utils import BlendMode, Ordering, PatchKeys, PytorchPadMode, ensure_tuple, optional_import
from monai.visualize import CAM, GradCAM, GradCAMpp

tqdm, has_tqdm = optional_import("tqdm", name="tqdm")

class Inferer(ABC):
    """
    A base class for model inference.
    Extend this class to support operations during inference, e.g. a sliding window method.

    Example code::

        device = torch.device("cuda:0")
        transform = Compose([ToTensor(), LoadImage(image_only=True)])
        data = transform(img_path).to(device)
        model = UNet(...).to(device)
        inferer = SlidingWindowInferer(...)

        model.eval()
        with torch.no_grad():
            pred = inferer(inputs=data, network=model)
        ...

    """

    @abstractmethod
    def __call__(self, inputs: torch.Tensor, network: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Run inference on `inputs` with the `network` model.

        Args:
            inputs: input of the model inference.
            network: model for inference.
            args: optional args to be passed to ``network``.
            kwargs: optional keyword args to be passed to ``network``.

        Raises:
            NotImplementedError: When the subclass does not override this method.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

class DiffusionInferer(Inferer):
    """
    DiffusionInferer takes a trained diffusion model and a scheduler and can be used to perform a signal forward pass
    for a training iteration, and sample from the model.

    Args:
        scheduler: diffusion scheduler.
    """

    def __init__(self, scheduler: Scheduler) -> None:  # type: ignore[override]
        super().__init__()

        self.scheduler = scheduler

    def __call__(  # type: ignore[override]
        self,
        inputs: torch.Tensor,
        diffusion_model: DiffusionModelUNet,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor | None = None,
        mode: str = "crossattn",
        seg: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Implements the forward pass for a supervised training iteration.

        Args:
            inputs: Input image to which noise is added.
            diffusion_model: diffusion model.
            noise: random noise, of the same shape as the input.
            timesteps: random timesteps.
            condition: Conditioning for network input.
            mode: Conditioning mode for the network.
            seg: if model is instance of SPADEDiffusionModelUnet, segmentation must be
            provided on the forward (for SPADE-like AE or SPADE-like DM)
        """
        if mode not in ["crossattn", "concat"]:
            raise NotImplementedError(f"{mode} condition is not supported")
        
        # Batch size = 12 --> 12 uniformally chosen timesteps for each image and their corresponding forward noise
        noisy_image: torch.Tensor = self.scheduler.add_noise(original_samples=inputs, noise=noise, timesteps=timesteps) # assume this is for now eq (1) of sdseg paper
        # z_t                       = noisy_image.clone()
        if mode == "concat":
            if condition is None:
                raise ValueError("Conditioning is required for concat condition")
            else:
                noisy_image = torch.cat([noisy_image, condition], dim=1) # concatenated z_t and z_c
                condition = None
        diffusion_model = (
            partial(diffusion_model, seg=seg)
            if isinstance(diffusion_model, SPADEDiffusionModelUNet)
            else diffusion_model
        )
        # I am going to assume this  prediction is z_o^tilde (source: trust me)
        prediction: torch.Tensor = diffusion_model(x=noisy_image, timesteps=timesteps, context=condition)
        
        # 
        # random_timestep chosen = random.randint(timesteps)
        # prediction is of the same shape as noisy_image i.e. (B, C, H, W), 
        return prediction # which is n_tilde

    @torch.no_grad()
    def sample(
        self,
        input_noise: torch.Tensor,
        diffusion_model: DiffusionModelUNet,
        scheduler: Scheduler | None = None,
        save_intermediates: bool | None = False,
        intermediate_steps: int | None = 100,
        conditioning: torch.Tensor | None = None,
        mode: str = "crossattn",
        verbose: bool = True,
        seg: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            input_noise: random noise, of the same shape as the desired sample.
            diffusion_model: model to sample from.
            scheduler: diffusion scheduler. If none provided will use the class attribute scheduler
            save_intermediates: whether to return intermediates along the sampling change
            intermediate_steps: if save_intermediates is True, saves every n steps
            conditioning: Conditioning for network input.
            mode: Conditioning mode for the network.
            verbose: if true, prints the progression bar of the sampling process.
            seg: if diffusion model is instance of SPADEDiffusionModel, segmentation must be provided.
        """
        if mode not in ["crossattn", "concat"]:
            raise NotImplementedError(f"{mode} condition is not supported")
        if mode == "concat" and conditioning is None:
            raise ValueError("Conditioning must be supplied for if condition mode is concat.")
        if not scheduler:
            scheduler = self.scheduler
        image = input_noise
        if verbose and has_tqdm:
            progress_bar = tqdm(scheduler.timesteps)
        else:
            progress_bar = iter(scheduler.timesteps)
        intermediates = []
        for t in progress_bar:
            # 1. predict noise model_output
            diffusion_model = (
                partial(diffusion_model, seg=seg)
                if isinstance(diffusion_model, SPADEDiffusionModelUNet)
                else diffusion_model
            )
            if mode == "concat" and conditioning is not None:
                model_input = torch.cat([image, conditioning], dim=1)
                model_output = diffusion_model(
                    model_input, timesteps=torch.Tensor((t,)).to(input_noise.device), context=None
                )
                # model output is noise meaning diffusion model unet always gives noise
            else:
                model_output = diffusion_model(
                    image, timesteps=torch.Tensor((t,)).to(input_noise.device), context=conditioning
                )

            # 2. compute previous image: x_t -> x_t-1
            image, _ = scheduler.step(model_output, t, image) # image corresponds to z_t
            # image is z_t-1, and noise_pred is n_tilde
            if save_intermediates and t % intermediate_steps == 0:
                intermediates.append(image)
        if save_intermediates:
            return image, intermediates, model_output
        else:
            return image, model_output
    @torch.no_grad()
    def get_likelihood(
        self,
        inputs: torch.Tensor,
        diffusion_model: DiffusionModelUNet,
        scheduler: Scheduler | None = None,
        save_intermediates: bool | None = False,
        conditioning: torch.Tensor | None = None,
        mode: str = "crossattn",
        original_input_range: tuple = (0, 255),
        scaled_input_range: tuple = (0, 1),
        verbose: bool = True,
        seg: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Computes the log-likelihoods for an input.

        Args:
            inputs: input images, NxCxHxW[xD]
            diffusion_model: model to compute likelihood from
            scheduler: diffusion scheduler. If none provided will use the class attribute scheduler.
            save_intermediates: save the intermediate spatial KL maps
            conditioning: Conditioning for network input.
            mode: Conditioning mode for the network.
            original_input_range: the [min,max] intensity range of the input data before any scaling was applied.
            scaled_input_range: the [min,max] intensity range of the input data after scaling.
            verbose: if true, prints the progression bar of the sampling process.
            seg: if diffusion model is instance of SPADEDiffusionModel, segmentation must be provided.
        """

        if not scheduler:
            scheduler = self.scheduler
        if scheduler._get_name() != "DDPMScheduler":
            raise NotImplementedError(
                f"Likelihood computation is only compatible with DDPMScheduler,"
                f" you are using {scheduler._get_name()}"
            )
        if mode not in ["crossattn", "concat"]:
            raise NotImplementedError(f"{mode} condition is not supported")
        if mode == "concat" and conditioning is None:
            raise ValueError("Conditioning must be supplied for if condition mode is concat.")
        if verbose and has_tqdm:
            progress_bar = tqdm(scheduler.timesteps)
        else:
            progress_bar = iter(scheduler.timesteps)
        intermediates = []
        noise = torch.randn_like(inputs).to(inputs.device)
        total_kl = torch.zeros(inputs.shape[0]).to(inputs.device)
        for t in progress_bar:
            timesteps = torch.full(inputs.shape[:1], t, device=inputs.device).long()
            noisy_image = self.scheduler.add_noise(original_samples=inputs, noise=noise, timesteps=timesteps)
            diffusion_model = (
                partial(diffusion_model, seg=seg)
                if isinstance(diffusion_model, SPADEDiffusionModelUNet)
                else diffusion_model
            )
            if mode == "concat" and conditioning is not None:
                noisy_image = torch.cat([noisy_image, conditioning], dim=1)
                model_output = diffusion_model(noisy_image, timesteps=timesteps, context=None)
            else:
                model_output = diffusion_model(x=noisy_image, timesteps=timesteps, context=conditioning)

            # get the model's predicted mean,  and variance if it is predicted
            if model_output.shape[1] == inputs.shape[1] * 2 and scheduler.variance_type in ["learned", "learned_range"]:
                model_output, predicted_variance = torch.split(model_output, inputs.shape[1], dim=1)
            else:
                predicted_variance = None

            # 1. compute alphas, betas
            alpha_prod_t = scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = scheduler.alphas_cumprod[t - 1] if t > 0 else scheduler.one
            beta_prod_t = 1 - alpha_prod_t
            beta_prod_t_prev = 1 - alpha_prod_t_prev

            # 2. compute predicted original sample from predicted noise also called
            # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
            if scheduler.prediction_type == "epsilon":
                pred_original_sample = (noisy_image - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            elif scheduler.prediction_type == "sample":
                pred_original_sample = model_output
            elif scheduler.prediction_type == "v_prediction":
                pred_original_sample = (alpha_prod_t**0.5) * noisy_image - (beta_prod_t**0.5) * model_output
            # 3. Clip "predicted x_0"
            if scheduler.clip_sample:
                pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

            # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
            # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
            pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * scheduler.betas[t]) / beta_prod_t
            current_sample_coeff = scheduler.alphas[t] ** (0.5) * beta_prod_t_prev / beta_prod_t

            # 5. Compute predicted previous sample µ_t
            # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
            predicted_mean = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * noisy_image

            # get the posterior mean and variance
            posterior_mean = scheduler._get_mean(timestep=t, x_0=inputs, x_t=noisy_image)
            posterior_variance = scheduler._get_variance(timestep=t, predicted_variance=predicted_variance)

            log_posterior_variance = torch.log(posterior_variance)
            log_predicted_variance = torch.log(predicted_variance) if predicted_variance else log_posterior_variance

            if t == 0:
                # compute -log p(x_0|x_1)
                kl = -self._get_decoder_log_likelihood(
                    inputs=inputs,
                    means=predicted_mean,
                    log_scales=0.5 * log_predicted_variance,
                    original_input_range=original_input_range,
                    scaled_input_range=scaled_input_range,
                )
            else:
                # compute kl between two normals
                kl = 0.5 * (
                    -1.0
                    + log_predicted_variance
                    - log_posterior_variance
                    + torch.exp(log_posterior_variance - log_predicted_variance)
                    + ((posterior_mean - predicted_mean) ** 2) * torch.exp(-log_predicted_variance)
                )
            total_kl += kl.view(kl.shape[0], -1).mean(dim=1)
            if save_intermediates:
                intermediates.append(kl.cpu())

        if save_intermediates:
            return total_kl, intermediates
        else:
            return total_kl

    def _approx_standard_normal_cdf(self, x):
        """
        A fast approximation of the cumulative distribution function of the
        standard normal. Code adapted from https://github.com/openai/improved-diffusion.
        """

        return 0.5 * (
            1.0 + torch.tanh(torch.sqrt(torch.Tensor([2.0 / math.pi]).to(x.device)) * (x + 0.044715 * torch.pow(x, 3)))
        )

    def _get_decoder_log_likelihood(
        self,
        inputs: torch.Tensor,
        means: torch.Tensor,
        log_scales: torch.Tensor,
        original_input_range: tuple = (0, 255),
        scaled_input_range: tuple = (0, 1),
    ) -> torch.Tensor:
        """
        Compute the log-likelihood of a Gaussian distribution discretizing to a
        given image. Code adapted from https://github.com/openai/improved-diffusion.

        Args:
            input: the target images. It is assumed that this was uint8 values,
                      rescaled to the range [-1, 1].
            means: the Gaussian mean Tensor.
            log_scales: the Gaussian log stddev Tensor.
            original_input_range: the [min,max] intensity range of the input data before any scaling was applied.
            scaled_input_range: the [min,max] intensity range of the input data after scaling.
        """
        if inputs.shape != means.shape:
            raise ValueError(f"Inputs and means must have the same shape, got {inputs.shape} and {means.shape}")
        bin_width = (scaled_input_range[1] - scaled_input_range[0]) / (
            original_input_range[1] - original_input_range[0]
        )
        centered_x = inputs - means
        inv_stdv = torch.exp(-log_scales)
        plus_in = inv_stdv * (centered_x + bin_width / 2)
        cdf_plus = self._approx_standard_normal_cdf(plus_in)
        min_in = inv_stdv * (centered_x - bin_width / 2)
        cdf_min = self._approx_standard_normal_cdf(min_in)
        log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
        log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
        cdf_delta = cdf_plus - cdf_min
        log_probs = torch.where(
            inputs < -0.999,
            log_cdf_plus,
            torch.where(inputs > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
        )
        return log_probs


class LatentDiffusionInferer(DiffusionInferer):
    """
    LatentDiffusionInferer takes a stage 1 model (VQVAE or AutoencoderKL), diffusion model, and a scheduler, and can
    be used to perform a signal forward pass for a training iteration, and sample from the model.

    Args:
        scheduler: a scheduler to be used in combination with `unet` to denoise the encoded image latents.
        scale_factor: scale factor to multiply the values of the latent representation before processing it by the
            second stage.
        ldm_latent_shape: desired spatial latent space shape. Used if there is a difference in the autoencoder model's latent shape.
        autoencoder_latent_shape:  autoencoder_latent_shape: autoencoder spatial latent space shape. Used if there is a
             difference between the autoencoder's latent shape and the DM shape.
    """

    def __init__(
        self,
        scheduler: Scheduler,
        scale_factor: float = 1.0,
        ldm_latent_shape: list | None = None,
        autoencoder_latent_shape: list | None = None,
    ) -> None:
        super().__init__(scheduler=scheduler)
        self.scale_factor = scale_factor
        if (ldm_latent_shape is None) ^ (autoencoder_latent_shape is None):
            raise ValueError("If ldm_latent_shape is None, autoencoder_latent_shape must be None, and vice versa.")
        self.ldm_latent_shape = ldm_latent_shape
        self.autoencoder_latent_shape = autoencoder_latent_shape
        if self.ldm_latent_shape is not None and self.autoencoder_latent_shape is not None:
            self.ldm_resizer = SpatialPad(spatial_size=self.ldm_latent_shape)
            self.autoencoder_resizer = CenterSpatialCrop(roi_size=self.autoencoder_latent_shape)

    def __call__(  # type: ignore[override]
        self,
        inputs: torch.Tensor,
        autoencoder_model: AutoencoderKL | VQVAE,
        diffusion_model: DiffusionModelUNet,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor | None = None,
        mode: str = "crossattn",
        seg: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Implements the forward pass for a supervised training iteration.

        Args:
            inputs: input image to which the latent representation will be extracted and noise is added.
            autoencoder_model: first stage model.
            diffusion_model: diffusion model.
            noise: random noise, of the same shape as the latent representation.
            timesteps: random timesteps.
            condition: conditioning for network input.
            mode: Conditioning mode for the network.
            seg: if diffusion model is instance of SPADEDiffusionModel, segmentation must be provided.
        """
        with torch.no_grad():
            latent = autoencoder_model.encode_stage_2_inputs(inputs) * self.scale_factor # latent_masks

        if self.ldm_latent_shape is not None:
            latent = torch.stack([self.ldm_resizer(i) for i in decollate_batch(latent)], 0)

        prediction: torch.Tensor = super().__call__(
            inputs=latent,
            diffusion_model=diffusion_model,
            noise=noise,
            timesteps=timesteps,
            condition=condition,
            mode=mode,
            seg=seg,
        )
        return prediction

    @torch.no_grad()
    def sample(  # type: ignore[override]
        self,
        input_noise: torch.Tensor,
        autoencoder_model: AutoencoderKL | VQVAE,
        diffusion_model: DiffusionModelUNet,
        scheduler: Scheduler | None = None,
        save_intermediates: bool | None = False,
        intermediate_steps: int | None = 100,
        conditioning: torch.Tensor | None = None,
        mode: str = "crossattn",
        verbose: bool = True,
        seg: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            input_noise: random noise, of the same shape as the desired latent representation.
            autoencoder_model: first stage model.
            diffusion_model: model to sample from.
            scheduler: diffusion scheduler. If none provided will use the class attribute scheduler.
            save_intermediates: whether to return intermediates along the sampling change
            intermediate_steps: if save_intermediates is True, saves every n steps
            conditioning: Conditioning for network input.
            mode: Conditioning mode for the network.
            verbose: if true, prints the progression bar of the sampling process.
            seg: if diffusion model is instance of SPADEDiffusionModel, or autoencoder_model
             is instance of SPADEAutoencoderKL, segmentation must be provided.
        """

        if (
            isinstance(autoencoder_model, SPADEAutoencoderKL)
            and isinstance(diffusion_model, SPADEDiffusionModelUNet)
            and autoencoder_model.decoder.label_nc != diffusion_model.label_nc
        ):
            raise ValueError(
                f"If both autoencoder_model and diffusion_model implement SPADE, the number of semantic"
                f"labels for each must be compatible, but got {autoencoder_model.decoder.label_nc} and"
                f"{diffusion_model.label_nc}"
            )
        
        # we have gotten now z_o_tilde and n_tilde
        outputs, noise_pred = super().sample(
            input_noise=input_noise,
            diffusion_model=diffusion_model,
            scheduler=scheduler,
            save_intermediates=save_intermediates,
            intermediate_steps=intermediate_steps,
            conditioning=conditioning,
            mode=mode,
            verbose=verbose,
            seg=seg,
        )

        if save_intermediates:
            latent, latent_intermediates = outputs
        else:
            latent = outputs

        if self.autoencoder_latent_shape is not None:
            latent = torch.stack([self.autoencoder_resizer(i) for i in decollate_batch(latent)], 0)
            latent_intermediates = [
                torch.stack([self.autoencoder_resizer(i) for i in decollate_batch(l)], 0) for l in latent_intermediates
            ]

        decode = autoencoder_model.decode_stage_2_outputs
        if isinstance(autoencoder_model, SPADEAutoencoderKL):
            decode = partial(autoencoder_model.decode_stage_2_outputs, seg=seg)
        image = decode(latent / self.scale_factor)

        if save_intermediates:
            intermediates = []
            for latent_intermediate in latent_intermediates:
                decode = autoencoder_model.decode_stage_2_outputs
                if isinstance(autoencoder_model, SPADEAutoencoderKL):
                    decode = partial(autoencoder_model.decode_stage_2_outputs, seg=seg)
                intermediates.append(decode(latent_intermediate / self.scale_factor))
            return image, intermediates, noise_pred # return n_tilde along with z_o_tilde

        else:
            return image, noise_pred

    @torch.no_grad()
    def get_likelihood(  # type: ignore[override]
        self,
        inputs: torch.Tensor,
        autoencoder_model: AutoencoderKL | VQVAE,
        diffusion_model: DiffusionModelUNet,
        scheduler: Scheduler | None = None,
        save_intermediates: bool | None = False,
        conditioning: torch.Tensor | None = None,
        mode: str = "crossattn",
        original_input_range: tuple | None = (0, 255),
        scaled_input_range: tuple | None = (0, 1),
        verbose: bool = True,
        resample_latent_likelihoods: bool = False,
        resample_interpolation_mode: str = "nearest",
        seg: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Computes the log-likelihoods of the latent representations of the input.

        Args:
            inputs: input images, NxCxHxW[xD]
            autoencoder_model: first stage model.
            diffusion_model: model to compute likelihood from
            scheduler: diffusion scheduler. If none provided will use the class attribute scheduler
            save_intermediates: save the intermediate spatial KL maps
            conditioning: Conditioning for network input.
            mode: Conditioning mode for the network.
            original_input_range: the [min,max] intensity range of the input data before any scaling was applied.
            scaled_input_range: the [min,max] intensity range of the input data after scaling.
            verbose: if true, prints the progression bar of the sampling process.
            resample_latent_likelihoods: if true, resamples the intermediate likelihood maps to have the same spatial
                dimension as the input images.
            resample_interpolation_mode: if use resample_latent_likelihoods, select interpolation 'nearest', 'bilinear',
                or 'trilinear;
            seg: if diffusion model is instance of SPADEDiffusionModel, or autoencoder_model
             is instance of SPADEAutoencoderKL, segmentation must be provided.
        """
        if resample_latent_likelihoods and resample_interpolation_mode not in ("nearest", "bilinear", "trilinear"):
            raise ValueError(
                f"resample_interpolation mode should be either nearest, bilinear, or trilinear, got {resample_interpolation_mode}"
            )
        latents = autoencoder_model.encode_stage_2_inputs(inputs) * self.scale_factor

        if self.ldm_latent_shape is not None:
            latents = torch.stack([self.ldm_resizer(i) for i in decollate_batch(latents)], 0)

        outputs = super().get_likelihood(
            inputs=latents,
            diffusion_model=diffusion_model,
            scheduler=scheduler,
            save_intermediates=save_intermediates,
            conditioning=conditioning,
            mode=mode,
            verbose=verbose,
            seg=seg,
        )

        if save_intermediates and resample_latent_likelihoods:
            intermediates = outputs[1]
            resizer = nn.Upsample(size=inputs.shape[2:], mode=resample_interpolation_mode)
            intermediates = [resizer(x) for x in intermediates]
            outputs = (outputs[0], intermediates)
        return outputs
