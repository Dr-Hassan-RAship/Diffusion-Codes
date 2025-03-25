# ------------------------------------------------------------------------------#
#
# File name                 : inference_ldm_utils.py
# Purpose                   : Helper functions for inference of Latent Diffusion Model (LDM)
# Usage                     : Used by inference_ldm.py
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh, Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 202410001@lums.edu.pk, hassan.mohyuddin@lums.edu.pk
#
# Last Date                 : March 11, 2025
#
# ------------------------------------------------------------------------------#
import os, csv, torch

import numpy                                        as np
import pandas                                       as pd
import nibabel                                      as nib
import matplotlib.pyplot                            as plt
import matplotlib

matplotlib.use("Agg")

from medpy                                          import metric
from monai.metrics                                  import MeanIoU
from torch.amp                                      import autocast
from monai.networks.nets                            import AutoencoderKL
from config_ldm_ddpm                                import *
from generative.networks.schedulers                 import DDPMScheduler, DDIMScheduler
from generative.inferers                            import LatentDiffusionInferer
from generative.networks.nets.diffusion_model_unet  import DiffusionModelUNet

# ------------------------------------------------------------------------------#


def check_or_create_folder(folder):
    """Check if folder exists, if not, create it."""
    if not os.path.exists(folder):
        os.makedirs(folder)

# ------------------------------------------------------------------------------#
def save_nifti(filename, savepath, data, affine=None, header=None):
    """Save data as a NIfTI file."""
    save_path = os.path.join(savepath, f"{filename}.nii.gz")
    check_or_create_folder(os.path.dirname(save_path))
    nifti_image = nib.Nifti1Image(data.astype(np.uint8), affine, header=header)
    nib.save(nifti_image, save_path)

# ------------------------------------------------------------------------------#
def load_autoencoder(device, train_loader, image=True, mask=True, epoch_mask = 500, epoch_image = 500):
    """Load a pre-trained autoencoder model (for image and mask) and get scale factor."""
    tup_image, tup_mask = None, None
    if image:
        # if epoch_image == N_EPOCHS: joint_dir = f'final_model.pth'
        # else                      : joint_dir = f'autoencoder_epoch_{epoch_image}.pth'
        joint_dir = f'autoencoder_epoch_{epoch_image}.pth'
        
        autoencoder_path = os.path.join(
            DAE_IMAGE_SNAPSHOT_DIR + "/models", joint_dir
        )
        autoencoder_params = DAE_IMAGE_PARAMS

        dae_image = AutoencoderKL(**autoencoder_params)
        dae_image.load_state_dict(
            torch.load(autoencoder_path, map_location=device, weights_only=True)
        )
        dae_image.to(device)
        dae_image.eval()

        sample_data = next(iter(train_loader))  # Get scale factor of latent space

        with torch.no_grad():
            with autocast("cuda", enabled=True):
                z = dae_image.encode_stage_2_inputs(
                    sample_data["clean_image"].to(device)
                )

            print(f"Loaded Image AutoencoderKL from {autoencoder_path}")

            z           = LDM_SCALE_FACTOR / torch.std(z)
            tup_image   = (dae_image, z)

    if mask:
        if epoch_mask == N_EPOCHS: joint_dir = f'final_model.pth'
        else                     : joint_dir = f'autoencoder_epoch_{epoch_mask}.pth'
        
        autoencoder_path = os.path.join(
            DAE_MASK_SNAPSHOT_DIR + "/models", joint_dir
        )
        autoencoder_params = DAE_MASK_PARAMS

        dae_mask = AutoencoderKL(**autoencoder_params)
        dae_mask.load_state_dict(
            torch.load(autoencoder_path, map_location=device, weights_only=True)
        )
        dae_mask.to(device)
        dae_mask.eval()

        sample_data = next(iter(train_loader))  # Get scale factor of latent space

        with torch.no_grad():
            with autocast("cuda", enabled=True):
                z = dae_mask.encode_stage_2_inputs(sample_data["clean_mask"].to(device))

        print(f"Loaded Mask AutoencoderKL from {autoencoder_path}")
        z           = LDM_SCALE_FACTOR / torch.std(z)
        tup_mask    = (dae_mask, z)

    # return tup_image only if image == True and tup_mask only if mask == True and
    # return both if both are true
    
    if image and not mask:
        return tup_image
    elif mask and not image:
        return tup_mask
    else:
        return tup_image, tup_mask

# ------------------------------------------------------------------------------#
def load_ldm_model(device, scale_factor):
    """Load the trained LDM model and scheduler."""
    model_path = (
        os.path.join(LDM_SNAPSHOT_DIR, "models", f"model_epoch_{do.MODEL_EPOCH}.pth")
        if do.MODEL_EPOCH != -1
        else os.path.join(LDM_SNAPSHOT_DIR, "models", "final_model.pth")
    )
    model_params = MODEL_PARAMS

    unet = DiffusionModelUNet(**model_params)
    unet.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    unet.to(device)
    unet.eval()

    scheduler = (
        DDIMScheduler(num_train_timesteps=do.TRAIN_TIMESTEPS, schedule=NOISE_SCHEDULER)
        if SCHEDULER == "DDIM"
        else DDPMScheduler(
            num_train_timesteps=do.TRAIN_TIMESTEPS, schedule=NOISE_SCHEDULER
        )
    )
    inferer = LatentDiffusionInferer(scheduler=scheduler, scale_factor=scale_factor)
    print(f"Loaded LDM model from {model_path} with {SCHEDULER} scheduler.")
    return unet, scheduler, inferer

# ------------------------------------------------------------------------------#
def calculate_metrics(prediction, label):
    """
    For calculating metrics such as dice, hd95, assd and miou.
    """
    miou_metric = MeanIoU()
    pred = prediction > 0
    gt = label > 0

    if (pred.sum() > 0) and (gt.sum() > 0):  # Main calculation
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        assd = metric.binary.assd(pred, gt)
        miou = miou_metric(torch.tensor(pred).unsqueeze(0).unsqueeze(0), torch.tensor(gt).unsqueeze(0).unsqueeze(0)).item()

    elif (pred.sum() == 0) and (gt.sum() == 0):  # Safeguard no 2
        dice = 1
        hd95 = 0
        assd = 0
        miou = 1

    else:
        dice = 0
        hd95 = np.inf
        assd = np.inf
        miou = 0

    return dice, hd95, assd, miou

# ------------------------------------------------------------------------------#
def save_groundtruth_image(image, save_folder, filename, mode="image"):
    """Save ground truth image as a JPEG file."""
    image = np.squeeze(image)
    filepath = os.path.join(save_folder, filename)
    if mode == "image":
        # tranpose image to (256, 256, 3) and normalize between 0 - 1 for each channel independantly for imsave
        image = np.transpose(image, (1, 2, 0))
        image = (image - image.min(axis=(0, 1))) / (
            image.max(axis=(0, 1)) - image.min(axis=(0, 1))
        )
        plt.imsave(filepath, image)
    else:
        plt.imsave(filepath, image, cmap="gray")
    print(f"{filename} saved at {save_folder}")

# ------------------------------------------------------------------------------#
def save_nifti(filename, savepath, data, affine, header, technique):
    """Save data (ground truth or predictions) as a NIfTI file."""
    full_path = os.path.join(savepath, f"{filename}_{technique}.nii.gz")
    check_or_create_folder(os.path.dirname(full_path))
    nifti_image = nib.Nifti1Image(data.astype(np.uint8), affine, header=header)
    nib.save(nifti_image, full_path)
    print(f"{technique.capitalize()} saved at {full_path}")

# ------------------------------------------------------------------------------#
def save_metrics_to_csv(metrics, csv_path, mode):
    """Save computed metrics to a CSV file."""
    with open(csv_path, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Patient ID", "Dice", "HD95", "ASSD", "Mean IoU"]) if mode == "mask" \
        else csv_writer.writerow(["Patient ID", "MSE", "SSIM"])   
        csv_writer.writerows(metrics)
        
    print(f"Metrics saved to {csv_path}")
    
# ------------------------------------------------------------------------------#
def visualize_samples(samples, output_dir):
    """
    Visualizes samples in a **horizontal grid**:
    - **Autoencoder Inference**: 
      - Top row: Ground Truth images 
      - Bottom row: Corresponding Reconstructions
    - **LDM Inference**:
      - Row 1: Ground Truth Images
      - Row 2: Ground Truth Masks
      - Row 3: Predicted Masks

    Args:
    - samples (list): 
      - Autoencoder: List of tuples (GT Image, Reconstruction).
      - LDM: List of tuples (GT Image, GT Mask, Predicted Mask).
    - output_dir (str): Directory where visualization will be saved.
    """

    num_rows = len(samples[0]); assert num_rows in [2, 3], 'each tuple should have either 2 or 3 elements.'
    _, axes  = plt.subplots(num_rows, len(samples), figsize = (len(samples) * 5, num_rows * 5))
    axes     = axes.flatten()

    # row-wise order: (gt_image, recon_image) | (gt_image, gt_mask, pred_mask) 
    for i, image_tuple in enumerate(samples):
        for j in range(num_rows):
            ax = axes[j * len(samples) + i] # Get the correct axis for this subplot
            ax.imshow(np.transpose(image_tuple[j], (1, 2, 0))) # Show the corresponding image
            ax.axis('off') # Hide the axes

    plt.tight_layout()

    save_path = os.path.join(output_dir, 'sample_' + ('images' if num_rows == 2 else 'masks') + '.png')
    plt.savefig(save_path)
    print(f'sample visualization saved at {save_path}')
    plt.show()
    plt.close()

#------------------------------------------------------------------------------#