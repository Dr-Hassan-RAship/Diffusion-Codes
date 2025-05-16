# ------------------------------------------------------------------------------#
#
# File name                 : inference_ldm_utils.py
# Purpose                   : Helper functions for inference of Latent Diffusion Model (LDM)
# Usage                     : Used by inference_ldm.py
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh, Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 202410001@lums.edu.pk, hassan.mohyuddin@lums.edu.pk
#
# Last Date                 : April 28, 2025
#
# ------------------------------------------------------------------------------#
import os, csv, torch

import numpy                                        as np
import nibabel                                      as nib
import matplotlib.pyplot                            as plt
import matplotlib

matplotlib.use("Agg")

from monai.metrics                                  import DiceMetric, MeanIoU
from medpy.metric.binary                            import assd, hd95
from config                                         import *
from utils                                          import *

# ------------------------------------------------------------------------------#
def save_nifti(filename, savepath, data, affine=None, header=None):
    """Save data as a NIfTI file."""
    save_path = os.path.join(savepath, f"{filename}.nii.gz")
    check_or_create_folder(os.path.dirname(save_path))
    nifti_image = nib.Nifti1Image(data.astype(np.uint8), affine, header=header)
    nib.save(nifti_image, save_path)
    
# ------------------------------------------------------------------------------#
def calculate_metrics(prediction, label):
    # """
    # For calculating metrics such as dice, hd95, assd and miou.
    # """
    
    pred = (prediction > 0).astype(np.uint8)
    gt   = (label > 0).astype(np.uint8)

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    miou_metric = MeanIoU(include_background=False)

    dice        = dice_metric(torch.tensor(pred)[None, None], torch.tensor(gt)[None, None]).item()
    miou        = miou_metric(torch.tensor(pred)[None, None], torch.tensor(gt)[None, None]).item()
    assd_val    = assd(pred, gt)
    hd95_val    = hd95(pred, gt)
    
    return dice, hd95_val, assd_val, miou

# ------------------------------------------------------------------------------#
def save_groundtruth_image(image, save_folder, filename, mode="image"):
    """Save ground truth image as a JPEG file."""
    image = np.squeeze(image)
    filepath = os.path.join(save_folder, filename)
    if mode == "image":
        # tranpose image to (256, 256, 3) and normalize between 0 - 1 for each channel independantly for imsave
        image  = np.transpose(image, (1, 2, 0))
        # normalize 
        image  = ((image + 1.0) / 2.0).clip(0, 1)

        plt.imsave(filepath, image)
    else:
        image = np.transpose(image, (1, 2, 0))
        image = (((image + 1.0) / 2.0))[:, :, 0]
        
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
    Visualizes samples in a horizontal grid:
      - For AutoencoderKL Inference: 
          Row 1: Ground Truth Images 
          Row 2: Reconstructions
      - For LDM Inference:
          Row 1: Ground Truth Images
          Row 2: Ground Truth Masks
          Row 3: Predicted Masks

    Args:
      samples (list): A list of tuples where each tuple contains:
                      - For autoencoder: (GT Image, Reconstruction)
                      - For LDM: (GT Image, GT Mask, Predicted Mask)
      output_dir (str): Directory where the visualization image will be saved.
    """
    num_samples = len(samples)
    num_rows = len(samples[0])
    assert num_rows in [2, 3], "Each sample tuple must have either 2 or 3 elements."

    # Titles based on inference type
    titles = {
        2: ["Ground Truth", "Reconstruction"],
        3: ["GT Image", "GT Mask", "Predicted Mask"]
    }

    # Create figure and axes
    fig, axes = plt.subplots(num_rows, num_samples, figsize=(num_samples * 4, num_rows * 4))

    # If only one sample, make axes 2D
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=1)

    for col_idx, sample in enumerate(samples):
        for row_idx in range(num_rows):
            ax = axes[row_idx][col_idx]
            img = sample[row_idx]

            # Convert CHW to HWC if needed
            if img.ndim == 3 and img.shape[0] in [1, 3]:
                img = np.transpose(img, (1, 2, 0))

            # Squeeze grayscale channels
            if img.shape[-1] == 1:
                img = img.squeeze(-1)

            ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
            ax.set_title(titles[num_rows][row_idx])
            ax.axis("off")

    plt.tight_layout()
    filename = "sample_reconstructions.png" if num_rows == 2 else "sample_masks.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path)
    print(f"Sample visualization saved at {save_path}")
    plt.close()

#------------------------------------------------------------------------------#