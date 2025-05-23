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
def calculate_metrics(prediction, label):
    # """
    # For calculating metrics such as dice, hd95, assd and miou.
    # """
    
    # pred = (prediction > 0).astype(np.uint8)
    pred = prediction
    gt   = (label > 0).astype(np.uint8)

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    miou_metric = MeanIoU(include_background=False)

    dice        = dice_metric(torch.tensor(pred)[None, None], torch.tensor(gt)[None, None]).item()
    miou        = miou_metric(torch.tensor(pred)[None, None], torch.tensor(gt)[None, None]).item()
    if pred.sum() > 0 and gt.sum() > 0:
        assd_val = assd(pred, gt)
        hd95_val = hd95(pred, gt)
    elif pred.sum() == 0 and gt.sum() == 0:
        assd_val = 0.0
        hd95_val = 0.0
    else:
        assd_val = float("inf")
        hd95_val = float("inf")
    
    return dice, hd95_val, assd_val, miou

# ------------------------------------------------------------------------------#
def save_groundtruth_image(image, save_folder, filename, mode="image"):
    """Save ground truth image as a JPEG file."""
    # image = np.squeeze(image)
    filepath = os.path.join(save_folder, filename)
    if mode == "image":
        # tranpose image to (256, 256, 3) and normalize between 0 - 1 for each channel independantly for imsave
        image  = np.transpose(image, (1, 2, 0))
        # normalize 
        image  = ((image + 1.0) / 2.0).clip(0, 1)

        plt.imsave(filepath, image)
    else:
        image = np.transpose(image, (1, 2, 0))
        image = image[:, :, 0]
        
        plt.imsave(filepath, image, cmap="gray")
    print(f"{filename} saved at {save_folder}")

# ------------------------------------------------------------------------------#
def save_metrics_to_csv(metrics, csv_path, headers=None):
    """Save computed metrics to a CSV file."""
    with open(csv_path, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(headers)
        csv_writer.writerows(metrics)
        
    print(f"Metrics saved to {csv_path}")
#------------------------------------------------------------------------------#
def visualize_intermediate_steps(intermediates, output_dir):
    """Visualize a horizontal grid of intermediate predictions."""
    if not intermediates or not isinstance(intermediates, (list, tuple)):
        print("No valid intermediates provided for visualization.")
        return
    decoded_images = [img.cpu() for img in intermediates]
    try:
        chain = torch.cat(decoded_images, dim=-1)
        plt.figure(figsize=(len(intermediates) * 2, 4))
        plt.imshow(chain[0, 0], vmin=0, vmax=1, cmap="gray")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "intermediate_steps.png"))
        plt.close()
        print("Intermediate steps visualization saved.")
    except Exception as e:
        print(f"Error visualizing intermediates: {e}")
#------------------------------------------------------------------------------#
def visualize_predictions(predictions_list, output_dir, model_epoch=-1):
    """Visualize ground truth images, masks, and predicted masks."""
    vis_dir = os.path.join(output_dir, f"epoch_{model_epoch}", "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    for i, (gt_image, gt_mask, pred_mask) in enumerate(predictions_list):
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(gt_image, cmap="gray")
        plt.title("Ground Truth Image")
        plt.axis("off")
        plt.subplot(1, 3, 2)
        plt.imshow(gt_mask, cmap="gray")
        plt.title("Ground Truth Mask")
        plt.axis("off")
        plt.subplot(1, 3, 3)
        plt.imshow(pred_mask, cmap="gray")
        plt.title("Predicted Mask")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f"sample_{i}.png"))
        plt.close()
    print(f"Visualizations saved in {vis_dir}")
#---------------------------------------------------------------------------------#
def compute_average_metrics(metrics_list):
    """Compute average metrics (Dice, HD95, ASSD, mIoU) per model epoch."""
    if not metrics_list:
        return []
    
    # Group metrics by model_epoch
    epoch_metrics = {}
    for metric in metrics_list:
        model_epoch = metric[0]
        if model_epoch not in epoch_metrics:
            epoch_metrics[model_epoch] = []
        epoch_metrics[model_epoch].append(metric[2:])  # [dice, hd95, assd, mean_iou]

    # Compute averages
    avg_metrics = []
    for model_epoch, metrics in epoch_metrics.items():
        metrics_array = torch.tensor(metrics, dtype=torch.float32)
        avg_dice = metrics_array[:, 0].mean().item()
        avg_hd95 = metrics_array[:, 1].mean().item()
        avg_assd = metrics_array[:, 2].mean().item()
        avg_miou = metrics_array[:, 3].mean().item()
        avg_metrics.append([model_epoch, avg_dice, avg_hd95, avg_assd, avg_miou])
    
    return avg_metrics
#--------------------------------------------------------------------------------#
# Potential Improvements:

# 1. Use multiprocessing or threading for parallel processing of metrics calculation.
# from concurrent.futures import ThreadPoolExecutor
# with ThreadPoolExecutor() as executor:
#     metrics = list(executor.map(calculate_metrics, preds, labels))