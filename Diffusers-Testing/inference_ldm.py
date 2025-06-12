# ------------------------------------------------------------------------------#
#
# File name                 : inference_ldm.py
# Purpose                   : Inference script for Latent Diffusion Model (LDM) segmentation
# Usage                     : python inference_ldm.py
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh, Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 202410001@lums.edu.pk, hassan.mohyuddin@lums.edu.pk
#
# Last Updated              : April 28, 2025
#
# ------------------------------------------------------------------------------#

import os, torch

import matplotlib.pyplot as plt

from tqdm                import tqdm
from config              import *
from dataset             import get_dataloaders
from architectures       import *
from utils               import *
from inference_ldm_utils import *

# ------------------------------------------------------------------------------#
@torch.no_grad()
def perform_inference(model, data_loader, device, output_dir, num_samples=5, model_epoch=-1, split="test"):
    """
    Perform inference on a dataset and save results.
    
    Args:
        model           : LDM_Segmentor model
        data_loader     : DataLoader for test or validation set
        device          : Device to run inference on (cuda/cpu)
        output_dir      : Directory to save results
        num_samples     : Number of samples to store for visualization
        model_epoch     : Epoch of the model (for naming and metrics)
        split           : Dataset split ("test" or "val")
    Returns:
        metrics_list: List of [model_epoch, patient_id, dice, hd95, assd, mean_iou] (val) or [patient_id, dice, hd95, assd, mean_iou] (test)
        predictions_list: List of (groundtruth_image, groundtruth_mask, predicted_mask) tuples
    """
    model.eval()
    predictions_list    = []
    metrics_list        = []

    progress_bar = tqdm(data_loader, desc=f"🚀 Inference (Epoch {model_epoch}, {split})", ncols=100)

    for batch in progress_bar:
        image       = batch["image"].to(device)  if split == "test" else batch["aug_image"].to(device)
        mask        = batch["mask"].to(device)   if split == "test" else batch["aug_mask"].to(device)
        patient_id  = batch["patient_id"].item() 
        B           = image.size(0)

        model_out           = model.inference(image)
        predicted_mask      = model_out['mask_hat']
        predicted_mask      = torch.clamp((predicted_mask + 1.0) / 2.0, min=0.0, max=1.0)
        predicted_mask      = predicted_mask.mean(dim=1, keepdim = True).repeat(1, 3, 1, 1)
        predicted_mask      = (predicted_mask > 0.5).float().cpu().numpy().squeeze()
        groundtruth_image   = image.cpu().numpy().squeeze()
        groundtruth_mask    = mask.cpu().numpy().squeeze()

        if split == 'test':
            patient_folder      = os.path.join(output_dir, f"{int(patient_id)}")
            os.makedirs(patient_folder, exist_ok=True)
            save_groundtruth_image(groundtruth_image, patient_folder, "Image_groundtruth.jpg", mode="image")
            save_groundtruth_image(groundtruth_mask, patient_folder, "Mask_groundtruth.jpg", mode="mask")
            save_groundtruth_image(predicted_mask, patient_folder, "Mask_predicted.jpg", mode="mask")

        if do.METRIC_REPORT:
            dice, hd95, assd, mean_iou = calculate_metrics(predicted_mask, groundtruth_mask)
            metrics_entry = [model_epoch, int(patient_id), dice, hd95, assd, mean_iou] if split == "val" else [int(patient_id), dice, hd95, assd, mean_iou]
            metrics_list.append(metrics_entry)

        # if num_samples > 0 and len(predictions_list) < num_samples:
        #     predictions_list.append((groundtruth_image, groundtruth_mask, predicted_mask))

    return metrics_list, predictions_list
# ------------------------------------------------------------------------------#
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Choose dataset split based on number of model epochs
    split = "val" if len(do.MODEL_EPOCHS) > 1 else "test"
    print(f"⚙️  Preparing {split} dataloader...")
    data_loader = get_dataloaders(BASE_DIR, split_ratio=SPLIT_RATIOS, split=split,
                                  trainsize=TRAINSIZE, batch_size=1, format=FORMAT
    )

    all_metrics = []
    for model_epoch in do.MODEL_EPOCHS:
        print(f"📦 Loading trained LDM model for epoch {model_epoch}...")
        try:            
            ckpt_path       = os.path.join(LDM_SNAPSHOT_DIR, "models", f"model_epoch_{model_epoch}.safetensors")
            model, _, _, _  = load_model_and_optimizer(ckpt_path, None, device, load_optim_dict=False)
            model.scheduler.set_timesteps(do.INFERENCE_TIMESTEPS)
            print(f"🔁 Loaded model checkpoint from: {ckpt_path}")
        except Exception as e:
            print(f"Error loading model for epoch {model_epoch}: {e}")
            continue
        
        output_dir = None
        if split == 'test':
           output_dir = os.path.join(do.SAVE_FOLDER, f"inference_{split}_M{model_epoch}")
           os.makedirs(output_dir, exist_ok = True)
        metrics_list, predictions_list = perform_inference(
            model, data_loader, device, output_dir, num_samples=do.NUM_SAMPLES,
            model_epoch=model_epoch, split=split
        )
        all_metrics.extend(metrics_list)
        # if predictions_list:
        #     visualize_predictions(predictions_list, output_dir, model_epoch=model_epoch)

    if do.METRIC_REPORT and all_metrics:
        os.makedirs(do.SAVE_FOLDER, exist_ok = True)
        if split == "val":
            metrics_path = os.path.join(do.SAVE_FOLDER, f'inference_{split}_M{do.MODEL_EPOCHS}.csv')	
            # Compute and save averaged metrics for validation
            avg_metrics = compute_average_metrics(all_metrics)
            avg_metrics.sort(key=lambda x: x[0])  # Sort by model_epoch
            save_metrics_to_csv(avg_metrics, metrics_path, headers=["Model_Epoch", "Avg_Dice", "Avg_HD95", "Avg_ASSD", "Avg_mIoU"])
        else:
            metrics_path = os.path.join(do.SAVE_FOLDER, f'inference_{split}_M{do.MODEL_EPOCHS}.csv')	
            # Save per-patient metrics for test
            all_metrics.sort(key=lambda x: x[0])  # Sort by patient_id
            save_metrics_to_csv(all_metrics, metrics_path, headers=["Patient_ID", "Dice", "HD95", "ASSD", "mIoU"])
        print(f"✅ Metrics saved in: {metrics_path}")

    print(f"✅ Inference complete. Results saved in: {do.SAVE_FOLDER}")
# ------------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
# ------------------------------------------------------------------------------ #
