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
def visualize_intermediate_steps(intermediates, output_dir):
    """Visualize a horizontal grid of intermediate predictions."""
    decoded_images = [img.cpu() for img in intermediates]
    chain = torch.cat(decoded_images, dim=-1)

    plt.figure(figsize=(10, 12))
    plt.imshow(chain[0, 0], vmin=0, vmax=1, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "intermediate_steps.png"))
    print("Intermediate steps visualization saved.")

# ------------------------------------------------------------------------------#
@torch.no_grad()
def perform_inference(model, test_loader, device, output_dir, num_samples=5):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    predictions_list = []
    metrics_list = []

    progress_bar = tqdm(test_loader, desc="🚀 Inference Running", ncols=100)

    for batch in progress_bar:
        image      = batch["image"].to(device)
        mask       = batch["mask"].to(device)
        patient_id = batch["patient_id"].item()

        # Generate multiple noise vectors for stochasticity
        B = image.size(0)
        
        # Run sampling
        t         = torch.full((B,), NUM_TRAIN_TIMESTEPS - 1, device=device).long() if do.ONE_X_ONE else model.scheduler.timesteps
        model_out = model.inference(image, t)

        predicted_mask    = model_out['mask_hat']
        predicted_mask    = (torch.sigmoid(predicted_mask) > 0.5).float().cpu().numpy().squeeze()
        groundtruth_image = image.cpu().numpy().squeeze()
        groundtruth_mask  = mask.cpu().numpy().squeeze()

        # Save predictions
        patient_folder = os.path.join(output_dir, f"{int(patient_id)}")
        os.makedirs(patient_folder, exist_ok=True)

        save_groundtruth_image(groundtruth_image, patient_folder, "Image_groundtruth.jpg", mode = "image")
        save_groundtruth_image(groundtruth_mask,  patient_folder, "Mask_groundtruth.jpg",  mode = "mask")
        save_groundtruth_image(predicted_mask,    patient_folder, "Mask_predicted.jpg",    mode = "mask")

        # Store metrics
        if do.METRIC_REPORT:
            dice, hd95, assd, mean_iou = calculate_metrics(predicted_mask, groundtruth_mask)
            metrics_list.append([int(patient_id), dice, hd95, assd, mean_iou])

        # Save random visualizations
        if num_samples > 0 and len(predictions_list) < num_samples:
            predictions_list.append((groundtruth_image, groundtruth_mask, predicted_mask))

    # Save CSV metrics
    if do.METRIC_REPORT:
        metrics_path = os.path.join(output_dir, "metrics.csv")
        metrics_list.sort(key = lambda x: x[0])
        save_metrics_to_csv(metrics_list, metrics_path, mode="mask")

    print(f"✅ Inference complete. Results saved in: {output_dir}")

# ------------------------------------------------------------------------------#
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("⚙️  Preparing test dataloader...")
    test_loader = get_dataloaders(
        BASE_DIR, split_ratio=SPLIT_RATIOS, split="test",
        trainsize=TRAINSIZE, batch_size=1, format=FORMAT
    )

    print("📦 Loading trained LDM model...")
    model = LDM_Segmentor().to(device)
    model.scheduler.set_timesteps(do.INFERENCE_TIMESTEPS)

    ckpt_path      = os.path.join(LDM_SNAPSHOT_DIR, "models", f"model_epoch_{do.MODEL_EPOCH}.safetensors")
    model, _, _, _ = load_model_and_optimizer(ckpt_path, None, device, load_optim_dict = False)
    print(f"🔁 Loaded full model checkpoint from: {ckpt_path}")

    perform_inference(model, test_loader, device, do.SAVE_FOLDER, num_samples=do.NUM_SAMPLES)

# ------------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
# ------------------------------------------------------------------------------#
