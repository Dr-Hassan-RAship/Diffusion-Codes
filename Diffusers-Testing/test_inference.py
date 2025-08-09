# ------------------------------------------------------------------------------#
#
# File name                 : inference_ldm.py
# Purpose                   : Inference script for Latent Diffusion Model (LDM) segmentation
# Usage                     : python inference_ldm.py
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh, Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 202410001@lums.edu.pk, hassan.mohyuddin@lums.edu.pk
#
# Last Updated              : May 21, 2025
#
# ------------------------------------------------------------------------------#

import os, torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import *
from dataset import get_dataloaders
from architectures import *
from utils import *
from inference_ldm_utils import *

# ------------------------------------------------------------------------------#
@torch.no_grad()
def perform_inference(model, data_loader, device, output_dir, num_samples=5, model_epoch=-1, split="test"):
    """
    Perform inference on a dataset and save results.
    
    Args:
        model: LDM_Segmentor model
        data_loader: DataLoader for test or validation set
        device: Device to run inference on (cuda/cpu)
        output_dir: Directory to save results
        num_samples: Number of samples to store for visualization
        model_epoch: Epoch of the model (for naming and metrics)
        split: Dataset split ("test" or "val")
    Returns:
        metrics_list: List of [model_epoch, patient_id, dice, hd95, assd, mean_iou] (val) or [patient_id, dice, hd95, assd, mean_iou] (test)
        predictions_list: List of (groundtruth_image, groundtruth_mask, predicted_mask) tuples
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    predictions_list = []
    metrics_list = []

    progress_bar = tqdm(data_loader, desc=f"🚀 Inference (Epoch {model_epoch}, {split})", ncols=100)

    for batch in progress_bar:
        if not all(k in batch for k in ["image", "mask", "patient_id"]):
            print("Invalid batch format. Skipping.")
            continue
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)
        patient_id = batch["patient_id"].item()
        B = image.size(0)
        if B != 1:
            print(f"Batch size {B} not supported. Skipping.")
            continue

        try:
            model_out = model.inference(image)
            predicted_mask = model_out['mask_hat']
            predicted_mask = torch.clamp((predicted_mask + 1.0) / 2.0, min=0.0, max=1.0)
            predicted_mask = predicted_mask.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
            predicted_mask = (predicted_mask > 0.5).float().cpu().numpy().squeeze()
            groundtruth_image = image.cpu().numpy().squeeze()
            groundtruth_mask = mask.cpu().numpy().squeeze()
        except Exception as e:
            print(f"Error processing batch for patient {patient_id}: {e}")
            continue

        patient_folder = os.path.join(output_dir, f"epoch_{model_epoch}", f"{int(patient_id)}")
        os.makedirs(patient_folder, exist_ok=True)
        save_groundtruth_image(groundtruth_image, patient_folder, "Image_groundtruth.jpg", mode="image")
        save_groundtruth_image(groundtruth_mask, patient_folder, "Mask_groundtruth.jpg", mode="mask")
        save_groundtruth_image(predicted_mask, patient_folder, "Mask_predicted.jpg", mode="mask")

        if do.METRIC_REPORT:
            try:
                dice, hd95, assd, mean_iou = calculate_metrics(predicted_mask, groundtruth_mask)
                metrics_entry = [model_epoch, int(patient_id), dice, hd95, assd, mean_iou] if split == "val" else [int(patient_id), dice, hd95, assd, mean_iou]
                metrics_list.append(metrics_entry)
            except Exception as e:
                print(f"Error calculating metrics for patient {patient_id}: {e}")

        if num_samples > 0 and len(predictions_list) < num_samples:
            predictions_list.append((groundtruth_image, groundtruth_mask, predicted_mask))

    return metrics_list, predictions_list

# ------------------------------------------------------------------------------#
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

# ------------------------------------------------------------------------------#
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

# ------------------------------------------------------------------------------#
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚙️  Using device: {device}")
    if device == "cpu":
        print("Warning: CUDA unavailable, performance may be slower.")

    # Choose dataset split based on number of model epochs
    split = "val" if len(do.MODEL_EPOCHS) > 1 else "test"
    print(f"⚙️  Preparing {split} dataloader...")
    try:
        data_loader = get_dataloaders(
            BASE_DIR, split_ratio=SPLIT_RATIOS, split=split,
            trainsize=TRAINSIZE, batch_size=1, format=FORMAT
        )
    except Exception as e:
        print(f"Error loading {split} dataloader: {e}")
        return

    all_metrics = []
    for model_epoch in do.MODEL_EPOCHS:
        print(f"📦 Loading trained LDM model for epoch {model_epoch}...")
        try:
            model = LDM_Segmentor().to(device)
            model.scheduler.set_timesteps(do.INFERENCE_TIMESTEPS)
            ckpt_path = os.path.join(LDM_SNAPSHOT_DIR, "models", f"model_epoch_{model_epoch}.safetensors")
            model, _, _, _ = load_model_and_optimizer(ckpt_path, None, device, load_optim_dict=False)
            print(f"🔁 Loaded model checkpoint from: {ckpt_path}")
        except Exception as e:
            print(f"Error loading model for epoch {model_epoch}: {e}")
            continue

        output_dir = os.path.join(do.SAVE_FOLDER, f"inference_{split}_M{model_epoch}")
        metrics_list, predictions_list = perform_inference(
            model, data_loader, device, output_dir, num_samples=do.NUM_SAMPLES,
            model_epoch=model_epoch, split=split
        )
        all_metrics.extend(metrics_list)
        if predictions_list:
            visualize_predictions(predictions_list, output_dir, model_epoch=model_epoch)

    if do.METRIC_REPORT and all_metrics:
        metrics_path = os.path.join(do.SAVE_FOLDER, "val_metrics.csv")
        if split == "val":
            # Compute and save averaged metrics for validation
            avg_metrics = compute_average_metrics(all_metrics)
            avg_metrics.sort(key=lambda x: x[0])  # Sort by model_epoch
            save_metrics_to_csv(avg_metrics, metrics_path, mode="mask", headers=["Model_Epoch", "Avg_Dice", "Avg_HD95", "Avg_ASSD", "Avg_mIoU"])
        else:
            # Save per-patient metrics for test
            all_metrics.sort(key=lambda x: x[0])  # Sort by patient_id
            save_metrics_to_csv(all_metrics, metrics_path, mode="mask", headers=["Patient_ID", "Dice", "HD95", "ASSD", "mIoU"])
        print(f"✅ Metrics saved in: {metrics_path}")

    print(f"✅ Inference complete. Results saved in: {do.SAVE_FOLDER}")

# ------------------------------------------------------------------------------#
if __name__ == "__main__":
    class InferenceConfig:
        N_PREDS = 1  # Unused, reserved for future use
        MODEL_EPOCHS = [2400]  # List of model epochs to load
        NUM_SAMPLES = 2
        INFERER_SCHEDULER = 'DDIM'
        TRAIN_TIMESTEPS = NUM_TRAIN_TIMESTEPS
        ONE_X_ONE = False
        INFERENCE_TIMESTEPS = 10 if INFERER_SCHEDULER == 'DDIM' else NUM_TRAIN_TIMESTEPS
        SAVE_FOLDER = LDM_SNAPSHOT_DIR + f"/inference-M{'-'.join(map(str, MODEL_EPOCHS)) if MODEL_EPOCHS else 'final'}-E{N_EPOCHS}-t{NUM_TRAIN_TIMESTEPS}-S{SCHEDULER}-SP{NUM_SAMPLES}-It{INFERENCE_TIMESTEPS}"
        SAVE_INTERMEDIATES = False
        METRIC_REPORT = True

        def validate(self):
            """Validate configuration parameters."""
            assert hasattr(self, 'TRAIN_TIMESTEPS'), "NUM_TRAIN_TIMESTEPS undefined in config"
            assert hasattr(self, 'N_EPOCHS'), "N_EPOCHS undefined in config"
            assert hasattr(self, 'SCHEDULER'), "SCHEDULER undefined in config"
            assert os.path.exists(LDM_SNAPSHOT_DIR), f"LDM_SNAPSHOT_DIR {LDM_SNAPSHOT_DIR} does not exist"
            assert isinstance(self.MODEL_EPOCHS, list) and len(self.MODEL_EPOCHS) > 0, "MODEL_EPOCHS must be a non-empty list"
            for epoch in self.MODEL_EPOCHS:
                assert isinstance(epoch, int) and epoch >= -1, f"Invalid model epoch: {epoch}"

    do = InferenceConfig()
    do.validate()
    main()
# ------------------------------------------------------------------------------#