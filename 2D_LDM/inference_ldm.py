# ------------------------------------------------------------------------------#
#
# File name                 : inference_ldm.py
# Purpose                   : Inference script for Latent Diffusion Model (LDM) segmentation
# Usage                     : python inference_ldm.py
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh, Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 202410001@lums.edu.pk, hassan.mohyuddin@lums.edu.pk
#
# Last Date                 : March 11, 2025
#
# ------------------------------------------------------------------------------#

import os, torch, random

import numpy                    as np
import matplotlib.pyplot        as plt

from tqdm                       import tqdm
from dataset                    import *
from inference_ldm_utils        import *
from config_ldm_ddpm            import *
# ------------------------------------------------------------------------------#
def inference(unet, aekl_image, aekl_mask, inferer, test_loader, device, output_dir, num_samples):
    """
    Perform inference using the LDM model with a progress bar.
    """
    unet.eval()
    aekl_image.eval()
    aekl_mask.eval()
    metrics_list     = []
    result_list      = []
    
    progress_bar = tqdm(test_loader, desc="LDM Inference Progress", ncols=100)

    with torch.no_grad():
        for batch in progress_bar:
            images            = batch["image"].to(device)
            groundtruth_image = batch["image"].cpu().numpy().squeeze()
            groundtruth_mask  = batch["mask"].cpu().numpy().squeeze()

            patient_id = batch["patient_id"]

            # Encode images to latent space
            z_c           = aekl_image.encode_stage_2_inputs(images).to(device) # Conditioned latent representation

            # Generate noise tensor for segmentation mask
            z_t_list = [torch.randn_like(z_c).to(device) for _ in range(do.N_PREDS)]

            # Perform LDM sampling # [talha] do.num_average times and then average predictions but in the case num_average > 1you dont need to average the intermediates only the predictions
            if do.SAVE_INTERMEDIATES:
                predicted_mask, intermediates = inferer.sample(input_noise        = z_t_list[0],
                                                               autoencoder_model  = aekl_mask,
                                                               diffusion_model    = unet,
                                                               scheduler          = inferer.scheduler,
                                                               save_intermediates = do.SAVE_INTERMEDIATES,
                                                               intermediate_steps = NUM_TRAIN_TIMESTEPS // 10,
                                                               conditioning       = z_c,
                                                               mode               = "concat")
            else:
                predicted_mask = torch.mean(torch.stack([inferer.sample(input_noise       = z_t,
                                                                        autoencoder_model = aekl_mask,
                                                                        diffusion_model   = unet,
                                                                        scheduler         = inferer.scheduler,
                                                                        conditioning      = z_c,
                                                                        mode              = "concat") for z_t in z_t_list]), dim = 0)

            # binarize predicted_mask
            predicted_mask    = (torch.sigmoid(predicted_mask) > 0.5).float().cpu().numpy().squeeze()
            # Save results
            patient_folder    = os.path.join(output_dir, f"{int(patient_id.item())}")
            check_or_create_folder(patient_folder)
            

            save_groundtruth_image(groundtruth_image, patient_folder, "Image_groundtruth.jpg", mode = 'image')
            save_groundtruth_image(groundtruth_mask, patient_folder, "Mask_groundtruth.jpg", mode = 'mask')
            save_groundtruth_image(predicted_mask, patient_folder, "Mask_predicted.jpg", mode = 'mask')

            # Compute segmentation metrics (Dice, HD95, ASSD, MeanIOU)
            if do.METRIC_REPORT:
                dice, hd95, assd, mean_iou = calculate_metrics(predicted_mask, groundtruth_mask)
                metrics_list.append([int(patient_id.item()), dice, hd95, assd, mean_iou])

            # Random sample collection
            if num_samples > 0 and len(result_list) < num_samples:
                result_list.append((groundtruth_image, groundtruth_mask, predicted_mask))

    # Save metrics
    if do.METRIC_REPORT:
        metrics_filename = os.path.join(output_dir, "metrics.csv")
        metrics_list.sort(key = lambda x: x[0])
        save_metrics_to_csv(metrics_list, metrics_filename, mode = 'mask')

    # # Visualize sample segmentations
    # if num_samples > 0 and len(result_list) >= num_samples:
    #     selected_samples = random.sample(result_list, num_samples)
    #     visualize_samples(selected_samples, output_dir)
    
    # visualize intermediate steps

    if do.SAVE_INTERMEDIATES:
        visualize_intermediate_steps(intermediates, output_dir)
    
    print(f"Inference complete! Results saved in {output_dir}")

# ------------------------------------------------------------------------------#
def visualize_intermediate_steps(intermediates, output_dir):
    """Visualizes a grid of intermediate steps showing LDM progression."""
    decoded_images = [img.cpu() for img in intermediates]
    chain          = torch.cat(decoded_images, dim=-1)

    plt.figure(figsize=(10, 12))
    plt.imshow(chain[0, 0], vmin=0, vmax=1, cmap="gray")
    plt.axis("off")
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "intermediate_steps.png"))
    print("Intermediate steps visualization saved.")

# ------------------------------------------------------------------------------#
def main():
    device       = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader = get_dataloaders(BASE_DIR, split_ratio=SPLIT_RATIOS, split="train", trainsize=TRAINSIZE, batch_size=BATCH_SIZE, format=FORMAT)
    test_loader  = get_dataloaders(BASE_DIR, split_ratio=SPLIT_RATIOS, split="test", trainsize=TRAINSIZE, batch_size=1, format=FORMAT)

    # Load pre-trained models
    tup_image, tup_mask      = load_autoencoder(device, train_loader, image=True, mask=True, epoch_mask = 100, epoch_image = 500)
    aekl_image, _            = tup_image
    aekl_mask, scale_factor  = tup_mask
    unet, _, inferer         = load_ldm_model(device, scale_factor)

    check_or_create_folder(do.SAVE_FOLDER)
    inference(unet, aekl_image, aekl_mask, inferer, test_loader, device, do.SAVE_FOLDER, do.NUM_SAMPLES)

# ------------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
# ------------------------------------------------------------------------------#
