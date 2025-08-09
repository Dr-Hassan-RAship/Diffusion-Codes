# ------------------------------------------------------------------------------#
#
# File name                 : inference_autoencoderkl.py
# Purpose                   : Inference script for AutoencoderKL models (Image/Mask)
# Usage                     : python inference_autoencoderkl.py --mode 'mask' --epoch_model 200 --num_samples 10
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh, Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 202410001@lums.edu.pk, hassan.mohyuddin@lums.edu.pk
#
# Last Date                 : March 25, 2025
#
# ------------------------------------------------------------------------------#

import os, torch, argparse, random

import                                  numpy as np

from torchvision                        import transforms
from torch.amp                          import autocast
from tqdm                               import tqdm
from monai.metrics.regression           import SSIMMetric
from dataset                            import *
from inference_ldm_utils                import *
from config                    import *


# ------------------------------------------------------------------------------#
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Inference script for AutoencoderKL (Image/Mask)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["image", "mask"],
        required=True,
        help="Specify whether to run inference for images or masks",
    )
    
    parser.add_argument(
        "--epoch_model",
        type=int,
        default=100,
        help="The epoch at which the model should be loaded",
    )
    
    parser.add_argument(
        "--num_samples",
        type=int,
        default=0,
        help="Number of random samples to visualize (0 = full test set)",
    )
    return parser.parse_args()


# ------------------------------------------------------------------------------#
def inference(autoencoderkl, test_loader, device, output_dir, mode, num_samples):
    """
    Perform inference using the AutoencoderKL model.
    """
    autoencoderkl.eval()
    metrics_list = []
    result_list = []

    progress_bar = tqdm(
        test_loader,
        desc=f"Inference Progress for {mode.capitalize()} AutoencoderKL",
        ncols=100,
    )
    # define normalization for reconstruction - same as clean image from dataset.py
    ssim_metric = SSIMMetric(spatial_dims=2)
    
    with torch.no_grad():
        for batch in progress_bar:
            if mode == "image":
                ae_input              = batch["image"].to(device)
                gt_input              = batch["image"].to(device)
                
            else:
                ae_input              = batch["mask"].to(device)
                gt_input              = batch["mask"].to(device)
                
            recon_norm = postprocess_and_rescaling(autoencoderkl, ae_input, mode)
            gt_input   = postprocess_and_rescaling(None, gt_input, mode)
            patient_id = batch["patient_id"]

            for i in range(ae_input.shape[0]):
                aekl_input    = gt_input[i].float().cpu().numpy().squeeze()
                aekl_recon    = recon_norm[i].float().cpu().numpy().squeeze()
                
                # print(f'aekl_input.shape {aekl_input.shape}, aekl_recon.shape {aekl_recon.shape}')

                patient_folder = os.path.join(output_dir, f"{patient_id[i]}")
                check_or_create_folder(patient_folder)

                save_groundtruth_image(aekl_input, patient_folder, f"{mode}_groundtruth.jpg")
                save_groundtruth_image(aekl_recon, patient_folder, f"{mode}_reconstructed.jpg")

                if mode == "mask":
                    dice, hd95, assd, miou = calculate_metrics(aekl_recon, aekl_input)
                    metrics_list.append([int(patient_id[i].item()), dice, hd95, assd, miou])
                else:
                    mse  = np.mean((aekl_input - aekl_recon) ** 2)
                    ssim = ssim_metric(torch.tensor(aekl_recon).unsqueeze(0), torch.tensor(aekl_input).unsqueeze(0))
                    metrics_list.append([int(patient_id[i].item()), mse, ssim.item()])  # append ssim if used

                if num_samples > 0 and len(result_list) < num_samples:
                    result_list.append((aekl_input, aekl_recon))

    metrics_filename = os.path.join(output_dir, "metrics.csv")
    metrics_list.sort(key=lambda x: x[0])
    save_metrics_to_csv(metrics_list, metrics_filename, mode)
    
    # [talha] --> debug this function as its giving problem?
    if num_samples > 0 and len(result_list) >= num_samples:
        print("Calling visualize_samples function...")
        selected_samples = random.sample(result_list, num_samples)
        visualize_samples(selected_samples, output_dir)

    print(f"Inference complete! Results saved in {output_dir}")

# ------------------------------------------------------------------------------#
def main():
    args   = parse_arguments()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    im_bool   = True if args.mode == "image" else False
    mask_bool = True if args.mode == "mask" else False

    
    if im_bool:
        output_dir = os.path.join(AEKL_IMAGE_SNAPSHOT_DIR, f"inference_results_SP{args.num_samples}")
        check_or_create_folder(output_dir)
    else:
        output_dir = os.path.join(AEKL_MASK_SNAPSHOT_DIR, f"inference_results_SP{args.num_samples}")
        check_or_create_folder(output_dir)

    train_loader = get_dataloaders(
        BASE_DIR,
        split_ratio  = SPLIT_RATIOS,
        split        = "train",
        trainsize    = TRAINSIZE,
        batch_size   = BATCH_SIZE,
        format       = FORMAT,
    )
    test_loader  = get_dataloaders(
        BASE_DIR,
        split_ratio  = SPLIT_RATIOS,
        split        = "test",
        trainsize    = TRAINSIZE,
        batch_size   = 1,
        format       = FORMAT,
    )
    tuple = load_autoencoder(device, train_loader, image = im_bool, mask = mask_bool, 
                             epoch_image = args.epoch_model if im_bool else 0, 
                             epoch_mask = args.epoch_model if mask_bool else 0)
    dae, _ = tuple

    inference(dae, test_loader, device, output_dir, args.mode, args.num_samples)


# ------------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
# ------------------------------------------------------------------------------#
