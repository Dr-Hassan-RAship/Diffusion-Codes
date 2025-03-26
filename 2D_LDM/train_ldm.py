# ------------------------------------------------------------------------------#
#
# File name                 : train_ldm.py
# Purpose                   : Train Latent Diffusion Model (LDM) for binary segmentation
# Usage                     : python train_ldm.py
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh, Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 202410001@lums.edu.pk, hassan.mohyuddin@lums.edu.pk
#
# Last Date                 : March 11, 2025
# ------------------------------------------------------------------------------#

import logging, os, torch, time

import torch.nn.functional as F

from torch.amp                      import GradScaler, autocast
from torch.utils.tensorboard        import SummaryWriter
# from generative.networks.nets       import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler
# from generative.inferers            import LatentDiffusionInferer
from config_ldm_ddpm                import *
from dataset                        import get_dataloaders
from inference_ldm_utils            import *
from utils                          import *
from source_autoencoderkl           import *
from source_inferer                 import *
from source_unet                    import *

# ------------------------------------------------------------------------------#
# Setup
# ------------------------------------------------------------------------------#
def setup_environment(seed: int, snapshot_dir: str):
    """Set up environment and determinism."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.multiprocessing.set_sharing_strategy("file_system")
    
    output_file = (
        "model_params.txt"  # Define the output text file pathweights_only=True
    )

    with open(
        f"{snapshot_dir}/{output_file}", "w"
    ) as f:  # Write the dictionary to a text file
        f.write("MODEL_PARAMS:\n")
        f.write(f"batch_size: {BATCH_SIZE}\n")
        [f.write(f"{key}: {value}\n") for key, value in MODEL_PARAMS.items()]

    print(f"MODEL_PARAMS written to {output_file}")

    return "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------------------#
# Training and Validation Functions
# ------------------------------------------------------------------------------#


def train_one_epoch(model, aekl_image, aekl_mask, train_loader, optimizer, inferer, scaler, scheduler, device, epoch, writer):
    """Train the LDM for one epoch."""
    epoch_loss = 0.0

    model.train()
    aekl_image.eval()
    aekl_mask.eval()

    for step, batch in enumerate(train_loader):
        image, mask  = batch['aug_image'].to(device), batch['aug_mask'].to(device)
        # noisy_image, noisy_mask = batch["noisy_image"].to(device), batch["noisy_mask"].to(device)

        optimizer.zero_grad(set_to_none = True)
        with autocast("cuda", enabled=True):
            # Assumed Pipeline:
            # 1. Encode images and masks
            # 2. Get random noise same shape as latent_masks
            # 3. Get timesteps corresponding to latent masks
            # 4. Call inferer with latent_images as conditioning and mode as 'concat' and autoencoder model as aekl_mask

            latent_images               = aekl_image.encode_stage_2_inputs(image).to(device)
            latent_masks                = aekl_mask.encode_stage_2_inputs(mask).to(device)
            noise                       = torch.randn_like(latent_masks).to(device) # (B, C, H, W)
            timesteps                   = torch.randint(0, scheduler.num_train_timesteps, (latent_masks.size(0),), device = device).long()
            z_T                         = scheduler.add_noise(original_samples = latent_masks, noise = noise, timesteps = timesteps)
            
            #[talha] Make sure z_t is same as the z_t we could have returned from inferer. 
            noise_pred                  = inferer(inputs            = mask,
                                                  noise             = noise,
                                                  diffusion_model   = model,
                                                  timesteps         = timesteps,
                                                  autoencoder_model = aekl_mask,
                                                  condition         = latent_images,
                                                  mode              = "concat")
            # Batchify loss_latent by making sure inferer returns noise_pred, z_t and the timesteps for it
            # Then calculate L1 loss between z_o_tilde (gotten from Eq(2) of paper) and latent_masks
            loss_noise             = F.l1_loss(noise_pred.float(), noise.float())
            
            alpha_bar_T            = scheduler.alphas_cumprod[timesteps][:, None, None, None]
            z_0_pred               = (1 / torch.sqrt(alpha_bar_T)) * (z_T - (torch.sqrt(1 - alpha_bar_T) * noise_pred))
            loss_latent            = F.l1_loss(z_0_pred.float(), latent_masks.float())

            # Then add both losses and backpropogate.
            loss                   = loss_noise + loss_latent

        # Using Monai Scaler for better precision training and better gradient calculation
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # cumulate losses and log for later debugging
        epoch_loss += loss.item()
        logging.info(f"[train] epoch: {epoch}\tbatch: {step}\tl_noise: {loss_noise.item()}\tl_latent: {loss_latent.item()}\tloss: {loss.item()}")
        # progress_bar.set_postfix({"Loss": epoch_loss / (step + 1)})
        writer.add_scalar(
            "Loss/Train Iteration", loss.item(), epoch * len(train_loader) + step
        )

    return epoch_loss / len(train_loader)


def validate_one_epoch(model, aekl_image, aekl_mask, val_loader, inferer, scheduler, device, epoch, writer):
    
    """Validate the LDM for one epoch."""
    val_loss = 0.0
    model.eval()
    aekl_image.eval()
    aekl_mask.eval()

    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            image, mask  = batch['aug_image'].to(device), batch['aug_mask'].to(device)
            
            # noisy_image, noisy_mask = batch["noisy_image"].to(device), batch["noisy_mask"].to(device)

            with autocast("cuda", enabled = True):
                latent_images               = aekl_image.encode_stage_2_inputs(image).to(device) 
                latent_masks                = aekl_mask.encode_stage_2_inputs(mask).to(device)
                noise                       = torch.randn_like(latent_masks).to(device)
                timesteps                   = torch.randint(0, scheduler.num_train_timesteps, (latent_masks.size(0),), device = device).long()
                z_T                         = scheduler.add_noise(original_samples = latent_masks, noise = noise, timesteps = timesteps)
                
                noise_pred                  = inferer(inputs            = mask,
                                                      noise             = noise,
                                                      diffusion_model   = model,
                                                      timesteps         = timesteps,
                                                      autoencoder_model = aekl_mask,
                                                      condition         = latent_images,
                                                      mode              = "concat")
                
                loss_noise             = F.l1_loss(noise_pred.float(), noise.float())
            
                alpha_bar_T            = scheduler.alphas_cumprod[timesteps][:, None, None, None]
                z_0_pred               = (1 / torch.sqrt(alpha_bar_T)) * (z_T - (torch.sqrt(1 - alpha_bar_T) * noise_pred))
                loss_latent            = F.l1_loss(z_0_pred.float(), latent_masks.float())

                # Then add both losses and backpropogate.
                loss                   = loss_noise + loss_latent
                val_loss               += loss.item()

            logging.info(f"[val] epoch: {epoch}\tbatch: {step}\tl_noise: {loss_noise.item()}\tl_latent: {loss_latent.item()}loss: {loss.item()}")
            writer.add_scalar(
                "Loss/Val Iteration", loss.item(), epoch * len(val_loader) + step
            )

    return val_loss / len(val_loader)


# ------------------------------------------------------------------------------#
# Main Training Loop
# ------------------------------------------------------------------------------#
def main():
    snapshot_dir = LDM_SNAPSHOT_DIR
    check_or_create_folder(snapshot_dir)
    
    device = setup_environment(SEED, snapshot_dir)
    models_dir = os.path.join(snapshot_dir, "models")
    
    check_or_create_folder(models_dir)
    writer = SummaryWriter(f"{snapshot_dir}/log")

    setup_logging(snapshot_dir)
    
    logging.info(f"ldm training parameters: epochs = {N_EPOCHS}, lr = {LR}, timesteps = {NUM_TRAIN_TIMESTEPS}, noise scheduler = {NOISE_SCHEDULER}, scheduler = {SCHEDULER}")
    print(f"Results logged in: {snapshot_dir}, TensorBoard logs in: {snapshot_dir}/log, Models saved in: {models_dir}\n")
    torch.manual_seed(SEED)

    train_loader = get_dataloaders(BASE_DIR, split_ratio = SPLIT_RATIOS, split = "train", trainsize = TRAINSIZE,
                                   batch_size = BATCH_SIZE,
                                   format     = FORMAT,)
    
    val_loader   = get_dataloaders(BASE_DIR, split_ratio = SPLIT_RATIOS, split = "val", trainsize = TRAINSIZE,
                                   batch_size = BATCH_SIZE,
                                   format     = FORMAT,)

    tup_image, tup_mask = load_autoencoder(device, train_loader, image = True, mask = True, epoch_image = 500, epoch_mask = 100)
    unet                = DiffusionModelUNet(**MODEL_PARAMS).to(device)

    scheduler = (
        DDIMScheduler(num_train_timesteps=NUM_TRAIN_TIMESTEPS, schedule=NOISE_SCHEDULER)
        if SCHEDULER == "DDIM"
        
        else DDPMScheduler(
            num_train_timesteps=NUM_TRAIN_TIMESTEPS, schedule=NOISE_SCHEDULER
        )
    )
    
    aekl_image, _         = tup_image  # scale_image compressed
    aekl_mask, scale_mask = tup_mask
    
    aekl_image, aekl_mask = aekl_image.to(device), aekl_mask.to(device)
    inferer               = LatentDiffusionInferer(scheduler = scheduler, scale_factor = scale_mask)
    
    optimizer             = torch.optim.AdamW(
    unet.parameters(), lr = LR, betas = (0.9, 0.999), weight_decay = 0.0001)
    
    scaler     = GradScaler("cuda")
    start_time = time.time()
    
    # Check if a checkpoint exists and resume training
    resume_epoch = 0
    checkpoint   = get_latest_checkpoint(models_dir, prefix = 'model_epoch_')
    if do.RESUME and checkpoint is not None:
        resume_epoch, checkpoint_path = checkpoint
        unet.load_state_dict(torch.load(checkpoint_path, map_location = device, weights_only = True))
        logging.info(f"Resuming training from epoch {resume_epoch}")
        print(f"Resuming training from epoch {resume_epoch}")
    else:
        print("Starting training from scratch.")

    eval_list = ["Epoch", "Train Loss", "Val Loss"]
    prepare_and_write_csv_file(snapshot_dir, eval_list)

    layout = prepare_writer_layout()
    writer.add_custom_scalars(layout)

    for epoch in range(resume_epoch, resume_epoch + N_EPOCHS):        # if RESUME_PATH:
        train_loss, val_loss = None, None

        train_loss = train_one_epoch(unet, aekl_image, aekl_mask, train_loader, optimizer, inferer, scaler, scheduler, device,
                                     epoch,
                                     writer,
        )
        logging.info(f"[train] epoch: {epoch}\tmean train loss: {train_loss}")
        print(f"[train] epoch: {epoch}\tmean train loss: {train_loss:.4f}")
        writer.add_scalar("loss/train epoch", train_loss, epoch)

        if (epoch + 1) % VAL_INTERVAL == 0:
            train_loss = validate_one_epoch(unet, aekl_image, aekl_mask, val_loader, inferer, scheduler, device,
                                            epoch,
                                            writer,
        )

            logging.info(f"[val] epoch: {epoch}\tmean val loss: {val_loss}")
            print(f"[val] epoch: {epoch}\tmean val loss: {val_loss:.4f}")
            writer.add_scalar("loss/val epoch", val_loss, epoch)

        if (epoch + 1) % MODEL_SAVE_INTERVAL == 0:
            model_path = os.path.join(models_dir, f"model_epoch_{epoch}.pth")
            torch.save(unet.state_dict(), model_path)
            logging.info(f"model saved to {model_path}")
            print(f"model saved to {model_path}")

        list_to_write = [epoch, train_loss, val_loss]
        prepare_and_write_csv_file(snapshot_dir, list_to_write)

    print(f"execution time: {(time.time() - start_time) // 60.0} minutes")
    final_model_path = os.path.join(models_dir, "final_model.pth")
    torch.save(unet.state_dict(), final_model_path)
    logging.info(f"model saved to {final_model_path}")
    print(f"model saved to {final_model_path}")
    writer.close()


# ------------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
# ------------------------------------------------------------------------------#
