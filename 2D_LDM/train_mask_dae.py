# ------------------------------------------------------------------------------#
#
# File name                 : train_mask_dae.py
# Purpose                   : Training script for the autoencoderkl for image
# Usage                     : python train_mask_dae.py
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh, Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 202410001@lums.edu.pk, hassan.mohyuddin@lums.edu.pk
#
# Last Date                 : March 11, 2025
# Note                      : Used AdamW optimizer for better performance
# ------------------------------------------------------------------------------#

import csv, logging, os, sys, torch, time

import torch.nn.functional as F

from torch.amp                import autocast, GradScaler
from torch.utils.tensorboard  import SummaryWriter
from tqdm                     import tqdm
from monai.networks.nets      import AutoencoderKL
from generative.networks.nets import PatchDiscriminator
from monai.losses             import PatchAdversarialLoss, PerceptualLoss
from torch.nn                 import MSELoss
from config_ldm_ddpm          import *
from dataset                  import get_dataloaders
from utils                    import *

# ------------------------------------------------------------------------------#
def initialize_components(device, snapshot_dir):
    """
    Initialize autoencoder, discriminator, losses, optimizers, and gradient scalers.

    Parameters:
    - device: The device (CPU or GPU) on which the components will run.
    - snapshot_dir: Directory to save snapshots.

    Returns:
    - autoencoderkl  : AutoencoderKL instance for encoding and decoding images.
    - discriminator  : PatchDiscriminator instance for adversarial loss calculation.
    - dice_loss      : Dice loss instance for segmentation loss computation.
    - perceptual_loss: Perceptual loss instance to compute feature-based reconstruction loss.
    - adv_loss       : Patch adversarial loss instance for discriminator loss computation.
    - optimizer_g    : Optimizer for the autoencoder generator.
    - optimizer_d    : Optimizer for the discriminator.
    - scaler_g       : Gradient scaler for mixed precision training (generator).
    - scaler_d       : Gradient scaler for mixed precision training (discriminator).
    """

    autoencoderkl      = AutoencoderKL(**DAE_MASK_PARAMS).to(device) # for encoding and decoding images
    discriminator      = PatchDiscriminator(**MASK_DISCRIM_PARAMS).to(device) # for adversarial training
    perceptual_loss    = PerceptualLoss(spatial_dims=2, network_type="alex").to(device)
    adv_loss           = PatchAdversarialLoss(criterion="least_squares")
    dice_loss          = DiceLoss(n_classes = 1).to(device)  # lets explore this later
    optimizer_g        = torch.optim.AdamW(autoencoderkl.parameters(),
                                           lr           = LR,
                                           betas        = (0.9, 0.999),
                                           weight_decay = 0.0001)
    optimizer_d        = torch.optim.AdamW(discriminator.parameters(),
                                           lr           = 5 * LR,
                                           betas        = (0.9, 0.999),
                                           weight_decay = 0.0001)
    scaler_g, scaler_d = GradScaler('cuda'), GradScaler('cuda') # for mixed precision training
    output_file        = "DAE_MASK_PARAMS.txt"

    with open(f"{snapshot_dir}/{output_file}", "w") as f: # Write the dictionary to a text file
        f.write("DAE_MASK_PARAMS:\n")
        f.write(f"batch_size: {BATCH_SIZE}\n")
        [f.write(f"{key}: {value}\n") for key, value in DAE_MASK_PARAMS.items()]

    print(f"DAE_MASK_PARAMS written to {output_file}")

    return autoencoderkl, discriminator, dice_loss, perceptual_loss, adv_loss, optimizer_g, optimizer_d, scaler_g, scaler_d

# ------------------------------------------------------------------------------#
def train_one_epoch(autoencoderkl, discriminator, dice_loss, perceptual_loss, adv_loss, train_loader, device, optimizers, scalers, epoch, writer):
    """
    Train the autoencoder and discriminator for one epoch.

    Parameters:
    - autoencoderkl  : The autoencoder model.
    - discriminator  : The discriminator model.
    - dice_loss      : Dice loss instance for segmentation loss computation.
    - perceptual_loss: Perceptual loss instance for feature-based reconstruction loss.
    - adv_loss       : Patch adversarial loss instance for discriminator loss computation.
    - train_loader   : DataLoader for the training set.
    - device         : Device to perform training on (CPU or GPU).
    - optimizers     : Tuple containing optimizers (optimizer_g, optimizer_d).
    - scalers        : Tuple containing gradient scalers (scaler_g, scaler_d).
    - epoch          : Current epoch number.
    - writer         : TensorBoard writer for logging.

    Returns:
    - epoch_total_loss: Average reconstruction loss for the epoch.
    - epoch_gen_loss  : Average generator loss for the epoch.
    - epoch_disc_loss : Average discriminator loss for the epoch.
    """
    optimizer_g, optimizer_d = optimizers
    scaler_g, scaler_d       = scalers
    epoch_total_loss         = 0.0
    epoch_gen_loss           = 0.0
    epoch_disc_loss          = 0.0
    alpha                    = 0.3 # for weight of signed distance map loss with segmentation loss
    mse_loss                 = MSELoss(reduction = 'mean').to(device)
    # progress_bar             = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
    # progress_bar.set_description(f"Epoch {epoch + 1}/{N_EPOCHS}")

    for step, batch in enumerate(train_loader): # Training loop
        # clean_mask = batch['clean_mask'].to(device)
        # noisy_mask = batch["noisy_mask"].to(device)

        # ----------------------------------------------------------------------#
        # Train Autoencoder
        # ----------------------------------------------------------------------#
        autoencoderkl.train(); discriminator.eval(); optimizer_g.zero_grad(set_to_none = True)
        with autocast('cuda', enabled = True): 
            # reconstruction = autoencoderkl.decode_stage_2_outputs(autoencoderkl.encode_stage_2_inputs(noisy_mask)) # reconstruction is logits
            clean_mask, noisy_mask = batch["clean_mask"].to(device), batch["noisy_mask"].to(device)
            reconstruction, z_mu, z_sigma = autoencoderkl(noisy_mask) # reconstruction is logitsreconstruction, z_mu, z_sigma = autoencoderkl(noisy_mask) # reconstruction is logitsreconstruction, z_mu, z_sigma = autoencoderkl(noisy_mask) # reconstruction is logitsreconstruction, z_mu, z_sigma = autoencoderkl(noisy_mask) # reconstruction is logits
            recon_sig                     = torch.sigmoid(reconstruction.to(device))
           
            with torch.no_grad(): # Compute Signed Distance Maps (SDMs) of the reconstruction and the original image
                recon_bin = (recon_sig > 0.5).float()
                clean_sdm = compute_sdm(clean_mask.detach().cpu().numpy())      # SDM for ground truth mask
                recon_sdm = compute_sdm(recon_bin.detach().cpu().numpy())  # SDM for predicted mask

            # Compute losses - replace total_loss with screenshot (use alpha = 0.3)
            
            seg_loss   = dice_loss(recon_sig, clean_mask)  # Segmentation loss summed across batch
            sdm_loss   = mse_loss(recon_sdm, clean_sdm)
            total_loss = seg_loss + (alpha * sdm_loss)
            # pass reconstruction_sdm and images_sdm to the perceptual loss function to ensure continous loss training
            p_loss     = perceptual_loss(recon_sdm.float(), clean_sdm.float()) # sdm_reconstruction and sdm_clean
            kl_loss    = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim = [1, 2, 3])
            kl_loss    = torch.sum(kl_loss) / kl_loss.shape[0]
            loss_g     = total_loss + (KL_WEIGHT * kl_loss) + (PERCEPTUAL_WEIGHT * p_loss)

            # placeholder values for gen_loss 
            # gen_loss   = 0.0
            # if epoch >= WARM_UP_EPOCHS:
            #     logits_fake  = discriminator(recon_sdm.contiguous().float())[-1] 
            #     gen_loss     = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
            #     loss_g      += ADV_WEIGHT * gen_loss

        # Backpropagation and optimization for autoencoder
        scaler_g.scale(total_loss).backward()
        scaler_g.step(optimizer_g)
        scaler_g.update()

        # ----------------------------------------------------------------------#
        # Train Discriminator
        # ----------------------------------------------------------------------#
        # autoencoderkl.eval(); discriminator.train()
    
        # with torch.no_grad():
        #     reconstruction, _, _ = autoencoderkl(noisy_mask)
        #     reconstruction       = (torch.sigmoid(reconstruction) > 0.5).float().to(device) # binarize the output
        #     recon_sdm            = compute_sdm(reconstruction.detach().cpu().numpy())

        # # placeholder values for disc_loss 
        # disc_loss = 0.0
        # if epoch >= WARM_UP_EPOCHS:
        #     optimizer_d.zero_grad(set_to_none = True)
        #     with autocast('cuda', enabled = True):
        #         logits_fake        = discriminator(recon_sdm.contiguous().detach())[-1] # reconstruction_sdm
        #         loss_d_fake        = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
        #         logits_real        = discriminator(clean_sdm.contiguous().detach())[-1] # sdm_clean
        #         loss_d_real        = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
        #         disc_loss          = 0.5 * (loss_d_fake + loss_d_real)

        #     # Backpropagation and optimization for discriminator
        #     scaler_d.scale(disc_loss).backward()
        #     scaler_d.step(optimizer_d)
        #     scaler_d.update()

        # ----------------------------------------------------------------------#
        # Update Loss Accumulators
        # ----------------------------------------------------------------------#
        epoch_total_loss     += total_loss.item()
        # if epoch >= WARM_UP_EPOCHS:
        #     epoch_gen_loss   += gen_loss.item()
        #     epoch_disc_loss  += disc_loss.item()
        #     logging.info(f'[train] epoch: {epoch}\tbatch: {step}\tseg loss: {seg_loss.item()}\tsdm loss: {sdm_loss.item()}\ttotal loss: {total_loss.item()}\tgen loss: {gen_loss.item()}\tdisc loss: {disc_loss.item()}')
        #     # progress_bar.set_postfix({"Reconstruction Loss": epoch_total_loss / (step + 1),
        #     #                           "Generator Loss"     : epoch_gen_loss / (step + 1),
        #     #                           "Discriminator Loss" : epoch_disc_loss / (step + 1)}) 
        # else:
        #     logging.info(f'[train] epoch: {epoch}\tbatch: {step}\tseg loss: {seg_loss.item()}\tsdm loss: {sdm_loss.item()}\ttotal loss: {total_loss.item()}')
        #     # progress_bar.set_postfix({"Reconstruction Loss": epoch_total_loss / (step + 1)}) 
        logging.info(f'[train] epoch: {epoch}\tbatch: {step}\tseg loss: {seg_loss.item()}\tsdm loss: {sdm_loss.item()}\ttotal loss: {total_loss.item()}')
        writer.add_scalar("Loss/Train Iteration", total_loss.item(), epoch * len(train_loader) + step)

    return [loss / len(train_loader) for loss in [epoch_total_loss, epoch_gen_loss, epoch_disc_loss]]

# ------------------------------------------------------------------------------#
def validate_and_save(autoencoderkl, val_loader, dice_loss, device, models_dir, epoch, writer):
    """
    Validate the autoencoder on the validation dataset and save the model.

    Parameters:
    - autoencoderkl : The autoencoder model.
    - val_loader    : DataLoader for the validation set.
    - dice_loss     : Dice loss function.
    - device        : Device to perform validation on (CPU or GPU).
    - models_dir    : Directory to save the model.
    - epoch         : Current epoch number.
    - writer        : TensorBoard writer for logging validation loss.

    Returns:
    - val_loss_avg  : Average validation loss across all batches.
    """
    autoencoderkl.eval()
    epoch_total_loss = 0.0
    mse_loss         = MSELoss(reduction = 'mean')
    alpha            = 0.3

    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            clean_mask, noisy_mask = batch["clean_mask"].to(device), batch["noisy_mask"].to(device)
            with autocast('cuda', enabled = True):
                val_reconstruction, _, _  = autoencoderkl(clean_mask)
                clean_mask, noisy_mask    = batch["clean_mask"].to(device), batch["noisy_mask"].to(device)
                val_reconstruction, _, _  = autoencoderkl(noisy_mask)
                val_reconstruction        = (torch.sigmoid(val_reconstruction) > 0.5).float().to(device) # binarize the output 
                clean_sdm                 = compute_sdm(clean_mask.detach().cpu().numpy()) # SDM for ground truth mask
                recon_sdm                 = compute_sdm(val_reconstruction.detach().cpu().numpy()) # SDM for predicted mask
                seg_loss                  = dice_loss(val_reconstruction, clean_mask) # Segmentation loss summed across batch
                sdm_loss                  = mse_loss(recon_sdm, clean_sdm)
                total_loss                = seg_loss + (alpha * sdm_loss)
                epoch_total_loss         += total_loss.item()

            logging.info(f'[val] epoch: {epoch}\tbatch: {step}\tseg loss: {seg_loss.item()}\tsdm loss: {sdm_loss.item()}\ttotal loss: {total_loss.item()}')
            writer.add_scalar("Loss/Val Iteration", total_loss.item(), epoch * len(val_loader) + step)

    if (epoch + 1) % MODEL_SAVE_INTERVAL == 0: # Save the model at specified intervals
        model_path = os.path.join(models_dir, f"autoencoder_epoch_{epoch + 1}.pth")
        torch.save(autoencoderkl.state_dict(), model_path)
        logging.info(f'model saved to {model_path}')
        print(f'model saved to {model_path}')

    return epoch_total_loss / len(val_loader)

# ------------------------------------------------------------------------------#
def main():
    """
    Main function for training the autoencoder component of the 2D Latent Diffusion Model (LDM).
    Sets up directories, initializes components, and manages the training loop.
    """
    device       = "cuda" if torch.cuda.is_available() else "cpu"
    snapshot_dir = DAE_MASK_SNAPSHOT_DIR; os.makedirs(snapshot_dir, exist_ok=True)
    models_dir   = os.path.join(snapshot_dir, "models"); os.makedirs(models_dir, exist_ok=True)
    writer       = SummaryWriter(f"{snapshot_dir}/log")
    autoencoderkl, discriminator, dice_loss, perceptual_loss, adv_loss, optimizer_g, optimizer_d, scaler_g, scaler_d = initialize_components(device, snapshot_dir)

    logging.basicConfig(filename = os.path.join(snapshot_dir, 'logs.txt'),
                        level    = logging.INFO,                            # Log message with level INFO  or higher
                        format   = "[%(asctime)s.%(msecs)03d] %(message)s", # Format of the log message
                        datefmt  = "%H:%M:%S")                              # Format of the timestamp
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(f"dae_mask training parameters: epochs = {N_EPOCHS}, lr = {LR}, timesteps = {NUM_TRAIN_TIMESTEPS}, noise scheduler = {NOISE_SCHEDULER}, scheduler = {SCHEDULER}")
    print(f"Results logged in: {snapshot_dir}")
    print(f"TensorBoard logs in: {snapshot_dir}/log")
    print(f"Models saved in: {models_dir}\n")
    torch.manual_seed(SEED)

    train_loader = get_dataloaders(BASE_DIR, split_ratio = SPLIT_RATIOS, split = "train", trainsize = TRAINSIZE, batch_size = BATCH_SIZE, format = FORMAT)
    val_loader   = get_dataloaders(BASE_DIR, split_ratio = SPLIT_RATIOS, split = "val", trainsize = TRAINSIZE, batch_size = BATCH_SIZE, format = FORMAT)
    start_time   = time.time()
    
    eval_list    = ['Epoch', 'Train Recon Loss', 'Train Gen Loss', 'Train Disc Loss', 'Val Recon Loss']
    prepare_and_write_csv_file(snapshot_dir, eval_list)
    
    layout       = prepare_writer_layout()
    writer.add_custom_scalars(layout)
        
    for epoch in range(N_EPOCHS): # Training loop
        train_total_loss, train_gen_loss, train_disc_loss, val_total_loss = None, None, None, None

        with open(os.path.join(snapshot_dir, 'logs.csv'), 'a') as csvfile:

            train_total_loss, train_gen_loss, train_disc_loss = train_one_epoch(
                autoencoderkl, discriminator, dice_loss, perceptual_loss, adv_loss, train_loader, device, 
                optimizers = (optimizer_g, optimizer_d), scalers = (scaler_g, scaler_d), epoch = epoch, writer = writer
            )
            logging.info(f'[train] epoch: {epoch}\tmean total loss: {train_total_loss}\tmean gen loss: {train_gen_loss}\tmean disc loss: {train_disc_loss}')
            print(f'[train] epoch: {epoch}\tmean total loss: {train_total_loss:.4f}')
            writer.add_scalar("loss/train epoch", train_total_loss, epoch)

            if epoch >= WARM_UP_EPOCHS:
                logging.info(f'[train] epoch: {epoch}\tmean gen loss: {train_gen_loss}\tmean disc loss: {train_disc_loss}')
                print(f'[train] epoch: {epoch}\tmean gen loss: {train_gen_loss:.4f}\tmean disc loss: {train_disc_loss:.4f}')

            if (epoch + 1) % VAL_INTERVAL == 0: # Validate and save model at intervals
                val_total_loss = validate_and_save(autoencoderkl, val_loader, dice_loss, device, models_dir, epoch, writer)
                logging.info(f'[val] epoch: {epoch}\tmean total loss: {val_total_loss}')
                print(f'[val] epoch: {epoch}\tmean total loss: {val_total_loss:.4f}')
                writer.add_scalar("loss/val epoch", val_total_loss, epoch)

            list_to_write = [epoch, train_total_loss, train_gen_loss, train_disc_loss, val_total_loss]
            prepare_and_write_csv_file(snapshot_dir, list_to_write)

    print(f'execution time: {time.time() - start_time} seconds')
    final_model_path = os.path.join(models_dir, 'final_model.pth')
    torch.save(autoencoderkl.state_dict(), final_model_path)
    logging.info(f'model saved to {final_model_path}')
    print(f'model saved to {final_model_path}')
    writer.close()

# ------------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
# ------------------------------------------------------------------------------#
