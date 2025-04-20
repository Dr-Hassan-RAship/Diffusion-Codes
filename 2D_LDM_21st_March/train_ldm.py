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

import csv, logging, os, sys, torch, time

import torch.nn.functional as F

from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler
from generative.inferers import LatentDiffusionInferer
from config import *
from dataset import get_dataloaders
from inference_ldm_utils import *
from utils import *


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


def train_one_epoch(
    model,
    dae_image,
    dae_mask,
    train_loader,
    optimizer,
    inferer,
    scaler,
    scheduler,
    device,
    epoch,
    writer,
):
    """Train the LDM for one epoch."""
    epoch_loss = 0.0
    # progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
    # progress_bar.set_description(f"Epoch {epoch + 1}/{N_EPOCHS}")
    model.train()
    dae_image.eval()
    dae_mask.eval()

    for step, batch in enumerate(train_loader):
        clean_image, clean_mask = batch["clean_image"].to(device), batch[
            "clean_mask"
        ].to(device)
        # noisy_image, noisy_mask = batch["noisy_image"].to(device), batch["noisy_mask"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=True):
            # Assumed Pipeline:
            # 1. Encode images and masks
            # 2. Get random noise same shape as latent_masks
            # 3. Get timesteps corresponding to latent masks
            # 4. Call inferer with latent_images as conditioning and mode as 'concat' and autoencoder model as dae_mask

            latent_images               = dae_image.encode_stage_2_inputs(clean_image).to(device)
            latent_masks                = dae_mask.encode_stage_2_inputs(clean_mask).to(device)
            noise                       = torch.randn_like(latent_masks).to(device) # (B, C, H, W)
            timesteps                   = torch.randint(0, scheduler.num_train_timesteps, (latent_masks.size(0),), device = device).long()
            z_T                         = scheduler.add_noise(original_samples = latent_masks, noise = noise, timesteps = timesteps)
            
            #[talha] Make sure z_t is same as the z_t we could have returned from inferer. 
            noise_pred                  = inferer(inputs            = clean_mask,
                                                  noise             = noise,
                                                  diffusion_model   = model,
                                                  timesteps         = timesteps,
                                                  autoencoder_model = dae_mask,
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


def validate_one_epoch(
    model, dae_image, dae_mask, val_loader, inferer, scheduler, device, epoch, writer
):
    """Validate the LDM for one epoch."""
    val_loss = 0.0
    model.eval()
    dae_image.eval()
    dae_mask.eval()

    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            clean_image, clean_mask = batch["clean_image"].to(device), batch[
                "clean_mask"
            ].to(device)
            # noisy_image, noisy_mask = batch["noisy_image"].to(device), batch["noisy_mask"].to(device)

            with autocast("cuda", enabled = True):
                latent_images               = dae_image.encode_stage_2_inputs(clean_image).to(device), 
                latent_masks                = dae_mask.encode_stage_2_inputs(clean_mask).to(device)
                noise                       = torch.randn_like(latent_masks).to(device)
                timesteps                   = torch.randint(0, scheduler.num_train_timesteps, (latent_masks.size(0),), device = device).long()
                z_T                         = scheduler.add_noise(original_samples = latent_masks, noise = noise, timesteps = timesteps)
                noise_pred                  = inferer(inputs            = clean_mask,
                                                      noise             = noise,
                                                      diffusion_model   = model,
                                                      timesteps         = timesteps,
                                                      autoencoder_model = dae_mask,
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

    logging.basicConfig(
        filename=os.path.join(snapshot_dir, "logs.txt"),
        level=logging.INFO,  # Log message with level INFO  or higher
        format="[%(asctime)s.%(msecs)03d] %(message)s",  # Format of the log message
        datefmt="%H:%M:%S",
    )  # Format of the timestamp
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(
        f"ldm training parameters: epochs = {N_EPOCHS}, lr = {LR}, timesteps = {NUM_TRAIN_TIMESTEPS}, noise scheduler = {NOISE_SCHEDULER}, scheduler = {SCHEDULER}"
    )
    print(
        f"Results logged in: {snapshot_dir}, TensorBoard logs in: {snapshot_dir}/log, Models saved in: {models_dir}\n"
    )
    torch.manual_seed(SEED)

    train_loader = get_dataloaders(
        BASE_DIR,
        split_ratio=SPLIT_RATIOS,
        split="train",
        trainsize=TRAINSIZE,
        batch_size=BATCH_SIZE,
        format=FORMAT,
    )
    val_loader = get_dataloaders(
        BASE_DIR,
        split_ratio=SPLIT_RATIOS,
        split="val",
        trainsize=TRAINSIZE,
        batch_size=BATCH_SIZE,
        format=FORMAT,
    )
    tup_image, tup_mask = load_autoencoder(
        device, train_loader, image=True, mask=True, epoch_image=250, epoch_mask=200
    )
    unet = DiffusionModelUNet(**MODEL_PARAMS).to(device)

    # if RESUME_PATH:
    #     epoch_add = int(RESUME_PATH.split("_")[-1][:-4])
    #     writer = SummaryWriter(f"{snapshot_dir}/log")
    #     unet.load_state_dict(
    #         torch.load(RESUME_PATH, map_location=device, weights_only=True)
    #     )
    #     unet.to(device)

    scheduler = (
        DDIMScheduler(num_train_timesteps=NUM_TRAIN_TIMESTEPS, schedule=NOISE_SCHEDULER)
        if SCHEDULER == "DDIM"
        else DDPMScheduler(
            num_train_timesteps=NUM_TRAIN_TIMESTEPS, schedule=NOISE_SCHEDULER
        )
    )
    dae_image, _ = tup_image  # scale_image compressed
    dae_mask, scale_mask = tup_mask
    dae_image, dae_mask = dae_image.to(device), dae_mask.to(device)
    inferer = LatentDiffusionInferer(scheduler=scheduler, scale_factor=scale_mask)
    optimizer = torch.optim.AdamW(
        unet.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=0.0001
    )
    scaler = GradScaler("cuda")
    start_time = time.time()

    eval_list = ["Epoch", "Train Loss", "Val Loss"]
    prepare_and_write_csv_file(snapshot_dir, eval_list)

    layout = prepare_writer_layout()
    writer.add_custom_scalars(layout)

    for epoch in range(N_EPOCHS):  # Training Loop
        # if RESUME_PATH:
        #     epoch += epoch_add + 1

        train_loss, val_loss = None, None

        train_loss = train_one_epoch(
            unet,
            dae_image,
            dae_mask,
            train_loader,
            optimizer,
            inferer,
            scaler,
            scheduler,
            device,
            epoch,
            writer,
        )
        logging.info(f"[train] epoch: {epoch}\tmean train loss: {train_loss}")
        print(f"[train] epoch: {epoch}\tmean train loss: {train_loss:.4f}")
        writer.add_scalar("loss/train epoch", train_loss, epoch)

        if (epoch + 1) % VAL_INTERVAL == 0:
            val_loss = validate_one_epoch(
                unet,
                dae_image,
                dae_mask,
                val_loader,
                inferer,
                scheduler,
                device,
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

    print(f"execution time: {time.time() - start_time} seconds")
    final_model_path = os.path.join(models_dir, "final_model.pth")
    torch.save(unet.state_dict(), final_model_path)
    logging.info(f"model saved to {final_model_path}")
    print(f"model saved to {final_model_path}")
    writer.close()


# ------------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
# ------------------------------------------------------------------------------#
