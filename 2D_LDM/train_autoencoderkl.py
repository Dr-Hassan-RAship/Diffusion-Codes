# ------------------------------------------------------------------------------#
#
# File name                 : train_image_dae.py
# Purpose                   : Training script for the autoencoderKL for both image and mask (without discriminator & generator)
# Usage                     : python train_autoencoderkl.py --mode 'image' [--resume]
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh, Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 202410001@lums.edu.pk, hassan.mohyuddin@lums.edu.pk
#
# Last Date                 : March 24, 2025
# Note                      : Used AdamW optimizer for better performance
# ------------------------------------------------------------------------------#

import logging, os, torch, argparse, time, re
import torch.nn.functional    as F

from torch.amp                import autocast, GradScaler
from torch.utils.tensorboard  import SummaryWriter
from monai.networks.nets      import AutoencoderKL
from config_ldm_ddpm          import *
from dataset                  import get_dataloaders
from utils                    import *
# ------------------------------------------------------------------------------#
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Training script for AutoencoderKL (Image/Mask)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["image", "mask"],
        required=True,
        help="Specify whether to run training for images or masks",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint if available"
    )
    return parser.parse_args()

# ------------------------------------------------------------------------------#
def get_latest_checkpoint(models_dir):
    """
    Search for the latest checkpoint in the models_dir. Checkpoint files should follow the pattern:
    'autoencoder_epoch_{epoch_number}.pth'
    
    Returns:
        (epoch, filepath) of the latest checkpoint, or None if not found.
    """
    checkpoint = None
    latest_epoch = -1
    for filename in os.listdir(models_dir):
        match = re.match(r'autoencoder_epoch_(\d+)\.pth', filename)
        if match:
            epoch = int(match.group(1))
            if epoch > latest_epoch:
                latest_epoch = epoch
                checkpoint = filename
    if checkpoint is not None:
        return latest_epoch, os.path.join(models_dir, checkpoint)
    else:
        return None

# ------------------------------------------------------------------------------#
def initialize_components(device, snapshot_dir):
    """
    Initialize autoencoder, optimizer, and gradient scaler.
    Writes training parameters to a text file.
    """
    autoencoderkl = AutoencoderKL(**AUTOENCODERKL_PARAMS).to(device)
    optimizer = torch.optim.AdamW(autoencoderkl.parameters(),
                                  lr=LR,
                                  betas=(0.9, 0.999),
                                  weight_decay=0.0001)
    scaler = GradScaler('cuda')
    output_file = "dae_image_params.txt"

    with open(os.path.join(snapshot_dir, output_file), "a") as f:
        f.write("DAE_IMAGE_PARAMS:\n")
        f.write(f"batch_size: {BATCH_SIZE}\n")
        for key, value in AUTOENCODERKL_PARAMS.items():
            f.write(f"{key}: {value}\n")
    print(f"DAE_IMAGE_PARAMS appended to {output_file}")

    return autoencoderkl, optimizer, scaler

# ------------------------------------------------------------------------------#
def train_one_epoch(autoencoderkl, train_loader, device, optimizer, scaler, epoch, writer, mode):
    """
    Train the autoencoder for one epoch.
    """
    epoch_recon_loss = 0.0
    for step, batch in enumerate(train_loader):
        gt_input = batch["aug_image"].to(device) if mode == 'image' else batch["aug_mask"].to(device)

        autoencoderkl.train()
        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda', enabled=True):
            reconstruction, _, _ = autoencoderkl(gt_input)
            reconstruction = torch.tanh(reconstruction).to(device)
            recon_loss = F.l1_loss(reconstruction.float(), gt_input.float())

        scaler.scale(recon_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_recon_loss += recon_loss.item()
        logging.info(f'[train] epoch: {epoch}\tbatch: {step}\trecon loss: {recon_loss.item():.4f}')
        writer.add_scalar("Loss/Train Iteration", recon_loss.item(), epoch * len(train_loader) + step)

    return epoch_recon_loss / len(train_loader)

# ------------------------------------------------------------------------------#
def validate_and_save(autoencoderkl, val_loader, device, models_dir, epoch, writer, mode):
    """
    Validate the autoencoder on the validation dataset and save the model checkpoint.
    """
    autoencoderkl.eval()
    epoch_recon_loss = 0.0
    with torch.no_grad():
        for val_step, val_batch in enumerate(val_loader):
            gt_input = val_batch["aug_image"].to(device) if mode == 'image' else val_batch["aug_mask"].to(device)
            with autocast('cuda', enabled=True):
                val_reconstruction, _, _ = autoencoderkl(gt_input)
                val_reconstruction = torch.tanh(val_reconstruction).to(device)
                recon_loss = F.l1_loss(gt_input.float(), val_reconstruction.float())
            epoch_recon_loss += recon_loss.item()
            logging.info(f'[val] epoch: {epoch}\tbatch: {val_step}\trecon_loss: {recon_loss:.4f}')
            writer.add_scalar("Loss/Val Iteration", recon_loss.item(), epoch * len(val_loader) + val_step)

    if (epoch + 1) % MODEL_SAVE_INTERVAL == 0:
        model_path = os.path.join(models_dir, f'autoencoderkl_epoch_{epoch + 1}.pth')
        torch.save(autoencoderkl.state_dict(), model_path)
        logging.info(f'model saved to {model_path}')
        print(f'model saved to {model_path}')

    return epoch_recon_loss / len(val_loader)

# ------------------------------------------------------------------------------#
def main():
    args = parse_arguments()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    snapshot_dir = AEKL_IMAGE_SNAPSHOT_DIR if args.mode == 'image' else AEKL_MASK_SNAPSHOT_DIR
    
    os.makedirs(snapshot_dir, exist_ok=True)
    models_dir = os.path.join(snapshot_dir, "models")
    
    os.makedirs(models_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(snapshot_dir, "log"))
    
    autoencoderkl, optimizer, scaler = initialize_components(device, snapshot_dir)
    
    setup_logging(snapshot_dir)
    logging.info(f"dae_image training parameters: epochs = {N_EPOCHS}, lr = {LR}, timesteps = {NUM_TRAIN_TIMESTEPS}, noise scheduler = {NOISE_SCHEDULER}, scheduler = {SCHEDULER}")
    
    print(f"Results logged in: {snapshot_dir}")
    print(f"TensorBoard logs in: {snapshot_dir}/log")
    print(f"Models saved in: {models_dir}\n")
    torch.manual_seed(SEED)

    train_loader = get_dataloaders(BASE_DIR, split_ratio=SPLIT_RATIOS, split='train', trainsize=TRAINSIZE, batch_size=BATCH_SIZE, format=FORMAT)
    val_loader   = get_dataloaders(BASE_DIR, split_ratio=SPLIT_RATIOS, split='val', trainsize=TRAINSIZE, batch_size=BATCH_SIZE, format=FORMAT)
    start_time   = time.time()

    # Check if a checkpoint exists and resume training
    resume_epoch = 0
    checkpoint = get_latest_checkpoint(models_dir)
    if args.resume and checkpoint is not None:
        resume_epoch, checkpoint_path = checkpoint
        autoencoderkl.load_state_dict(torch.load(checkpoint_path, map_location=device))
        logging.info(f"Resuming training from epoch {resume_epoch}")
        print(f"Resuming training from epoch {resume_epoch}")
    else:
        print("Starting training from scratch.")

    # Prepare CSV logging and TensorBoard custom scalars
    eval_list = ['Epoch', 'Train Recon Loss', 'Val Recon Loss']
    prepare_and_write_csv_file(snapshot_dir, eval_list)
    layout = prepare_writer_layout()
    writer.add_custom_scalars(layout)

    # Run training loop (if resuming, continue from resume_epoch+1)
    for epoch in range(resume_epoch, resume_epoch + N_EPOCHS):
        train_recon_loss = train_one_epoch(autoencoderkl, train_loader, device, optimizer, scaler, epoch, writer, args.mode)
        logging.info(f'[train] epoch: {epoch}\tmean recon loss: {train_recon_loss:.4f}')
        print(f'[train] epoch: {epoch}\tmean recon loss: {train_recon_loss:.4f}')
        writer.add_scalar("loss/train epoch", train_recon_loss, epoch)

        if (epoch + 1) % VAL_INTERVAL == 0:
            val_recon_loss = validate_and_save(autoencoderkl, val_loader, device, models_dir, epoch, writer, args.mode)
            logging.info(f'[val] epoch: {epoch}\tmean recon loss: {val_recon_loss:.4f}')
            print(f'[val] epoch: {epoch}\tmean recon loss: {val_recon_loss:.4f}')
            writer.add_scalar("loss/val epoch", val_recon_loss, epoch)

        list_to_write = [epoch, train_recon_loss, val_recon_loss]
        prepare_and_write_csv_file(snapshot_dir, list_to_write)
            
    print(f'execution time: {(time.time() - start_time) // 60.0} minutes')
    final_model_path = os.path.join(models_dir, 'final_model.pth')
    torch.save(autoencoderkl.state_dict(), final_model_path)
    logging.info(f'model saved to {final_model_path}')
    print(f'model saved to {final_model_path}')
    writer.close()
    
#------------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
    
#------------------------------------------------------------------------------#