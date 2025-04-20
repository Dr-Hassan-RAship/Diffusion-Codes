# ------------------------------------------------------------------------------#
#
# File name                 : train_image_aekl.py
# Purpose                   : Training script for the autoencoderKL for both image and mask (without discriminator & generator)
# Usage                     : python train_autoencoderkl.py --mode 'image' [--resume]
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh, Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 202410001@lums.edu.pk, hassan.mohyuddin@lums.edu.pk
#
# Last Date                 : March 24, 2025
# Note                      : Used AdamW optimizer for better performance
# ------------------------------------------------------------------------------#

import logging, os, torch, argparse, time
import torch.nn.functional    as F

from torch.amp                import autocast, GradScaler
from torch.utils.tensorboard  import SummaryWriter
from monai.networks.nets      import AutoencoderKL
from config          import *
from dataset                  import *
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
def initialize_components(device, snapshot_dir, mode):
    """
    Initialize autoencoder, optimizer, and gradient scaler.
    Writes training parameters to a text file.
    """
    autoencoderkl = AutoencoderKL(**AUTOENCODERKL_PARAMS).to(device)
    optimizer     = torch.optim.AdamW(autoencoderkl.parameters(),
                                  lr=LR,
                                  betas=(0.9, 0.999),
                                  weight_decay=0.0001)
    scaler        = GradScaler('cuda')
    output_file   = f"aekl_{mode}_params.txt"

    txt_path      = os.path.join(snapshot_dir, output_file)
    
    if not os.path.exists(txt_path):
        with open(txt_path, "a") as f:
            f.write("AUTOENCODERKL_PARAMS:\n")
            f.write(f"batch_size: {BATCH_SIZE}\n")
            for key, value in AUTOENCODERKL_PARAMS.items():
                f.write(f"{key}: {value}\n")
        print(f"AUTOENCODERKL_PARAMS added to {output_file}")
    else:
        print(f'Params txt file already exists!')

    return autoencoderkl, optimizer, scaler

# ------------------------------------------------------------------------------#
def train_one_epoch(autoencoderkl, train_loader, device, optimizer, scaler, epoch, writer, mode):
    """
    Train the autoencoder for one epoch.
    """
    epoch_recon_loss = 0.0
    autoencoderkl.train(); optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(train_loader):
        gt_input = batch["aug_image"].to(device) if mode == 'image' else batch["aug_mask"].to(device)
        
        with autocast('cuda', enabled=True):
            reconstruction, _, _ = autoencoderkl(gt_input)
            reconstruction       = torch.tanh(reconstruction).to(device)
            recon_loss           = F.l1_loss(reconstruction.float(), gt_input.float())

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
    
    epoch_recon_loss = 0.0
    autoencoderkl.eval()
    with torch.no_grad():
        for val_step, val_batch in enumerate(val_loader):
            gt_input = val_batch["aug_image"].to(device) if mode == 'image' else val_batch["aug_mask"].to(device)

            with autocast('cuda', enabled=True):
                val_reconstruction, _, _ = autoencoderkl(gt_input)
                val_reconstruction       = torch.tanh(val_reconstruction).to(device)
                recon_loss               = F.l1_loss(gt_input.float(), val_reconstruction.float())
                
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

    # Setup Logging
    setup_logging(snapshot_dir)
    if args.resume:
        print('--------------------------------------------------TRAINING RESUMING!---------------------------------------------------------------------')
        # Prepare Tensorboard Writer
        writer = SummaryWriter(os.path.join(snapshot_dir, "log_resume"))
    else:
        # Prepare Tensorboard Writer
        writer = SummaryWriter(os.path.join(snapshot_dir, "log"))
    logging.info(f"aekl_{args.mode} training parameters: epochs = {N_EPOCHS}, lr = {LR}, timesteps = {NUM_TRAIN_TIMESTEPS}, noise scheduler = {NOISE_SCHEDULER}, scheduler = {SCHEDULER}")
    
    print(f"Results logged in: {snapshot_dir}")
    print(f"TensorBoard logs in: {snapshot_dir}/log")
    print(f"Models saved in: {models_dir}\n")
    torch.manual_seed(SEED)
    
    # Initialize components
    autoencoderkl, optimizer, scaler = initialize_components(device, snapshot_dir, args.mode)
    
    # Dataloaders
    train_loader = get_dataloaders(
                BASE_DIR, split_ratio = SPLIT_RATIOS, split = 'train',
                trainsize = TRAINSIZE, batch_size = BATCH_SIZE, format = FORMAT
            )
    val_loader   = get_dataloaders(
        BASE_DIR, split_ratio = SPLIT_RATIOS, split = 'val',
        trainsize = TRAINSIZE, batch_size = BATCH_SIZE, format = FORMAT
    )
    # Resume or start fresh
    resume_epoch, autoencoderkl, train_loader_override, val_loader_override, optimizer_override = validate_resume_training(
    autoencoderkl, snapshot_dir, models_dir, mode = args.mode, device = device, args = args, prefix = 'autoencoderkl_epoch_')

    # Fall back to default loaders if not overridden
    train_loader = train_loader_override or train_loader
    val_loader   = val_loader_override or val_loader
    optimizer    = optimizer_override  or optimizer
    
    # Prepare CSV logging and TensorBoard custom scalars
    eval_list = ['Epoch', 'Train Recon Loss', 'Val Recon Loss']
    if not os.path.exists(os.path.join(snapshot_dir, 'logs.csv')):
        prepare_and_write_csv_file(snapshot_dir, eval_list, write_header=True)
    writer.add_custom_scalars(prepare_writer_layout())

    # Run training loop (if resuming, continue from resume_epoch+1)
    # Launch tensorboard
    launch_tensorboard(os.path.join(snapshot_dir, "log_resume"))
    start_time   = time.time()
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
        prepare_and_write_csv_file(snapshot_dir, list_to_write, write_header = False)
            
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