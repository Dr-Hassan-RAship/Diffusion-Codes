# ------------------------------------------------------------------------------#
#
# File name                 : train.py
# Purpose                   : Training script for Latent Diffusion Model (LDM) segmentation
# Usage                     : python train.py
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh, Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 202410001@lums.edu.pk, hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : April 28, 2025
# ------------------------------------------------------------------------------#

import os, time, argparse, logging, torch

from torch.amp                      import autocast, GradScaler
from torch.optim                    import AdamW
from torch.utils.tensorboard        import SummaryWriter

from diffusers.optimization         import get_cosine_schedule_with_warmup
from config                         import *
from dataset                        import get_dataloaders
from architectures                  import *
from utils                          import *
from modeling_utils                 import *

# ------------------------------------------------------------------------------ #
def initialize_new_session(device):
    """Creates a new model and optimizer from scratch."""
    model     = LDM_Segmentor().to(device)
    optimizer = AdamW(model.parameters(), lr=OPT['lr'], betas=OPT['betas'], weight_decay=OPT['weight_decay'])
    scheduler = None 
    # Scheduler
    if USE_SCHEDULER:
        total_steps  = (SPLIT_RATIOS[0] // BATCH_SIZE) * N_EPOCHS # for periteration
        # total_steps  = N_EPOCHS
        warmup_steps = int(OPT['warmup_ratio'] * total_steps)
        scheduler    = get_cosine_schedule_with_warmup(optimizer = optimizer, num_warmup_steps = warmup_steps,
                                                    num_cycles = OPT['period'], num_training_steps=total_steps
        )
    
    return model, optimizer, scheduler, 0

# ------------------------------------------------------------------------------ #
def trainer(model, optimizer, scheduler, train_loader, val_loader, device, scaler, snapshot_dir, writer, resume_epoch):
    
    for epoch in range(resume_epoch, resume_epoch + N_EPOCHS):
        print(f"\n🚀 Starting Epoch {epoch}")

        torch.cuda.empty_cache(); model.train(); epoch_loss = 0.0
        # ---- Training ---- #
        for step, batch in enumerate(train_loader):
            image = batch["aug_image"].to(device, dtype = torch.float16)
            mask  = batch["aug_mask"].to(device, dtype = torch.float16)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device, enabled=True):
                _, loss  = model(image, mask) # [nehal flag version] 

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            writer.add_scalar("Loss/Train Iteration", loss.item(), epoch * len(train_loader) + step)
            logging.info(f"[train] epoch: {epoch} batch: {step} loss: {loss.item():.4f}")
            
            if USE_SCHEDULER:
                scheduler.step()        
                writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)
            
        epoch_loss /= len(train_loader)
        # logging.info(f"[train] epoch: {epoch} mean loss: {epoch_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        logging.info(f"[train] epoch: {epoch} mean loss: {epoch_loss:.4f}")
        writer.add_scalar("loss/train epoch", epoch_loss, epoch)
          
        # ---- Validation ---- #
        if (epoch + 1) % VAL_INTERVAL == 0:
            val_loss = validator(model, val_loader, device, epoch, writer)
            logging.info(f"[val] epoch: {epoch} mean val loss: {val_loss:.4f}")
            writer.add_scalar("loss/val epoch", val_loss, epoch)
        else:
            val_loss = 0.0

        # ---- Save Model and Optimizer ---- #
        if (epoch + 1) % MODEL_SAVE_INTERVAL == 0:
            ckpt_path = os.path.join(snapshot_dir, "models", f"model_epoch_{epoch + 1}")
            save_checkpoint(model, optimizer, scheduler, epoch + 1, ckpt_path)
            logging.info(f"Checkpoint saved to {ckpt_path}")

        prepare_and_write_csv_file(snapshot_dir, [epoch, epoch_loss, val_loss], write_header=(epoch == 0))

# ------------------------------------------------------------------------------ #
def validator(model, val_loader, device, epoch, writer):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            image = batch["aug_image"].to(device, dtype=torch.float16)
            mask  = batch["aug_mask"].to(device, dtype=torch.float16)

            with autocast(device, enabled=True):
                _, loss  = model(image, mask)

            val_loss += loss.item()
            writer.add_scalar("Loss/Val Iteration", loss.item(), epoch * len(val_loader) + step)

    return val_loss / len(val_loader)

# ------------------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args   = parser.parse_args()

    snapshot_dir = LDM_SNAPSHOT_DIR
    models_dir   = os.path.join(snapshot_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    device = setup_environment(SEED, snapshot_dir)
    setup_logging(snapshot_dir)
    writer = SummaryWriter(os.path.join(snapshot_dir, "log" if args.resume else "log"))
    print(f"Results logged in: {snapshot_dir}, TensorBoard logs in: {snapshot_dir}/log, Models saved in: {models_dir}\n")
    
    
    if args.resume:
        resume_epoch, weights_path, opt_path = get_latest_checkpoint(models_dir)
        if weights_path and opt_path:
            model, optimizer, scheduler, _ = load_model_and_optimizer(weights_path, opt_path, device, load_optim_dict = False)
            logging.info(f"✅ Resumed from epoch {resume_epoch} (model: {weights_path})")
        else:
            logging.warning("⚠️ No valid checkpoint found. Starting from scratch.")
            model, optimizer, scheduler, resume_epoch = initialize_new_session(device)
    else:
        logging.info("🚀 Starting new training session")
        model, optimizer, scheduler, resume_epoch = initialize_new_session(device)

    scaler       = GradScaler(device)
    train_loader = get_dataloaders(BASE_DIR, SPLIT_RATIOS, "train", TRAINSIZE, BATCH_SIZE, 6, FORMAT)
    val_loader   = get_dataloaders(BASE_DIR, SPLIT_RATIOS, "val", TRAINSIZE, BATCH_SIZE, 6, FORMAT)

    start = time.time()

    trainer(model, optimizer, scheduler, train_loader, val_loader, device, scaler, snapshot_dir, writer, resume_epoch)

    total_time = (time.time() - start) / 60
    logging.info(f"✅ Training finished in {total_time:.2f} minutes.")
    writer.close()
# ------------------------------------------------------------------------------ #
if __name__ == "__main__":
    main()
# ------------------------------------------------------------------------------ #
