# ------------------------------------------------------------------------------#
#
# File name                 : train_aekl.py
# Purpose                   : Training script for the autoencoderKL for both image and mask (without discriminator & generator)
# Usage                     : python train_aekl.py --mode 'image' [--resume] [--pretrained]
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh, Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 202410001@lums.edu.pk, hassan.mohyuddin@lums.edu.pk
#
# Last Date                 : April 16, 2025
# Note                      : Used AdamW optimizer for better performance
# ------------------------------------------------------------------------------#

import argparse, os, time, logging
import torch
import torch.nn.functional as F

from accelerate                        import Accelerator
from diffusers                         import AutoencoderKL
from torch.utils.tensorboard           import SummaryWriter
from config                            import TRAIN_CONFIG, AEKL_CONFIG
from dataset                           import *
from utils                             import *

# ---------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="Train AutoencoderKL using diffusers")
    parser.add_argument("--mode", choices=["image", "mask"], required=True, help="Mode: image or mask")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained Hugging Face weights")
    return parser.parse_args()

# ---------------------------------------------------------------------------- #
def initialize_model(pretrained: bool):
    if pretrained:
        model = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
    else:
        model = AutoencoderKL(**AEKL_CONFIG)
    return model

# ---------------------------------------------------------------------------- #
def train_one_epoch(model, dataloader, optimizer, accelerator, epoch, writer, mode):
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(dataloader):
        gt = batch["aug_image"] if mode == "image" else batch["aug_mask"]
        with accelerator.autocast():
            recon = model(gt).sample
            recon = torch.tanh(recon)
            loss = F.l1_loss(recon, gt)

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        writer.add_scalar("Loss/Train Iteration", loss.item(), epoch * len(dataloader) + step)

    return total_loss / len(dataloader)

# ---------------------------------------------------------------------------- #
def validate(model, dataloader, accelerator, epoch, writer, mode):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            gt = batch["aug_image"] if mode == "image" else batch["aug_mask"]
            with accelerator.autocast():
                recon = model(gt).sample
                recon = torch.tanh(recon)
                loss = F.l1_loss(recon, gt)
            total_loss += loss.item()
            writer.add_scalar("Loss/Val Iteration", loss.item(), epoch * len(dataloader) + step)
    return total_loss / len(dataloader)

# ---------------------------------------------------------------------------- #
def main():
    args        = parse_args()
    accelerator = Accelerator()
    device      = accelerator.device

    snapshot_dir = TRAIN_CONFIG[f"{args.mode.upper()}_SNAPSHOT_DIR"]
    models_dir   = os.path.join(snapshot_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    setup_logging(snapshot_dir)
    writer = SummaryWriter(log_dir=os.path.join(snapshot_dir, "log"))

    train_loader = get_dataloaders(split="train", mode=args.mode)
    val_loader = get_dataloaders(split="val", mode=args.mode)

    # Model & Optimizer
    model = initialize_model(args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN_CONFIG["LR"])

    resume_epoch = 0
    if args.resume:
        latest_epoch, ckpt_path = get_latest_checkpoint(models_dir, prefix="autoencoderkl_epoch_")
        if ckpt_path:
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            resume_epoch = latest_epoch + 1
            logging.info(f"Resumed from epoch {resume_epoch}")
            print(f"✅ Resumed from epoch {resume_epoch}")

    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

    launch_tensorboard(os.path.join(snapshot_dir, "log"))
    for epoch in range(resume_epoch, resume_epoch + TRAIN_CONFIG["N_EPOCHS"]):
        train_loss = train_one_epoch(model, train_loader, optimizer, accelerator, epoch, writer, args.mode)
        val_loss = validate(model, val_loader, accelerator, epoch, writer, args.mode)

        logging.info(f"[E{epoch}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        writer.add_scalar("loss/train epoch", train_loss, epoch)
        writer.add_scalar("loss/val epoch", val_loss, epoch)

        if (epoch + 1) % TRAIN_CONFIG["MODEL_SAVE_INTERVAL"] == 0:
            ckpt_path = os.path.join(models_dir, f"autoencoderkl_epoch_{epoch + 1}.pth")
            accelerator.save(model.state_dict(), ckpt_path)
            logging.info(f"Checkpoint saved at {ckpt_path}")

    final_model = os.path.join(models_dir, "final_model.pth")
    accelerator.save(model.state_dict(), final_model)
    print(f"🚀 Final model saved at {final_model}")
    writer.close()

# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
