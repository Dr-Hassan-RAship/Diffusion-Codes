import os, time, argparse, logging, torch
from torch.amp import autocast, GradScaler
from torch.nn.functional import l1_loss
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from config import *
from dataset import get_dataloaders
from architectures import LDM_Segmentor
from utils import setup_logging, prepare_and_write_csv_file

# ------------------------------------------------------------------------------ #
def save_model_checkpoint(model, optimizer, epoch, path):
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(ckpt, path)

# ------------------------------------------------------------------------------ #
def trainer(model, optimizer, train_loader, val_loader, device, scaler, snapshot_dir, writer, resume_epoch):
    for epoch in range(resume_epoch, resume_epoch + N_EPOCHS):
        print(f"\n🚀 Starting Epoch {epoch}")
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(train_loader):
            image = batch["aug_image"].to(device, dtype=torch.float16)
            mask  = batch["aug_mask"].to(device, dtype=torch.float16)
            t     = torch.randint(0, model.scheduler.config.num_train_timesteps, (image.size(0),), device=device).long()

            optimizer.zero_grad(set_to_none=True)
            with autocast(device, enabled=True):
                out  = model(image, mask, t)
                loss = l1_loss(out["mask_hat"], mask)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            writer.add_scalar("Loss/Train Iteration", loss.item(), epoch * len(train_loader) + step)
            logging.info(f"[train] epoch: {epoch} batch: {step} loss: {loss.item():.4f}")

        epoch_loss /= len(train_loader)
        logging.info(f"[train] epoch: {epoch} mean loss: {epoch_loss:.4f}")
        writer.add_scalar("loss/train epoch", epoch_loss, epoch)

        # ---- Validation ---- #
        if (epoch + 1) % VAL_INTERVAL == 0:
            val_loss = validator(model, val_loader, device, epoch, writer)
            logging.info(f"[val] epoch: {epoch} mean val loss: {val_loss:.4f}")
            writer.add_scalar("loss/val epoch", val_loss, epoch)
        else:
            val_loss = 0.0

        # ---- Save Model ---- #
        if (epoch + 1) % MODEL_SAVE_INTERVAL == 0:
            ckpt_path = os.path.join(snapshot_dir, "models", f"model_epoch_{epoch + 1}.pth")
            save_model_checkpoint(model, optimizer, epoch, ckpt_path)
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
            t     = torch.randint(0, model.scheduler.config.num_train_timesteps, (image.size(0),), device=device).long()

            with autocast(device, enabled=True):
                out = model(image, mask, t)
                loss = l1_loss(out["mask_hat"], mask)

            val_loss += loss.item()
            writer.add_scalar("Loss/Val Iteration", loss.item(), epoch * len(val_loader) + step)

    return val_loss / len(val_loader)

# ------------------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    snapshot_dir = LDM_SNAPSHOT_DIR
    models_dir   = os.path.join(snapshot_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Setup environment
    setup_logging(snapshot_dir)
    logging.info("Starting new LDM training session")
    writer = SummaryWriter(os.path.join(snapshot_dir, "log_resume" if args.resume else "log"))
    
    print(f"Results logged in: {snapshot_dir}, TensorBoard logs in: {snapshot_dir}/log, Models saved in: {models_dir}\n")
    torch.manual_seed(SEED)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LDM_Segmentor().to(device)
    optimizer = AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=0.0001)
    scaler = GradScaler(device)

    train_loader = get_dataloaders(BASE_DIR, SPLIT_RATIOS, "train", TRAINSIZE, BATCH_SIZE, FORMAT)
    val_loader   = get_dataloaders(BASE_DIR, SPLIT_RATIOS, "val", TRAINSIZE, BATCH_SIZE, FORMAT)

    resume_epoch = 0
    if args.resume:
        ckpt_list = [f for f in os.listdir(os.path.join(snapshot_dir, "models")) if f.startswith("model_epoch_")]
        if ckpt_list:
            latest_ckpt = sorted(ckpt_list, key=lambda f: int(f.split("_")[-1].split(".")[0]))[-1]
            ckpt_path = os.path.join(snapshot_dir, "models", latest_ckpt)
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            resume_epoch = state["epoch"] + 1
            logging.info(f"✅ Resumed from {ckpt_path} at epoch {resume_epoch}")

    start = time.time()
    trainer(model, optimizer, train_loader, val_loader, device, scaler, snapshot_dir, writer, resume_epoch)
    total_time = (time.time() - start) / 60
    logging.info(f"Training finished in {total_time:.2f} minutes.")
    writer.close()

# ------------------------------------------------------------------------------ #
if __name__ == "__main__":
    main()
