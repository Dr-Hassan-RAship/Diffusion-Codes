import os, time, argparse, logging, torch
from torch.amp import autocast, GradScaler
from torch.nn.functional import l1_loss
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from config import *
from dataset import get_dataloaders
from architectures import *
from utils import *
from modelling_utils import *

# ------------------------------------------------------------------------------ #
def initialize_new_session(device):
    """Creates a new model and optimizer from scratch."""
    model     = LDM_Segmentor().to(device)
    optimizer = AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=0.0001)
    return model, optimizer, 0

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
                loss = l1_loss(out['noise_hat'], out['noise']) # + l1_loss(out["z0_hat"], out['z0']) [nehal flag version] 

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            torch.cuda.empty_cache()
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

        # ---- Save Model and Optimizer ---- #
        if (epoch + 1) % MODEL_SAVE_INTERVAL == 0:
            ckpt_path = os.path.join(snapshot_dir, "models", f"model_epoch_{epoch + 1}")
            save_checkpoint(model, optimizer, epoch, ckpt_path)
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
                out  = model(image, mask, t)
                loss = l1_loss(out["mask_hat"], mask)

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

    setup_logging(snapshot_dir)
    writer = SummaryWriter(os.path.join(snapshot_dir, "log_resume2" if args.resume else "log"))
    print(f"Results logged in: {snapshot_dir}, TensorBoard logs in: {snapshot_dir}/log_resume2, Models saved in: {models_dir}\n")
    torch.manual_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.resume:
        latest_epoch, weights_path, opt_path = get_latest_checkpoint(models_dir)
        if weights_path and opt_path:
            model, optimizer, resume_epoch = load_model_and_optimizer(weights_path, opt_path, device)
            logging.info(f"✅ Resumed from epoch {resume_epoch} (model: {weights_path})")
        else:
            logging.warning("⚠️ No valid checkpoint found. Starting from scratch.")
            model, optimizer, resume_epoch = initialize_new_session(device)
    else:
        logging.info("🚀 Starting new training session")
        model, optimizer, resume_epoch = initialize_new_session(device)

    scaler       = GradScaler(device)
    train_loader = get_dataloaders(BASE_DIR, SPLIT_RATIOS, "train", TRAINSIZE, BATCH_SIZE, FORMAT)
    val_loader   = get_dataloaders(BASE_DIR, SPLIT_RATIOS, "val", TRAINSIZE, BATCH_SIZE, FORMAT)

    start = time.time()

    trainer(model, optimizer, train_loader, val_loader, device, scaler, snapshot_dir, writer, resume_epoch)

    total_time = (time.time() - start) / 60
    logging.info(f"✅ Training finished in {total_time:.2f} minutes.")
    writer.close()
# ------------------------------------------------------------------------------ #
if __name__ == "__main__":
    main()
# ------------------------------------------------------------------------------ #