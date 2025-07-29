# Basic Package
import torch
import argparse
import numpy as np
import yaml
import logging
import time
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from monai.losses.dice import DiceLoss

# Own Package
from data.image_dataset import Image_Dataset
from utils.tools import *
from utils.get_logger import open_log
from utils.load_ckpt import * # contains function save_checkpoint
from utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from utils.metrics      import *
from networks.latent_mapping_model import ResAttnUNet_DS
from networks.models.distributions import DiagonalGaussianDistribution
from networks.novel.sft.sft_block import *
from networks.novel.sft.patchify import *

from tensorboardX import SummaryWriter

# Note the encoder output is AutoeencoderTinyOuput(latents = ouptut) and Decoder is DecoderOutput(sample = output)


def get_multi_loss(criterion, out_dict, label, is_ds=True, key_list=None):
    keys = key_list if key_list is not None else list(out_dict.keys())
    if is_ds:
        multi_loss = sum([criterion(out_dict[key], label) for key in keys])
    else:
        multi_loss = criterion(out_dict["out"], label)
    return multi_loss


def get_vae_encoding_mu_and_sigma(encoder_posterior, scale_factor):
    if isinstance(encoder_posterior, DiagonalGaussianDistribution):
        mean, logvar = encoder_posterior.mu_and_sigma()
    else:
        raise NotImplementedError(
            f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented"
        )
    return scale_factor * mean, logvar


def vae_decode(vae_model, pred_mean, scale_factor):
    z = 1.0 / scale_factor * pred_mean
    pred_seg = vae_model.decode(
        z
    ).sample  # [CHANGED] --> has channels = 3 according to config
    pred_seg = torch.mean(
        pred_seg, dim=1, keepdim=True
    )  # [CHANGED] --> Taking mean across channels dimension resulting in 1 channel
    pred_seg = torch.clamp(
        (pred_seg + 1.0) / 2.0, min=0.0, max=1.0
    )  # (B, 1, H, W) # [CHANGED] --> Bringing the range to (0, 1) as per Kvasir-SEG dataset
    return pred_seg


# [CHANGED] --> added the path to the Kvasir-SEG config file
def arg_parse() -> argparse.ArgumentParser.parse_args:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="./configs/busi_train.yaml",
        type=str,
        help="load the config file",
    )
    args = parser.parse_args()
    return args


def run_trainer() -> None:
    args = arg_parse()
    configs = yaml.load(open(args.config), Loader=yaml.FullLoader)
    # time.strftime("%Y%m%d%H%M", time.localtime(time.time()))
    configs["snapshot_path"] = os.path.join(
        configs["snapshot_path"],
        'epoch_1500',
    )
    configs["log_path"] = os.path.join(configs["snapshot_path"], "logs")

    # Output folder and save fig folder
    os.makedirs(configs["snapshot_path"], exist_ok=True)
    os.makedirs(configs["log_path"], exist_ok=True)

    # Set GPU ID
    gpus = ",".join([str(i) for i in configs["GPUs"]])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    # Fix seed (for repeatability)
    seed_reproducer(configs["seed"])

    # Open log file
    open_log(args, configs)
    logging.info(configs)
    print_options(configs)

    # Define summary writer
    writer = SummaryWriter(configs["log_path"])
    ds_list = ["level2", "level1", "out"]

    # Get data loader
    train_dataset = Image_Dataset(configs["pickle_file_path"], stage="train", excel = False)
    valid_dataset = Image_Dataset(configs["pickle_file_path"], stage="test", excel = False)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=configs["batch_size"],
        pin_memory=True,
        drop_last=True,
        shuffle=True,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=configs["batch_size"],
        pin_memory=True,
        drop_last=False,
        shuffle=False,
    )
    

    # Define networks
    mapping_model = get_cuda(
        ResAttnUNet_DS(
            in_channel=configs["in_channel"],
            out_channels=configs["out_channels"],
            num_res_blocks=configs["num_res_blocks"],
            ch=configs["ch"],
            ch_mult=configs["ch_mult"],
        )
    )
    # Getting tiny-vae (with residual_autoencoding) default: frozen and eval
    vae_train = True
    if configs['vae_model'] == 'tiny_vae':
        logging.info("Initializing TinyVAE")
        vae_model = get_tiny_autoencoder(train = vae_train, residual_autoencoding = False)
    else:
        logging.info("Initializing LiteVAE")
        tiny_vae  = get_tiny_autoencoder(train = vae_train, residual_autoencoding = False) # for the segmentation latent and decoding at the end.
        vae_model = get_lite_vae(model_version = configs['vae_model'])

    scale_factor = 1.0
    
    if configs['patchify']:
        patch_model = LearnablePatchify(patch_size = 28).to(dtype = torch.float32, device = 'cuda')
        sft_model   = SFTModule(original_channels = configs['in_channel'], guidance_channels = 64).to(dtype = torch.float32, device = 'cuda')
        patch_model.train(); sft_model.train()

    # Define optimizers

    param_groups = list(mapping_model.parameters())
    if vae_train:
        param_groups += list(vae_model.parameters())
        logging.info("Training both mapping model and VAE model")
    
    if configs['patchify']:
        param_groups += list(patch_model.parameters()) + list(sft_model.parameters())
        logging.info("Training patch_model and sft_model as well")

    logging.info(f"Mapping Model trainable params: {count_params(mapping_model)} out of {sum(p.numel() for p in mapping_model.parameters())}")
    logging.info(f"VAE Model trainable params: {count_params(vae_model)} out of {sum(p.numel() for p in vae_model.parameters())}")
    logging.info(f"Patch Model and SFT Model trainable params: {count_params(patch_model) + count_params(sft_model)}")
    
    optimizer = torch.optim.AdamW(param_groups, lr=configs["lr"])
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer, warmup_epochs=5, max_epochs=configs["epochs"]
    )

    # Define loss functions
    mse_loss = torch.nn.MSELoss(reduction="mean")
    dice_loss = DiceLoss()

    # For Tensorboard Visualization
    iter_num = 0
    best_valid_loss = np.inf
    best_valid_loss_rec = np.inf
    best_valid_dice = 0
    best_valid_dice_epoch = 0
    best_valid_loss_dice = np.inf

    # Network training
    for epoch in range(1, configs["epochs"] + 1):
        epoch_start_time = time.time()
        mapping_model.train()
        vae_model.train() if vae_train else vae_model.eval()
        tiny_vae.eval() if configs['vae_model'] != 'tiny_vae' else None

        T_loss = []
        T_loss_Rec = []
        T_loss_Dice = []
        T_loss_valid = []
        T_loss_Rec_valid = []
        T_loss_Dice_valid = []
        T_Dice_valid = []

        ### Training phase
        for batch_data in tqdm(train_dataloader, desc="Train: "):
            # [CHANGED] --> In case the vae model is lite_vae, the haar transform expects (0, 1) or (0, 255) so we will scale the input accordingly
            # Note: no change needed regarding the segmentation as it is being used by tiny vae.
            img_rgb = batch_data["img"]
            img_rgb = img_rgb / 255.0
            if configs['vae_model'] == 'tiny_vae':   
                img_rgb = 2.0 * img_rgb - 1.0

            seg_raw = batch_data["seg"]
            seg_raw = seg_raw.permute(0, 3, 1, 2) / 255.0
            seg_rgb = 2.0 * seg_raw - 1.0
                
            # [CHANGED] --> Taking mean across channels dimension resulting in 1 channel which matches the channel dimension
            # of pred_seg gotten from the decoder. Same thing in Validation
            seg_img = torch.mean(seg_raw, dim=1, keepdim=True)
            name = batch_data["name"]
            
            if configs['vae_model'] == 'tiny_vae':
                img_latent_mean_aug, seg_latent_mean = (
                    vae_model.encode(get_cuda(img_rgb)).latents,
                    vae_model.encode(get_cuda(seg_rgb)).latents,
                )
            else:
                img_latent_mean_aug, seg_latent_mean = (
                    vae_model(get_cuda(img_rgb)),
                    tiny_vae.encode(get_cuda(seg_rgb)).latents,
                )

            if configs['patchify']:
                patched_img         = patch_model(img_rgb.to('cuda'))
                img_latent_mean_aug = sft_model(x = img_latent_mean_aug, ref = patched_img)

            # latent matching [CHANGED] --> recieves the grountruth latent mask representation and predicted and computes mse loss
            out_latent_mean_dict = mapping_model(img_latent_mean_aug)
            loss_Rec = configs["w_rec"] * get_multi_loss(
                mse_loss,
                out_latent_mean_dict,
                seg_latent_mean,
                is_ds=True,
                key_list=ds_list,
            )

            # image matching [CHANGED] --> computes the predicted mask on different levels as the predicted latent representation
            # was on different levels (see line 228 - 233 of latent_mapping_model.py)
            pred_seg_dict = {}
            for level_name in ds_list:
                pred_seg_dict[level_name] = vae_decode(
                    tiny_vae if configs['vae_model'] != 'tiny_vae' else vae_model, out_latent_mean_dict[level_name], scale_factor
                )

            # [CHANGED] --> similar to Loss_Rec
            loss_Dice = configs["w_dice"] * get_multi_loss(
                dice_loss,
                pred_seg_dict,
                get_cuda(seg_img),
                is_ds=True,
                key_list=ds_list,
            )

            loss = loss_Rec + loss_Dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1
            if iter_num % 10 == 0:
                writer.add_scalar("loss/loss", loss, iter_num)
                writer.add_scalar("loss/loss_Rec", loss_Rec, iter_num)
                writer.add_scalar("loss/loss_Dice", loss_Dice, iter_num)

            T_loss.append(loss.item())
            T_loss_Rec.append(loss_Rec.item())
            T_loss_Dice.append(loss_Dice.item())

        scheduler.step()
        writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

        T_loss = np.mean(T_loss)
        T_loss_Rec = np.mean(T_loss_Rec)
        T_loss_Dice = np.mean(T_loss_Dice)

        logging.info("Train:")
        logging.info(
            "loss: {:.4f}, loss_Rec: {:.4f}, loss_Dice: {:.4f}".format(
                T_loss, T_loss_Rec, T_loss_Dice
            )
        )

        # [CHANGED] --> Added Tensorboard logging for training loss per epoch
        writer.add_scalar("train/loss", T_loss, epoch)
        writer.add_scalar("train/loss_Rec", T_loss_Rec, epoch)
        writer.add_scalar("train/loss_Dice", T_loss_Dice, epoch)

        ### Validation phase [CHANGED] --> almost same as Train except the dice score is being calculated here as well
        ### Also note that the validation set is the same as the test set in the inference/valid.py file.
        for batch_data in tqdm(valid_dataloader, desc="Valid: "):
            img_rgb = batch_data["img"]
            img_rgb = img_rgb / 255.0  # [CHANGED] V.V.V Imp!  --> SCALE CORRECTION
            if configs['vae_model'] == 'tiny_vae':
                img_rgb = 2.0 * img_rgb - 1.0

            seg_raw = batch_data["seg"]
            seg_raw = seg_raw.permute(0, 3, 1, 2) / 255.0
            seg_rgb = 2.0 * seg_raw - 1.0
            seg_img = torch.mean(seg_raw, dim=1, keepdim=True)
            name = batch_data["name"]

            mapping_model.eval()
            vae_model.eval()
            tiny_vae.eval() if configs['vae_model'] != 'tiny_vae' else None
            patch_model.eval(); sft_model.eval()

            with torch.no_grad():
                if configs['vae_model'] == 'tiny_vae':
                    img_latent_mean, seg_latent_mean = (
                        vae_model.encode(get_cuda(img_rgb)).latents,
                        vae_model.encode(get_cuda(seg_rgb)).latents,
                    )
                else:
                    img_latent_mean, seg_latent_mean = (
                        vae_model(get_cuda(img_rgb)),
                        tiny_vae.encode(get_cuda(seg_rgb)).latents,
                    )

                if configs['patchify']:
                    patched_img         = patch_model(img_rgb.to('cuda'))
                    img_latent_mean     = sft_model(x = img_latent_mean, ref = patched_img)

                out_latent_mean_dict = mapping_model(img_latent_mean)
                pred_seg = vae_decode(
                    tiny_vae if configs['vae_model'] != 'tiny_vae' else vae_model, out_latent_mean_dict["out"], scale_factor
                )

                loss_Rec = configs["w_rec"] * mse_loss(
                    out_latent_mean_dict["out"], seg_latent_mean
                )
                loss_Dice = configs["w_dice"] * dice_loss(pred_seg, get_cuda(seg_img))

                loss = loss_Rec + loss_Dice

                # calc dice
                pred_seg = pred_seg.cpu()
                reduce_axis = list(range(1, len(seg_img.shape)))

                intersection = torch.sum(seg_img * pred_seg, dim=reduce_axis)
                y_o = torch.sum(seg_img, dim=reduce_axis)
                y_pred_o = torch.sum(pred_seg, dim=reduce_axis)
                denominator = y_o + y_pred_o
                dice_raw = (2.0 * intersection) / denominator
                dice_value = dice_raw.mean()

                T_Dice_valid.append(dice_value.item())
                T_loss_valid.append(loss.item())
                T_loss_Rec_valid.append(loss_Rec.item())
                T_loss_Dice_valid.append(loss_Dice.item())

        T_Dice_valid = np.mean(T_Dice_valid)
        T_loss_valid = np.mean(T_loss_valid)
        T_loss_Rec_valid = np.mean(T_loss_Rec_valid)
        T_loss_Dice_valid = np.mean(T_loss_Dice_valid)

        writer.add_scalar("valid/dice", T_Dice_valid, epoch)
        writer.add_scalar("valid/loss", T_loss_valid, epoch)
        writer.add_scalar("valid/loss_Rec", T_loss_Rec_valid, epoch)
        writer.add_scalar("valid/loss_Dice", T_loss_Dice_valid, epoch)

        logging.info("Valid:")
        logging.info(
            "loss: {:.4f}, loss_Rec: {:.4f}, loss_Dice: {:.4f}, Dice: {:.4f}".format(
                T_loss_valid, T_loss_Rec_valid, T_loss_Dice_valid, T_Dice_valid
            )
        )

        if T_Dice_valid > best_valid_dice:
            save_name = f"best_valid_dice_{epoch}.pth"
            if vae_train:
                save_checkpoint(mapping_model, save_name, configs["snapshot_path"], vae_model = vae_model, vae_model_save = True)
            else:
                save_checkpoint(mapping_model, save_name, configs["snapshot_path"])
            best_valid_dice = T_Dice_valid
            best_valid_dice_epoch = epoch
            logging.info("Save best valid Dice !")

        if T_loss_valid < best_valid_loss:
            save_name = f"best_valid_loss_{epoch}.pth"
            if vae_train:
                save_checkpoint(mapping_model, save_name, configs["snapshot_path"], vae_model = vae_model, vae_model_save = True)
            else:
                save_checkpoint(mapping_model, save_name, configs["snapshot_path"])
            best_valid_loss = T_loss_valid
            logging.info("Save best valid Loss All !")

        if T_loss_Rec_valid < best_valid_loss_rec:
            save_name = f"best_valid_loss_rec_{epoch}.pth"
            if vae_train:
                save_checkpoint(mapping_model, save_name, configs["snapshot_path"], vae_model = vae_model, vae_model_save = True)
            else:
                save_checkpoint(mapping_model, save_name, configs["snapshot_path"])
            best_valid_loss_rec = T_loss_Rec_valid
            logging.info("Save best valid Loss Rec !")

        if T_loss_Dice_valid < best_valid_loss_dice:
            save_name = f"best_valid_loss_dice_{epoch}.pth"
            if vae_train:
                save_checkpoint(mapping_model, save_name, configs["snapshot_path"], vae_model = vae_model, vae_model_save = True)
            else:
                save_checkpoint(mapping_model, save_name, configs["snapshot_path"])
            best_valid_loss_dice = T_loss_Dice_valid
            logging.info("Save best valid Loss Dice !")

        if epoch % configs["save_freq"] == 0:
            save_name = "{}_epoch_{:0>4}.pth".format("latent_mapping_model", epoch)
            if vae_train:
                save_checkpoint(mapping_model, save_name, configs["snapshot_path"], vae_model = vae_model, vae_model_save = True)
            else:
                save_checkpoint(mapping_model, save_name, configs["snapshot_path"])

        logging.info("Current learning rate: {:.5f}".format(scheduler.get_last_lr()[0]))
        logging.info(
            "epoch %d / %d \t Time Taken: %d sec"
            % (epoch, configs["epochs"], time.time() - epoch_start_time)
        )
        logging.info(
            "best valid dice: {:.4f} at epoch: {}".format(
                best_valid_dice, best_valid_dice_epoch
            )
        )
        logging.info("\n")

    writer.close()


if __name__ == "__main__":
    run_trainer()
