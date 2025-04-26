# ------------------------------------------------------------------------------#
#
# File name                 : utils.py
# Purpose                   : Helper functions in general

# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh, Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 202410001@lums.edu.pk, hassan.mohyuddin@lums.edu.pk
#
# Last Date                 : March 24, 2025
#-------------------------------------------------------------------------------#
import os, csv, sys, logging, re, subprocess, logging, torch

from config                   import *
from diffusers                import AutoencoderKL
from dataset                  import *

from   config import *
from   safetensors.torch import save_model as save_safetensors
from   safetensors.torch import load_file as load_safetensors
from   glob import glob
from   torch.optim import AdamW
from   architectures import *

#-----------------------------------------------------------------------------#

def prepare_and_write_csv_file(snapshot_dir, list_entries, write_header=False):
    """
    Write entries to logs.csv. Optionally write header if `write_header=True`.
    """
    csv_path = os.path.join(snapshot_dir, 'logs.csv')
    with open(csv_path, 'a', newline='') as csvfile:
        csv_logger = csv.writer(csvfile)
        if write_header:
            csv_logger.writerow(list_entries)
        else:
            csv_logger.writerow(list_entries)
        csvfile.flush()
#------------------------------------------------------------------------------#
def prepare_writer_layout():
    layout = {
        "Evaluation": {
            "Loss" : ["Multiline", ["loss/train epoch", "loss/val epoch"]]
        }
    }
    
    return layout
#-------------------------------------------------------------------------------#
def setup_logging(snapshot_dir, log_filename="logs.txt", level=logging.INFO, console=True):
    """
    Sets up logging to file and optionally to stdout.

    Args:
        snapshot_dir (str): Path to directory where log file will be saved.
        log_filename (str): Name of the log file.
        level (int): Logging level (e.g., logging.INFO).
        console (bool): If True, also log to stdout.
    """
    os.makedirs(snapshot_dir, exist_ok=True)
    log_path = os.path.join(snapshot_dir, log_filename)

    logging.basicConfig(
        filename = log_path,
        level    = level,
        format   = "[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt  = "%H:%M:%S",
        force    = True
    )

    if console:
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        
#--------------------------------------------------------------------------------------#
def save_checkpoint(model, optimizer, epoch, path):
    """
    Save model weights (deduplicated) + optimizer state to disk.
    """
    weights_path = path + ".safetensors"

    # 1) Save deduplicated model weights
    save_safetensors(model, weights_path) 

    # 2) Save optimizer + epoch if and only if we are at last epoch
    if epoch == 1500:
        opt_path     = path + ".opt.pt"
        ckpt = {
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(ckpt, opt_path)
    
#--------------------------------------------------------------------------------------#
def get_latest_checkpoint(models_dir):
    """Returns the latest (epoch_num, weights_path, opt_path) tuple."""
    
    safetensors_files = sorted(
        glob(os.path.join(models_dir, "model_epoch_*.safetensors")),
        key = lambda x: int(x.split("_")[-1].split(".")[0])
    )
    if not safetensors_files:
        return None, None, None

    weights_path = safetensors_files[-1]
    epoch_num    = int(weights_path.split("_")[-1].split(".")[0])
    opt_path     = os.path.join(models_dir, f"model_epoch_{epoch_num}.opt.pt")
    return epoch_num, weights_path, opt_path

#--------------------------------------------------------------------------------------#
def load_model_and_optimizer(weights_path, opt_path, device, load_optim_dict = True):
    """Loads model and optimizer from safetensors + pt files without RAM duplication."""
    
    epoch     = None
    model     = LDM_Segmentor().to("cpu")
    optimizer = AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=0.0001)

    # Load pretrained VAE weights directly into model.vae without retaining duplicate
    model.vae.load_state_dict(
        AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").state_dict()
    )
    for p in model.vae.parameters():
        p.requires_grad = False
    model.vae.eval()

    # Load deduplicated model weights from .safetensors
    state_dict = load_safetensors(weights_path)
    model.load_state_dict(state_dict, strict=False)
    
    del state_dict

    # Move to GPU
    model = model.to(device)

    # Load optimizer state
    if load_optim_dict:
        opt_state = torch.load(opt_path, map_location="cpu", weights_only = True)
        optimizer.load_state_dict(opt_state["optimizer_state_dict"])
        epoch     = opt_state["epoch"] + 1

        del opt_state

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    
    return model, optimizer, epoch

#--------------------------------------------------------------------------------------#