# ------------------------------------------------------------------------------#
#
# File name                 : utils.py
# Purpose                   : Helper functions in general

# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh, Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 202410001@lums.edu.pk, hassan.mohyuddin@lums.edu.pk
#
# Last Date                 : March 24, 2025
#-------------------------------------------------------------------------------#
import os, csv, sys, logging, logging, torch

from config                     import *
from diffusers                  import AutoencoderKL
from diffusers.optimization     import get_cosine_schedule_with_warmup
from dataset                    import *

from safetensors.torch          import save_model as save_safetensors
from safetensors.torch          import load_file as load_safetensors
from glob                       import glob
from torch.optim                import AdamW
from architectures              import *

#-----------------------------------------------------------------------------#
def check_or_create_folder(folder):
    """Check if folder exists, if not, create it."""
    if not os.path.exists(folder):
        os.makedirs(folder)

#-----------------------------------------------------------------------------#
def setup_environment(seed: int, snapshot_dir: str):
    """Set up environment and determinism."""
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.multiprocessing.set_sharing_strategy("file_system")
    
    output_file = (
        "params.txt"  # Define the output text file pathweights_only=True
    )

    with open(f"{snapshot_dir}/{output_file}", "w") as f:  # Write the dictionary to a text file
        f.write(f"batch_size: {BATCH_SIZE}\n")
        f.write("-----------------------------------------------------\n")
        f.write("UNET_PARAMS:\n")
        [f.write(f"{key}: {value}\n") for key, value in UNET_PARAMS.items()]
        f.write("-----------------------------------------------------\n")
        f.write("OPTIMIZER_PARAMS:\n")
        f.write(f"use_scheduler: {USE_SCHEDULER}\n")
        for key, value in (list(OPT.items())[:4] if USE_SCHEDULER else OPT.items()):
            f.write(f"{key}: {value}\n")

    print(f"UNET_PARAMS and OPTIMIZER_PARAMS written to {output_file}")

    return "cuda" if torch.cuda.is_available() else "cpu"

#-------------------------------------------------------------------#
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
def save_checkpoint(model, optimizer, scheduler, epoch, path):
    """
    Save model weights (deduplicated) + optimizer state to disk.
    """
    weights_path = path + ".safetensors"

    # 1) Save deduplicated model weights
    save_safetensors(model, weights_path) 

    # 2) Save optimizer + epoch if and only if we are at last epoch
    if epoch % N_EPOCHS == 0:
        opt_path     = path + ".opt.pt"
        if USE_SCHEDULER:
            ckpt = {
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch"               : epoch,
            }
            torch.save(ckpt, opt_path)
        else:
            ckpt = {
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch"               : epoch,
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
def load_model_and_optimizer(weights_path, opt_path, device, load_optim_dict=True):
    """Loads model, optimizer, and scheduler from safetensors + pt files."""
    
    epoch     = 0
    scheduler = None
    
    model = LDM_Segmentor().to("cpu")

    # Reload VAE weights to avoid shared tensor duplication
    model.vae.load_state_dict(
        AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").state_dict()
    )
    for p in model.vae.parameters():
        p.requires_grad = False
    model.vae.eval()

    # Load deduplicated weights
    state_dict = load_safetensors(weights_path)
    model.load_state_dict(state_dict, strict=False)
    del state_dict

    model = model.to(device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=OPT['lr'], betas=OPT['betas'], weight_decay=OPT['weight_decay'])

    # Scheduler
    # total_steps  = (SPLIT_RATIOS[0] // BATCH_SIZE) * N_EPOCHS
    # warmup_steps = int(WARMUP_RATIO * total_steps)
    # scheduler    = get_cosine_schedule_with_warmup(optimizer  = optimizer, num_warmup_steps = warmup_steps,
    #                                             num_cycles = 0.1, num_training_steps=total_steps
    # )

    if load_optim_dict and opt_path is not None and os.path.exists(opt_path):
        opt_state = torch.load(opt_path, map_location="cpu")
        optimizer.load_state_dict(opt_state["optimizer_state_dict"])
        if "scheduler_state_dict" in opt_state:
            scheduler.load_state_dict(opt_state["scheduler_state_dict"])
        epoch = opt_state["epoch"] + 1
        
        del opt_state

        # Move optimizer tensors to device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    return model, optimizer, scheduler, epoch
#--------------------------------------------------------------------------------------#
