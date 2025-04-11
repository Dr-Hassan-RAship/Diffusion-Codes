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

from config_ldm_ddpm          import *
from monai.networks.nets      import AutoencoderKL
from dataset                  import *
from source_unet              import *

#------------------------------------------------------------------------------#
def validate_resume_params(snapshot_dir, mode, current_batch_size, current_model_params):
    """
    Validates the current training configuration against saved parameters.
    If mismatched, it logs the differences and uses the saved configuration.
    """
    joint_dir           = f"aekl_{mode}_params.txt" if mode != 'ldm' else f"model_params.txt" 
    txt_file            = os.path.join(snapshot_dir, joint_dir)
    override_params     = current_model_params.copy()
    override_batch_size = current_batch_size

    if not os.path.exists(txt_file):
        logging.warning(f"Parameter file {txt_file} not found. Cannot validate consistency.")
        return override_batch_size, override_params

    with open(txt_file, "r") as f:
        lines = f.readlines()

    saved_params = {}
    for line in lines[1:]:  # skip header
        if ":" not in line:
            continue
        key, value = line.strip().split(":", 1)
        key, value = key.strip(), value.strip()
        try:
            parsed_value = eval(value)
        except:
            parsed_value  = value
        saved_params[key] = parsed_value

    mismatch_batch = False
    if "batch_size" in saved_params and saved_params["batch_size"] != current_batch_size:
        logging.warning(f"Batch size mismatch! (config: {current_batch_size}, saved: {saved_params['batch_size']})")
        override_batch_size = saved_params["batch_size"]
        mismatch_batch      = True

    mismatch_params = False
    for key in current_model_params:
        if key in saved_params and saved_params[key] != current_model_params[key]:
            logging.warning(f"Param mismatch for '{key}'! (config: {current_model_params[key]}, saved: {saved_params[key]})")
            override_params[key] = saved_params[key]
            mismatch_params      = True
    
    return override_batch_size, override_params, mismatch_batch, mismatch_params

#-----------------------------------------------------------------------------#
def launch_tensorboard(log_dir):
    """
    Launch TensorBoard in a subprocess.

    Args:
        log_dir (str): Path to the log directory to visualize.
    """
    try:
        subprocess.Popen(["tensorboard", "--logdir", log_dir, "--port", "6006"])
        logging.info(f"TensorBoard launched at http://localhost:6006/ (logdir: {log_dir})")
        print(f"🔍 TensorBoard launched at http://localhost:6006/ (logdir: {log_dir})")
    except Exception as e:
        logging.warning(f"Failed to launch TensorBoard: {e}")
        print(f"⚠️ Could not launch TensorBoard: {e}")

#------------------------------------------------------------------------------#
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
        
#----------------------------------------------------------------------------------#
def get_latest_checkpoint(models_dir, prefix, resume_epoch=None):
    """
    Get the latest or specified checkpoint.

    Args:
        models_dir (str): Path to the models directory.
        prefix (str): Checkpoint filename prefix, e.g., 'autoencoderkl_epoch_'.
        resume_epoch (int or None): If specified, return the checkpoint for this epoch and delete newer ones.

    Returns:
        (int, str): Tuple of (epoch, checkpoint path), or (None, None) if not found.
    """
    checkpoint   = None
    latest_epoch = -1

    pattern      = re.compile(rf'{re.escape(prefix)}(\d+)\.pth')
    epoch_files  = []

    for filename in os.listdir(models_dir):
        match = pattern.match(filename)
        if match:
            epoch = int(match.group(1))
            epoch_files.append((epoch, filename))

    if not epoch_files:
        return None, None

    if resume_epoch is not None:
        # Find checkpoint at requested epoch
        for epoch, fname in epoch_files:
            if epoch == resume_epoch:
                checkpoint = fname

        # Delete all checkpoints after the requested epoch
        for epoch, fname in epoch_files:
            if epoch > resume_epoch:
                path_to_delete = os.path.join(models_dir, fname)
                os.remove(path_to_delete)
                print(f"🗑️ Deleted newer checkpoint: {path_to_delete}")

        if checkpoint:
            return resume_epoch, os.path.join(models_dir, checkpoint)
        else:
            print(f"⚠️ No checkpoint found for epoch {resume_epoch}")
            return None, None

    else:
        # Default behavior: return the latest
        latest_epoch, checkpoint = max(epoch_files, key = lambda x: x[0])
        return latest_epoch, os.path.join(models_dir, checkpoint)
#-----------------------------------------------------------------------------------#
def validate_resume_training(model, snapshot_dir, models_dir, mode, device, args, prefix):
    """
    Resume training if --resume is specified and checkpoint exists.
    Validates params from the saved txt file.

    Returns:
        resume_epoch (int): Epoch to resume from.
        model (nn.Module): Updated model if parameter mismatch.
        train_loader (DataLoader): Updated if batch size mismatch.
        val_loader (DataLoader): Updated if batch size mismatch.
    """
    
    resume_epoch             = 0
    override_batch_size      = BATCH_SIZE
    override_params          = AUTOENCODERKL_PARAMS if mode != 'ldm' else MODEL_PARAMS
    train_loader, val_loader = None, None  # initialize as None
    optimizer                = None
    
    latest_epoch, ckpt_path = get_latest_checkpoint(models_dir, prefix=prefix, resume_epoch = None)
    print(f'latest_epoch: {latest_epoch}, ckpt_path: {ckpt_path}')

    if args.resume and ckpt_path is not None:
        # Validate parameter consistency
        override_batch_size, override_params, mismatch_batch, mismatch_params = validate_resume_params(
            snapshot_dir,
            mode                 = mode,
            current_batch_size   = BATCH_SIZE,
            current_model_params = AUTOENCODERKL_PARAMS if mode != 'ldm' else MODEL_PARAMS
        )
        
        # Re-initialize dataloaders if batch size differs
        if mismatch_batch:
            train_loader = get_dataloaders(
                BASE_DIR, split_ratio = SPLIT_RATIOS, split = 'train',
                trainsize = TRAINSIZE, batch_size = override_batch_size, format = FORMAT
            )
            val_loader   = get_dataloaders(
                BASE_DIR, split_ratio = SPLIT_RATIOS, split = 'val',
                trainsize = TRAINSIZE, batch_size = override_batch_size, format = FORMAT
            )

        # Re-initialize model if model param mismatch
        if mismatch_params:
            model = AutoencoderKL(**override_params).to(device) if mode != 'ldm' else DiffusionModelUNet(**MODEL_PARAMS).to(device)
            # Load model weights
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
            
            optimizer     = torch.optim.AdamW(model.parameters(),
                                  lr=LR,
                                  betas=(0.9, 0.999),
                                  weight_decay=0.0001)

    else:
        logging.info("Starting training from scratch.")
        print("🚀 Starting training from scratch.")
    
    resume_epoch = latest_epoch + 1
    logging.info(f"Resuming training from epoch {resume_epoch} using checkpoint {ckpt_path}")
    print(f"✅ Resuming training from epoch {resume_epoch}")
    return resume_epoch, model, train_loader, val_loader, optimizer
#-------------------------------------------------------------------------------------#