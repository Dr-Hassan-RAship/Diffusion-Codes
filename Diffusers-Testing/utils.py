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
from monai.networks.nets      import AutoencoderKL
from dataset                  import *

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