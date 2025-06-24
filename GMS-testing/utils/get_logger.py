# ------------------------------------------------------------------------------#
#
# File name                 : get_logger.py
# Purpose                   : Utility for initializing logging to both file and console.
# Usage                     : Called during training/inference to initialize log files.
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh, Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 202410001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : June 21, 2025
# ------------------------------------------------------------------------------#


# --------------------------- Module imports -----------------------------------#
import os, logging

from configs.config             import *

# --------------------------- Open log file ------------------------------------#
def open_log(phase, log_path):
    """
    Creates a log file at the specified path in the config.
    Deletes existing log file with the same name if it exists.

    Args:
        args   : Command-line arguments, must include `args.config` (path to YAML).
        log_path: Path to save the log file, typically derived from `args.config`.

    Returns:
        None
    """

    log_name      = f'{phase}-config'
    log_filepath  = os.path.join(log_path, f'{log_name}.txt')

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    if os.path.isfile(log_filepath):
        os.remove(log_filepath)

    initLogging(log_filepath)


# --------------------------- Logger initializer -------------------------------#
def initLogging(logFilename):
    """
    Initializes Python logging to write logs to both a file and the console.

    Args:
        logFilename : Full path of the log file.

    Returns:
        None
    """

    logging.basicConfig(
        level     = logging.INFO,
        format    = '[%(asctime)s-%(levelname)s] %(message)s',
        datefmt   = '%y-%m-%d %H:%M:%S',
        filename  = logFilename,
        filemode  = 'w'
    )

    # Set up console output
    console        = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter      = logging.Formatter('[%(asctime)s-%(levelname)s] %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

# --------------------------------- End -----------------------------------------#
