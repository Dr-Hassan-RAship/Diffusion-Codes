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


# --------------------------- Open log file ------------------------------------#
def open_log(phase, config):
    """
    Creates a log file at the specified path in the config.
    Deletes existing log file with the same name if it exists.

    Args:
        args   : Command-line arguments, must include `args.config` (path to YAML).
        config : Dictionary-like object that must contain `log_path`.

    Returns:
        None
    """

    log_savepath  = config['log_path']
    log_name      = f'{phase}-yaml'
    log_filepath  = os.path.join(log_savepath, f'{log_name}.txt')

    if not os.path.exists(log_savepath):
        os.makedirs(log_savepath)

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
