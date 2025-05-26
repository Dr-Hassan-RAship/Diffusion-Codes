# ------------------------------------------------------------------------------#
#
# File name                 : config.py
# Purpose                   : Configuration for 2D Latent Diffusion Model (LDM)
# Usage                     : Imported by train_ldm.py and inference_ldm.py
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh, Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 20240001@lums.edu.pk, hassan.mohyuddin@lums.edu.pk
#
# Last Date                 : April 28, 2025
#
# ------------------------------------------------------------------------------#
# Dataset configuration
BASE_DIR            = "/media/ee/New Volume/Datasets/Kvasir-SEG"        # Path to the dataset root directory
TRAINSIZE           = 352                   # Target size for resizing images and masks
BATCH_SIZE          = 3                     # Batch size for dataloaders
SPLIT_RATIOS        = (800, 100, 100)       # Train, validation, test split ratios # (800, 100, 100)
FORMAT              = True                  # If True, train/val/test subdirectories already exist
CLASSIFICATION_TYPE = 'binary'              # 'binary' or 'multiclass'

#-------------------------------------------------------------------------------#
# Optimizer Configuration
OPT              = { "optimizer"      : "AdamW",
                     "lr"              : 5e-5,
                     "weight_decay"    : 1e-2,
                     "betas"           : (0.9, 0.999)
}
# Scheduler_Configuration
USE_SCHEDULER    = False
SCHEDULER_TYPE   = 'cosine_warmup'  # Options: 'cosine_warmup', 'lambda_warmup_cosine', 'cosine_annealing', 'ddpm'
SCHEDULER_KWARGS = {
    'lr_min': 1e-6,
    'lr_max': 1.0,
    'lr_start': 0.0,
    'eta_min': 1e-6,
    'verbosity_interval': 100
}

# ------------------------------------------------------------------------------#
# Training configuration
SEED                = 23          # Random seed for reproducibility
N_EPOCHS            = 5000          # Number of training epochs
VAL_INTERVAL        = 5             # Validate every n epochs (can reduce)
MODEL_SAVE_INTERVAL = 1             # Save model every n epochs
NUM_TRAIN_TIMESTEPS = 1000          # i.e., diffusion steps (T)

#--------------------------------------------------------------------------------#
NOISE_SCHEDULER     = "linear" # {linear, scaled_linear, squaredcos_cap_v2, sigmoid}
BETA_START          = 0.0015
BETA_END            = 0.0195
SCHEDULER           = 'DDPM'        # {DDPM, DDIM}
ETA                 = 0.0           # Weight for noise added in DDIM (eta = 1 for DDPM, eta = 0 for deterministic and DDIM)
VAR_NOISE           = False
DETERMINISTIC_TAU   = True          # Whether to use deterministic vae latent representation or not
DETERMINISTIC_ENC   = False

# ------------------------------------------------------------------------------#
# Experiment configuration
OPTIONAL_INFO   = f"with_{NOISE_SCHEDULER}_noise_scheduler"
# OPTIONAL_INFO   = f'with_new_data_split'
EXPERIMENT_NAME = f'machine--B{BATCH_SIZE}-E{N_EPOCHS}-V{VAL_INTERVAL}-T{NUM_TRAIN_TIMESTEPS}-S{SCHEDULER}'
RUN             = '02_' + OPTIONAL_INFO

#"block_out_channels": (192, 384, 384, 768, 768), #  (128, 256, 256, 512)
# ------------------------------------------------------------------------------#
# Model configuration for Diffusion i.e., UNET --> matched with SDSeg
UNET_PARAMS = { "sample_size"       : TRAINSIZE // 8,
                "in_channels"       : 8,  # Using latent space input (z = 4 + concatenation), so latent dimensions match autoencoder
                "out_channels"      : 4,  # Latent space output before decoder
                "layers_per_block"  : 2,
                "block_out_channels": (44, 44, 88, 88, 176, 176), #  (1, 1, 2, 2, 4, 4)
                "down_block_types"  : ("DownBlock2D",) * 4 + ("AttnDownBlock2D",) + ("DownBlock2D",), 
                "up_block_types"    : ("UpBlock2D",) * 1 + ("AttnUpBlock2D",) + ("UpBlock2D",) * 4,
                "norm_num_groups"   : 44,
                "attention_head_dim": 8
              } # num_head_channels = model_channels (192) // num_heads (8)

LDM_SNAPSHOT_DIR     = "./results/" + RUN + f"/ldm-" + EXPERIMENT_NAME
# LDM_SCALE_FACTOR   = 1.0

# ------------------------------------------------------------------------------#
# Placeholder for inference configuration
class InferenceConfig:
    N_PREDS             = 1
    MODEL_EPOCHS        = [460]              # Epoch of the list of models to load. 
    NUM_SAMPLES         = 2                 # Number of samples 
    INFERER_SCHEDULER   = 'DDIM'
    TRAIN_TIMESTEPS     = NUM_TRAIN_TIMESTEPS
    ONE_X_ONE           = True # make it False if training
    INFERENCE_TIMESTEPS = 10 if INFERER_SCHEDULER == 'DDIM' else NUM_TRAIN_TIMESTEPS
    SAVE_FOLDER         = LDM_SNAPSHOT_DIR + f"/inference-M{MODEL_EPOCHS if MODEL_EPOCHS != -1 else N_EPOCHS}-E{N_EPOCHS}-t{NUM_TRAIN_TIMESTEPS}-S{SCHEDULER}-SP{NUM_SAMPLES}-It{INFERENCE_TIMESTEPS}"  # Save folder for inference results
    SAVE_INTERMEDIATES  = False
    METRIC_REPORT       = True

do = InferenceConfig()
# ------------------------------------------------------------------------------#
