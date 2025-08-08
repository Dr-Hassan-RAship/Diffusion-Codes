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
BASE_DIR            = "/media/ee/DATA/Talha_Nehal/Datasets/Kvasir-SEG"  # Path to the dataset root directory
TRAINSIZE           = 256  # Target size for resizing images and masks
BATCH_SIZE          = 6  # Batch size for dataloaders
SPLIT_RATIOS        = (800, 100, 100)  # Train, validation, test split ratios # (800, 100, 100)
FORMAT              = True  # If True, train/val/test subdirectories already exist
CLASSIFICATION_TYPE = "binary"  # 'binary' or 'multiclass'

# -------------------------------------------------------------------------------#
OPT = {
    "optimizer"     : "AdamW",
    "lr"            : 5e-5,
    "weight_decay"  : 1e-3,
    "betas"         : (0.9, 0.999),
    "warmup_ratio"  : 0.1,
}
# Scheduler_Configuration
USE_SCHEDULER       = False
SCHEDULER_TYPE      = "ddpm"  # Options: 'cosine_warmup', 'lambda_warmup_cosine', 'cosine_annealing', 'ddpm'
SCHEDULER_KWARGS    = {
    "lr_min"                : 1e-6,
    "lr_max"                : 5e-5,
    "lr_start"              : 0.0,
    "verbosity_interval"    : 100,
}

# ------------------------------------------------------------------------------#
# Training configuration
SEED                = 23  # Random seed for reproducibility
N_EPOCHS            = 5000  # Number of training epochs
VAL_INTERVAL        = 5  # Validate every n epochs (can reduce)
MODEL_SAVE_INTERVAL = 1  # Save model every n epochs
NUM_TRAIN_TIMESTEPS = 1000  # i.e., diffusion steps (T)

# --------------------------------------------------------------------------------#
# Diffusion Settings

NUM_TRAIN_TIMESTEPS = 1000  # i.e., diffusion steps (T)
BETA_START          = 0.0015  # 0.0001 default
BETA_END            = 0.0195  # 0.02 default
NOISE_SCHEDULER     = "linear"  # {linear, scaled_linear, squaredcos_cap_v2}, squaredcos_cap_v2 is cosine schedule
SCHEDULER           = "DDPM"  # {DDPM, DDIM}
ETA                 = 0.0  # Weight for noise added in DDIM (eta = 1 for DDPM, eta = 0 for deterministic DDIM)
VAR_NOISE           = False
DETERMINISTIC_TAU   = True  # Whether to use deterministic vae latent representation or not
DETERMINISTIC_ENC   = False

# ------------------------------------------------------------------------------#
# Experiment configuration
OPTIONAL_INFO   = f"tau_theta_learnable_and_backend_z0"
EXPERIMENT_NAME = f'machine--B{BATCH_SIZE}-E{N_EPOCHS}-V{VAL_INTERVAL}-T{NUM_TRAIN_TIMESTEPS}-S{SCHEDULER}'
RUN             = '02_' + OPTIONAL_INFO

# ------------------------------------------------------------------------------#
# Model configuration for Diffusion i.e., UNET --> matched with SDSeg
UNET_PARAMS = { "sample_size"       : TRAINSIZE // 8,
                "in_channels"       : 8,  # Using latent space input (z = 4 + concatenation), so latent dimensions match autoencoder
                "out_channels"      : 4,  # Latent space output before decoder
                "layers_per_block"  : 2,
                "block_out_channels": (192, 384, 384, 768, 768), #  (128, 256, 256, 512)
                "down_block_types"  : ("DownBlock2D",) * 3 + ('AttnDownBlock2D',) * 1 + ('DownBlock2D',) * 1,
                "up_block_types"    : ("UpBlock2D",) * 1 + ('AttnUpBlock2D',) * 1 + ('UpBlock2D',) * 3,
              } # num_head_channels = model_channels (192) // num_heads (8)

LDM_SNAPSHOT_DIR     = "./results/" + RUN + f"/ldm-" + EXPERIMENT_NAME
LDM_SCALE_FACTOR     = 0.18125


# ------------------------------------------------------------------------------#
# Placeholder for inference configuration
class InferenceConfig:
    N_PREDS             = 1
    MODEL_EPOCHS        = [x for x in range(10, 771, 10)]            # Epoch of the list of models to load.
    NUM_SAMPLES         = 2                  # Number of samples
    INFERER_SCHEDULER   = 'DDIM'
    TRAIN_TIMESTEPS     = NUM_TRAIN_TIMESTEPS
    ONE_X_ONE           = False # make it False if training
    INFERENCE_TIMESTEPS = 10 if INFERER_SCHEDULER == 'DDIM' else NUM_TRAIN_TIMESTEPS
    SAVE_FOLDER         = LDM_SNAPSHOT_DIR + f"/inference-M{MODEL_EPOCHS}-E{N_EPOCHS}-t{NUM_TRAIN_TIMESTEPS}-S{SCHEDULER}-SP{NUM_SAMPLES}-It{INFERENCE_TIMESTEPS}"  # Save folder for inference results
    SAVE_INTERMEDIATES  = False
    METRIC_REPORT       = True

do = InferenceConfig()
# ------------------------------------------------------------------------------#

# CONSIDER THE USAGE OF AUTOENCODER TINY FOR INCREASED BATCH SIZE
"""
AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float16)
{
  "_class_name": "AutoencoderTiny",
  "_diffusers_version": "0.30.0",
  "act_fn": "relu",
  "decoder_block_out_channels": [
    64,
    64,
    64,
    64
  ],
  "encoder_block_out_channels": [
    64,
    64,
    64,
    64
  ],
  "force_upcast": false,
  "in_channels": 3,
  "latent_channels": 4,
  "latent_magnitude": 3,
  "latent_shift": 0.5,
  "num_decoder_blocks": [
    3,
    3,
    3,
    1
  ],
  "num_encoder_blocks": [
    1,
    3,
    3,
    3
  ],
  "out_channels": 3,
  "scaling_factor": 1.0,
  "shift_factor": 0.0,
  "upsample_fn": "nearest",
  "upsampling_scaling_factor": 2
}
"""
