# ------------------------------------------------------------------------------#
#
# File name                 : config_ldm_ddpm.py
# Purpose                   : Configuration for 2D Latent Diffusion Model (LDM)
# Usage                     : Imported by train_ldm.py and inference_ldm.py
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh, Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 20240001@lums.edu.pk, hassan.mohyuddin@lums.edu.pk
#
# Last Date                 : March 25, 2025
#
# ------------------------------------------------------------------------------#
# Dataset configuration
BASE_DIR            = "/media/ee/DATA/Talha_Nehal/Datasets/Kvasir-SEG"        # Path to the dataset root directory
TRAINSIZE           = 256                   # Target size for resizing images and masks
BATCH_SIZE          = 4                     # Batch size for dataloaders
SPLIT_RATIOS        = (800, 100, 100)       # Train, validation, test split ratios
FORMAT              = True                  # If True, train/val/test subdirectories already exist
CLASSIFICATION_TYPE = 'binary'              # 'binary' or 'multiclass'

# ------------------------------------------------------------------------------#
# Training configuration
SEED                = 1337          # Random seed for reproducibility
N_EPOCHS            = 500          # Number of training epochs
LR                  = 1.0e-5        # Learning rate for the optimizer # [talha] change it to be -5
VAL_INTERVAL        = 1             # Validate every n epochs (can reduce)
MODEL_SAVE_INTERVAL = 10             # Save model every n epochs
NUM_TRAIN_TIMESTEPS = 1000          # i.e., diffusion steps (T)
NOISE_SCHEDULER     = "linear_beta" # {linear_beta, cosine_beta}
SCHEDULER           = 'DDPM'        # {DDPM, DDIM}
DETERMINISTIC       = False         # Whether to use deterministic vae latent representation or not

# ------------------------------------------------------------------------------#
# Experiment configuration
OPTIONAL_INFO   = "with_sdseg_settings"
EXPERIMENT_NAME = f'machine--B{BATCH_SIZE}-E{N_EPOCHS}-V{VAL_INTERVAL}-T{NUM_TRAIN_TIMESTEPS}-S{SCHEDULER}'
RUN             = '01_' + OPTIONAL_INFO

# ------------------------------------------------------------------------------#
# Model configuration for Diffusion i.e., UNET --> matched with SDSeg
UNET_PARAMS = { "sample_size"        : TRAINSIZE // 8,
                "in_channels"        : 8,  # Using latent space input (z = 4 + concatenation), so latent dimensions match autoencoder
                "out_channels"       : 4,  # Latent space output before decoder
                "layers_per_block"   : 2,
                "block_out_channels" : (192, 384, 384, 768, 768), #  (128, 256, 256, 512)
                "down_block_types"   : ("DownBlock2D",) * 3 + ("AttnDownBlock2D",) + ("DownBlock2D",), 
                "up_block_types"     : ("UpBlock2D",) * 1 + ("AttnUpBlock2D",) + ("UpBlock2D",) * 3,
              } # num_head_channels = model_channels (192) // num_heads (8)

LDM_SNAPSHOT_DIR     = "./results/" + RUN + f"/ldm-" + EXPERIMENT_NAME
LDM_SCALE_FACTOR     = 1.0

# ------------------------------------------------------------------------------#
# Placeholder for inference configuration
class InferenceConfig:
    N_PREDS             = 1
    RESUME              = False
    MODEL_EPOCH         = 500               # Epoch of the model to load (-1 for final model)
    NUM_SAMPLES         = 10                 # Number of samples 
    SAVE_FOLDER         = LDM_SNAPSHOT_DIR + f"/inference-M{MODEL_EPOCH if MODEL_EPOCH != -1 else N_EPOCHS}-E{N_EPOCHS}-t{NUM_TRAIN_TIMESTEPS}-S{SCHEDULER}-SP{NUM_SAMPLES}"  # Save folder for inference results
    INFERER_SCHEDULER   = 'DDIM'
    TRAIN_TIMESTEPS     = NUM_TRAIN_TIMESTEPS
    INFERENCE_TIMESTEPS = NUM_TRAIN_TIMESTEPS // 10 if SCHEDULER == 'DDIM' else NUM_TRAIN_TIMESTEPS
    SAVE_INTERMEDIATES  = False
    METRIC_REPORT       = True

do = InferenceConfig()
# ------------------------------------------------------------------------------#