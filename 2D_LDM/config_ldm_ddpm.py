# ------------------------------------------------------------------------------#
#
# File name                 : config_ldm_ddpm.py
# Purpose                   : Configuration for 2D Latent Diffusion Model (LDM)
# Usage                     : Imported by train_ldm.py and inference_ldm.py
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh, Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 20240001@lums.edu.pk, hassan.mohyuddin@lums.edu.pk
#
# Last Date                 : March 13, 2025
#
# ------------------------------------------------------------------------------#

# Dataset configuration
BASE_DIR            = "/media/ee/DATA/Talha_Nehal/Datasets/Kvasir-SEG"        # Path to the dataset root directory
DIMENSION           = '2d'                  # '2d' or '3d'
TRAINSIZE           = 256                   # Target size for resizing images and masks
BATCH_SIZE          = 5                     # Batch size for dataloaders
SPLIT_RATIOS        = (600, 200, 200)       # Train, validation, test split ratios
FORMAT              = True                  # If True, train/val/test subdirectories already exist
CLASSIFICATION_TYPE = 'binary'              # 'binary' or 'multiclass'

# ------------------------------------------------------------------------------#
# Augmentation Configuration for Image and Mask
AUGMENTATION_CONFIG = {"image": {"option_1": {"random_flip"    : True,
                                              "random_crop"    : True,
                                              "random_rotation": True,
                                              "color_enhance"  : True,
                                              "swap_patches"   : False},
                                 "option_2": {"random_flip"    : True,
                                              "random_crop"    : True,
                                              "random_rotation": False,
                                              "color_enhance"  : False,
                                              "swap_patches"   : True}},
                       "mask": {"random_flip"   : True,
                                "random_crop"   : True,
                                "random_pepper" : True,
                                "swap_pixels"   : True,
                                "morph"         : True}}

# Select which augmentation option to use for images
IMAGE_AUGMENTATION_OPTION = "option_1"  # Change to "option_2" for swapping patches

# ------------------------------------------------------------------------------#
# Training configuration
SEED                = 1337          # Random seed for reproducibility
N_EPOCHS            = 250           # Number of training epochs
LR                  = 1.0e-4        # Learning rate for the optimizer
VAL_INTERVAL        = 1             # Validate every n epochs (can reduce)
MODEL_SAVE_INTERVAL = 1             # Save model every n epochs
NUM_TRAIN_TIMESTEPS = 1000          # i.e., diffusion steps (T)
NOISE_SCHEDULER     = "linear_beta" # {linear_beta, cosine_beta}
SCHEDULER           = 'DDIM'        # {DDPM, DDIM}
# RESUME_PATH         = LDM_SNAPSHOT_DIR + f'/models/model_epoch_499.pth'

NUM_INFERENCE_TIMESTEPS = NUM_TRAIN_TIMESTEPS // 10 if SCHEDULER == 'DDIM' else NUM_TRAIN_TIMESTEPS

# ------------------------------------------------------------------------------#
# Experiment configuration
OPTIONAL_INFO   = "with_sdseg_settings"
EXPERIMENT_NAME = f'machine--B{BATCH_SIZE}-E{N_EPOCHS}-V{VAL_INTERVAL}-T{NUM_TRAIN_TIMESTEPS}-S{SCHEDULER}'
RUN             = '02_' + OPTIONAL_INFO

# ------------------------------------------------------------------------------#
# Autoencoder configuration (for both dae image and masks)

# [talha] --> make sure latent dim is 32 ! not 64 x 64

DAE_IMAGE_SNAPSHOT_DIR = "./results/" + RUN + "/dae-image"
DAE_MASK_SNAPSHOT_DIR  = "./results/" + RUN + "/dae-mask"
DAE_IMAGE_PARAMS       = {"spatial_dims"              : 2,
                          "in_channels"               : 3,
                          "latent_channels"           : 4, # (= Z in SDSeg paper)
                          "out_channels"              : 3,
                          "channels"                  : (128, 256, 512, 512), # to match SDSeg paper i.e. 32 latent dim
                          "num_res_blocks"            : 2,
                          "attention_levels"          : (False, False, False, False),
                          "with_encoder_nonlocal_attn": True, # (as per SDSeg paper to ensure middle block of encoder is as required)
                          "with_decoder_nonlocal_attn": True, # (as per SDSeg paper to ensure middle block of decoder is as required)
                          "use_flash_attention"       : True}
DAE_MASK_PARAMS        = {"spatial_dims"              : 2,
                          "in_channels"               : 3,
                          "latent_channels"           : 4, # (= Z in SDSeg paper)
                          "out_channels"              : 3,
                          "channels"                  : (128, 256, 512, 512), # to match SDSeg paper i.e. 32 latent dim
                          "num_res_blocks"            : 2,
                          "attention_levels"          : (False, False, False, False),
                          "with_encoder_nonlocal_attn": True, # (as per SDSeg paper to ensure middle block of encoder is as required)
                          "with_decoder_nonlocal_attn": True, # (as per SDSeg paper to ensure middle block of decoder is as required)
                          "use_flash_attention"       : True}
IMAGE_DISCRIM_PARAMS   = {"spatial_dims": 2,
                          "num_layers_d": 3,
                          "num_channels": 64,
                          "in_channels" : 3,
                          "out_channels": 3}
MASK_DISCRIM_PARAMS    = {"spatial_dims": 2,
                          "num_layers_d": 3,
                          "num_channels": 64,
                          "in_channels" : 1,
                          "out_channels": 1}

# ------------------------------------------------------------------------------#
# Additional loss and training parameters for autoencoder
KL_WEIGHT         = 1e-6  # Weight for KL loss
PERCEPTUAL_WEIGHT = 0.001 # Weight for perceptual loss
ADV_WEIGHT        = 0.01  # Weight for adversarial loss
WARM_UP_EPOCHS    = 10    # Warm-up epochs before applying adversarial loss

# ------------------------------------------------------------------------------#
# Model configuration for Diffusion i.e., UNET --> [talha] try to modify the params to match SDSeg params config
MODEL_PARAMS = {"spatial_dims"     : 2 if DIMENSION == "2d" else 3,
                "in_channels"      : 8,  # Using latent space input (z = 4 + concatenation), so latent dimensions match autoencoder
                "out_channels"     : 4,  # Latent space output before decoder
                "num_channels"     : (192, 384, 384, 768, 768), # (192, 384, 384, 768, 768)
                "attention_levels" : (True, True, True, True, True),
                "num_res_blocks"   : 2,
                "num_head_channels": 24} # num_head_channels = model_channels (192) // num_heads (8)


# ------------------------------------------------------------------------------#
LDM_SNAPSHOT_DIR     = "./results/" + RUN + f"/ldm-{DIMENSION}-{CLASSIFICATION_TYPE}-" + EXPERIMENT_NAME
LDM_SCALE_FACTOR     = 1.0
LDM_SCHEDULER_PARAMS = {"schedule"           : NOISE_SCHEDULER}
# ------------------------------------------------------------------------------#
# Placeholder for inference configuration
class InferenceConfig:
    # TEST_FOLDER         = os.path.join(BASE_DIR, "Test")
    N_PREDS             = 1
    AUGMENT             = False
    MODEL_EPOCH         = 700                # Epoch of the model to load (-1 for final model)
    NUM_SAMPLES         = 10                # Number of samples 
    SAVE_FOLDER         = LDM_SNAPSHOT_DIR + f"/inference-A{AUGMENT}-M{MODEL_EPOCH if MODEL_EPOCH != -1 else N_EPOCHS}-E{N_EPOCHS}-t{NUM_TRAIN_TIMESTEPS}-S{SCHEDULER}-SP{NUM_SAMPLES}"  # Save folder for inference results
    TRAIN_TIMESTEPS     = NUM_TRAIN_TIMESTEPS
    INFERENCE_TIMESTEPS = NUM_INFERENCE_TIMESTEPS
    SAVE_INTERMEDIATES  = True
    METRIC_REPORT       = True

do = InferenceConfig()
# ------------------------------------------------------------------------------#