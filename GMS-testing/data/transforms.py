# ------------------------------------------------------------------------------#
#
# File name                 : transforms.py
# Purpose                   : Centralized Albumentations transform builders.
# Usage                     : build_transforms("train", 224)
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh, Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 202410001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : June 21, 2025
# ------------------------------------------------------------------------------#


# --------------------------- Module imports -----------------------------------#
import albumentations as A
from   albumentations.pytorch import ToTensorV2

# --------------------------- Transform builder --------------------------------#
def build_transforms(stage: str = "train", img_size: int = 224):
    """
    Returns an Albumentations `Compose` transform for the given stage.

    Args:
        stage     : "train" | "val" | "test"
        img_size  : Final resize dimension (square).

    Returns:
        Albumentations Compose object.
    """
    # Note instead of A.ShiftScaleRotate(shift_limit = 0.15, scale_limit = 0.1, rotate_limit = 20, p = 0.4)
    # use A.Affine(translate_percent=0.15, scale=0.1, rotate=20, p=0.4) which is more general and does not give warning
    if stage == "train":
        tfm = A.Compose([
            A.ToFloat(max_value = 255.0),
            A.HorizontalFlip(p = 0.5),
            A.VerticalFlip(p = 0.5),
            A.RandomRotate90(p = 0.5),
            A.ColorJitter(0.1, 0.1, 0.1, 0.1, p = 0.2),
            A.Affine(translate_percent=0.15, scale=0.1, rotate=20, p=0.4),
            A.Resize(img_size, img_size),
            ToTensorV2(),
        ])
    else:  # val / test
        tfm = A.Compose([
            A.ToFloat(max_value = 255.0),
            A.Resize(img_size, img_size),
            ToTensorV2(),
        ])
    return tfm

# --------------------------------- End -----------------------------------------#
