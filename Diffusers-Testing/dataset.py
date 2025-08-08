# ------------------------------------------------------------------------------#
#
# File name                 : dataset.py
# Purpose                   : Data loader and splitter for 2D LDM segmentation
# Usage (command)           : Used as a module for training and inference
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh, Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 202410001@lums.edu.pk, hassan.mohyuddin@lums.edu.pk
#
# Last Date                 : March 25, 2025
#
# ------------------------------------------------------------------------------#

import os, random, shutil, cv2, torch, PIL, glob

import numpy                                as np
import torchvision.transforms.functional    as TF

from PIL                                    import Image
from torchvision                            import transforms

from torch.utils.data                       import DataLoader, Dataset
from config                                 import *


# ------------------------------------------------------------------------------#
def split_dataset(base_dir, split_ratios=(800, 100, 100)):
    """
    Split the dataset into training, validation, and test subsets.
    Cleans up existing splits before recreating them.
    """

    image_dir = os.path.join(base_dir, "images")
    mask_dir = os.path.join(base_dir, "masks")

    assert os.path.exists(image_dir), "Images folder not found!"
    assert os.path.exists(mask_dir), "Masks folder not found!"

    # Clean old splits if any
    for split in ["train", "val", "test"]:
        split_img_dir = os.path.join(base_dir, split, "images")
        split_mask_dir = os.path.join(base_dir, split, "masks")
        if os.path.exists(split_img_dir):
            shutil.rmtree(split_img_dir)
        if os.path.exists(split_mask_dir):
            shutil.rmtree(split_mask_dir)

    # Recreate fresh directories
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(base_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, split, "masks"), exist_ok=True)

    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))

    assert len(image_files) == len(
        mask_files
    ), "Number of images and masks do not match!"
    for img_file, mask_file in zip(image_files, mask_files):
        assert (
            img_file.split(".")[0] == mask_file.split(".")[0]
        ), "Image and mask files do not match!"

    paired_files = list(zip(image_files, mask_files))
    random.shuffle(paired_files)

    total_files = sum(split_ratios)
    assert total_files <= len(
        paired_files
    ), f"Split exceeds dataset size ({len(paired_files)})."

    train_files = paired_files[: split_ratios[0]]
    val_files = paired_files[split_ratios[0] : split_ratios[0] + split_ratios[1]]
    test_files = paired_files[split_ratios[0] + split_ratios[1] :]

    for split, files in zip(
        ["train", "val", "test"], [train_files, val_files, test_files]
    ):
        for img_file, mask_file in files:
            shutil.copy(
                os.path.join(image_dir, img_file),
                os.path.join(base_dir, split, "images", img_file),
            )
            shutil.copy(
                os.path.join(mask_dir, mask_file),
                os.path.join(base_dir, split, "masks", mask_file),
            )

    print("✅ Dataset successfully split into train, val, and test.")

# ------------------------------------------------------------------------------#
class KvasirPolypDataset(Dataset):
    """
    Custom Dataset for Kvasir polyp segmentation.

    Parameters:
    - base_dir : Base directory containing `images` and `masks` subfolders.
    - split    : Subset to load ('train', 'val', or 'test').
    - trainsize: Target size for resizing images and masks.
    - augment  : Whether to apply data augmentation.
    """

    def __init__(self, base_dir, split="train", trainsize=256):
        self.split = split
        self.trainsize = trainsize

        self.image_dir = os.path.join(base_dir, self.split, "images")
        self.mask_dir  = os.path.join(base_dir, self.split, "masks")

        self.image_files = sorted(os.listdir(self.image_dir))
        self.mask_files = sorted(os.listdir(self.mask_dir))

        # insert assert statement that the order matches for the names of the image and mask files
        assert len(self.image_files) == len(
            self.mask_files
        ), "Number of images and masks do not match!"

        self.transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        )

    # --------------------------------------------------------------------------#
    def __len__(self):
        return len(self.image_files)

    # --------------------------------------------------------------------------#
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        assert (img_path.split(".")[0]).rsplit("/", 1)[-1] == mask_path.split(".")[
            0
        ].rsplit("/", 1)[-1], "Image and mask file name do not match!"

        # Read image and corresponding mask
        img = Image.fromarray(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
        mask = Image.fromarray(cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB))

        # Resize image and mask
        img = img.resize((self.trainsize, self.trainsize), resample=PIL.Image.BICUBIC)
        mask = mask.resize((self.trainsize, self.trainsize), resample=PIL.Image.NEAREST)

        # Processing Pipeline for SDSeg involves transform followed by dynamic range setting for image (regardless of split)
        # but for mask the dynamic range setting is only true for 'train' and 'val' splits
        # however we will still do it and then later map to 0-1 in inference if needed

        if self.split == "train" or self.split == "val":
            img, mask = utilize_transformation(img, mask, self.transform)

            # Adjust dynamic range in [-1, +1]; np.float32
            img = np.array(img).astype(np.float32) / 255.0
            img = (img * 2.0) - 1.0  # Shape: (256, 256, 3)
            mask = (np.array(mask) > 128).astype(np.float32)
            mask = (mask * 2.0) - 1.0

            # Confirm dynamic ranges
            assert np.max(img) <= 1.0 and np.min(img) >= -1.0
            assert np.max(mask) <= 1.0 and np.min(mask) >= -1.0

            # Convert img and mask to tensor and permute dimensions
            img, mask = torch.from_numpy(img).permute(2, 0, 1), torch.from_numpy(
                mask
            ).permute(2, 0, 1)

            return {"patient_id": idx, "aug_image": img, "aug_mask": mask}

        else:
            # Adjust dynamic range in [-1, +1]; np.float32
            img = np.array(img).astype(np.float32) / 255.0
            img = (img * 2.0) - 1.0
            mask = (np.array(mask) > 128).astype(np.float32)
            mask = (mask * 2.0) - 1.0

            # Confirm dynamic ranges
            assert np.max(img) <= 1.0 and np.min(img) >= -1.0
            assert np.max(mask) <= 1.0 and np.min(mask) >= -1.0

            # Convert img and mask to tensor and permute dimensions
            img, mask = torch.from_numpy(img).permute(2, 0, 1), torch.from_numpy(
                mask
            ).permute(2, 0, 1)

            return {"patient_id": idx, "image": img, "mask": mask}


# ------------------------------------------------------------------------------#
def get_dataloaders(base_dir, split_ratio, split="train", trainsize=256, batch_size=16, num_workers=4, format=True):
    """Get train, validation, and test dataloaders."""

    if not format:
        split_dataset(base_dir, split_ratios=split_ratio)
    else:
        print(f"Dataset already split into train, val and test directories")

    dataset = KvasirPolypDataset(base_dir, split=split, trainsize=trainsize)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True if split == "train" else False,
                            num_workers=num_workers,
                            pin_memory=True,
    )

    return dataloader


# -----------------------------------------------------------------------------#
def utilize_transformation(img, mask, transforms_op):
    """
    Applying transformations to img and mask with same random seed
    """

    state = torch.get_rng_state()
    mask = transforms_op(mask)
    torch.set_rng_state(state)
    img = transforms_op(img)

    return img, mask


# -----------------------------------------------------------------------------#
