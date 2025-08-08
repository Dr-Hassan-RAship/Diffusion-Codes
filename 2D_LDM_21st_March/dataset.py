# ------------------------------------------------------------------------------#
#
# File name                 : dataset.py
# Purpose                   : Data loader and splitter for 2D LDM segmentation
# Usage (command)           : Used as a module for training and inference
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh, Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 202410001@lums.edu.pk, hassan.mohyuddin@lums.edu.pk
#
# Last Date                 : March 13, 2025
#
# ------------------------------------------------------------------------------#

import os, random, shutil

from torchvision        import transforms
from torch.utils.data   import DataLoader, Dataset
from custom_transforms  import *
from config    import *
from custom_transforms  import *

# ------------------------------------------------------------------------------#
def split_dataset(base_dir, split_ratios=(600, 200, 200)):
    """
    Split the dataset into training, validation, and test subsets.

    Parameters:
    - base_dir    : Path to the Kvasir dataset containing `images` and `masks` subfolders.
    - split_ratios: Tuple specifying the number of samples for train, val, and test splits.

    Returns:
    - None: The dataset will be split into subfolders `train`, `val`, and `test` inside `images` and `masks`.
    """
    image_dir = os.path.join(base_dir, "images")
    mask_dir  = os.path.join(base_dir, "masks")

    assert os.path.exists(image_dir), "Images folder not found!"
    assert os.path.exists(mask_dir), "Masks folder not found!"

    os.makedirs(os.path.join(base_dir, "train", "images"), exist_ok = True)
    os.makedirs(os.path.join(base_dir, "train", "masks"),  exist_ok = True)
    os.makedirs(os.path.join(base_dir, "val", "images"),   exist_ok = True)
    os.makedirs(os.path.join(base_dir, "val", "masks"),    exist_ok = True)
    os.makedirs(os.path.join(base_dir, "test", "images"),  exist_ok = True)
    os.makedirs(os.path.join(base_dir, "test", "masks"),   exist_ok = True)

    image_files = sorted(os.listdir(image_dir))
    mask_files  = sorted(os.listdir(mask_dir))
    
    # assert for ensurin image_files and mask_files have the same naming for each index
    for img_file, mask_file in zip(image_files, mask_files):
        assert img_file.split(".")[0] == mask_file.split(".")[0], "Image and mask files do not match!"

    # asserting length of image_files and mask_files are equal
    assert len(image_files) == len(mask_files), "Number of images and masks do not match!"

    paired_files = list(zip(image_files, mask_files))
    random.shuffle(paired_files)

    train_files = paired_files[:split_ratios[0]]
    val_files   = paired_files[split_ratios[0]:split_ratios[0] + split_ratios[1]]
    test_files  = paired_files[split_ratios[0] + split_ratios[1]:]

    for split, files in zip(["train", "val", "test"], [train_files, val_files, test_files]):
        for img_file, mask_file in files:
            shutil.copy(os.path.join(image_dir, img_file), os.path.join(base_dir, split, "images", img_file))
            shutil.copy(os.path.join(mask_dir, mask_file), os.path.join(base_dir, split, "masks", mask_file))

    print("Dataset successfully split into train, val, and test subsets.")

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
    def __init__(self, base_dir, split="train", trainsize=256, augment=False):
        self.image_dir   = os.path.join(base_dir, split, "images")
        self.mask_dir    = os.path.join(base_dir, split, "masks")
        self.image_files = sorted(os.listdir(self.image_dir))
        self.mask_files  = sorted(os.listdir(self.mask_dir))
        
        # insert assert statement that the order matches for the names of the image and mask files
        assert len(self.image_files) == len(self.mask_files), "Number of images and masks do not match!"

        self.trainsize   = trainsize
        self.augment     = augment

        self.img_transform  = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),])
        
        self.mask_transform = transforms.Compose([transforms.Resize((self.trainsize, self.trainsize)),
                                                  transforms.ToTensor(),])
    # --------------------------------------------------------------------------#
    def __len__(self):
        return len(self.image_files)

    # --------------------------------------------------------------------------#
    def __getitem__(self, idx):
        img_path  = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        assert (img_path.split(".")[0]).rsplit('/', 1)[-1] == mask_path.split(".")[0].rsplit('/', 1)[-1], "Image and mask file name do not match!"

        img       = Image.open(img_path).convert("RGB") 
        mask      = Image.open(mask_path).convert("L")
        
        if self.augment:
            clean_img, clean_mask, noisy_img, noisy_mask = apply_augmentations(img, mask)

            clean_img, noisy_img  = self.img_transform(clean_img), self.img_transform(noisy_img)
            clean_mask, noisy_mask = self.mask_transform(clean_mask), self.mask_transform(noisy_mask)

            return {"patient_id": idx, "clean_image": clean_img, "noisy_image": noisy_img, 
                    "clean_mask": clean_mask, "noisy_mask": noisy_mask}
       
        # return the below if augment is false
        if not self.augment:
            clean_img, clean_mask = img.copy(), mask.copy()
            clean_img  = self.img_transform(clean_img)
            clean_mask = self.mask_transform(clean_mask)
            return {"patient_id": idx, "image": clean_img, "mask": clean_mask}
        
# ------------------------------------------------------------------------------#
def get_dataloaders(base_dir, split_ratio, split = 'train', trainsize = 256, batch_size = 16, num_workers = 4, format = True):
    """Get train, validation, and test dataloaders."""
    
    augment = True 
    
    if not format          : split_dataset(base_dir, split_ratios = split_ratio)
    else                   : print(f'Dataset already split into train, val and test directories')
    if split == 'test' and not do.AUGMENT: augment = False
    # if not do.AUGMENT: 
    #     if split != 'train'    : augment = False
    
    dataset    = KvasirPolypDataset(base_dir, split = split, trainsize = trainsize, augment = augment)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True if split == 'train' else False, num_workers = num_workers, pin_memory = True)

    return dataloader

# ------------------------------------------------------------------------------#