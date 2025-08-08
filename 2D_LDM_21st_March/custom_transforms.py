# ------------------------------------------------------------------------------#
#
# File name                 : custom_transforms.py
# Purpose                   : Custom transforms for image and mask augmentation
# Usage                     : Imported by dataset.py
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh, Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 20240001@lums.edu.pk, hassan.mohyuddin@lums.edu.pk
#
# Last Date                 : March 11, 2025
#
# ------------------------------------------------------------------------------#

import os, random, torch
import numpy               as np

from PIL                   import Image, ImageEnhance
from skimage.morphology    import disk, erosion, dilation, opening, closing

from config       import *


# ------------------------------------------------------------------------------#
def cv_random_flip(img, mask):
    """Randomly flip the image and mask horizontally."""
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

    return img, mask


# ------------------------------------------------------------------------------#
def randomCrop(image, mask):
    """
    Randomly crops an image and mask
    
    Args:
        image (PIL.Image): Input image.
        mask (PIL.Image): Corresponding mask.

    Returns:
        PIL.Image, PIL.Image: Cropped image and mask.
    """
    border = 30
    image_width, image_height = image.size


    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)

    # Randomly determine the cropping region
    left = np.random.randint(0, image_width - crop_win_width + 1)
    top = np.random.randint(0, image_height - crop_win_height + 1)
    right = left + crop_win_width
    bottom = top + crop_win_height

    # Crop image and mask
    image = image.crop((left, top, right, bottom))
    mask = mask.crop((left, top, right, bottom))

    return image, mask


# ------------------------------------------------------------------------------#
def random_rotation(img, mask):
    """Randomly rotate the image and mask within a specified angle range. The source code uses bicubic for mask/label too"""
    if random.random() > 0.8:
        angle = random.randint(-15, 15)
        img = img.rotate(angle, resample=Image.BICUBIC)
        mask = mask.rotate(angle, resample=Image.NEAREST)

    return img, mask


# ------------------------------------------------------------------------------#
def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)

    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)

    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)

    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)

    return image


# ------------------------------------------------------------------------------#
def swap_pixels(mask, prob=0.25):

    mask = (np.array(mask))[:, :, np.newaxis]
    w, h, c = mask.shape
    for ch in range(c):
        img = mask[:, :, ch]
        if np.random.uniform(low=0.0, high=1.0) <= prob:
            sz = np.random.randint(3, 9)
            footprint = disk(sz)
            indices = np.asarray(np.where(np.abs(img - erosion(img, footprint)) > 0))
            remove_idx = [idx for idx, x in enumerate(indices[0]) if (x + sz >= w)]
            indices = np.delete(indices, remove_idx, 1)
            remove_idx = [idx for idx, x in enumerate(indices[1]) if (x + sz >= h)]
            indices = np.delete(indices, remove_idx, 1)
            indices = indices.astype(int)

            idx = np.random.choice(
                range(len(indices[0])), size=len(indices[0]) // 10
            ).astype(int)
            x_l_vals = img[indices[0, idx], indices[1, idx] - sz]
            x_vals = img[indices[0, idx], indices[1, idx]]
            img[indices[0, idx], indices[1, idx] - sz] = x_vals
            img[indices[0, idx], indices[1, idx]] = x_l_vals

            idx = np.random.choice(
                range(len(indices[0])), size=len(indices[0]) // 10
            ).astype(int)
            x_l_vals = img[indices[0, idx], indices[1, idx] + sz]
            x_vals = img[indices[0, idx], indices[1, idx]]
            img[indices[0, idx], indices[1, idx] + sz] = x_vals
            img[indices[0, idx], indices[1, idx]] = x_l_vals

            idx = np.random.choice(
                range(len(indices[0])), size=len(indices[0]) // 10
            ).astype(int)
            x_l_vals = img[indices[0, idx] + sz, indices[1, idx]]
            x_vals = img[indices[0, idx], indices[1, idx]]
            img[indices[0, idx] + sz, indices[1, idx]] = x_vals
            img[indices[0, idx], indices[1, idx]] = x_l_vals

            idx = np.random.choice(
                range(len(indices[0])), size=len(indices[0]) // 10
            ).astype(int)
            x_l_vals = img[indices[0, idx] - sz, indices[1, idx]]
            x_vals = img[indices[0, idx], indices[1, idx]]
            img[indices[0, idx] - sz, indices[1, idx]] = x_vals
            img[indices[0, idx], indices[1, idx]] = x_l_vals
    
    return Image.fromarray(mask.squeeze(-1))


# ------------------------------------------------------------------------------#
def morph_mask(mask, prob=0.25):
    """
    Apply morphological operations to the image and mask.
    """
    mask = np.array(mask)[:, :, np.newaxis]
    w, h, c = mask.shape
    for ch in range(c):
        img = mask[:, :, ch]
        if np.random.uniform(low=0.0, high=1.0) <= prob:
            morph_flag = np.random.randint(0, 4)
            footprint = disk(np.random.randint(1, 5))
            if morph_flag == 0:
                img = erosion(img, footprint)
            elif morph_flag == 1:
                img = dilation(img, footprint)
            elif morph_flag == 2:
                img = opening(img, footprint)
            elif morph_flag == 3:
                img = closing(img, footprint)

        mask[:, :, ch] = img

    return Image.fromarray(mask.squeeze(-1))


# ------------------------------------------------------------------------------#
def randomPeper(mask):
    mask = (np.array(mask))
    noiseNum = int(0.0015 * mask.shape[0] * mask.shape[1])
    for i in range(noiseNum):
        randX = random.randint(0, mask.shape[0] - 1)
        randY = random.randint(0, mask.shape[1] - 1)

        if random.randint(0, 1) == 0:
            mask[randX, randY] = 0
        else:
            mask[randX, randY] = 255

    return Image.fromarray(mask)


# ------------------------------------------------------------------------------#
def random_patch_swap(tensor, patch_size, num_swaps):
    """
    Randomly swaps patches of size `patch_size x patch_size` in the tensor.

    Args:
        tensor (torch.Tensor): Input tensor (image).
        patch_size (int): Size of the square patches to swap.
        num_swaps (int): Number of patch swaps to perform.

    Returns:
        torch.Tensor: Tensor with randomly swapped patches.
    """
    tensor = torch.from_numpy(np.array(tensor))

    _, h, w = tensor.shape  # Get the height and width of the image

    for _ in range(num_swaps):
        # Ensure different patches are selected
        while True:
            x1, y1 = np.random.randint(0, w - patch_size), np.random.randint(0, h - patch_size)
            x2, y2 = np.random.randint(0, w - patch_size), np.random.randint(0, h - patch_size)

            # Ensure patches do not overlap
            if (x1 != x2 or y1 != y2):
                break

        # Extract patches
        patch1 = tensor[:, y1 : y1 + patch_size, x1 : x1 + patch_size].clone()
        patch2 = tensor[:, y2 : y2 + patch_size, x2 : x2 + patch_size].clone()

        # Swap patches
        tensor[:, y1 : y1 + patch_size, x1 : x1 + patch_size] = patch2
        tensor[:, y2 : y2 + patch_size, x2 : x2 + patch_size] = patch1

    return tensor

# ------------------------------------------------------------------------------#
def apply_augmentations(img, mask):
    """Apply selected augmentations dynamically based on config."""
    if AUGMENTATION_CONFIG['image'][IMAGE_AUGMENTATION_OPTION]["random_flip"]:
        img, mask    = cv_random_flip(img, mask)

    if AUGMENTATION_CONFIG['image'][IMAGE_AUGMENTATION_OPTION]["random_crop"]:
        img, mask    = randomCrop(img, mask)

    if AUGMENTATION_CONFIG['image'][IMAGE_AUGMENTATION_OPTION]["random_rotation"]:
        img, mask    = random_rotation(img, mask)

    # Tap out the clean image and the clean mask here!
    clean_img, clean_mask = img.copy(), mask.copy() 

    # assert clean mask should be a binary image and should contain unique values equal to the num_classes

    if AUGMENTATION_CONFIG['image'][IMAGE_AUGMENTATION_OPTION]["color_enhance"]:
        img          = colorEnhance(img)

    if AUGMENTATION_CONFIG['image'][IMAGE_AUGMENTATION_OPTION]["swap_patches"]:
        img          = random_patch_swap(img, patch_size=5, num_swaps=10)

    if AUGMENTATION_CONFIG["mask"]["random_pepper"]:
        mask         = randomPeper(mask)

    if AUGMENTATION_CONFIG["mask"]["swap_pixels"]:
        mask         = swap_pixels(mask)

    if AUGMENTATION_CONFIG["mask"]["morph"]:
        mask         = morph_mask(mask)

    return clean_img, clean_mask, img, mask

# ------------------------------------------------------------------------------#
