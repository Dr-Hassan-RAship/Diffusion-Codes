# ------------------------------------------------------------------------------#
#
# File name                 : utils.py
# Purpose                   : Helper functions in general

# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh, Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 202410001@lums.edu.pk, hassan.mohyuddin@lums.edu.pk
#
# Last Date                 : March 24, 2025
#-------------------------------------------------------------------------------#
import torch, os, csv, sys, logging

import numpy                as np

from scipy.ndimage          import distance_transform_edt
from skimage                import segmentation as skimage_seg

import torch.nn             as nn
import torch.nn.functional  as F

#------------------------------------------------------------------------------#
def pad_tensor_symmetrically(tensor: torch.Tensor) -> torch.Tensor:
    """
    Symmetrically pads a 3D tensor along the depth dimension (S) to ensure it has exactly 16 slices.

    Args:
        tensor (torch.Tensor): A 3D tensor of shape (H, W, S), where H is height, W is width, and S is the number of slices.

    Returns:
        torch.Tensor: A padded 3D tensor of shape (H, W, 16). If S >= 16, the original tensor is returned unchanged.


    Example Usage:
        >>> tensor        = torch.randn(32, 32, 10)
        >>> padded_tensor = pad_tensor_symmetrically(tensor)
        >>> print(padded_tensor.shape)
        torch.Size([32, 32, 16])
    """
    # Ensure the input tensor is 3D
    if tensor.ndim != 3:
        raise ValueError(f"Input tensor must be 3D (H x W x S). Got shape: {tensor.shape}")

    H, W, S = tensor.shape

    # Check if padding is needed
    if S < 16:
        padding_total = 16 - S  # Total number of padding slices needed
        is_odd        = padding_total % 2 != 0  # Check if padding_total is odd

        # Calculate padding for top and bottom
        if is_odd:
            # Randomly choose to add the extra slice to the top or bottom
            extra          = np.random.choice(['top', 'bottom'])
            padding_top    = padding_total // 2 + (1 if extra == 'top' else 0)
            padding_bottom = padding_total // 2 + (1 if extra == 'bottom' else 0)
        else:
            # Split padding equally between top and bottom
            padding_top = padding_bottom = padding_total // 2

        # Create zero-padding tensors for top and bottom
        padding_top_slices    = torch.zeros((H, W, padding_top), dtype=tensor.dtype, device=tensor.device)
        padding_bottom_slices = torch.zeros((H, W, padding_bottom), dtype=tensor.dtype, device=tensor.device)

        # Concatenate the padding slices with the original tensor along the depth dimension (axis=2)
        padded_tensor         = torch.cat([padding_top_slices, tensor, padding_bottom_slices], dim=2)
    else:
        # No padding needed if S >= 16
        padded_tensor = tensor

    return padded_tensor

#------------------------------------------------------------------------------#
def compute_sdm(mask: np.ndarray, device: str = "cuda") -> torch.Tensor:
    """
    Compute the Signed Distance Map (SDM) for a binary polyp mask.

    Args:
        mask (np.ndarray): A binary mask of shape (B, 1, H, W), where B is the batch size.
        device (str)    : The device to store the resulting tensor ("cpu" or "cuda").

    Returns:
        torch.Tensor: The Signed Distance Map of shape (B, 1, H, W), normalized to [-1, 1].
    """
    # Ensure the input is a NumPy array
    if not isinstance(mask, np.ndarray):
        raise TypeError("Input mask must be a NumPy array.")

    B, C, H, W = mask.shape
    assert C == 1, "Mask should have a single channel (B, 1, H, W)."
    
    sdm = np.zeros((B, 1, H, W), dtype=np.float32)  # Initialize SDM tensor

    for i in range(B):  # Iterate over the batch
        binary_mask = mask[i, 0].astype(np.uint8)  # Ensure it's binary (0 or 1)

        if np.all(binary_mask == 1):  # Fully foreground case
            sdm[i, 0] = -1.0
            continue
        if np.all(binary_mask == 0):  # Fully background case
            sdm[i, 0] = 1.0
            continue

        pos_dist = distance_transform_edt(binary_mask)   # Distance inside the mask
        neg_dist = distance_transform_edt(1 - binary_mask)  # Distance outside the mask

        # Normalize distances to [-1, 1]
        sdf = (neg_dist - np.min(neg_dist)) / (np.max(neg_dist) - np.min(neg_dist)) - \
              (pos_dist - np.min(pos_dist)) / (np.max(pos_dist) - np.min(pos_dist))

        # Set boundary pixels to 0
        boundary = skimage_seg.find_boundaries(binary_mask, mode="inner").astype(np.uint8)
        sdf[boundary == 1] = 0

        sdm[i, 0] = sdf  # Store computed SDM

    # Convert SDM to PyTorch tensor and move to the desired device
    return torch.from_numpy(sdm).to(device)


#------------------------------------------------------------------------------#
class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes  = n_classes

    def _one_hot_encoder(self, input_tensor):
        # Shape will be [24, 224, 224, 2]
        one_hot_labels  = F.one_hot(input_tensor, num_classes=2)

        # Rearrange the dimensions to [24, 2, 224, 224]
        one_hot_labels  = one_hot_labels.permute(0, 3, 1, 2)
        return one_hot_labels.float()

    def _dice_loss(self, score, target):
        target          = target.float()
        smooth          = 1e-10
        intersect       = torch.sum(score * target)
        y_sum           = torch.sum(target * target)
        z_sum           = torch.sum(score * score)
        loss            = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss            = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:         inputs = torch.softmax(inputs, dim=1)
        if weight is None:  weight = [1] * self.n_classes

        # Already one-hot encoded
        #target          = self._one_hot_encoder(target)

        assert inputs.size() == target.size(), 'predict & target shape do not match'

        class_wise_dice, loss = [], 0.0
        for i in range(0, self.n_classes):
            dice        = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss        += dice * weight[i]

        return loss / self.n_classes

#------------------------------------------------------------------------------#
def prepare_and_write_csv_file(snapshot_dir, list_entries):
    """
    For writing csv file with entries in eval_list
    """
    
    with open(os.path.join(snapshot_dir, 'logs.csv'), 'a') as csvfile:
        csv_logger = csv.writer(csvfile)
        csv_logger.writerow(list_entries)
        csvfile.flush()
#------------------------------------------------------------------------------#
def prepare_writer_layout():
    layout = {
        "Evaluation": {
            "Loss" : ["Multiline", ["loss/train epoch", "loss/val epoch"]]
        }
    }
    
    return layout
#-------------------------------------------------------------------------------#
def setup_logging(snapshot_dir, log_filename="logs.txt", level=logging.INFO, console=True):
    """
    Sets up logging to file and optionally to stdout.

    Args:
        snapshot_dir (str): Path to directory where log file will be saved.
        log_filename (str): Name of the log file.
        level (int): Logging level (e.g., logging.INFO).
        console (bool): If True, also log to stdout.
    """
    os.makedirs(snapshot_dir, exist_ok=True)
    log_path = os.path.join(snapshot_dir, log_filename)

    logging.basicConfig(
        filename=log_path,
        level=level,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S"
    )

    if console:
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
#---------------------------------------------------------------------------------#
def utilize_transformation(img, mask, transforms_op):
    """
    Applying transformations to img and mask with same random seed
    """
    
    state = torch.get_rng_state()
    mask  = transforms_op(mask)
    torch.set_rng_state(state)
    img   = transforms_op(img)
    
    return img, mask
    
#----------------------------------------------------------------------------------#