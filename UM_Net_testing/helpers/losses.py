#------------------------------------------------------------------------------#
# File name         : losses.py
# Purpose           : Contains various loss functions. Accumulated from various
#                     sources on Github.
#
# Authors           : Shujah Ur Rehman, Hassan Mohy-ud-Din
# Email             : 21060003@lums.edu.pk, hassan.mohyuddin@lums.edu.pk
#
# Last Date         : March 17, 2025
#------------------------------------------------------------------------------#

import torch, math
import numpy                    as np
import torch.nn                 as nn
from torch.autograd             import Variable
from torch.nn                   import functional as F

# ---------------------------------------------------------------------------- #
# The loss below is from Talha's E1D3 code -
# https://github.com/Clinical-and-Translational-Imaging-Lab/brats-e1d3/blob/main/e1d3/utils/losses.py
class DiceLoss_E1D3(nn.Module):
    """ - (dice score), includes softmax """

    def __init__(self, num_classes=4, reduction_dims=(0, 2, 3)):        # (B,C,H,W) so reduce dimensions across 'B','H','W'
        super(DiceLoss, self).__init__()
        self.num_classes        = num_classes
        self._REDUCTION_DIMS    = reduction_dims
        self._EPS               = 1e-7

    def forward(self, y_pred, y_true):
        """
        y_pred: (B, C, H, W), without softmax
        y_true: (B, H, W), dtype='long'
        """
        y_true      = F.one_hot(y_true, num_classes=self.num_classes).permute(0, 3, 1, 2)
        numerator   = 2.0 * torch.sum(y_true * y_pred, dim=self._REDUCTION_DIMS)
        denominator = torch.sum(y_true * y_true, dim=self._REDUCTION_DIMS) + torch.sum(y_pred * y_pred, dim=self._REDUCTION_DIMS)

        return 1 - torch.mean((numerator + self._EPS) / (denominator + self._EPS))

# ---------------------------------------------------------------------------- #
class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes  = n_classes

    def _one_hot_encoder(self, input_tensor):
        # Shape will be [24, 256, 256, 4]
        # Rearrange the dimensions to [24, 4, 256, 256]
        one_hot_labels  = F.one_hot(input_tensor, num_classes=self.n_classes)
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

        # One-hot encoding
        # target          = self._one_hot_encoder(target)
        print(f'target.shape: {target.shape}')
        assert inputs.size() == target.size(), 'predict & target shape do not match'

        class_wise_dice, loss = [], 0.0
        for i in range(0, self.n_classes):
            dice        = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss        += dice * weight[i]

        return loss / self.n_classes

# ---------------------------------------------------------------------------- #
class Block_DiceLoss(nn.Module):
    def __init__(self, n_classes, block_num):
        super(Block_DiceLoss, self).__init__()
        self.n_classes  = n_classes
        self.block_num  = block_num
        self.dice_loss  = DiceLoss_SLC(self.n_classes)

    # ------------------------------------------------------------------------ #
    def forward(self, inputs, target, weight=None, softmax=False):
        shape           = inputs.shape
        img_size        = shape[-1]
        div_num         = self.block_num
        block_size      = math.ceil(img_size / self.block_num)
        if target is not None:
            loss        = []
            for i in range(div_num):
                for j in range(div_num):
                    block_features  = inputs[:, :, i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
                    block_labels    = target[:, i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
                    tmp_loss        = self.dice_loss(block_features, block_labels.unsqueeze(1))
                    loss.append(tmp_loss)
            loss = torch.stack(loss).mean()
        return loss

# ---------------------------------------------------------------------------- #
def entropy_loss(p, C=2):
    """
    Calculates the entropy loss of a probability distribution.

    Args:
        p: Probability distribution tensor.
        C: Number of classes (default is 2).

    Returns:
        ent: Entropy loss.
    """
    # Calculate the negative sum of each element in p multiplied by the logarithm of the corresponding element in p
    y1 = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1) / torch.tensor(np.log(C)).cuda()

    # Calculate the mean of the entropy values for each instance
    ent = torch.mean(y1)

    # Return the entropy loss
    return ent

# ---------------------------------------------------------------------------- #
def softmax_dice_loss(input_logits, target_logits):
    """
    Takes softmax on both sides and returns MSE loss.

    Args:
        input_logits: Input logits tensor.
        target_logits: Target logits tensor.

    Returns:
        mean_dice: Mean dice loss.
    """
    # Ensure input and target logits have the same size
    assert input_logits.size() == target_logits.size()

    # Apply softmax to input and target logits along the second dimension
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    # Get the number of classes (dimension 1 of input logits)
    n = input_logits.shape[1]

    # Initialize dice loss
    dice = 0

    # Calculate dice loss for each class
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])

    # Calculate the mean dice loss
    mean_dice = dice / n

    return mean_dice

# ---------------------------------------------------------------------------- #
def entropy_loss_map(p, C=2):
    """
    Calculates the entropy loss for each instance in a probability distribution.

    Args:
        p: Probability distribution tensor.
        C: Number of classes (default is 2).

    Returns:
        ent: Entropy loss tensor.
    """
    # Calculate the negative sum of each element in p multiplied by the logarithm of the corresponding element in p
    ent = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True) / torch.tensor(np.log(C)).cuda()

    # Return the entropy loss tensor
    return ent

# ---------------------------------------------------------------------------- #
def softmax_mse_loss(input_logits, target_logits, sigmoid=False):
    """
    Takes softmax on both sides and returns MSE loss.

    Args:
        input_logits: Input logits tensor.
        target_logits: Target logits tensor.
        sigmoid: Flag indicating whether to apply sigmoid function to logits (default is False).

    Returns:
        mse_loss: Mean squared error loss tensor.
    """
    # Ensure input and target logits have the same size
    assert input_logits.size() == target_logits.size()

    # Apply softmax or sigmoid to input and target logits based on the sigmoid flag
    if sigmoid:
        input_softmax   = torch.sigmoid(input_logits)
        target_softmax  = torch.sigmoid(target_logits)
    else:
        input_softmax   = F.softmax(input_logits, dim=1)
        target_softmax  = F.softmax(target_logits, dim=1)

    # Calculate mean squared error loss
    mse_loss = (input_softmax - target_softmax)**2

    return mse_loss

# ---------------------------------------------------------------------------- #
def softmax_kl_loss(input_logits, target_logits, sigmoid=False):
    """
    Takes softmax on both sides and returns KL divergence loss.

    Args:
        input_logits: Input logits tensor.
        target_logits: Target logits tensor.
        sigmoid: Flag indicating whether to apply sigmoid function to logits (default is False).

    Returns:
        kl_div: KL divergence loss tensor.
    """
    # Ensure input and target logits have the same size
    assert input_logits.size() == target_logits.size()

    # Apply softmax or sigmoid to input and target logits based on the sigmoid flag
    if sigmoid:
        input_log_softmax   = torch.log(torch.sigmoid(input_logits))
        target_softmax      = torch.sigmoid(target_logits)
    else:
        input_log_softmax   = F.log_softmax(input_logits, dim=1)
        target_softmax      = F.softmax(target_logits, dim=1)

    # Calculate KL divergence loss
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='mean')

    return kl_div

# ---------------------------------------------------------------------------- #
def symmetric_mse_loss(input1, input2):
    """
    Calculates the symmetric mean squared error (MSE) loss between two input tensors.

    Args:
        input1: First input tensor.
        input2: Second input tensor.

    Returns:
        mse_loss: Symmetric MSE loss tensor.
    """
    # Ensure input tensors have the same size
    assert input1.size() == input2.size()

    # Calculate the mean squared error loss
    mse_loss = torch.mean((input1 - input2)**2)

    return mse_loss

# ---------------------------------------------------------------------------- #
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        """
        Focal Loss module for handling imbalanced classification problems.

        Args:
            gamma: Focusing parameter that controls the degree of focusing (default is 2).
            alpha: Class weight parameter. It can be a single value, a list of values, or None (default is None).
            size_average: Boolean flag indicating whether to return the mean of the loss over the batch (default is True).
        """
        super(FocalLoss, self).__init__()

        self.gamma = gamma
        self.alpha = alpha

        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])

        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)

        self.size_average = size_average

    def forward(self, input, target):
        """
        Compute the forward pass of the Focal Loss module.

        Args:
            input: Predicted logits or probabilities from the model.
            target: Ground truth labels.

        Returns:
            loss: Focal Loss tensor.
        """
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C

        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)

        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)

            at      = self.alpha.gather(0, target.data.view(-1))
            logpt   = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

# ---------------------------------------------------------------------------- #
def entropy_minimization(p):
    """
    Compute the entropy minimization loss.

    Args:
        p: Probability distribution tensor.

    Returns:
        ent: Entropy minimization loss value.
    """
    y1 = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1)
    ent = torch.mean(y1)

    return ent

# ---------------------------------------------------------------------------- #
def entropy_map(p):
    """
    Compute the entropy map.

    Args:
        p: Probability distribution tensor.

    Returns:
        ent_map: Entropy map tensor.
    """
    ent_map = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)
    return ent_map

# ---------------------------------------------------------------------------- #
def compute_kl_loss(p, q):
    """
    Compute the KL divergence loss between probability distributions p and q.

    Args:
        p: Probability distribution tensor.
        q: Probability distribution tensor.

    Returns:
        loss: KL divergence loss value.
    """
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss

# ---------------------------------------------------------------------------- #
