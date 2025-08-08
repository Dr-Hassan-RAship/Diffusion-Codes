#------------------------------------------------------------------------------#
# File name         : fom.py
# Purpose           : Figures of merit utilized in segmentation tasks.
#                     Picked from Github
#
# Editors           : Syed Muqeem Mahmood, Shujah Ur Rehman, Hassan Mohy-ud-Din
# Email             : 24100025@lums.edu.pk, 21060003@lums.edu.pk,
#                     hassan.mohyuddin@lums.edu.pk
#
# Last Date         : March 3, 2025
#------------------------------------------------------------------------------#

import torch, cv2, math

import numpy                        as np
import pandas                       as pd
import torch.nn.functional          as F

from PIL                            import Image
from torch                          import nn
from functools                      import wraps
from medpy                          import metric
from numpy                          import average, dot, linalg
from sklearn.metrics                import roc_auc_score, jaccard_score

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
def auc_on_batch(masks, pred):
    '''Computes the mean Area Under ROC Curve over a batch during training'''

    aucs                = []
    for i in range(pred.shape[1]):
        prediction      = pred[i][0].cpu().detach().numpy()
        # print("www",np.max(prediction), np.min(prediction))

        mask            = masks[i].cpu().detach().numpy()
        # print("rrr",np.max(mask), np.min(mask))
        aucs.append(roc_auc_score(mask.reshape(-1), prediction.reshape(-1)))
    return np.mean(aucs)

#------------------------------------------------------------------------------#
def iou_on_batch(pred, masks):
    ious                = []
    for i in range(pred.shape[0]):
        pred_tmp        = pred[i].cpu().detach().numpy()
        mask_tmp        = masks[i].cpu().detach().numpy()

        pred_tmp[pred_tmp >= 0.5]   = 1
        pred_tmp[pred_tmp < 0.5]    = 0
        mask_tmp[mask_tmp > 0]      = 1
        mask_tmp[mask_tmp <= 0]     = 0

        ious.append(jaccard_score(mask_tmp.reshape(-1), pred_tmp.reshape(-1)))
    return np.mean(ious)

#------------------------------------------------------------------------------#
def dice_coef(y_pred, y_true):
    smooth              = 1e-5

    # Flatten the tensors
    y_true_f            = y_true.contiguous().view(-1)
    y_pred_f            = y_pred.contiguous().view(-1)

    # Calculate intersection and sums
    intersection        = torch.sum(y_true_f * y_pred_f)
    y_true_sum          = torch.sum(y_true_f)
    y_pred_sum          = torch.sum(y_pred_f)

    # Compute Dice coefficient
    dice                = (2. * intersection + smooth) / (y_true_sum + y_pred_sum + smooth)
    return dice.item()  # Convert to Python scalar if needed

#------------------------------------------------------------------------------#
def save_on_batch(images1, masks, pred, names, vis_path):
    '''Computes the mean Area Under ROC Curve over a batch during training'''

    for i in range(pred.shape[0]):
        pred_tmp        = pred[i][0].cpu().detach().numpy()
        mask_tmp        = masks[i].cpu().detach().numpy()

        pred_tmp[pred_tmp >= 0.5]   = 255
        pred_tmp[pred_tmp < 0.5]    = 0
        mask_tmp[mask_tmp > 0]      = 255
        mask_tmp[mask_tmp <= 0]     = 0

        cv2.imwrite(vis_path + names[i][:-4] + "_pred.jpg", pred_tmp)
        cv2.imwrite(vis_path + names[i][:-4] + "_gt.jpg", mask_tmp)

#------------------------------------------------------------------------------#
# Unification images processing
def get_thum(image, size=(224, 224), greyscale=False):
    image               = image.resize(size, Image.ANTIALIAS)
    if greyscale: image = image.convert('L')
    return image

#------------------------------------------------------------------------------#
# Calculate the cosine distance between pictures
def img_similarity_vectors_via_numpy(image1, image2):

    image1              = get_thum(image1)
    image2              = get_thum(image2)
    images              = [image1, image2]

    vectors, norms      = [], []
    for image in images:
        vector          = []

        for pixel_turple in image.getdata(): vector.append(average(pixel_turple))
        vectors.append(vector)
        norms.append(linalg.norm(vector, 2))

    a, b                = vectors
    a_norm, b_norm      = norms
    res                 = dot(a / a_norm, b / b_norm)
    return res

#------------------------------------------------------------------------------#
def compute_iou(pred, label):
    pred[pred >= 0.5]   = 1
    pred[pred < 0.5]    = 0
    label[label > 0]    = 1
    label[label <= 0]   = 0

    return jaccard_score(label.reshape(-1), pred.reshape(-1))

#------------------------------------------------------------------------------#
def test_single_slice(pred, label):
    """
    Calculate metrics in a single slice
    This function is exclusively used only in 'inference.py'
    """

    if (pred.sum() > 0) and (label.sum() > 0):                 # Main calculation
        dice    = metric.binary.dc(pred, label)
        IoU     = compute_iou(label, pred)

    elif (pred.sum() == 0) or (label.sum() == 0):              # Safeguard no 1
        dice    = 0
        IoU     = 0

    elif (pred.sum() == 0) and (label.sum() == 0):             # Safeguard no 2
        dice    = 1
        IoU     = 1

    return (dice, IoU)

#------------------------------------------------------------------------------#
# https://github.com/mmany/pytorch-GDL
class GradientDifferenceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(GradientDifferenceLoss, self).__init__()

    def forward(self, inputs, targets, by_batch):
        gradient_diff_i = (torch.diff(inputs, dim=-1) - torch.diff(targets, dim=-1)).pow(2)
        gradient_diff_j = (torch.diff(inputs, dim=-2) - torch.diff(targets, dim=-2)).pow(2)
        gradient_diff   = torch.sum(gradient_diff_i) + torch.sum(gradient_diff_j)

        if by_batch == True:
            norm_div    = inputs.shape[0] * inputs.shape[1]
            loss_gdl    = gradient_diff/norm_div
        else:
            loss_gdl    = gradient_diff/inputs.numel()

        return loss_gdl

#------------------------------------------------------------------------------#
# https://github.com/mmany/pytorch-GDL
class MSE_L1_and_GDL(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(MSE_L1_and_GDL, self).__init__()

    def forward(self, inputs, targets, lambda_mse_l1, lambda_gdl, by_batch, type='l2'):

        if type == 'l2':
            error           = torch.sum((inputs - targets).pow(2))
        elif type == 'l1':
            l1_error        = torch.nn.L1Loss(reduction='sum')
            error           = l1_error(inputs, targets)
            #error           = torch.sum(torch.abs(inputs - targets))

        if by_batch == True:
            norm_div        = inputs.shape[0] * inputs.shape[1]
            error           = error/norm_div
        else:
            error           = error/inputs.numel()

        gradient_diff_i     = (torch.diff(inputs, dim=-1) - torch.diff(targets, dim=-1)).pow(2)
        gradient_diff_j     = (torch.diff(inputs, dim=-2) - torch.diff(targets, dim=-2)).pow(2)
        gradient_diff       = torch.sum(gradient_diff_i) + torch.sum(gradient_diff_j)

        if by_batch == True:
            norm_div        = inputs.shape[0] * inputs.shape[1]
            loss_gdl        = gradient_diff/norm_div
        else:
            loss_gdl        = gradient_diff/inputs.numel()

        loss                = (lambda_mse_l1*error) + (lambda_gdl*loss_gdl)

        return loss, error, loss_gdl

#------------------------------------------------------------------------------#
# github.com/chongyangma/cs231n/blob/master/assignments/assignment3/style_transfer_pytorch.py
def tv_loss(Imat, tv_weight=0.01):
    """
    Compute total variation loss.

    Inputs:
    - Imat: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.

    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for Imat weighted by tv_weight.
    """
    w_variance  = torch.sum(torch.pow(Imat[:,:,:,:-1] - Imat[:,:,:,1:], 2))
    h_variance  = torch.sum(torch.pow(Imat[:,:,:-1,:] - Imat[:,:,1:,:], 2))
    loss        = tv_weight * (h_variance + w_variance)
    return loss

#------------------------------------------------------------------------------#
def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

#------------------------------------------------------------------------------#
def barlow_loss(c, lam=0.0051):
    "Compute loss for Barlow twins."
    on_diag             = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag            = off_diagonal(c).pow_(2).sum()
    loss                = on_diag + lam * off_diag
    return loss

#------------------------------------------------------------------------------#
