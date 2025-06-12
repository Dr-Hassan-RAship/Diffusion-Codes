#------------------------------------------------------------------------------#
# File name         : fom.py
# Purpose           : Figures of merit utilized in segmentation tasks.
#                     Picked from Github
#
# Editors           : Syed Muqeem Mahmood, Shujah Ur Rehman, Hassan Mohy-ud-Din
# Email             : 24100025@lums.edu.pk, 21060003@lums.edu.pk,
#                     hassan.mohyuddin@lums.edu.pk
#
# Last Date         : March 17, 2025
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
