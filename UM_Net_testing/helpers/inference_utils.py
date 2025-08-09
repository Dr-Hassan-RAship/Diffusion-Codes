#------------------------------------------------------------------------------#
# File name         : inference_utils.py
# Purpose           : Contains functions used in inference.py
# Usage (command)   : from inference_utils.py import *
#
# Authors           : Shujah Ur Rehman, Hassan Mohy-ud-Din
# Email             : 21060003@lums.edu.pk, hassan.mohyuddin@lums.edu.pk
#
# Last Date         : March 17, 2025
#------------------------------------------------------------------------------#

import os, sys, math, torch

import numpy                        as np
import nibabel                      as nib

from medpy                          import metric
from helpers.train_utils            import compute_sdf
from skimage                        import transform    as sktform

# from small_unet                     import (UNet, MCNet2d_v1, MCNet2d_v2,
#                                             UNet_DTC, UNet_URPC, MCNet_MR, UNet_E1D2)

#------------------------------------------------------------------------------#
def net_factory(self):

    experiments = ['FS_100p', 'FS_10p', 'FS_20p', 'ACMT-Ent', 'ACMT-MU',
                   'ACMT-SErr', 'uamt', 'mean_teacher', 'dae_mt', 'slc']

    if any(substring in self.experiment_name for substring in experiments):
        net     = UNet(in_chns              = self.input_channels,
                       class_num            = self.num_classes,
                       initialization       = self.initialize).cuda()

    if 'small_slc' in self.experiment_name:
        net     = UNet_SLC(in_chns          = self.input_channels,
                           class_num        = self.num_classes,
                           initialization   = self.initialize).cuda()

    if 'urpc' in self.experiment_name:
        net     = UNet_URPC(in_chns         = self.input_channels,
                            class_num       = self.num_classes,
                            initialization  = self.initialize).cuda()

    if 'MCNet' == self.experiment_name:
        net     = MCNet2d_v1(in_chns        = self.input_channels,
                             class_num      = self.num_classes,
                             initialization = self.initialize).cuda()

    if 'MCNet_plus' == self.experiment_name:
        net     = MCNet2d_v2(in_chns        = self.input_channels,
                             class_num      = self.num_classes,
                             initialization = self.initialize).cuda()

    if 'mutual_reliable' == self.experiment_name:
        net     = MCNet_MR(in_chns          = self.input_channels,
                           class_num        = self.num_classes,
                           initialization   = self.initialize).cuda()

    if 'ACMT-PErr' in self.experiment_name:
        net     = UNet_PErr(in_chns         = self.input_channels,
                            class_num       = self.num_classes,
                            initialization  = self.initialize).cuda()

    if 'dtc' in self.experiment_name:
        net     = UNet_DTC(in_chns          = self.input_channels,
                           class_num        = self.num_classes,
                           initialization   = self.initialize).cuda()

    if 'sassnet' in self.experiment_name:
        net     = UNet_DTC(in_chns          = self.input_channels,
                           class_num        = self.num_classes,
                           initialization   = self.initialize).cuda()

    if 'AMBW' in self.experiment_name:
        net     = UNet_E1D2(in_chns                 = self.input_channels,
                            class_num               = self.num_classes,
                            initialization          = self.initialize,
                            dropout_inside_encoder  = False).cuda()

    if 'E1D2_FS' in self.experiment_name:
        net     = UNet_E1D2(in_chns                 = self.input_channels,
                        class_num                   = self.num_classes,
                        initialization              = self.initialize,
                        dropout_inside_encoder      = True).cuda()

    if 'Aux_Dec' in self.experiment_name:
        net     = UNet_E1D2(in_chns                 = self.input_channels,
                        class_num                   = self.num_classes,
                        initialization              = self.initialize,
                        dropout_inside_encoder      = True).cuda()
    return net

#------------------------------------------------------------------------------#
def rearrange_dims(sa_prob_map):
        ''' Rearrange dimensions from [z, nC, x, y] to [nC, x, y, z]'''

        SA_prob = []
        for j in range(sa_prob_map.shape[0]):
             SA_prob += [sa_prob_map[j,:,:,:]]

        SA_prob = np.stack(SA_prob, axis = 3)

        return SA_prob
#------------------------------------------------------------------------------#
def minmax_norm(image):
    '''
    Performs min-max normalization with NumPy.
    '''
    norm_image = (image - image.min()) / (image.max() - image.min())
    return norm_image

#------------------------------------------------------------------------------#
def read_nifti_image(image, islabel=False):
    '''
    Read nifti images and extract relevant fields.
    '''

    if isinstance(image, str): image = nib.load(image)

    if islabel:
        img         = image.get_fdata().astype(np.uint8)
    else:
        img         = image.get_fdata().astype(np.float32)

    i_affine        = image.affine
    i_header        = image.header
    i_pixsize       = (image.header['pixdim'][1],
                       image.header['pixdim'][2],
                       image.header['pixdim'][3])
    i_shape         = image.shape

    return i_affine, i_header, i_pixsize, i_shape, img

#------------------------------------------------------------------------------#
def test_single_volume(prediction, label, classes, pixsize, dim):
    """
    Calculate metrics for each class in a single volume
    This function is exclusively used only in 'inference.py'
    """

    metric_list = []
    for i in range(1, classes):
        pred    = (prediction == i)
        gt      = (label == i)

        if (pred.sum() > 0) and (gt.sum() > 0):                 # Main calculation
            dice    = metric.binary.dc(pred, gt)
            hd95    = metric.binary.hd95(pred, gt, voxelspacing=pixsize)
            assd    = metric.binary.assd(pred, gt, voxelspacing=pixsize)

        elif (pred.sum() == 0) or (gt.sum() == 0):              # Safeguard no 1
            dice    = 0
            hd95    = np.sqrt((dim[0] * pixsize[0])**2 +
                              (dim[1] * pixsize[1])**2 +
                              (dim[2] * pixsize[2])**2)
            assd    = hd95

        elif (pred.sum() == 0) and (gt.sum() == 0):             # Safeguard no 2
            dice    = 1
            hd95    = 0
            assd    = 0

        metric_list.append([dice, hd95, assd])

    return metric_list

#------------------------------------------------------------------------------#
def performance_in_training(score, target, classes, mode):
    """
    This function is used to compute segmentation peformance during training
    and validation.
    """
    metrics = []
    if mode == 'Training':
        for i in range(0, classes):
            # Extract the binary masks for the current class
            pred    = (score[:, i, :, :]).cpu().detach().numpy()
            label   = (target == i).squeeze(1).cpu().detach().numpy()

            # Flatten the ground truth and predicted masks
            label   = label.flatten()
            pred    = pred.flatten()

            # Calculate the intersection between the masks
            intersection = np.sum(label * pred)

            # Add a smoothing term to avoid division by zero
            smooth = 0.0001

            # Calculate the Dice coefficient
            dice = (2. * intersection + smooth) / (np.sum(label) + np.sum(pred) + smooth)

            metrics.append(dice)

    elif mode == 'Validation':
        for i in range(1, classes):
            pred                = score == i
            label               = target == i
            pred[pred > 0]      = 1
            label[label > 0]    = 1
            metrics.append(metric.binary.dc(pred, label))

    return metrics

#------------------------------------------------------------------------------#
