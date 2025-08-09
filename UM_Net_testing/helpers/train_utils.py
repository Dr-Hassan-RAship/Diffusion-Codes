#------------------------------------------------------------------------------#
# File name         : train_utils.py
# Purpose           : This file includes several functions utilized in
#                     SmS models. Some functions are picked from github.
#
# Authors           : Shujah Ur Rehman, Dr. Hassan Mohy-ud-Din
# Email             : 21060003@lums.edu.pk, hassan.mohyuddin@lums.edu.pk

# Last Date         : March 17, 2025
#------------------------------------------------------------------------------#

import os, sys, random, torch, math
import numpy                                                    as np
import torch.optim                                              as optim

from skimage                    import segmentation             as skimage_seg
from scipy.ndimage              import distance_transform_edt   as distance

from helpers.cosineWarm         import *
from helpers.LARS               import *
from torch.optim.lr_scheduler   import CosineAnnealingWarmRestarts

try:
    from scipy.special          import comb
except:
    from scipy.misc             import comb

#------------------------------------------------------------------------------#
def sigmoid_rampup(current, rampup_length):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase   = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

#------------------------------------------------------------------------------#
def get_current_consistency_weight(self, iter):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return self.consistency * sigmoid_rampup(iter, self.total_iterations)

#------------------------------------------------------------------------------#
def get_current_entropy_thresh(self, iter):
    H = (self.Ent_th + 0.25 * sigmoid_rampup(iter, self.total_iterations)) * np.log(self.num_classes)
    return H

#------------------------------------------------------------------------------#
def update_ema_variables(self, model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha       = min(1 - 1 / (global_step + 1), alpha)

    # The mul_() method multiplies the ema_param.data tensor by the alpha scalar,
    # and then adds the result to the 1-alpha scalar times the param.data tensor
    # using the add_() method. The mul_() and add_() methods modify the
    # ema_params.data tensor in-place, meaning that the changes are made directly
    # to the original object, and no new object is created. If we do
    # this -> (.add((1-alpha) * param.data)), then a new tensor is created.
    # .add_() is more memory efficient.
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

#------------------------------------------------------------------------------#
def sharpening(P, Temp=0.1):
    invTemp     = 1/Temp
    P_sharpen   = (P ** invTemp) / (P ** invTemp + (1 - P) ** invTemp)

    return P_sharpen

#------------------------------------------------------------------------------#
def entropy_map(p, C=2):
    entMap = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1) / torch.tensor(np.log(C)).cuda()
    return entMap

#------------------------------------------------------------------------------#
def optim_policy(self, model):
    if self.optimizer == "SGD":
        optimizer   = optim.SGD(model.parameters(),
                                lr              = self.eta_zero,
                                momentum        = 0.9,
                                nesterov        = False,
                                weight_decay    = 0.0001)

    elif self.optimizer == "Adam":
        optimizer   = optim.Adam(model.parameters(),
                                 lr             = self.eta_zero,
                                 betas          = (0.9, 0.999))

    elif self.optimizer == "AdamW":
        optimizer   = optim.AdamW(model.parameters(),
                                  lr            = self.eta_zero,
                                  betas         = (0.9, 0.999),
                                  weight_decay  = 5e-4)

    elif self.optimizer == "LARS":
        param_weights, param_biases = [], []
        for param in model.parameters():
            if param.ndim == 1:
                param_biases.append(param)
            else:
                param_weights.append(param)

        parameters  = [{'params': param_weights}, {'params': param_biases}]

        optimizer   = LARS(parameters,
                           lr                       = 0,
                           weight_decay             = 1e-6,
                           weight_decay_filter      = True,
                           lars_adaptation_filter   = True)

    return optimizer

#------------------------------------------------------------------------------#
def LR_schedule(self, optimizer, iter_count, current_LR_value):

    if self.LR_policy == "Constant":
        lr_ = self.eta_zero

    elif self.LR_policy == "PolyDecay":
        lr_ = self.eta_zero * (1 - iter_count / self.total_iterations) ** self.lr_decay_rate

    elif self.LR_policy == "PolyGrowth":
        lr_ = ((self.eta_N - self.eta_zero) * ((iter_count / self.total_iterations) ** self.lr_decay_rate)) + self.eta_zero

    elif self.LR_policy == "StepDecay":
        if (iter_count % self.iter_step == 0):
            lr_ = self.eta_zero * self.lr_decay_rate ** (iter_count // self.iter_step)
        else:
            lr_ = current_LR_value

    elif self.LR_policy == "CosineLR":
        #lr_ = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=eta_zero)
        lr_ = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=self.eta_min)

    else:
        print("The specified learning rate policy is not defined")

    for param_group in optimizer.param_groups: param_group['lr'] = lr_

    return param_group, optimizer, lr_

#------------------------------------------------------------------------------#
def compute_sdf(img_gt, out_shape, S):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0          ; x in segmentation boundary
             -inf|x-y|  ; x in segmentation
             +inf|x-y|  ; x out of segmentation
    normalize sdf to [-1, 1]
    """

    img_gt          = img_gt.astype(np.uint8)
    normalized_sdf  = np.zeros(out_shape)
    Nx, Ny          = img_gt.shape[1], img_gt.shape[2]

    for b in range(out_shape[0]):                       # batch size
        posmask     = img_gt[b].astype(np.bool)

        if S == 0:
            if np.all(posmask==1): normalized_sdf[b] = -1 * np.ones((Nx, Ny)); continue

        if S == 1 or S == 2 or S == 3:
            if np.all(posmask==0): normalized_sdf[b] = np.ones((Nx, Ny)); continue

        if posmask.any():
            negmask             = ~posmask
            posdis              = distance(posmask)
            negdis              = distance(negmask)
            boundary            = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
            sdf                 = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
            sdf[boundary==1]    = 0
            normalized_sdf[b]   = sdf

            # assert np.min(sdf) == -1.0, print(np.min(posdis), np.max(posdis), np.min(negdis), np.max(negdis))
            # assert np.max(sdf) ==  1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))

    return normalized_sdf

#------------------------------------------------------------------------------#
def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """
    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

#------------------------------------------------------------------------------#
def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.
       Control points should be a list of lists, or list of tuples
       such as [ [1,1],
                 [2,3],
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000
        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t                   = np.linspace(0.0, 1.0, nTimes)
    polynomial_array    = np.array([bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)])

    xvals   = np.dot(xPoints, polynomial_array)
    yvals   = np.dot(yPoints, polynomial_array)

    return xvals, yvals

#------------------------------------------------------------------------------#
def nonlinear_transformation(slices, flag=True):

        if flag: # increasing function applied to image
            random_num = random.random()
            if random_num <= 0.4:
                return (slices + 1) / 2

            if random_num > 0.4 and random_num <= 0.7:
                # 1st and 4th index are end-points, while 2nd and 3rd are control points,
                points_2            = [[-1, -1], [-0.5, 0.5], [0.5, -0.5], [1, 1]]
                xvals_2, yvals_2    = bezier_curve(points_2, nTimes=10000)
                xvals_2             = np.sort(xvals_2)
                yvals_2             = np.sort(yvals_2)
                nonlinear_slices_2  = np.interp(slices, xvals_2, yvals_2)
                return (nonlinear_slices_2 + 1) / 2

            if random_num > 0.7:
                points_4            = [[-1, -1], [-0.75, 0.75], [0.75, -0.75], [1, 1]]
                xvals_4, yvals_4    = bezier_curve(points_4, nTimes=10000)
                xvals_4             = np.sort(xvals_4)
                yvals_4             = np.sort(yvals_4)
                nonlinear_slices_4  = np.interp(slices, xvals_4, yvals_4)
                return (nonlinear_slices_4 + 1) / 2

        else: # decreasing function applied to image.
              # Note that only the x-coordinates are being sorted below.
              # this results in a decreasing function.
            random_num = random.random()

            if random_num <= 0.4:
                points_1            = [[-1, -1], [-1, -1], [1, 1], [1, 1]]
                xvals_1, yvals_1    = bezier_curve(points_1, nTimes=10000)
                xvals_1             = np.sort(xvals_1)
                nonlinear_slices_1  = np.interp(slices, xvals_1, yvals_1)
                nonlinear_slices_1[nonlinear_slices_1 == 1] = -1
                return (nonlinear_slices_1 + 1) / 2

            if random_num > 0.4 and random_num <= 0.7:
                points_3            = [[-1, -1], [-0.5, 0.5], [0.5, -0.5], [1, 1]]
                xvals_3, yvals_3    = bezier_curve(points_3, nTimes=10000)
                xvals_3             = np.sort(xvals_3)
                nonlinear_slices_3  = np.interp(slices, xvals_3, yvals_3)
                nonlinear_slices_3[nonlinear_slices_3 == 1] = -1
                return (nonlinear_slices_3 + 1) / 2

            if random_num > 0.7:
                points_5            = [[-1, -1], [-0.75, 0.75], [0.75, -0.75], [1, 1]]
                xvals_5, yvals_5    = bezier_curve(points_5, nTimes=10000)
                xvals_5             = np.sort(xvals_5)
                nonlinear_slices_5  = np.interp(slices, xvals_5, yvals_5)
                nonlinear_slices_5[nonlinear_slices_5 == 1] = -1
                return (nonlinear_slices_5 + 1) / 2

#------------------------------------------------------------------------------#
def pad_slices(tensor: torch.Tensor, max_slices: int=16) -> torch.Tensor:
    """
    Symmetrically pads a 3D tensor along the depth dimension (S) to ensure it
    has exactly 16 slices. This was required for 3D cardiac image segmentation.

    Ref: https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
    """

    H, W, S = tensor.shape

    if S < max_slices:
        nslices_to_pad  = max_slices - S
        padd_tuple      = ((nslices_to_pad // 2), (nslices_to_pad - (nslices_to_pad // 2)))
        #tensor          = torch.as_tensor(tensor)
        padded_tensor   = F.pad(tensor, padd_tuple, "constant", 0)
    else:
        # No padding needed if S >= max_slices
        padded_tensor = tensor

    return padded_tensor

#------------------------------------------------------------------------------#
def calculate_discrepancy(prob_map, gt):
    squared_error = torch.pow(prob_map - gt, 2)
    sum_error     = torch.sum(squared_error)
    var           = torch.log(1 + sum_error)
    omega         = torch.exp(-var)
    return var, omega
#------------------------------------------------------------------------------#
