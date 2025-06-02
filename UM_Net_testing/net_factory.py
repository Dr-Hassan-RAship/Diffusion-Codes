#------------------------------------------------------------------------------#
# File name         : net_factory.py
# Purpose           : Several encoder-decoder architectures for segmentation
#
# Editors           : Syed Muqeem Mahmood, Shujah Ur Rehman, Hassan Mohy-ud-Din
# Email             : 24100025@lums.edu.pk, 21060003@lums.edu.pk,
#                     hassan.mohyuddin@lums.edu.pk
#
# Original Source   : picked from github (semi-supervised segmentation methods)
#                     and Huisi Wu et al. TMI, 2024.
#
# Last Date         : March 13, 2025
#------------------------------------------------------------------------------#

from networks.Res_UNet          import ResUNet
from networks.small_unet        import UNet             as smUNet
from networks.small_unet        import UNet_DTC         as smDTC
from networks.LViT_UNet         import UNet             as LViT_UNet
from networks.barlow_model      import BarlowTwins      as Barlow

#------------------------------------------------------------------------------#
def net_factory(net_type, in_chns=3, class_num=2, pretrain=False, concatF=False, init=None):

    if net_type == "small_unet":
        net = smUNet(n_channels=in_chns, n_classes=class_num, initialization=init)

    elif net_type == "res_unet":
        net = ResUNet(n_channels=in_chns, n_classes=class_num, pretrain=pretrain, concatF=concatF)

    elif net_type == "lvit_unet":
        net = LViT_UNet(n_channels=in_chns, n_classes=class_num)

    elif net_type == "barlow_enc":
        net = Barlow(n_channels=in_chns, n_classes=class_num, pretrain=pretrain)

    return net

#------------------------------------------------------------------------------#
