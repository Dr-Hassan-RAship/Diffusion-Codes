#------------------------------------------------------------------------------#
#
# File name         : Res-UNet.py
# Purpose           : UNet architecture with a ResNet encoder and UNet decoder
#
# Editors           : Shujah Ur Rehman, Hassan Mohy-ud-Din
# Email             : 21060003@lums.edu.pk, hassan.mohyuddin@lums.edu.pk
#
# Original Source   : github.com/zhangbaiming/FedSemiSeg/.../unet34.py
#
# Last Date         : October 28, 2024
#
#------------------------------------------------------------------------------#

import torch
import torch.nn                         as nn
from torchvision.models                 import resnet34, ResNet34_Weights
from networks.fmodule                   import FModule

#------------------------------------------------------------------------------#
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

#------------------------------------------------------------------------------#
class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='bilinear', align_corners=False):
        super(Upsample, self).__init__()

        self.align_corners  = align_corners
        self.mode           = mode
        self.scale_factor   = scale_factor
        self.size           = size

    def forward(self, x):
        return nn.functional.interpolate(x,
                                         size           = self.size,
                                         scale_factor   = self.scale_factor,
                                         mode           = self.mode,
                                         align_corners  = self.align_corners)

#------------------------------------------------------------------------------#
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters, use_transpose=True):
        super(DecoderBlock, self).__init__()

        if use_transpose:
            self.up_op  = nn.ConvTranspose2d(in_channels // 4, in_channels // 4,
                                             3, stride=2, padding=1,
                                             output_padding=1)
        else:
            self.up_op  = Upsample(scale_factor=2, align_corners=True)

        self.conv1      = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1      = nn.BatchNorm2d(in_channels // 4)
        self.relu1      = nn.ReLU(inplace=True)

        self.norm2      = nn.BatchNorm2d(in_channels // 4)
        self.relu2      = nn.ReLU(inplace=True)

        self.conv3      = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3      = nn.BatchNorm2d(n_filters)
        self.relu3      = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.up_op(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

#------------------------------------------------------------------------------#
class DecoderBlockConcat(nn.Module):
    def __init__(self, in_channels, out_channels, use_transpose=True):
        super(DecoderBlockConcat, self).__init__()

        if use_transpose:
            self.up         = nn.ConvTranspose2d(in_channels, out_channels,
                                                kernel_size=2, stride=2)
        else:
            self.up         = Upsample(scale_factor=2, align_corners=True)

        self.conv_bn_relu   = nn.Sequential(nn.Conv2d(2 * out_channels, out_channels,
                                                      kernel_size=3, padding=1),
                                            nn.BatchNorm2d(out_channels),
                                            nn.ReLU(inplace=True))

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x  = torch.cat((x1, x2), dim=1)
        x  = self.conv_bn_relu(x)
        return x

#------------------------------------------------------------------------------#
class ResUNet(FModule):
    def __init__(self, n_channels=3, n_classes=2, pretrain=False, concatF=False):
        super(ResUNet, self).__init__()

        self.concatF        = concatF
        filters             = [64, 128, 256, 512]

        if pretrain == True:
            resnet          = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        else:
            resnet          = resnet34()

        self.firstconv      = resnet.conv1
        self.firstbn        = resnet.bn1
        self.firstrelu      = resnet.relu
        self.firstmaxpool   = resnet.maxpool
        self.encoder1       = resnet.layer1
        self.encoder2       = resnet.layer2
        self.encoder3       = resnet.layer3
        self.encoder4       = resnet.layer4

        if self.concatF ==  False:
            self.decoder4   = DecoderBlock(filters[3], filters[2])
            self.decoder3   = DecoderBlock(filters[2], filters[1])
            self.decoder2   = DecoderBlock(filters[1], filters[0])
        else:
            self.decoder4   = DecoderBlockConcat(filters[3], filters[2])
            self.decoder3   = DecoderBlockConcat(filters[2], filters[1])
            self.decoder2   = DecoderBlockConcat(filters[1], filters[0])

        self.decoder1       = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1   = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1     = nn.ReLU(inplace=True)
        self.finalconv2     = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2     = nn.ReLU(inplace=True)
        self.finalconv3     = nn.Conv2d(32, n_classes, 3, padding=1)

    def forward(self, x):
        x       = self.firstconv(x)
        x       = self.firstbn(x)
        x       = self.firstrelu(x)
        x       = self.firstmaxpool(x)

        e1      = self.encoder1(x)
        e2      = self.encoder2(e1)
        e3      = self.encoder3(e2)
        e4      = self.encoder4(e3)

        # Decoder
        if self.concatF == False:
            d4  = self.decoder4(e4) + e3
            d3  = self.decoder3(d4) + e2
            d2  = self.decoder2(d3) + e1
        else:
            d4  = self.decoder4(e4, e3)
            d3  = self.decoder3(d4, e2)
            d2  = self.decoder2(d3, e1)

        d1      = self.decoder1(d2)

        out     = self.finaldeconv1(d1)
        out     = self.finalrelu1(out)
        out     = self.finalconv2(out)
        out     = self.finalrelu2(out)
        out     = self.finalconv3(out)

        return out#, [e4, d4, d3, d2, d1]

#------------------------------------------------------------------------------#
class ModifiedResUNet(nn.Module):
    def __init__(self, original_model=None, n_classes=2, concatF=False, ft_option=None):
        super(ModifiedResUNet, self).__init__()

        self.n_classes      = n_classes
        self.concatF        = concatF
        filters             = [64, 128, 256, 512]

        self.firstconv      = original_model.firstconv
        self.firstbn        = original_model.firstbn
        self.firstrelu      = original_model.firstrelu
        self.firstmaxpool   = original_model.firstmaxpool

        self.encoder1       = original_model.encoder1
        self.encoder2       = original_model.encoder2
        self.encoder3       = original_model.encoder3
        self.encoder4       = original_model.encoder4

        if ft_option == 'decoder-only':
            if self.concatF ==  False:
                self.decoder4   = DecoderBlock(filters[3], filters[2])
                self.decoder3   = DecoderBlock(filters[2], filters[1])
                self.decoder2   = DecoderBlock(filters[1], filters[0])
            else:
                self.decoder4   = DecoderBlockConcat(filters[3], filters[2])
                self.decoder3   = DecoderBlockConcat(filters[2], filters[1])
                self.decoder2   = DecoderBlockConcat(filters[1], filters[0])

            self.decoder1       = DecoderBlock(filters[0], filters[0])

            self.finaldeconv1   = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
            self.finalrelu1     = nn.ReLU(inplace=True)
            self.finalconv2     = nn.Conv2d(32, 32, 3, padding=1)
            self.finalrelu2     = nn.ReLU(inplace=True)
            self.finalconv3     = nn.Conv2d(32, n_classes, 3, padding=1)

        elif ft_option == 'last-layer':
            self.decoder4       = original_model.decoder4
            self.decoder3       = original_model.decoder3
            self.decoder2       = original_model.decoder2
            self.decoder1       = original_model.decoder1

            self.finaldeconv1   = original_model.finaldeconv1
            self.finalrelu1     = original_model.finalrelu1
            self.finalconv2     = original_model.finalconv2
            self.finalrelu2     = original_model.finalrelu2

            self.finalconv3     = nn.Conv2d(32, self.n_classes, 3, padding=1)

        elif ft_option == 'complete':
            self.decoder4       = original_model.decoder4
            self.decoder3       = original_model.decoder3
            self.decoder2       = original_model.decoder2
            self.decoder1       = original_model.decoder1

            self.finaldeconv1   = original_model.finaldeconv1
            self.finalrelu1     = original_model.finalrelu1
            self.finalconv2     = original_model.finalconv2
            self.finalrelu2     = original_model.finalrelu2
            self.finalconv3     = original_model.finalconv3

    def forward(self, x):
        x       = self.firstconv(x)
        x       = self.firstbn(x)
        x       = self.firstrelu(x)
        x       = self.firstmaxpool(x)

        # Encoder
        e1      = self.encoder1(x)
        e2      = self.encoder2(e1)
        e3      = self.encoder3(e2)
        e4      = self.encoder4(e3)

        # Decoder
        if self.concatF == False:
            d4  = self.decoder4(e4) + e3
            d3  = self.decoder3(d4) + e2
            d2  = self.decoder2(d3) + e1
        else:
            d4  = self.decoder4(e4, e3)
            d3  = self.decoder3(d4, e2)
            d2  = self.decoder2(d3, e1)

        d1      = self.decoder1(d2)

        out     = self.finaldeconv1(d1)
        out     = self.finalrelu1(out)
        out     = self.finalconv2(out)
        out     = self.finalrelu2(out)
        out     = self.finalconv3(out)

        return out

#------------------------------------------------------------------------------#
class ModifiedBarlowTwins(nn.Module):
    def __init__(self, original_model=None, n_classes=2, concatF=False, ft_option=None):
        super(ModifiedBarlowTwins, self).__init__()

        self.n_classes          = n_classes
        self.concatF            = concatF
        filters                 = [64, 128, 256, 512]
        barlow_seg              = original_model.backbone

        # Encoder
        self.firstlayer         = nn.Sequential(*list(barlow_seg.children())[:3])
        self.maxpool            = list(barlow_seg.children())[3]
        self.encoder1           = barlow_seg.layer1
        self.encoder2           = barlow_seg.layer2
        self.encoder3           = barlow_seg.layer3
        self.encoder4           = barlow_seg.layer4

        # Decoder
        if ft_option == 'decoder-only':
            if self.concatF ==  False:
                self.decoder4   = DecoderBlock(filters[3], filters[2])
                self.decoder3   = DecoderBlock(filters[2], filters[1])
                self.decoder2   = DecoderBlock(filters[1], filters[0])
            else:
                self.decoder4   = DecoderBlockConcat(filters[3], filters[2])
                self.decoder3   = DecoderBlockConcat(filters[2], filters[1])
                self.decoder2   = DecoderBlockConcat(filters[1], filters[0])

            self.decoder1       = DecoderBlock(filters[0], filters[0])

            self.finaldeconv1   = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
            self.finalrelu1     = nn.ReLU(inplace=True)
            self.finalconv2     = nn.Conv2d(32, 32, 3, padding=1)
            self.finalrelu2     = nn.ReLU(inplace=True)
            self.finalconv3     = nn.Conv2d(32, n_classes, 3, padding=1)

        elif ft_option == 'complete':
            self.decoder4       = barlow_seg.decoder4
            self.decoder3       = barlow_seg.decoder3
            self.decoder2       = barlow_seg.decoder2
            self.decoder1       = barlow_seg.decoder1
            self.finaldeconv1   = barlow_seg.finaldeconv1
            self.finalrelu1     = barlow_seg.finalrelu1
            self.finalconv2     = barlow_seg.finalconv2
            self.finalrelu2     = barlow_seg.finalrelu2
            self.finalconv3     = barlow_seg.finalconv3

    def forward(self, x):
        x       = self.firstlayer(x)                    # torch.Size([24, 64, 112, 112])
        x       = self.maxpool(x)                       # torch.Size([24, 64, 56, 56])

        # Encoder
        e1      = self.encoder1(x)                      # torch.Size([24, 64, 56, 56])
        e2      = self.encoder2(e1)                     # torch.Size([24, 128, 28, 28])
        e3      = self.encoder3(e2)                     # torch.Size([24, 256, 14, 14])
        e4      = self.encoder4(e3)                     # torch.Size([24, 512, 7, 7])

        # Decoder
        if self.concatF == False:
            d4  = self.decoder4(e4) + e3
            d3  = self.decoder3(d4) + e2
            d2  = self.decoder2(d3) + e1
        else:
            d4  = self.decoder4(e4, e3)
            d3  = self.decoder3(d4, e2)
            d2  = self.decoder2(d3, e1)

        d1      = self.decoder1(d2)

        out     = self.finaldeconv1(d1)
        out     = self.finalrelu1(out)
        out     = self.finalconv2(out)
        out     = self.finalrelu2(out)
        out     = self.finalconv3(out)

        return out

#------------------------------------------------------------------------------#
if __name__ == '__main__':
    # compute FLOPS & PARAMETERS

    #--------------------------------------------------------------------------#
    import torch, ipdb
    from torchinfo              import summary
    from pt_flops               import get_model_complexity_info
    #--------------------------------------------------------------------------#

    model               = ResUNet(n_channels    = 3,
                                  n_classes     = 2,
                                  pretrain      = False,
                                  concatF       = False).cuda()
    summary(model, input_size = (24, 3, 224, 224), depth = 4)

    with torch.cuda.device(0):
        macs, params    = get_model_complexity_info(model, (3, 224, 224),
                                                    as_strings=True,
                                                    print_per_layer_stat=True,
                                                    verbose=True)

        print('{:<30}  {:<8}'.format('Computational complexity: ' , macs))
        print('{:<30}  {:<8}'.format('Number of parameters: '     , params))

    ipdb.set_trace()

#------------------------------------------------------------------------------#
