import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


class SELayer(nn.Module):
    """
    (SE) layer implementation.
    This layer applies channel-wise attention to the input feature maps by
    learning a set of weights for each channel. It uses global average pooling
    to capture global spatial information, followed by a two-layer fully connected
    network to compute the attention weights.
    Args:
        channel (int): The number of input channels.
        reduction (int, optional): The reduction ratio for the intermediate
            fully connected layer. Default is 16.
    Methods:
        forward(x):
            Applies the SE operation to the input tensor.
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
            Returns:
                torch.Tensor: Output tensor with the same shape as the input, where
                each channel is scaled by its corresponding attention weight.
    """
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        return x * y.expand_as(x)


class NonLocalBlock(nn.Module):
    """
    A PyTorch implementation of the Non-Local Block for capturing long-range dependencies in feature maps.
    Args:
        in_channels (int): Number of input channels.
        inter_channels (int, optional): Number of intermediate channels. If None, it is set to `in_channels // 2`.
        sub_sample (bool, optional): If True, applies max pooling to reduce spatial dimensions. Default is True.
        bn_layer (bool, optional): If True, includes a BatchNorm2d layer after the final convolution. Default is True.
    Attributes:
        sub_sample (bool): Indicates whether max pooling is applied.
        in_channels (int): Number of input channels.
        inter_channels (int): Number of intermediate channels.
        g (nn.Module): Convolutional layer or sequential module for generating g(x).
        W (nn.Module): Sequential module or convolutional layer for generating W(y).
        theta (nn.Conv2d): Convolutional layer for generating theta(x).
        phi (nn.Module): Convolutional layer or sequential module for generating phi(x).
    Methods:
        forward(x):
            Computes the output tensor by capturing long-range dependencies in the input tensor.
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            Returns:
                torch.Tensor: Output tensor of the same shape as the input tensor.
    """
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NonLocalBlock, self).__init__()

        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                          kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                               kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, nn.MaxPool2d(kernel_size=(2, 2)))
            self.phi = nn.Sequential(self.phi, nn.MaxPool2d(kernel_size=(2, 2)))

    def forward(self, x):
        
        # Note: h * w is replaced with (h // 2) * (w // 2) if self.sub_sample == True
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1) # (b, h * w, inter_channels)

        # Query - Features
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1) # (b, h * w, inter_channels)
        
        # Key - Features
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1) # (b, inter_channels, h * w)
        
        # Attention Map - similarity between every pair of positions in the feature map 
        f       = torch.matmul(theta_x, phi_x) # (b, h * w, inter_channels) @ (b, inter_channels, h * w) --> (b, h * w, h * w)
        # Normalize Attention Weights along last dimension
        f_div_C = F.softmax(f, dim=-1)

        # Feature Aggregation 
        y = torch.matmul(f_div_C, g_x) # (b, h * w, h * w) @ (b, h * w, inter_channels) --> (b, h * w, inter_channels)
        y = y.permute(0, 2, 1).contiguous() # (b, inter_channels, h * w)
        y = y.view(batch_size, self.inter_channels, *x.size()[2:]) # (b, inter_channels, h * w) if self.sub_sample = True
        
        W_y = self.W(y) # (b, c, h , w)
        z = W_y + x # skip connection

        return z


class HPPF(nn.Module):
    """
    Hierarchical Pooling and Feature Fusion (HPPF) module.
    This module performs hierarchical pooling and feature fusion on input tensors 
    to generate enhanced feature representations. It combines features from multiple 
    input tensors, applies adaptive pooling, attention mechanisms, and convolutional 
    operations to produce the output.
    Args:
        in_channels (int): Number of input channels for the module.
    Methods:
        forward(x1, x2, x3):
            Forward pass of the HPPF module.
            Args:
                x1 (torch.Tensor): First input tensor with shape (batch_size, channels, height, width).
                x2 (torch.Tensor): Second input tensor with shape (batch_size, channels, height, width).
                x3 (torch.Tensor): Third input tensor with shape (batch_size, channels, height, width).
            Returns:
                torch.Tensor: Output tensor after hierarchical pooling, attention, and feature fusion.
    """
    def __init__(self, in_channels):
        super(HPPF, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, in_channels // 16, 1, 1), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, in_channels // 64, 1, 1), nn.ReLU(inplace=True))
        self.avg   = nn.AdaptiveAvgPool2d(1)
        self.max1  = nn.AdaptiveMaxPool2d(4)
        self.max2  = nn.AdaptiveMaxPool2d(8)
        self.mlp   = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, kernel_size=1),
            nn.Sigmoid())
        self.feat_conv = nn.Sequential(nn.Conv2d(in_channels, in_channels // 3, 3, 1, 1),
                                       nn.BatchNorm2d(in_channels // 3),
                                       nn.ReLU(inplace=True))

    def forward(self, x1, x2, x3):
        # Interpolate x2 and x3 to match the size of x1 at spatial dimensions (h, w)
        x2 = F.interpolate(x2, size=x1.size()[2:], mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, size=x1.size()[2:], mode='bilinear', align_corners=True)
        
        # Concatenate the input tensors along the channel dimension
        feat = torch.cat((x1, x2, x3), 1) # (feat: [b, 3 * c, h, w])

        b, c, h, w = feat.size()
        
        # Apply hierarchical pooling and feature fusion
        y1 = self.avg(feat) # (y1: [b, c, 1, 1])
        
        y2 = self.conv1(self.max1(feat)) # (y2: [b, 3 * c, 4, 4] --> [b, 3 * c, 1, 1] after reshape)
        
        y3 = self.conv2(self.max2(feat)) # (y3: [b, 3 * c, 8, 8] --> [b, 3 * c, 1, 1] after reshape)
        
        # Reshape y1, y2, and y3 to match the expected dimensions for attention computation
        y2 = y2.reshape(b, c, 1, 1)
        y3 = y3.reshape(b, c, 1, 1)
        z  = (y1 + y2 + y3) // 3 # (z: [b, 3 * c, 1, 1])
        
        attention = self.mlp(z)
        output1 = attention * feat
        output2 = self.feat_conv(output1)

        return output2


class ALGM(nn.Module):
    """
    ALGM (Adaptive Local-Global Module) is a PyTorch module designed for feature extraction and 
    context aggregation using a combination of local and global operations. It supports cascading 
    outputs and flexible output configurations.
    Args:
        mid_ch (int): The number of middle channels used for feature extraction.
        pool_size (tuple): A tuple specifying the padding/dilation sizes for convolutional layers.
        out_list (tuple): A tuple specifying the output channel sizes for each output layer.
        cascade (bool): If True, enables cascading outputs with additional processing.
    Attributes:
        cascade (bool): Indicates whether cascading outputs are enabled.
        out_list (tuple): Stores the output channel sizes for each output layer.
        LGmodule (nn.ModuleList): A list of local-global modules for feature extraction.
        LGoutmodel (nn.ModuleList): A list of output layers for generating final outputs.
        conv1 (nn.Sequential): Initial convolutional layer for feature extraction.
        conv2 (nn.Sequential): Convolutional layer used for cascading output processing.
    Methods:
        forward(x, y=None):
            Performs forward propagation through the module.
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, mid_ch, height, width).
                y (list of torch.Tensor, optional): List of tensors for cascading outputs. 
                    Each tensor should match the spatial dimensions of the input tensor.
            Returns:
                list of torch.Tensor: A list of output tensors, each corresponding to an 
                output layer defined in `out_list`.
    """
    def __init__(self, mid_ch, pool_size=(), out_list=(), cascade=False):
        super(ALGM, self).__init__()
        in_channels = mid_ch // 4
        self.cascade = cascade
        self.out_list = out_list
        size = [1, 2, 3]
        LGlist = []
        LGoutlist = []

        LGlist.append(NonLocalBlock(in_channels))
        for i in size:
            LGlist.append(nn.Sequential(
                nn.Conv2d(in_channels*i, in_channels, 3, stride=1, padding=pool_size[i-1], dilation=pool_size[i-1]),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)))
        self.LGmodule = nn.ModuleList(LGlist)

        for j in range(len(self.out_list)):
            LGoutlist.append(nn.Sequential(SELayer(in_channels*4),
                                           nn.Conv2d(in_channels * 4, self.out_list[j], 3, 1, 1),
                                           nn.BatchNorm2d(self.out_list[j]),
                                           nn.ReLU(inplace=True)))
        self.LGoutmodel = nn.ModuleList(LGoutlist)
        self.conv1 = nn.Sequential(nn.Conv2d(mid_ch, in_channels, 3, 1, 1),
                                   nn.BatchNorm2d(in_channels),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

    def forward(self, x, y=None):
        xsize = x.size()[2:]
        x = self.conv1(x)
        lg_context = []
        for i in range(2):
            lg_context.append(self.LGmodule[i](x))
        x1 = torch.cat((x, lg_context[1]), 1)
        lg_context.append(self.LGmodule[2](x1))
        x2 = torch.cat((x, lg_context[1], lg_context[2]), 1)
        lg_context.append(self.LGmodule[3](x2))
        lg_context = torch.cat(lg_context, dim=1)

        output = []
        for i in range(len(self.LGoutmodel)):
            out = self.LGoutmodel[i](lg_context)
            if self.cascade is True and y is not None:
                m = self.conv2(abs(F.interpolate(y[i], xsize, mode='bilinear', align_corners=True) - out))
                out = out + m
            output.append(out)

        return output


class CBAM(nn.Module):
    """
    CBAM (Convolutional Block Attention Module) implementation.
    CBAM is a lightweight attention module that can be integrated into 
    convolutional neural networks to improve feature representation by 
    focusing on important regions in both channel and spatial dimensions.
    Args:
        channel (int): Number of input channels.
        reduction (int, optional): Reduction ratio for the channel attention module. 
                                   Default is 16.
    Methods:
        forward(x):
            Applies the CBAM attention mechanism to the input tensor.
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, channel, height, width).
            Returns:
                torch.Tensor: Output tensor after applying channel and spatial attention.
    """
    def __init__(self, channel, reduction=16):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=False))
        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        c_avg = self.mlp(self.avg_pool(x))
        c_max = self.mlp(self.max_pool(x))
        c_out = self.sigmoid(c_avg + c_max)
        y1 = c_out * x

        s_avg = torch.mean(y1, dim=1, keepdim=True)
        s_max, _ = torch.max(y1, dim=1, keepdim=True)
        s_out = torch.cat((s_max, s_avg), 1)
        s_out = self.sigmoid(self.conv(s_out))
        output = s_out * y1

        return output


class RCG(nn.Module):
    """
    Residual Contextual Guidance (RCG) Module.
    This module processes input feature maps and edge maps to refine features 
    using attention mechanisms and residual connections.
    Attributes:
        conv1 (nn.Sequential): A sequential layer consisting of a 2D convolution, 
            batch normalization, and ReLU activation for feature extraction.
        mlp (nn.Sequential): A sequential layer consisting of a 1x1 convolution 
            followed by a sigmoid activation for generating attention weights.
    Methods:
        forward(pre, edge, f):
            Args:
                pre (torch.Tensor): Pre-computed attention map (shape: [batch_size, 1, height, width]).
                edge (torch.Tensor): Edge map (shape: [batch_size, 1, height, width]).
                f (torch.Tensor): Input feature map (shape: [batch_size, channels, height, width]).
            Returns:
                torch.Tensor: Refined feature map after applying attention and residual connections 
                (shape: [batch_size, channels, height, width]).
    """
    def __init__(self):
        super(RCG, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1),
                                 nn.Sigmoid())

    def forward(self, pre, edge, f):
        f_att = torch.sigmoid(pre)
        r_att = -1 * f_att + 1
        r = r_att * f

        edge1 = F.interpolate(edge, size=f.size()[2:], mode='bilinear', align_corners=True)
        x1 = torch.cat((edge1, r), 1)
        x2 = self.conv1(x1)
        x3 = self.mlp(x2)
        x4 = x3 * x2
        output = x4 + f

        return output

