# ================================================== lip_resnet_encoder.py =====
# File Name  : lip_resnet_encoder.py
# Purpose    : Pre-trained ResNet-LIP encoder that outputs a (B, C_latent, H/8, W/8)
#              representation compatible with the GMS latent diffusion pipeline.
# Usage      :
#     from lip_resnet_encoder import LIPResNetEncoder
#     enc = LIPResNetEncoder(backbone='resnet50',   # or 'resnet34'
#                            pretrained=True,
#                            latent_channels=4,     # to match VAE Z_I
#                            freeze=True)           # optional fine-tune flag
#     Z_lip = enc(img_rgb)                          # shape (B, 4, 28, 28) if input 224×224
#     Z      = torch.cat([Z_I, Z_lip], dim=1)       # (B, 8, 28, 28)
# ------------------------------------------------------------------------------
# Authors    : <your-name>
# Last Modif : 2025-07-03
# ==============================================================================

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------ constants #
COEFF              = 12.0            # soft-gate amplification
BOTTLENECK_WIDTH   = 128             # width of internal logit bottleneck
ALLOWED_BACKBONES  = ['resnet34', 'resnet50']   # LIP weights usually exist for 50/101

# ----------------------------------------------------------------- primitives #
def conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, 3, stride, padding = 1, bias=False)

def conv1x1(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, 1, stride, padding = 0, bias=False)


# ----------------------------------------------------------- LIP core op -----#
def lip2d(x, logit, kernel=3, stride=2, padding=1, margin=0.0):
    """
    Local Importance-based Pooling (progressive version).
    Args
    ----
    x     : (B, C, H, W) feature map
    logit : (B, 1, H, W) importance logits G(I)
    """
    w = logit.exp()                          # non-negative importance
    a = F.avg_pool2d(x * w, kernel, stride, padding)
    b = F.avg_pool2d(w,     kernel, stride, padding).add(margin)
    return a / b


# ---------------------------------------------------------- helper modules ---#
class SoftGate(nn.Module):
    """σ(x) * COEFF → keeps values in (0, COEFF)."""
    def forward(self, x):                     
        return torch.sigmoid(x) * COEFF


class BottleneckShared(nn.Module):
    """Shared feature extractor producing logit feature map G(I)."""
    def __init__(self, in_ch):
        super().__init__()
        rp = BOTTLENECK_WIDTH
        self.logit = nn.Sequential(OrderedDict([
            ('conv1', conv1x1(in_ch, rp)),
            ('bn1'  , nn.InstanceNorm2d(rp, affine=True)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', conv3x3(rp, rp)),
            ('bn2'  , nn.InstanceNorm2d(rp, affine=True)),
            ('relu2', nn.ReLU(inplace=True)),
        ]))

    def forward(self, x):
        return self.logit(x)


class BottleneckLIP(nn.Module):
    """
    LIP downsample branch used inside stride-2 residual blocks.
    Operates on the *main* path (x already reduced to `planes` channels).
    """
    def __init__(self, planes):
        super().__init__()
        rp = BOTTLENECK_WIDTH
        self.planes = planes
        self.postprocessing = nn.Sequential(OrderedDict([
            ('conv', conv1x1(rp, planes)),
            ('bn'  , nn.InstanceNorm2d(planes, affine=True)),
            ('gate', SoftGate()),
        ]))
        self.obj = BottleneckShared(in_ch = self.planes)
        # initialise conv to zero so we start as average pooling
        self.postprocessing[0].weight.data.fill_(0.0)

    # ---- hooks --------------------------------------------------------------#
    def forward(self, x):
        return lip2d(x, self.postprocessing(self.obj(x)))
    def forward_with_shared(self, x, shared):             # used by backbone
        return lip2d(x, self.postprocessing(shared))

class SimplifiedLIP(nn.Module):
    """3x3 stride-2 LIP layer used to replace the first max-pool."""
    def __init__(self, channels):
        # input is (1, 64, 112, 112)
        super().__init__()
        self.logit = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(channels, channels, kernel_size = 3, padding=1, bias=False)),
            ('bn'  , nn.InstanceNorm2d(channels, affine = True)),
            ('gate', SoftGate()),
        ]))
        nn.init.constant_(self.logit[0].weight, 0.0)

    def forward(self, x):
        return lip2d(x, self.logit(x))


# ------------------------ ResNet blocks -----------------------#
class BasicBlock(nn.Module):
    """Standard BasicBlock (ResNet34) – strides handled via LIP in downsample."""
    expansion = 1

    def __init__(self, in_ch, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(in_ch, planes, stride)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2   = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride     = stride
        nn.init.constant_(self.bn2.weight, 0.0)

    def forward(self, x):                                
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out =       self.bn2(self.conv2(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.relu(out + residual)
        return out


class Bottleneck(nn.Module):
    """Bottleneck block (ResNet50/101) with LIP-downsample when stride==2."""
    expansion = 4

    def __init__(self, in_ch, planes, stride=1, downsample=None):
        super().__init__()
        self.stride = stride
        # -------- --------------------  conv-branch ---------------------------------------------- #
        if stride == 2:                                  # use LIP
            self.bottleneck_shared = BottleneckShared(in_ch)
            self.conv1  = conv1x1(in_ch, planes)
            self.bn1    = nn.BatchNorm2d(planes)
            self.conv2  = nn.Sequential(
                BottleneckLIP(planes),
                conv1x1(planes, planes),
            )
            self.bn2    = nn.BatchNorm2d(planes)
            self.conv3  = conv1x1(planes, planes * self.expansion)
            self.bn3    = nn.BatchNorm2d(planes * self.expansion)

        else:                                            # standard
            self.conv1 = conv1x1(in_ch, planes)
            self.bn1   = nn.BatchNorm2d(planes)
            self.conv2 = conv3x3(planes, planes, stride)
            self.bn2   = nn.BatchNorm2d(planes)
            self.conv3 = conv1x1(planes, planes * self.expansion)
            self.bn3   = nn.BatchNorm2d(planes * self.expansion)

        self.relu       = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.bn3.weight.data.zero_()

    def forward(self, x):                                 
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))

        if self.stride == 2:                              # LIP pathway
            shared = self.bottleneck_shared(x)
            out    = self.conv2[0].forward_with_shared(out, shared)
            out    = self.conv2[1](out)
        else:
            out = self.conv2(out)

        out = self.relu(self.bn2(out))
        out =        self.bn3(self.conv3(out))

        if self.downsample is not None:
            if self.stride == 2:
                residual = self.downsample[0].forward_with_shared(x, shared)
                residual = self.downsample[1](residual)
                residual = self.downsample[2](residual)
            else:
                residual = self.downsample(x)

        return self.relu(out + residual)


# ------------------------------------------------------------- ResNet stem ---#
class _LIPResNetBackbone(nn.Module):              # internal helper
    def __init__(self, block, layers, num_classes = 1000):
        super().__init__()
        self.inplanes = 64
        # ---- stem ---------------------------------------------------------- #
        self.conv1   = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        # Downsampled already done by 2 --> (1, 64, 112, 112)
        self.bn1     = nn.BatchNorm2d(64)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = SimplifiedLIP(64)           # stride-2 LIP instead of max-pool

        # ---- stages -------------------------------------------------------- #
        self.layer1 = self._make_layer(block,  64, layers[0])      # stride 1
        self.layer2 = self._make_layer(block, 128, layers[1], 2)   # stride 2  -> 1/8
        self.layer3 = self._make_layer(block, 256, layers[2], 2)   # stride 2  -> 1/16
        self.layer4 = self._make_layer(block, 512, layers[3], 2)   # stride 2  -> 1/32

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(512 * block.expansion, num_classes)
        # weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        self.fc.weight.data.normal_(0, 0.01)
        self.fc.bias.data.zero_()

    # ----------------------------------------------------------------------- #
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 2:
                downsample = nn.Sequential(OrderedDict([
                    ('lip', BottleneckLIP(self.inplanes)),
                    ('conv', conv1x1(self.inplanes, planes * block.expansion)),
                    ('bn'  , nn.BatchNorm2d(planes * block.expansion)),
                ]))
            else:
                downsample = nn.Sequential(OrderedDict([
                    ('conv', conv1x1(self.inplanes, planes * block.expansion, stride)),
                    ('bn'  , nn.BatchNorm2d(planes * block.expansion)),
                ]))

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    # ----------------------------------------------------------------------- #
    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)      # 1/4
        x = self.layer2(x)      # 1/8  <- we will cut here
        feat_1_8 = x            # (B, C, H/8, W/8)
        x = self.layer3(x)      # 1/16
        x = self.layer4(x)      # 1/32

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        x = self.fc(x)
        return feat_1_8, x      # return both for flexibility

# ========================================= public encoder ====================#
class LIPResNetEncoder(nn.Module):
    """
    Pretrained ResNet-LIP encoder that returns a latent grid of spatial
    resolution H/8 x W/8 (28x28 for 224-pixel inputs).

    Parameters
    ----------
    backbone        : 'resnet34' | 'resnet50'
    pretrained      : load official LIP weights (.pth) if available
    latent_channels : output channel count so that torch.cat([Z_I, Z_LIP], 1) works
    freeze          : if True, encoder is kept in eval mode & no grads
    """

    _layers_cfg = {
        'resnet34': (BasicBlock , [3, 4, 6, 3], 128),   # C_out after 1/8 stage
        'resnet50': (Bottleneck, [3, 4, 6, 3], 512),
    }

    _url_or_path = {
        # update with your actual .pth paths
        'resnet34': None,
        'resnet50': './SD-VAE-weights/lip_resnet-50.pth',
    }

    def __init__(self,
                 backbone        = 'resnet50',
                 pretrained      = True,
                 latent_channels = 4,
                 model_freeze    = True,
                 lip_freeze      = False):
        super().__init__()
        if backbone not in ALLOWED_BACKBONES:
            raise ValueError(f"backbone must be one of {ALLOWED_BACKBONES}")

        block, layers, c_raw = self._layers_cfg[backbone]
        self.body  = _LIPResNetBackbone(block, layers)

        # 1×1 projection to desired latent depth
        self.proj = nn.Conv2d(c_raw, latent_channels, 1)
        nn.init.kaiming_normal_(self.proj.weight, mode='fan_out')
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0)

        # ---------------------------------- load weights -------------------- #
        if pretrained and self._url_or_path[backbone] is not None:
            state = torch.load(self._url_or_path[backbone], map_location='cpu')
            missing, _ = self.body.load_state_dict(state, strict=False)
            if missing:
                print("[LIPResNetEncoder] Warning: unmatched keys:", missing)

        # ---------------------------------- freeze option but lip ------------------- #
        if model_freeze:
            for p in self.parameters():
                p.requires_grad_(False)
            if not lip_freeze:
                for name, p in self.named_parameters():
                    if (".shared." in name) or (".logit." in name) or (".lip" in name):
                        p.requires_grad_(True)
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen    = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        print(f"Trainable LIP params : {trainable/1e6:.2f} M")
        print(f"Frozen backbone parms: {frozen/1e6:.2f} M")
    # ----------------------------------------------------------------------- #
    @torch.no_grad() if not torch.is_grad_enabled() else lambda f, *a, **kw: f
    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor, RGB image in [-1, 1], shape (B, 3, H, W)

        Returns
        -------
        z_lip : torch.Tensor, shape (B, latent_channels, H/8, W/8)
        """
        feat_1_8, x = self.body(x)          # keep only the 1/8 feature map
        z_lip       = self.proj(feat_1_8)
        return feat_1_8, x, z_lip
# ============================================================================ #
