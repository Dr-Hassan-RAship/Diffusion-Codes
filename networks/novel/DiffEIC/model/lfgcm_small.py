import torch
import torch.nn as nn
from ..model.layers import *
    
class Encoder(nn.Module):
    def __init__(self, in_nc, mid_nc, out_nc, prior_nc, sft_ks):
        super().__init__()

        self.sft_1_8 = nn.Sequential(
            conv3x3(4, prior_nc),
            nn.GELU(),
            conv1x1(prior_nc, prior_nc)
        )

        self.sft_1_16 = nn.Sequential(
            conv3x3(prior_nc, prior_nc, stride=2),
            nn.GELU(),
            conv1x1(prior_nc, prior_nc)
        )

        self.sft_1_16_2 = nn.Sequential(
            conv3x3(prior_nc, prior_nc),
            nn.GELU(),
            conv1x1(prior_nc, prior_nc)
        )

        self.g_a1 = nn.Sequential(
            ResidualBlockWithStride(in_nc, mid_nc[2]),
            ResidualBottleneck(mid_nc[2]),
            ResidualBottleneck(mid_nc[2]),
            ResidualBottleneck(mid_nc[2]),
        )

        self.g_a1_ref = SFT(mid_nc[2], prior_nc, ks=sft_ks)

        self.g_a2  = nn.Sequential(
            ResidualBlockWithStride(mid_nc[2], mid_nc[3]),
            ResidualBottleneck(mid_nc[3]),
            ResidualBottleneck(mid_nc[3]),
            ResidualBottleneck(mid_nc[3]),
        )
        self.g_a2_ref = SFT(mid_nc[3], prior_nc, ks=sft_ks)


        self.g_a3 = conv3x3(mid_nc[3],mid_nc[3])
        self.g_a4 = SFTResblk(mid_nc[3], prior_nc, ks=sft_ks)
        self.g_a5 = SFTResblk(mid_nc[3], prior_nc, ks=sft_ks)

        self.g_a6 = conv3x3(mid_nc[3], out_nc)

    def forward(self, x, feature):
        sft_feature = self.sft_1_8(feature)
        x = self.g_a1(x)
        x = self.g_a1_ref(x, sft_feature)

        sft_feature = self.sft_1_16(sft_feature)
        x = self.g_a2(x)
        x = self.g_a2_ref(x, sft_feature)

        sft_feature = self.sft_1_16_2(sft_feature)
        x = self.g_a3(x)
        x = self.g_a4(x, sft_feature)
        x = self.g_a5(x, sft_feature)
        x = self.g_a6(x)

        return x
    
class Decoder(nn.Module):
    def __init__(self, N, M, out_nc, prior_nc, sft_ks):
        super().__init__()

        self.g_s2 = nn.Sequential(
            conv3x3(M,N),
            ResidualBottleneck(N),
            ResidualBottleneck(N),
            ResidualBottleneck(N))
        self.g_s3 = nn.Sequential(
            ResidualBlockUpsample(N, N),
            ResidualBottleneck(N),
            ResidualBottleneck(N),
            ResidualBottleneck(N))
        self.g_s4 = conv3x3(N, out_nc)

    def forward(self, x):
        x = self.g_s2(x)
        x = self.g_s3(x)
        x = self.g_s4(x)

        return x

class LFGCM(nn.Module):
    def __init__(self, in_nc, out_nc, enc_mid, N, M, prior_nc, sft_ks, slice_num, slice_ch):
        super().__init__()

        self.slice_num = slice_num
        self.slice_ch = slice_ch

        self.encoder = Encoder(in_nc, enc_mid, M, prior_nc, sft_ks)
        self.decoder = Decoder(N, M, out_nc, prior_nc, sft_ks)

    def forward(self, x, ref):
        y = self.encoder(x, ref)
        output = self.decoder(y)

        return output

if __name__ == "__main__":
    x = torch.randn(1,3,512,512)
    y = torch.randn(1,4,64,64)
    
    model = LFGCM(3,4,[192,192,192,192],128,192,64,3,5,[16,16,32,64,64])
    z = model(x,y)
