# The purpose of this file is to define the SFTblock consisting of a 3 x 3 conv + LeakyReLU + 3 x 3 conv (x 2)
# Defining two classes SFT and SFTResblock. And any other functions or classes (utility nature) should be in the sft folder

# The nature of this script lies in the fact that the class SFT is being used as a guidance mechanism for the original ZI latent gotten from LiteVAE

# some sample imports
import torch
import torch.nn as nn
import torch.nn.functional as F

class SFT(nn.Module):
    def __init__(self, original_channels, guidance_channels, ks=3, nhidden=128):
        super().__init__()
        pw = ks // 2

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(guidance_channels, nhidden, kernel_size=ks, padding=pw),
            nn.LeakyReLU(0.2, True)
        )
        self.mlp_gamma = nn.Conv2d(nhidden, original_channels, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, original_channels, kernel_size=ks, padding=pw)

    def forward(self, x, ref):
        # Note the avg pool is no longer needed as the spatial dimensons already matched
        # ref     = F.adaptive_avg_pool2d(ref, x.size()[2:]) # avg pool on the spatial dimensions of ref to match x
        actv    = self.mlp_shared(ref)
        gamma   = self.mlp_gamma(actv)
        beta    = self.mlp_beta(actv)
        out     = x * (1 + gamma) + beta

        return out

class SFTResblk(nn.Module):
    def __init__(self, original_channels, guidance_channels, ks=3, dtype = torch.float32, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        
        self.conv_0 = nn.Conv2d(original_channels, original_channels, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(original_channels, original_channels, kernel_size=3, padding=1)

        self.norm_0 = SFT(original_channels, guidance_channels, ks=ks).to(dtype = dtype, device = device)
        self.norm_1 = SFT(original_channels, guidance_channels, ks=ks).to(dtype = dtype, device = device)
    
    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)
    
    def forward(self, x, ref):
        dx = self.conv_0(self.actvn(self.norm_0(x, ref)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, ref)))
        out = x + dx

        return out

class SFTModule(nn.Module):
    def __init__(self, original_channels, guidance_channels, dtype = torch.float32, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.sftresblk = SFTResblk(original_channels, guidance_channels).to(dtype = torch.float32, device = device)

    def forward(self, x, ref):
        x = self.sftresblk(x, ref)
        return x

if __name__ == "__main__":
    # Example usage
    original = torch.randn(2, 10, 28, 28)
    guidance = torch.randn(2, 64, 28, 28)
    sft_module = SFTModule(original_channels=10, guidance_channels=64)
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sft_module = sft_module.to(device)
    original = original.to(device)
    guidance = guidance.to(device)
    output = sft_module(original, guidance)
    print("Output shape:", output.shape)