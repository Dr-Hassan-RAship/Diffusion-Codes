import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnablePatchify(nn.Module):
    def __init__(self, patch_size=28):
        super(LearnablePatchify, self).__init__()
        self.patch_size = patch_size

        # Learnable projection over patches (input and output channels = 3 to preserve RGB)
        self.proj = nn.Conv2d(
            in_channels  = 3,
            out_channels = 3,
            kernel_size  = patch_size,
            stride       = patch_size
        )
    
    # (B, 3, 224, 224) --> (B, , 28, 28) and if we choose kernel size of higher number then we will have N2 << N1

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, "Image dimensions must be divisible by patch size"

        # Apply learnable projection
        patches = self.proj(x)  # Shape: (B, 3, H/patch, W/patch)

        # Convert to (B, N, 3)
        patches = patches.permute(0, 2, 3, 1)  # (B, H/P, W/P, 3)
        N = (H // self.patch_size) * (W // self.patch_size)
        patches = patches.reshape(B, N, 3)  # Flatten spatial grid

        # Upsample patches to 28x28 with 3 channels
        out = patches.reshape(B * N, 3, 1, 1)  # shape: (B*N, 3, 1, 1)
        out = nn.functional.interpolate(out, size=(self.patch_size, self.patch_size), mode='bilinear', align_corners=False)
        out = out.reshape(B, N, 3, self.patch_size, self.patch_size).mean(dim = 2)  # shape: (B, N, 3, 28, 28)
        return out

# Alternative 2: Making a patchify class which is non-learnable. (B, 3, 224, 224) --> (B, 64, 28, 28)

def extract_patches_mean(arr, kernel_size=28, stride=28):
    """
    Extracts non-overlapping patches from a 4D image tensor and averages across channels.

    Args:
        arr (Tensor): Input tensor of shape [B, C, H, W]
        kernel_size (int): Height and width of each patch
        stride (int): Stride between patches

    Returns:
        Tensor: Patch tensor of shape [B, N_patches, H_patch, W_patch], averaged across channels
    """
    B, C, H, W = arr.shape

    # Step 1: Unfold to get flat patches
    patches = F.unfold(arr, kernel_size=kernel_size, stride=stride)  # [B, C*K*K, N_patches]

    # Step 2: Rearrange to [B*N_patches, C, K, K]
    patches = patches.transpose(1, 2).contiguous().view(-1, C, kernel_size, kernel_size)

    # Step 3: Group back per image: [B, N_patches, C, K, K]
    N_patches = patches.shape[0] // B
    patches = patches.reshape(B, N_patches, C, kernel_size, kernel_size)

    # Step 4: Average across channels → [B, N_patches, K, K]
    patches = patches.mean(dim=2)

    return patches

# Example usage
# if __name__ == "__main__":
#     B, C, H, W = 2, 3, 224, 224
#     model = LearnablePatchify(patch_size=28)

#     x = torch.randn(B, C, H, W)
#     y = model(x)
#     print("Output shape:", y.shape)  # (2, 64, 3, 28, 28)

#     # check the number of learnable parameters in the class using torchinfo.summary
#     from torchinfo import summary
#     print(summary(model, input_size=(B, C, H, W), verbose=0))



