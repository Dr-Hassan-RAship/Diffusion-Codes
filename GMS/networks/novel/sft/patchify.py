import torch
import torch.nn as nn

class LearnablePatchify(nn.Module):
    def __init__(self, patch_size=28):
        super(LearnablePatchify, self).__init__()
        self.patch_size = patch_size

        # Learnable projection over patches (input and output channels = 3 to preserve RGB)
        self.proj = nn.Conv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=patch_size,
            stride=patch_size
        )

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



