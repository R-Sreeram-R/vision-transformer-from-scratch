import torch 
from torch import nn 
import random 

# Convert 2D image inot 1D embedding vector

class PatchEmbedding(nn.Module):

    def __init__(self, in_channels=3, patch_size=16, 
                 embedding_dim=768):
        super().__init__()

        self.patcher = nn.Conv2d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0)
        self.patch_size = patch_size
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self,x):

        image_res = x.shape[-1]

        assert image_res % self.patch_size == 0, f"Input image size must be divisible by patch size, image shape: {image_res}, patch size: {self.patch_size}"
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)

        return x_flattened.permute(0,2,1) # check once



