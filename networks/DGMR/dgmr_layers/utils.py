import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange



def space2depth(
    x: torch.tensor=None, 
    factor: int=2
    ):
    """
    Relocate pixels at (H, W) dimension to channel.
    Args:
        x: torch.tensor, tensor to be transformed.
        factor: int, factor of size reduction.
    """
    B, C, H, W = x.shape
    x = nn.Unfold(factor, stride=factor)(x)
    return x.view(B, C * factor ** 2, H // factor, W // factor)

def depth2space(
    x: torch.tensor=None,
    factor: int=2
    ):
    """
    Relocate pixels at channel to (H, W) dimension.
    Args:
        x: torch.tensor, tensor to be transformed.
        factor: int, factor of size expansion.
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H*W)
    x = nn.Fold((H*factor, W*factor), kernel_size=(factor, factor), stride=factor)(x)
    return x

class Space2Depth(nn.Module):
    """
    Relocate pixels at (H, W) dimension to channel.
    See space2depth.
    """
    def __init__(self, *args):
        super(Space2Depth, self).__init__()
    def forward(self, x):
        if len(x.shape) == 4:
            return space2depth(x)
        if len(x.shape) == 5:
            B, C, T, H, W = x.shape
            x = x.permute(0, 2, 1, 3, 4) # B, T, C, H, W
            x = x.reshape(B*T, C, H, W)
            x = space2depth(x)
            x = x.view(B,  T, -1, x.shape[-2], x.shape[-1])
            x = x.permute(0, 2, 1, 3, 4) # B, C, T, H, W
            return x

class Identity(nn.Module):
    def __init__(self, **kwargs):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

def random_crop(x, size=128, padding=False):
    B, C, H, W = x.shape
    if padding:
        # TODO: add padding=True method
        pass
    else:
        h_idx = random.randint(0, H-size)
        w_idx = random.randint(0, W-size)
        x = x[:, :, h_idx:h_idx+size, w_idx:w_idx+size]
    return x

class Rearrange(nn.Module):
    def __init__(self, rearrange_type):
        super(Rearrange, self).__init__()
        self.rearrange_type = rearrange_type

    def forward(self, x):
        x = rearrange(x, self.rearrange_type)
        return x