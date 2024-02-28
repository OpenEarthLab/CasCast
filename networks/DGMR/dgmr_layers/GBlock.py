"""
Skilful precipitation nowcasting using deep generative models of radar, from DeepMind
https://arxiv.org/abs/2104.00954
"""

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

from .utils import Identity, depth2space, Rearrange

class GBlockCell(nn.Module):
    """
    Component of G Block in https://arxiv.org/abs/2104.00954, upsampling convolution block.
    Args:
        in_channels: int, number of channels of input tensor.
        out_channels: int, number of channels of output tensor.
        upsample: bool, whether to apple upsampling function.
    Return:
        torch.tensor
    """
    def __init__(
        self,
        in_channels: int=None,
        out_channels: int=None,
        upsample: bool=True
        ):
        super(GBlockCell, self).__init__()
        Scaling = (nn.Upsample if upsample else Identity)
        ReLU = nn.LeakyReLU(0.2)
        # self.conv3x3 = nn.Sequential(
        #         nn.BatchNorm2d(in_channels),
        #         ReLU, #nn.ReLU(inplace=True),
        #         Scaling(scale_factor=2),
        #         spectral_norm(nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False), eps=1e-4),
        #         nn.BatchNorm2d(in_channels),
        #         ReLU, #nn.ReLU(inplace=True),
        #         spectral_norm(nn.Conv2d(in_channels, out_channels, 3, 1, 1), eps=1e-4)
        #         )
        self.conv3x3 = nn.Sequential(
                Rearrange('b c h w -> b h w c'),
                nn.LayerNorm(in_channels),
                Rearrange('b h w c -> b c h w'),
                ReLU, #nn.ReLU(inplace=True),
                Scaling(scale_factor=2),
                spectral_norm(nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False), eps=1e-4),
                Rearrange('b c h w -> b h w c'),
                nn.LayerNorm(in_channels),
                Rearrange('b h w c -> b c h w'),
                ReLU, #nn.ReLU(inplace=True),
                spectral_norm(nn.Conv2d(in_channels, out_channels, 3, 1, 1), eps=1e-4)
                )
        self.conv1x1 = nn.Sequential(
            Scaling(scale_factor=2),
            spectral_norm(nn.Conv2d(in_channels, out_channels, 1, 1, 0), eps=1e-4)
            )


    def forward(self, x):
        conv3x3 = self.conv3x3(x)
        conv1x1 = self.conv1x1(x)
        return conv3x3 + conv1x1

class GBlock(nn.Module):
    """
    G Block in https://arxiv.org/abs/2104.00954, upsampling convolution block.
    Args:
        in_channels: int, number of channels of input tensor.
        out_channels: int, number of channels of output tensor.
    Return:
        torch.tensor
    """
    def __init__(
        self,
        in_channels: int=None,
        out_channels: int=None
        ):
        super(GBlock, self).__init__()
        self.conv = spectral_norm(nn.Conv2d(in_channels, in_channels, 1, 1, 0, bias=False), eps=1e-4)
        self.g_block = GBlockCell(in_channels, in_channels, upsample=False)
        self.g_block_up = GBlockCell(in_channels, out_channels, upsample=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.g_block(x)
        x = self.g_block_up(x)
        return x

class LastGBlock(nn.Module):
    """
    Final G Block in https://arxiv.org/abs/2104.00954, a convolution block.
    Args:
        in_channels: int, number of channels of input tensor.
    Return:
        torch.tensor
    """
    def __init__(
        self,
        in_channels: int=None
        ):
        super(LastGBlock, self).__init__()
        self.conv = spectral_norm(nn.Conv2d(in_channels, in_channels, 1, 1, 0, bias=False), eps=1e-4)
        self.g_block = GBlockCell(in_channels, in_channels, upsample=False)
        self.g_block_up = GBlockCell(in_channels, in_channels, upsample=True)
        ReLU = nn.LeakyReLU(0.2)
        # self.conv_out = nn.Sequential(
        #         nn.BatchNorm2d(in_channels),
        #         ReLU,#nn.ReLU(inplace=True),
        #         spectral_norm(nn.Conv2d(in_channels, 4, 1, 1, 0), eps=1e-4)
        #         )
        self.conv_out = nn.Sequential(
                Rearrange('b c h w -> b h w c'),
                nn.LayerNorm(in_channels),
                Rearrange('b h w c -> b c h w'),
                ReLU,#nn.ReLU(inplace=True),
                spectral_norm(nn.Conv2d(in_channels, 4, 1, 1, 0), eps=1e-4)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.g_block(x)
        x = self.g_block_up(x)
        x = self.conv_out(x)
        x = depth2space(x)
        return x