"""
Skilful precipitation nowcasting using deep generative models of radar, from DeepMind
https://arxiv.org/abs/2104.00954
"""

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

from .utils import Identity

class DBlock(nn.Module):
    """
    D Block in https://arxiv.org/abs/2104.00954, downsampling 2-D convolution block.
    Args:
        in_channels: int, number of channels of input tensor.
        out_channels: int, number of channels of output tensor.
        relu: bool, whether to apply ReLU function.
        downsample: bool, whether to apply scaling function.
    Return:
        torch.tensor
    """
    def __init__(
        self, 
        in_channels:int=None, 
        out_channels:int=None, 
        relu:bool=True, 
        downsample:bool=True
        ):
        super(DBlock, self).__init__()
        Scaling = (nn.AvgPool2d(2, 2) if downsample else Identity())
        ReLU = (nn.LeakyReLU(0.2) if relu else Identity()) #nn.ReLU() if relu else Identity())

        self.conv1x1 = nn.Sequential(
                spectral_norm(nn.Conv2d(in_channels, out_channels, 1, 1, 0), eps=1e-4),
                Scaling
                )
        self.conv3x3 = nn.Sequential(
                ReLU,
                spectral_norm(nn.Conv2d(in_channels, in_channels, 3, 1, 1), eps=1e-4),
                nn.LeakyReLU(0.2), #nn.ReLU(inplace=True),
                spectral_norm(nn.Conv2d(in_channels, out_channels, 3, 1, 1), eps=1e-4),
                Scaling
                )

    def forward(self, x):
        conv1x1 = self.conv1x1(x)
        conv3x3 = self.conv3x3(x)
        return conv1x1 + conv3x3

class D3Block(nn.Module):
    """
    D Block in https://arxiv.org/abs/2104.00954, downsampling 3-D convolution block.
    Args:
        in_channels: int, number of channels of input tensor.
        out_channels: int, number of channels of output tensor.
        relu: bool, whether to apply ReLU function.
        downsample: bool, whether to apply scaling function.
    Return:
        torch.tensor
    """
    def __init__(
        self, 
        in_channels:int=None, 
        out_channels:int=None, 
        relu:bool=True, 
        downsample:bool=True
        ):
        super(D3Block, self).__init__()
        Scaling = (nn.AvgPool3d(2, 2) if downsample else Identity())
        ReLU = (nn.LeakyReLU(0.2) if relu else Identity()) #nn.ReLU() if relu else Identity())

        self.conv1x1 = nn.Sequential(
                spectral_norm(nn.Conv3d(in_channels, out_channels, 1, 1, "same"), eps=1e-4),
                Scaling
                )
        self.conv3x3 = nn.Sequential(
                ReLU,
                spectral_norm(nn.Conv3d(in_channels, in_channels, 3, 1, "same"), eps=1e-4),
                nn.LeakyReLU(0.2), #nn.ReLU(inplace=True),
                spectral_norm(nn.Conv3d(in_channels, out_channels, 3, 1, "same"), eps=1e-4),
                Scaling
                )

    def forward(self, x):
        conv1x1 = self.conv1x1(x)
        conv3x3 = self.conv3x3(x)
        return conv1x1 + conv3x3