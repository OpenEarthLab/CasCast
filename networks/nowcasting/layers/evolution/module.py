import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class DoubleConv(nn.Module):

    def   __init__(self, in_channels, out_channels, kernel=3, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels, mid_channels, kernel_size=kernel, padding=kernel//2)),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            spectral_norm(nn.Conv2d(mid_channels, out_channels, kernel_size=kernel, padding=kernel//2)),
        )
        self.single_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=kernel // 2))
        )

    def forward(self, x):
        shortcut = self.single_conv(x)
        x = self.double_conv(x)
        x = x + shortcut
        return x

class Down(nn.Module):

    def __init__(self, in_channels, out_channels, kernel=3):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, kernel)
        )

    def forward(self, x):
        x = self.maxpool_conv(x)
        return x

class Up(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True, kernel=3):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, kernel=kernel, mid_channels=in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, kernel)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Up_S(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True, kernel=3):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, kernel=kernel, mid_channels=in_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, kernel)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

