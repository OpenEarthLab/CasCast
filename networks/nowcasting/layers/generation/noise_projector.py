import torch
import torch.nn as nn
from ..utils import spectral_norm

class Noise_Projector(nn.Module):
    def __init__(self, input_length, configs):
        super(Noise_Projector, self).__init__()
        self.input_length = input_length
        self.conv_first = spectral_norm(nn.Conv2d(self.input_length, self.input_length * 2, kernel_size=3, padding=1))
        self.L1 = ProjBlock(self.input_length * 2, self.input_length * 4)
        self.L2 = ProjBlock(self.input_length * 4, self.input_length * 8)
        self.L3 = ProjBlock(self.input_length * 8, self.input_length * 16)
        self.L4 = ProjBlock(self.input_length * 16, self.input_length * 32)

    def forward(self, x):
        x = self.conv_first(x)
        x = self.L1(x)
        x = self.L2(x)
        x = self.L3(x)
        x = self.L4(x)

        return x


class ProjBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ProjBlock, self).__init__()
        self.one_conv = spectral_norm(nn.Conv2d(in_channel, out_channel-in_channel, kernel_size=1, padding=0))
        self.double_conv = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)),
            nn.ReLU(),
            spectral_norm(nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1))
        )

    def forward(self, x):
        x1 = torch.cat([x, self.one_conv(x)], dim=1)
        x2 = self.double_conv(x)
        output = x1 + x2
        return output