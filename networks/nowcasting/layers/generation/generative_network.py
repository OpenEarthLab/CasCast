import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from ..generation.module import GenBlock
from ..evolution.module import *

class Generative_Encoder(nn.Module):
    def __init__(self, n_channels, base_c=64):
        super(Generative_Encoder, self).__init__()
        base_c = base_c
        self.inc = DoubleConv(n_channels, base_c, kernel=3)
        self.down1 = Down(base_c * 1, base_c * 2, 3)
        self.down2 = Down(base_c * 2, base_c * 4, 3)
        self.down3 = Down(base_c * 4, base_c * 8, 3)

    def forward(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        return x

class Generative_Decoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf

        ic = opt.ic_feature
        self.fc = nn.Conv2d(ic, 8 * nf, 3, padding=1)

        self.head_0 = GenBlock(8 * nf, 8 * nf, opt)

        self.G_middle_0 = GenBlock(8 * nf, 4 * nf, opt, double_conv=True)
        self.G_middle_1 = GenBlock(4 * nf, 4 * nf, opt, double_conv=True)

        self.up_0 = GenBlock(4 * nf, 2 * nf, opt)

        self.up_1 = GenBlock(2 * nf, 1 * nf, opt, double_conv=True)
        self.up_2 = GenBlock(1 * nf, 1 * nf, opt, double_conv=True)

        final_nc = nf * 1

        self.conv_img = nn.Conv2d(final_nc, self.opt.gen_oc, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2)
    
    def forward(self, x, evo):
        x = self.fc(x)
        x = self.head_0(x, evo)
        x = self.up(x)
        x = self.G_middle_0(x, evo)
        x = self.G_middle_1(x, evo)
        x = self.up(x)
        x = self.up_0(x, evo)
        x = self.up(x)
        x = self.up_1(x, evo)
        x = self.up_2(x, evo)
        x = self.conv_img(F.leaky_relu(x, 2e-1))
        return x