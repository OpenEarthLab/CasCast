# SPADE Module and Block are adapted from Nvidia SPADE project (https://github.com/NVlabs/SPADE).

import re
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

class GenBlock(nn.Module):
    def __init__(self, fin, fout, opt, use_se=False, dilation=1, double_conv=False):
        super().__init__()
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        self.opt = opt
        self.double_conv = double_conv

        self.pad = nn.ReflectionPad2d(dilation)
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=0, dilation=dilation)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=0, dilation=dilation)

        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        self.conv_0 = spectral_norm(self.conv_0)
        self.conv_1 = spectral_norm(self.conv_1)
        if self.learned_shortcut:
            self.conv_s = spectral_norm(self.conv_s)

        ic = opt.evo_ic
        
        self.norm_0 = SPADE(fin, ic)
        self.norm_1 = SPADE(fmiddle, ic)
        if self.learned_shortcut:
            self.norm_s = SPADE(fin, ic)

    def forward(self, x, evo):
        x_s = self.shortcut(x, evo)
        dx = self.conv_0(self.pad(self.actvn(self.norm_0(x, evo))))
        if self.double_conv:
            dx = self.conv_1(self.pad(self.actvn(self.norm_1(dx, evo))))

        out = x_s + dx

        return out

    def shortcut(self, x, evo):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, evo))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()

        ks = 3

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        nhidden = 64
        ks = 3
        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.ReflectionPad2d(pw),
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=0),
            nn.ReLU()
        )
        self.pad = nn.ReflectionPad2d(pw)
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=0)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=0)

    def forward(self, x, evo):

        normalized = self.param_free_norm(x)
        evo = F.adaptive_avg_pool2d(evo, output_size=x.size()[2:])

        actv = self.mlp_shared(evo)

        gamma = self.mlp_gamma(self.pad(actv))
        beta = self.mlp_beta(self.pad(actv))

        out = normalized * (1 + gamma) + beta

        return out




