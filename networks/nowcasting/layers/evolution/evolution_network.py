import torch.nn.functional as F
from ..evolution.module import *

class Evolution_Network(nn.Module):
    def __init__(self, n_channels, n_classes, base_c=64, bilinear=True):
        super(Evolution_Network, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        base_c = base_c
        self.inc = DoubleConv(n_channels, base_c)
        self.down1 = Down(base_c * 1, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)

        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c * 1, bilinear)
        self.outc = OutConv(base_c * 1, n_classes)
        self.gamma = nn.Parameter(torch.zeros(1, n_classes, 1, 1), requires_grad=True)

        self.up1_v = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2_v = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3_v = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4_v = Up(base_c * 2, base_c * 1, bilinear)
        self.outc_v = OutConv(base_c * 1, n_classes * 2)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x) * self.gamma

        v = self.up1_v(x5, x4)
        v = self.up2_v(v, x3)
        v = self.up3_v(v, x2)
        v = self.up4_v(v, x1)
        v = self.outc_v(v)
        return x, v

