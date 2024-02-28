import torch
import torch.nn as nn


class RangeNorm(torch.nn.Module):
    def __init__(self, normalized_shape):
        super(RangeNorm, self).__init__()
        #self.shape = 1
        #for i in normalized_shape:
        #    self.shape *= i

    def forward(self, x):
        if torch.eq(x.std(), 0.0):
            return x
        shape_ = x.shape
        x_ = x.view(shape_[0], -1)
        min_ = x_.min(dim=-1, keepdim=True)[0]
        max_ = x_.max(dim=-1, keepdim=True)[0]
        return ((2 * x_ - (min_ + max_)) / (max_ - min_)).view(shape_)


class MIMNBase(nn.Module):
    def __init__(self, num_hidden, width, filter_size, tln=1):
        super(MIMNBase, self).__init__()
        """Initialize the basic Conv LSTM cell.
        Args:
            layer_name: layer names for different convlstm layers.
            filter_size: int tuple thats the height and width of the filter.
            num_hidden: number of units in output tensor.
            tln: whether to apply tensor layer normalization.
        """
        if tln:
            if tln == 1:
                norm = nn.LayerNorm
            elif tln == 2:
                norm = RangeNorm
        self.filter_size = filter_size
        self.paddings = self.filter_size // 2
        self.num_hidden = num_hidden
        self.s2s_conv = nn.ModuleList([nn.Sequential(nn.Conv2d(self.num_hidden, self.num_hidden,
                                                               kernel_size=self.filter_size, stride=1,
                                                               padding=self.paddings, bias=False),
                                       norm([num_hidden, *width]) if tln else None) for _ in range(4)])
        self.i2s_conv = nn.ModuleList([nn.Sequential(nn.Conv2d(self.num_hidden, self.num_hidden,
                                                               kernel_size=self.filter_size, stride=1,
                                                               padding=self.paddings, bias=False),
                                       norm([num_hidden, *width]) if tln else None) for _ in range(4)])
        self._forget_bias = 1.0
        self.ct_weight = nn.Parameter((torch.normal(torch.zeros((1, self.num_hidden*2, *width), dtype=torch.float), 1)))
        self.oc_weight = nn.Parameter((torch.normal(torch.zeros((1, self.num_hidden, *width), dtype=torch.float), 1)))

    def forward(self, x, h_t, c_t):
        i_h, g_h, f_h, o_h = [conv(h_t) for conv in self.s2s_conv]

        ct_activation = torch.mul(c_t.repeat(1, 2, 1, 1), self.ct_weight)
        i_c, f_c = torch.split(ct_activation, self.num_hidden, dim=1)

        i_ = i_h + i_c
        f_ = f_h + f_c
        g_ = g_h
        o_ = o_h

        if x != None:
            i_x, g_x, f_x, o_x = [conv(x) for conv in self.i2s_conv]

            i_ += i_x
            f_ += f_x
            g_ += g_x
            o_ += o_x

        i_ = torch.sigmoid(i_)
        f_ = torch.sigmoid(f_ + self._forget_bias)
        c_new = f_ * c_t + i_ * torch.tanh(g_)

        o_c = torch.mul(c_new, self.oc_weight)

        h_new = torch.sigmoid(o_ + o_c) * torch.tanh(c_new)

        return h_new, c_new


class MIMN(MIMNBase):
    def __init__(self, num_hidden, width, filter_size, tln=1):
        super(MIMN, self).__init__(num_hidden, width, filter_size, tln)


def ChSh(x, order):
    if order == 2:
        return 2*x**2-1
    elif order == 3:
        return 4*x**3-3*x


class MIMNPl(MIMNBase):
    def __init__(self, num_hidden, width, filter_size, tln=1, order=2):
        super(MIMNPl, self).__init__(num_hidden, width, filter_size, tln)
        self.order = order

    def forward(self, x, h_t, c_t):
        i_h, g_h, f_h, o_h = [ChSh(conv(h_t), self.order) for conv in self.s2s_conv]

        ct_activation = torch.mul(c_t.repeat(1, 2, 1, 1), self.ct_weight)
        i_c, f_c = torch.split(ct_activation, self.num_hidden, dim=1)

        i_ = i_h + i_c
        f_ = f_h + f_c
        g_ = g_h
        o_ = o_h

        if x != None:
            i_x, g_x, f_x, o_x = [ChSh(conv(x), self.order) for conv in self.i2s_conv]

            i_ += i_x
            f_ += f_x
            g_ += g_x
            o_ += o_x

        i_ = torch.sigmoid(i_)
        f_ = torch.sigmoid(f_ + self._forget_bias)
        c_new = f_ * c_t + i_ * torch.tanh(g_)

        o_c = torch.mul(c_new, self.oc_weight)

        h_new = torch.sigmoid(o_ + o_c) * torch.tanh(c_new)

        return h_new, c_new


class MIMNOA(MIMNBase):
    def __init__(self, num_hidden, width, filter_size, tln=1, act=None):
        super(MIMNOA, self).__init__(num_hidden, width, filter_size, tln)
        self.act = act

    def forward(self, x, h_t, c_t):
        i_h, g_h, f_h, o_h = [self.act(conv(h_t)).clone() for conv in self.s2s_conv]
        ct_activation = torch.mul(c_t.repeat(1, 2, 1, 1), self.ct_weight)
        i_c, f_c = torch.split(ct_activation, self.num_hidden, dim=1)

        i_ = i_h + i_c
        f_ = f_h + f_c
        g_ = g_h
        o_ = o_h

        if x != None:
            i_x, g_x, f_x, o_x = [self.act(conv(x)) for conv in self.i2s_conv]

            i_ += i_x
            f_ += f_x
            g_ += g_x
            o_ += o_x

        i_ = torch.sigmoid(i_)
        f_ = torch.sigmoid(f_ + self._forget_bias)
        c_new = f_ * c_t + i_ * torch.tanh(g_)

        o_c = torch.mul(c_new, self.oc_weight)

        h_new = torch.sigmoid(o_ + o_c) * torch.tanh(c_new)

        return h_new, c_new


