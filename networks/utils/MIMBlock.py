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


class MIMBlockBase(nn.Module):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, tln=False):
        super().__init__()
        if tln:
            if tln == 1:
                norm = nn.LayerNorm
            elif tln == 2:
                norm = RangeNorm
        self.num_hidden = num_hidden
        #self.device = device
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        self.conv_s2s = nn.ModuleList([nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
            norm([num_hidden, *width]) if tln else None) for _ in range(4)])
        self.conv_i2s = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_channel, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
            norm([num_hidden, *width]) if tln else None) for _ in range(4)])
        self.conv_x = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_channel, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
            norm([num_hidden, *width]) if tln else None) for _ in range(4)])
        self.conv_h = nn.ModuleList([nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
            norm([num_hidden, *width]) if tln else None) for _ in range(3)])
        self.conv_m = nn.ModuleList([nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
            norm([num_hidden, *width]) if tln else None) for _ in range(4)])
        self.conv_o = nn.Sequential(
            nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0),
            norm([num_hidden, *width]) if tln else None
        )
        self.ct_weight = nn.Parameter((torch.normal(torch.zeros((1, self.num_hidden * 2, *width),
                                                                dtype=torch.float), 1)))
        self.oc_weight = nn.Parameter((torch.normal(torch.zeros((1, self.num_hidden, *width), dtype=torch.float), 1)))

    def MIMS(self, x, h_t, c_t):
        i_h, g_h, f_h, o_h = [conv_s2s(h_t) for conv_s2s in self.conv_s2s]

        ct_activation = torch.mul(c_t.repeat([1, 2, 1, 1]), self.ct_weight)
        i_c, f_c = torch.split(ct_activation, self.num_hidden, dim=1)

        i_ = i_h + i_c
        f_ = f_h + f_c
        g_ = g_h
        o_ = o_h

        if x != None:
            i_x, g_x, f_x, o_x = [conv_i2s(x) for conv_i2s in self.conv_i2s]

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

    def forward(self, x, diff_h, h, c, m, lstm_c):
        i_s, g_s, f_s, o_s = [conv_m(m) for conv_m in self.conv_m]
        i_t, g_t, o_t = [conv_h(h) for conv_h in self.conv_h]
        i_x, g_x, f_x, o_x = [conv_x(x) for conv_x in self.conv_x]

        i = torch.sigmoid(i_x + i_t)
        i_ = torch.sigmoid(i_x + i_s)
        g = torch.tanh(g_x + g_t)
        g_ = torch.tanh(g_x + g_s)
        f_ = torch.sigmoid(f_x + f_s + self._forget_bias)
        o = torch.sigmoid(o_x + o_t + o_s)
        new_m = f_ * m + i_ * g_
        c, lstm_c = self.MIMS(diff_h, c, lstm_c)  # self.convlstm_c
        new_c = c + i * g
        cell = torch.cat([new_c, new_m], 1)
        cell = self.conv_o(cell)
        new_h = o * torch.tanh(cell)

        return new_h, new_c, new_m, lstm_c

class MIMBlock(MIMBlockBase):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, tln=False):
        super(MIMBlock, self).__init__(in_channel, num_hidden, width, filter_size, stride, tln)


def ChSh(x, order):
    if order == 2:
        return 2*x**2-1
    elif order == 3:
        return 4*x**3-3*x


class MIMBlockPl(MIMBlockBase):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, device, tln=False, order=2):
        super(MIMBlockPl, self).__init__(in_channel, num_hidden, width, filter_size, stride, device, tln)
        self.order = order

    def MIMS(self, x, h_t, c_t):
        i_h, g_h, f_h, o_h = [ChSh(conv_s2s(h_t), self.order) for conv_s2s in self.conv_s2s]

        ct_activation = torch.mul(c_t.repeat([1, 2, 1, 1]), self.ct_weight)
        i_c, f_c = torch.split(ct_activation, self.num_hidden, dim=1)

        i_ = i_h + i_c
        f_ = f_h + f_c
        g_ = g_h
        o_ = o_h

        if x != None:
            i_x, g_x, f_x, o_x = [ChSh(conv_i2s(x), self.order) for conv_i2s in self.conv_i2s]

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

    def forward(self, x, diff_h, h, c, m, lstm_c):
        i_s, g_s, f_s, o_s = [ChSh(conv_m(m), self.order) for conv_m in self.conv_m]
        i_t, g_t, o_t = [ChSh(conv_h(h), self.order) for conv_h in self.conv_h]
        i_x, g_x, f_x, o_x = [ChSh(conv_x(x), self.order) for conv_x in self.conv_x]

        i = torch.sigmoid(i_x + i_t)
        i_ = torch.sigmoid(i_x + i_s)
        g = torch.tanh(g_x + g_t)
        g_ = torch.tanh(g_x + g_s)
        f_ = torch.sigmoid(f_x + f_s + self._forget_bias)
        o = torch.sigmoid(o_x + o_t + o_s)
        new_m = f_ * m + i_ * g_
        c, lstm_c = self.MIMS(diff_h, c, lstm_c)  # self.convlstm_c
        new_c = c + i * g
        cell = torch.cat([new_c, new_m], 1)
        cell = ChSh(self.conv_o(cell), self.order)
        new_h = o * torch.tanh(cell)

        return new_h, new_c, new_m, lstm_c


class MIMBlockOA(MIMBlockBase):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, device, tln=False, act=None):
        super(MIMBlockOA, self).__init__(in_channel, num_hidden, width, filter_size, stride, device, tln)
        self.act = act

    def MIMS(self, x, h_t, c_t):
        i_h, g_h, f_h, o_h = [self.act(conv_s2s(h_t)).clone() for conv_s2s in self.conv_s2s]
        ct_activation = torch.mul(c_t.repeat([1, 2, 1, 1]), self.ct_weight)
        i_c, f_c = torch.split(ct_activation, self.num_hidden, dim=1)

        i_ = i_h + i_c
        f_ = f_h + f_c
        g_ = g_h
        o_ = o_h

        if x != None:
            i_x, g_x, f_x, o_x = [self.act(conv_i2s(x)) for conv_i2s in self.conv_i2s]

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

    def forward(self, x, diff_h, h, c, m, lstm_c):
        i_s, g_s, f_s, o_s = [self.act(conv_m(m)) for conv_m in self.conv_m]
        i_t, g_t, o_t = [self.act(conv_h(h)) for conv_h in self.conv_h]
        i_x, g_x, f_x, o_x = [self.act(conv_x(x)) for conv_x in self.conv_x]

        i = torch.sigmoid(i_x + i_t)
        i_ = torch.sigmoid(i_x + i_s)
        g = torch.tanh(g_x + g_t)
        g_ = torch.tanh(g_x + g_s)
        f_ = torch.sigmoid(f_x + f_s + self._forget_bias)
        o = torch.sigmoid(o_x + o_t + o_s)
        new_m = f_ * m + i_ * g_
        c, lstm_c = self.MIMS(diff_h, c, lstm_c)  # self.convlstm_c
        new_c = c + i * g
        cell = torch.cat([new_c, new_m], 1)
        cell = self.act(self.conv_o(cell))
        new_h = o * torch.tanh(cell)

        return new_h, new_c, new_m, lstm_c

