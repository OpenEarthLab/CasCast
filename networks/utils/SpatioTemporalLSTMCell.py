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


class SpatioTemporalLSTMCellBase(nn.Module):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, layer_norm):
        super(SpatioTemporalLSTMCellBase, self).__init__()
        if layer_norm:
            if layer_norm == 1:
                norm = nn.LayerNorm
            elif layer_norm == 2:
                norm = RangeNorm
        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        self.conv_x = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_channel, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
            norm([num_hidden, *width]) if layer_norm else None
        ) for _ in range(7)])
        self.conv_h = nn.ModuleList([nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
            norm([num_hidden, *width]) if layer_norm else None
        ) for _ in range(4)])
        self.conv_m = nn.ModuleList([nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
            norm([num_hidden, *width]) if layer_norm else None
        ) for _ in range(3)])
        self.conv_o = nn.Sequential(
            nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
            norm([num_hidden, *width]) if layer_norm else None
        )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0)

    def forward(self, x_t, h_t, c_t, m_t):
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = [conv_x(x_t) for conv_x in self.conv_x]
        i_h, f_h, g_h, o_h = [conv_h(h_t) for conv_h in self.conv_h]
        i_m, f_m, g_m = [conv_m(m_t) for conv_m in self.conv_m]

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        m_new = f_t_prime * m_t + i_t_prime * g_t_prime

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new


class SpatioTemporalLSTMCell(SpatioTemporalLSTMCellBase):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, layer_norm):
        super(SpatioTemporalLSTMCell, self).__init__(in_channel, num_hidden, width, filter_size, stride, layer_norm)


def ChSh(x, order):
    if order == 2:
        return 2*torch.pow(x, 2)-1
    elif order == 3:
        return 4*x**3-3*x


class SpatioTemporalLSTMCellPoly(SpatioTemporalLSTMCellBase):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, layer_norm, order):
        super(SpatioTemporalLSTMCellPoly, self).__init__(
            in_channel, num_hidden, width, filter_size, stride, layer_norm)
        self.order = order

    def forward(self, x_t, h_t, c_t, m_t):
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = [ChSh(conv_x(x_t), self.order) for conv_x in self.conv_x]
        i_h, f_h, g_h, o_h = [ChSh(conv_h(h_t), self.order) for conv_h in self.conv_h]
        i_m, f_m, g_m = [ChSh(conv_m(m_t), self.order) for conv_m in self.conv_m]
        print(ChSh(self.conv_x[0](x_t), self.order))
        print(self.conv_x[0](x_t))

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        m_new = f_t_prime * m_t + i_t_prime * g_t_prime

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new


class SpatioTemporalLSTMCellOtherAct(SpatioTemporalLSTMCellBase):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, layer_norm, act):
        super(SpatioTemporalLSTMCellOtherAct, self).__init__(
            in_channel, num_hidden, width, filter_size, stride, layer_norm)
        self.act = act

    def forward(self, x_t, h_t, c_t, m_t):
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = [self.act(conv_x(x_t)) for conv_x in self.conv_x]
        i_h, f_h, g_h, o_h = [self.act(conv_h(h_t)) for conv_h in self.conv_h]
        i_m, f_m, g_m = [self.act(conv_m(m_t)) for conv_m in self.conv_m]

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        m_new = f_t_prime * m_t + i_t_prime * g_t_prime

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new


class SpatioTemporalLSTMCellv2(nn.Module):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, tln):
        super(SpatioTemporalLSTMCellv2, self).__init__()
        if tln:
            if tln == 1:
                norm = nn.LayerNorm
            elif tln == 2:
                norm = RangeNorm
        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        self.conv_x = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_channel, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
            norm([num_hidden, *width]) if tln else None
        ) for _ in range(4)])
        self.conv_h = nn.ModuleList([nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
            norm([num_hidden, *width]) if tln else None
        ) for _ in range(4)])
        self.conv_m = nn.ModuleList([nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
            norm([num_hidden, *width]) if tln else None
        ) for _ in range(4)])
        self.conv_o = nn.Sequential(
            nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0),
            norm([num_hidden, *width]) if tln else None
        )

    def forward(self, x_t, h_t, c_t, m_t):
        i_x, g_x, f_x, o_x = [conv_x(x_t) for conv_x in self.conv_x]
        i_t, g_t, f_t, o_t = [conv_h(h_t) for conv_h in self.conv_h]
        i_s, g_s, f_s, o_s = [conv_m(m_t) for conv_m in self.conv_m]

        i = torch.sigmoid(i_x + i_t)
        i_ = torch.sigmoid(i_x + i_s)
        g = torch.tanh(g_x + g_t)
        g_ = torch.tanh(g_x + g_s)
        f = torch.sigmoid(f_x + f_t + self._forget_bias)
        f_ = torch.sigmoid(f_x + f_s + self._forget_bias)
        o = torch.sigmoid(o_x + o_t + o_s)
        new_m = f_ * m_t + i_ * g_
        new_c = f * c_t + i * g
        cell = torch.cat([new_c, new_m], 1)
        cell = self.conv_o(cell)
        new_h = o * torch.tanh(cell)

        return new_h, new_c, new_m



