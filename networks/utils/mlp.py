""" MLP module w/ dropout and configurable activation layer
Hacked together by / Copyright 2020 Ross Wightman
"""
from turtle import forward
from torch import nn as nn
import torch
from networks.utils.utils import window_partition, window_reverse, PeriodicPad2d, SElayer
import math
# from networks.utils.moe_utils import TaskMoE, router_z_loss_func, load_balancing_loss_func
from megatron_utils.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class GluMlp(nn.Module):
    """ MLP w/ GLU style gating
    See: https://arxiv.org/abs/1612.08083, https://arxiv.org/abs/2002.05202
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.Sigmoid, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        assert hidden_features % 2 == 0

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features // 2, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def init_weights(self):
        # override init of fc1 w/ gate portion set to weight near zero, bias=1
        fc1_mid = self.fc1.bias.shape[0] // 2
        nn.init.ones_(self.fc1.bias[fc1_mid:])
        nn.init.normal_(self.fc1.weight[fc1_mid:], std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x, gates = x.chunk(2, dim=-1)
        x = x * self.act(gates)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class GatedMlp(nn.Module):
    """ MLP as used in gMLP
    """
    def __init__(
            self, in_features, hidden_features=None, out_features=None, window_size=[4,8], act_layer=nn.GELU, 
            norm_layer=nn.LayerNorm, bias=True, drop=0., get_weight=False):
        super().__init__()
        self.window_size = window_size
        self.get_weight = get_weight

        out_features = out_features or in_features
        if not get_weight:
            hidden_features = hidden_features or in_features * 2
        else:
            hidden_features = hidden_features or in_features
        self.hidden_features = hidden_features
        total_window_size = 1
        for i in window_size:
            total_window_size *= i

        self.norm = norm_layer(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        if not get_weight:
            self.norm1 = norm_layer(hidden_features//2)
        else:
            self.norm1 = norm_layer(hidden_features)
        self.spatial_fc = nn.Linear(total_window_size, total_window_size, bias=bias)
        if not self.get_weight:
            self.fc2 = nn.Linear(hidden_features//2, out_features, bias=bias)
            self.drop = nn.Dropout(drop)

    def forward(self, x):
        B, H, W, C = x.shape
        shortcut = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        if not self.get_weight:
            u, v = x.chunk(2, dim=-1) 
        else:
            u = x
        u = self.norm1(u)
        u = u.reshape(B, -1, C).permute(0, 2, 1)
        u = self.spatial_fc(u)
        u = u.permute(0, 2, 1).reshape(B, H, W, C)
        if not self.get_weight:
            x = (u + 1.) * v
            x = self.fc2(x)
            x = self.drop(x)
            x = x + shortcut
            return x
        else:
            return u


class ConvMlp(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims
    """
    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU,
            norm_layer=None, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias)
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

class MAGMlp(nn.Module):
    """ Multi-Axis Gated MLP Block(MAXIM: Multi-Axis MLP for Image Processing)
    """
    def __init__(self, dim, window_size=[4, 8], act_layer=nn.GELU, bias=True, drop=0., get_weight=False):
        super().__init__()
        self.window_size = window_size
        self.get_weight = get_weight

        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim, bias=bias)
        self.act = act_layer()
        self.local_gmlp = GatedMlp(dim // 2, window_size=window_size, act_layer=act_layer, drop=drop, get_weight=get_weight)
        self.global_gmlp = GatedMlp(dim // 2, window_size=window_size, act_layer=act_layer, drop=drop, get_weight=get_weight)
        self.fc2 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        shortcut = x
        B, H, W, C = x.shape
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        lb, gb = x.chunk(2, dim=-1)
        local_window = window_partition(lb, self.window_size)
        global_window_size = [H//self.window_size[0], W//self.window_size[1]]
        global_window = window_partition(gb, global_window_size).reshape(B, *self.window_size, -1, C//2).permute(0, 3, 1, 2, 4).reshape(-1, *self.window_size, C//2)
        local_window = self.local_gmlp(local_window)
        global_window = self.global_gmlp(global_window)
        global_window = global_window.reshape(B, -1, *self.window_size, C//2).permute(0, 2, 3, 1, 4).reshape(-1, *global_window_size, C//2)
        gb = window_reverse(global_window, global_window_size, H=H, W=W)
        lb = window_reverse(local_window, self.window_size, H=H, W=W)
        x = torch.cat([lb, gb], dim=-1)
        x = self.fc2(x)
        x = self.drop(x)
        if not self.get_weight:
            x = shortcut + x
        return x

class RCAB(nn.Module):
    """Residual Channel Attention Block(MAXIM: Multi-Axis MLP for Image Processing)
    """
    def __init__(self, dim, reduction=4) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.pad1 = PeriodicPad2d([1,1])
        self.conv1 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=0)
        self.act = nn.LeakyReLU()
        self.pad2 = PeriodicPad2d([1,1])
        self.conv2 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=0)

        self.selayer = SElayer(dim, reduction=reduction)
        

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.pad2(x)
        x = self.conv2(x)
        x = self.selayer(x)
        x = x.permute(0, 2, 3, 1)
        return shortcut + x

class RDCAB(nn.Module):
    def __init__(self, dim, reduction=4, bias = True, drop=0.1) -> None:
        super().__init__()
        
        self.norm = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=dim, bias=bias, drop=drop)
        self.selayer = SElayer(dim, reduction=reduction)

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        x = self.mlp(x).permute(0,3,1,2)
        x = self.selayer(x).permute(0,2,3,1)
        return shortcut + x



class DWMlp(nn.Module):
    """hilo中mlp, 可用于代替位置编码
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# class Mlp_withmoe(nn.Module):
#     """ MLP as used in Vision Transformer, MLP-Mixer and related networks
#     """
#     def __init__(self, in_features, attr_len, attr_hidden_size, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.,
#                 num_experts=1, expert_capacity=1., router_bias=True, router_noise=1e-2, is_scale_prob=True, drop_tokens=True):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.attr_len = attr_len
#         self.in_features = in_features

#         self.mlp = Mlp(in_features=in_features, hidden_features=hidden_features, out_features=out_features,
#                     act_layer=act_layer, bias=bias, drop=drop)

#         self.mlp = TaskMoE(hidden_size=attr_hidden_size, expert=self.mlp, num_experts=num_experts, attr_len=attr_len,
#                             expert_capacity=expert_capacity, router_bias=router_bias, router_noise=router_noise,
#                             is_scale_prob=is_scale_prob, drop_tokens=drop_tokens)


#     def forward(self, x, attr=None):

#         if self.attr_len > self.in_features and attr is not None:
#             x, gate_decision = self.mlp(x, torch.cat((x, attr),dim=-1))
#         elif attr is not None:
#             x, gate_decision = self.mlp(x, attr)
#         else:
#             x, gate_decision = self.mlp(x, x)

#         expert_index, router_probs, router_logits = gate_decision
#         z_loss = router_z_loss_func(router_logits=router_logits)
#         balance_loss = load_balancing_loss_func(router_probs=router_probs, expert_indices=expert_index)
#         return x, z_loss, balance_loss


class Mlp_parallel(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, 
                 bias=True, drop=0., use_cpu_initialization=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = ColumnParallelLinear(in_features, hidden_features, gather_output=False, 
                                        async_tensor_model_parallel_allreduce=False,
                                        use_cpu_initialization=use_cpu_initialization)
        self.act = act_layer()
        # self.drop1 = nn.Dropout(drop)
        self.fc2 = RowParallelLinear(hidden_features, out_features, input_is_parallel=True,
                                     use_cpu_initialization=use_cpu_initialization)

    def forward(self, x):
        Bs, H, W, _ = x.shape
        x = x.reshape(Bs, H*W, -1)
        x, _ = self.fc1(x)
        x = self.act(x)
        x, _ = self.fc2(x)
        x = x.reshape(Bs, H, W, -1)
        return x

