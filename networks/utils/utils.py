from turtle import forward
import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Optional
import numpy as np
from timm.models.layers import to_2tuple
from einops import rearrange



class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class GroupNorm(nn.Module):
    def __init__(self, in_channels):
        super(GroupNorm, self).__init__()
        self.gn = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

    def forward(self, x):
        return self.gn(x)



class attn_norm(nn.Module):
    def __init__(self, dim=-1, method='softmax') -> None:
        super().__init__()
        if method == 'softmax':
            self.attn_norm = nn.Softmax(dim=dim)
        elif method == 'squared_relu':
            self.attn_norm = nn.ReLU()
        elif method == 'softmax_plus':
            self.attn_norm = nn.Softmax(dim=dim)

        self.method = method

    def forward(self, x):
        if self.method == 'softmax':
            return self.attn_norm(x)
        else:
            mask = x > -torch.inf / 10
            l = x.shape[-1]
            if self.method == 'squared_relu':
                return self.attn_norm(x)**2
            elif self.method == 'softmax_plus':
                scale = np.log(l)/np.log(512) * mask + 1 - mask * 1
                return self.attn_norm(x * scale)



def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


def window_partition(x, window_size: tuple):
    """
    将feature map按照window_size划分成一个个没有重叠的window
    Args:
        x: (B, H, W, C)
        window_size (tuple): window size(Wt, Wh, Ww)
    Returns:
        windows: (num_windows*B, window_size, C)
    """
    if len(window_size) == 3:
        B, T, H, W, C = x.shape
        x = x.view(B, T // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
        # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
        # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
        windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0], window_size[1], window_size[2], C)
    elif len(window_size) == 2:
        B, H, W, C = x.shape
        x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
        # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows



def window_reverse(windows, window_size, T=1, H=1, W=1):
    """
    将一个个window还原成一个feature map
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    if len(window_size) == 3:
        B = int(windows.shape[0] / (T * H * W / window_size[0] / window_size[1] / window_size[2]))
        # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
        x = windows.view(B, T // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1)
        # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
        # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, T, H, W, -1)
    elif len(window_size) == 2:
        B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
        # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
        x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
        # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
        # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class ScaleOffset(nn.Module):
    def __init__(self, dim, scale=True, offset=True) -> None:
        super().__init__()
        if scale:
            self.gamma = nn.Parameter(torch.zeros(dim))
            nn.init.normal_(self.gamma, std=.02)
        else:
            self.gamma = None
        if offset:
            self.beta = nn.Parameter(torch.zeros(dim))
        else:
            self.beta = None
    
    def forward(self, input):
        if self.gamma is not None:
            output = input * self.gamma
        else:
            output = input
        if self.beta is not None:
            output = output + self.beta
        else:
            output = output
        
        return output


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, patch_size=[1, 1, 1], in_c=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_c
        self.embed_dim = embed_dim
        if len(patch_size) == 2:
            self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        elif len(patch_size) == 3:
            self.proj = nn.Conv3d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        if len(self.patch_size) == 3:
            _, _, T, H, W = x.shape
        elif len(self.patch_size) == 2:
            _, _, H, W = x.shape

        # 下采样patch_size倍
        x = self.proj(x)
        # _, _, T, H, W = x.shape
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        if len(self.patch_size) == 3:
            return x, T//self.patch_size[-3], H//self.patch_size[-2], W//self.patch_size[-1]
        elif len(self.patch_size) == 2:
            return x, 1, H//self.patch_size[-2], W//self.patch_size[-1]


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SElayer(nn.Module):
    def __init__(self, dim, reduction=4) -> None:
        super().__init__()

        hidden_dim = dim // reduction
        self.channel_conv1 = nn.Conv2d(dim, hidden_dim, 1, 1, 0)
        self.act1 = nn.ReLU()
        self.channel_conv2 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        y = torch.mean(x, dim=[2, 3], keepdim=True)
        y = self.channel_conv1(y)
        y = self.act1(y)
        y = self.channel_conv2(y)
        x = x*self.act2(y)
        return x



class PeriodicPad2d(nn.Module):
    """ 
        pad longitudinal (left-right) circular 
        and pad latitude (top-bottom) with zeros
    """
    def __init__(self, pad_width):
       super(PeriodicPad2d, self).__init__()
       self.pad_width = to_2tuple(pad_width)

    def forward(self, x):
        # pad left and right circular
        out = F.pad(x, (self.pad_width[1], self.pad_width[1], 0, 0), mode="circular") 
        # pad top and bottom zeros
        out = F.pad(out, (0, 0, self.pad_width[0], self.pad_width[0]), mode="constant", value=0) 
        return out




class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            GroupNorm(in_channels),
            Swish(),
            PeriodicPad2d(1),
            nn.Conv2d(in_channels, out_channels, 3, 1, 0),
            GroupNorm(out_channels),
            Swish(),
            PeriodicPad2d(1),
            nn.Conv2d(out_channels, out_channels, 3, 1, 0)
        )
        if in_channels != out_channels:
            self.channel_up = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        if self.in_channels != self.out_channels:
            return self.block(x) + self.channel_up(x)
        else:
            return x + self.block(x)


class UpSampleBlock(nn.Module):
    def __init__(self, channels):
        super(UpSampleBlock, self).__init__()
        self.up = nn.ConvTranspose2d(channels, channels, kernel_size=(2, 2), stride=(2, 2))
        self.ppad = PeriodicPad2d(1)
      
        self.conv = nn.Conv2d(channels, channels, 3, 1, 0)

    def forward(self, x):
        # x = F.interpolate(x, scale_factor=2.)
        x = self.up(x)
        x = self.ppad(x)
        return self.conv(x)


class DownSampleBlock(nn.Module):
    def __init__(self, channels):
        super(DownSampleBlock, self).__init__()
        self.ppad = PeriodicPad2d(1)
        self.conv = nn.Conv2d(channels, channels, 3, 2, 0)

    def forward(self, x):
        x = self.ppad(x)
        return self.conv(x)


class NonLocalBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = GroupNorm(in_channels)
        self.q = torch.nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, 1, 1, 0)

    def forward(self, x):
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape

        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w)

        attn = torch.bmm(q, k)
        attn = attn * (int(c) ** (-0.5))
        attn = F.softmax(attn, dim=2)

        attn = attn.permute(0, 2, 1)
        A = torch.bmm(v, attn)
        A = A.reshape(b, c, h, w)

        A = self.proj_out(A)

        return x + A


class PatchExpand(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H, W, C
        """
        x = self.expand(x)
        B, H, W, C = x.shape

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        # x = x.view(B,-1,C//4)
        x= self.norm(x)

        return x



class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, window_length, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.window_length = window_length
        if window_length == 3:
            self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(8 * dim)
        elif window_length == 2:
            self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(4 * dim)

    def forward(self, x, T, H, W):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        if self.window_length == 3:
            assert L == T * H * W, "input feature has wrong size"
            if T > 1:
                x = x.view(B, T, H, W, C)
            else:
                x = x.view(B, H, W, C)
        elif self.window_length == 2:
            assert L == H * W, "input feature has wrong size"
            x = x.view(B, H, W, C)

        if len(x.shape) == 5:
            x0 = x[:, 0::2, 0::2, 0::2, :]  # [B, H/2, W/2, C]
            x1 = x[:, 1::2, 0::2, 0::2, :]  # [B, H/2, W/2, C]
            x2 = x[:, 0::2, 1::2, 0::2, :]  # [B, H/2, W/2, C]
            x3 = x[:, 1::2, 1::2, 0::2, :]  # [B, H/2, W/2, C]
            x4 = x[:, 0::2, 0::2, 1::2, :]  # [B, H/2, W/2, C]
            x5 = x[:, 1::2, 0::2, 1::2, :]  # [B, H/2, W/2, C]
            x6 = x[:, 0::2, 1::2, 1::2, :]  # [B, H/2, W/2, C]
            x7 = x[:, 1::2, 1::2, 1::2, :]  # [B, H/2, W/2, C]
            x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # [B, H/2, W/2, 4*C]
            x = x.view(B, -1, 8 * C)  # [B, H/2*W/2, 4*C]
        elif len(x.shape) == 4:
            x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
            x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
            x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
            x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
            x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
            x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4*C]
        else:
            print(x.shape)

        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2*W/2, 2*C]

        return x


def round_to(dat, c):
    return dat + (dat - dat % c) % c

def get_activation(act, inplace=False, **kwargs):
    """

    Parameters
    ----------
    act
        Name of the activation
    inplace
        Whether to perform inplace activation

    Returns
    -------
    activation_layer
        The activation
    """
    if act is None:
        return lambda x: x
    if isinstance(act, str):
        if act == 'leaky':
            negative_slope = kwargs.get("negative_slope", 0.1)
            return nn.LeakyReLU(negative_slope, inplace=inplace)
        elif act == 'identity':
            return nn.Identity()
        elif act == 'elu':
            return nn.ELU(inplace=inplace)
        elif act == 'gelu':
            return nn.GELU()
        elif act == 'relu':
            return nn.ReLU()
        elif act == 'sigmoid':
            return nn.Sigmoid()
        elif act == 'tanh':
            return nn.Tanh()
        elif act == 'softrelu' or act == 'softplus':
            return nn.Softplus()
        elif act == 'softsign':
            return nn.Softsign()
        else:
            raise NotImplementedError('act="{}" is not supported. '
                                      'Try to include it if you can find that in '
                                      'https://pytorch.org/docs/stable/nn.html'.format(act))
    else:
        return act

class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        """Root Mean Square Layer Normalization proposed in "[NeurIPS2019] Root Mean Square Layer Normalization"

        Parameters
        ----------
        d
            model size
        p
            partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        eps
            epsilon value, default 1e-8
        bias
            whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed

def get_norm_layer(normalization: str = 'layer_norm',
                   axis: int = -1,
                   epsilon: float = 1e-5,
                   in_channels: int = 0, **kwargs):
    """Get the normalization layer based on the provided type

    Parameters
    ----------
    normalization
        The type of the layer normalization from ['layer_norm']
    axis
        The axis to normalize the
    epsilon
        The epsilon of the normalization layer
    in_channels
        Input channel

    Returns
    -------
    norm_layer
        The layer normalization layer
    """
    if isinstance(normalization, str):
        if normalization == 'layer_norm':
            assert in_channels > 0
            assert axis == -1
            norm_layer = nn.LayerNorm(normalized_shape=in_channels, eps=epsilon, **kwargs)
        elif normalization == 'rms_norm':
            assert axis == -1
            norm_layer = RMSNorm(d=in_channels, eps=epsilon, **kwargs)
        else:
            raise NotImplementedError('normalization={} is not supported'.format(normalization))
        return norm_layer
    elif normalization is None:
        return nn.Identity()
    else:
        raise NotImplementedError('The type of normalization must be str')


def _generalize_padding(x, pad_t, pad_h, pad_w, padding_type, t_pad_left=False):
    """

    Parameters
    ----------
    x
        Shape (B, T, H, W, C)
    pad_t
    pad_h
    pad_w
    padding_type
    t_pad_left

    Returns
    -------
    out
        The result after padding the x. Shape will be (B, T + pad_t, H + pad_h, W + pad_w, C)
    """
    if pad_t == 0 and pad_h == 0 and pad_w == 0:
        return x

    assert padding_type in ['zeros', 'ignore', 'nearest']
    B, T, H, W, C = x.shape

    if padding_type == 'nearest':
        return F.interpolate(x.permute(0, 4, 1, 2, 3), size=(T + pad_t, H + pad_h, W + pad_w)).permute(0, 2, 3, 4, 1)
    else:
        if t_pad_left:
            return F.pad(x, (0, 0, 0, pad_w, 0, pad_h, pad_t, 0))
        else:
            return F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_t))


def _generalize_unpadding(x, pad_t, pad_h, pad_w, padding_type):
    assert padding_type in['zeros', 'ignore', 'nearest']
    B, T, H, W, C = x.shape
    if pad_t == 0 and pad_h == 0 and pad_w == 0:
        return x

    if padding_type == 'nearest':
        return F.interpolate(x.permute(0, 4, 1, 2, 3), size=(T - pad_t, H - pad_h, W - pad_w)).permute(0, 2, 3, 4, 1)
    else:
        return x[:, :(T - pad_t), :(H - pad_h), :(W - pad_w), :].contiguous()

def apply_initialization(m,
                         linear_mode="0",
                         conv_mode="0",
                         norm_mode="0",
                         embed_mode="0"):
    if isinstance(m, nn.Linear):

        if linear_mode in ("0", ):
            nn.init.kaiming_normal_(m.weight,
                                    mode='fan_in', nonlinearity="linear")
        elif linear_mode in ("1", ):
            nn.init.kaiming_normal_(m.weight,
                                    a=0.1,
                                    mode='fan_out',
                                    nonlinearity="leaky_relu")
        else:
            raise NotImplementedError
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        if conv_mode in ("0", ):
            nn.init.kaiming_normal_(m.weight,
                                    a=0.1,
                                    mode='fan_out',
                                    nonlinearity="leaky_relu")
        else:
            raise NotImplementedError
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        if norm_mode in ("0", ):
            if m.elementwise_affine:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        else:
            raise NotImplementedError
    elif isinstance(m, nn.GroupNorm):
        if norm_mode in ("0", ):
            if m.affine:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        else:
            raise NotImplementedError
    # # pos_embed already initialized when created
    elif isinstance(m, nn.Embedding):
        if embed_mode in ("0", ):
            nn.init.trunc_normal_(m.weight.data, std=0.02)
        else:
            raise NotImplementedError
    else:
        pass