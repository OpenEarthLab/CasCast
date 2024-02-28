# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm, mmseg, setr, xcit and swin code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/fudan-zvg/SETR
# https://github.com/facebookresearch/xcit/
# https://github.com/microsoft/Swin-Transformer
# --------------------------------------------------------'
import math
import os
import torch
import numpy as np
from functools import partial

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as checkpoint_train
from einops import rearrange
# try:
#     import spring.linklink as link
# except:
#     import linklink as link

from timm.models.layers import drop_path, to_2tuple, trunc_normal_

# from core.utils import NestedTensor
# from ..ckpt import checkpoint_wrapper
from dict_recursive_update import recursive_update


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self):
        return 'p={}'.format(self.drop_prob)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, window_size=None, rel_pos_spatial=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.rel_pos_spatial = rel_pos_spatial
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.window_size = window_size
        if COMPAT:
            if COMPAT == 2:
                self.rel_pos_h = nn.Parameter(torch.zeros(2 * window_size[0] - 1, head_dim))
                self.rel_pos_w = nn.Parameter(torch.zeros(2 * window_size[1] - 1, head_dim))
            else:
                q_size = window_size[0]
                kv_size = q_size
                rel_sp_dim = 2 * q_size - 1
                self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
                self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = ((q * self.scale) @ k.transpose(-2, -1))
        if self.rel_pos_spatial:
            raise
            attn = calc_rel_pos_spatial(attn, q, self.window_size, self.window_size, self.rel_pos_h, self.rel_pos_w)

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def calc_rel_pos_spatial(
        attn,
        q,
        q_shape,
        k_shape,
        rel_pos_h,
        rel_pos_w,
):
    """
    Spatial Relative Positional Embeddings.
    Source: https://github.com/facebookresearch/mvit/
    """
    sp_idx = 0
    q_h, q_w = q_shape
    k_h, k_w = k_shape

    # Scale up rel pos if shapes for q and k are different.
    q_h_ratio = max(k_h / q_h, 1.0)
    k_h_ratio = max(q_h / k_h, 1.0)
    dist_h = (torch.arange(q_h)[:, None] * q_h_ratio - torch.arange(k_h)[None, :] * k_h_ratio)
    dist_h += (k_h - 1) * k_h_ratio
    q_w_ratio = max(k_w / q_w, 1.0)
    k_w_ratio = max(q_w / k_w, 1.0)
    dist_w = (torch.arange(q_w)[:, None] * q_w_ratio - torch.arange(k_w)[None, :] * k_w_ratio)
    dist_w += (k_w - 1) * k_w_ratio

    Rh = rel_pos_h[dist_h.long()]
    Rw = rel_pos_w[dist_w.long()]

    B, n_head, q_N, dim = q.shape

    r_q = q[:, :, sp_idx:].reshape(B, n_head, q_h, q_w, dim)
    rel_h = torch.einsum("byhwc,hkc->byhwk", r_q, Rh)
    rel_w = torch.einsum("byhwc,wkc->byhwk", r_q, Rw)

    attn[:, :, sp_idx:, sp_idx:] = (
            attn[:, :, sp_idx:, sp_idx:].view(B, -1, q_h, q_w, k_h, k_w)
            + rel_h[:, :, :, :, :, None]
            + rel_w[:, :, :, :, None, :]
    ).view(B, -1, q_h * q_w, k_h * k_w)

    return attn


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, rel_pos_spatial=False):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.rel_pos_spatial=rel_pos_spatial

        if COMPAT:
            q_size = window_size[0]
            kv_size = window_size[1]
            rel_sp_dim = 2 * q_size - 1
            self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, H, W):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        x = x.reshape(B_, H, W, C)
        pad_l = pad_t = 0
        pad_r = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_b = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]

        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        x = window_partition(x, self.window_size[0])  # nW*B, window_size, window_size, C
        x = x.view(-1, self.window_size[1] * self.window_size[0], C)  # nW*B, window_size*window_size, C

        B_w = x.shape[0]
        N_w = x.shape[1]
        qkv = self.qkv(x).reshape(B_w, N_w, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)   --> (batchsize, heads, len, head_dim)

        attn = ((q * self.scale) @ k.transpose(-2, -1))
        if self.rel_pos_spatial:
            raise

        attn = attn.softmax(dim=-1)
        _attn_mask = (torch.isinf(attn) + torch.isnan(attn))
        attn = attn.masked_fill(_attn_mask, 0)

        x = (attn @ v).transpose(1, 2).reshape(B_w, N_w, C)
        x = self.proj(x)

        x = x.view(-1, self.window_size[1], self.window_size[0], C)
        x = window_reverse(x, self.window_size[0], Hp, Wp)  # B H' W' C

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B_, H * W, C)

        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, window=False, rel_pos_spatial=False, prompt=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if not window:
            self.attn = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias,
                window_size=window_size, rel_pos_spatial=rel_pos_spatial)
        else:
            self.attn = WindowAttention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias,
                window_size=window_size, rel_pos_spatial=rel_pos_spatial
            )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, x, H, W, mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])  # could be dynamic
        self.num_patches = self.patch_shape[0] * self.patch_shape[1]  # could be dynamic
        self.img_size = img_size
        self.patch_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, mask=None, **kwargs):
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]

        x = x.flatten(2).transpose(1, 2)

        if mask is not None:
            mask = F.interpolate(mask[None].float(), size=(Hp, Wp)).to(torch.bool)[0]

        return x, (Hp, Wp), mask


class Norm2d(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class _ViT(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, out_chans=227, num_classes=80, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False,
                 drop_path_rate=0., norm_layer=None, window=True,
                 use_abs_pos_emb=False, interval=3, bn_group=None, test_pos_mode='simple_interpolate',time_embed=False,
                 learnable_pos=False, rel_pos_spatial=False, lms_checkpoint_train=False,
                 prompt=None, pad_attn_mask=False, freeze_iters=0,
                 act_layer='GELU', pre_ln=False, mask_input=False, ending_norm=True,
                 round_padding=False, compat=False):
        super().__init__()
        self.pad_attn_mask = pad_attn_mask  # only effective for detection task input w/ NestedTensor wrapping
        self.lms_checkpoint_train = lms_checkpoint_train
        self.freeze_iters = freeze_iters
        self.mask_input = mask_input
        self.ending_norm = ending_norm
        self.round_padding = round_padding
        self.patch_size = patch_size
        self.img_size = img_size
        self.time_embed = time_embed


        self.ori_Hp, self.ori_Hw = img_size[0] // patch_size[0], \
                                   img_size[1] // patch_size[1]

        global COMPAT
        COMPAT = compat

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        
        num_patches = self.patch_embed.num_patches

        if use_abs_pos_emb:
            if self.time_embed:
                self.pos_embed = nn.Parameter(torch.zeros(24, num_patches, embed_dim), requires_grad=learnable_pos)
            else:
                self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=learnable_pos)

            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.patch_shape, cls_token=False)

            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        else:
            raise

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop_path=dpr[i], norm_layer=norm_layer,
                window_size=(14, 14) if ((i + 1) % interval != 0) else self.patch_embed.patch_shape,
                window=((i + 1) % interval != 0) if window else False,
                rel_pos_spatial=rel_pos_spatial, prompt=prompt,
                act_layer=QuickGELU if act_layer == 'QuickGELU' else nn.GELU
            )
            self.blocks.append(block)

        self.ln_pre = norm_layer(embed_dim) if pre_ln else nn.Identity()  # for clip model only
        self.norm = norm_layer(embed_dim)

        self.final = nn.Linear(embed_dim, out_chans*patch_size[-1]*patch_size[-2], bias=False)

        ### duplicated init, only affects network weights and has no effect given pretrain
        self.apply(self._init_weights)
        self.fix_init_weight()
        ###
        self.test_pos_mode = test_pos_mode


    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def _normalization(x):
        assert len(x.shape) == 4
        x = x.sub(torch.tensor([123.675, 116.280, 103.530]).view(1, 3, 1, 1).cuda()).div(torch.tensor([58.395, 57.120, 57.375]).view(1, 3, 1, 1).cuda())
        return x

    def get_num_layers(self):
        return len(self.blocks)

    def forward_features(self, x, mask=None):
        B, C, H, W = x.shape
            
        x, (Hp, Wp), mask = self.patch_embed(x, mask)
        batch_size, seq_len, _ = x.size()
        if self.test_pos_mode is False:
            if x.size(1) == self.pos_embed.size(1):
                x = x + self.pos_embed  # BxHWxC
            else: # take top-left if pos_embed > x's dimension
                x = x + self.pos_embed.reshape(1, self.patch_embed.patch_shape[0],
                                               self.patch_embed.patch_shape[1],
                                               self.pos_embed.size(2))[:,:Hp, :Wp, :].reshape(1, x.size(1),
                                                                                              self.pos_embed.size(2))
        elif self.test_pos_mode == 'learnable_simple_interpolate':
            patch_shape = (Hp, Wp)
            pos_embed = self.pos_embed
            x = x + get_abs_pos(pos_embed, False, (self.ori_Hp, self.ori_Hw), patch_shape)
        else:
            raise NotImplementedError

        x = self.ln_pre(x)  # effective for clip model only, otherwise nn.Identity

        for i, blk in enumerate(self.blocks):
            # *Warning*: official ckpt implementation leads to NaN loss in many cases, use fairscale if that's the case
            if self.lms_checkpoint_train == True:
                x = checkpoint_train(lambda x: blk(x, Hp, Wp, mask), x, preserve_rng_state=True)
            else:
                x = blk(x, Hp, Wp)
        if self.ending_norm:
            x = self.norm(x)  # b h*w c

        x = self.final(x)
        x = x.view(B, Hp, Wp,-1)
        res = rearrange(
            x,
            "b h w (p1 p2 c_out) -> b c_out (h p1) (w p2)",
            p1=self.patch_size[-2],
            p2=self.patch_size[-1],
            h=self.img_size[0] // self.patch_size[-2],
            w=self.img_size[1] // self.patch_size[-1],
        )
        return res

    def forward(self, input_var, **kwargs):
        output = {}

        x = input_var  #['image']
        _, _, C, _, _ = x.shape
        ## (b, t, c, h, w) -> (b, (t c), h, w)
        x = rearrange(x, 'b t c h w -> b (t c) h w')
        # if self.time_embed:
        #     hour = kwargs['hour_label']
        #     month =  kwargs['month_label']
        # else:
        #     hour=None
        #     month=None
        
        if self.round_padding:
            stride = self.patch_embed.patch_size
            assert stride[0] == stride[1]
            stride = max(stride[0], self.round_padding)
            output["prepad_input_size"] = [x.shape[-2], x.shape[-1]]  # h, w for sem_seg_postprocess
            target_size = (torch.tensor((x.shape[-1], x.shape[-2])) + (stride - 1)).div(stride, rounding_mode="floor") * stride  # w, h
            padding_size = [  # [l,r,t,b]
                0,
                target_size[0] - x.shape[-1],
                0,
                target_size[1] - x.shape[-2],
                ]
            x = F.pad(x, padding_size, value=0.).contiguous()

        output = self.forward_features(x)
        ## (b, (t c), h, w) -> (b, t, c, h, w)
        output = rearrange(output, 'b (t c) h w -> b t c h w', c=C)
        return output

    def init_weights(self, pretrained='',):
        import os
        if os.path.isfile(pretrained):
            if pretrained.endswith('.tar'):
                pretrained_dict = torch.load(pretrained)['state_dict']
                print('=> loading pretrained model {}'.format(pretrained))
                model_dict = self.state_dict()
                pretrained_dict_filter ={}
                for k, v in pretrained_dict.items():
                    # import pdb
                    # pdb.set_trace()
                    if k[23:] in model_dict.keys() and "pos_embed" not in k:
                        pretrained_dict_filter.update({k[23:]: v})

                #for k, _ in pretrained_dict.items():
                #    logger.info(
                #        '=> loading {} pretrained model {}'.format(k, pretrained))
            elif pretrained.endswith('.pth'):
                pretrained_dict = torch.load(pretrained)['model']#
                print('=> loading pretrained model {}'.format(pretrained))
                model_dict = self.state_dict()


                pretrained_dict_filter ={}
                for k, v in pretrained_dict.items():
                    # import pdb
                    # pdb.set_trace()
                    if k in model_dict.keys() and "pos_embed" not in k and "patch_embed" not in k:
                        pretrained_dict_filter.update({k: v})
            print(
                "Missing keys: {}".format(list(set(model_dict) - set(pretrained_dict_filter)
                                               )))
            model_dict.update(pretrained_dict_filter)
            # import pdb
            # pdb.set_trace()
            self.load_state_dict(model_dict)



# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_size = to_2tuple(grid_size)
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    # import pdb
    # pdb.set_trace()
    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_abs_pos(abs_pos, has_cls_token, ori_hw, hw):
    """
    Calculate absolute positional embeddings. If needed, resize embeddings and remove cls_token
        dimension for the original embeddings.
    Args:
        abs_pos (Tensor): absolute positional embeddings with (1, num_position, C).
        has_cls_token (bool): If true, has 1 embedding in abs_pos for cls token.
        hw (Tuple): size of input image tokens.
    Returns:
        Absolute positional embeddings after processing with shape (1, H, W, C)
    """
    embed_num, _, emde_dim = abs_pos.size()
    h, w = hw
    if has_cls_token:
     abs_pos = abs_pos[:, 1:]
    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))

    ori_hp, ori_hw = ori_hw

    # import  pdb
    # pdb.set_trace()
    assert ori_hp, ori_hw == xy_num

    if ori_hp != h or ori_hw != w:
        new_abs_pos = F.interpolate(
            abs_pos.reshape(embed_num, ori_hp, ori_hw, -1).permute(0, 3, 1, 2),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )

        return new_abs_pos.permute(0, 2, 3, 1).reshape(embed_num, h*w, -1)
    else:
        return abs_pos.reshape(embed_num, h*w, -1)


# @BACKBONES.register_module()
# class MAEvitBackbone(nn.Module):
class ViT(nn.Module):
    def __init__(self, arch='vit_custom', patch_size=(16,16), in_chans=227, out_chans=227,
                learnable_pos= True, window= True, drop_path_rate= 0., round_padding= True,
                pad_attn_mask= True , test_pos_mode= 'learnable_simple_interpolate', # to_do: ablation
                lms_checkpoint_train= False, img_size= (128,256), **kwargs):
        super().__init__()
        backbone_kwargs = {'learnable_pos':learnable_pos, 'window':window, 'drop_path_rate':drop_path_rate,
                'round_padding': round_padding, 'pad_attn_mask': pad_attn_mask , 'test_pos_mode': test_pos_mode, # to_do: ablation
                  'lms_checkpoint_train': lms_checkpoint_train, 'img_size': img_size, 'patch_size': patch_size,
                  'in_chans': in_chans, 'out_chans': out_chans, 'norm_layer': partial(nn.LayerNorm, eps=1e-6),}
        # base_default_dict =dict(
        #     drop_path_rate=0, use_abs_pos_emb=True,  # as in table 11
        #     patch_size=patch_size, in_chans=in_chans, out_chans=out_chans, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        #     norm_layer=partial(nn.LayerNorm, eps=1e-6),
        #     # learnable_pos= True,
        # )

        # large_default_dict =dict(
        #     drop_path_rate=0, use_abs_pos_emb=True,  # as in table 11
        #     patch_size=patch_size,in_chans=in_chans, out_chans=out_chans, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        #     norm_layer=partial(nn.LayerNorm, eps=1e-6),
        #     # learnable_pos= True,
        # )

        # huge_default_dict =dict(
        #     drop_path_rate=0, use_abs_pos_emb=True,  # as in table 11
        #     patch_size=patch_size,in_chans=in_chans, out_chans=out_chans, embed_dim=2048, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        #     norm_layer=partial(nn.LayerNorm, eps=1e-6),
        #     # learnable_pos= True,
        # )

        # if arch == "vit_small":
        #     recursive_update(base_default_dict, kwargs)
        #     self.net = ViT(**base_default_dict)

        # elif arch == "vit_base":
        #     recursive_update(large_default_dict, kwargs)
        #     self.net = ViT(**large_default_dict)

        # elif arch == "vit_large":
        #     recursive_update(huge_default_dict, kwargs)
        #     self.net = ViT(**huge_default_dict)
        if arch == "vit_custom":
            custom_dict = kwargs.get('arch_dict', {})
            recursive_update(custom_dict, backbone_kwargs)
            self.net = _ViT(**custom_dict)

        else:
            raise Exception("Architecture undefined!")

    def forward(self, input, **kwargs):
        return  self.net(input, **kwargs)



if __name__ == '__main__':
    print('start')
    b = 16
    inp_length, pred_length = 13, 12
    total_length = inp_length + pred_length
    c = 4
    h, w = 48, 48
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') #
    input_data = torch.randn((b, inp_length, c, h, w)).to(device) #torch.randn((b, inp_length, c, h, w)).to(device)
    target = torch.randn((b, pred_length, c, h, w)).to(device) #
    # backbone_kwargs = {
    #     'arch': 'vit_custom',
    #     'patch_size': [4, 4],
    #     'in_chans': inp_length,
    #     'out_chans': pred_length,
    #     'learnable_pos': True,
    #     'window': True,
    #     'drop_path_rate': 0.,
    #     'round_padding': True,
    #     'pad_attn_mask': True , # to_do: ablation
    #     'test_pos_mode': 'learnable_simple_interpolate', # to_do: ablation
    #     'lms_checkpoint_train': False,
    #     'img_size': [h,w],
    #     'arch_dict':{
    #         'use_abs_pos_emb': True,
    #         'embed_dim': 256,
    #         'depth': 12,
    #         'num_heads': 8,
    #         'mlp_ratio':4,
    #         'qkv_bias':True
    #     }
    # }
    print('load yaml from config')
    import yaml
    cfg_path = '/mnt/cache/gongjunchao/workdir/radar_forecasting/configs/sevir/latent_vit.yaml'
    with open(cfg_path, 'r') as cfg_file:
      cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)
    backbone_kwargs = cfg_params['model']['params']['sub_model']['vit']

    print('end')
    model = ViT(**backbone_kwargs)
    model.to(device)
    # import pdb; pdb.set_trace()
    import torch.nn.functional as F
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for i in range(5):
        start.record()
        pred = model(input_data)
        loss = F.mse_loss(pred, target)
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is None:
                print(f'{n} has no grad')
        end.record()
        torch.cuda.synchronize()
        print(start.elapsed_time(end))
        memory = torch.cuda.memory_reserved() / (1024. * 1024)
        print("memory:", memory)
        
    from fvcore.nn.parameter_count import parameter_count_table
    from fvcore.nn.flop_count import flop_count
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    flops = FlopCountAnalysis(model, input_data)
    print(flop_count_table(flops))

## srun -p ai4earth --quotatype=spot --ntasks-per-node=1  --cpus-per-task=4 --gres=gpu:1 python -u ViT.py ##