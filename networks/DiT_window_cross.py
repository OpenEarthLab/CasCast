# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from einops import rearrange
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


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

def patch_partition(x, patch_size, x_H, x_W):
    """
    use avg pooling to partition image into patches
    Args:
        x: (B, L, C)
        patch_size (int): patch size
        x_H: height of image after finest patch embedding
        x_W: Width of image after finest patch embedding
    Returns:
        x: (B, L/patch_size**2,  C)
    """
    x = rearrange(x, 'b (h w) c -> b c h w', h=x_H, w=x_W)
    avg_x = F.avg_pool2d(x, kernel_size=patch_size, stride=patch_size)
    avg_x = rearrange(avg_x, 'b c h w -> b (h w) c')
    return avg_x 

def patch_reverse(x, patch_size, x_H, x_W):
    """
    recover fine patch from pooling patch
    Args:
        x: (B, L/patch_size**2,  C)
        patch_size (int): patch size
        x_H: height of image after finest patch embedding
        x_W: Width of image after finest patch embedding
    Returns:
        x: (B, L, C)
    """
    x = rearrange(x, 'b (h w) c -> b c h w', h=x_H//patch_size, w=x_W//patch_size)
    x = F.interpolate(x, size=(x_H, x_W), mode='nearest')
    x = rearrange(x, 'b c h w -> b (h w) c')
    return x


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


# class LabelEmbedder(nn.Module):
#     """
#     Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
#     """
#     def __init__(self, num_classes, hidden_size, dropout_prob):
#         super().__init__()
#         use_cfg_embedding = dropout_prob > 0
#         self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
#         self.num_classes = num_classes
#         self.dropout_prob = dropout_prob

#     def token_drop(self, labels, force_drop_ids=None):
#         """
#         Drops labels to enable classifier-free guidance.
#         """
#         if force_drop_ids is None:
#             drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
#         else:
#             drop_ids = force_drop_ids == 1
#         labels = torch.where(drop_ids, self.num_classes, labels)
#         return labels

#     def forward(self, labels, train, force_drop_ids=None):
#         use_dropout = self.dropout_prob > 0
#         if (train and use_dropout) or (force_drop_ids is not None):
#             labels = self.token_drop(labels, force_drop_ids)
#         embeddings = self.embedding_table(labels)
#         return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, x_H, x_W, patch_size, **kwargs):
        """
        x: (N, L, D)
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)

        x_down = patch_partition(x=x, patch_size=patch_size, x_H=x_H, x_W=x_W)
        attn_x_down = self.attn(modulate(self.norm1(x_down), shift_msa, scale_msa))
        attn_x = patch_reverse(x=attn_x_down, patch_size=patch_size, x_H=x_H, x_W=x_W)
        x = x + gate_msa.unsqueeze(1) * attn_x

        x_down = patch_partition(x=x, patch_size=patch_size, x_H=x_H, x_W=x_W)
        ffn_x = self.mlp(modulate(self.norm2(x_down), shift_mlp, scale_mlp))
        ffn_x = patch_reverse(x=ffn_x, patch_size=patch_size, x_H=x_H, x_W=x_W) 
        x = x + gate_mlp.unsqueeze(1) * ffn_x
        return x

class DiTWindowBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, window_size=24, **block_kwargs):
        super().__init__()
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, x_H, x_W, patch_size=None, **kwargs):
        bs, N, chan = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = modulate(self.norm1(x), shift_msa, scale_msa)
        #########################
        x_window = rearrange(x, 'b (h w) c -> b h w c', h=x_H, w=x_W)
        x_window = window_partition(x_window, self.window_size)
        x_window = x_window.view(-1, self.window_size * self.window_size, chan)
        ## window attn ##
        x_window = self.attn(x_window)
        x_window = x_window.view(-1, self.window_size, self.window_size, chan)
        x = window_reverse(x_window, self.window_size, x_H, x_W)
        x = rearrange(x, 'b h w c -> b (h w) c')
        x = x + gate_msa.unsqueeze(1) * x
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class AgentCrossAttention(nn.Module):
    def __init__(self, dim, inner_dim, num_patches, context_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 sr_ratio=1, agent_num=49, **kwargs):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_patches = num_patches
        window_size = (int(num_patches ** 0.5), int(num_patches ** 0.5))
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = inner_dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, inner_dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(inner_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.agent_token_conv = nn.Conv2d((inner_dim+context_dim), inner_dim, kernel_size=3, padding=1)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.agent_num = agent_num
        self.dwc = nn.Conv2d(in_channels=inner_dim, out_channels=inner_dim, kernel_size=(3, 3), padding=1, groups=inner_dim)
        self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window_size[0] // sr_ratio, 1))
        self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window_size[1] // sr_ratio))
        self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window_size[0], 1, agent_num))
        self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window_size[1], agent_num))
        trunc_normal_(self.an_bias, std=.02)
        trunc_normal_(self.na_bias, std=.02)
        trunc_normal_(self.ah_bias, std=.02)
        trunc_normal_(self.aw_bias, std=.02)
        trunc_normal_(self.ha_bias, std=.02)
        trunc_normal_(self.wa_bias, std=.02)
        self.pool_size = int(agent_num ** 0.5)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(self.pool_size, self.pool_size))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, context, H, W, c_H, c_W):
        b, n, c = x.shape
        num_heads = self.num_heads
        q = self.q(x)
        inner_c = q.shape[-1]
        context_c = context.shape[-1]
        head_dim = inner_c // num_heads

        if self.sr_ratio > 1:
            import pdb; pdb.set_trace()
            x_ = x.permute(0, 2, 1).reshape(b, c, H, W)
            x_ = self.sr(x_).reshape(b, c, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(b, -1, 2, c).permute(2, 0, 1, 3)
        else:
            kv = self.kv(x).reshape(b, -1, 2, inner_c).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]
        pool_x_tokens = self.pool(q.reshape(b, H, W, inner_c).permute(0, 3, 1, 2)).reshape(b, inner_c, -1).permute(0, 2, 1)
        pool_context_tokens = self.pool(context.reshape(b, c_H, c_W, context_c).permute(0, 3, 1, 2)).reshape(b, context_c, -1).permute(0, 2, 1)
        agent_tokens = torch.concat([pool_x_tokens, pool_context_tokens], dim=2).reshape(b, -1, self.pool_size, self.pool_size)
        agent_tokens = self.agent_token_conv(agent_tokens).reshape(b, inner_c, -1) 

        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n // self.sr_ratio ** 2, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n // self.sr_ratio ** 2, num_heads, head_dim).permute(0, 2, 1, 3)
        agent_tokens = agent_tokens.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)

        kv_size = (self.window_size[0] // self.sr_ratio, self.window_size[1] // self.sr_ratio)
        position_bias1 = nn.functional.interpolate(self.an_bias, size=kv_size, mode='bilinear')
        position_bias1 = position_bias1.reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias = position_bias1 + position_bias2
        agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)
        agent_attn = self.attn_drop(agent_attn)
        agent_v = agent_attn @ v

        agent_bias1 = nn.functional.interpolate(self.na_bias, size=self.window_size, mode='bilinear')
        agent_bias1 = agent_bias1.reshape(1, num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)
        agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, num_heads, -1, self.agent_num).repeat(b, 1, 1, 1)
        agent_bias = agent_bias1 + agent_bias2
        q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias)
        q_attn = self.attn_drop(q_attn)
        x = q_attn @ agent_v

        x = x.transpose(1, 2).reshape(b, n, inner_c)
        v = v.transpose(1, 2).reshape(b, H // self.sr_ratio, W // self.sr_ratio, inner_c).permute(0, 3, 1, 2)
        if self.sr_ratio > 1:
            v = nn.functional.interpolate(v, size=(H, W), mode='bilinear')
        x = x + self.dwc(v).permute(0, 2, 3, 1).reshape(b, n, inner_c)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class DiTCrossBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, context_dim=27, num_patches=1, **block_kwargs):
        super().__init__()
        self.context_dim = context_dim
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn_cross = AgentCrossAttention(dim=hidden_size, inner_dim=hidden_size//2, num_patches=num_patches, context_dim=context_dim, num_heads=16,
                                              agent_num= 100,
                                            qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                                            sr_ratio=1) 
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )
    
    def forward(self, x, c, context, c_H, c_W, x_H, x_W, patch_size):
        """
        x: (N, L, D)
        """
        shift_msa, scale_msa, gate_msa, shift_cross_msa, scale_cross_msa, gate_cross_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(9, dim=1)

        x_down = patch_partition(x=x, patch_size=patch_size, x_H=x_H, x_W=x_W)
        attn_x_down = self.attn(modulate(self.norm1(x_down), shift_msa, scale_msa))
        attn_x = patch_reverse(x=attn_x_down, patch_size=patch_size, x_H=x_H, x_W=x_W)
        x = x + gate_msa.unsqueeze(1) * attn_x

        ## agent cross ##
        x_down = patch_partition(x=x, patch_size=patch_size, x_H=x_H, x_W=x_W)
        cross_attn_x_down = self.attn_cross(modulate(self.norm_cross(x_down), shift_cross_msa, scale_cross_msa), context, H=x_H//patch_size, W=x_W//patch_size, c_H=c_H, c_W=c_W)
        cross_attn_x_down = patch_reverse(x=cross_attn_x_down, patch_size=patch_size, x_H=x_H, x_W=x_W)
        x = x + gate_cross_msa.unsqueeze(1) * cross_attn_x_down

        x_down = patch_partition(x=x, patch_size=patch_size, x_H=x_H, x_W=x_W)
        ffn_x = self.mlp(modulate(self.norm2(x_down), shift_mlp, scale_mlp))
        ffn_x = patch_reverse(x=ffn_x, patch_size=patch_size, x_H=x_H, x_W=x_W) 
        x = x + gate_mlp.unsqueeze(1) * ffn_x
        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class MultiscaleBlocks(nn.Module):
    def __init__(self, hidden_size, num_heads, context_dim, num_patches, mlp_ratio=4.0, patch_size=[1, 2, 2, 2], **block_kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.multiscale_blocks = nn.ModuleList([])    
        for ps in patch_size:
            if ps == 1:
                self.multiscale_blocks.append(DiTWindowBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, window_size=24, **block_kwargs))
            else:
                self.multiscale_blocks.append(DiTCrossBlock(hidden_size, num_heads, context_dim=context_dim, mlp_ratio=mlp_ratio,
                                                             num_patches=num_patches//ps**2, **block_kwargs))
        
    def forward(self, x, c, context, x_H, x_W, c_H, c_W):
        for i, block in enumerate(self.multiscale_blocks):
            x= block(x=x, c=c, x_H=x_H, x_W=x_W, context=context, c_H=c_H, c_W=c_W,  patch_size=self.patch_size[i])
        return x

class _DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        block_pair_depth = 6,
        num_heads=16,
        mlp_ratio=4.0,
        out_channels = 1,
        block_patch_size = [2, 2, 2, 1],
        cond_field_downsample_ratio = 2,
        cond_field_channel = 27,
        # class_dropout_prob=0.1,
        # num_classes=1000,
        learn_sigma=False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        # self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        # self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.cond_field_downsample_ratio = cond_field_downsample_ratio
        self.cond_field_hiddensize = cond_field_channel *cond_field_downsample_ratio
        self.blocks = nn.ModuleList([
            MultiscaleBlocks(hidden_size, num_heads, context_dim=self.cond_field_hiddensize, mlp_ratio=mlp_ratio, patch_size=block_patch_size, num_patches=num_patches) for _ in range(block_pair_depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self.cond_field_encoder = nn.Conv2d(cond_field_channel, self.cond_field_hiddensize, kernel_size=3, stride=cond_field_downsample_ratio, padding=1)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # # Initialize label embedding table:
        # nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for blockPair in self.blocks: 
            for block in blockPair.multiscale_blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, context, y=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        c_h, c_w = context.shape[-2:]
        x_H, x_W = x.shape[-2:]
        x_H, x_W = x_H // self.patch_size, x_W // self.patch_size
        c_h = c_h // self.cond_field_downsample_ratio
        c_w = c_w // self.cond_field_downsample_ratio

        context_hid = self.cond_field_encoder(context) ## (b, c, h, w)
        context_hid = rearrange(context_hid, 'b c h w -> b (h w) c')

        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        # y = self.y_embedder(y, self.training)    # (N, D)
        # c = t + y                                # (N, D)
        c = t
        for block in self.blocks:
            x = block(x, c, context_hid, x_H=x_H, x_W=x_W, c_H=c_h, c_W=c_w)                      # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
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
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return _DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return _DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return _DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_1(**kwargs):
    return _DiT(depth=24, hidden_size=1024, patch_size=1, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return _DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return _DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return _DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return _DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return _DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return _DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return _DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return _DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return _DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/1':DiT_L_1, 'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}

class DiT_window(nn.Module):
    def __init__(self, arch, config):
        super().__init__()
        self.arch = arch
        self.config = config
        if arch not in DiT_models:
            raise ValueError(f'Unrecognized DiT model architecture {arch}')
        self.model = DiT_models[arch](**self.config)

    def forward(self, x, timesteps, cond, context=None, **kwargs):
        """
        x: (b, t, c, h, w)
        cond: (b, t, c, h, w)
        context: (b, 3, 9, 30, 30)
        """
        b, t, _, h, w = x.shape
        x = rearrange(x, 'b t c h w -> b (t c) h w')
        cond = rearrange(cond, 'b t c h w -> b (t c) h w')
        context = rearrange(context, 'b t c h w -> b (t c) h w')
        inp = torch.cat([x, cond], dim=1)
        out = self.model(x=inp, t=timesteps, context=context)
        out = rearrange(out, 'b (t c) h w -> b t c h w', t=t)
        return out
    


if __name__ == "__main__":
    print('start')
    b = 8
    inp_length, pred_length = 12, 12
    c = 2
    h, w = 48, 48
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') #
    input_data = torch.randn((b, inp_length, c, h, w)).to(device) #torch.randn((b, inp_length, c, h, w)).to(device)
    cond = input_data
    target = torch.randn((b, pred_length, c, h, w)).to(device) #
    t = torch.randint(0, 1000, (b,)).to(device)
    ####################
    cond_field = torch.randn((b, 3, 9, 30, 30)).to(device)
    ########################################################################
    # print('load yaml from config')
    # import yaml
    # cfg_path = '/mnt/cache/gongjunchao/workdir/radar_forecasting/configs/sevir/unet2d_refine.yaml'
    # with open(cfg_path, 'r') as cfg_file:
    #   cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)
    # backbone_kwargs = cfg_params['model']['params']['sub_model']['unet2d']

    backbone_kwargs = {
        'arch': 'DiT-L/1',
        'config': {
        'input_size': h,
        'in_channels': 2*inp_length*c,
        'mlp_ratio': 4.0,
        'learn_sigma': False,
        'out_channels': inp_length*c,
        'cond_field_downsample_ratio':1,
        'cond_field_channel':27,
        }
    }
    model = DiT_window(**backbone_kwargs)
    model.to(device)
    print('end')

    import torch.nn.functional as F
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for i in range(5):
        start.record()
        pred = model(x=input_data, timesteps=t, cond=cond, context=cond_field)
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
    flops = FlopCountAnalysis(model, (input_data, t, cond, cond_field))
    print(flop_count_table(flops))
# srun -p ai4earth --kill-on-bad-exit=1 --quotatype=auto --gres=gpu:1 python -u DiT_window_cross.py #