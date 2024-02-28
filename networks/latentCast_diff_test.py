# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.append('/mnt/cache/gongjunchao/workdir/radar_forecasting')
import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from einops import rearrange
from networks.ldm.ldm_attention import (default, _ATTN_PRECISION, exists, einsum, repeat)



def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


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

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
            # dim,
            # num_heads=8,
            # qkv_bias=False,
            # qk_norm=False,
            # attn_drop=0.,
            # proj_drop=0.,
            # norm_layer=nn.LayerNorm,
            # flash_attention=True,
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION =="fp32":
            with torch.autocast(enabled=False, device_type = 'cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        
        del q, k
    
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


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

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class CrossBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, cross_heads, split_num,
                mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = CrossAttention(query_dim=hidden_size, context_dim=hidden_size, heads=cross_heads, 
                                         dim_head=hidden_size//cross_heads, dropout=0)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size//split_num * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        # self.latent2enc_mlp = Mlp(in_features=hidden_size, hidden_features=hidden_size*mlp_ratio,
        #                           out_features=hidden_size, act_layer=approx_gelu, drop=0)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, latent_x, enc_x, c):
        """
        latent_x: [b, l, t*d]
        enc_x: [b, t, l, d]
        c: [b, t*d]
        """
        ## TODO warning: maybe loss many infomation 
        # import pdb; pdb.set_trace()
        x = rearrange(enc_x, 'b t l d -> b l (t d)')
        # latent_x = self.latent2enc_mlp(latent_x)
        # latent_x = rearrange(latent_x, '-> ... c s')
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.cross_attn(x=modulate(self.norm1(x), shift_msa, scale_msa), context=latent_x)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
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
        split_num=12,
        enc_depth=6,
        latent_depth=12,
        dec_depth=6,
        num_heads=16,
        mlp_ratio=4.0,
        out_channels = 1,
        # class_dropout_prob=0.1,
        # num_classes=1000,
        learn_sigma=False,
        single_heads_num=1,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        # self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.split_num = split_num
        self.single_heads_num = single_heads_num

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size//split_num, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size//split_num)
        self.aggr_t_embedder = TimestepEmbedder(hidden_size)
        # self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size//split_num), requires_grad=False)

        
        self.encoder_blocks = nn.ModuleList([
            DiTBlock(hidden_size//split_num, single_heads_num, mlp_ratio=mlp_ratio) for _ in range(enc_depth)
        ])
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(latent_depth)
        ])
        self.distribute_block = CrossBlock(hidden_size, cross_heads=num_heads, split_num=split_num, mlp_ratio=mlp_ratio)
        self.decoder_blocks = nn.ModuleList([
            DiTBlock(hidden_size//split_num, single_heads_num, mlp_ratio=mlp_ratio) for _ in range(dec_depth)
        ])
        self.final_layer = FinalLayer(hidden_size//split_num, patch_size, self.out_channels)
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

        nn.init.normal_(self.aggr_t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.aggr_t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.encoder_blocks+self.blocks+self.decoder_blocks:
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

    def forward(self, x, t):
        """
        Forward pass of DiT.
        x: (N, T, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        """
        B, T, _, _, _ = x.shape
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.x_embedder(x) + self.pos_embed  # (N*split_num, T, D), where T = H * W / patch_size ** 2

        ## embeding timesteps ##
        sub_t = self.t_embedder(t)                   # (N, D)
        D = sub_t.shape[-1]
        sub_t = sub_t.unsqueeze(1).expand(B, T, D) #(N*split_num, D)
        sub_t = rearrange(sub_t, 'b t d -> (b t) d').contiguous()
        aggr_t = self.aggr_t_embedder(t)
        ### enc ###
        # import pdb; pdb.set_trace()
        enc_x = x
        for block in self.encoder_blocks:
            enc_x = block(enc_x, sub_t)
        ### latent ###         
        # import pdb; pdb.set_trace()
        latent_x = rearrange(enc_x, '(b t) l d -> b l (t d)', b=B).contiguous()
        for block in self.blocks:
            latent_x = block(latent_x, aggr_t)
        ## distribute latent ##
        # import pdb; pdb.set_trace()
        enc_x = rearrange(enc_x, '(b t) l d -> b t l d', b=B).contiguous()
        # import pdb; pdb.set_trace()
        dec_x = self.distribute_block(latent_x, enc_x, aggr_t)
        dec_x = rearrange(dec_x, 'b l (t d) -> (b t) l d', t=T).contiguous() ##TODO: maybe add linear map before rearrange()
        ### dec ###
        for block in self.decoder_blocks:
            dec_x = block(dec_x, sub_t)         
        # import pdb; pdb.set_trace()                                  
        out = self.final_layer(dec_x, sub_t)                # (N, T, patch_size ** 2 * out_channels)
        out = self.unpatchify(out)                   # (N, out_channels, H, W)
        out = rearrange(out, '(b t) c h w -> b t c h w', b=B)  # (N, T, out_channels, H, W)
        return out

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

# def DiT_XL_2(**kwargs):
#     return _DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

# def DiT_XL_4(**kwargs):
#     return _DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

# def DiT_XL_8(**kwargs):
#     return _DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

# def DiT_L_2(**kwargs):
#     return _DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

# def DiT_L_4(**kwargs):
#     return _DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

# def DiT_L_8(**kwargs):
#     return _DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

# def DiT_B_2(**kwargs):
#     return _DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

# def DiT_B_4(**kwargs):
#     return _DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

# def DiT_B_8(**kwargs):
#     return _DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

# def DiT_S_2(**kwargs):
#     return _DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

# def DiT_S_4(**kwargs):
#     return _DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

# def DiT_S_8(**kwargs):
#     return _DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)

def DiT_custom(**kwargs):
    return _DiT(**kwargs)


DiT_models = {
    # 'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    # 'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    # 'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    # 'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
    'DiT-custom': DiT_custom
}

class latentCast_diff_test(nn.Module):
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
        """
        b, t, _, h, w = x.shape
        inp = torch.cat([x, cond], dim=2)
        out = self.model(x=inp, t=timesteps)
        return out
    


if __name__ == "__main__":
    print('start')
    b = 8
    inp_length, pred_length = 12, 12
    c = 4
    h, w = 48, 48
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') #
    input_data = torch.randn((b, inp_length, c, h, w)).to(device) #torch.randn((b, inp_length, c, h, w)).to(device)
    cond = input_data
    target = torch.randn((b, pred_length, c, h, w)).to(device) #
    t = torch.randint(0, 1000, (b,)).to(device)
    ########################################################################
    # print('load yaml from config')
    # import yaml
    # cfg_path = '/mnt/cache/gongjunchao/workdir/radar_forecasting/configs/sevir/unet2d_refine.yaml'
    # with open(cfg_path, 'r') as cfg_file:
    #   cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)
    # backbone_kwargs = cfg_params['model']['params']['sub_model']['unet2d']

    backbone_kwargs = {
        'arch': 'DiT-custom',
        'config': {
        'input_size': h,
        'in_channels': 2*c,
        'mlp_ratio': 4.0,
        'learn_sigma': False,
        'out_channels': c,
        'hidden_size':1152, 'patch_size':2, 'num_heads':16,
        'enc_depth':3, 'dec_depth':3, 'latent_depth': 18, 
        'split_num':12
        }
    }
    model = latentCast_diff_test(**backbone_kwargs)
    model.to(device)
    print('end')

    import torch.nn.functional as F
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for i in range(5):
        start.record()
        pred = model(x=input_data, timesteps=t, cond=cond)
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
    flops = FlopCountAnalysis(model, (input_data, t, cond))
    print(flop_count_table(flops))
# srun -p ai4earth --kill-on-bad-exit=1 --quotatype=auto --gres=gpu:1 python -u latentCast_diff_test.py #