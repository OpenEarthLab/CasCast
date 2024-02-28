import torch.nn as nn
import functools
import torch
import numpy as np
from . import layers, layerspp
from einops import rearrange

ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANppGN
get_act = layers.get_act
default_initializer = layers.default_init


def get_sigmas(config):

    T = getattr(config, 'num_classes')

    if config.sigma.sigma_dist == 'geometric':
        return torch.logspace(np.log10(config.sigma.sigma_begin), np.log10(config.sigma.sigma_end),
                              T)

    elif config.sigma.sigma_dist == 'linear':
        return torch.linspace(config.sigma.sigma_begin, config.sigma.sigma_end,
                              T)

    elif config.sigma.sigma_dist == 'cosine':
        t = torch.linspace(T, 0, T+1)/T
        s = 0.008
        f = torch.cos((t + s)/(1 + s) * np.pi/2)**2
        return f[:-1]/f[-1]

    else:
        raise NotImplementedError('sigma distribution not supported')
    
def get_act(config):
  """Get activation functions from the config file."""
  return nn.SiLU()



class SPADE_NCSNpp(nn.Module):
  """NCSN++ model with SPADE normalization"""

  def __init__(self, config):
    super().__init__()
    self.config = config
    self.act = act = get_act(config)
    self.register_buffer('sigmas', get_sigmas(config))
    self.is3d = (config.arch in ["unetmore3d", "unetmorepseudo3d"])
    self.pseudo3d = (config.arch == "unetmorepseudo3d")
    if self.is3d:
      from . import layers3d

    self.channels = channels = config.channels
    self.num_frames = num_frames = config.num_frames
    self.num_frames_cond = num_frames_cond = config.num_frames_cond
    self.n_frames = num_frames

    self.nf = nf = config.ngf*self.num_frames if self.is3d else config.ngf # We must prevent problems by multiplying by num_frames
    ch_mult = config.ch_mult
    self.num_res_blocks = num_res_blocks = config.num_res_blocks
    self.attn_resolutions = attn_resolutions = config.attn_resolutions
    dropout = getattr(config, 'dropout', 0.0)
    resamp_with_conv = True
    self.num_resolutions = num_resolutions = len(ch_mult)
    self.all_resolutions = all_resolutions = [config.image_size // (2 ** i) for i in range(num_resolutions)]

    self.conditional = conditional = getattr(config, 'time_conditional', True)  # noise-conditional
    self.cond_emb = getattr(config, 'cond_emb', False)
    fir = True
    fir_kernel = [1, 3, 3, 1]
    self.skip_rescale = skip_rescale = True
    self.resblock_type = resblock_type = 'biggan'
    self.embedding_type = embedding_type = 'positional'
    init_scale = 0.0
    assert embedding_type in ['fourier', 'positional']

    self.spade_dim = spade_dim = getattr(config, "spade_dim", 128)

    modules = []
    # timestep/noise_level embedding; only for continuous training
    if embedding_type == 'fourier':
      # Gaussian Fourier features embeddings.

      modules.append(layerspp.GaussianFourierProjection(
        embedding_size=nf, scale=16
      ))
      embed_dim = 2 * nf

    elif embedding_type == 'positional':
      embed_dim = nf

    else:
      raise ValueError(f'embedding type {embedding_type} unknown.')

    temb_dim = None

    if conditional:
      modules.append(nn.Linear(embed_dim, nf * 4))
      modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
      nn.init.zeros_(modules[-1].bias)
      modules.append(nn.Linear(nf * 4, nf * 4))
      modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
      nn.init.zeros_(modules[-1].bias)
      temb_dim = nf * 4

      if self.cond_emb:
        modules.append(torch.nn.Embedding(num_embeddings=2, embedding_dim=nf // 2)) # makes it 8 times smaller (16 if ngf=32) since it should be small because there are only two possible values: 
        temb_dim += nf // 2

    if self.pseudo3d:
      conv3x3 = functools.partial(layers3d.ddpm_conv3x3_pseudo3d, n_frames=self.num_frames, act=self.act) # Activation here as per https://arxiv.org/abs/1809.04096
      conv1x1_cond = functools.partial(layers3d.ddpm_conv1x1_pseudo3d, n_frames=self.channels, act=self.act)
    elif self.is3d:
      conv3x3 = functools.partial(layers3d.ddpm_conv3x3_3d, n_frames=self.num_frames)
      conv1x1_cond = functools.partial(layers3d.ddpm_conv1x1_3d, n_frames=self.channels)
    else:
      conv3x3 = layerspp.conv3x3
      conv1x1 = conv1x1_cond = layerspp.conv1x1

    if self.is3d:
      AttnBlock = functools.partial(layers3d.AttnBlockpp3d,
                                    init_scale=init_scale,
                                    skip_rescale=skip_rescale,
                                    n_head_channels=config.n_head_channels,
                                    n_frames=self.num_frames,
                                    act=None) # No activation here as per https://github.com/facebookresearch/TimeSformer/blob/main/timesformer/models/vit.py#L131
    else:
      AttnBlock = functools.partial(layerspp.AttnBlockpp,
                                    init_scale=init_scale,
                                    skip_rescale=skip_rescale, n_head_channels=config.n_head_channels)

    Upsample = functools.partial(layerspp.Upsample,
                                 with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

    Downsample = functools.partial(layerspp.Downsample,
                                   with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

    ResnetBlockDDPM = layerspp.ResnetBlockDDPMppSPADE
    ResnetBlockBigGAN = layerspp.ResnetBlockBigGANppSPADE

    if resblock_type == 'ddpm':
      ResnetBlock = functools.partial(ResnetBlockDDPM,
                                      act=act,
                                      dropout=dropout,
                                      init_scale=init_scale,
                                      skip_rescale=skip_rescale,
                                      temb_dim=temb_dim,
                                      is3d=self.is3d,
                                      pseudo3d=self.pseudo3d,
                                      n_frames=self.num_frames,
                                      num_frames_cond=num_frames_cond,
                                      cond_ch=num_frames_cond*channels,
                                      spade_dim=spade_dim,
                                      act3d=True) # Activation here as per https://arxiv.org/abs/1809.04096

    elif resblock_type == 'biggan':
      ResnetBlock = functools.partial(ResnetBlockBigGAN,
                                      act=act,
                                      dropout=dropout,
                                      fir=fir,
                                      fir_kernel=fir_kernel,
                                      init_scale=init_scale,
                                      skip_rescale=skip_rescale,
                                      temb_dim=temb_dim,
                                      is3d=self.is3d,
                                      pseudo3d=self.pseudo3d,
                                      n_frames=self.num_frames,
                                      num_frames_cond=num_frames_cond,
                                      cond_ch=num_frames_cond*channels,
                                      spade_dim=spade_dim,
                                      act3d=True) # Activation here as per https://arxiv.org/abs/1809.04096

    else:
      raise ValueError(f'resblock type {resblock_type} unrecognized.')

    # Downsampling block

    modules.append(conv3x3(channels*self.num_frames, nf))
    hs_c = [nf]

    in_ch = nf
    for i_level in range(num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(num_res_blocks):
        out_ch = nf * ch_mult[i_level]
        modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
        in_ch = out_ch

        if all_resolutions[i_level] in attn_resolutions:
          modules.append(AttnBlock(channels=in_ch))
        hs_c.append(in_ch)

      if i_level != num_resolutions - 1:
        if resblock_type == 'ddpm':
          modules.append(Downsample(in_ch=in_ch))
        else:
          modules.append(ResnetBlock(down=True, in_ch=in_ch))

        hs_c.append(in_ch)

    # Middle Block
    in_ch = hs_c[-1]
    modules.append(ResnetBlock(in_ch=in_ch))
    modules.append(AttnBlock(channels=in_ch))
    modules.append(ResnetBlock(in_ch=in_ch))

    pyramid_ch = 0
    # Upsampling block
    for i_level in reversed(range(num_resolutions)):
      for i_block in range(num_res_blocks + 1):
        out_ch = nf * ch_mult[i_level]
        in_ch_old = hs_c.pop()
        modules.append(ResnetBlock(in_ch=in_ch + in_ch_old,
                                     out_ch=out_ch))
        in_ch = out_ch

      if all_resolutions[i_level] in attn_resolutions:
        modules.append(AttnBlock(channels=in_ch))

      if i_level != 0:
        if resblock_type == 'ddpm':
          modules.append(Upsample(in_ch=in_ch))
        else:
          modules.append(ResnetBlock(in_ch=in_ch, up=True))

    assert not hs_c

    modules.append(layerspp.get_act_norm(act=act, act_emb=act, norm='spade', ch=in_ch, is3d=self.is3d, n_frames=self.num_frames, num_frames_cond=num_frames_cond,
                                         cond_ch=num_frames_cond*channels, spade_dim=spade_dim, cond_conv=conv3x3, cond_conv1=conv1x1_cond))
    modules.append(conv3x3(in_ch, channels*self.num_frames, init_scale=init_scale))

    self.all_modules = nn.ModuleList(modules)

  def forward(self, x, time_cond, cond=None, cond_mask=None):
    """
    don't use cond_mask
    """
    # timestep/noise_level embedding; only for continuous training
    ## tensor to device ##
    if self.sigmas.device != x.device:
      self.sigmas = self.sigmas.to(x.device)

    ## x, cond: (b, t, c, h, w) -> (b, t*c, h, w)
    _, T, _, _, _ = x.shape
    x = rearrange(x, 'b t c h w -> b (t c) h w')
    cond = rearrange(cond, 'b t c h w -> b (t c) h w')

    modules = self.all_modules
    m_idx = 0

    # if cond is not None:
    #   x = torch.cat([x, cond], dim=1) # B, (num_frames+num_frames_cond)*C, H, W

    if self.is3d: # B, N*C, H, W -> B, C*N, H, W : subtle but important difference!
      B, NC, H, W = x.shape
      CN = NC
      x = x.reshape(B, self.num_frames, self.channels, H, W).permute(0, 2, 1, 3, 4).reshape(B, CN, H, W)
      if cond is not None:
        B, NC, H, W = cond.shape
        CN = NC
        cond = cond.reshape(B, self.num_frames_cond, self.channels, H, W).permute(0, 2, 1, 3, 4).reshape(B, CN, H, W)

    if self.embedding_type == 'fourier':
      # Gaussian Fourier features embeddings.
      used_sigmas = time_cond
      temb = modules[m_idx](torch.log(used_sigmas))
      m_idx += 1
    elif self.embedding_type == 'positional':
      # Sinusoidal positional embeddings.
      timesteps = time_cond
      used_sigmas = self.sigmas[time_cond.long()]
      temb = layers.get_timestep_embedding(timesteps, self.nf)
    else:
      raise ValueError(f'embedding type {self.embedding_type} unknown.')

    if self.conditional:
      temb = modules[m_idx](temb)
      m_idx += 1
      temb = modules[m_idx](self.act(temb)) # b x k
      m_idx += 1
      if self.cond_emb:
        if cond_mask is None:
          cond_mask = torch.ones(x.shape[0], device=x.device, dtype=torch.int32)
        temb = torch.cat([temb, modules[m_idx](cond_mask)], dim=1) # b x (k/8 + k)
        m_idx += 1
    else:
      temb = None

    # Downsampling block
    input_pyramid = None

    x = x.contiguous()
    hs = [modules[m_idx](x)]
    m_idx += 1
    for i_level in range(self.num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(self.num_res_blocks):
        h = modules[m_idx](hs[-1], temb, cond=cond)
        m_idx += 1
        if h.shape[-1] in self.attn_resolutions:
          h = modules[m_idx](h)
          m_idx += 1

        hs.append(h)

      if i_level != self.num_resolutions - 1:
        if self.resblock_type == 'ddpm':
          h = modules[m_idx](hs[-1], cond=cond)
        else:
          h = modules[m_idx](hs[-1], temb, cond=cond)
        m_idx += 1
        hs.append(h)

    # Middle Block

    # ResBlock
    h = hs[-1]
    h = modules[m_idx](h, temb, cond=cond)
    m_idx += 1
    # AttnBlock
    h = modules[m_idx](h)
    m_idx += 1

    # ResBlock
    h = modules[m_idx](h, temb, cond=cond)
    m_idx += 1

    pyramid = None
    # Upsampling block
    for i_level in reversed(range(self.num_resolutions)):
      for i_block in range(self.num_res_blocks + 1): 
        if self.is3d:
          # Get h and h_old
          B, CN, H, W = h.shape
          h = h.reshape(B, -1, self.num_frames, H, W)
          prev = hs.pop().reshape(B, -1, self.num_frames, H, W)
          # Concatenate
          h_comb = torch.cat([h, prev], dim=1) # B, C, N, H, W
          h_comb = h_comb.reshape(B, -1, H, W)
        else:
          prev = hs.pop()
          h_comb = torch.cat([h, prev], dim=1)
        h = modules[m_idx](h_comb, temb, cond=cond)
        m_idx += 1

      if h.shape[-1] in self.attn_resolutions:
        h = modules[m_idx](h)
        m_idx += 1

      if i_level != 0:
        if self.resblock_type == 'ddpm':
          h = modules[m_idx](h, cond=cond)
          m_idx += 1
        else:
          h = modules[m_idx](h, temb, cond=cond)
          m_idx += 1

    assert not hs
    # GroupNorm
    h = modules[m_idx](h, cond=cond)
    m_idx += 1

    # conv3x3_last
    h = modules[m_idx](h)
    m_idx += 1

    assert m_idx == len(modules)

    if self.is3d: # B, C*N, H, W -> B, N*C, H, W subtle but important difference!
      B, CN, H, W = h.shape
      NC = CN
      h = h.reshape(B, self.channels, self.num_frames, H, W).permute(0, 2, 1, 3, 4).reshape(B, NC, H, W)

    ## (b, t*c, h, w) -> (b, t, c, h, w)
    h = rearrange(h, 'b (t c) h w -> b t c h w', t=T)
    return h

  