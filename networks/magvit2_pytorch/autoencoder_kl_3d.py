from beartype.typing import Union, Tuple, Optional, List
from beartype import beartype
from torch.nn import Module, ModuleList
import pickle
from torch import nn, einsum, Tensor
from einops import rearrange, repeat, reduce, pack, unpack
import torch


from .magvit2_pytorch import (
    CausalConv3d, cast_tuple, ResidualUnit,
    ResidualUnitMod, SpatialDownsample2x, SpatialUpsample2x,
    TimeDownsample2x, TimeUpsample2x, SpaceAttention,
    FeedForward, LinearSpaceAttention, TokenShift,
    TimeAttention, SimpleGateLoopLayer, ToTimeSequence,
    TokenShift, Rearrange, Sequential,
    Residual, SameConv2d, divisible_by,
     default, exists, safe_get_index,
     pad_at_dim)
from networks.prediff.utils.distributions import DiagonalGaussianDistribution


class AutoencoderKL_3d(Module):
    def __init__(
        self,
        *,
        image_size,
        layers: Tuple[Union[str, Tuple[str, int]], ...] = (
            'residual',
            'residual',
            'residual'
        ),
        latent_dim = 4,
        residual_conv_kernel_size = 3,
        channels = 3,
        init_dim = 64,
        max_dim = float('inf'),
        dim_cond = None,
        dim_cond_expansion_factor = 4.,
        input_conv_kernel_size: Tuple[int, int, int] = (7, 7, 7),
        output_conv_kernel_size: Tuple[int, int, int] = (3, 3, 3),
        pad_mode: str = 'constant',
        attn_dim_head = 32,
        attn_heads = 8,
        attn_dropout = 0.,
        flash_attn = True,
        separate_first_frame_encoding = False,
        gateloop_use_jax = False
    ):
        super().__init__()

        # for autosaving the config

        _locals = locals()
        _locals.pop('self', None)
        _locals.pop('__class__', None)
        self._configs = pickle.dumps(_locals)

        # image size

        self.channels = channels
        self.image_size = image_size

        # initial encoder

        self.conv_in = CausalConv3d(channels, init_dim, input_conv_kernel_size, pad_mode = pad_mode)

        # whether to encode the first frame separately or not

        self.conv_in_first_frame = nn.Identity()
        self.conv_out_first_frame = nn.Identity()

        if separate_first_frame_encoding:
            self.conv_in_first_frame = SameConv2d(channels, init_dim, input_conv_kernel_size[-2:])
            self.conv_out_first_frame = SameConv2d(init_dim, channels, output_conv_kernel_size[-2:])

        self.separate_first_frame_encoding = separate_first_frame_encoding

        # encoder and decoder layers

        self.encoder_layers = ModuleList([])
        self.decoder_layers = ModuleList([])

        self.conv_out = CausalConv3d(init_dim, channels, output_conv_kernel_size, pad_mode = pad_mode)

        dim = init_dim
        dim_out = dim

        layer_fmap_size = image_size
        time_downsample_factor = 1
        has_cond_across_layers = []
    
        for layer_def in layers:
            layer_type, *layer_params = cast_tuple(layer_def)

            has_cond = False

            if layer_type == 'residual':
                encoder_layer = ResidualUnit(dim, residual_conv_kernel_size)
                decoder_layer = ResidualUnit(dim, residual_conv_kernel_size)

            elif layer_type == 'consecutive_residual':
                num_consecutive, = layer_params
                encoder_layer = Sequential(*[ResidualUnit(dim, residual_conv_kernel_size) for _ in range(num_consecutive)])
                decoder_layer = Sequential(*[ResidualUnit(dim, residual_conv_kernel_size) for _ in range(num_consecutive)])

            elif layer_type == 'cond_residual':
                assert exists(dim_cond), 'dim_cond must be passed into VideoTokenizer, if tokenizer is to be conditioned'

                has_cond = True

                encoder_layer = ResidualUnitMod(dim, residual_conv_kernel_size, dim_cond = int(dim_cond * dim_cond_expansion_factor))
                decoder_layer = ResidualUnitMod(dim, residual_conv_kernel_size, dim_cond = int(dim_cond * dim_cond_expansion_factor))
                dim_out = dim

            elif layer_type == 'compress_space':
                dim_out = safe_get_index(layer_params, 0)
                dim_out = default(dim_out, dim * 2)
                dim_out = min(dim_out, max_dim)

                encoder_layer = SpatialDownsample2x(dim, dim_out)
                decoder_layer = SpatialUpsample2x(dim_out, dim)

                assert layer_fmap_size > 1
                layer_fmap_size //= 2

            elif layer_type == 'compress_time':
                dim_out = safe_get_index(layer_params, 0)
                dim_out = default(dim_out, dim * 2)
                dim_out = min(dim_out, max_dim)

                encoder_layer = TimeDownsample2x(dim, dim_out)
                decoder_layer = TimeUpsample2x(dim_out, dim)

                time_downsample_factor *= 2

            elif layer_type == 'attend_space':
                attn_kwargs = dict(
                    dim = dim,
                    dim_head = attn_dim_head,
                    heads = attn_heads,
                    dropout = attn_dropout,
                    flash = flash_attn
                )

                encoder_layer = Sequential(
                    Residual(SpaceAttention(**attn_kwargs)),
                    Residual(FeedForward(dim))
                )

                decoder_layer = Sequential(
                    Residual(SpaceAttention(**attn_kwargs)),
                    Residual(FeedForward(dim))
                )

            elif layer_type == 'linear_attend_space':
                attn_kwargs = dict(
                    dim = dim,
                    dim_head = attn_dim_head,
                    heads = attn_heads,
                    dropout = attn_dropout,
                    flash = flash_attn
                )

                encoder_layer = Sequential(
                    Residual(LinearSpaceAttention(**attn_kwargs)),
                    Residual(FeedForward(dim))
                )

                decoder_layer = Sequential(
                    Residual(LinearSpaceAttention(**attn_kwargs)),
                    Residual(FeedForward(dim))
                )

            elif layer_type == 'gateloop_time':
                gateloop_kwargs = dict(
                    use_jax_associative_scan = gateloop_use_jax
                )

                encoder_layer = ToTimeSequence(Residual(SimpleGateLoopLayer(dim = dim)))
                decoder_layer = ToTimeSequence(Residual(SimpleGateLoopLayer(dim = dim)))

            elif layer_type == 'attend_time':
                attn_kwargs = dict(
                    dim = dim,
                    dim_head = attn_dim_head,
                    heads = attn_heads,
                    dropout = attn_dropout,
                    causal = True,
                    flash = flash_attn
                )

                encoder_layer = Sequential(
                    Residual(TokenShift(TimeAttention(**attn_kwargs))),
                    Residual(TokenShift(FeedForward(dim, dim_cond = dim_cond)))
                )

                decoder_layer = Sequential(
                    Residual(TokenShift(TimeAttention(**attn_kwargs))),
                    Residual(TokenShift(FeedForward(dim, dim_cond = dim_cond)))
                )

            elif layer_type == 'cond_attend_space':
                has_cond = True

                attn_kwargs = dict(
                    dim = dim,
                    dim_cond = dim_cond,
                    dim_head = attn_dim_head,
                    heads = attn_heads,
                    dropout = attn_dropout,
                    flash = flash_attn
                )

                encoder_layer = Sequential(
                    Residual(SpaceAttention(**attn_kwargs)),
                    Residual(FeedForward(dim))
                )

                decoder_layer = Sequential(
                    Residual(SpaceAttention(**attn_kwargs)),
                    Residual(FeedForward(dim))
                )

            elif layer_type == 'cond_linear_attend_space':
                has_cond = True

                attn_kwargs = dict(
                    dim = dim,
                    dim_cond = dim_cond,
                    dim_head = attn_dim_head,
                    heads = attn_heads,
                    dropout = attn_dropout,
                    flash = flash_attn
                )

                encoder_layer = Sequential(
                    Residual(LinearSpaceAttention(**attn_kwargs)),
                    Residual(FeedForward(dim, dim_cond = dim_cond))
                )

                decoder_layer = Sequential(
                    Residual(LinearSpaceAttention(**attn_kwargs)),
                    Residual(FeedForward(dim, dim_cond = dim_cond))
                )

            elif layer_type == 'cond_attend_time':
                has_cond = True

                attn_kwargs = dict(
                    dim = dim,
                    dim_cond = dim_cond,
                    dim_head = attn_dim_head,
                    heads = attn_heads,
                    dropout = attn_dropout,
                    causal = True,
                    flash = flash_attn
                )

                encoder_layer = Sequential(
                    Residual(TokenShift(TimeAttention(**attn_kwargs))),
                    Residual(TokenShift(FeedForward(dim, dim_cond = dim_cond)))
                )

                decoder_layer = Sequential(
                    Residual(TokenShift(TimeAttention(**attn_kwargs))),
                    Residual(TokenShift(FeedForward(dim, dim_cond = dim_cond)))
                )

            else:
                raise ValueError(f'unknown layer type {layer_type}')

            self.encoder_layers.append(encoder_layer)
            self.decoder_layers.insert(0, decoder_layer)

            dim = dim_out
            has_cond_across_layers.append(has_cond)

        # add a final norm just before quantization layer

        self.encoder_layers.append(Sequential(
            Rearrange('b c ... -> b ... c'),
            nn.LayerNorm(dim),
            Rearrange('b ... c -> b c ...'),
        ))
        has_cond_across_layers.append(has_cond)
        # add quantization layer to encoder_layers #
        self.encoder_layers.append(
           nn.Conv3d(dim_out, latent_dim*2, (1, 3, 3), padding=(0, 1, 1)) 
        )
        has_cond_across_layers.append(has_cond)

        ## add dequantization layer to decoder_layers #
        self.decoder_layers.insert(0, nn.Conv3d(latent_dim, dim_out, (1, 3, 3), padding=(0, 1, 1)))
        ######################################################################
        self.time_downsample_factor = time_downsample_factor
        self.time_padding = time_downsample_factor - 1

        self.fmap_size = layer_fmap_size

        # use a MLP stem for conditioning, if needed

        self.has_cond_across_layers = has_cond_across_layers
        self.has_cond = any(has_cond_across_layers)

        self.encoder_cond_in = nn.Identity()
        self.decoder_cond_in = nn.Identity()

        if has_cond:
            self.dim_cond = dim_cond

            self.encoder_cond_in = Sequential(
                nn.Linear(dim_cond, int(dim_cond * dim_cond_expansion_factor)),
                nn.SiLU()
            )

            self.decoder_cond_in = Sequential(
                nn.Linear(dim_cond, int(dim_cond * dim_cond_expansion_factor)),
                nn.SiLU()
            )

    @beartype
    def encode(
        self,
        video: Tensor,
        quantize = False,
        cond: Optional[Tensor] = None,
        video_contains_first_frame = True
    ):
        encode_first_frame_separately = self.separate_first_frame_encoding and video_contains_first_frame

        # whether to pad video or not

        if video_contains_first_frame:
            video = pad_at_dim(video, (self.time_padding, 0), value = 0., dim = 2)

        # conditioning, if needed

        assert (not self.has_cond) or exists(cond), '`cond` must be passed into tokenizer forward method since conditionable layers were specified'

        if exists(cond):
            assert cond.shape == (video.shape[0], self.dim_cond)

            cond = self.encoder_cond_in(cond)
            cond_kwargs = dict(cond = cond)

        # initial conv
        # taking into account whether to encode first frame separately

        if encode_first_frame_separately:
            first_frame, video = video[:, :, 0], video[:, :, 1:]
            xff = self.conv_in_first_frame(first_frame)

        x = self.conv_in(video)

        if encode_first_frame_separately:
            x, _ = pack([xff, x], 'b c * h w')

        # encoder layers

        for fn, has_cond in zip(self.encoder_layers, self.has_cond_across_layers):

            layer_kwargs = dict()

            if has_cond:
                layer_kwargs = cond_kwargs

            x = fn(x, **layer_kwargs)

        ## TODO: add distribution ##
        moments = x
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    @beartype
    def decode(
        self,
        quantized: Tensor,
        cond: Optional[Tensor] = None,
        video_contains_first_frame = True
    ):
        decode_first_frame_separately = self.separate_first_frame_encoding and video_contains_first_frame

        batch = quantized.shape[0]

        # conditioning, if needed

        assert (not self.has_cond) or exists(cond), '`cond` must be passed into tokenizer forward method since conditionable layers were specified'

        if exists(cond):
            assert cond.shape == (batch, self.dim_cond)

            cond = self.decoder_cond_in(cond)
            cond_kwargs = dict(cond = cond)

        # decoder layers

        x = quantized

        for fn, has_cond in zip(self.decoder_layers, reversed(self.has_cond_across_layers)):

            layer_kwargs = dict()

            if has_cond:
                layer_kwargs = cond_kwargs

            x = fn(x, **layer_kwargs)

        # to pixels

        if decode_first_frame_separately:
            left_pad, xff, x = x[:, :, :self.time_padding], x[:, :, self.time_padding], x[:, :, (self.time_padding + 1):]

            out = self.conv_out(x)
            outff = self.conv_out_first_frame(xff)

            video, _ = pack([outff, out], 'b c * h w')

        else:
            video = self.conv_out(x)

            # if video were padded, remove padding

            if video_contains_first_frame:
                video = video[:, :, self.time_padding:]

        return video

    @beartype
    def forward(
            self,
            sample: torch.Tensor,
            sample_posterior: bool = False,
            return_posterior: bool = False,
            generator: Optional[torch.Generator] = None,
            video_contains_first_frame: bool = False
        ):
        r"""
        Args:
            sample (`torch.FloatTensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_posterior (`bool`, *optional*, defaults to `False`):
                Whether or not to return `posterior` along with `dec` for calculating the training loss.
        """
        assert sample.ndim in {5}
        assert sample.shape[-2:] == (self.image_size, self.image_size)
        batch, channels, frames = sample.shape[:3]
        assert divisible_by(frames - int(video_contains_first_frame), self.time_downsample_factor), f'number of frames {frames} minus the first frame ({frames - int(video_contains_first_frame)}) must be divisible by the total downsample factor across time {self.time_downsample_factor}'

        x = sample
        posterior = self.encode(x, video_contains_first_frame=video_contains_first_frame)
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z, video_contains_first_frame=video_contains_first_frame)
        if return_posterior:
            return dec, posterior
        else:
            return dec