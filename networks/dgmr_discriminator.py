if __name__ == "__main__":
    import sys
    sys.path.append('/mnt/cache/gongjunchao/workdir/empty/DGMR-pytorch')

import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

from networks.DGMR.dgmr_layers.ConvGRU import ConvGRUCell
from networks.DGMR.dgmr_layers.DBlock import DBlock, D3Block
from networks.DGMR.dgmr_layers.GBlock import GBlock, LastGBlock
from networks.DGMR.dgmr_layers.LatentStack import LBlock, SpatialAttention
from networks.DGMR.dgmr_layers.utils import random_crop, space2depth


class SpatialDiscriminator(nn.Module):
    def __init__(
        self,
        n_frame: int=10,
        debug: bool=False
        ):
        super().__init__()
        self.n_frame = n_frame
        self.debug = debug

        self.avgpooling = nn.AvgPool2d(2)
        self.d_blocks = nn.ModuleList([
                DBlock(4, 48, relu=False, downsample=True), # 4 -> (3 * 4) * 4 = 48
                DBlock(48, 96, downsample=True), # 48 -> (6 * 4) * 4 = 96
                DBlock(96, 192, downsample=True), # 96 -> (12 * 4) * 4 = 192
                DBlock(192, 384, downsample=True), # 192 -> (24 * 4) * 4 = 384
                DBlock(384, 768, downsample=True), # 384 -> (48 * 4) * 4 = 768
                DBlock(768, 768, downsample=False) # 768 -> 768, no downsample no * 4
                ])
        self.linear = nn.Sequential(
                nn.LayerNorm(768),
                spectral_norm(nn.Linear(768, 1))
                )
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = x.unsqueeze(2)
        B, N, C, H, W = x.shape # batch_size, total_frames, channel=1, height, width
        indices = random.sample(range(N), self.n_frame)
        x = x[:, indices, :, :, :]
        if self.debug: print(f"Picked x: {x.shape}")
        x = x.view(B*self.n_frame, C, H, W)
        if self.debug: print(f"Reshaped: {x.shape}")
        x = self.avgpooling(x)
        if self.debug: print(f"Avg pool: {x.shape}")
        x = space2depth(x)
        if self.debug: print(f"S2Dshape: {x.shape}")

        for i, block in enumerate(self.d_blocks):
            x = block(x)
            if self.debug: print(f"D block{i}: {x.shape}")

        # sum pooling
        x = self.relu(x)
        x = torch.sum(x, dim=(-1, -2))
        if self.debug: print(f"Sum pool: {x.shape}")

        x = self.linear(x)
        if self.debug: print(f"Linear : {x.shape}")

        x = x.view(B, self.n_frame, -1)
        if self.debug: print(f"Reshaped: {x.shape}")

        x = torch.sum(x, dim=1)
        if self.debug: print(f"Sum up : {x.shape}")

        return x

class TemporalDiscriminator(nn.Module):
    def __init__(
        self,
        crop_size: int=256,
        debug: bool=False
        ):
        super().__init__()
        self.crop_size = crop_size
        self.debug = debug

        self.avgpooling = nn.AvgPool3d(2)
        self.d3_blocks = nn.ModuleList([
                D3Block(4, 48, relu=False, downsample=True), # C: 4 -> 48, T -> T/2
                D3Block(48, 96, downsample=True) # C: 48 -> 96, T/2 -> T/4 (not exactly the same as DGMR)
                ])
        self.d_blocks = nn.ModuleList([
                DBlock(96, 192, downsample=True), # 96 -> (12 * 4) * 4 = 192
                DBlock(192, 384, downsample=True), # 192 -> (24 * 4) * 4 = 384
                DBlock(384, 768, downsample=True), # 384 -> (48 * 4) * 4 = 768
                DBlock(768, 768, downsample=False) # 768 -> 768, no downsample no * 4
                ])
        self.linear = nn.Sequential(
                nn.LayerNorm(768),
                spectral_norm(nn.Linear(768, 1))
                )
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = random_crop(x, size=self.crop_size).to(x.device)
        x = x.unsqueeze(1)
        B, C, T, H, W = x.shape

        x = x.permute(0, 2, 1, 3, 4).view(B*T, C, H, W) # -> B, T, C, H, W
        if self.debug: print(f"Cropped : {x.shape}")

        x = space2depth(x) # B*T, C, H, W
        x = x.view(B, T, -1, x.shape[-2], x.shape[-1]).permute(0, 2, 1, 3, 4) # -> B, C, T, H, W
        if self.debug: print(f"S2Dshape: {x.shape}")

        for i, block3d in enumerate(self.d3_blocks):
            x = block3d(x)
            if self.debug: print(f"3D block: {x.shape}")

        B, C, T, H, W  = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)
        if self.debug: print(f"Reshaped: {x.shape}")

        for i, block in enumerate(self.d_blocks):
            x = block(x)
            if self.debug: print(f"D block{i}: {x.shape}")

        # sum pooling
        x = self.relu(x)
        x = torch.sum(x, dim=(-1, -2))
        if self.debug: print(f"Sum pool: {x.shape}")

        x = self.linear(x)
        if self.debug: print(f"Linear : {x.shape}")

        x = x.view(B, T, -1)
        if self.debug: print(f"Reshaped: {x.shape}")

        x = torch.sum(x, dim=1)
        if self.debug: print(f"Sum up : {x.shape}")

        return x
    
class _DGMRDiscriminators(nn.Module):
    def __init__(
        self,
        n_frame: int=10,
        crop_size: int=128,
        **kwargs
        ):
        super().__init__()
        self.spatial_discriminator = SpatialDiscriminator(n_frame=n_frame)
        self.temporal_discriminator = TemporalDiscriminator(crop_size=crop_size)

    def forward(self, prev_x, future_x):
        s_score = self.spatial_discriminator(future_x)
        continuous_x = torch.cat([prev_x, future_x], dim=1)
        t_score = self.temporal_discriminator(continuous_x)
        return torch.cat((s_score, t_score), dim=1)

class DGMRDiscriminators(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = _DGMRDiscriminators(**config)
        self.disc_start = config.get('disc_start', 25001)

    def forward(self, prev_x, future_x):
        """
        prev_x: (b, t_prev, c, h, w)
        future_x: (b, t_pred, c, h, w) or (b, k, t_pred, c, h, w)
        """
        assert len(future_x.shape) == 5 or len(future_x.shape) == 6
        if len(future_x.shape) == 5:
            prev_x = prev_x.squeeze(2)
            future_x = future_x.squeeze(2)
            logits = self.net(prev_x, future_x)
        elif len(future_x.shape) == 6:
            prev_x = prev_x.squeeze(2)
            logits_list = []
            k = future_x.shape[1]
            for member in range(k):
                future_x_member = future_x[:, member].squeeze(2)
                logits = self.net(prev_x, future_x_member)
                logits_list.append(logits)
            logits = torch.stack(logits_list, dim=1)
        else:
            raise NotImplementedError

        return logits

    def compute_loss(self, tar_digit, pred_digit, step):
        """
        tar_digit: (b, 2) ## "2" is the concat of spatial and temporal
        pred_digit: (b, k, 2) 
        """
        assert pred_digit.shape[1] == 1, "Only support k=1"
        pred_digit = pred_digit.squeeze(1)

        if step < self.disc_start:
            disc_coeff = 0.
        else:
            disc_coeff = 1.


        # ## hinge gan loss ##
        # ## spatial disc loss ##
        # spatial_loss_disc_true = F.relu(1. - tar_digit[:, 0]).mean() * disc_coeff
        # spatial_loss_disc_fake = F.relu(1. + pred_digit[:, 0]).mean() * disc_coeff
        # spatial_loss_disc = spatial_loss_disc_true + spatial_loss_disc_fake

        # ## temporal disc loss ##
        # temporal_loss_disc_true = F.relu(1. - tar_digit[:, 1]).mean() * disc_coeff
        # temporal_loss_disc_fake = F.relu(1. + pred_digit[:, 1]).mean() * disc_coeff
        # temporal_loss_disc = temporal_loss_disc_true + temporal_loss_disc_fake
        
        ## bce gan loss ##
        tar_digit = torch.clamp(tar_digit, 0., 1.)
        pred_digit = torch.clamp(pred_digit, 0., 1.)
        ## spatial disc loss ##
        spatial_loss_disc_true = F.binary_cross_entropy(tar_digit[:, 0], 1. * torch.ones_like(tar_digit[:, 0]))* disc_coeff  
        spatial_loss_disc_fake = F.binary_cross_entropy(pred_digit[:, 0], 0. * torch.ones_like(pred_digit[:, 0]))* disc_coeff 
        spatial_loss_disc = spatial_loss_disc_true + spatial_loss_disc_fake

        ## temporal disc loss ##
        temporal_loss_disc_true = F.binary_cross_entropy(tar_digit[:, 1], 1. * torch.ones_like(tar_digit[:, 1]))* disc_coeff 
        temporal_loss_disc_fake = F.binary_cross_entropy(pred_digit[:, 1], 0. * torch.ones_like(pred_digit[:, 1]))* disc_coeff 
        temporal_loss_disc = temporal_loss_disc_true + temporal_loss_disc_fake

        disc_loss_dict = {
            'spatial_loss_disc_true': spatial_loss_disc_true,
            'spatial_loss_disc_fake': spatial_loss_disc_fake,
            'spatial_loss_disc': spatial_loss_disc,
            'temporal_loss_disc_true': temporal_loss_disc_true,
            'temporal_loss_disc_fake': temporal_loss_disc_fake,
            'temporal_loss_disc': temporal_loss_disc,
            'loss_disc': spatial_loss_disc + temporal_loss_disc
        }
        return disc_loss_dict

if __name__ == "__main__":#
    print('start')
    b = 8
    inp_length, pred_length = 13, 12
    c = 1
    h, w = 384, 384 #384, 384   
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') #
    input_data = torch.randn((b, inp_length, c, h, w)).to(device) #torch.randn((b, inp_length, c, h, w)).to(device)
    target = torch.randn((b, pred_length, c, h, w)).to(device) #
    t = torch.randint(0, 1000, (b,)).to(device)
    # print('load yaml from config')
    # import yaml
    # cfg_path = '/mnt/cache/gongjunchao/workdir/radar_forecasting/configs/sevir/ldm_unet_pred.yaml'
    # with open(cfg_path, 'r') as cfg_file:
    #   cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)
    # backbone_kwargs = cfg_params['model']['params']['sub_model']['unet2d']
    backbone_kwargs = {
        'n_frame': 8,
        'crop_size': 196
    }
    
    model = DGMRDiscriminators(backbone_kwargs)
    model.to(device)
    print('end')

    k = 1
    import torch.nn.functional as F
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for i in range(5):
        start.record()
        logits_fake = model(prev_x=input_data, future_x=target.unsqueeze(1).expand(-1, k, -1, -1, -1, -1))
        logits_true = model(prev_x=input_data, future_x=target)
        loss = model.compute_loss(tar_digit=logits_true, pred_digit=logits_fake)['loss_disc']
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
    flops = FlopCountAnalysis(model, (input_data, target))
    print(flop_count_table(flops))

# srun -p ai4earth --kill-on-bad-exit=1 --quotatype=spot --gres=gpu:1 python -u dgmr_discriminator.py #