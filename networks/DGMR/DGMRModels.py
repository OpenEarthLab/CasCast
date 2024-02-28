if __name__ == "__main__":
    import sys
    sys.path.append('/mnt/cache/gongjunchao/workdir/radar_forecasting')
import sys
import torch
import random
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

from networks.DGMR.dgmr_layers.ConvGRU import ConvGRUCell
from networks.DGMR.dgmr_layers.DBlock import DBlock, D3Block
from networks.DGMR.dgmr_layers.GBlock import GBlock, LastGBlock
from networks.DGMR.dgmr_layers.LatentStack import LBlock, SpatialAttention
from networks.DGMR.dgmr_layers.utils import random_crop, space2depth

from einops import rearrange

class DGMRGenerator(nn.Module):
    def __init__(
        self,
        in_step: int=4,
        out_step: int=6,
        debug: bool=False
        ):
        super().__init__()
        self.in_step = in_step
        self.out_step = out_step
        self.debug = debug

        # self.d_block0 = nn.ModuleList([DBlock(self.in_step*4, self.in_step*8, relu=False, downsample=True) for _ in range(3)])
        # self.d_block1 = nn.ModuleList([DBlock(self.in_step*8, self.in_step*16, relu=False, downsample=True) for _ in range(3)])
        # self.d_block2 = nn.ModuleList([DBlock(self.in_step*16, self.in_step*32, relu=False, downsample=True) for _ in range(3)])
        # self.d_block3 = nn.ModuleList([DBlock(self.in_step*32, self.in_step*64, relu=False, downsample=True) for _ in range(3)])

        self.d_block0 = DBlock(self.in_step*4, self.in_step*8, relu=False, downsample=True)
        self.d_block1 = DBlock(self.in_step*8, self.in_step*16, relu=False, downsample=True)
        self.d_block2 = DBlock(self.in_step*16, self.in_step*32, relu=False, downsample=True)
        self.d_block3 = DBlock(self.in_step*32, self.in_step*64, relu=False, downsample=True)
    

        self.conv0 = nn.Sequential(
                nn.utils.parametrizations.spectral_norm(nn.Conv2d(self.in_step*8, 48, 3, 1, 1)),
                nn.ReLU()
                )
        self.conv1 = nn.Sequential(
                nn.utils.parametrizations.spectral_norm(nn.Conv2d(self.in_step*16, 96, 3, 1, 1)),
                nn.ReLU()
                )
        self.conv2 = nn.Sequential(
                nn.utils.parametrizations.spectral_norm(nn.Conv2d(self.in_step*32, 192, 3, 1, 1)),
                nn.ReLU()
                )
        self.conv3 = nn.Sequential(
                nn.utils.parametrizations.spectral_norm(nn.Conv2d(self.in_step*64, 384, 3, 1, 1)),
                nn.ReLU()
                ) 
        self.latent_conv = nn.Sequential(
                    nn.utils.parametrizations.spectral_norm(nn.Conv2d(8, 8, 3, 1, 1)),
                    LBlock(8, 24),
                    LBlock(24, 48),
                    LBlock(48, 192),
                    SpatialAttention(),
                    LBlock(192, 768)
                    )

        # self.g_block0 = nn.ModuleList([GBlock(384, 384) for _ in range(self.out_step)])
        # self.g_block1 = nn.ModuleList([GBlock(192, 192) for _ in range(self.out_step)])
        # self.g_block2 = nn.ModuleList([GBlock(96, 96) for _ in range(self.out_step)])
        # self.g_block3 = nn.ModuleList([LastGBlock(48) for _ in range(self.out_step)])
        # self.gru_layer0 = nn.ModuleList([ConvGRUCell(768, 384) if _ > 0 else ConvGRUCell(768, 384) for _ in range(self.out_step)])
        # self.gru_layer1 = nn.ModuleList([ConvGRUCell(384, 192) if _ > 0 else ConvGRUCell(384, 192) for _ in range(self.out_step)])
        # self.gru_layer2 = nn.ModuleList([ConvGRUCell(192, 96) if _ > 0 else ConvGRUCell(192, 96) for _ in range(self.out_step)])
        # self.gru_layer3 = nn.ModuleList([ConvGRUCell(96, 48) if _ > 0 else ConvGRUCell(96, 48) for _ in range(self.out_step)])
        self.g_block0 = GBlock(384, 384)
        self.g_block1 = GBlock(192, 192)
        self.g_block2 = GBlock(96, 96)
        self.g_block3 = LastGBlock(48)
        self.gru_layer0 = ConvGRUCell(768, 384)
        self.gru_layer1 = ConvGRUCell(384, 192)
        self.gru_layer2 = ConvGRUCell(192, 96)
        self.gru_layer3 = ConvGRUCell(96, 48)

    def forward(self, x0, return_noise=False):
        # import pdb; pdb.set_trace()
        B, C, H, W = x0.shape

        ##### conditioning stack #####
        x0 = space2depth(x0) # 256 -> 128
        if self.debug: print(f"s2d    : {x0.shape}")

        ### downsample with so many Ds ###
        ### We want x0, x1, x2, x3 for next step
        temp_x0, temp_x1, temp_x2, temp_x3 = [], [], [], []
        d0 = self.d_block0(x0)
        d1 = self.d_block1(d0)
        d2 = self.d_block2(d1)
        d3 = self.d_block3(d2)
        temp_x0.append(d0)
        temp_x1.append(d1)
        temp_x2.append(d2)
        temp_x3.append(d3)

        x0 = torch.cat(temp_x0, dim=1)
        if self.debug: print(f"new x0 : {x0.shape}")
        x1 = torch.cat(temp_x1, dim=1)
        if self.debug: print(f"new x1 : {x1.shape}")
        x2 = torch.cat(temp_x2, dim=1)
        if self.debug: print(f"new x2 : {x2.shape}")
        x3 = torch.cat(temp_x3, dim=1)
        if self.debug: print(f"new x3 : {x3.shape}")

        del temp_x0, temp_x1, temp_x2, temp_x3

        x0 = self.conv0(x0)
        if self.debug: print(f"conv 0 : {x0.shape}")

        x1 = self.conv1(x1)
        if self.debug: print(f"conv 1 : {x1.shape}")

        x2 = self.conv2(x2)
        if self.debug: print(f"conv 2 : {x2.shape}")

        x3 = self.conv3(x3)
        if self.debug: print(f"conv 3 : {x3.shape}")

        ##### sampler #####
        # import pdb; pdb.set_trace()
        outputs = []
        for t in range(self.out_step):
            noise = self.latent_conv(torch.randn((B, 8, H//32, W//32)).to(x0.device))

            if self.debug: print(f"init x3: {x3.shape}")
            x3 = self.gru_layer0(noise, x3)
            if self.debug: print(f"1st GRU: {x3.shape}")
            g = self.g_block0(x3)
            x2 = self.gru_layer1(g, x2)
            if self.debug: print(f"2nd GRU: {x2.shape}")
            g = self.g_block1(x2)
            x1 = self.gru_layer2(g, x1)
            if self.debug: print(f"3rd GRU: {x1.shape}")
            g = self.g_block2(x1)
            x0 = self.gru_layer3(g, x0)
            if self.debug: print(f"4th GRU: {x0.shape}")
            g = self.g_block3(x0) 
            outputs.append(g)
            #noises.append(noise.detach().cpu().numpy()[0])

        outputs = torch.cat(outputs, dim=1)
        if self.debug: print(f"outputs: {outputs.shape}")
        #if return_noise:
        #    return outputs, noises
        #else:
        return outputs

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
                nn.BatchNorm1d(768),
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
                nn.BatchNorm1d(768),
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

class DGMRDiscriminators(nn.Module):
    def __init__(
        self,
        n_frame: int=10,
        crop_size: int=128
        ):
        super().__init__()
        self.spatial_discriminator = SpatialDiscriminator(n_frame=n_frame)
        self.temporal_discriminator = TemporalDiscriminator(crop_size=crop_size)

    def forward(self, x, y):
        inputs = torch.cat((x, y), dim=1)
        s_score = self.spatial_discriminator(inputs)
        t_score = self.temporal_discriminator(inputs)
        return torch.cat((s_score, t_score), dim=0)

if __name__ == "__main__":
    print('start')
    b = 8
    inp_length, pred_length = 13, 12
    c = 1
    h, w = 384, 384 #384, 384   
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') #
    input_data = torch.randn((b, pred_length* c, h, w)).to(device) #torch.randn((b, inp_length, c, h, w)).to(device)
    cond = torch.randn((b, inp_length* c, h, w)).to(device)
    target = torch.randn((b, pred_length* c, h, w)).to(device) #
    t = torch.randint(0, 1000, (b,)).to(device)
    # print('load yaml from config')
    # import yaml
    # cfg_path = '/mnt/cache/gongjunchao/workdir/radar_forecasting/configs/sevir/ldm_unet_pred.yaml'
    # with open(cfg_path, 'r') as cfg_file:
    #   cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)
    # backbone_kwargs = cfg_params['model']['params']['sub_model']['unet2d']
    backbone_kwargs = {
        'in_step': inp_length,
        'out_step': pred_length,
        'debug': False
    }
    
    model = DGMRGenerator(**backbone_kwargs)
    model.to(device)
    print('end')

    import torch.nn.functional as F
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for i in range(5):
        start.record()
        pred = model(cond)
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
    flops = FlopCountAnalysis(model, (cond))
    print(flop_count_table(flops))

# srun -p ai4earth --kill-on-bad-exit=1 --quotatype=auto --gres=gpu:1 python -u DGMRModels.py #