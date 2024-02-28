if __name__ == "__main__":
    import sys
    sys.path.append('/mnt/cache/gongjunchao/workdir/radar_forecasting')

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from networks.nowcasting.layers.generation.noise_projector import Noise_Projector
from networks.nowcasting.layers.generation.generative_network import Generative_Encoder, Generative_Decoder
import torch.nn.functional as F



class Dict(dict):
            __setattr__ = dict.__setitem__
            __getattr__ = dict.__getitem__

def dictToObj(dictObj):
        if not isinstance(dictObj, dict):
            return dictObj
        d = Dict()
        for k, v in dictObj.items():
            d[k] = dictToObj(v)
        return d


class convective_net(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        self.configs = configs
        self.gen_enc = Generative_Encoder(self.configs.total_length, base_c=self.configs.ngf)
        self.gen_dec = Generative_Decoder(self.configs)
        self.proj = Noise_Projector(self.configs.ngf, configs)
    
    def forward(self, x_inp, x_pred, k=3):
        """
        x_inp: (b, t, c, h, w)
        x_pred: (b, t, c, h, w)
        k: number of samples
        """
        ## (b, t, 1, h, w) -> (b, t, h, w) ##
        x_inp = x_inp[:, :, 0, :, :]
        x_pred = x_pred[:, :, 0, :, :]
        batch, _, height, width = x_inp.shape

        ## (b, t, h, w) -> (k*b, t, h, w) ##
        x_inp = x_inp.unsqueeze(0).expand(k, -1, -1, -1, -1)
        x_pred = x_pred.unsqueeze(0).expand(k, -1, -1, -1, -1)
        x_inp = rearrange(x_inp, 'k b t h w -> (k b) t h w')
        x_pred = rearrange(x_pred, 'k b t h w -> (k b) t h w')
        adec_feat = self.gen_enc(torch.cat([x_inp, x_pred], dim=1))
        
        if height % 32 == 0:
            noise = torch.randn(k*batch, self.configs.ngf, height // 32, width // 32).to(x_inp.device)
            noise_feature = self.proj(noise).reshape(k*batch, -1, 4, 4, 8, 8).permute(0, 1, 4, 5, 2, 3).reshape(k*batch, -1, height // 8, width // 8)
        else:
            ## for meteonet with 400 as height ##
            assert height == 400
            noise = torch.randn(k*batch, self.configs.ngf, 416 // 32, 416 // 32).to(x_inp.device)
            noise_feature = self.proj(noise).reshape(k*batch, -1, 4, 4, 8, 8).permute(0, 1, 4, 5, 2, 3).reshape(k*batch, -1, 416 // 8, 416 // 8)
            noise_feature = F.interpolate(noise_feature, size=(height // 8, width // 8), mode='bilinear', align_corners=True)
        

        feature = torch.cat([adec_feat, noise_feature], dim=1)
        gen_result = self.gen_dec(feature, x_pred)
        
        # (k*b, t, h, w) -> (k*b, t, 1, h, w) #
        gen_result = gen_result.unsqueeze(2)

        return gen_result

class Nowcast_Generator(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        self.configs = dictToObj(configs)
        self.net = convective_net(self.configs)
        max_pooling_loss_kernel = 5 
        max_pooling_loss_stride = 2
        self.max_pooling = nn.MaxPool2d(kernel_size=max_pooling_loss_kernel, stride=max_pooling_loss_stride)


    def K_MAX_pooling_loss(self, tar_x, pred_x):
        ## generator loss ##
        k, b, t, _, _, _ = pred_x.shape
        ## pred: (k, b, t_tar, c, h, w) -> ((k b t_tar), c, h, w)
        pred_x = rearrange(pred_x, 'k b t c h w -> (k b t) c h w')
        ## tar: (b, t_tar, c, h, w) -> ((b t_tar), c, h, w)
        tar_x = rearrange(tar_x, 'b t c h w -> (b t) c h w')

        pooled_tar = self.max_pooling(tar_x)
        pooled_pred = self.max_pooling(pred_x)
        pooled_pred = rearrange(pooled_pred, '(k b t) c h w -> k (b t) c h w', k=k, b=b, t=t)
        pooled_pred = torch.mean(pooled_pred, dim=0)

        loss = F.l1_loss(pooled_tar, pooled_pred) ## TODO: add w(x) coefficient ##
        return loss

    def compute_loss(self, pred_digits, tar_x, pred_x):
         """
         pred_digits: ((k b), c, h, w))
         tar_x: (b, t_tar, c, h, w)
         pred_x: (k, b, t_tar, c, h, w)
         """
         loss_gen = F.binary_cross_entropy(pred_digits, 1. * torch.ones_like(pred_digits))
         loss_dict = {}
         loss_dict['K_MAX_pooling_loss'] = self.K_MAX_pooling_loss(tar_x=tar_x, pred_x=pred_x)
         loss_dict['loss_gen'] = loss_gen
         loss_dict['total_loss'] = 6*loss_gen + 20*loss_dict['K_MAX_pooling_loss']
         return loss_dict 

    def forward(self, inp_x, coarse_x, k=3):
        """
        inp_x: (b, t_in, c, h, w)
        tar_x: (b, t_tar, c, h, w)
        coarse_x: (b, t_tar, c, h, w)
        """
        b, _, _, _, _ = inp_x.shape
        gen_result = self.net(x_inp=inp_x, x_pred=coarse_x, k=k)
        gen_result = rearrange(gen_result, '(k b) t c h w -> k b t c h w', b=b)
        return gen_result

if __name__ == "__main__":
    print('start')
    b = 8
    inp_length, pred_length = 12, 12
    total_length = inp_length + pred_length
    c = 1
    h, w = 400, 400
    device =  torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    input_data = torch.randn((b, inp_length, c, h, w)).to(device)
    pred_data = torch.randn((b, pred_length, c, h, w)).to(device)
    pred_digits = torch.clamp(torch.randn((3*b, 1, 24, 24)).to(device), min=0, max=1)

    print('load yaml from config')
    import yaml
    cfg_path = '/mnt/cache/gongjunchao/workdir/radar_forecasting/configs/meteonet/nowcast_gan.yaml'
    with open(cfg_path, 'r') as cfg_file:
      cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)
    backbone_kwargs = cfg_params['model']['params']['sub_model']['nowcast_generator']
    # backbone_kwargs = {
    #     'total_length': 20,
    #     'input_length': 10,
    #     'ngf': 32,
    #     'img_height': 480,
    #     'img_width': 480,
    #     'ic_feature': 32 * 10,
    #     'gen_oc': 20-10,
    #     'evo_ic': 20-10,

    # }
    print('end')
    gen_model = Nowcast_Generator(backbone_kwargs)
    gen_model.to(device)

    import torch.nn.functional as F
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for i in range(5):
        start.record()
        pred= gen_model(inp_x=input_data, coarse_x=pred_data)
        loss_dict = gen_model.compute_loss(pred_digits=pred_digits, tar_x=pred_data, pred_x=pred)
        loss = loss_dict['total_loss']
        # loss =F.mse_loss(pred, pred_data)
        loss.backward()
        for n, p in gen_model.named_parameters():
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
    flops = FlopCountAnalysis(gen_model, (input_data, pred_data))
    print(flop_count_table(flops))

## srun -p ai4earth --kill-on-bad-exit=1 --quotatype=auto --gres=gpu:1 python -u nowcast_generator.py ##