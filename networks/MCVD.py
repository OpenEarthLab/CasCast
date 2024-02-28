if __name__ == "__main__":
    import sys
    sys.path.append('/mnt/cache/gongjunchao/workdir/radar_forecasting')
try:
    from .mcvd import  SPADE_NCSNpp
except:
    from networks.mcvd.mcvd import SPADE_NCSNpp

import torch
import torch.nn as nn


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


class MCVD(nn.Module):
    def __init__(self, config, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.net = SPADE_NCSNpp(dictToObj(Dict(config)))
    
    def forward(self, x, timesteps, cond, **kwargs):
        out = self.net(x=x, time_cond=timesteps, cond=cond)
        return out

if __name__ == "__main__":
    print('start')
    b = 1
    seq_length = 12
    c = 1
    h, w = 384, 384
    device =  torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    cond = torch.randn((b, seq_length, c, h, w)).to(device)
    x = torch.randn((b, seq_length, c, h, w)).to(device)
    target = torch.randn((b, seq_length, c, h, w)).to(device)
    t = torch.randint(0, 1000, (b,)).to(device)

    print('load yaml from config')
    import yaml
    cfg_path = '/mnt/cache/gongjunchao/workdir/radar_forecasting/configs/sevir/mcvd_refine.yaml'
    with open(cfg_path, 'r') as cfg_file:
      cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)
    backbone_kwargs = cfg_params['model']['params']['sub_model']['mcvd']

    # backbone_kwargs = {
    #     'arch': 'unetmore',
    #     'sigma': {'sigma_dist':'linear', 'sigma_begin':0.02, 'sigma_end':0.0001
    #               },
    #     'channels': 1,
    #     'num_frames': 12,
    #     'num_frames_cond': 12,
    #     'ngf': 64,
    #     'ch_mult': (1, 2, 3, 4),
    #     'num_res_blocks': 2,
    #     'attn_resolutions': (8, 16, 32),
    #     'dropout': 0.0,
    #     'image_size': 384,
    #     'time_conditional': True,
    #     'cond_emb': False,
    #     'spade_dim': 128,
    #     'n_head_channels': 64,
    #     'num_classes': 1000,
    # }

    print('end')
    model = MCVD(backbone_kwargs)
    model.to(device)

    import torch.nn.functional as F
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for i in range(5):
        start.record()
        pred = model(x=x, timesteps=t, cond=cond)
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
    flops = FlopCountAnalysis(model, (x, t, cond))
    print(flop_count_table(flops))

## srun -p ai4earth --kill-on-bad-exit=1 --quotatype=auto --gres=gpu:1 python -u MCVD.py ##
