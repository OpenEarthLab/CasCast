if __name__ == "__main__":
    import sys
    sys.path.append('/mnt/cache/gongjunchao/workdir/radar_forecasting')

import torch
import torch.nn as nn
from networks.ldm.unet2d_openai import UNetModel
from einops import rearrange

class Unet2d_cond(nn.Module):
    def __init__(self, config, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.net = UNetModel(**config)
    
    def forward(self, x, timesteps, cond, context, **kwargs):
        inp = torch.cat([cond, x], dim=1)
        b, _, c, h, w = inp.shape
        inp = inp.reshape(b, -1, h, w).contiguous()
        context = rearrange(context, 'b t c h w -> b (h w) (c t)').contiguous()
        out = self.net(x=inp, timesteps=timesteps, context=context)
        out = out.reshape(b, -1, c, h, w)
        return out

if __name__ == "__main__":
    print('start')
    b = 1
    inp_length, pred_length = 12, 12
    c = 4
    h, w = 48, 48
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') #
    input_data = torch.randn((b, inp_length, c, h, w)).to(device) #torch.randn((b, inp_length, c, h, w)).to(device)
    cond = input_data
    target = torch.randn((b, pred_length, c, h, w)).to(device) #
    t = torch.randint(0, 1000, (b,)).to(device)
    field_context = torch.randn((b, 3, 9, 30, 30)).to(device)
    print('load yaml from config')
    import yaml
    cfg_path = '/mnt/cache/gongjunchao/workdir/radar_forecasting/configs/sevir/ldm_cond_unet.yaml'
    with open(cfg_path, 'r') as cfg_file:
      cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)
    backbone_kwargs = cfg_params['model']['params']['sub_model']['unet2d_cond']
    
    model = Unet2d_cond(config=backbone_kwargs)
    model.to(device)
    print('end')

    import torch.nn.functional as F
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for i in range(5):
        start.record()
        pred = model(x=input_data, timesteps=t, cond=cond, context=field_context)
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
    flops = FlopCountAnalysis(model, (input_data, t, cond, field_context))
    print(flop_count_table(flops))

## srun -p ai4earth --quotatype=auto --ntasks-per-node=1  --cpus-per-task=4 --gres=gpu:1 python -u unet2d_cond.py ##