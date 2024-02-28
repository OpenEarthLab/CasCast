if __name__ == "__main__":
   ## load yaml ##
   import sys 
   sys.path.append('/mnt/cache/gongjunchao/workdir/integration_for_assimilation')

import numpy as np


import torch.nn as nn
import torch



class _persistent(nn.Module):
    """
    IDLE outputs what input is.
    """
    def __init__(self, **kwargs):
      super().__init__()
      self.i_param = torch.nn.Parameter(torch.Tensor(1))
      pass
      

    def forward(self, frames_tensor, mask_true, input_length=10, total_length=20, reverse_scheduled_sampling=0):
      """
      frames_tensor: b, t, c, h, w
      """
      next_frames = []
      for t in range(total_length - 1):
         next_frames.append(frames_tensor[:, input_length-1])
      next_frames = torch.stack(next_frames, dim=1)
      return next_frames + self.i_param * 0
   
class persistent(nn.Module):
   """
   persistent model.
   """
   def __init__(self, **kwargs) -> None:
      super().__init__()
      self.net = _persistent()

   def forward(self, frames_tensor, mask_true, input_length=10, total_length=20, reverse_scheduled_sampling=0, **kwargs):
        out = self.net(frames_tensor, mask_true, input_length=input_length, total_length=total_length, reverse_scheduled_sampling=reverse_scheduled_sampling)
        return out

if __name__ == "__main__":
   ## load yaml ##
   # earth_patch_size_latitude((128,256), (4,4))
    print('start')
    b = 24
    inp_length, pred_length = 13, 12
    total_length = inp_length + pred_length
    c = 1
    h, w = 384, 384
    device =  torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    input_data = torch.randn((b, total_length, c, h, w)).to(device)
    mask_true = torch.ones((b, total_length, c, h, w)).to(device)
    target = torch.randn((b, total_length-inp_length, c, h, w)).to(device)

    print('load yaml from config')
    import yaml
    cfg_path = '/mnt/cache/gongjunchao/workdir/radar_forecasting/configs/sevir/PredRNN.yaml'
    with open(cfg_path, 'r') as cfg_file:
      cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)
    backbone_kwargs = cfg_params['model']['params']['sub_model']['PredRNN']
    # backbone_kwargs = {
    #     'num_layers':4,
    #     'num_hidden':[128, 128, 128, 128],
    #     'img_channel':1,
    #     'img_width':480, 
    #     'img_height': 480,
    #     'patch_size': 10,
    #     'stride':1, 
    #     'filter_size':3
    # }
    print('end')
    model = persistent(**backbone_kwargs)
    model.to(device)

    import torch.nn.functional as F
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for i in range(5):
        start.record()
        pred = model(input_data, mask_true=mask_true, input_length=13, total_length=25)
        loss = F.mse_loss(pred, input_data[:, 1:])
        loss.backward()
        for n, p in model.net.named_parameters():
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
    flops = FlopCountAnalysis(model, (input_data, mask_true))
    print(flop_count_table(flops))

## srun -p ai4earth --kill-on-bad-exit=1 --quotatype=auto --gres=gpu:1 python -u persistent.py ## 