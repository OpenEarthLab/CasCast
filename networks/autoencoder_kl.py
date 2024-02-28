if __name__ == "__main__":
    import sys
    sys.path.append("/mnt/cache/gongjunchao/workdir/radar_forecasting")

import torch
import torch.nn as nn
from networks.prediff.taming.autoencoder_kl import AutoencoderKL


class autoencoder_kl(nn.Module):
    def __init__(self, config):
        super(autoencoder_kl, self).__init__()
        self.config = config
        self.net = AutoencoderKL(**config)

    def forward(self, sample, sample_posterior=True, return_posterior=True, generator=None):
        pred, posterior = self.net(sample, sample_posterior, return_posterior, generator)
        out = [pred, posterior]
        return out



if __name__ == "__main__":
    print('start')
    b = 1
    c = 1
    h, w = 384, 384
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') #
    input_data = torch.randn((b, c, h, w)).to(device) #torch.randn((b, inp_length, c, h, w)).to(device)
    target = torch.randn((b, c, h, w)).to(device) #
    # backbone_kwargs = {
    #     'in_channels': 1,
    #     'out_channels': 1,
    #     'down_block_types': ['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
    #     'up_block_types': ['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'], 
    #     'block_out_channels': [128, 256, 512, 512],
    #     'layers_per_block': 2,
    #     'latent_channels': 64,
    #     'norm_num_groups': 32,
    # }
    print('load yaml from config')
    import yaml
    cfg_path = '/mnt/cache/gongjunchao/workdir/radar_forecasting/configs/sevir/autoencoder_kl_gan.yaml'
    with open(cfg_path, 'r') as cfg_file:
      cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)
    backbone_kwargs = cfg_params['model']['params']['sub_model']['autoencoder_kl']
    model = autoencoder_kl(backbone_kwargs)
    model.to(device)
    print('end')

    import torch.nn.functional as F
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for i in range(5):
        start.record()
        pred, posterior = model(sample=input_data, sample_posterior=True, return_posterior=True, generator=None)
        loss = F.mse_loss(pred, target)
        loss.backward()
        # for n, p in model.named_parameters():
        #     if p.grad is None:
        #         print(f'{n} has no grad')
        end.record()
        torch.cuda.synchronize()
        print(start.elapsed_time(end))
        memory = torch.cuda.memory_reserved() / (1024. * 1024)
        print("memory:", memory)
        
    from fvcore.nn.parameter_count import parameter_count_table
    from fvcore.nn.flop_count import flop_count
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    flops = FlopCountAnalysis(model, (input_data))
    print(flop_count_table(flops))

# srun -p ai4earth --kill-on-bad-exit=1 --quotatype=auto --gres=gpu:1 python -u autoencoder_kl.py #

