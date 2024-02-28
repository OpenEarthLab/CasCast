if __name__ == "__main__":
    import sys
    sys.path.append("/mnt/cache/gongjunchao/workdir/radar_forecasting")

from networks.magvit2_pytorch.autoencoder_kl_3d import AutoencoderKL_3d 
import torch
from networks.lpipsWithDisc import lpipsWithDisc
import torch.nn as nn
from einops import rearrange



class autoencoder_3d(nn.Module):
    def __init__(self, config):
        super(autoencoder_3d, self).__init__()
        self.config = config
        self.net = AutoencoderKL_3d(**config)
    
    def get_last_layer_weight(self):
        return self.net.conv_out.conv.weight
    
    def forward(self, sample,
            sample_posterior=True,
            return_posterior=True,
            generator=None,
            video_contains_first_frame=False, **kwargs):
        """
        inputs: (b, t, c, h, w)
        """
        b, t, c, h, w = sample.shape
        sample = rearrange(sample, 'b t c h w -> b c t h w')
        rec, posterior = self.net(sample=sample, sample_posterior=sample_posterior, return_posterior=return_posterior,
                                    generator=generator, video_contains_first_frame=video_contains_first_frame)
        rec = rearrange(rec, 'b c t h w -> b t c h w')
        out = [rec, posterior]
        return out
    
if __name__ == "__main__":
    print('start')
    b = 3
    inp_length, pred_length = 12, 12
    c = 1
    h, w = 384, 384
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') #
    input_data = torch.randn((b, inp_length, c, h, w)).to(device) #torch.randn((b, inp_length, c, h, w)).to(device)
    target = torch.randn((b, pred_length, c, h, w)).to(device) #
    ########################################################################
    print('load yaml from config')
    import yaml
    cfg_path = '/mnt/cache/gongjunchao/workdir/radar_forecasting/configs/sevir/autoencoder_3d.yaml'
    with open(cfg_path, 'r') as cfg_file:
      cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)
    backbone_kwargs = cfg_params['model']['params']['sub_model']['autoencoder_3d']

    print('load yaml from config')
    import yaml
    cfg_path = '/mnt/cache/gongjunchao/workdir/radar_forecasting/configs/sevir/autoencoder_3d.yaml'
    with open(cfg_path, 'r') as cfg_file:
      cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)
    loss_backbone_kwargs = cfg_params['model']['params']['sub_model']['lpipsWithDisc']
    loss_model = lpipsWithDisc(loss_backbone_kwargs)
    loss_model.to(device)

    model = autoencoder_3d(backbone_kwargs)
    model.to(device)
    print('end')

    import torch.nn.functional as F
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for i in range(5):
        start.record()
        rec, posterior = model(sample=input_data,
            sample_posterior=True,
            return_posterior=True,
            generator=None,
            video_contains_first_frame=False)
        last_layer_weight = model.get_last_layer_weight()
        aeloss, loss_dict = loss_model(inputs=rearrange(input_data, 'b t c h w -> (b t) c h w'),
                                reconstructions=rearrange(rec, 'b t c h w -> (b t) c h w'),
                                posteriors=posterior, optimizer_idx=0, global_step=0, last_layer=last_layer_weight)
        aeloss.backward()
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
    flops = FlopCountAnalysis(model, (input_data))
    print(flop_count_table(flops))
    
    # srun -p ai4earth --kill-on-bad-exit=1 --quotatype=auto --gres=gpu:1 python -u autoencoder_3d.py #
