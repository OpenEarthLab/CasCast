if __name__ == "__main__":
    import sys
    sys.path.append("/mnt/cache/gongjunchao/workdir/radar_forecasting")

import torch
import torch.nn as nn
from networks.prediff.taming.autoencoder_kl import AutoencoderKL
from networks.prediff.taming.losses.contperceptual import LPIPSWithDiscriminator
from networks.prediff.utils.distributions import DiagonalGaussianDistribution

class lpipsWithDisc(nn.Module):
    def __init__(self, config):
        super(lpipsWithDisc, self).__init__()
        self.config = config
        self.net = LPIPSWithDiscriminator(**config)

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx, global_step, mask=None, last_layer=None, split='train'):
        loss, loss_dict = self.net(inputs=inputs, reconstructions=reconstructions, posteriors=posteriors, 
        optimizer_idx=optimizer_idx, global_step=global_step, mask=mask, last_layer=last_layer, split=split)
        out = [loss, loss_dict]
        return out



if __name__ == "__main__":
    print('start')
    b = 1
    c = 1
    h, w = 384, 384
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') #
    input_data = torch.randn((b, c, h, w)).to(device) #torch.randn((b, inp_length, c, h, w)).to(device)
    target = torch.randn((b, c, h, w)).to(device) #
    ## important part: posterior
    # posterior = DiagonalGaussianDistribution(torch.randn((b, 64, h//8, w//8)).to(device))
    # last_layer_weight = torch.randn((1, 128, 3, 3), requires_grad=True).to(device) #
    global_step = 0

    # backbone_kwargs = {
    #     'disc_start': 50001, ## not default
    #     'logvar_init': 0.0,
    #     'kl_weight': 1e-6,
    #     'pixelloss_weight': 1.0,
    #     'disc_num_layers': 3,
    #     'disc_in_channels': 1,
    #     'disc_factor': 1.0,
    #     'disc_weight': 0.5,
    #     'perceptual_weight': 0.0
    # }
    print('load yaml from config')
    import yaml
    cfg_path = '/mnt/cache/gongjunchao/workdir/radar_forecasting/configs/sevir/autoencoder_kl_gan.yaml'
    with open(cfg_path, 'r') as cfg_file:
      cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)
    backbone_kwargs = cfg_params['model']['params']['sub_model']['lpipsWithDisc']
    model = lpipsWithDisc(backbone_kwargs)
    model.to(device)

    ## autoencoder ##
    backbone_kwargs = {
        'in_channels': 1,
        'out_channels': 1,
        'down_block_types': ['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
        'up_block_types': ['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'], 
        'block_out_channels': [128, 256, 512, 512],
        'layers_per_block': 2,
        'latent_channels': 64,
        'norm_num_groups': 32,
    }
    autoencoder = AutoencoderKL(**backbone_kwargs)
    autoencoder.to(device)
    print('end')

    import torch.nn.functional as F
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for i in range(5):
        start.record()
        rec, posterior = autoencoder(sample=input_data, sample_posterior=True, return_posterior=True, generator=None)
        last_layer_weight = autoencoder.decoder.conv_out.weight
        aeloss, loss_dict = model(inputs=input_data, reconstructions=rec, posteriors=posterior, optimizer_idx=0, global_step=global_step, last_layer=last_layer_weight)
        loss = aeloss
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
    flops = FlopCountAnalysis(model, (input_data, target, posterior, 0, global_step, None, last_layer_weight))
    print(flop_count_table(flops))

# srun -p ai4earth --kill-on-bad-exit=1 --quotatype=auto --gres=gpu:1 python -u lpipsWithDisc.py #