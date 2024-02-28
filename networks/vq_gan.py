if __name__ == "__main__":
    import sys
    sys.path.append("/mnt/cache/gongjunchao/workdir/radar_forecasting")
from networks.vqgan.vqgan import VQGAN
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

class vq_gan(nn.Module):
    def __init__(self, config):
        super(vq_gan, self).__init__()
        self.config = config
        self.net = VQGAN(dictToObj(Dict(config)))

    def forward(self, x):
        decoded_images, codebook_indices, q_loss = self.net(x)    
        out = [decoded_images, codebook_indices, q_loss]
        return out

if __name__ == "__main__":
    print('start')
    b = 1
    c = 1
    h, w = 384, 384
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') #
    input_data = torch.randn((b, c, h, w)).to(device) #torch.randn((b, inp_length, c, h, w)).to(device)
    target = torch.randn((b, c, h, w)).to(device) #
    print('load yaml from config')
    import yaml
    cfg_path = '/mnt/cache/gongjunchao/workdir/radar_forecasting/configs/sevir/vqgan_pretrain.yaml'
    with open(cfg_path, 'r') as cfg_file:
      cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)
    backbone_kwargs = cfg_params['model']['params']['sub_model']['vq_gan']

    # backbone_kwargs = {
    #     "latent_dim":256,
    #     "image_size":384,
    #     "num_codebook_vectors":1024, 
    #     "image_channels":1,
    #     'beta': 0.25,
    #     # beta=0.25, image_channels=3, dataset_path='C:\\Users\\dome\\datasets\\flowers', 
    #     # device='cuda', batch_size=6, epochs=50, learning_rate=2.25e-05, 
    #     # beta1=0.5, beta2=0.9, disc_start=10000, disc_factor=1.0, 
    #     # l2_loss_factor=1.0, perceptual_loss_factor=1.0
    # }
    
    model = vq_gan(config=backbone_kwargs)
    model.to(device)
    print('end')

    import torch.nn.functional as F
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for i in range(5):
        start.record()
        decoded_images, codebook_indices, q_loss = model(input_data)
        loss = F.mse_loss(decoded_images, target)
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
    flops = FlopCountAnalysis(model, (input_data))
    print(flop_count_table(flops))

## srun -p ai4earth --quotatype=spot --ntasks-per-node=1  --cpus-per-task=4 --gres=gpu:1 python -u vq_gan.py ##