if __name__ == "__main__":
    import sys
    sys.path.append('/mnt/cache/gongjunchao/workdir/radar_forecasting')
from networks.U_VIT.libs.uvit import UViT as _UViT
import torch
import torch.nn as nn
from einops import rearrange



#################################################################################
#                                   UViT Configs                                  #
#################################################################################

def UViT_H_2(**kwargs):
    return _UViT(depth=28, embed_dim=1152, patch_size=2, num_heads=16, 
                mlp_ratio=4, **kwargs)

def UViT_L_2(**kwargs):
    return _UViT(depth=20, embed_dim=1024, patch_size=2, num_heads=16,
                mlp_ratio=4, **kwargs)


UViT_models = {
    'UViT-H/2': UViT_H_2,
    'UViT-L/2':  UViT_L_2,   
}


class UViT(nn.Module):
    def __init__(self, arch, config):
        super().__init__()
        self.arch = arch
        self.config = config
        if arch not in UViT_models:
            raise ValueError(f'Unrecognized UViT model architecture {arch}')
        self.model = UViT_models[arch](**self.config)
    
    def forward(self, x, timesteps, cond, context=None, **kwargs):
        """
        x: (b, t, c, h, w)
        cond: (b, t, c, h, w)
        """
        b, t, _, h, w = x.shape
        x = rearrange(x, 'b t c h w -> b (t c) h w')
        cond = rearrange(cond, 'b t c h w -> b (t c) h w')
        inp = torch.cat([x, cond], dim=1)
        out = self.model(x=inp, timesteps=timesteps)
        out = rearrange(out, 'b (t c) h w -> b t c h w', t=t)
        return out


if __name__ == '__main__':
    print('start')
    b = 32
    inp_length, pred_length = 12, 12
    c = 4
    h, w = 48, 48
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') #
    dtype = torch.float16
    input_data = torch.randn((b, pred_length, c, h, w)).to(device) #torch.randn((b, inp_length, c, h, w)).to(device)
    cond = input_data
    target = torch.randn((b, pred_length, c, h, w)).to(device) #
    t = torch.randint(0, 1000, (b,)).to(device)
    ########################################################################
    # print('load yaml from config')
    # import yaml
    # cfg_path = '/mnt/cache/gongjunchao/workdir/radar_forecasting/configs/sevir/unet2d_refine.yaml'
    # with open(cfg_path, 'r') as cfg_file:
    #   cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)
    # backbone_kwargs = cfg_params['model']['params']['sub_model']['unet2d']
    backbone_kwargs = {
        'arch': 'UViT-L/2',
        'config':{
            'img_size':48,
            'in_chans':c*pred_length*2,
            'out_chans':c*pred_length,
            'qkv_bias':False,
            'mlp_time_embed':False,
            'num_classes':0,
            'use_checkpoint':False,
            'conv':False
        }
    }
    ## huge ##
    # backbone_kwargs = {
    #     'img_size':48,
    #     'patch_size':2,
    #     'in_chans':c*pred_length*2,
    #     'out_chans':c*pred_length,
    #     'embed_dim':1152,
    #     'depth':28,
    #     'num_heads':16,
    #     'mlp_ratio':4,
    #     'qkv_bias':False,
    #     'mlp_time_embed':False,
    #     'num_classes':0,
    #     'use_checkpoint':False,
    #     'conv':False
    # }
    ## large ##
    #     config.nnet = d(
    #     name='uvit',
    #     img_size=64,
    #     patch_size=4,
    #     in_chans=4,
    #     embed_dim=1024,
    #     depth=20,
    #     num_heads=16,
    #     mlp_ratio=4,
    #     qkv_bias=False,
    #     mlp_time_embed=False,
    #     num_classes=1001,
    #     use_checkpoint=True
    # )
    # backbone_kwargs = {
    #     img_size=64,
    #     patch_size=4,
    #     in_chans=4,
    #     embed_dim=1152,
    #     depth=28,
    #     num_heads=16,
    #     mlp_ratio=4,
    #     qkv_bias=False,
    #     mlp_time_embed=False,
    #     num_classes=1001,
    #     use_checkpoint=True,
    #     conv=False
    # }

    model = UViT(**backbone_kwargs)
    model.to(device)
    print('end')

    import torch.nn.functional as F
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for i in range(5):
        start.record()
        pred = model(x=input_data, timesteps=t, cond=cond)
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
    flops = FlopCountAnalysis(model, (input_data, t, cond))
    print(flop_count_table(flops))
## srun -p ai4earth --quotatype=auto --ntasks-per-node=1  --cpus-per-task=4 --gres=gpu:1 python -u UViT.py ##