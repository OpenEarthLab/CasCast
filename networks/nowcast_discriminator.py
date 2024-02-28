import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F



class LBlock(nn.Module):
    def __init__(self, in_channel, out_channel, group_num, downsample=False):
        super(LBlock, self).__init__()
        self.one_conv = nn.Sequential(nn.GroupNorm(group_num, in_channel), nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1))
        self.double_conv = nn.Sequential(
            nn.GroupNorm(group_num, in_channel), ## TODO: adjust according to in_channel
            nn.ReLU(),
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        )
        if downsample:
            self.out_fn = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            self.out_fn = nn.Identity()

    def forward(self, x):
        x1 = self.one_conv(x)
        x2 = self.double_conv(x)
        output = x1 + x2
        output = self.out_fn(output)
        return output
    
class _Temporal_Discriminator(nn.Module):
    def __init__(self, inp_len, pred_len, 
                 first_group_num=5, L1_group_num=30, in_chan = 180,
                 ) -> None:
        """
        first_group_num: group number of the first conv layer. 5 for sevir
        L1_group_num: group number of the first LBlock. 30 for sevir
        in_chan: in_chan of L layer. 180 for sevir
        """
        super().__init__()
        self.hidden_dim = 128
        self.conv_first_1 = nn.Sequential(nn.GroupNorm(first_group_num, (inp_len + pred_len)), nn.Conv2d((inp_len + pred_len), 64, kernel_size=9, 
                                                    stride=(2, 2), padding=4)) ## check h,w
        self.conv_first_2 = nn.Conv3d(in_channels=1, out_channels=4,  kernel_size=(4, 9, 9), stride=(1, 2, 2), padding=(0, 4, 4))
        self.conv_first_3 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(4, 9, 9), stride=(1, 2, 2), padding=(0, 4, 4))

        in_chan = in_chan
        self.L1 = LBlock(in_chan, self.hidden_dim, group_num=L1_group_num, downsample=True) ## in channel is the flatten and cat of 3 conv features
        self.L2 = LBlock(self.hidden_dim, self.hidden_dim * 2, group_num=32, downsample=True)
        self.L3 = LBlock(self.hidden_dim * 2,  self.hidden_dim * 4, group_num=32, downsample=True)
        self.L4 = LBlock(self.hidden_dim * 4,  self.hidden_dim * 4, group_num=32,)
        self.last_conv = nn.Sequential(nn.GroupNorm(32, self.hidden_dim * 4), nn.LeakyReLU(0.2), nn.Conv2d(self.hidden_dim * 4, 1, kernel_size=3, padding=1))

    def forward(self, inp_x, pred_x):
        ## (b, t, c, h, w) ##
        x_cat = torch.cat([inp_x, pred_x], dim=1)
        x_cat = rearrange(x_cat, 'b t c h w -> b (t c) h w')
        x_feat_1 = self.conv_first_1(x_cat)
        x_feat_2 = self.conv_first_2(rearrange(pred_x, 'b t c h w -> b c t h w'))
        x_feat_3 = self.conv_first_3(rearrange(inp_x, 'b t c h w -> b c t h w'))

        x_feat_2 = rearrange(x_feat_2, 'b c t h w -> b (c t) h w')
        x_feat_3 = rearrange(x_feat_3, 'b c t h w -> b (c t) h w')
        x_feat = torch.cat([x_feat_1, x_feat_2, x_feat_3], dim=1)

        x_feat = self.L1(x_feat)
        x_feat = self.L2(x_feat)
        x_feat = self.L3(x_feat)
        x_feat = self.L4(x_feat)

        x_feat = self.last_conv(x_feat)
        x_digits = torch.sigmoid(x_feat)

        return x_digits



class Temporal_Discriminator(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.config = kwargs
        self.net = _Temporal_Discriminator(**self.config)

    
    def compute_loss(self, tar_digit, pred_digit):
        """
        tar_digit: (b, 1, h, w)
        pred_digit: (k*b, 1, h, w)
        """
        b, _, _, _ = tar_digit.shape
        kb = pred_digit.shape[0]
        k = kb//b
        ## (b, 1, h, w) -> (k*b, 1, h, w)
        tar_digit = tar_digit.unsqueeze(0).expand(k, -1, -1, -1, -1)
        tar_digit = rearrange(tar_digit, 'k b c h w -> (k b) c h w')
        loss_disc_fake = F.binary_cross_entropy(pred_digit, 0. * torch.ones_like(pred_digit))
        loss_disc_real = F.binary_cross_entropy(tar_digit, 1. * torch.ones_like(tar_digit))
        
        return {'loss_disc_fake': loss_disc_fake, 'loss_disc_real': loss_disc_real, 'loss_disc': loss_disc_fake + loss_disc_real}

    def forward(self, inp_x, x):
        """
        inp_x: (b, t_inp, c, h, w)
        x: (b, t_tar, c, h, w)/(k, b, t_tar, c, h, w) k indicate k samples.
        """
        if len(x.shape) == 5:
            pass
        elif len(x.shape) == 6:
            k, b, t, c, h, w = x.shape
            x = rearrange(x, 'k b t c h w -> (k b) t c h w')
            inp_x = inp_x.unsqueeze(0).expand(k, -1, -1, -1, -1, -1)
            inp_x = rearrange(inp_x, 'k b t c h w -> (k b) t c h w')
        else:
            raise NotImplementedError
        x_digit = self.net(inp_x = inp_x, pred_x = x)
        return x_digit

if __name__ == "__main__":
    print('start')
    b = 2
    inp_length, pred_length = 12, 12
    total_length = inp_length + pred_length
    c = 1
    h, w = 400, 400
    device =  torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    input_data = torch.randn((b, inp_length, c, h, w)).to(device)
    pred_data = torch.randn((3, b, pred_length, c, h, w)).to(device)
    output_data = torch.randn((b, pred_length, c, h, w)).to(device)
    mask_true = torch.ones((b, total_length, c, h, w)).to(device)
    target = torch.randn((b, total_length-inp_length, c, h, w)).to(device)

    print('load yaml from config')
    import yaml
    cfg_path = '/mnt/cache/gongjunchao/workdir/radar_forecasting/configs/meteonet/nowcast_gan.yaml'
    with open(cfg_path, 'r') as cfg_file:
      cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)
    backbone_kwargs = cfg_params['model']['params']['sub_model']['nowcast_discriminator']
    # backbone_kwargs = {
    #     'inp_len': inp_length,
    #     'pred_len': pred_length,
    #     'first_group_num': 6,
    #     'L1_group_num': 43,
    #     'in_chan': 172
    # }

    print('end')
    model = Temporal_Discriminator(**backbone_kwargs)
    model.to(device)

    import torch.nn.functional as F
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for i in range(5):
        start.record()
        pred_digit = model(inp_x=input_data, x=pred_data)
        tar_digit = model(inp_x=input_data, x=target)
        loss_dict = model.compute_loss(tar_digit=tar_digit, pred_digit=pred_digit) 
        loss = loss_dict['loss_disc']
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
    flops = FlopCountAnalysis(model, (input_data, pred_data))
    print(flop_count_table(flops))

## srun -p ai4earth --kill-on-bad-exit=1 --quotatype=auto --gres=gpu:1 python -u nowcast_discriminator.py ##