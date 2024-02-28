if __name__ == "__main__":
    import sys
    sys.path.append('/mnt/cache/gongjunchao/workdir/radar_forecasting')

import torch
import torch.nn as nn

from einops import rearrange

class SpatioTemporalLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, layer_norm):
        super(SpatioTemporalLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        if layer_norm:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 7, width, width])
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, width, width])
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 3, width, width])
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden, width, width])
            )
        else:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x_t, h_t, c_t, m_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        m_new = f_t_prime * m_t + i_t_prime * g_t_prime

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new


class _PredRNN(nn.Module):
    def __init__(self, num_layers, num_hidden, img_channel,
                 img_height, img_width, stride, filter_size, patch_size=1, layer_norm=True):
        super(_PredRNN, self).__init__()

        self.filter_size = filter_size
        self.patch_size = patch_size
        self.img_channel = img_channel
        self.img_height = img_height
        self.img_width = img_width
        self.stride = stride

        self.frame_channel = patch_size * patch_size * img_channel
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        self.width = img_width // patch_size
        self.height = img_height // patch_size

        self.MSE_criterion = nn.MSELoss()

        self.conv_first = nn.Conv2d(img_channel, self.frame_channel, kernel_size=patch_size, stride=patch_size, padding=0, bias=False)
        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, num_hidden[i], self.width, filter_size,
                                       stride, layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Sequential(
                                   nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False), torch.nn.ReLU(),
                                   nn.ConvTranspose2d(self.frame_channel, self.img_channel, kernel_size=patch_size, stride=patch_size, padding=0, bias=False)
                                   )


    def forward(self, frames_tensor, mask_true, input_length=10, total_length=20, reverse_scheduled_sampling=0):
        #  [batch, length, channel, height, width] #
        frames = frames_tensor
        mask_true = mask_true

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], self.height, self.width]).to(frames.device)
            h_t.append(zeros)
            c_t.append(zeros)

        memory = torch.zeros([batch, self.num_hidden[0], self.height, self.width]).to(frames.device)

        for t in range(total_length - 1):
            # reverse schedule sampling
            if reverse_scheduled_sampling == 1:
                if t == 0:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            else:
                if t < input_length:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - input_length] * frames[:, t] + \
                          (1 - mask_true[:, t - input_length]) * x_gen
                    
            net = self.conv_first(net)
            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory)

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 2, 3, 4).contiguous()
        # loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])
        return next_frames


class PredRNN(nn.Module):
    def __init__(self, num_layers, num_hidden, img_channel,
                 img_height, img_width, stride, filter_size, patch_size=1, layer_norm=True, **kwargs):
        super().__init__()
        self.net = _PredRNN(num_layers=num_layers, num_hidden=num_hidden, img_channel=img_channel, img_height= img_height,
                            img_width=img_width, stride=stride, filter_size=filter_size, 
                            patch_size=patch_size, layer_norm=layer_norm)
    
    def forward(self, frames_tensor, mask_true, input_length=10, total_length=20, reverse_scheduled_sampling=0, **kwargs):
        out = self.net(frames_tensor, mask_true, input_length=input_length, total_length=total_length, reverse_scheduled_sampling=reverse_scheduled_sampling)
        return out
    

if __name__ == "__main__":
    # earth_patch_size_latitude((128,256), (4,4))
    print('start')
    b = 8
    inp_length, pred_length = 12, 12
    total_length = inp_length + pred_length
    c = 1
    h, w = 400, 400
    device =  torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    input_data = torch.randn((b, total_length, c, h, w)).to(device)
    mask_true = torch.ones((b, total_length, c, h, w)).to(device)
    target = torch.randn((b, total_length-inp_length, c, h, w)).to(device)

    print('load yaml from config')
    import yaml
    cfg_path = '/mnt/cache/gongjunchao/workdir/radar_forecasting/configs/meteonet/PredRNN.yaml'
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
    model = PredRNN(**backbone_kwargs)
    model.to(device)

    import torch.nn.functional as F
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for i in range(5):
        start.record()
        pred = model(input_data, mask_true=mask_true, input_length=inp_length, total_length=total_length)
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
    flops = FlopCountAnalysis(model, (input_data, mask_true, inp_length, total_length))
    print(flop_count_table(flops))

# srun -p ai4earth --kill-on-bad-exit=1 --quotatype=auto --gres=gpu:1 python -u PredRNN.py #