import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class tf_Conv3d(nn.Module):

    def __init__(self, in_channels, out_channels, *vargs, **kwargs):
        super(tf_Conv3d, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, *vargs, **kwargs)

    def forward(self, input):
        return F.interpolate(self.conv3d(input), size=input.shape[-3:], mode="nearest")
    
class Eidetic3DLSTMCell(nn.Module):

    def __init__(self, in_channel, num_hidden, window_length,
                 height, width, filter_size, stride, layer_norm):
        super(Eidetic3DLSTMCell, self).__init__()

        self._norm_c_t = nn.LayerNorm([num_hidden, window_length, height, width])
        self.num_hidden = num_hidden
        self.padding = (0, filter_size[1] // 2, filter_size[2] // 2) 
        self._forget_bias = 1.0
        if layer_norm:
            self.conv_x = nn.Sequential(
                tf_Conv3d(in_channel, num_hidden * 7, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 7, window_length, height, width])
            )
            self.conv_h = nn.Sequential(
                tf_Conv3d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, window_length, height, width])
            )
            self.conv_gm = nn.Sequential(
                tf_Conv3d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, window_length, height, width])
            )
            self.conv_new_cell = nn.Sequential(
                tf_Conv3d(num_hidden, num_hidden, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden, window_length, height, width])
            )
            self.conv_new_gm = nn.Sequential(
                tf_Conv3d(num_hidden, num_hidden, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden, window_length, height, width])
            )
        else:
            self.conv_x = nn.Sequential(
                tf_Conv3d(in_channel, num_hidden * 7, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_h = nn.Sequential(
                tf_Conv3d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_gm = nn.Sequential(
                tf_Conv3d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_new_cell = nn.Sequential(
                tf_Conv3d(num_hidden, num_hidden, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_new_gm = nn.Sequential(
                tf_Conv3d(num_hidden, num_hidden, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
        self.conv_last = tf_Conv3d(num_hidden * 2, num_hidden, kernel_size=1,
                                   stride=1, padding=0, bias=False)
    
    def _attn(self, in_query, in_keys, in_values):
        batch, num_channels, t, width, height = in_query.shape
        query = rearrange(in_query, 'b (k c) t w h -> b k (c t w h)', k=1)
        keys = rearrange(in_keys, 'b (k c) t w h -> b k (c t w h)', c=num_channels) 
        values = rearrange(in_values, 'b (k c) t w h -> b k (c t w h)', c=num_channels) 
        attn = torch.einsum('bxc,byc->bxy', query, keys)
        attn = torch.softmax(attn, dim=2)
        attn = torch.einsum("bxy,byc->bxc", attn, values)
        attn = rearrange(attn, 'b k (c t w h) -> b (k c) t w h', k=1, c=num_channels, h=height, w=width)
        return attn

    def forward(self, x_t, h_t, c_t, global_memory, eidetic_cell):
        h_concat = self.conv_h(h_t)
        i_h, g_h, r_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)

        x_concat = self.conv_x(x_t)
        i_x, g_x, r_x, o_x, temp_i_x, temp_g_x, temp_f_x = \
            torch.split(x_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        r_t = torch.sigmoid(r_x + r_h)
        g_t = torch.tanh(g_x + g_h)
        new_cell = c_t + self._attn(r_t, eidetic_cell, eidetic_cell)
        new_cell = self._norm_c_t(new_cell) + i_t * g_t

        new_global_memory = self.conv_gm(global_memory)
        i_m, f_m, g_m, m_m = torch.split(new_global_memory, self.num_hidden, dim=1)

        temp_i_t = torch.sigmoid(temp_i_x + i_m)
        temp_f_t = torch.sigmoid(temp_f_x + f_m + self._forget_bias)
        temp_g_t = torch.tanh(temp_g_x + g_m)
        new_global_memory = temp_f_t * torch.tanh(m_m) + temp_i_t * temp_g_t
        
        o_c = self.conv_new_cell(new_cell)
        o_m = self.conv_new_gm(new_global_memory)

        output_gate = torch.tanh(o_x + o_h + o_c + o_m)

        memory = torch.cat((new_cell, new_global_memory), 1)
        memory = self.conv_last(memory)

        output = torch.tanh(memory) * torch.sigmoid(output_gate)

        return output, new_cell, global_memory
    
class _E3DLSTM_Model(nn.Module):
    r"""E3D-LSTM Model

    Implementation of `EEidetic 3D LSTM: A Model for Video Prediction and Beyond
    <https://openreview.net/forum?id=B1lKS2AqtX>`_.

    """

    def __init__(self, num_layers, num_hidden, in_shape,
                 patch_size, stride, layer_norm=True, **kwargs):
        super(_E3DLSTM_Model, self).__init__()
        T, C, H, W = in_shape

        self.frame_channel = patch_size * patch_size * C
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        self.window_length = 2
        self.window_stride = 1

        self.height = H // patch_size
        self.width = W // patch_size
        self.MSE_criterion = nn.MSELoss()
        self.L1_criterion = nn.L1Loss()

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                Eidetic3DLSTMCell(in_channel, num_hidden[i],
                                  self.window_length, self.height, self.width, (2, 5, 5),
                                  stride, layer_norm))
        self.conv_enc = nn.Conv3d(C, self.frame_channel,
                                    kernel_size=(1, patch_size, patch_size),
                                    stride=(1, patch_size, patch_size),
                                    padding=0, bias=False)
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv3d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=(self.window_length, 1, 1),
                                   stride=(self.window_length, 1, 1), padding=0, bias=False)
        self.conv_dec = nn.ConvTranspose2d(self.frame_channel, C, kernel_size=patch_size,
                                           stride=patch_size, padding=0, bias=False)
        

    def forward(self, frames_tensor, mask_true, input_length=10, total_length=20, 
                # pre_seq_length=10, aft_seq_length=10,
                reverse_scheduled_sampling=0):
        pre_seq_length = input_length
        aft_seq_length = total_length - input_length
        # [batch, length, channel, height, width]
        frames = frames_tensor
        mask_true = mask_true[:, :, :, :self.height, :self.width]
        frames = self.conv_enc(frames.permute(0, 2, 1, 3, 4).contiguous())
        frames = frames.permute(0, 2, 1, 3, 4).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []
        c_history = []
        input_list = []

        for t in range(self.window_length - 1):
            input_list.append(
                torch.zeros_like(frames[:, 0]))

        for i in range(self.num_layers):
            zeros = torch.zeros(
                [batch, self.num_hidden[i], self.window_length, self.height, self.width]).to(frames_tensor.device)
            h_t.append(zeros)
            c_t.append(zeros)
            c_history.append(zeros)

        memory = torch.zeros(
            [batch, self.num_hidden[0], self.window_length, self.height, self.width]).to(frames_tensor.device)

        for t in range(pre_seq_length + aft_seq_length - 1):
            # reverse schedule sampling
            if reverse_scheduled_sampling == 1:
                if t == 0:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            else:
                if t < pre_seq_length:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - pre_seq_length] * frames[:, t] + \
                          (1 - mask_true[:, t - pre_seq_length]) * x_gen
        
            input_list.append(net)

            if t % (self.window_length - self.window_stride) == 0:
                net = torch.stack(input_list[t:], dim=0) # window b c h w
                net = net.permute(1, 2, 0, 3, 4).contiguous() # b c window h w

            for i in range(self.num_layers):
                if t == 0:
                    c_history[i] = c_t[i]
                else:
                    c_history[i] = torch.cat((c_history[i], c_t[i]), 1)
                
                input = net if i == 0 else h_t[i-1]
                h_t[i], c_t[i], memory = self.cell_list[i](input, h_t[i], c_t[i], memory, c_history[i])
            
            x_gen = self.conv_last(h_t[self.num_layers - 1]).squeeze(2)
            next_frames.append(self.conv_dec(x_gen))

        # [length, batch, channel, height, width] -> [batch, length, channel, height, width]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 2, 3, 4).contiguous()

        return next_frames
    

class E3DLSTM(nn.Module):
    r"""E3D-LSTM Model

    Implementation of `EEidetic 3D LSTM: A Model for Video Prediction and Beyond
    <https://openreview.net/forum?id=B1lKS2AqtX>`_.

    """

    def __init__(self, num_layers, num_hidden, in_shape,
                 patch_size, stride, layer_norm=True, **kwargs):
        super(E3DLSTM, self).__init__()
        self.net = _E3DLSTM_Model(num_layers=num_layers, num_hidden=num_hidden, in_shape=in_shape,
                                 patch_size=patch_size, stride=stride)
        

    def forward(self, frames_tensor, mask_true, input_length=10, total_length=20,
                reverse_scheduled_sampling=0, **kwargs):
        out = self.net(frames_tensor=frames_tensor, mask_true=mask_true, input_length=input_length, total_length=total_length, reverse_scheduled_sampling=reverse_scheduled_sampling)
        return out

if __name__ == "__main__":
    print('start')
    b = 16
    inp_length, pred_length = 10, 10
    total_length = inp_length + pred_length
    c = 1
    h, w = 480, 480
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') #
    input_data = torch.randn((b, total_length, c, h, w)).to(device)
    mask_true = torch.ones((b, total_length, c, h, w)).to(device)
    target = torch.randn((b, total_length-inp_length, c, h, w)).to(device)

    # print('load yaml from config')
    # import yaml
    # cfg_path = '/mnt/cache/gongjunchao/workdir/radar_forecasting/configs/hko7/ConvGRU.yaml'
    # with open(cfg_path, 'r') as cfg_file:
    #   cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)
    # backbone_kwargs = cfg_params['model']['params']['sub_model']['ConvGRU']
    backbone_kwargs = {
        'num_layers':3, 
        'num_hidden':[64, 64, 64], 
        'in_shape': [total_length, 1, h, w],
        'patch_size': 10, 
        'stride': 1
    }
    print('end')
    model = E3DLSTM(**backbone_kwargs)
    model.to(device)

    import torch.nn.functional as F
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for i in range(5):
        start.record()
        pred = model(input_data, mask_true=mask_true)
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

# srun -p ai4earth --kill-on-bad-exit=1 --quotatype=auto --gres=gpu:1 python -u e3d_lstm.py #