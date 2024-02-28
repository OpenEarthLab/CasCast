if __name__ == "__main__":
    import sys
    sys.path.append('/mnt/cache/gongjunchao/workdir/radar_forecasting')

import torch
import torch.nn as nn

from einops import rearrange

class CGRU_cell(nn.Module):
    """
    ConvGRU Cell
    """
    def __init__(self, shape, input_channels, h2h_filter_size, i2h_filter_size,
                 h2h_pad, i2h_pad,
                 num_features, residual_connection=True, **kwargs):
        super(CGRU_cell, self).__init__()
        self.shape = (shape, shape)
        self.input_channels = input_channels
        self.residual_connection = residual_connection

        if self.residual_connection:
            self.res_conv = nn.Conv2d(in_channels=self.input_channels, out_channels=num_features,
                                       kernel_size=1, stride=1, padding=0)

        # kernel_size of input_to_state equals state_to_state
        self.h2h_filter_size = h2h_filter_size
        self.i2h_filter_size = i2h_filter_size

        self.num_features = num_features
        self.h2h_pad = h2h_pad
        self.i2h_pad = i2h_pad
        self.conv_i2h = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channels,
                      out_channels=self.num_features*3, kernel_size=self.i2h_filter_size, 
                      stride=1, padding=self.i2h_pad),
            nn.GroupNorm(self.num_features*3 // 8, self.num_features*3))
        self.conv_h2h = nn.Sequential(
            nn.Conv2d(self.num_features,
                      self.num_features*3, self.h2h_filter_size, 1,
                      padding=self.h2h_pad),
            nn.GroupNorm(self.num_features*3 // 8, self.num_features*3))

    def forward(self, inputs=None, hidden_state=None, seq_len=10):
        """
        inputs: (b, t, c, h, w)
        """
        # seq_len=10 for moving_mnist
        if hidden_state is None:
            htprev = torch.zeros(inputs.shape[0], self.num_features,
                                 self.shape[0], self.shape[1]).to(inputs.device)
        else:
            htprev = hidden_state
        output_inner = []
        for index in range(seq_len):
            if inputs is None:
                x = torch.zeros(htprev.shape[0], self.input_channels,
                                self.shape[0], self.shape[1]).to(htprev.device)
            else:
                x = inputs[:, index, ...]
            i2h = self.conv_i2h(x)  # W * X_t, dim(b, c, h, w) 
            h2h = self.conv_h2h(htprev)  # U * H_t-1, dim(b, c, h, w)

            i2h_u, i2h_r, i2h_m = torch.chunk(i2h, 3, dim=1) ## chunk along dim c
            h2h_u, h2h_r, h2h_m = torch.chunk(h2h, 3, dim=1)
            update_state = torch.sigmoid(i2h_u + h2h_u) # Z_t
            reset_state = torch.sigmoid(i2h_r + h2h_r) # R_t
            mem_state = torch.tanh(i2h_m + reset_state * h2h_m) # H~_t

            htnext = (1 - update_state) * mem_state + update_state * htprev # H_t
            htprev = htnext

            if self.residual_connection:
                output_inner.append(htnext + self.res_conv(x))
            else:
                output_inner.append(htnext)

        output_inner = torch.stack(output_inner, dim=1) ## b, t, c, h, w
        return output_inner, htnext


class _ConvGRU(nn.Module):
    def __init__(self, stack_num, hidden_dims,
                 first_conv, last_deconv, 
                 h2h_kernel_size=[], h2h_pad=[], 
                 i2h_kernel_size=[], i2h_pad=[],
                 downsample=[], upsample=[],
                 featmap_size = [96, 32, 16], 
                 residual_connection=True, **kwargs):
        """
        __C.MODEL.ENCODER_FORECASTER.FIRST_CONV = (8, 7, 5, 1)  # Num filter, kernel, stride, pad
        __C.MODEL.ENCODER_FORECASTER.LAST_DECONV = (8, 7, 5, 1)  # Num filter, kernel, stride, pad
        __C.MODEL.ENCODER_FORECASTER.FEATMAP_SIZE = [96, 32, 16]
        __C.MODEL.ENCODER_FORECASTER.DOWNSAMPLE = [(5, 3, 1),
                                           (3, 2, 1)]  # (kernel, stride, pad) for conv2d
        __C.MODEL.ENCODER_FORECASTER.UPSAMPLE = [(5, 3, 1),
                                                (4, 2, 1)]  # (kernel, stride, pad) for deconv2d
        __C.MODEL.ENCODER_FORECASTER.RNN_BLOCKS.RES_CONNECTION = True
        __C.MODEL.ENCODER_FORECASTER.RNN_BLOCKS.LAYER_TYPE = ["ConvGRU", "ConvGRU", "ConvGRU"]
        __C.MODEL.ENCODER_FORECASTER.RNN_BLOCKS.STACK_NUM = [2, 3, 3]
        # These features are used for both ConvGRU
        __C.MODEL.ENCODER_FORECASTER.RNN_BLOCKS.NUM_FILTER = [32, 64, 64]
        __C.MODEL.ENCODER_FORECASTER.RNN_BLOCKS.H2H_KERNEL = [(5, 5), (5, 5), (3, 3)]
        __C.MODEL.ENCODER_FORECASTER.RNN_BLOCKS.H2H_DILATE = [(1, 1), (1, 1), (1, 1)]
        __C.MODEL.ENCODER_FORECASTER.RNN_BLOCKS.I2H_KERNEL = [(3, 3), (3, 3), (3, 3)]
        __C.MODEL.ENCODER_FORECASTER.RNN_BLOCKS.I2H_PAD = [(1, 1), (1, 1), (1, 1)]
        """
        super().__init__()
        assert len(hidden_dims) == stack_num, "hidden_dims must have the same length as stack_num"
        self.stack_num = stack_num
        self.featmap_size = featmap_size

        self.first_conv = nn.Conv2d(in_channels=1, out_channels=first_conv[0], kernel_size=first_conv[1], stride=first_conv[2], padding=first_conv[3])
        self.last_deconv = nn.ConvTranspose2d(in_channels=hidden_dims[0], out_channels=1, kernel_size=last_deconv[1], stride=last_deconv[2], padding=last_deconv[3])

        self.encoder = nn.ModuleList()
        for i in range(stack_num):
            self.encoder.append(CGRU_cell(shape=self.featmap_size[i], input_channels=first_conv[0] if i==0 else hidden_dims[i], 
                                          h2h_filter_size=h2h_kernel_size[i], h2h_pad=h2h_pad[i],
                                          i2h_filter_size=i2h_kernel_size[i], i2h_pad=i2h_pad[i],
                                          num_features=hidden_dims[i], residual_connection=residual_connection if i != stack_num-1 else False))
            if i != stack_num - 1:
                ## downsample
                self.encoder.append(nn.Conv2d(in_channels=hidden_dims[i], out_channels=hidden_dims[i+1], kernel_size=downsample[i][0], stride=downsample[i][1], padding=downsample[i][2]))

        self.forecaster = nn.ModuleList()
        for i in range(stack_num):
            inverse_i = stack_num - i - 1
            self.forecaster.append(CGRU_cell(shape=self.featmap_size[inverse_i], input_channels=hidden_dims[inverse_i],
                                              h2h_filter_size=h2h_kernel_size[inverse_i], h2h_pad=h2h_pad[inverse_i],
                                              i2h_filter_size=i2h_kernel_size[inverse_i], i2h_pad=i2h_pad[inverse_i],
                                              num_features=hidden_dims[inverse_i], residual_connection=residual_connection))
            if inverse_i != 0 :
                ## upsample ##
                self.forecaster.append(nn.ConvTranspose2d(in_channels=hidden_dims[inverse_i], out_channels=hidden_dims[inverse_i-1], kernel_size=upsample[inverse_i-1][0], stride=upsample[inverse_i-1][1], padding=upsample[inverse_i-1][2]))

    def forward(self, inputs, pred_len=10):
        """
        inputs: (b, t, c, h, w)
        """
        b, t, c, h, w = inputs.size()
        inputs = rearrange(inputs, 'b t c h w -> (b t) c h w')
        x = self.first_conv(inputs)
        x = rearrange(x, '(b t) c h w -> b t c h w', b=b, t=t)

        encoded_h_next = []
        ## rnn encoding ##
        for i, layer in enumerate(self.encoder):
            if isinstance(layer, CGRU_cell):
                output_inner, h_next = layer(inputs=x if i ==0 else down_output_inner, hidden_state=None, seq_len=t) 
                encoded_h_next.append(h_next)
            elif isinstance(layer, nn.Conv2d):
                ## check the size of output_inner
                output_inner = rearrange(output_inner, 'b t c h w -> (b t) c h w')
                down_output_inner = layer(output_inner) ## resize to (b, t, c, h, w)
                down_output_inner = rearrange(down_output_inner, '(b t) c h w -> b t c h w', b=b, t=t)
            else:
                raise NotImplementedError
        
        ## rnn decoding ##
        gru_count = 0
        for i, layer in enumerate(self.forecaster):
            if isinstance(layer, CGRU_cell):
                output_inner, _ = layer(inputs=None if i == 0 else up_output_inner, hidden_state=encoded_h_next[-gru_count-1], seq_len=pred_len)
                gru_count += 1
            elif isinstance(layer, nn.ConvTranspose2d):
                output_inner = rearrange(output_inner, 'b t c h w -> (b t) c h w')
                up_output_inner = layer(output_inner) ##resize to (b, t, c, h, w)
                up_output_inner = rearrange(up_output_inner, '(b t) c h w -> b t c h w', b=b, t=pred_len)
            else:
                raise NotImplementedError
        
        ## output ##
        output_inner = rearrange(output_inner, 'b t c h w -> (b t) c h w')
        pred = torch.sigmoid(self.last_deconv(output_inner))
        pred = rearrange(pred, '(b t) c h w -> b t c h w', b=b, t=pred_len)

        return pred


class ConvGRU(nn.Module):
    def __init__(self, stack_num, hidden_dims,
                 first_conv, last_deconv, 
                 h2h_kernel_size=[], h2h_pad=[], 
                 i2h_kernel_size=[], i2h_pad=[],
                 downsample=[], upsample=[],
                 featmap_size = [96, 32, 16], 
                 residual_connection=True, **kwargs):
        super().__init__()
        self.net = _ConvGRU(stack_num=stack_num, hidden_dims=hidden_dims, first_conv=first_conv,
                            last_deconv=last_deconv, h2h_kernel_size=h2h_kernel_size, h2h_pad=h2h_pad,
                            i2h_kernel_size=i2h_kernel_size, i2h_pad=i2h_pad,
                            downsample=downsample, upsample=upsample,
                            featmap_size=featmap_size, residual_connection=residual_connection)
    
    def forward(self, data, pred_len, **kwargs):
        out = self.net(data, pred_len)
        return out
    

if __name__ == "__main__":
    # earth_patch_size_latitude((128,256), (4,4))
    print('start')
    b = 1
    t, pred_len = 5, 20
    c = 1
    h, w = 480, 480
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    input_data = torch.randn((b, t, c, h, w)).to(device)
    target = torch.randn((b, pred_len, c, h, w)).to(device)

    print('load yaml from config')
    import yaml
    cfg_path = '/mnt/cache/gongjunchao/workdir/radar_forecasting/configs/hko7/ConvGRU.yaml'
    with open(cfg_path, 'r') as cfg_file:
      cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)
    backbone_kwargs = cfg_params['model']['params']['sub_model']['ConvGRU']

    print('end')
    model = ConvGRU(**backbone_kwargs)
    model.to(device)

    import torch.nn.functional as F
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for i in range(5):
        start.record()
        pred = model(input_data, pred_len=pred_len)
        loss = F.mse_loss(pred, target)
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
    flops = FlopCountAnalysis(model, (input_data, pred_len))
    print(flop_count_table(flops))

# srun -p ai4earth --kill-on-bad-exit=1 --quotatype=auto --gres=gpu:1 python -u ConvGRU.py #