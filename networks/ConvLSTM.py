import torch
import torch.nn as nn

from openstl.modules import ConvLSTMCell


class ConvLSTM_Model(nn.Module):
    r"""ConvLSTM Model

    Implementation of `Convolutional LSTM Network: A Machine Learning Approach
    for Precipitation Nowcasting <https://arxiv.org/abs/1506.04214>`_.

    """

    def __init__(self, num_layers, num_hidden, configs, **kwargs):
        super(ConvLSTM_Model, self).__init__()
        T, C, H, W = configs.in_shape

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * C
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        self.height = H // configs.patch_size
        self.width = W // configs.patch_size
        self.MSE_criterion = nn.MSELoss()

        self.encoder = nn.Conv2d(C, self.frame_channel, kernel_size=configs.patch_size, 
                                 stride=configs.patch_size, padding=0)
        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                ConvLSTMCell(in_channel, num_hidden[i], self.height, self.width, configs.filter_size,
                                       configs.stride, configs.layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)
        
        self.decoder = nn.ConvTranspose2d(self.frame_channel, C, kernel_size=configs.patch_size, 
                                          stride=configs.patch_size, padding=0)

    def forward(self, frames_tensor, mask_true, **kwargs):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        # frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        # mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()
        frames = frames_tensor

        batch = frames.shape[0]

        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], self.height, self.width]).to(frames.device)
            h_t.append(zeros)
            c_t.append(zeros)

        for t in range(self.configs.pre_seq_length + self.configs.aft_seq_length - 1):
            # reverse schedule sampling
            if self.configs.reverse_scheduled_sampling == 1:
                if t == 0:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            else:
                if t < self.configs.pre_seq_length:
                    net = frames[:, t]
                else:
                    try:
                        net = mask_true[:, t - self.configs.pre_seq_length] * frames[:, t] + \
                            (1 - mask_true[:, t - self.configs.pre_seq_length]) * x_gen
                    except:
                        import pdb; pdb.set_trace()

            net = self.encoder(net)
            h_t[0], c_t[0] = self.cell_list[0](net, h_t[0], c_t[0])

            for i in range(1, self.num_layers):
                h_t[i], c_t[i] = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i])

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            x_gen = self.decoder(x_gen)
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=1)
        # if kwargs.get('return_loss', True):
        #     loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])
        # else:
        #     loss = None

        return next_frames

class Dict(dict):
            __setattr__ = dict.__setitem__
            __getattr__ = dict.__getitem__

class ConvLSTM(nn.Module):
    #  __init__(self, num_layers, num_hidden, configs, **kwargs):

    def __init__(self, num_layers, num_hidden, configs,  **kwargs) -> None:
        super().__init__()
        self.net = ConvLSTM_Model(num_layers, num_hidden, self.dictToObj(configs), **kwargs)
    
    def forward(self, frames_tensor, mask_true, **kwargs):

        return self.net(frames_tensor, mask_true)
    
    def dictToObj(self, dictObj):
        if not isinstance(dictObj, dict):
            return dictObj
        d = Dict()
        for k, v in dictObj.items():
            d[k] = self.dictToObj(v)
        return d

    

if __name__ == '__main__':
    print('start')
    b = 8
    inp_len, pred_len = 12, 12
    total_length = inp_len + pred_len
    c = 1
    h, w = 400, 400
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    input_data = torch.randn((b, inp_len+pred_len, c, h, w)).to(device)
    target = torch.randn((b, pred_len, c, h, w)).to(device)
    mask_true = torch.ones((b, total_length, c, h, w)).to(device)
    print('load yaml from config')
    import yaml
    cfg_path = '/mnt/cache/gongjunchao/workdir/radar_forecasting/configs/meteonet/ConvLSTM.yaml'
    with open(cfg_path, 'r') as cfg_file:
      cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)
    backbone_kwargs = cfg_params['model']['params']['sub_model']['ConvLSTM']

    # ## input_shape, input_dim, hidden_dims, n_layers, kernel_size,device
    # class Dict(dict):
    #     __setattr__ = dict.__setitem__
    #     __getattr__ = dict.__getitem__
    # def dictToObj(dictObj):
    #     if not isinstance(dictObj, dict):
    #         return dictObj
    #     d = Dict()
    #     for k, v in dictObj.items():
    #         d[k] = dictToObj(v)
    #     return d

    # backbone_kwargs = {
    #     'num_layers': 4, #[inp_len, c, h, w]
    #     'num_hidden': [128, 128, 128 ,128],
    #     'configs': dictToObj({'in_shape': [inp_len, c, h, w],
    #                 'patch_size': 10,
    #                 'filter_size': 5,
    #                 'stride': 1,
    #                 'layer_norm': 0,
    #                 'pre_seq_length': inp_len,
    #                  'aft_seq_length': pred_len,
    #                  'reverse_scheduled_sampling': 0,
    #                    })
    # }
    print('end')
    # def __init__(self, num_layers, num_hidden, configs, **kwargs):
    model = ConvLSTM(**backbone_kwargs)
    model.to(device)

    import torch.nn.functional as F
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for i in range(5):
        start.record()
        pred = model(input_data, mask_true)
        # loss = F.mse_loss(pred, input_data[:, 1:])
        loss = F.mse_loss(pred, input_data[:, 1:])
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
    flops = FlopCountAnalysis(model, (input_data, mask_true))
    print(flop_count_table(flops))

    # srun -p ai4earth --kill-on-bad-exit=1 --quotatype=auto --gres=gpu:1 python -u ConvLSTM.py #