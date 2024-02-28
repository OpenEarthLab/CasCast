if __name__ == "__main__":
    import sys
    sys.path.append('/mnt/cache/gongjunchao/workdir/radar_forecasting')
import torch
import torch.nn as nn
from networks.utils.SpatioTemporalLSTMCell import SpatioTemporalLSTMCellv2 as stlstm
from networks.utils.MIMBlock import MIMBlock as mimblock
from networks.utils.MIMN import MIMN as mimn
from networks.utils.MIMBlock import MIMBlockPl as mimblock_pl
from networks.utils.MIMN import MIMNPl as mimn_pl
from networks.utils.MIMBlock import MIMBlockOA as mimblock_oa
from networks.utils.MIMN import MIMNOA as mimn_oa

class MyObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)
            
class mimBase(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(mimBase, self).__init__()
        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * configs.img_channel
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.patch_size = configs.patch_size
        height = (configs.img_height-1) // configs.patch_size + 1
        width = (configs.img_width-1) // configs.patch_size + 1
        self.hw_new = [height, width]

        self.stlstm_layer = []
        self.stlstm_layer_diff = []
        
		## encoder and decoder ##
        self.input_encoder = nn.Conv2d(in_channels=configs.img_channel, out_channels=self.frame_channel, 
                                       kernel_size=configs.patch_size, stride=configs.patch_size, padding=0,
								 		bias=False)
        self.decoder = nn.ConvTranspose2d(in_channels=self.frame_channel, out_channels=configs.img_channel, 
                                          kernel_size=configs.patch_size, stride=configs.patch_size, padding=0)

    def forward(self, frames, mask_true, input_length, total_length):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        batch, length, n_channel, height, width = frames.shape
        lstm_c, cell_state, hidden_state, cell_state_diff, hidden_state_diff = [], [], [], [], []
        device = frames.device
        
		## init states ##
        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], *self.hw_new]).to(device=device)
            cell_state.append(zeros)
            hidden_state.append(zeros)
        for i in range(self.num_layers-1):
            zeros = torch.zeros([batch, self.num_hidden[i], *self.hw_new]).to(device=device)
            lstm_c.append(zeros)
            cell_state_diff.append(zeros)
            hidden_state_diff.append(zeros)

        st_memory = torch.zeros([batch, self.num_hidden[0], *self.hw_new]).to(device=device)
        next_frames = []

        for t in range(total_length-1):
            if t < input_length:
                x_gen = frames[:, t]
            else:
                x_gen = mask_true[:, t - input_length] * frames[:, t] + \
                      (1 - mask_true[:, t - input_length]) * x_gen

			## encode x_gen ##
            x_gen = self.input_encoder(x_gen) 
            preh = hidden_state[0]
            hidden_state[0], cell_state[0], st_memory = self.stlstm_layer[0](
                x_gen, hidden_state[0], cell_state[0], st_memory)
            for i in range(1, self.num_layers):
                if t > 0:
                    if i == 1:
                        hidden_state_diff[i - 1], cell_state_diff[i - 1] = self.stlstm_layer_diff[i - 1](
                            hidden_state[i - 1] - preh, hidden_state_diff[i - 1], cell_state_diff[i - 1])
                    else:
                        hidden_state_diff[i - 1], cell_state_diff[i - 1] = self.stlstm_layer_diff[i - 1](
                            hidden_state_diff[i - 2], hidden_state_diff[i - 1], cell_state_diff[i - 1])
                else:
                    self.stlstm_layer_diff[i - 1](torch.zeros_like(hidden_state[i - 1]).to(device=device),
                                                  torch.zeros_like(hidden_state[i - 1]).to(device=device),
                                                  torch.zeros_like(hidden_state[i - 1]).to(device=device))
                preh = hidden_state[i]
                hidden_state[i], cell_state[i], st_memory, lstm_c[i-1] = self.stlstm_layer[i](
                    hidden_state[i - 1], hidden_state_diff[i - 1], hidden_state[i],
                    cell_state[i], st_memory, lstm_c[i-1])
            x_gen = self.conv_last(hidden_state[self.num_layers-1])
            ## decode x_gen ##
            x_gen = self.decoder(x_gen)

            next_frames.append(x_gen)

        next_frames = torch.stack(next_frames, dim=1).contiguous()
        return next_frames


class mim(mimBase):
    def __init__(self, num_layers, num_hidden, configs):
        super(mim, self).__init__(num_layers, num_hidden, configs)

        for i in range(num_layers):
            if i == 0:
                num_hidden_in = self.frame_channel
            else:
                num_hidden_in = num_hidden[i - 1]
            if i < 1:
                new_stlstm_layer = stlstm(num_hidden_in,
                                          num_hidden[i],
                                          self.hw_new,
                                          configs.filter_size,
                                          configs.stride,
                                          tln=configs.norm) #.to(device=device)
            else:
                new_stlstm_layer = mimblock(num_hidden_in,
                                            num_hidden[i],
                                            self.hw_new,
                                            configs.filter_size,
                                            configs.stride,
                                            tln=configs.norm) #.to(device=device)
            self.stlstm_layer.append(new_stlstm_layer)

        for i in range(num_layers - 1):
            new_stlstm_layer = mimn(num_hidden[i + 1],
                                    self.hw_new,
                                    configs.filter_size,
                                    tln=configs.norm) #.to(device=device)
            self.stlstm_layer_diff.append(new_stlstm_layer)
        self.stlstm_layer = nn.ModuleList(self.stlstm_layer)
        self.stlstm_layer_diff = nn.ModuleList(self.stlstm_layer_diff)
        self.conv_last = nn.Conv2d(num_hidden[num_layers-1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False) #.to(device=device)


class mimpl(mimBase):
    def __init__(self, num_layers, num_hidden, configs):
        super(mimpl, self).__init__(num_layers, num_hidden, configs)

        order = 2 if 'plq' in configs.model_name else 3

        for i in range(num_layers):
            if i == 0:
                num_hidden_in = self.frame_channel
            else:
                num_hidden_in = num_hidden[i - 1]
            if i < 1:
                new_stlstm_layer = stlstm(num_hidden_in, num_hidden[i], self.hw_new, configs.filter_size,
                                          configs.stride, tln=1).to(device=device)
            elif i % 2 == 1:
                new_stlstm_layer = mimblock_pl(num_hidden_in, num_hidden[i], self.hw_new, configs.filter_size,
                                            configs.stride, configs.device, tln=configs.norm, order=order).to(
                    device=device)
            else:
                new_stlstm_layer = mimblock(num_hidden_in, num_hidden[i], self.hw_new, configs.filter_size,
                                            configs.stride, configs.device, tln=1).to(device=device)
            self.stlstm_layer.append(new_stlstm_layer)

        for i in range(num_layers - 1):
            if i % 2 == 1:
                new_stlstm_layer = mimn_pl(num_hidden[i + 1], self.hw_new, configs.filter_size,
                                           tln=configs.norm, order=order).to(device=device)
            else:
                new_stlstm_layer = mimn(num_hidden[i + 1], self.hw_new, configs.filter_size,
                                        tln=1).to(device=device)
            self.stlstm_layer_diff.append(new_stlstm_layer)
        self.stlstm_layer = nn.ModuleList(self.stlstm_layer)
        self.stlstm_layer_diff = nn.ModuleList(self.stlstm_layer_diff)

        self.conv_last = nn.Conv2d(num_hidden[num_layers-1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False).to(device=device)


class mimoa(mimBase):
    def __init__(self, num_layers, num_hidden, configs):
        super(mimoa, self).__init__(num_layers, num_hidden, configs)
        if configs.activator == 'relu':
            activator = nn.ReLU()
        elif configs.activator == 'sigmoid':
            activator = nn.Sigmoid()
        else:
            print('please define the activation function you use here')
            exit(1)

        for i in range(num_layers):
            if i == 0:
                num_hidden_in = self.frame_channel
            else:
                num_hidden_in = num_hidden[i - 1]
            if i < 1:
                new_stlstm_layer = stlstm(num_hidden_in, num_hidden[i], self.hw_new, configs.filter_size,
                                          configs.stride, tln=1).to(device=device)
            elif i % 2 == 1:
                new_stlstm_layer = mimblock_oa(num_hidden_in, num_hidden[i], self.hw_new, configs.filter_size,
                                            configs.stride, configs.device, tln=configs.norm, act=activator).to(
                    device=device)
            else:
                new_stlstm_layer = mimblock(num_hidden_in, num_hidden[i], self.hw_new, configs.filter_size,
                                            configs.stride, configs.device, tln=1).to(device=device)
            self.stlstm_layer.append(new_stlstm_layer)

        for i in range(num_layers - 1):
            if i % 2 == 1:
                new_stlstm_layer = mimn_oa(num_hidden[i + 1], self.hw_new, configs.filter_size,
                                           tln=configs.norm, act=activator).to(device=device)
            else:
                new_stlstm_layer = mimn(num_hidden[i + 1], self.hw_new, configs.filter_size,
                                        tln=1).to(device=device)
            self.stlstm_layer_diff.append(new_stlstm_layer)
        self.stlstm_layer = nn.ModuleList(self.stlstm_layer)
        self.stlstm_layer_diff = nn.ModuleList(self.stlstm_layer_diff)

        self.conv_last = nn.Conv2d(num_hidden[num_layers-1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False).to(device=device)

	

class MIM(nn.Module):
	def __init__(self, num_layers, num_hidden, configs, **kwargs) -> None:
		super().__init__()
		self.net = mim(num_layers=num_layers, num_hidden=num_hidden, configs=MyObject(configs))
	
	def forward(self, frames_tensor, mask_true, input_length=10, total_length=20, reverse_scheduled_sampling=0, **kwargs):
		"""
		Args:
			frame_tensor: tensor of shape (batch_size, total_length, c, h, w)
			mask_true: tensor of shape (batch_size, total_length, c, h, w)
			input_length: int
			total_length: int
			reverse_scheduled_sampling: float
		Returns:
			pred_tensor: tensor of shape (batch_size, pred_length, c, h, w)
		"""
		# frame_tensor: (batch_size, total_length, c, h, w)
		# mask_true: (batch_size, total_length, c, h, w)
		# pred_tensor: (batch_size, total_length-1, c, h, w)
		pred_tensor = self.net(frames_tensor, mask_true, input_length=input_length, total_length=total_length)
		return pred_tensor

if __name__ == "__main__":
    print('start')
    b = 1
    inp_length, pred_length = 13, 12
    total_length = inp_length + pred_length
    c = 1
    h, w = 384, 384
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') #
    input_data = torch.randn((b, total_length, c, h, w)).to(device)
    mask_true = torch.ones((b, total_length, c, h, w)).to(device)
    target = torch.randn((b, total_length-inp_length, c, h, w)).to(device)

    print('load yaml from config')
    import yaml
    cfg_path = '/mnt/cache/gongjunchao/workdir/radar_forecasting/configs/sevir/MIM.yaml'
    with open(cfg_path, 'r') as cfg_file:
      cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)
    backbone_kwargs = cfg_params['model']['params']['sub_model']['MIM']
    # backbone_kwargs = {
    #     'num_layers': 4,
    #     'num_hidden': [64, 64, 64, 64],
    #     'configs': {'patch_size':10, 'img_channel':1,
    #                 'img_height':h, 'img_width':w,
    #             	'filter_size':3,
    #                 'stride':1, 'norm':1}
    #     # 'configs': {'patch_size':10, 'img_channel':1,
    #     #             'img_height':h, 'img_width':w,
    #     #             'input_length':10, 'total_length':20,
    #     #             'device':device, 'filter_size':3,
    #     #             'stride':1, 'norm':1}
    # }
    print('end')
    model = MIM(**backbone_kwargs)
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
    flops = FlopCountAnalysis(model, (input_data, mask_true))
    print(flop_count_table(flops))

# srun -p ai4earth --kill-on-bad-exit=1 --quotatype=auto --gres=gpu:1 python -u MIM.py #