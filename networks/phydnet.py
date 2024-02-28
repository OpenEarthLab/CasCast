import random
import torch
from torch import nn

from openstl.modules import PhyCell, PhyD_ConvLSTM, PhyD_EncoderRNN, K2M


class PhyDNet_Model(nn.Module):
    r"""PhyDNet Model

    Implementation of `Disentangling Physical Dynamics from Unknown Factors for
    Unsupervised Video Prediction <https://arxiv.org/abs/2003.01460>`_.

    """

    def __init__(self, configs, **kwargs):
        super(PhyDNet_Model, self).__init__()
        self.pre_seq_length = configs.pre_seq_length
        self.aft_seq_length = configs.aft_seq_length
        _, C, H, W = configs.in_shape
        patch_size = configs.patch_size #if configs.patch_size in [2, 4] else 4
        input_shape = (H // patch_size, W // patch_size)

        self.phycell = PhyCell(input_shape=input_shape, input_dim=64, F_hidden_dims=[49],
                               n_layers=1, kernel_size=(7,7))
        self.convcell = PhyD_ConvLSTM(input_shape=input_shape, input_dim=64, hidden_dims=[128,128,128,64],
                                      n_layers=4, kernel_size=(3,3))
        self.encoder = PhyD_EncoderRNN(self.phycell, self.convcell,
                                       in_channel=C, patch_size=patch_size)
        self.k2m = K2M([7,7])
    
    # def forward(self, input_tensor, target_tensor, constraints, teacher_forcing_ratio=0.0):
    #     loss = 0
    #     for ei in range(self.pre_seq_length - 1):
    #         _, _, output_image, _, _ = self.encoder(input_tensor[:,ei,:,:,:], (ei==0))
    #         loss += self.criterion(output_image, input_tensor[:,ei+1,:,:,:])

    #     decoder_input = input_tensor[:,-1,:,:,:]
    #     use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    #     for di in range(self.aft_seq_length):
    #         _, _, output_image, _, _ = self.encoder(decoder_input)
    #         target = target_tensor[:,di,:,:,:]
    #         loss += self.criterion(output_image, target)
    #         if use_teacher_forcing:
    #             decoder_input = target
    #         else:
    #             decoder_input = output_image

    #     for b in range(0, self.encoder.phycell.cell_list[0].input_dim):
    #         filters = self.encoder.phycell.cell_list[0].F.conv1.weight[:,b,:,:]
    #         m = self.k2m(filters.double()).float()
    #         loss += self.criterion(m, constraints)

    #     return loss

    def forward(self, input_tensor, target_tensor, teacher_forcing_ratio=0.0):
        next_frames = []
        moments = []
        for ei in range(self.pre_seq_length - 1):
            _, _, output_image, _, _ = self.encoder(input_tensor[:,ei,:,:,:], (ei==0))
            next_frames.append(output_image)

        decoder_input = input_tensor[:,-1,:,:,:]
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        for di in range(self.aft_seq_length):
            _, _, output_image, _, _ = self.encoder(decoder_input)
            next_frames.append(output_image)
            target = target_tensor[:,di,:,:,:]
            if use_teacher_forcing:
                decoder_input = target
            else:
                decoder_input = output_image

        for b in range(0, self.encoder.phycell.cell_list[0].input_dim):
            filters = self.encoder.phycell.cell_list[0].F.conv1.weight[:,b,:,:]
            m = self.k2m(filters).float()
            moments.append(m)
        next_frames = torch.stack(next_frames, dim=1)
        moments = torch.stack(moments, dim=1)
        return [next_frames, moments]

    # def inference(self, input_tensor, target_tensor, constraints, **kwargs):
    #     with torch.no_grad():
    #         loss = 0
    #         for ei in range(self.pre_seq_length - 1):
    #             encoder_output, encoder_hidden, output_image, _, _  = \
    #                 self.encoder(input_tensor[:,ei,:,:,:], (ei==0))
    #             if kwargs.get('return_loss', True):
    #                 loss += self.criterion(output_image, input_tensor[:,ei+1,:,:,:])

    #         decoder_input = input_tensor[:,-1,:,:,:]
    #         predictions = []

    #         for di in range(self.aft_seq_length):
    #             _, _, output_image, _, _ = self.encoder(decoder_input, False, False)
    #             decoder_input = output_image
    #             predictions.append(output_image)
    #             if kwargs.get('return_loss', True):
    #                 loss += self.criterion(output_image, target_tensor[:,di,:,:,:])

    #         for b in range(0, self.encoder.phycell.cell_list[0].input_dim):
    #             filters = self.encoder.phycell.cell_list[0].F.conv1.weight[:,b,:,:]
    #             m = self.k2m(filters.double()).float()
    #             if kwargs.get('return_loss', True):
    #                 loss += self.criterion(m, constraints)

    #         return torch.stack(predictions, dim=1), loss

class Dict(dict):
            __setattr__ = dict.__setitem__
            __getattr__ = dict.__getitem__

class PhyDNet(nn.Module):
    def __init__(self, configs, **kwargs) -> None:
        super().__init__()
        self.net = PhyDNet_Model(self.dictToObj(configs), **kwargs)
    
    def forward(self, input_tensor, target_tensor, teacher_forcing_ratio=0.0):
        out = self.net(input_tensor, target_tensor, teacher_forcing_ratio)
        return out

    def dictToObj(self, dictObj):
        if not isinstance(dictObj, dict):
            return dictObj
        d = Dict()
        for k, v in dictObj.items():
            d[k] = self.dictToObj(v)
        return d
    
    # def inference(self, input_tensor, target_tensor, **kwargs):
    ## equal to teacher_foracing_ratio = 0.0
    #     out = self.net.inference(input_tensor, target_tensor, **kwargs)
    #     return out

if __name__ == "__main__":
    print('start')
    b = 8
    inp_len, pred_len = 12, 12
    total_length = inp_len + pred_len
    c = 1
    h, w = 400, 400
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    input_data = torch.randn((b, inp_len, c, h, w)).to(device)
    target = torch.randn((b, pred_len, c, h, w)).to(device)
    mask_true = torch.ones((b, total_length, c, h, w)).to(device)
    print('load yaml from config')
    import yaml
    cfg_path = '/mnt/cache/gongjunchao/workdir/radar_forecasting/configs/meteonet/PhyDNet.yaml'
    with open(cfg_path, 'r') as cfg_file:
      cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)
    backbone_kwargs = cfg_params['model']['params']['sub_model']['PhyDNet']

    ## input_shape, input_dim, hidden_dims, n_layers, kernel_size,device
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
    #     'configs': dictToObj({'in_shape': [inp_len, c, h, w],
    #                 'patch_size': 4,
    #                 'pre_seq_length': inp_len,
    #                  'aft_seq_length': pred_len
    #                    })
    # }
    print('end')
    # def __init__(self, num_layers, num_hidden, configs, **kwargs):
    model = PhyDNet(**backbone_kwargs)
    model.to(device)
    
    def _get_constraints():
        constraints = torch.zeros((49, 7, 7)).to(device)
        ind = 0
        for i in range(0, 7):
            for j in range(0, 7):
                constraints[ind,i,j] = 1
                ind +=1
        return constraints 
    
    constraints = _get_constraints()
    constraints = constraints.unsqueeze(1)
    constraints = constraints.expand(49, 64, 7, 7).to(device)
    import torch.nn.functional as F
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for i in range(5):
        start.record()
        pred, moments = model(input_tensor=input_data, target_tensor=target)
        loss_pred = F.mse_loss(pred, torch.cat([input_data[:, 1:], target], dim=1))
        loss_moments = F.mse_loss(moments, constraints)
        loss = loss_pred + loss_moments
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
    flops = FlopCountAnalysis(model, (input_data, target))
    print(flop_count_table(flops))

    # srun -p ai4earth --kill-on-bad-exit=1 --quotatype=auto --gres=gpu:1 python -u phydnet.py #