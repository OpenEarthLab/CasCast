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