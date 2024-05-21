import torch
import torch.nn as nn
from networks.prediff.taming.autoencoder_kl import AutoencoderKL


class autoencoder_kl(nn.Module):
    def __init__(self, config):
        super(autoencoder_kl, self).__init__()
        self.config = config
        self.net = AutoencoderKL(**config)

    def forward(self, sample, sample_posterior=True, return_posterior=True, generator=None):
        pred, posterior = self.net(sample, sample_posterior, return_posterior, generator)
        out = [pred, posterior]
        return out


