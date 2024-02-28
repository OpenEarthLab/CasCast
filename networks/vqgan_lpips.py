if __name__ == "__main__":
    import sys
    sys.path.append("/mnt/cache/gongjunchao/workdir/radar_forecasting")

"""
Code is adopted from `LPIPSWithDiscriminator` in https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/losses/contperceptual.py.
Enable `channels != 3`.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

from networks.prediff.taming.losses.lpips import LPIPS
from networks.prediff.taming.losses.model import NLayerDiscriminator, weights_init


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


class vqgan_lpips(nn.Module):
    def __init__(self, disc_start, logvar_init=0.0,  pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_loss="hinge"):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.pixel_weight = pixelloss_weight
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, inputs, reconstructions, optimizer_idx,
                global_step, mask=None, last_layer=None, cond=None, split="train",
                weights=None):
        r"""
        Changes compared with original implementation:
        1. use `inputs = inputs.contiguous()` and `reconstructions = reconstructions.contiguous()` to avoid later duplicated `.contiguous()`.
        2. only feed RGB channels into `self.perceptual_loss()`.
        3. add `mask`

        Parameters
        ----------
        inputs, reconstructions:    torch.Tensor
            shape = (b, c, h, w)
            channels should be ["red", "green", "blue", ...]
        mask:   torch.Tensor
            shape = (b, 1, h, w)
            1 for non-masking, 0 for masking

        Returns
        -------
        loss, log
        """
        batch_size = inputs.shape[0]
        inputs = inputs.contiguous()
        reconstructions = reconstructions.contiguous()
        if mask is not None:
            # TODO: how to handle corrupted pixels? E.g. recons without mask?
            inputs = inputs * mask
            reconstructions = reconstructions * mask

        rec_loss = torch.abs(inputs - reconstructions)
        if self.perceptual_weight > 0:
            # Only RGB channels
            p_loss = self.perceptual_loss(inputs[:, :3, ...], reconstructions[:, :3, ...])
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / batch_size
        nll_loss = torch.sum(nll_loss) / batch_size

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions)
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions, cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = weighted_nll_loss + d_weight * disc_factor * g_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/logvar".format(split): self.logvar.detach(),
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.detach())
                logits_fake = self.discriminator(reconstructions.detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log


class vqgan_LPIPS(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.net = vqgan_lpips(**config)

    def forward(self, inputs, reconstructions, optimizer_idx, global_step,  mask=None, last_layer=None, split='train'):
        loss, loss_dict = self.net(inputs=inputs, reconstructions=reconstructions, 
        optimizer_idx=optimizer_idx, global_step=global_step, mask=mask, last_layer=last_layer, split=split)
        out = [loss, loss_dict]
        return out


if __name__ == "__main__":
    print('start')
    b = 1
    c = 1
    h, w = 384, 384
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') #
    input_data = torch.randn((b, c, h, w)).to(device) #torch.randn((b, inp_length, c, h, w)).to(device)
    target = torch.randn((b, c, h, w)).to(device) #
    global_step = 0

    # backbone_kwargs = {
    #     'disc_start': 50001, ## not default
    #     'logvar_init': 0.0,
    #     'kl_weight': 1e-6,
    #     'pixelloss_weight': 1.0,
    #     'disc_num_layers': 3,
    #     'disc_in_channels': 1,
    #     'disc_factor': 1.0,
    #     'disc_weight': 0.5,
    #     'perceptual_weight': 0.0
    # }
    print('load yaml from config')
    import yaml
    cfg_path = '/mnt/cache/gongjunchao/workdir/radar_forecasting/configs/sevir/vqgan_pretrain.yaml'
    with open(cfg_path, 'r') as cfg_file:
      cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)
    backbone_kwargs = cfg_params['model']['params']['sub_model']['vqgan_LPIPS']
    model = vqgan_LPIPS(backbone_kwargs)
    model.to(device)

    ## autoencoder ##
    backbone_kwargs_ed = cfg_params['model']['params']['sub_model']['vq_gan']
    from networks.vq_gan import vq_gan
    autoencoder = vq_gan(backbone_kwargs_ed)
    autoencoder.to(device)
    print('end')

    import torch.nn.functional as F
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for i in range(5):
        start.record()
        decoded_images, codebook_indices, q_loss = autoencoder(x=input_data)
        last_layer_weight = autoencoder.net.decoder.conv_out.weight
        aeloss, loss_dict = model(inputs=input_data, reconstructions=decoded_images, optimizer_idx=0, global_step=global_step, last_layer=last_layer_weight)
        loss = aeloss
        loss.backward()
        # for n, p in model.named_parameters():
        #     if p.grad is None:
        #         print(f'{n} has no grad')
        end.record()
        torch.cuda.synchronize()
        print(start.elapsed_time(end))
        memory = torch.cuda.memory_reserved() / (1024. * 1024)
        print("memory:", memory)
        
    from fvcore.nn.parameter_count import parameter_count_table
    from fvcore.nn.flop_count import flop_count
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    flops = FlopCountAnalysis(model, (input_data, target, 0, global_step, None, last_layer_weight))
    print(flop_count_table(flops))

# srun -p ai4earth --kill-on-bad-exit=1 --quotatype=auto --gres=gpu:1 python -u vqgan_lpips.py #