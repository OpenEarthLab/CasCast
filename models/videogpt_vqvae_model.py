import torch
from models.model import basemodel
import torch.cuda.amp as amp
from torch.functional import F
from torch.distributions import Normal
import time
import copy
from megatron_utils import mpu
import numpy as np

import utils.misc as utils

import wandb

### paralle ensemble ###
from megatron_utils.parallel_state import get_ensemble_parallel_group

class videogpt_vqvae_model(basemodel):
    def __init__(self, logger, **params) -> None:
        super().__init__(logger, **params)
        self.logger_print_time = False
        self.data_begin_time = time.time()

    def data_preprocess(self, data):
        data_dict = {}
        inp_data = data['data_samples'].float().to(self.device, non_blocking=True, dtype=self.data_type)
        tar_data = data['data_samples'].float().to(self.device, non_blocking=True, dtype=self.data_type)
        data_dict.update({'inputs': inp_data, 'data_samples': tar_data})
        return data_dict

    def get_last_layer(self):
        if utils.get_world_size() == 1 :
            last_layer = self.model[list(self.model.keys())[0]].net.decoder.conv_out.weight
        else:
            last_layer = self.model[list(self.model.keys())[0]].module.net.decoder.conv_out.weight
        return last_layer

    def train_one_step(self, batch_data, step):
        data_dict = self.data_preprocess(batch_data)
        inp, tar = data_dict['inputs'], data_dict['data_samples']
        ## first: train encoder+decoder+codebook ##
        recon_loss, x_recon, vq_output = self.model[list(self.model.keys())[0]](x=inp)
        
        loss = recon_loss + vq_output['commitment_loss']
        self.optimizer[list(self.model.keys())[0]].zero_grad()
        loss.backward()
        self.optimizer[list(self.model.keys())[0]].step()

        
        if self.visualizer_type is None:
            pass
        elif self.visualizer_type == 'hko7_visualizer' and (step) % self.visualizer_step==0:
            self.visualizer.save_dbz_image(pred_image=refined_prediction[0], target_img=tar, step=step)
        elif self.visualizer_type == 'sevir_visualizer' and (step) % self.visualizer_step==0:
            self.visualizer.save_pixel_image(pred_image=x_recon, target_img=tar, step=step) ## (k, b, t, c, h, w) -> (b, t, c, h, w)
        else:
            pass
        
        loss_dict = {}
        loss_dict.update({'recon_loss': recon_loss.item()})
        loss_dict.update({'commitment_loss': vq_output['commitment_loss'].item()})
        loss_dict.update({'perplexity': vq_output['perplexity'].item()})
        loss_dict.update({'MSE': F.mse_loss(x_recon.detach(), tar).item()})
        return loss_dict


    def test_one_step(self, batch_data):
        data_dict = self.data_preprocess(batch_data)
        inp, tar = data_dict['inputs'], data_dict['data_samples']
        recon_loss, x_recon, vq_output = self.model[list(self.model.keys())[0]](x=inp)
        loss_dict = {}
        loss_dict.update({'recon_loss': recon_loss.item()})
        loss_dict.update({'commitment_loss': vq_output['commitment_loss'].item()})
        loss_dict.update({'perplexity': vq_output['perplexity'].item()})
        loss_dict.update({'MSE': F.mse_loss(x_recon.detach(), tar).item()})
        return loss_dict
    
    
    
    @torch.no_grad()
    def test(self, test_data_loader, epoch):
        metric_logger = utils.MetricLogger(delimiter="  ", sync=True)
        # set model to eval
        for key in self.model:
            self.model[key].eval()
        data_loader = test_data_loader

        ## save some results ##
        self.num_results2save = 0
        self.id_results2save = 0
        for step, batch in enumerate(data_loader):
            if self.debug and step>= 2 and self.sub_model_name[0] != "IDLE":
                break
            # if self.debug and step>= 2:
            #     break
            if isinstance(batch, int):
                batch = None

            loss = self.test_one_step(batch)
            metric_logger.update(**loss)

        self.logger.info('  '.join(
                [f'Epoch [{epoch + 1}](val stats)',
                 "{meters}"]).format(
                    meters=str(metric_logger)
                 ))

        return metric_logger
    

    def eval_step(self, batch_data):
        data_dict = self.data_preprocess(batch_data)
        inp, tar = data_dict['inputs'], data_dict['data_samples']
        prediction = self.model[list(self.model.keys())[0]](inp)

        ### the official hko7 evaluator receive input tensor shape: b, t, h, w ##
        losses = {}
        data_dict = {}
        if self.metrics_type == 'hko7_official':
            data_dict.update({'gt': tar.squeeze(2).cpu().numpy()})
            data_dict.update({'pred': prediction.squeeze(2).cpu().numpy()})
            self.eval_metrics.update(gt=data_dict['gt'], pred=data_dict['pred'], mask=self.eval_metrics._exclude_mask)
            csi, mse, mae = self.eval_metrics.calculate_stat()
            for i, thr in enumerate(self.eval_metrics._thresholds):
                losses.update({f'CSI_{thr}': csi[:, i].mean()})
        elif self.metrics_type == 'SEVIRSkillScore':
            ## to pixel ##
            data_dict['gt'] = tar.squeeze(2) * 255
            data_dict['pred'] = prediction.squeeze(2) * 255
            self.eval_metrics.update(target=data_dict['gt'].cpu(), pred=data_dict['pred'].cpu())
            metrics = self.eval_metrics.compute()
            csi_total = 0
            for i, thr in enumerate(self.eval_metrics.threshold_list):
                losses.update({f'CSI_{thr}': metrics[thr
                ]['csi']})
                csi_total += metrics[thr]['csi']
            losses.update({'CSI_m': csi_total / len(self.eval_metrics.threshold_list)})
        
        return losses
    
    @torch.no_grad()
    def test_final(self, test_data_loader, predict_length):
        self.test_data_loader = test_data_loader
        metric_logger = utils.MetricLogger(delimiter="  ", sync=True)
        # set model to eval
        for key in self.model:
            self.model[key].eval()

        if utils.get_world_size() > 1:
            rank = mpu.get_data_parallel_rank()
            world_size = mpu.get_data_parallel_world_size()
        else:
            rank = 0
            world_size = 1

        if test_data_loader is not None:
            data_loader = test_data_loader
        else:
            raise ValueError("test_data_loader is None")

        from megatron_utils.tensor_parallel.data import get_data_loader_length
        total_step = get_data_loader_length(test_data_loader)
        ## save some results ##
        self.num_results2save = 5
        self.id_results2save = 0
        for step, batch in enumerate(data_loader):
            if isinstance(batch, int):
                batch = None
            losses = self.eval_step(batch_data=batch)
            metric_logger.update(**losses)

            self.logger.info("#"*80)
            self.logger.info(step)
            if step % 10 == 0 or step == total_step-1:
                self.logger.info('  '.join(
                [f'Step [{step + 1}](val stats)',
                 "{meters}"]).format(
                    meters=str(metric_logger)
                 ))
        return None