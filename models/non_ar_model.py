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
import math
import wandb

### paralle ensemble ###
from megatron_utils.parallel_state import get_ensemble_parallel_group

class non_ar_model(basemodel):
    def __init__(self, logger, **params) -> None:
        super().__init__(logger, **params)
        self.logger_print_time = False
        self.data_begin_time = time.time()
        self.rss_training = params.get('rss_training', False)

    def data_preprocess(self, data):
        data_dict = {}
        inp_data = data['inputs'].float().to(self.device, non_blocking=True, dtype=self.data_type)
        tar_data = data['data_samples'].float().to(self.device, non_blocking=True, dtype=self.data_type)

        data_dict.update({'inputs': inp_data, 'data_samples': tar_data})
        return data_dict


    def train_one_step(self, batch_data, step):
        data_dict = self.data_preprocess(batch_data)
        inp, tar = data_dict['inputs'], data_dict['data_samples']
        self.optimizer[list(self.model.keys())[0]].zero_grad()
        prediction = self.model[list(self.model.keys())[0]](inp)
        loss = self.loss(prediction, tar)
        loss.backward()
        self.optimizer[list(self.model.keys())[0]].step()
        
        if self.visualizer_type is None:
            pass
        elif self.visualizer_type == 'sevir_visualizer' and (step) % self.visualizer_step==0:
            self.visualizer.save_pixel_image(pred_image=prediction, target_img=tar, step=step)
        else:
            pass
        return {self.loss_type: loss.item()}
    


    def test_one_step(self, batch_data):
        data_dict = self.data_preprocess(batch_data)
        inp, tar = data_dict['inputs'], data_dict['data_samples']
        with torch.no_grad():
            prediction = self.model[list(self.model.keys())[0]](inp)
        loss_records = {}
        ## evaluate other metrics ##
        data_dict = {}
        data_dict['gt'] = tar
        data_dict['pred'] = prediction
        MSE_loss = torch.mean((prediction - tar) ** 2).item()
        loss = self.loss(prediction, tar)

        ## evaluation ##
        if self.metrics_type == 'SEVIRSkillScore':
            data_dict['gt'] = data_dict['gt']
            data_dict['pred'] = data_dict['pred']
            self.eval_metrics.update(target=data_dict['gt'], pred=data_dict['pred'])
            loss_records.update({'MSE': MSE_loss})
        else: 
            raise not NotImplementedError

        return loss_records
    
    @torch.no_grad()
    def test(self, test_data_loader, epoch):
        metric_logger = utils.MetricLogger(delimiter="  ", sync=True)
        # set model to eval
        for key in self.model:
            self.model[key].eval()
        data_loader = test_data_loader

        for step, batch in enumerate(data_loader):
            if self.debug and step>= 2:
                break

            loss = self.test_one_step(batch)
            metric_logger.update(**loss)

        ## compute metrics ##
        losses = {}
        metrics = self.eval_metrics.compute()
        for thr, thr_dict in metrics.items():
            for k, v in thr_dict.items():
                losses.update({f'{thr}-{k}': v})
        self.eval_metrics.reset()
        metric_logger.update(**losses)
        self.logger.info('  '.join(
                [f'Epoch [{epoch + 1}](val stats)',
                 "{meters}"]).format(
                    meters=str(metric_logger)
                 ))

        return metric_logger
    

    def eval_step(self, batch_data, step):
        data_dict = self.data_preprocess(batch_data)
        inp, tar = data_dict['inputs'], data_dict['data_samples']
        prediction = self.model[list(self.model.keys())[0]](inp)

        losses = {}
        data_dict = {}
        if self.metrics_type == 'SEVIRSkillScore':
            data_dict['gt'] = tar
            data_dict['pred'] = prediction
            self.eval_metrics.update(target=data_dict['gt'], pred=data_dict['pred'])
            ############
            sf_dict = self.eval_metrics.get_single_frame_metrics(target=data_dict['gt'], pred=data_dict['pred'])
            crps_dict = self.eval_metrics.get_crps(target=data_dict['gt'], pred=data_dict['pred'])
            losses.update(sf_dict)
            losses.update(crps_dict)
            ############
        if self.visualizer_type == 'sevir_visualizer' and (step) % 1000 == 0:
            self.visualizer.save_pixel_image(pred_image=data_dict['pred'], target_img=data_dict['gt'], step=step)
        else:
            pass

        losses.update({'MSE': torch.mean((prediction - tar) ** 2).item()})
        return losses
    
    @torch.no_grad()
    def test_final(self, test_data_loader, predict_length):
        self.test_data_loader = test_data_loader
        metric_logger = utils.MetricLogger(delimiter="  ", sync=True)
        # set model to eval
        for key in self.model:
            self.model[key].eval()

        if test_data_loader is not None:
            data_loader = test_data_loader
        else:
            raise ValueError("test_data_loader is None")

        from megatron_utils.tensor_parallel.data import get_data_loader_length
        total_step = get_data_loader_length(test_data_loader)

        for step, batch in enumerate(data_loader):
            losses = self.eval_step(batch_data=batch, step=step)
            metric_logger.update(**losses)

            self.logger.info("#"*80)
            self.logger.info(step)
            if step % 10 == 0 or step == total_step-1:
                self.logger.info('  '.join(
                [f'Step [{step + 1}](val stats)',
                 "{meters}"]).format(
                    meters=str(metric_logger)
                 ))
        ####################################################
        ####################################################
        ####################################################
        metrics = self.eval_metrics.compute()
        for thr, thr_dict in metrics.items():
            for k, v in thr_dict.items():
                losses.update({f'{thr}-{k}': v})
        ###################################################
        ## directly print ##
        metric_logger.update(**losses)
        self.logger.info('final results: {meters}'.format(meters=str(metric_logger)))
        ##################################################
        return None
    