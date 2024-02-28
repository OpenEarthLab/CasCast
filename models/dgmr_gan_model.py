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

class dgmr_gan_model(basemodel):
    def __init__(self, logger, **params) -> None:
        super().__init__(logger, **params)
        self.logger_print_time = False
        self.data_begin_time = time.time()
        self.rss_training = params.get('rss_training', False)
    #     ## load checkpoint for advective predictor ##

    # def advective_predictor_load(checkpoint_path):


    def data_preprocess(self, data):
        data_dict = {}
        inp_data = data['inputs'].float().to(self.device, non_blocking=True, dtype=self.data_type)
        tar_data = data['data_samples'].float().to(self.device, non_blocking=True, dtype=self.data_type)

        data_dict.update({'inputs': inp_data, 'data_samples': tar_data})
        return data_dict



    def train_one_step(self, batch_data, step):
        data_dict = self.data_preprocess(batch_data)
        inp, tar = data_dict['inputs'], data_dict['data_samples']
        ## 1. updating discriminator ##
        with torch.no_grad():
            refined_prediction = self.model[list(self.model.keys())[0]](x=inp, k=1)
            refined_prediction = refined_prediction.detach() ## b, k, t, c, h, w

        # torch.autograd.set_detect_anomaly(True)
        pred_digit = self.model[list(self.model.keys())[1]](prev_x=inp, future_x=refined_prediction)
        tar_digit =  self.model[list(self.model.keys())[1]](prev_x=inp, future_x=tar)
        ## discriminator loss ##
        if utils.get_world_size() == 1 :
            disc_loss_dict = self.model[list(self.model.keys())[1]].compute_loss(tar_digit=tar_digit, pred_digit=pred_digit, step=step)
        else:
            disc_loss_dict = self.model[list(self.model.keys())[1]].module.compute_loss(tar_digit=tar_digit, pred_digit=pred_digit, step=step)

        total_discriminator_loss = disc_loss_dict['loss_disc']
        total_discriminator_loss.backward()
        self.optimizer[list(self.model.keys())[1]].step()
        self.optimizer[list(self.model.keys())[1]].zero_grad()

        ## 2. updating generator ##
        refined_prediction = self.model[list(self.model.keys())[0]](x=inp, k=1) ## b, k, t, c, h, w
        pred_digit = self.model[list(self.model.keys())[1]](prev_x=inp, future_x=refined_prediction)
        ## generator loss ##
        if utils.get_world_size() == 1:
            gen_loss_dict = self.model[list(self.model.keys())[0]].compute_loss(pred_digits=pred_digit, tar_x=tar, pred_x=refined_prediction, step=step)
        else:
            gen_loss_dict = self.model[list(self.model.keys())[0]].module.compute_loss(pred_digits=pred_digit, tar_x=tar, pred_x=refined_prediction, step=step)
        total_generator_loss = gen_loss_dict['total_loss']
        total_generator_loss.backward()
        self.optimizer[list(self.model.keys())[1]].step()
        self.optimizer[list(self.model.keys())[1]].zero_grad()

        
        # if (utils.get_world_size() > 1 and mpu.get_data_parallel_rank() == 0) or utils.get_world_size() == 1:
        #             wandb.log({f'train_discriminator_fake': disc_loss_dict['loss_disc_fake'].item(),
        #                        f'train_discriminator_real': disc_loss_dict['loss_disc_real'].item(),
        #                         f'train_generator_k_maxpool_loss': gen_loss_dict['K_MAX_pooling_loss'].item(),
        #                         f'train_generator_adv_loss': gen_loss_dict['loss_gen'].item()})
        
        if self.visualizer_type is None:
            pass
        elif self.visualizer_type == 'hko7_visualizer' and (step) % self.visualizer_step==0:
            self.visualizer.save_dbz_image(pred_image=refined_prediction[0], target_img=tar, step=step)
        elif self.visualizer_type == 'sevir_visualizer' and (step) % self.visualizer_step==0:
            self.visualizer.save_pixel_image(pred_image=refined_prediction[:, 0], target_img=tar, step=step) ## (b, k, t, c, h, w) -> (b, t, c, h, w)
        else:
            pass

        return {
                f'spatial_loss_disc_true': disc_loss_dict['spatial_loss_disc_true'].item(),
                f'spatial_loss_disc_fake': disc_loss_dict['spatial_loss_disc_fake'].item(),
                f'temporal_loss_disc_true': disc_loss_dict['temporal_loss_disc_true'].item(),
                f'temporal_loss_disc_fake': disc_loss_dict['temporal_loss_disc_fake'].item(),
                f'loss_disc': disc_loss_dict['loss_disc'].item(),
                f'spatial_loss_gen': gen_loss_dict['spatial_loss_gen'].item(),
                f'temporal_loss_gen': gen_loss_dict['temporal_loss_gen'].item(),
                f'pixel_loss':  gen_loss_dict['pixel_loss'].item(),
                f'gne_total_loss': gen_loss_dict['total_loss'].item()
                }
    


    def test_one_step(self, batch_data):
        data_dict = self.data_preprocess(batch_data)
        inp, tar = data_dict['inputs'], data_dict['data_samples']
        with torch.no_grad():
            prediction = self.model[list(self.model.keys())[0]](x=inp, k=1)
        loss_records = {}
        ## evaluate other metrics ##
        data_dict = {}
        data_dict['gt'] = tar
        data_dict['pred'] = prediction.squeeze(1)
        MSE_loss = torch.mean((prediction - tar) ** 2).item()
        loss = self.loss(prediction, tar)

        ## evaluation ##
        if self.metrics_type == 'hko7_official':
            data_dict['gt'] = data_dict['gt'].squeeze(2).cpu().numpy()
            data_dict['pred'] = data_dict['pred'].squeeze(2).cpu().numpy()
            self.eval_metrics.update(gt=data_dict['gt'], pred=data_dict['pred'], mask=self.eval_metrics._exclude_mask)
            csi, mse, mae = self.eval_metrics.calculate_stat()
            for i, thr in enumerate(self.eval_metrics._thresholds):
                loss_records.update({f'CSI_{thr}': csi[:, i].mean()})
            loss_records.update({'MSE': MSE_loss})
        elif self.metrics_type == 'SEVIRSkillScore':
            data_dict['gt'] = data_dict['gt']
            data_dict['pred'] = data_dict['pred']
            self.eval_metrics.update(target=data_dict['gt'], pred=data_dict['pred'])
            loss_records.update({'MSE': MSE_loss})
            # csi_total = 0
            # ## to pixel ##
            # data_dict['gt'] = data_dict['gt'].squeeze(2) * 255
            # data_dict['pred'] = data_dict['pred'].squeeze(2) * 255
            # self.eval_metrics.update(target=data_dict['gt'].cpu(), pred=data_dict['pred'].cpu())
            # metrics = self.eval_metrics.compute()
            # for i, thr in enumerate(self.eval_metrics.threshold_list):
            #     loss_records.update({f'CSI_{thr}': metrics[thr
            #     ]['csi']})
            #     csi_total += metrics[thr]['csi']
            # loss_records.update({'CSI_m': csi_total / len(self.eval_metrics.threshold_list)})
            # loss_records.update({'MSE': MSE_loss})
            # if (utils.get_world_size() > 1 and mpu.get_data_parallel_rank() == 0) or utils.get_world_size() == 1:
            #     wandb.log({f'val_CSI_m': loss_records['CSI_m'] })
        else:
            metrics_loss = self.eval_metrics.evaluate_batch(data_dict)
            loss_records.update(metrics_loss)
        
        ## log to wandb ##
        # if (utils.get_world_size() > 1 and mpu.get_data_parallel_rank() == 0) or utils.get_world_size() == 1:
        #     wandb.log({f'val_{self.loss_type}': loss.item() })

        return loss_records
    
    # @torch.no_grad()
    # def test(self, test_data_loader, epoch):
    #     metric_logger = utils.MetricLogger(delimiter="  ", sync=True)
    #     # set model to eval
    #     for key in self.model:
    #         self.model[key].eval()
    #     data_loader = test_data_loader

    #     ## save some results ##
    #     self.num_results2save = 0
    #     self.id_results2save = 0
    #     for step, batch in enumerate(data_loader):
    #         if self.debug and step>= 2 and self.sub_model_name[0] != "IDLE":
    #             break
    #         # if self.debug and step>= 2:
    #         #     break
    #         if isinstance(batch, int):
    #             batch = None

    #         loss = self.test_one_step(batch)
    #         metric_logger.update(**loss)

    #     self.logger.info('  '.join(
    #             [f'Epoch [{epoch + 1}](val stats)',
    #              "{meters}"]).format(
    #                 meters=str(metric_logger)
    #              ))

    #     return metric_logger

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

        ## compute metrics ##
        metrics_records = {}
        csi_total = 0
        metrics = self.eval_metrics.compute()
        for i, thr in enumerate(self.eval_metrics.threshold_list):
            metrics_records.update({f'CSI_{thr}': metrics[thr
            ]['csi']})
            csi_total += metrics[thr]['csi']
        metrics_records.update({'CSI_m': csi_total / len(self.eval_metrics.threshold_list)})
        metric_logger.update(**metrics_records)
        self.eval_metrics.reset()
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