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
from einops import rearrange

import wandb

### paralle ensemble ###
from megatron_utils.parallel_state import get_ensemble_parallel_group

class nowcast_gan_model(basemodel):
    def __init__(self, logger, **params) -> None:
        super().__init__(logger, **params)
        self.logger_print_time = False
        self.data_begin_time = time.time()
        self.rss_training = params.get('rss_training', False)

        ## reset the best mse score of deterministic model ##
        self.metric_best = None

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
        ## first, generate coarse advective field ##
        with torch.no_grad():
            prediction = self.model[list(self.model.keys())[0]](inp)
            prediction = prediction.detach()
        ## 1. updating discriminator ##
        with torch.no_grad():
            refined_prediction = self.model[list(self.model.keys())[1]](inp_x=inp, coarse_x=prediction, k=3)
            refined_prediction = refined_prediction.detach()

        # torch.autograd.set_detect_anomaly(True)
        pred_digit = self.model[list(self.model.keys())[2]](inp_x=inp, x=refined_prediction)
        tar_digit =  self.model[list(self.model.keys())[2]](inp_x=inp, x=tar)
        ## discriminator loss ##
        if utils.get_world_size() == 1 :
            disc_loss_dict = self.model[list(self.model.keys())[2]].compute_loss(tar_digit=tar_digit, pred_digit=pred_digit)
        else:
            disc_loss_dict = self.model[list(self.model.keys())[2]].module.compute_loss(tar_digit=tar_digit, pred_digit=pred_digit)

        total_discriminator_loss = disc_loss_dict['loss_disc']
        total_discriminator_loss.backward()
        self.optimizer[list(self.model.keys())[2]].step()
        self.optimizer[list(self.model.keys())[2]].zero_grad()

        ## 2. updating generator ##
        refined_prediction = self.model[list(self.model.keys())[1]](inp_x=inp, coarse_x=prediction, k=3)
        pred_digit = self.model[list(self.model.keys())[2]](inp_x=inp, x=refined_prediction)
        ## generator loss ##
        if utils.get_world_size() == 1:
            gen_loss_dict = self.model[list(self.model.keys())[1]].compute_loss(pred_digits=pred_digit, tar_x=tar, pred_x=refined_prediction)
        else:
            gen_loss_dict = self.model[list(self.model.keys())[1]].module.compute_loss(pred_digits=pred_digit, tar_x=tar, pred_x=refined_prediction)
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
            self.visualizer.save_pixel_image(pred_image=refined_prediction[0], target_img=tar, step=step)
        elif self.visualizer_type == 'sevir_visualizer' and (step) % self.visualizer_step==0:
            self.visualizer.save_pixel_image(pred_image=refined_prediction[0], target_img=tar, step=step) ## (k, b, t, c, h, w) -> (b, t, c, h, w)
        elif self.visualizer_type == 'meteonet_visualizer' and (step) % self.visualizer_step==0:
            self.visualizer.save_pixel_image(pred_image=refined_prediction[0], target_img=tar, step=step)
        else:
            pass

        return {f'train_discriminator_fake': disc_loss_dict['loss_disc_fake'].item(),
                               f'train_discriminator_real': disc_loss_dict['loss_disc_real'].item(),
                                f'train_generator_k_maxpool_loss': gen_loss_dict['K_MAX_pooling_loss'].item(),
                                f'train_generator_adv_loss': gen_loss_dict['loss_gen'].item()}
    


    @ torch.no_grad()
    def test_one_step(self, batch_data):
        data_dict = self.data_preprocess(batch_data)
        inp, tar = data_dict['inputs'], data_dict['data_samples']

        prediction = self.model[list(self.model.keys())[0]](inp)
        refined_prediction = self.model[list(self.model.keys())[1]](inp_x=inp, coarse_x=prediction, k=3)
        ens_refined_prediction = refined_prediction.mean(dim=0)
        prediction = ens_refined_prediction
        
        loss_records = {}
        ## evaluate other metrics ##
        data_dict = {}
        data_dict['gt'] = tar
        data_dict['pred'] = prediction
        MSE_loss = torch.mean((prediction - tar) ** 2).item()
        loss = self.loss(prediction, tar)

        ## evaluation ##
        if self.metrics_type == 'SEVIRSkillScore':
            import pdb; pdb.set_trace()
            csi_total = 0
            ## to pixel ##
            data_dict['gt'] = data_dict['gt'].squeeze(2) * 255
            data_dict['pred'] = data_dict['pred'].squeeze(2) * 255
            self.eval_metrics.update(target=data_dict['gt'].cpu(), pred=data_dict['pred'].cpu())
            metrics = self.eval_metrics.compute()
            for i, thr in enumerate(self.eval_metrics.threshold_list):
                loss_records.update({f'CSI_{thr}': metrics[thr
                ]['csi']})
                csi_total += metrics[thr]['csi']
            loss_records.update({'CSI_m': csi_total / len(self.eval_metrics.threshold_list)})
            loss_records.update({'MSE': MSE_loss})
            if (utils.get_world_size() > 1 and mpu.get_data_parallel_rank() == 0) or utils.get_world_size() == 1:
                wandb.log({f'val_CSI_m': loss_records['CSI_m'] })
        elif self.metrics_type == 'METEONETScore':
            data_dict['gt'] = data_dict['gt']
            data_dict['pred'] = data_dict['pred']
            self.eval_metrics.update(target=data_dict['gt'], pred=data_dict['pred'])
            loss_records.update({'MSE': MSE_loss})
        elif self.metrics_type == 'hko7_official':
            data_dict['gt'] = data_dict['gt']
            data_dict['pred'] = data_dict['pred']
            self.eval_metrics.update(target=data_dict['gt'], pred=data_dict['pred'])
            loss_records.update({'MSE': MSE_loss})
        else:
            metrics_loss = self.eval_metrics.evaluate_batch(data_dict)
            loss_records.update(metrics_loss)
        
        # ## log to wandb ##
        # if (utils.get_world_size() > 1 and mpu.get_data_parallel_rank() == 0) or utils.get_world_size() == 1:
        #     wandb.log({f'val_{self.loss_type}': loss.item() })

        loss_records.update({f'MSE': MSE_loss})

        return loss_records
    
    
    @torch.no_grad()
    def test(self, test_data_loader, epoch):
        metric_logger = utils.MetricLogger(delimiter="  ", sync=True)
        # set model to eval
        for key in self.model:
            self.model[key].eval()
        data_loader = test_data_loader

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
    
    @torch.no_grad()
    def eval_step(self, batch_data, step):
        data_dict = self.data_preprocess(batch_data)
        inp, tar = data_dict['inputs'], data_dict['data_samples']
        with torch.no_grad():
            prediction = self.model[list(self.model.keys())[0]](inp)
        refined_prediction = self.model[list(self.model.keys())[1]](inp_x=inp, coarse_x=prediction, k=1)
        sample_predictions = rearrange(refined_prediction, 'k b t c h w -> b k t c h w')
        ## evaluate other metrics ##
        # data_dict = {}
        # data_dict['gt'] = tar
        # data_dict['pred'] = prediction
        # MSE_loss = torch.mean((prediction - tar) ** 2).item()
        # loss = self.loss(prediction, tar)

        losses = {}
        ## evaluate other metrics ##
        data_dict = {}
        if self.metrics_type == 'SEVIRSkillScore':
            data_dict['gt'] =  tar
            data_dict['pred'] = sample_predictions.mean(dim=1)
            self.eval_metrics.update(target=data_dict['gt'], pred=data_dict['pred'])
            ############
            sf_dict = self.eval_metrics.get_single_frame_metrics(target=data_dict['gt'], pred=data_dict['pred'])
            crps_dict = self.eval_metrics.get_crps(target=data_dict['gt'], pred=sample_predictions)
            losses.update(sf_dict)
            losses.update(crps_dict)
            ############
            # self.fvd_computer.update(videos_real=self.fvd_computer.preprocess(data_dict['gt'].repeat(1, 1, 3, 1, 1)), videos_fake=self.fvd_computer.preprocess(data_dict['pred'].repeat(1, 1, 3, 1, 1)))
            ############
        elif self.metrics_type == 'hko7_official':
            data_dict['gt'] =  tar
            data_dict['pred'] = sample_predictions.mean(dim=1)
            self.eval_metrics.update(target=data_dict['gt'], pred=data_dict['pred'])
            ############
            sf_dict = self.eval_metrics.get_single_frame_metrics(target=data_dict['gt'], pred=data_dict['pred'])
            crps_dict = self.eval_metrics.get_crps(target=data_dict['gt'], pred=sample_predictions)
            losses.update(sf_dict)
            losses.update(crps_dict)
        elif self.metrics_type == 'METEONETScore':
            data_dict['gt'] =  tar
            data_dict['pred'] = sample_predictions.mean(dim=1)
            self.eval_metrics.update(target=data_dict['gt'], pred=data_dict['pred'])
            ###########
            sf_dict = self.eval_metrics.get_single_frame_metrics(target=data_dict['gt'], pred=data_dict['pred'])
            crps_dict = self.eval_metrics.get_crps(target=data_dict['gt'], pred=sample_predictions)
            losses.update(sf_dict)
            losses.update(crps_dict)
            ############
        ## save image ##
        if self.visualizer_type == 'sevir_visualizer' and (step) % 1 == 0:
            # self.visualizer.save_pixel_image(pred_image=data_dict['pred'], target_img=data_dict['gt'], step=step)
            #######################################################################################################
            model_name = list(self.model.keys())[0]
            ceph_prefix = f'radar:s3://radar_visualization/sevir/nowcast_{model_name}_{self.visualizer.sub_dir}'
            self.visualizer.save_vil_last_image_and_npy(pred_image=data_dict['pred'], target_img=data_dict['gt'], step=step, ceph_prefix=ceph_prefix)
        elif self.visualizer_type == 'hko7_visualizer' and (step) % 1 == 0:
            # self.visualizer.save_pixel_image(pred_image=data_dict['pred'], target_img=data_dict['gt'], step=step)
            #######################################################################################################
            model_name = list(self.model.keys())[0]
            ceph_prefix = f'radar:s3://radar_visualization/hko7/nowcast_{model_name}_{self.visualizer.sub_dir}'
            self.visualizer.save_hko7_last_image_and_npy(pred_image=data_dict['pred'], target_img=data_dict['gt'], step=step, ceph_prefix=ceph_prefix)
        elif self.visualizer_type == 'meteonet_visualizer' and (step) % 1 == 0:
            # self.visualizer.save_pixel_image(pred_image=data_dict['pred'], target_img=data_dict['gt'], step=step)
            model_name = list(self.model.keys())[0]
            ceph_prefix = f'radar:s3://radar_visualization/meteonet/nowcast_{model_name}_{self.visualizer.sub_dir}'
            self.visualizer.save_meteo_last_image_and_npy(pred_image=data_dict['pred'], target_img=data_dict['gt'], step=step, ceph_prefix=ceph_prefix)
        else:
            pass
        
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
            if isinstance(batch, int):
                batch = None
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
        # fvd = self.fvd_computer.compute()
        # losses.update({'fvd':fvd})
        ####################################################
        metrics = self.eval_metrics.compute()
        for thr, thr_dict in metrics.items():
            for k, v in thr_dict.items():
                losses.update({f'{thr}-{k}': v})
        ###################################################
        try:
            metric_logger.update(**losses)
            self.logger.info('final results: {meters}'.format(meters=str(metric_logger)))
        except:
            ## save as excel ##
            import pandas as pd
            df = pd.DataFrame.from_dict(losses)
            df.to_excel(f'{self.visualizer.exp_dir}/{self.visualizer.sub_dir}_losses.xlsx')
        return None