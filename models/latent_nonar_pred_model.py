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
from tqdm.auto import tqdm
import torch.distributed as dist

import wandb

from einops import rearrange

class latent_nonar_pred_model(basemodel):
    def __init__(self, logger, **params) -> None:
        super().__init__(logger, **params)
        self.logger_print_time = False
        self.data_begin_time = time.time()
        
        self.autoencoder_ckpt_path = self.extra_params.get("autoencoder_checkpoint_path", None)
        print(f'load from autoencoder_ckpt_path: {self.autoencoder_ckpt_path}')
        self.load_checkpoint(self.autoencoder_ckpt_path, load_model=True, load_optimizer=False, load_scheduler=False, load_epoch=False, load_metric_best=False)
        
        self.metric_epoch_interval = self.extra_params.get("metric_epoch_interval", 1)

    def data_preprocess(self, data):
        data_dict = {}
        inp_data = data['inputs'].float().to(self.device, non_blocking=True, dtype=self.data_type)
        tar_data = data['data_samples']['latent'].float().to(self.device, non_blocking=True, dtype=self.data_type)
        original_tar = data['data_samples']['original'].float().to(self.device, non_blocking=True, dtype=self.data_type)
        data_dict.update({'inputs': inp_data, 'data_samples': tar_data, 'original': original_tar})
        return data_dict

    @torch.no_grad()
    def decode_stage(self, z):
        z = z
        if utils.get_world_size() == 1 :
            z = self.model[list(self.model.keys())[1]].net.decode(z)
        else:
            z = self.model[list(self.model.keys())[1]].module.net.decode(z)
        return z

    def train_one_step(self, batch_data, step):
        data_dict = self.data_preprocess(batch_data)
        inp, tar = data_dict['inputs'], data_dict['data_samples'] 
        original_tar = data_dict['original']
        
        z_prediction = self.model[list(self.model.keys())[0]](inp)
        loss = self.loss(z_prediction, tar)

        self.optimizer[list(self.model.keys())[0]].zero_grad()
        loss.backward()
        self.optimizer[list(self.model.keys())[0]].step()
        ################################################################

        if self.visualizer_type is None:
            pass
        elif self.visualizer_type == 'hko7_visualizer' and (step) % self.visualizer_step==0:
            import pdb; pdb.set_trace() ##TODO
            self.visualizer.save_dbz_image(pred_image=prediction, target_img=tar, step=step)
        elif self.visualizer_type == 'sevir_visualizer' and (step) % self.visualizer_step==0:
            sub_z_prediction = z_prediction[0] #tar[0]
            sub_pixel_prediction = self.decode_stage(sub_z_prediction).unsqueeze(0)
            sub_original_tar = original_tar[:1]
            self.visualizer.save_pixel_image(pred_image=sub_pixel_prediction, target_img=sub_original_tar, step=step)
        else:
            pass
        return {self.loss_type: loss.item()}
        
    @torch.no_grad()
    def test_one_step(self, batch_data):
        data_dict = self.data_preprocess(batch_data)
        inp, tar = data_dict['inputs'], data_dict['data_samples'] 
        original_tar = data_dict['original']

        b = original_tar.shape[0]

        z_prediction = self.model[list(self.model.keys())[0]](inp)
        MSE_loss = F.mse_loss(z_prediction, tar)
        loss_records = {'MSE': MSE_loss.item()}
        ## evaluation ##
        if self.metrics_type == 'hko7_official':
            import pdb; pdb.set_trace()
            data_dict['gt'] = data_dict['gt'].squeeze(2).cpu().numpy()
            data_dict['pred'] = data_dict['pred'].squeeze(2).cpu().numpy()
            self.eval_metrics.update(gt=data_dict['gt'], pred=data_dict['pred'], mask=self.eval_metrics._exclude_mask)
            csi, mse, mae = self.eval_metrics.calculate_stat()
            for i, thr in enumerate(self.eval_metrics._thresholds):
                loss_records.update({f'CSI_{thr}': csi[:, i].mean()})
            loss_records.update({'MSE': MSE_loss})
        elif self.metrics_type == 'SEVIRSkillScore' and self.metric_epoch%self.metric_epoch_interval == 0:
            data_dict['gt'] = original_tar
            z_prediction_flat = rearrange(z_prediction, 'b t c h w -> (b t) c h w').contiguous()
            pixel_prediction = self.decode_stage(z_prediction_flat)
            pixel_prediction = rearrange(pixel_prediction, '(b t) c h w -> b t c h w', b=b)
            data_dict['pred'] = pixel_prediction

            self.eval_metrics.update(target=data_dict['gt'], pred=data_dict['pred'])
            sf_dict = self.eval_metrics.get_single_frame_metrics(target=data_dict['gt'], pred=data_dict['pred'])
            crps_dict = self.eval_metrics.get_crps(target=data_dict['gt'], pred=data_dict['pred'])
            loss_records.update(sf_dict)
            loss_records.update(crps_dict)
        else:
            loss_records.update({'MSE': MSE_loss.item()})

        return loss_records
    
    @torch.no_grad()
    def test(self, test_data_loader, epoch):
        metric_logger = utils.MetricLogger(delimiter="  ", sync=True)
        # set model to eval
        for key in self.model:
            self.model[key].eval()
        data_loader = test_data_loader

        self.metric_epoch = int(epoch)
        ## save some results ##
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
        if self.metric_epoch%self.metric_epoch_interval == 0:         
            metrics_records = {}
            csi_total = 0
            metrics = self.eval_metrics.compute()
            for i, thr in enumerate(self.eval_metrics.threshold_list):
                metrics_records.update({f'CSI_{thr}': metrics[thr
                ]['csi']})
                csi_total += metrics[thr]['csi']
            metrics_records.update({'CSI_m': csi_total / len(self.eval_metrics.threshold_list)})
            self.eval_metrics.reset()
            metric_logger.update(**metrics_records)

        self.logger.info('  '.join(
                [f'Epoch [{epoch + 1}](val stats)',
                 "{meters}"]).format(
                    meters=str(metric_logger)
                 ))

        return metric_logger
    
    def eval_step(self, batch_data, step):
        data_dict = self.data_preprocess(batch_data)
        inp, tar = data_dict['inputs'], data_dict['data_samples']
        original_tar = data_dict['original']
        b, t, c, h, w = tar.shape
        ## inp is coarse prediction in latent space, tar is gt in latent space
        z_tar = tar
        z_coarse_prediction = inp
        ## scale ##
        z_tar = z_tar * self.scale_factor
        z_coarse_prediction = z_coarse_prediction * self.scale_factor
        ## sample image ##
        losses = {}
        z_sample_prediction = self.denoise(template_data=z_tar, cond_data=z_coarse_prediction, bs=tar.shape[0], vis=True, cfg=2, ensemble_member=1)
        len_shape_prediction = len(z_sample_prediction.shape)
        assert len_shape_prediction == 6
        n = z_sample_prediction.shape[1]
        sample_predictions = []
        for i in range(n):
            member_z_sample_prediction = z_sample_prediction[:, i, ...]
            member_z_sample_prediction = rearrange(member_z_sample_prediction, 'b t c h w -> (b t) c h w').contiguous()
            member_sample_prediction = self.decode_stage(member_z_sample_prediction)
            member_sample_prediction = rearrange(member_sample_prediction, '(b t) c h w -> b t c h w', t=t)
            sample_predictions.append(member_sample_prediction) 
        sample_predictions = torch.stack(sample_predictions, dim=1)
        ## evaluate other metrics ##
        data_dict = {}
        if self.metrics_type == 'SEVIRSkillScore':
            data_dict['gt'] =  original_tar
            data_dict['pred'] = sample_predictions.mean(dim=1)
            self.eval_metrics.update(target=data_dict['gt'], pred=data_dict['pred'])
            ############
            sf_dict = self.eval_metrics.get_single_frame_metrics(target=data_dict['gt'], pred=data_dict['pred'])
            crps_dict = self.eval_metrics.get_crps(target=data_dict['gt'], pred=sample_predictions)
            losses.update(sf_dict)
            losses.update(crps_dict)
            ############
            self.fvd_computer.update(videos_real=self.fvd_computer.preprocess(data_dict['gt'].repeat(1, 1, 3, 1, 1)), videos_fake=self.fvd_computer.preprocess(data_dict['pred'].repeat(1, 1, 3, 1, 1)))
            ############
        ## save image ##
        if self.visualizer_type == 'sevir_visualizer' and (step) % 200 == 0:
            self.visualizer.save_pixel_image(pred_image=data_dict['pred'], target_img=data_dict['gt'], step=step)
        else:
            pass
        
        return losses
    
    @torch.no_grad()
    def test_final(self, test_data_loader, predict_length):
        self.test_data_loader = test_data_loader
        metric_logger = utils.MetricLogger(delimiter="  ", sync=True)
        self.scale_factor = 0.6790621876716614 #48x48x4 0.6786020398139954
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

        from utils.metrics import cal_FVD
        self.fvd_computer = cal_FVD(use_gpu=True)

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
        fvd = self.fvd_computer.compute()
        losses.update({'fvd':fvd})
        ####################################################
        metrics = self.eval_metrics.compute()
        for thr, thr_dict in metrics.items():
            for k, v in thr_dict.items():
                losses.update({f'{thr}-{k}': v})
        ###################################################
        metric_logger.update(**losses)
        self.logger.info('final results: {meters}'.format(meters=str(metric_logger)))
        return None