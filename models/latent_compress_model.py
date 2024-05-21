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
import io
import wandb
import pandas as pd
import datetime
import os

from einops import rearrange

class latent_compress_model(basemodel):
    def __init__(self, logger, **params) -> None:
        super().__init__(logger, **params)
        self.logger_print_time = False
        self.data_begin_time = time.time()

        ## load pretrained checkpoint ##
        self.predictor_ckpt_path = self.extra_params.get("predictor_checkpoint_path", None)
        print(f'load from predictor_ckpt_path: {self.predictor_ckpt_path}')
        self.load_checkpoint(self.predictor_ckpt_path, load_model=True, load_optimizer=False, load_scheduler=False, load_epoch=False, load_metric_best=False)
        
        self.autoencoder_ckpt_path = self.extra_params.get("autoencoder_checkpoint_path", None)
        print(f'load from autoencoder_ckpt_path: {self.autoencoder_ckpt_path}')
        self.load_checkpoint(self.autoencoder_ckpt_path, load_model=True, load_optimizer=False, load_scheduler=False, load_epoch=False, load_metric_best=False)

        self.scale_factor = 1.0

        self.latent_size = params.get('latent_size', '48x48x1')
        self.model_name = params.get('model_name', 'gt')
        self.latent_data_save_dir = params.get('latent_data_save_dir', 'latent_data')
        
    def data_preprocess(self, data):
        data_dict = {}
        inp_data = data['inputs'].float().to(self.device, non_blocking=True, dtype=self.data_type)
        tar_data = data['data_samples'].float().to(self.device, non_blocking=True, dtype=self.data_type)
        data_dict.update({'inputs': inp_data, 'data_samples': tar_data, 'file_name': data['file_name']})
        return data_dict

    @torch.no_grad()
    def encode_stage(self, x):
        if utils.get_world_size() == 1 :
            z = self.model[list(self.model.keys())[0]].net.encode(x)
        else:
            z = self.model[list(self.model.keys())[0]].module.net.encode(x)
        return z.sample() * self.scale_factor

    @torch.no_grad()
    def decode_stage(self, z):
        z = z/self.scale_factor
        if utils.get_world_size() == 1 :
            z = self.model[list(self.model.keys())[0]].net.decode(z)
        else:
            z = self.model[list(self.model.keys())[0]].module.net.decode(z)
        return z

    def trainer(self, train_data_loader, valid_data_loader, test_data_loader, max_epoches, max_steps, checkpoint_savedir=None, save_ceph=False, resume=False):
        
        self.test_data_loader = test_data_loader
        self.valid_data_loader = valid_data_loader
        self.train_data_loader = train_data_loader
        ## load temporal mean and std for delta-prediction model ##

        ## the dir of saving models and prediction results ##
        self.checkpoint_savedir = checkpoint_savedir

        if 'sevir' in self.autoencoder_ckpt_path or self.metrics_type == 'SEVIRSkillScore':
            self.z_savedir = 'sevir_latent' 
        else:
            raise NotImplementedError

        if 'TrainingSampler' in self.sampler_type:
            self._iter_trainer(train_data_loader, test_data_loader, max_steps) 
        else:
            self._epoch_trainer(train_data_loader, valid_data_loader, test_data_loader, max_epoches)

    def _epoch_trainer(self, train_data_loader, valid_data_loader, test_data_loader, max_epoches):
        assert max_epoches == 1, "only support 1 epoch"
        for epoch in range(self.begin_epoch, max_epoches):
            if train_data_loader is not None:
                train_data_loader.sampler.set_epoch(epoch)

            self.train_one_epoch(train_data_loader, epoch, max_epoches)
            self.train_one_epoch(valid_data_loader, epoch, max_epoches)
            self.train_one_epoch(test_data_loader, epoch, max_epoches)

    @torch.no_grad()
    def train_one_epoch(self, train_data_loader, epoch, max_epoches):
        import datetime
        from megatron_utils.tensor_parallel.data import get_data_loader_length
        for key in self.lr_scheduler:
            if not self.lr_scheduler_by_step[key]:
                self.lr_scheduler[key].step(epoch)

        end_time = time.time()           
        for key in self.optimizer:              # only train model which has optimizer
            self.model[key].train()

        metric_logger = utils.MetricLogger(delimiter="  ", sync=True)
        iter_time = utils.SmoothedValue(window_size=8, fmt='{avg:.3f}')
        data_time = utils.SmoothedValue(window_size=8, fmt='{avg:.3f}')

        header = 'Epoch [{epoch}/{max_epoches}][{step}/{max_step}]'

        data_loader = train_data_loader
        self.train_data_loader = train_data_loader

        ## reset eval_metrics ##
        self.eval_metrics.reset()

        max_step = get_data_loader_length(train_data_loader)
        for step, batch in enumerate(data_loader):
            if (self.debug and step >=2):
                self.logger.info("debug mode: break from train loop")
                break
        
            # record data read time
            data_time.update(time.time() - end_time)
            if self.debug:
                print(f'data_time: {str(data_time)}')
            loss = self.train_one_step(batch, step)
            # record and time
            iter_time.update(time.time() - end_time)
            end_time = time.time()
            metric_logger.update(**loss)

            # output to logger
            if (step+1) % self.log_step == 0 or step+1 == max_step:
                eta_seconds = iter_time.global_avg * (max_step - step - 1 + max_step * (max_epoches-epoch-1))
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                self.logger.info(
                    metric_logger.delimiter.join(
                        [header,
                        "eta: {eta}",
                        "time: {time}",
                        "data: {data}",
                        "memory: {memory:.0f}",
                        ]
                    ).format(
                        epoch=epoch+1, max_epoches=max_epoches, step=step+1, max_step=max_step,
                        eta=eta_string,
                        time=str(iter_time),
                        data=str(data_time),
                        memory=torch.cuda.memory_reserved() / (1024. * 1024),
                    ))
        
        losses = {}
        ####################################################
        metrics = self.eval_metrics.compute()
        for thr, thr_dict in metrics.items():
            for k, v in thr_dict.items():
                losses.update({f'{thr}-{k}': v})
        ###################################################
        metric_logger.update(**losses)
        self.logger.info('final results: {meters}'.format(meters=str(metric_logger)))
    
    def save_latents(self, latent_data, file_names):
        dir_path = os.path.dirname(file_names[0])
        os.makedirs(dir_path, exist_ok=True)
        for i in range(len(file_names)):
            np.save(file_names[i], latent_data[i].cpu().numpy())
        return
    
    def get_save_names(self, file_names, size, model, data_source='sevir'):
        if data_source == 'sevir':
            save_names = []
            for file_name in file_names:
                split = file_name.split('/')[-2]
                sevir_name = file_name.split('/')[-1]
                save_name = os.path.join(self.latent_data_save_dir, self.z_savedir,
                                         size, model, f'{split}'+'_2h', sevir_name) #f'radar:s3://{self.z_savedir}/{size}/{model}/{split}/{sevir_name}'
                save_names.append(save_name)
        else:
            raise NotImplementedError
        return save_names
    
    @torch.no_grad()
    def train_one_step(self, batch_data, step):
        data_dict = self.data_preprocess(batch_data)
        inp, tar = data_dict['inputs'], data_dict['data_samples']
        file_name = data_dict['file_name']
        b, t, c, h, w = tar.shape
        with torch.no_grad():
            ## first, generate coarse advective field ##
            if self.model_name == 'gt':
                tar = tar
            elif self.model_name == 'earthformer':
                tar = self.model[list(self.model.keys())[1]](inp)
            ## second: encode to latent space ##
            z_tar = self.encode_stage(tar.reshape(-1, c, h, w).contiguous())
            rec_tar = self.decode_stage(z_tar)
            rec_tar = rearrange(rec_tar, '(b t) c h w -> b t c h w', b=b)
            z_tar = rearrange(z_tar, '(b t) c h w -> b t c h w', b=b)

        ## self.get_save_names中model的超参在使用时如果是'gt'，则表示保存的是真实的latent，如果是'EarthFormer'，则表示保存的是EarthFormer预测的latent
        gt_save_names = self.get_save_names(file_names=file_name, size=self.latent_size, model=self.model_name, 
                                            data_source='sevir')
        self.save_latents(latent_data=z_tar, file_names=gt_save_names)
        loss = self.loss(rec_tar, tar) ## important: rescale the loss

        ## compute csi ##
        if self.metrics_type == 'SEVIRSkillScore':
            data_dict['gt'] = tar
            data_dict['pred'] = rec_tar
            self.eval_metrics.update(target=data_dict['gt'], pred=data_dict['pred'])
        else:
            raise NotImplementedError

        return {self.loss_type: loss.item()}
    
        
    @torch.no_grad()
    def test_one_step(self, batch_data):
        pass
    
    
    @torch.no_grad()
    def test(self, test_data_loader, epoch):
        pass
    

    def eval_step(self, batch_data, step):
        pass
    
    @torch.no_grad()
    def test_final(self, test_data_loader, predict_length):
        pass