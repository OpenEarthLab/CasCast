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

from einops import rearrange

class latent_compare_model(basemodel):
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
        
    def data_preprocess(self, data):
        data_dict = {}
        inp_data = data['inputs'].float().to(self.device, non_blocking=True, dtype=self.data_type)
        tar_data = data['data_samples']['latent'].float().to(self.device, non_blocking=True, dtype=self.data_type)
        original_tar = data['data_samples']['original'].float().to(self.device, non_blocking=True, dtype=self.data_type)
        data_dict.update({'inputs': inp_data, 'data_samples': tar_data, 'original': original_tar})
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

        from petrel_client.client import Client
        self.z_savedir = 'sevir_latent' 
        self.z_client = Client(conf_path="~/petreloss.conf")

        if 'TrainingSampler' in self.sampler_type:
            self._iter_trainer(train_data_loader, test_data_loader, max_steps) 
        else:
            self._epoch_trainer(train_data_loader, valid_data_loader, test_data_loader, max_epoches)

    def _epoch_trainer(self, train_data_loader, valid_data_loader, test_data_loader, max_epoches):
        assert max_epoches == 1, "only support 1 epoch"
        for epoch in range(self.begin_epoch, max_epoches):
            if train_data_loader is not None:
                train_data_loader.sampler.set_epoch(epoch)

            # self.train_one_epoch(train_data_loader, epoch, max_epoches)
            self.train_one_epoch(valid_data_loader, epoch, max_epoches)
            self.train_one_epoch(test_data_loader, epoch, max_epoches)

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

        max_step = get_data_loader_length(train_data_loader)
        for step, batch in enumerate(data_loader):
            if (self.debug and step >=2) or self.sub_model_name[0] == "IDLE":
                self.logger.info("debug mode: break from train loop")
                break
            if isinstance(batch, int):
                batch = None
        
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
                        "{meters}"
                        ]
                    ).format(
                        epoch=epoch+1, max_epoches=max_epoches, step=step+1, max_step=max_step,
                        eta=eta_string,
                        time=str(iter_time),
                        data=str(data_time),
                        memory=torch.cuda.memory_reserved() / (1024. * 1024),
                        meters=str(metric_logger)
                    ))
        
        losses = {}
        ####################################################
        metrics = self.eval_metrics.compute()
        self.eval_metrics.reset()
        for thr, thr_dict in metrics.items():
            for k, v in thr_dict.items():
                losses.update({f'{thr}-{k}': v})
        ###################################################
        metric_logger.update(**losses)
        self.logger.info('final results: {meters}'.format(meters=str(metric_logger)))
    
    def save_latents(self, latent_data, file_names):
        for latent, file_name in zip(latent_data, file_names):
            with io.BytesIO() as f:
                np.save(f, latent.cpu().numpy())
                f.seek(0)
                self.z_client.put(file_name, f.read())
        pass
    
    def get_save_names(self, file_names, size, model):
        save_names = []
        for file_name in file_names:
            split = file_name.split('/')[-2]
            sevir_name = file_name.split('/')[-1]
            save_name = f'radar:s3://{self.z_savedir}/{size}/{model}/{split}/{sevir_name}'
            save_names.append(save_name)
        return save_names
    
    @torch.no_grad()
    def train_one_step(self, batch_data, step):
        data_dict = self.data_preprocess(batch_data)
        inp, tar = data_dict['inputs'], data_dict['data_samples'] 
        original_tar = data_dict['original']
        z_tar = tar
        z_coarse_prediction = inp
        b, t, c, h, w = z_tar.shape
        ## first, generate coarse advective field ##
        with torch.no_grad():
            z_coarse_prediction = z_coarse_prediction.reshape(-1, c, h, w).contiguous()
            rec_coarse_prediction = self.decode_stage(z_coarse_prediction)
            rec_coarse_prediction = rearrange(rec_coarse_prediction, '(b t) c h w -> b t c h w', b=b)
            z_coarse_prediction = rearrange(z_coarse_prediction, '(b t) c h w -> b t c h w', b=b)
        z_loss = self.loss(z_tar, z_coarse_prediction) 
        pixel_loss = self.loss(original_tar, rec_coarse_prediction)

        ## compute csi ##
        if self.metrics_type == 'SEVIRSkillScore':
            data_dict['gt'] = original_tar
            data_dict['pred'] = rec_coarse_prediction
            self.eval_metrics.update(target=data_dict['gt'], pred=data_dict['pred'])
        return {f'z-{self.loss_type}': z_loss.item(), f'pixel-{self.loss_type}': pixel_loss.item()}
    
        
    @torch.no_grad()
    def test_one_step(self, batch_data):
        data_dict = self.data_preprocess(batch_data)
        inp, tar = data_dict['inputs'], data_dict['data_samples']
        b, t, c, h, w = tar.shape
        ## first, generate coarse advective field ##
        with torch.no_grad():
            coarse_prediction = self.model[list(self.model.keys())[0]](inp)
            coarse_prediction = coarse_prediction.detach() 
            ## second: encode to latent space ##
            z_tar = self.encode_stage(tar.reshape(-1, c, h, w).contiguous())
            z_tar = rearrange(z_tar, '(b t) c h w -> b t c h w', b=b)
            z_coarse_prediction = self.encode_stage(coarse_prediction.reshape(-1, c, h, w).contiguous())
            z_coarse_prediction = rearrange(z_coarse_prediction, '(b t) c h w -> b t c h w', b=b)
        ## third: train diffusion model ##
        ## sample noise to add ##
        noise = torch.randn_like(z_tar)
        ## sample random timestep for each ##
        bs = inp.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=inp.device)
        noisy_tar = self.noise_scheduler.add_noise(z_tar, noise, timesteps)

        ## predict the noise residual ##
        noise_pred = self.model[list(self.model.keys())[1]](x=noisy_tar, timesteps=timesteps, cond=z_coarse_prediction)

        loss_records = {}
        ## evaluate other metrics ##
        data_dict = {}
        data_dict['gt'] = noise
        data_dict['pred'] = noise_pred
        MSE_loss = torch.mean((noise_pred - noise) ** 2).item()
        loss = self.loss(noise_pred, noise) ## important: rescale the loss

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
            # if (utils.get_world_size() > 1 and mpu.get_data_parallel_rank() == 0) or utils.get_world_size() == 1:
            #     wandb.log({f'val_CSI_m': loss_records['CSI_m'] })
        else:
            loss_records.update({'MSE': MSE_loss})
        
        ## log to wandb ##
        # if (utils.get_world_size() > 1 and mpu.get_data_parallel_rank() == 0) or utils.get_world_size() == 1:
        #     wandb.log({f'val_{self.loss_type}': loss.item() })

        return loss_records
    
    
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
    

    def eval_step(self, batch_data, step):
        data_dict = self.data_preprocess(batch_data)
        inp, tar = data_dict['inputs'], data_dict['data_samples']
        ## first predict coarse prediction ##
        coarse_prediction = self.model[list(self.model.keys())[0]](inp)
        ## noise prediction ##
        ## sample noise to add ##
        noise = torch.randn_like(tar)
        ## sample random timestep for each ##
        bs = inp.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=inp.device) 
        noisy_tar = self.noise_scheduler.add_noise(tar, noise, timesteps)

        ## predict the noise residual ##
        noise_pred = self.model[list(self.model.keys())[1]](x=noisy_tar, timesteps=timesteps, cond=coarse_prediction)
        
        ## loss ##
        loss_records = {}
        MSE_loss = torch.mean((noise_pred - noise) ** 2).item()
        loss_records.update({'MSE': MSE_loss})

        ################### generate images ###################
        ## sample image ##
        refined_pred = self.denoise(template_data=tar, cond_data=coarse_prediction, bs=tar.shape[0])
        ## evaluate other metrics ##
        data_dict = {}
        if self.metrics_type == 'SEVIRSkillScore':
            data_dict['gt'] = tar
            data_dict['pred'] = refined_pred
            self.eval_metrics.update(target=data_dict['gt'], pred=data_dict['pred'])

        ## save image ##
        if self.visualizer_type == 'sevir_visualizer' and (step) % 1000 == 0:
            self.visualizer.save_pixel_image(pred_image=refined_pred, target_img=tar, step=step)
        else:
            pass
        
        return loss_records
    
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
        #############################################################################
        metrics = self.eval_metrics.compute()
        csi_total = 0
        for i, thr in enumerate(self.eval_metrics.threshold_list):
            losses.update({f'CSI_{thr}': metrics[thr
            ]['csi']})
            csi_total += metrics[thr]['csi']
        losses.update({'CSI_m': csi_total / len(self.eval_metrics.threshold_list)})
        metric_logger.update(**losses)
        self.logger.info('final results: {meters}'.format(meters=str(metric_logger)))
        return None