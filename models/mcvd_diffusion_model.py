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


class mcvd_diffusion_model(basemodel):
    def __init__(self, logger, **params) -> None:
        super().__init__(logger, **params)
        self.logger_print_time = False
        self.data_begin_time = time.time()

        self.diffusion_kwargs = params.get('diffusion_kwargs', {})

        ## init noise scheduler ##
        self.noise_scheduler_kwargs = self.diffusion_kwargs.get('noise_scheduler', {})
        self.noise_scheduler_type = list(self.noise_scheduler_kwargs.keys())[0]
        if self.noise_scheduler_type == 'DDPMScheduler':
            from src.diffusers import DDPMScheduler
            self.noise_scheduler = DDPMScheduler(**self.noise_scheduler_kwargs[self.noise_scheduler_type])
            num_train_timesteps = self.noise_scheduler_kwargs[self.noise_scheduler_type]['num_train_timesteps']
            self.noise_scheduler.set_timesteps(num_train_timesteps)
        elif self.noise_scheduler_type == 'DPMSolverMultistepScheduler':
            from src.diffusers import DPMSolverMultistepScheduler
            import pdb; pdb.set_trace()
            self.noise_scheduler = DPMSolverMultistepScheduler(**self.noise_scheduler_kwargs[self.noise_scheduler_type])
            num_train_timesteps = self.noise_scheduler_kwargs[self.noise_scheduler_type]['num_train_timesteps']
            self.noise_scheduler.set_timesteps(num_train_timesteps)
        else:
            raise NotImplementedError

        ## important: scale the noise to get a reasonable noise process ##
        self.noise_scale = self.noise_scheduler_kwargs.get('noise_scale', 1.0)
        self.logger.info(f'####### noise scale: {self.noise_scale} ##########')

        ## init noise scheduler for sampling ##
        self.sample_noise_scheduler_type = 'DDIMScheduler'
        if self.sample_noise_scheduler_type == 'DDIMScheduler':
            print("############# USING SAMPLER: DDIMScheduler #############")
            from src.diffusers import DDIMScheduler
            self.sample_noise_scheduler = DDIMScheduler(**self.noise_scheduler_kwargs[self.noise_scheduler_type])
            ## set num of inference
            self.sample_noise_scheduler.set_timesteps(20)
        elif self.sample_noise_scheduler_type == 'DDPMScheduler':
            print("############# USING SAMPLER: DDPMScheduler #############")
            from src.diffusers import DDPMScheduler
            self.sample_noise_scheduler = DDPMScheduler(**self.noise_scheduler_kwargs[self.noise_scheduler_type])
            self.sample_noise_scheduler.set_timesteps(1000)
        else:
            raise NotImplementedError
        
        ## scale factor ##
        self.scale_factor = 1.0 ## 1/std

        ## classifier_free_guidance ##
        self.classifier_free_guidance_kwagrs = self.diffusion_kwargs.get('classifier_free_guidance', {})
        self.p_uncond = self.classifier_free_guidance_kwagrs.get('p_uncond', 0.0)
        self.guidance_weight = self.classifier_free_guidance_kwagrs.get('guidance_weight', 0.0)

    def data_preprocess(self, data):
        data_dict = {}
        inp_data = data['inputs'].float().to(self.device, non_blocking=True, dtype=self.data_type)
        tar_data = data['data_samples'].float().to(self.device, non_blocking=True, dtype=self.data_type)
        data_dict.update({'inputs': inp_data, 'data_samples': tar_data})
        return data_dict
    
    @torch.no_grad()
    def denoise(self, template_data, cond_data, bs=1):
        """
        denoise from gaussian.
        """
        _, t, c, h, w = template_data.shape
        cond_data = cond_data[:bs, ...]
        generator = torch.Generator(device=template_data.device) #torch.manual_seed(0)
        generator.manual_seed(0)
        latents = torch.randn(
            (bs, t, c, h, w),
            generator=generator,
            device=template_data.device,
        ) 
        latents = latents * self.sample_noise_scheduler.init_noise_sigma
        ## iteratively denoise ##
        print("start sampling")
        for t in tqdm(self.sample_noise_scheduler.timesteps) if self.debug else self.sample_noise_scheduler.timesteps:
            ## predict the noise residual ##
            timestep = torch.ones((bs,), device=template_data.device) * t
            noise_pred = self.model[list(self.model.keys())[1]](x=latents, timesteps=timestep, cond=cond_data)
            ## compute the previous noisy sample x_t -> x_{t-1} ##
            latents = self.sample_noise_scheduler.step(noise_pred, t, latents).prev_sample
        print("end sampling")

        return latents

    @torch.no_grad()
    def init_scale_factor(self, z_tar):
        del self.scale_factor
        self.logger.info("### USING STD-RESCALING ###")
        _std = z_tar.std()
        _mean = z_tar.mean()
        if utils.get_world_size() == 1 :
            pass
        else:
            dist.barrier()
            dist.all_reduce(_std)
            _std = _std / dist.get_world_size()
            ## center to zero ##
            dist.barrier()
            dist.all_reduce(_mean)
            _mean = _mean / dist.get_world_size()
        scale_factor = 1/_std
        self.logger.info(f'####### scale factor: {scale_factor.item()} ##########')
        self.register_buffer('scale_factor', scale_factor)
        self.logger.info(f'####### MOVE mean: {_mean.item()} ##########')
        self.register_buffer('move_mean', _mean)


    def train_one_step(self, batch_data, step):
        data_dict = self.data_preprocess(batch_data)
        inp, tar = data_dict['inputs'], data_dict['data_samples']
        ## first, generate coarse advective field ##
        with torch.no_grad():
            coarse_prediction = self.model[list(self.model.keys())[0]](inp)
            coarse_prediction = coarse_prediction.detach() 
            ## init scale_factor ##
            if self.scale_factor == 1.0:
                self.init_scale_factor(tar)
            ## important: scale coarse_prediction to [0, 255] ##
            scale_coarse_prediction = (coarse_prediction - self.move_mean) * self.scale_factor
        scale_tar = (tar - self.move_mean) * self.scale_factor

        ## classifier free guidance ##
        p = torch.rand(1)
        if p < self.p_uncond: ## discard condition
            scale_coarse_prediction_cond = torch.zeros_like(scale_coarse_prediction)
        else:
            scale_coarse_prediction_cond = scale_coarse_prediction
        ## sample noise to add ##
        noise = torch.randn_like(tar)
        ## sample random timestep for each ##
        bs = tar.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=inp.device)
        noisy_tar = self.noise_scheduler.add_noise(scale_tar, noise, timesteps)

        ## predict the noise residual ##
        noise_pred = self.model[list(self.model.keys())[1]](x=noisy_tar, timesteps=timesteps, cond=scale_coarse_prediction_cond) ## TODO: important scale

        loss = self.loss(noise_pred, noise) ## important: rescale the loss
        loss.backward()

        ## update params of mcvd ##
        self.optimizer[list(self.model.keys())[1]].step()
        self.optimizer[list(self.model.keys())[1]].zero_grad()

        # if (utils.get_world_size() > 1 and mpu.get_data_parallel_rank() == 0) or utils.get_world_size() == 1:
        #             wandb.log({f'train_{self.loss_type}': loss.item() })
        
        if self.visualizer_type is None:
            pass
        elif self.visualizer_type == 'hko7_visualizer' and (step) % self.visualizer_step==0:
            import pdb; pdb.set_trace() ##TODO sample image
            self.visualizer.save_dbz_image(pred_image=prediction, target_img=tar, step=step)
        elif self.visualizer_type == 'sevir_visualizer' and (step) % self.visualizer_step==0:
            scale_sample_prediction = self.denoise(template_data=tar, cond_data=scale_coarse_prediction, bs=1)
            sample_prediction = (scale_sample_prediction / self.scale_factor) + self.move_mean
            # import pdb; pdb.set_trace() ##TODO sample image
            self.visualizer.save_pixel_image(pred_image=sample_prediction, target_img=tar, step=step)
        else:
            pass
        return {self.loss_type: loss.item()}
    

    @torch.no_grad()
    def test_one_step(self, batch_data):
        data_dict = self.data_preprocess(batch_data)
        inp, tar = data_dict['inputs'], data_dict['data_samples']
        ## first predict coarse prediction ##
        coarse_prediction = self.model[list(self.model.keys())[0]](inp)
        scale_coarse_prediction = (coarse_prediction - self.move_mean) * self.scale_factor
        scale_tar = (tar - self.move_mean) * self.scale_factor
        ## noise prediction ##
        ## sample noise to add ##
        noise = torch.randn_like(tar)
        ## sample random timestep for each ##
        bs = inp.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=inp.device)
        noisy_tar = self.noise_scheduler.add_noise(scale_tar, noise, timesteps)

        ## predict the noise residual ##
        noise_pred = self.model[list(self.model.keys())[1]](x=noisy_tar, timesteps=timesteps, cond=scale_coarse_prediction)

        loss_records = {}
        ## evaluate other metrics ##
        data_dict = {}
        data_dict['gt'] = noisy_tar
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
            if (utils.get_world_size() > 1 and mpu.get_data_parallel_rank() == 0) or utils.get_world_size() == 1:
                wandb.log({f'val_CSI_m': loss_records['CSI_m'] })
        else:
            loss_records.update({'MSE': MSE_loss})
        loss_records.update({'MSE': MSE_loss})
        ## log to wandb ##
        # if (utils.get_world_size() > 1 and mpu.get_data_parallel_rank() == 0) or utils.get_world_size() == 1:
        #     wandb.log({f'val_{self.loss_type}': loss.item() })

        return loss_records
    
    def test_data_preprocess(self, data):
        inp_node_data = torch.concat(tuple(data['inp_node_array'][0]), dim=0).float().to(self.device, non_blocking=True)
        inp_edge_data = torch.concat(tuple(data['inp_edge_array'][0]), dim=0).float().to(self.device, non_blocking=True)
        tar_node_datas = torch.concat(tuple(data['gt_pred_node_array']), dim=0).float() ## avoid out of memory 
        tar_edge_datas = torch.concat(tuple(data['gt_pred_edge_array']), dim=0).float()
        return inp_node_data, inp_edge_data, tar_node_datas, tar_edge_datas
    
    def multi_step_predict(self, batch_data, data_std, step, predict_length, base_index, **kwargs):
        # batch_len = batch_data[0].__len__()
        # index = (step + 1) * batch_len + base_index ## to obatin data from dataset
        inp_node, inp_edge, tar_node_datas, tar_edge_datas = self.test_data_preprocess(batch_data)
        metric_steps = self.test_data_loader.dataset.sample_steps[self.test_data_loader.dataset.input_length:]

        metrics_losses = []
        for i in range(predict_length):
            predict = self.model[list(self.model.keys())[0]]([inp_node, inp_edge])
            node_prediction, edge_prediction = predict
            inp_node = node_prediction
            inp_edge = edge_prediction
            if (i+1) in metric_steps:
                gt_ind = metric_steps.index(i+1)
                tar_node, tar_edge = tar_node_datas[gt_ind].to(self.device, non_blocking=True), tar_edge_datas[gt_ind].to(self.device, non_blocking=True)
                var_names = self.test_data_loader.dataset.get_var_names(type='node')
                data_dict = {}
                data_dict['gt'] = tar_node
                data_dict['pred'] = node_prediction
                data_dict['std'] = data_std['node_std']
                data_dict['mean'] = self.node_mean
                node_metrics = self.eval_metrics.evaluate_batch(data_dict, var_names=var_names)
                var_names = self.test_data_loader.dataset.get_var_names(type='edge')
                data_dict = {}
                data_dict['gt'] = tar_edge
                data_dict['pred'] = edge_prediction
                data_dict['std'] = data_std['edge_std']
                data_dict['mean'] = self.edge_mean
                edge_metrics = self.eval_metrics.evaluate_batch(data_dict, var_names=var_names)
                node_metrics.update(edge_metrics)
                metrics_losses.append(node_metrics)
                save_flag1 = (not torch.distributed.is_initialized() or mpu.get_tensor_model_parallel_rank() == 0) and (self.num_results2save > self.id_results2save)
                save_flag2 = (gt_ind < self.test_save_steps)
                if save_flag1 and save_flag2:
                    set_num = False 
                    self.save_test_results(tar_node, node_prediction, type='node', dataset=f'test_s{self.test_data_loader.dataset.sample_steps[gt_ind+1]}', set_num=set_num)
                    self.save_test_results(tar_edge, edge_prediction, type='edge', dataset=f'test_s{self.test_data_loader.dataset.sample_steps[gt_ind+1]}', set_num=set_num)
        self.id_results2save += 1
        return metrics_losses
    
    
    @torch.no_grad()
    def visualize_one_step(self, batch_data, epoch, step):
        data_dict = self.data_preprocess(batch_data)
        inp_node_data, inp_edge_data, tar_node_data, tar_edge_data = data_dict["inp_node_array"], data_dict["inp_edge_array"], data_dict["gt_pred_node_array"], data_dict["gt_pred_edge_array"]
        predict = self.model[list(self.model.keys())[0]]([inp_node_data, inp_edge_data])
        node_prediction, edge_prediction = predict

        ## denormalization ##
        node_prediction = node_prediction * self.node_std + self.node_mean
        tar_node_data = tar_node_data * self.node_std + self.node_mean
        edge_prediction = edge_prediction * self.edge_std + self.edge_mean
        tar_edge_data = tar_edge_data * self.edge_std + self.edge_mean

        ## get the index of variable to be visualized in data ##
        visual_node_var_idx_dict, visual_edge_var_idx_dict = self.test_data_loader.dataset.get_var_idx_dict(self.visual_vars)
        for var, ind in visual_node_var_idx_dict.items():
            _pred = node_prediction[:, :, ind] 
            _tar = tar_node_data[:, :, ind]
            self.plt_node(_pred, _tar, var=var, epoch=epoch, step=step)
        for var, ind in visual_edge_var_idx_dict.items():
            ## TODO: plt edge
            pass
        
        return None
    
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