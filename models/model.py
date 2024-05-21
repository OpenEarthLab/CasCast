import torch
import torch.nn as nn

# from networks.PVFlash import BidirectionalTransformer
from utils.builder import get_optimizer, get_lr_scheduler
import utils.misc as utils
import time
import datetime
from pathlib import Path
import torch.cuda.amp as amp
from utils.checkpoint_ceph import checkpoint_ceph
import os
from collections import OrderedDict
from torch.functional import F
from fvcore.nn import FlopCountAnalysis, flop_count_table
from megatron_utils import mpu
from utils.misc import is_dist_avail_and_initialized
from megatron_utils.tensor_parallel.data import broadcast_data,get_data_loader_length
import numpy as np
import matplotlib.pyplot as plt
# from terminaltables import AsciiTable
import wandb




class basemodel(nn.Module):
    def __init__(self, logger, **params) -> None:
        super().__init__()
        self.model = {}
        self.sub_model_name = []
        self.params = params
        self.pred_type = self.params.get("pred_type", None)
        self.sampler_type = self.params.get("sampler_type", "DistributedSampler")
        self.data_type = self.params.get("data_type", "fp32")
        if self.data_type == "bf16":
            self.data_type = torch.bfloat16
        elif self.data_type == "fp16":
            self.data_type = torch.float16
        elif self.data_type == "fp32":
            self.data_type = torch.float32
        else:
            raise NotImplementedError
        # gjc: debug #
        self.debug = self.params.get("debug", False)
        self.visual_vars = self.params.get("visual_vars", None)
        self.run_dir = self.params.get("run_dir", None)

        self.logger = logger
        self.save_best_param = self.params.get("save_best", "MSE")
        self.metric_best = None
        self.constants_len = self.params.get("constants_len", 0) ## computed in train.py
        self.extra_params = params.get("extra_params", {})
        self.loss_type = self.extra_params.get("loss_type", "LpLoss")
        self.enabled_amp = self.extra_params.get("enabled_amp", False)
        self.log_step = self.extra_params.get("log_step", 20)
        self.save_epoch_interval = self.extra_params.get("save_epoch_interval", 1)
        self.test_save_steps = self.extra_params.get("test_save_steps", 0)
        self.ceph_checkpoint_path = params.get("ceph_checkpoint_path", None)
        self.metrics_type = params.get("metrics_type", 'None')

        self.begin_epoch = 0
        self.begin_step = 0
        self.metric_best = 1000

        self.gscaler = amp.GradScaler(enabled=self.enabled_amp)

        if self.ceph_checkpoint_path is None:
            self.checkpoint_ceph = None #checkpoint_ceph()
        else:
            self.checkpoint_ceph = checkpoint_ceph(checkpoint_dir=self.ceph_checkpoint_path)

        self.use_ceph = self.params.get('use_ceph', True)

        ## build network ##
        sub_model = params.get('sub_model', {})
        for key in sub_model:
            if key == 'EarthFormer_xy':
                from networks.earthformer_xy import EarthFormer_xy
                self.model[key] = EarthFormer_xy(**sub_model['EarthFormer_xy'])
            elif key == 'autoencoder_kl':
                from networks.autoencoder_kl import autoencoder_kl
                self.model[key] = autoencoder_kl(config=sub_model['autoencoder_kl'])
            elif key == 'lpipsWithDisc':
                from networks.lpipsWithDisc import lpipsWithDisc
                self.model[key] = lpipsWithDisc(config=sub_model['lpipsWithDisc'])
            elif key == 'casformer':
                from networks.casformer import CasFormer
                self.model[key] = CasFormer(**sub_model['casformer'])
            else:
                raise NotImplementedError('Invalid model type.')
            self.sub_model_name.append(key)
        
        # load optimizer and lr_scheduler
        self.optimizer = {}
        self.lr_scheduler = {}
        self.lr_scheduler_by_step = {}

        optimizer = params.get('optimizer', {})
        lr_scheduler = params.get('lr_scheduler', {})
        for key in self.sub_model_name:
            if key in optimizer:
                self.optimizer[key] = get_optimizer(self.model[key], optimizer[key])
            if key in lr_scheduler:
                self.lr_scheduler_by_step[key] = lr_scheduler[key].get('by_step', False)
                self.lr_scheduler[key] = get_lr_scheduler(self.optimizer[key], lr_scheduler[key])

        # load metrics
        eval_metrics_list = params.get('metrics_list', [])
        self.eval_metrics_list = eval_metrics_list
        eval_metrics_vars = params.get('metrics_vars', None)
        if self.metrics_type == 'SEVIRSkillScore':
            from utils.metrics import SEVIRSkillScore
            seq_len = params.get("sevir_seq_len", 12)
            self.eval_metrics = SEVIRSkillScore(layout='NTCHW', seq_len=seq_len, dist_eval=True if is_dist_avail_and_initialized() else False)
        elif self.metrics_type == 'None':
            self.eval_metrics = None
        else:
            raise NotImplementedError

        ## build visualizer ##
        self.visualizer_params = params.get("visualizer", {})
        self.visualizer_type = self.visualizer_params.get("visualizer_type", None)
        self.visualizer_step = self.visualizer_params.get("visualizer_step", 100)
        if self.visualizer_type == 'sevir_visualizer':
            from utils.visualizer import sevir_visualizer
            self.visualizer = sevir_visualizer(exp_dir=self.run_dir)
        else:
            raise NotImplementedError


        for key in self.model:
            self.model[key].eval()

        self.checkpoint_path = self.extra_params.get("checkpoint_path", None)
        if self.checkpoint_path is None:
            self.logger.info("finetune checkpoint path not exist")
        else:
            self.load_checkpoint(self.checkpoint_path, load_model=True, load_optimizer=False, load_scheduler=False, load_epoch=False, load_metric_best=False)
        
        if self.loss_type == "MSELoss":
            self.loss = self.MSELoss
        else: 
            raise NotImplementedError()
        
    def to(self, device):
        self.device = device
        for key in self.model:
            self.model[key].to(device, dtype=self.data_type)

        for key in self.optimizer:
            for state in self.optimizer[key].state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device, dtype=self.data_type)

    def MSELoss(self, pred, target, **kwargs):
        return torch.mean((pred-target)**2)

    def train_one_step(self, batch_data, step):
        input, target = self.data_preprocess(batch_data)
        if len(self.model) == 1:
            predict = self.model[list(self.model.keys())[0]](input)
        else:
            raise NotImplementedError('Invalid model type.')

        loss = self.loss(predict, target)
        if len(self.optimizer) == 1:
            self.optimizer[list(self.optimizer.keys())[0]].zero_grad()
            loss.backward()
            self.optimizer[list(self.optimizer.keys())[0]].zero_grad()
        else:
            raise NotImplementedError('Invalid model type.')
        
        return loss

    
    def test_one_step(self, batch_data):
        input, target = self.data_preprocess(batch_data)
        if len(self.model) == 1:
            predict = self.model[list(self.model.keys())[0]](input)

        data_dict = {}
        data_dict['gt'] = target
        data_dict['pred'] = predict
        if MetricsRecorder is not None:
            loss = self.eval_metrics(data_dict)
        else:
            raise NotImplementedError('No Metric Exist.')
        return loss


    def train_one_epoch(self, train_data_loader, epoch, max_epoches):

        for key in self.lr_scheduler:
            if not self.lr_scheduler_by_step[key]:
                self.lr_scheduler[key].step(epoch)

        end_time = time.time()           
        for key in self.optimizer:              # only train model which has optimizer
            self.model[key].train()

        metric_logger = utils.MetricLogger(delimiter="  ", sync=True)
        iter_time = utils.SmoothedValue(window_size=8, fmt='{avg:.3f}')
        data_time = utils.SmoothedValue(window_size=8, fmt='{avg:.3f}')

        max_step = get_data_loader_length(train_data_loader)
        header = 'Epoch [{epoch}/{max_epoches}][{step}/{max_step}]'

        data_loader = train_data_loader
        self.train_data_loader = train_data_loader
        for step, batch in enumerate(data_loader):
            ## step lr ##
            for key in self.lr_scheduler:
                if self.lr_scheduler_by_step[key]:
                    self.lr_scheduler[key].step(epoch*max_step+step)

            if (self.debug and step >=2):
                self.logger.info("debug mode: break from train loop")
                break
            if isinstance(batch, int):
                batch = None
        
            # record data read time
            data_time.update(time.time() - end_time)
            if self.debug:
                print(f'data_time: {str(data_time)}')

            loss = self.train_one_step(batch, step)

            # record loss and time
            metric_logger.update(**loss)
            iter_time.update(time.time() - end_time)
            end_time = time.time()

            # output to logger
            if (step+1) % self.log_step == 0 or step+1 == max_step:
                eta_seconds = iter_time.global_avg * (max_step - step - 1 + max_step * (max_epoches-epoch-1))
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                self.logger.info(
                    metric_logger.delimiter.join(
                        [header,
                        "lr: {lr}",
                        "eta: {eta}",
                        "time: {time}",
                        "data: {data}",
                        "memory: {memory:.0f}",
                        "{meters}"
                        ]
                    ).format(
                        epoch=epoch+1, max_epoches=max_epoches, step=step+1, max_step=max_step,
                        lr=self.optimizer[list(self.optimizer.keys())[0]].param_groups[0]["lr"],
                        eta=eta_string,
                        time=str(iter_time),
                        data=str(data_time),
                        memory=torch.cuda.memory_reserved() / (1024. * 1024),
                        meters=str(metric_logger)
                    ))
                

    def load_checkpoint(self, checkpoint_path, load_model=True, load_optimizer=True, load_scheduler=True, load_epoch=True, load_metric_best=True,
                        **kwargs):
        if utils.get_world_size() > 1 and mpu.get_tensor_model_parallel_world_size() > 1:
            path1, path2 = checkpoint_path.split('.')
            checkpoint_path = f"{path1}_{mpu.get_tensor_model_parallel_rank()}{path2}"

        if self.use_ceph:
            checkpoint_dict = self.checkpoint_ceph.load_checkpoint(checkpoint_path)
            if checkpoint_dict is None:
                self.logger.info("checkpoint is not exist")
                return
        elif os.path.exists(checkpoint_path):
            if checkpoint_path is None or checkpoint_path=='None':
                self.logger.info("checkpoint is not exist")
                return
            checkpoint_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        else:
            self.logger.info("checkpoint is not exist")
            return
        checkpoint_model = checkpoint_dict['model']
        checkpoint_optimizer = checkpoint_dict['optimizer']
        checkpoint_lr_scheduler = checkpoint_dict['lr_scheduler']
        ### load model for lora training ##
        lora = kwargs.get('lora', False)
        lora_base_model = kwargs.get('lora_base_model', 'DiT')
        if lora:
            print(f"load model weight for lora training !!!")
            if lora_base_model + '_lora' in checkpoint_model.keys():
                print(f"load {lora_base_model}_lora already exsits in ckpt")
            else:
                checkpoint_model[lora_base_model+'_lora'] = checkpoint_model[lora_base_model]

        ###################################
        if load_model:
            ckpt_submodels = list(checkpoint_model.keys())
            submodels = list(self.model.keys())
            for key in checkpoint_model:
                if key not in submodels:
                    print(f"warning!!!!!!!!!!!!!: skip load of {key}")
                    continue
                new_state_dict = OrderedDict()
                for k, v in checkpoint_model[key].items():
                    if "module" == k[:6]:
                        name = k[7:]
                    else:
                        name = k
                    new_state_dict[name] = v
                self.model[key].load_state_dict(new_state_dict, strict=False)
        ######################################
        if load_optimizer:
            resume = kwargs.get('resume', False)
            for key in checkpoint_optimizer:
                self.optimizer[key].load_state_dict(checkpoint_optimizer[key])
                if resume: #for resume train
                    self.optimizer[key].param_groups[0]['capturable'] = True
        if load_scheduler:
            for key in checkpoint_lr_scheduler:
                self.lr_scheduler[key].load_state_dict(checkpoint_lr_scheduler[key])
        if load_epoch:
            self.begin_epoch = checkpoint_dict['epoch']
            self.begin_step = 0 if 'step' not in checkpoint_dict.keys() else checkpoint_dict['step']
        if load_metric_best and 'metric_best' in checkpoint_dict:
            self.metric_best = checkpoint_dict['metric_best']
        if 'amp_scaler' in checkpoint_dict:
            self.gscaler.load_state_dict(checkpoint_dict['amp_scaler'])
        self.logger.info("last epoch:{epoch}, metric best:{metric_best}".format(epoch=checkpoint_dict['epoch'], metric_best=checkpoint_dict['metric_best']))


    def save_checkpoint(self, epoch, checkpoint_savedir, save_type='save_best', step=0): 
        checkpoint_savedir = Path(checkpoint_savedir)
        # checkpoint_path = checkpoint_savedir / '{}'.format('checkpoint_best.pth' \
        #                     if save_type == 'save_best' else 'checkpoint_latest.pth')

    
        # print(save_type, checkpoint_path)

        if (utils.get_world_size() > 1 and mpu.get_tensor_model_parallel_world_size() == 1) or utils.get_world_size() == 1:
            if save_type == "save_best":
                checkpoint_path = checkpoint_savedir / '{}'.format('checkpoint_best.pth')
            else:
                checkpoint_path = checkpoint_savedir / '{}'.format('checkpoint_latest.pth')
        else:
            if save_type == "save_best":
                checkpoint_path = checkpoint_savedir / f'checkpoint_best_{mpu.get_tensor_model_parallel_rank()}.pth'
            else:
                checkpoint_path = checkpoint_savedir / f'checkpoint_latest_{mpu.get_tensor_model_parallel_rank()}.pth'


        if utils.get_world_size() > 1 and mpu.get_data_parallel_rank() == 0:
            if self.use_ceph:
                self.checkpoint_ceph.save_checkpoint(
                    checkpoint_path,
                    {
                    'step':             step,
                    'epoch':            epoch+1,
                    'model':            {key: self.model[key].module.state_dict() for key in self.model},
                    'optimizer':        {key: self.optimizer[key].state_dict() for key in self.optimizer},
                    'lr_scheduler':     {key: self.lr_scheduler[key].state_dict() for key in self.lr_scheduler},
                    'metric_best':      self.metric_best,
                    'amp_scaler':       self.gscaler.state_dict(),
                    # "max_logvar":       self.max_logvar if hasattr(self, 'max_logvar') else None,
                    # "min_logvar":       self.min_logvar if hasattr(self, 'min_logvar') else None,
                    }
                )
            else:
                torch.save(
                    {
                    'step':             step,
                    'epoch':            epoch+1,
                    'model':            {key: self.model[key].module.state_dict() for key in self.model},
                    'optimizer':        {key: self.optimizer[key].state_dict() for key in self.optimizer},
                    'lr_scheduler':     {key: self.lr_scheduler[key].state_dict() for key in self.lr_scheduler},
                    'metric_best':      self.metric_best,
                    'amp_scaler':       self.gscaler.state_dict(),
                    # "max_logvar":       self.max_logvar if hasattr(self, 'max_logvar') else None,
                    # "min_logvar":       self.min_logvar if hasattr(self, 'min_logvar') else None,
                    }, checkpoint_path
                )
        elif utils.get_world_size() == 1:
            if self.use_ceph:
                self.checkpoint_ceph.save_checkpoint(
                    checkpoint_path,
                    {
                    'step':             step,
                    'epoch':            epoch+1,
                    'model':            {key: self.model[key].state_dict() for key in self.model},
                    'optimizer':        {key: self.optimizer[key].state_dict() for key in self.optimizer},
                    'lr_scheduler':     {key: self.lr_scheduler[key].state_dict() for key in self.lr_scheduler},
                    'metric_best':      self.metric_best,
                    'amp_scaler':       self.gscaler.state_dict(),
                    # "max_logvar":       self.max_logvar if hasattr(self, 'max_logvar') else None,
                    # "min_logvar":       self.min_logvar if hasattr(self, 'min_logvar') else None,
                    }
                )
            else:
                torch.save(
                    {
                    'step':             step,
                    'epoch':            epoch+1,
                    'model':            {key: self.model[key].state_dict() for key in self.model},
                    'optimizer':        {key: self.optimizer[key].state_dict() for key in self.optimizer},
                    'lr_scheduler':     {key: self.lr_scheduler[key].state_dict() for key in self.lr_scheduler},
                    'metric_best':      self.metric_best,
                    'amp_scaler':       self.gscaler.state_dict(),
                    # "max_logvar":       self.max_logvar if hasattr(self, 'max_logvar') else None,
                    # "min_logvar":       self.min_logvar if hasattr(self, 'min_logvar') else None,
                    }, checkpoint_path
                )


    def whether_save_best(self, metric_logger):
        metric_now = metric_logger.meters[self.save_best_param].global_avg
        if self.metric_best is None:
            self.metric_best = metric_now
            return True
        if metric_now < self.metric_best:
            self.metric_best = metric_now
            return True
        return False



    def trainer(self, train_data_loader, test_data_loader, max_epoches, max_steps, checkpoint_savedir=None, save_ceph=False, resume=False):
        
        self.test_data_loader = test_data_loader
        self.train_data_loader = train_data_loader
        ## load temporal mean and std for delta-prediction model ##

        ## the dir of saving models and prediction results ##
        self.checkpoint_savedir = checkpoint_savedir

        if 'TrainingSampler' in self.sampler_type:
            self._iter_trainer(train_data_loader, test_data_loader, max_steps) 
        else:
            self._epoch_trainer(train_data_loader, test_data_loader, max_epoches)

    
    def _epoch_trainer(self, train_data_loader, test_data_loader, max_epoches):
        for epoch in range(self.begin_epoch, max_epoches):
            if train_data_loader is not None:
                train_data_loader.sampler.set_epoch(epoch)

            ## gjc: debug mode ##
            self.train_one_epoch(train_data_loader, epoch, max_epoches)

            # # update lr_scheduler
            if utils.get_world_size() > 1:
                for key in self.model:
                    utils.check_ddp_consistency(self.model[key])

            ## gjc: debug mode ##
            metric_logger = self.test(test_data_loader, epoch)

            # save model
            if self.checkpoint_savedir is not None:
                if self.whether_save_best(metric_logger):
                    self.save_checkpoint(epoch, self.checkpoint_savedir, save_type='save_best')
                if (epoch + 1) % 1 == 0:
                    self.save_checkpoint(epoch, self.checkpoint_savedir, save_type='save_latest')


    def _iter_trainer(self, train_data_loader, test_data_loader, max_steps):
        end_time = time.time()

        metric_logger = utils.MetricLogger(delimiter="  ", sync=True)
        iter_time = utils.SmoothedValue(window_size=8, fmt='{avg:.3f}')
        data_time = utils.SmoothedValue(window_size=8, fmt='{avg:.3f}')

        epoch_step = get_data_loader_length(train_data_loader)
        header = '[{step}/{epoch_step}/{max_steps}]'
        
        data_iter = iter(train_data_loader)
        for step in range(self.begin_step, max_steps):
            ## step lr ##
            for key in self.lr_scheduler:
                self.lr_scheduler[key].step(step)
            ## load data ##
            batch = next(data_iter)
            data_time.update(time.time() - end_time)
            if self.debug:
                print(f'data_time: {str(data_time)}')
            ## train_one_step ##
            loss = self.train_one_step(batch, step)
            ## record loss and time ##
            metric_logger.update(**loss)
            iter_time.update(time.time() - end_time)
            end_time = time.time()

            ## output to logger ##
            if (step+1) % self.log_step == 0 or step+1 == max_steps:
                eta_seconds = iter_time.global_avg*(max_steps - step - 1)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                self.logger.info(
                     metric_logger.delimiter.join(
                        [header,
                        "lr: {lr}",
                        "eta: {eta}",
                        "iter_time: {time}",
                        "data: {data}",
                        "memory: {memory:.0f}",
                        "{meters}"
                        ]
                ).format(
                    step=step+1, epoch_step=epoch_step, max_steps=max_steps,
                    lr=self.optimizer[list(self.optimizer.keys())[0]].param_groups[0]["lr"],
                    eta=eta_string,
                    time=str(iter_time),
                    data=str(data_time),
                    memory=torch.cuda.memory_reserved() / (1024. * 1024),
                    meters=str(metric_logger)
                )
                )

            ## test and save ##
            if (step + 1) % epoch_step == 0 or step+1 == max_steps or (self.debug and step >= 2):
                ## test ##
                train_data_type = self.data_type
                self.data_type = torch.float32
                for key in self.sub_model_name:
                    self.model[key].to(self.device, dtype=self.data_type)
                metric_logger = self.test(test_data_loader, epoch=float(f'{(step+1)/epoch_step:.2f}'))
                self.data_type = train_data_type
                for key in self.sub_model_name:
                    self.model[key].to(self.device, dtype=self.data_type)
                ## save ##
                cur_epoch = int((step+1)/epoch_step)
                save_flag = cur_epoch%self.save_epoch_interval == 0 or step+1 == max_steps
                if save_flag:
                    assert self.checkpoint_savedir is not None
                    if self.whether_save_best(metric_logger):
                        self.save_checkpoint(epoch=(step+1)/epoch_step, checkpoint_savedir=self.checkpoint_savedir, save_type='save_best', step=step+1)
                    self.save_checkpoint(epoch=(step+1)/epoch_step, checkpoint_savedir=self.checkpoint_savedir, save_type='save_latest', step=step+1)


                ## reset metric logger of training loop ##
                metric_logger = utils.MetricLogger(delimiter="  ", sync=True)
                
    @torch.no_grad()
    def test(self, test_data_loader, epoch):
        metric_logger = utils.MetricLogger(delimiter="  ", sync=True)
        # set model to eval
        for key in self.model:
            self.model[key].eval()

        max_step = get_data_loader_length(test_data_loader)

        if test_data_loader is None:
            data_loader = range(max_step)
        else:
            data_loader = test_data_loader

        ## save some results ##
        self.num_results2save = 0
        self.id_results2save = 0
        for step, batch in enumerate(data_loader):
            if self.debug and step>= 2 and self.sub_model_name[0] != "IDLE":
                break
            if isinstance(batch, int):
                batch = None

            loss = self.test_one_step(batch)
            # if step < self.num_results2save and (not torch.distributed.is_initialized() or mpu.get_tensor_model_parallel_rank() == 0) and self.visual_vars is not None: 
            #     self.visualize_one_step(batch, epoch, step)
            metric_logger.update(**loss)

        
        self.logger.info('  '.join(
                [f'Epoch [{epoch + 1}](val stats)',
                 "{meters}"]).format(
                    meters=str(metric_logger)
                 ))

        return metric_logger

    @torch.no_grad()
    def test_final(self, test_data_loader, predict_length):
        pass
    

    










