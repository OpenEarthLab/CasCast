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

class vqgan_model(basemodel):
    def __init__(self, logger, **params) -> None:
        super().__init__(logger, **params)
        self.logger_print_time = False
        self.data_begin_time = time.time()

    def data_preprocess(self, data):
        data_dict = {}
        inp_data = data['inputs'].float().to(self.device, non_blocking=True, dtype=self.data_type)
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
        decoded_images, codebook_indices, q_loss_dict = self.model[list(self.model.keys())[0]](x=inp)
        aeloss, log_dict_ae = self.model[list(self.model.keys())[1]](inputs=tar, reconstructions=decoded_images, 
                                                                  optimizer_idx=0, global_step=step, mask=None, 
                                                                  last_layer=self.get_last_layer(), split='train')
        
        q_loss = q_loss_dict['q_loss']
        loss = aeloss + q_loss
        self.optimizer[list(self.model.keys())[0]].zero_grad()
        loss.backward()
        self.optimizer[list(self.model.keys())[0]].step()

        ## second: train the discriminator ##
        disloss, log_dict_disc = self.model[list(self.model.keys())[1]](inputs=tar, reconstructions=decoded_images,
                                                                     optimizer_idx=1, global_step=step,
                                                                    mask=None, last_layer=self.get_last_layer(), split="train")
        self.optimizer[list(self.model.keys())[1]].zero_grad()
        disloss.backward()
        self.optimizer[list(self.model.keys())[1]].step()


        
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
            self.visualizer.save_pixel_image(pred_image=decoded_images.unsqueeze(1), target_img=tar.unsqueeze(1), step=step) ## (k, b, t, c, h, w) -> (b, t, c, h, w)
        else:
            pass
        
        loss_dict = log_dict_disc
        loss_dict.update(log_dict_ae)
        loss_dict.update(q_loss_dict)
        new_loss_dict = {}
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                new_loss_dict.update({k: v.item()})
            else:
                new_loss_dict.update({k: v})
        return loss_dict


    def test_one_step(self, batch_data):
        data_dict = self.data_preprocess(batch_data)
        inp, tar = data_dict['inputs'], data_dict['data_samples']
       ## first: train encoder+decoder+logvar ##
        decoded_images, codebook_indices, q_loss_dict = self.model[list(self.model.keys())[0]](x=inp)
        aeloss, log_dict_ae = self.model[list(self.model.keys())[1]](inputs=tar, reconstructions=decoded_images, 
                                                                  optimizer_idx=0, global_step=0, mask=None, 
                                                                  last_layer=self.get_last_layer(), split='val')

        ## second: train the discriminator ##
        disloss, log_dict_disc = self.model[list(self.model.keys())[1]](inputs=tar, reconstructions=decoded_images,
                                                                     optimizer_idx=1, global_step=0,
                                                                    mask=None, last_layer=self.get_last_layer(), split="val")

        
        loss_dict = log_dict_disc
        loss_dict.update(log_dict_ae)
        loss_dict.update(q_loss_dict)
        new_loss_dict = {}
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                new_loss_dict.update({k: v.item()})
            else:
                new_loss_dict.update({k: v})

        loss_dict.update({'MSE': F.mse_loss(decoded_images, tar).item()})
        return loss_dict
    
    
    
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