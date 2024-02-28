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

### paralle ensemble ###
from megatron_utils.parallel_state import get_ensemble_parallel_group

class Iter_model(basemodel):
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


    def train_one_step(self, batch_data, step):
        data_dict = self.data_preprocess(batch_data)
        inp, tar = data_dict['inputs'], data_dict['data_samples']
        self.optimizer[list(self.model.keys())[0]].zero_grad()
        with amp.autocast(enabled=self.enabled_amp):
            prediction = self.model[list(self.model.keys())[0]](inp, pred_len=tar.shape[1])
            loss = self.loss(prediction, tar)
            self.gscaler.scale(loss).backward()
            
        self.gscaler.step(self.optimizer[list(self.model.keys())[0]])
        self.gscaler.update()
        # if (not torch.distributed.is_initialized()) or (mpu.get_tensor_model_parallel_rank() == 0 and self.num_results2save > self.id_results2save):
        #     self.save_test_results(tar_node_data, node_prediction, type='node', dataset='train', set_num=False)
        #     self.save_test_results(tar_edge_data, edge_prediction, type='edge', dataset='train', set_num=True)
        # ## replay buffer ##
        # ### store data into replay buffer ###
        # if hasattr(self.train_data_loader.dataset, "use_replay_buffer") and self.train_data_loader.dataset.use_replay_buffer:
        #     for idx, _ in enumerate(data_dict["tar_file_id"]):
        #         inp_file_id = data_dict["tar_file_id"][idx]
        #         if not self.train_data_loader.dataset.check_file_id(inp_file_id):
        #             continue
        #         tar_file_id = self.train_data_loader.dataset.get_tar_file_id(inp_file_id)
        #         time_step = data_dict["time_step"][idx] + 1
        #         self.train_data_loader.dataset.replay_buffer.store(inp_data={'inp_node':node_prediction[idx:idx+1].detach().cpu(), 'inp_edge':edge_prediction[idx:idx+1].detach().cpu()}
        #                             , tar_idx=tar_file_id, time_step=time_step)

        return {self.loss_type: loss.item()}
    


    def test_one_step(self, batch_data):
        data_dict = self.data_preprocess(batch_data)
        inp, tar = data_dict['inputs'], data_dict['data_samples']
        prediction = self.model[list(self.model.keys())[0]](inp, pred_len=tar.shape[1])
        loss_records = {}
        # loss = self.loss(prediction, tar)

        # # if ((not torch.distributed.is_initialized()) or (mpu.get_tensor_model_parallel_rank() == 0)) and self.num_results2save > self.id_results2save:
        # #     self.save_test_results(tar_node_data, node_prediction, type='node', dataset='test', set_num=False)
        # #     self.save_test_results(tar_edge_data, edge_prediction, type='edge', dataset='test', set_num=True)

        # loss_records = {}
        # ## evaluate the prediction ##
        # if 'HSS' or 'CSI' in self.eval_metrics_list:
        #     thresholds = [0.5, 2, 5, 10, 30]
        #     ## compute HSS and CSI ##
        #     for thr in thresholds:
        #         pixel_threshold = self.train_data_loader.dataset.rainfall_to_pixel(thr)
        #         threshold_tar = torch.where(tar >= pixel_threshold, torch.tensor(1), torch.tensor(0))
        #         threshold_pred = torch.where(prediction >= pixel_threshold, torch.tensor(1), torch.tensor(0))
        #         data_dict = {}
        #         data_dict['gt'] = threshold_tar
        #         data_dict['pred'] = threshold_pred
        #         HSS_metrics_loss = self.eval_metrics.evaluate_batch_metric(data_dict, metric='HSS')
        #         CSI_metrics_loss = self.eval_metrics.evaluate_batch_metric(data_dict, metric='CSI')
        #         loss_records.update({f'HSS_{thr}': HSS_metrics_loss['HSS'], f'CSI_{thr}': CSI_metrics_loss['CSI']})

        ## evaluate other metrics ##
        data_dict = {}
        data_dict['gt'] = tar
        data_dict['pred'] = prediction
        # data_dict['std'] = self.node_std
        # data_dict['mean'] = self.node_mean
        metrics_loss = self.eval_metrics.evaluate_batch(data_dict)
        loss_records.update(metrics_loss)
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
    

    def eval_step(self, batch_data):
        data_dict = self.data_preprocess(batch_data)
        inp, tar = data_dict['inputs'], data_dict['data_samples']
        prediction = self.model[list(self.model.keys())[0]](inp, pred_len=tar.shape[1])

        ### the official hko7 evaluator receive input tensor shape: b, t, h, w ##
        losses = {}
        data_dict = {'gt': tar.squeeze(2).cpu().numpy()}
        data_dict = {'pred': prediction.squeeze(2).cpu().numpy()}
        csi, mse, mae = self.eval_metrics.update(gt=data_dict['gt'], pred=data_dict['pred'], mask=self.eval_metrics.mask)

        for i, thr in enumerate(self.eval_metrics._threshold):
            losses.update({f'CSI_{thr}': csi[:, i].mean()})
        
        # for t in range(len(mse)):
        #     losses.update({f'MSE_{t}': mse[t].mean()})
        #     losses.update({f'MAE_{t}': mae[t].mean()})
        
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
                for i in range(predict_length):
                    self.logger.info('  '.join(
                            [f'final valid {i}th step predict (val stats)',
                            "{meters}"]).format(
                                meters=str(metric_logger[i])
                            ))
        return None