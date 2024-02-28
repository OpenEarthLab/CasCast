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

class Iter_delta_model(basemodel):
    def __init__(self, logger, **params) -> None:
        super().__init__(logger, **params)
        self.logger_print_time = False
        self.data_begin_time = time.time()
        self.z_score_delta = (params.get('extra_params', {})).get('z_score_delta', False)

    def data_preprocess(self, data):
        data_dict = {}
        ## TODO: [0] is for single frame model
        tar_node_data = torch.concat(tuple(data['gt_pred_node_array'][0]), dim=0).float().to(self.device, non_blocking=True, dtype=self.data_type)
        tar_edge_data = torch.concat(tuple(data['gt_pred_edge_array'][0]), dim=0).float().to(self.device, non_blocking=True, dtype=self.data_type)
        inp_node_data = torch.concat(tuple(data['inp_node_array'][0]), dim=0).float().to(self.device, non_blocking=True, dtype=self.data_type)
        inp_edge_data = torch.concat(tuple(data['inp_edge_array'][0]), dim=0).float().to(self.device, non_blocking=True, dtype=self.data_type)
        data_dict.update({'gt_pred_node_array': tar_node_data, 'gt_pred_edge_array': tar_edge_data, 'inp_node_array': inp_node_data, 'inp_edge_array': inp_edge_data})
        ## information of replay buffer ##
        if "time_step" in data.keys():
            data_dict['time_step'] = data['time_step']
        if "tar_file_id" in data.keys():
            data_dict['tar_file_id'] = data['tar_file_id']
        return data_dict


    def train_one_step(self, batch_data, step):
        data_dict = self.data_preprocess(batch_data)
        inp_node_data, inp_edge_data, tar_node_data, tar_edge_data = data_dict["inp_node_array"], data_dict["inp_edge_array"], data_dict["gt_pred_node_array"], data_dict["gt_pred_edge_array"]
        delta_node, delta_edge = tar_node_data-inp_node_data, tar_edge_data-inp_edge_data
        if self.z_score_delta:
            delta_node = (delta_node - self.t_node_mean) / (self.t_node_std + 0.01)
            delta_edge = (delta_edge - self.t_edge_mean) / (self.t_edge_std + 0.01) 
        self.optimizer[list(self.model.keys())[0]].zero_grad()
        with amp.autocast(enabled=self.enabled_amp):
            predict = self.model[list(self.model.keys())[0]]([inp_node_data, inp_edge_data])
            delta_node_prediction, delta_edge_prediction = predict
            node_loss = self.loss(delta_node_prediction, delta_node)
            edge_loss = self.loss(delta_edge_prediction, delta_edge)
            loss = node_loss + edge_loss
            self.gscaler.scale(loss).backward()
            
        self.gscaler.step(self.optimizer[list(self.model.keys())[0]])
        self.gscaler.update()
        # if (not torch.distributed.is_initialized()) or (mpu.get_tensor_model_parallel_rank() == 0 and self.num_results2save > self.id_results2save):
        #     self.save_test_results(tar_node_data, node_prediction, type='node', dataset='train', set_num=False)
        #     self.save_test_results(tar_edge_data, edge_prediction, type='edge', dataset='train', set_num=True)

        ## store data into replay buffer ###
        if hasattr(self.train_data_loader.dataset, "use_replay_buffer") and self.train_data_loader.dataset.use_replay_buffer:
            node_prediction = delta_node_prediction.float() + inp_node_data.float()
            edge_prediction = delta_edge_prediction.float() + inp_edge_data.float()
            for idx, _ in enumerate(data_dict["tar_file_id"]):
                inp_file_id = data_dict["tar_file_id"][idx]
                if not self.train_data_loader.dataset.check_file_id(inp_file_id):
                    continue
                tar_file_id = self.train_data_loader.dataset.get_tar_file_id(inp_file_id)
                time_step = data_dict["time_step"][idx] + 1
                self.train_data_loader.dataset.replay_buffer.store(inp_data={'inp_node':node_prediction[idx:idx+1].detach().cpu(), 'inp_edge':edge_prediction[idx:idx+1].detach().cpu()}
                                    , tar_idx=tar_file_id, time_step=time_step)

        return {self.loss_type: loss.item(), "delta_node_loss": node_loss.item(), "delta_edge_loss": edge_loss.item()}
    


    def test_one_step(self, batch_data):
        data_dict = self.data_preprocess(batch_data)
        inp_node_data, inp_edge_data, tar_node_data, tar_edge_data = data_dict["inp_node_array"], data_dict["inp_edge_array"], data_dict["gt_pred_node_array"], data_dict["gt_pred_edge_array"]
        delta_node, delta_edge = tar_node_data-inp_node_data, tar_edge_data-inp_edge_data
        if self.z_score_delta:
            delta_node = (delta_node - self.t_node_mean) / (self.t_node_std + 0.01)
            delta_edge = (delta_edge - self.t_edge_mean) / (self.t_edge_std + 0.01) 
        predict = self.model[list(self.model.keys())[0]]([inp_node_data, inp_edge_data])
        delta_node_prediction, delta_edge_prediction = predict

        node_loss = self.loss(delta_node_prediction, delta_node)
        edge_loss = self.loss(delta_edge_prediction, delta_edge)
        loss = node_loss + edge_loss

        # if ((not torch.distributed.is_initialized()) or (mpu.get_tensor_model_parallel_rank() == 0)) and self.num_results2save > self.id_results2save:
        #     self.save_test_results(tar_node_data, node_prediction, type='node', dataset='test', set_num=False)
        #     self.save_test_results(tar_edge_data, edge_prediction, type='edge', dataset='test', set_num=True)

        ## evaluate the prediction ##
        ## node ##
        data_dict = {}
        data_dict['gt'] = delta_node
        data_dict['pred'] = delta_node_prediction
        data_dict['std'] = self.node_std
        data_dict['mean'] = self.node_mean
        metrics_loss = self.eval_metrics.evaluate_batch(data_dict, var_names=self.node_var_names)
        metrics_loss.update({"node_MSE":metrics_loss['MSE'], "node_MAE":metrics_loss['MAE']})
        del metrics_loss['MSE'], metrics_loss['MAE']

        ## edge ##
        data_dict = {}
        data_dict['gt'] = delta_edge
        data_dict['pred'] = delta_edge_prediction
        data_dict['std'] = self.edge_std
        data_dict['mean'] = self.edge_mean
        edge_metrics = self.eval_metrics.evaluate_batch(data_dict, var_names=self.edge_var_names)
        edge_metrics.update({"edge_MSE":edge_metrics['MSE'], "edge_MAE":edge_metrics['MAE']})
        del edge_metrics['MSE'], edge_metrics['MAE']
        metrics_loss.update(edge_metrics)

        ## update MAE for save_best ##
        metrics_loss.update({"MAE": metrics_loss['node_MAE'] + metrics_loss['edge_MAE']})
        metrics_loss.update({"MSE": metrics_loss['node_MSE'] + metrics_loss['edge_MSE']})
        return metrics_loss
    
    def test_data_preprocess(self, data):
        inp_node_data = torch.concat(tuple(data['inp_node_array'][0]), dim=0).float().to(self.device, non_blocking=True, dtype=self.data_type)
        inp_edge_data = torch.concat(tuple(data['inp_edge_array'][0]), dim=0).float().to(self.device, non_blocking=True, dtype=self.data_type)
        tar_node_datas = torch.concat(tuple(data['gt_pred_node_array']), dim=0).float() ## avoid out of memory 
        tar_edge_datas = torch.concat(tuple(data['gt_pred_edge_array']), dim=0).float()
        return inp_node_data, inp_edge_data, tar_node_datas, tar_edge_datas
    
    def multi_step_predict(self, batch_data, data_std, step, predict_length, base_index, **kwargs):
        # batch_len = batch_data[0].__len__()
        # index = (step + 1) * batch_len + base_index ## to obatin data from dataset
        inp_node, inp_edge, tar_node_datas, tar_edge_datas = self.test_data_preprocess(batch_data)
        ## TODO ## 注意/std时为了防止nan值加了0.01
        # if self.z_score_delta:
        #     delta_node = (delta_node - self.t_node_mean) / self.t_node_std
        #     delta_edge = (delta_edge - self.t_edge_mean) / self.t_edge_std 
        metric_steps = self.test_data_loader.dataset.sample_steps[self.test_data_loader.dataset.input_length:]

        metrics_losses = []
        for i in range(predict_length):
            predict = self.model[list(self.model.keys())[0]]([inp_node, inp_edge])
            node_prediction, edge_prediction = predict
            inp_node = node_prediction
            inp_edge = edge_prediction
            if (i+1) in metric_steps:
                gt_ind = metric_steps.index(i+1)
                tar_node, tar_edge = tar_node_datas[gt_ind].to(self.device, non_blocking=True, dtype=self.data_type), tar_edge_datas[gt_ind].to(self.device, non_blocking=True, dtype=self.data_type)
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
            if isinstance(batch, int):
                batch = None

            loss = self.test_one_step(batch)
            metric_logger.update(**loss)

        self.mulvar_logger(metric_logger, epoch)
        # self.logger.info('  '.join(
        #         [f'Epoch [{epoch + 1}](val stats)',
        #          "{meters}"]).format(
        #             meters=str(metric_logger)
        #          ))

        return metric_logger
    
    def mulvar_logger(self, metric_logger, epoch):
        ## log general metrics ##
        general_metrics = ["node_MSE", "node_MAE", "edge_MAE", "edge_MSE"]
        general_info = []
        for metric in general_metrics:
            meter = metric_logger.meters[metric]
            general_info.append(f'{metric}: {meter.global_avg:.5f}')
        self.logger.info(" ".join([
                f'Epoch [{epoch + 1}](train stats) ',
                " ".join(general_info), "\n"
        ]))
        ## log mulvar metrics ##
        var_wise_metrics = "channel_RMSE"
        self.logger.info(f"{var_wise_metrics} \n")
        table_data = []
        table_head = ["var_name"] + self.eval_metrics.var_names
        table_data.append(table_head)
        for h in self.eval_metrics.var_heights:
            row_data = [h]
            for  var in self.eval_metrics.var_names:
                row_data.append(f"{metric_logger.meters[f'{var_wise_metrics}_{var}-{h}'].global_avg:.5f}")
            table_data.append(row_data)
        table = AsciiTable(table_data)
        self.logger.info(table.table)
        self.logger.info('\n')

        ## save table ##
        # 创建Workbook对象
        workbook = Workbook()
        # 获取默认的工作表
        sheet = workbook.active
        # 将AsciiTable内容写入工作表
        for row in table_data:
            sheet.append(row)
        # 保存为XLSX文件
        workbook.save(f'{self.run_dir}/table.xlsx')

