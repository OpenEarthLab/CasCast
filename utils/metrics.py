if __name__ == "__main__":
    import sys
    sys.path.append('/mnt/cache/gongjunchao/workdir/radar_forecasting')
import torch
import numpy as np
from typing import  Optional, Sequence
from torchmetrics import Metric
from utils.misc import is_dist_avail_and_initialized
import torch.distributed as dist
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import einops
import torch.nn.functional as F
from einops import rearrange
import scipy
from torchvision.transforms.functional import center_crop


# @torch.jit.script
# def lat(j: torch.Tensor, num_lat: int) -> torch.Tensor:
#     return 90. - j * 180./float(num_lat-1)

# @torch.jit.script
# def latitude_weighting_factor_torch(j: torch.Tensor, num_lat: int, s: torch.Tensor) -> torch.Tensor:
#     return num_lat * torch.cos(3.1416/180. * lat(j, num_lat))/s

# @torch.jit.script
# def weighted_rmse_torch_channels(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#     #takes in arrays of size [n, c, h, w]  and returns latitude-weighted rmse for each chann
#     num_lat = pred.shape[2]
#     #num_long = target.shape[2]
#     lat_t = torch.arange(start=0, end=num_lat, device=pred.device)

#     s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
#     weight = torch.reshape(latitude_weighting_factor_torch(lat_t, num_lat, s), (1, 1, -1, 1))
#     result = torch.sqrt(torch.mean(weight * (pred - target)**2., dim=(-1,-2)))
#     return result

# @torch.jit.script
# def weighted_rmse_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#     result = weighted_rmse_torch_channels(pred, target)
#     return torch.mean(result, dim=0)

# @torch.jit.script
# def weighted_acc_torch_channels(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#     #takes in arrays of size [n, c, h, w]  and returns latitude-weighted acc
#     num_lat = pred.shape[2]
#     #num_long = target.shape[2]
#     lat_t = torch.arange(start=0, end=num_lat, device=pred.device)
#     s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
#     weight = torch.reshape(latitude_weighting_factor_torch(lat_t, num_lat, s), (1, 1, -1, 1))
#     result = torch.sum(weight * pred * target, dim=(-1,-2)) / torch.sqrt(torch.sum(weight * pred * pred, dim=(-1,-2)) * torch.sum(weight * target *
#     target, dim=(-1,-2)))
#     return result

# @torch.jit.script
# def weighted_acc_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#     result = weighted_acc_torch_channels(pred, target)
#     return torch.mean(result, dim=0)

# class Metrics(object):
#     """
#     Define metrics for evaluation, metrics include:

#         - MSE, masked MSE;

#         - RMSE, masked RMSE;

#         - REL, masked REL;

#         - MAE, masked MAE;

#         - Threshold, masked threshold.
#     """
#     def __init__(self, epsilon = 1e-8, **kwargs):
#         """
#         Initialization.

#         Parameters
#         ----------

#         epsilon: float, optional, default: 1e-8, the epsilon used in the metric calculation.
#         """
#         super(Metrics, self).__init__()
#         self.epsilon = epsilon
    
#     def channel_MSE(self, pred, gt, data_mask, clim_time_mean_daily, data_std):
#         """
#         MSE metric.

#         Parameters
#         ----------

#         pred: tensor, required, the predicted;

#         gt: tensor, required, the ground-truth

#         Returns
#         -------

#         The MSE metric.
#         """
#         sample_mse = torch.mean((pred - gt) ** 2)
#         return sample_mse.item()
    
#     def MSE(self, pred, gt, data_mask, clim_time_mean_daily, data_std, **kwargs):
#         """
#         MSE metric.

#         Parameters
#         ----------

#         pred: tensor, required, the predicted;

#         gt: tensor, required, the ground-truth

#         Returns
#         -------

#         The MSE metric.
#         """
#         sample_mse = torch.mean((pred - gt) ** 2)
#         return sample_mse.item()
#     # def Channel_MSE(self, pred, gt, data_mask, clim_time_mean_daily, data_std):
#     #     channel_mse = torch.mean((pred - gt) ** 2, dim=[0,2,3])
#     #     return channel_mse
    
#     # def Channel_MAE(self, pred, gt, data_mask, clim_time_mean_daily, data_std):
#     #     channel_mae = torch.mean(torch.abs(pred - gt), dim=[0,2,3])
#     #     return channel_mae

#     # def Position_MSE(self, pred, gt, data_mask, clim_time_mean_daily, data_std):
#     #     position_mse = torch.mean((pred - gt) ** 2, dim=[0, 1]).reshape(-1)
#     #     return position_mse
    
#     def RMSE(self, pred, gt, data_mask, clim_time_mean_daily, data_std, **kwargs):
#         """
#         RMSE metric.

#         Parameters
#         ----------

#         pred: tensor, required, the predicted;

#         gt: tensor, required, the ground-truth;


#         Returns
#         -------

#         The RMSE metric.
#         """
#         sample_mse = torch.mean((pred - gt) ** 2, dim = [0, 1]) * data_std
#         return torch.sqrt(sample_mse)
    
#     def channel_MAE(self, pred, gt, data_mask, clim_time_mean_daily, data_std):
#         """
#         MAE metric.

#         Parameters
#         ----------
#         pred: tensor, required, the predicted

#         gt: tensor, required, the ground-truth

#         Returns
#         -------
        
#         The MAE metric.
#         """
#         sample_mae = torch.mean(torch.abs(pred - gt), dim=[0, 1]) * data_std
#         return sample_mae
    
#     def MAE(self, pred, gt, data_mask, clim_time_mean_daily, data_std, **kwargs):
#         """
#         MAE metric.

#         Parameters
#         ----------
#         pred: tensor, required, the predicted

#         gt: tensor, required, the ground-truth

#         Returns
#         -------
        
#         The MAE metric.
#         """
#         sample_mae = torch.mean(torch.abs(pred - gt))
#         return sample_mae.item()
    
#     def MedianAE(self, pred, gt, data_mask, clim_time_mean_daily, data_std):
#         """
#         MedianAE metric.

#         Parameters
#         ----------
#         pred: tensor, required, the predicted

#         gt: tensor, required, the ground-truth

#         Returns
#         -------
        
#         The MedianAE metric.
#         """
#         B, N, C = pred.shape
#         ae = torch.abs(pred - gt).reshape(B*N, C)
#         median_ae = torch.median(ae, dim=0)[0] * data_std
#         return median_ae
        

#     def MAPE(self, pred, gt, data_mask, clim_time_mean_daily, data_std, data_mean):
#         """
#         MAPE metric.

#         Parameters
#         ----------
#         pred: tensor, required, the predicted

#         gt: tensor, required, the ground-truth

#         Returns
#         -------
        
#         The MAPE metric.
#         """
#         pred = pred * data_std + data_mean
#         gt = gt * data_std + data_mean
#         ape = torch.abs(pred - gt) / torch.clamp(torch.abs(gt), min=1e-2)
#         sample_mape = torch.mean(ape, dim=[0, 1])
#         return sample_mape

#     def MedianAPE(self, pred, gt, data_mask, clim_time_mean_daily, data_std, data_mean):
#         """
#         MedianAPE metric.

#         Parameters
#         ----------
#         pred: tensor, required, the predicted

#         gt: tensor, required, the ground-truth

#         Returns
#         -------
        
#         The MedianAPE metric.
#         """
#         pred = pred * data_std + data_mean
#         gt = gt * data_std + data_mean
#         B, N, C = pred.shape
#         ape = torch.abs(pred - gt) / torch.clamp(torch.abs(gt), min=1e-16)
#         ape = ape.reshape(B*N, C)
#         sample_median_ape = torch.median(ape, dim=0)[0]
#         return sample_median_ape
    
#     def channel_RMSE(self, pred, gt, data_mask, clim_time_mean_daily, data_std, **kwargs):
#         """
#         channel RMSE metric.

#         Parameters
#         ----------
#         pred: tensor, required, the predicted

#         gt: tensor, required, the ground-truth

#         Returns
#         -------
        
#         The RMSE metric.
#         """
#         sample_mse = torch.mean((pred - gt) ** 2, dim=[0, 1]) * data_std
#         return torch.sqrt(sample_mse)



    
#     def gCRPS(self, pred, gt, data_std, member_std, mode="mean", eps=1e-15):
#         """
#         gCRPS metric.

#         Parameters
#         ----------
#         pred: tensor, required, the predicted mean

#         gt: tensor, required, the ground-truth

#         pred_std: tensor, required, the predicted std

#         Returns
#         -------
        
#         The gCRPS metric.
#         """
#         if member_std is None:
#             sample_mae = torch.mean(torch.abs(pred - gt), dim=[0, 2, 3]) * data_std
#             return sample_mae
#         else:
#             assert mode in ['mean', 'raw'], 'CRPS mode should be mean or raw'
#             data_std = data_std.reshape(1, -1, 1, 1)
#             pred_std = member_std * data_std
#             normed_diff = ((pred - gt)*data_std + eps) / (pred_std + eps)
#             _frac_sqrt_pi = 1 / np.sqrt(np.pi)
#             _normal_dist = torch.distributions.Normal(0, 1)
#             try:
#                 cdf = _normal_dist.cdf(normed_diff)
#                 pdf = _normal_dist.log_prob(normed_diff).exp()
#             except:
#                 print(normed_diff)
#                 raise ValueError
#             crps = (pred_std + eps) * (normed_diff * (2 * cdf - 1) + 2 * pdf - _frac_sqrt_pi)
#             if mode == "mean":
#                 return torch.mean(crps, dim=[0, 2, 3])
#             return crps
    
#     def HSS(self, pred, gt, data_mask, clim_time_mean_daily, data_std, **kwargs):
#         """
#         HSS metric for pred tensor with shape (b, t, c, h, w). 

#         Parameters
#         ----------

#         pred: tensor, required, the predicted;

#         gt: tensor, required, the ground-truth

#         Returns
#         -------

#         The HSS metric.
#         """
#         TP = torch.sum(pred * gt, dim=[2, 3, 4])
#         FN = torch.sum((1 - pred) * gt, dim=[2, 3, 4])
#         FP = torch.sum((1 - pred) * (1 - gt), dim=[2, 3, 4])
#         TN = torch.sum(pred * (1 - gt), dim=[2, 3, 4])
#         HSS = (TP * TN - FN * FP) / ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN) + 1e-6)
#         return HSS.mean()
    
#     def CSI(self, pred, gt, data_mask, clim_time_mean_daily, data_std, **kwargs):
#         """
#         CSI metric for pred tensor with shape (b, t, c, h, w).

#         Parameters
#         ----------

#         pred: tensor, required, the predicted;

#         gt: tensor, required, the ground-truth

#         Returns
#         -------

#         The CSI metric.
#         """
#         eps = 1e-6
#         TP = torch.sum(pred * gt, dim=[2, 3, 4])
#         FN = torch.sum((1 - pred) * gt, dim=[2, 3, 4])
#         FP = torch.sum((1-pred) * (1 - gt), dim=[2, 3, 4])
#         TN = torch.sum(pred * (1 - gt), dim=[2, 3, 4])
#         CSI = TP / (TP + FN + FP + 1e-6)
#         return CSI.mean()


# class MetricsRecorder(object):
#     """
#     Metrics Recorder.
#     """
#     def __init__(self, metrics_list, epsilon = 1e-7, **kwargs):
#         """
#         Initialization.

#         Parameters
#         ----------

#         metrics_list: list of str, required, the metrics name list used in the metric calcuation.

#         epsilon: float, optional, default: 1e-8, the epsilon used in the metric calculation.
#         """
#         super(MetricsRecorder, self).__init__()
#         self.epsilon = epsilon
#         self.metrics = Metrics(epsilon = epsilon)
#         self.metric_str_list = metrics_list
#         self.metrics_list = []
#         for metric in metrics_list:
#             try:
#                 metric_func = getattr(self.metrics, metric)
#                 self.metrics_list.append([metric, metric_func, {}])
#             except Exception:
#                 raise NotImplementedError('Invalid metric type.')
    
#     def evaluate_batch(self, data_dict, **kwargs):
#         """
#         Evaluate a batch of the samples.

#         Parameters
#         ----------

#         data_dict: pred and gt

#         Returns
#         -------

#         The metrics dict.
#         """
#         pred = data_dict['pred']            # (B, C, H, W)
#         gt = data_dict['gt']
#         data_mask = None
#         clim_time_mean_daily = None
#         data_std = data_dict["std"] if "std" in data_dict.keys() else None
#         data_mean = data_dict["mean"] if "mean" in data_dict.keys() else None
#         if "clim_mean" in data_dict:
#             clim_time_mean_daily = data_dict['clim_mean']    #(C, H, W)
#             data_std = data_dict["std"]
#         var_names = kwargs.get('var_names', None)

#         losses = {}
#         for metric_line in self.metrics_list:
#             metric_name, metric_func, metric_kwargs = metric_line
#             if metric_name == 'CSI' or metric_name == 'HSS':
#                 continue
#             if metric_name == "gCRPS":
#                 member_std = data_dict.get('member_std', None)
#                 loss = metric_func(pred=pred, gt=gt, data_std=data_std, member_std=member_std)
#             else:
#                 loss = metric_func(pred, gt, data_mask, clim_time_mean_daily, data_std, data_mean=data_mean)
#             if isinstance(loss, torch.Tensor):
#                 for i in range(len(loss)):
#                     if var_names is not None:
#                         losses[metric_name+'_'+var_names[i]] = loss[i].item()
#                     else:
#                         losses[metric_name+str(i)] = loss[i].item()
#             else:
#                 losses[metric_name] = loss

#         return losses
    
#     def evaluate_batch_metric(self, data_dict, metric, **kwargs):
#         """
#         Evaluate a batch of the samples on specific metric

#         Parameters
#         ----------

#         data_dict: pred and gt

#         Returns
#         -------

#         The metrics dict.
#         """
#         pred = data_dict['pred']            # (B, C, H, W)
#         gt = data_dict['gt']
#         data_mask = None
#         clim_time_mean_daily = None
#         data_std = data_dict["std"] if "std" in data_dict.keys() else None
#         data_mean = data_dict["mean"] if "mean" in data_dict.keys() else None
#         if "clim_mean" in data_dict:
#             clim_time_mean_daily = data_dict['clim_mean']    #(C, H, W)
#             data_std = data_dict["std"]
#         var_names = kwargs.get('var_names', None)

#         losses = {}
#         for metric_line in self.metrics_list:
#             ## choose corresponding metric ##
#             metric_name, metric_func, metric_kwargs = metric_line
#             if metric_name == metric:
#                 loss = metric_func(pred=pred, gt=gt, data_mask=data_mask, clim_time_mean_daily=clim_time_mean_daily, data_std=data_std, data_mean=data_mean)
#                 losses[metric_name] = loss
#                 break

#         return losses

# class MulVar_MetricsRecorder(object):
#     """
#     Metrics Recorder of multivariables. It should record the general metrics and variable-wise metrics.
#     """
#     def __init__(self, metrics_list, epsilon = 1e-7, **kwargs):
#         """
#         Initialization.

#         Parameters
#         ----------

#         metrics_list: list of str, required, the metrics name list used in the metric calcuation.

#         epsilon: float, optional, default: 1e-8, the epsilon used in the metric calculation.
#         """
#         super(MulVar_MetricsRecorder, self).__init__()
#         self.epsilon = epsilon
#         self.metrics = Metrics(epsilon = epsilon)
#         self.metric_str_list = metrics_list
#         self.metrics_list = []
#         eval_metrics_vars_kwargs = kwargs.get('eval_metrics_vars', {})
#         self.var_names = eval_metrics_vars_kwargs.get('names', ["theta", "rho"])
#         self.var_heights = eval_metrics_vars_kwargs.get('heights', [0, 10])
#         self.vars = [f'{name}-{height}' for name in self.var_names for height in self.var_heights]
#         for metric in metrics_list:
#             try:
#                 metric_func = getattr(self.metrics, metric)
#                 self.metrics_list.append([metric, metric_func, {}])
#             except Exception:
#                 raise NotImplementedError('Invalid metric type.')
    
#     def evaluate_batch(self, data_dict, **kwargs):
#         """
#         Evaluate a batch of the samples.

#         Parameters
#         ----------

#         data_dict: pred and gt

#         Returns
#         -------

#         The metrics dict.
#         """
#         pred = data_dict['pred']            # (B, C, H, W)
#         gt = data_dict['gt']
#         data_mask = None
#         clim_time_mean_daily = None
#         data_std = data_dict["std"] if "std" in data_dict.keys() else None
#         data_mean = data_dict["mean"] if "mean" in data_dict.keys() else None
#         if "clim_mean" in data_dict:
#             clim_time_mean_daily = data_dict['clim_mean']    #(C, H, W)
#             data_std = data_dict["std"]
#         var_names = kwargs.get('var_names', None)

#         losses = {}
#         for metric_line in self.metrics_list:
#             metric_name, metric_func, metric_kwargs = metric_line
#             if metric_name == 'CSI' or metric_name == 'HSS':
#                 pass
#             if metric_name == "gCRPS":
#                 member_std = data_dict.get('member_std', None)
#                 loss = metric_func(pred=pred, gt=gt, data_std=data_std, member_std=member_std)
#             else:
#                 loss = metric_func(pred=pred, gt=gt, data_mask=data_mask, clim_time_mean_daily=clim_time_mean_daily, data_std=data_std, data_mean=data_mean)
#             if isinstance(loss, torch.Tensor):
#                 for i in range(len(loss)):
#                     if var_names is not None:
#                         losses[metric_name+'_'+var_names[i]] = loss[i].item()
#                     else:
#                         losses[metric_name+str(i)] = loss[i].item()
#             else:
#                 losses[metric_name] = loss

#         return losses




# class HKO7_MetricsRecorder(object):
#     """
#     Metrics Recorder.
#     """
#     def __init__(self, epsilon = 1e-7, **kwargs):
#         """
#         Initialization.

#         Parameters
#         ----------

#         metrics_list: list of str, required, the metrics name list used in the metric calcuation.

#         epsilon: float, optional, default: 1e-8, the epsilon used in the metric calculation.
#         """
#         super(HKO7_MetricsRecorder, self).__init__()
#         self.epsilon = epsilon
#         ## load excluded mask ##
#         with np.load('/mnt/cache/gongjunchao/workdir/empty/radar-forecasting/nwp/datasets/hko/data_provider/mask_dat.npz') as dat:
#             exclude_mask = 1-dat['exclude_mask'][:] ## set the useful zone to 1
#         self.excluded_mask = exclude_mask
#         ##########################
#         self.rf_threshold = [0.5, 2, 5, 10, 30]
#         self.threshold = [ self.rainfall_to_pixel(rf) for rf in self.rf_threshold] 
#         self._total_TP = 0
#         self._total_FN = 0
#         self._total_FP = 0
#         self._total_TN = 0
    
#     def rainfall_to_pixel(self, rainfall_intensity, a=58.53, b=1.56):
#         # dBZ = 10b log(R) +10log(a)
#         dBR = np.log10(rainfall_intensity) * 10.0
#         dBZ = dBR * b + 10.0 * np.log10(a)
#         pixel_vals = (dBZ + 10.0) / 70.0
#         return pixel_vals
    
#     def evaluate_batch(self, data_dict, **kwargs):
#         """
#         Evaluate a batch of the samples.

#         Parameters
#         ----------

#         data_dict: pred and gt

#         Returns
#         -------

#         The metrics dict.
#         """
#         pred = data_dict['pred']            # (B, T, C, H, W)
#         gt = data_dict['gt']
#         losses = {}

#         mse = torch.mean((pred - gt) ** 2, dim=[0, 1, 2, 3, 4])
#         mae = torch.mean(torch.abs(pred - gt), dim=[0, 1, 2, 3, 4])
#         losses.update({'MSE': mse.item(), 'MAE': mae.item()})

#         ## calculate TP, FN, FP, TN dependent on T ## (seq_len, len(threshold))
#         ## reshape mask from (480, 480) to (1, 1, 1, 480, 480
#         data_mask = self.excluded_mask.reshape(1, 1, 1, 480, 480)
#         data_mask = torch.from_numpy(data_mask).to(pred.device)
#         TPs, FNs, FPs, TNs = [], [], [], []
#         for threshold in self.threshold:
#             tmp_pred, tmp_gt = (pred >= threshold).to(dtype=torch.uint8), (gt >= threshold).to(dtype=torch.uint8)
#             TP = torch.sum(tmp_pred*tmp_gt*data_mask , dim=[0, 2, 3, 4])
#             TN = torch.sum(tmp_pred*(1-tmp_gt)*data_mask, dim=[0, 2, 3, 4])
#             FP = torch.sum((1-tmp_pred)*tmp_gt*data_mask , dim=[0, 2, 3, 4])
#             FN = torch.sum((1-tmp_pred)*(1-tmp_gt)*data_mask, dim=[0, 2, 3, 4])
#             TPs.append(TP)
#             FPs.append(FP)
#             TNs.append(TN)
#             FNs.append(FN)
#         ## dim should be (seq_len, len(threshold))
#         TPs = torch.stack(TPs, dim=1)
#         FPs = torch.stack(FPs, dim=1)
#         TNs = torch.stack(TNs, dim=1)
#         FNs = torch.stack(FNs, dim=1)

#         self._total_TP += TPs
#         self._total_FP += FPs
#         self._total_TN += TNs
#         self._total_FN += FNs

#         ## calculate metrics ##
#         ## http://www.wxonline.info/topics/verif2.html ##
#         a = TPs #self._total_TP
#         b = TNs #self._total_TN
#         c = FPs #self._total_FP
#         d = FNs #self._total_FN
#         csi = a / (a +b +c + self.epsilon)
#         hss = 2 * (a*d - b*c)/((a+c)*(c+d) + (a+b)*(b+d) + self.epsilon)

#         for i, threshold in enumerate(self.rf_threshold):
#             ## temporal_mean ##
#             tm_csi = torch.mean(csi, dim=0)
#             tm_hss = torch.mean(hss, dim=0)
#             losses.update({'csi_{}'.format(threshold): tm_csi[i].item(), 'hss_{}'.format(threshold): tm_hss[i].item()})
#         return losses


@torch.no_grad()
def cal_SSIM(gt, pred, is_img=True):
    '''
    iter_cal=True, gt.shape=pred.shape=[nb b t c h w]
    iter_cal=Fasle, gt.shape=pred.shape=[n t c h w]
    '''
    cal_ssim = StructuralSimilarityIndexMeasure(data_range=int(torch.max(gt)-torch.min(gt)) ).to(gt.device)
    if is_img:
        pred = torch.maximum(pred, torch.min(gt))
        pred = torch.minimum(pred, torch.max(gt))
    pred = einops.rearrange(pred, 'n t c h w -> (n t) c h w')
    gt = einops.rearrange(gt, 'n t c h w -> (n t) c h w')
    ssim = cal_ssim(pred, gt).cpu()
    
    # print(ssim)
    # ssim = cal_ssim_2(pred, gt).cpu()
    
    return ssim.item()

@torch.no_grad()
def cal_PSNR(gt, pred, is_img=True):
    '''
    gt.shape=pred.shape=[n t c h w]
    '''
    cal_psnr = PeakSignalNoiseRatio().to(gt.device)
    if is_img:
        pred = torch.maximum(pred, torch.min(gt))
        pred = torch.minimum(pred, torch.max(gt))
    pred = einops.rearrange(pred, 'n t c h w -> (n t) c h w')
    gt = einops.rearrange(gt, 'n t c h w -> (n t) c h w')
    psnr = 0
    for n in range(pred.shape[0]):
        psnr += cal_psnr(pred[n], gt[n]).cpu()
    return (psnr / pred.shape[0]).item()

@torch.no_grad()
def cal_CRPS(gt, pred, type='avg', scale=4, mode='mean', eps=1e-10):
    """
    gt: (b, t, c, h, w)
    pred: (b, n, t, c, h, w)
    """
    assert mode in ['mean', 'raw'], 'CRPS mode should be mean or raw'
    _normal_dist = torch.distributions.Normal(0, 1)
    _frac_sqrt_pi = 1 / np.sqrt(np.pi)

    b, n, t, _, _, _ = pred.shape
    gt = rearrange(gt, 'b t c h w -> (b t) c h w')
    pred = rearrange(pred, 'b n t c h w -> (b n t) c h w')
    if type == 'avg':
        pred = F.avg_pool2d(pred, scale, stride=scale)
        gt = F.avg_pool2d(gt, scale, stride=scale)
    elif type == 'max':
        pred = F.max_pool2d(pred, scale, stride=scale)
        gt = F.max_pool2d(gt, scale, stride=scale)
    else:
        gt = gt
        pred = pred
    gt = rearrange(gt, '(b t) c h w -> b t c h w', b=b)
    pred = rearrange(pred, '(b n t) c h w -> b n t c h w', b=b, n=n)

    pred_mean = torch.mean(pred, dim=1)
    pred_std = torch.std(pred, dim=1) if n > 1 else torch.zeros_like(pred_mean)
    normed_diff = (pred_mean - gt + eps) / (pred_std + eps)
    cdf = _normal_dist.cdf(normed_diff)
    pdf = _normal_dist.log_prob(normed_diff).exp()

    crps = (pred_std + eps) * (normed_diff * (2 * cdf - 1) + 2 * pdf - _frac_sqrt_pi)
    if mode == "mean":
        return torch.mean(crps).item()
    return crps.item()
    


def _threshold(target, pred ,T):
    t = (target >= T).float()
    p = (pred >= T).float()
    is_nan = torch.logical_or(torch.isnan(target),
                              torch.isnan(pred))
    t[is_nan] = 0
    p[is_nan] = 0
    return t, p


# #################################
# #### HKO7 official evaluator ####
# #################################
# #################################
# try:
#     import cPickle as pickle
# except:
#     import pickle
# import numpy as np
# import logging
# import os

# # try:
# #     from models.numba_accelerated import get_GDL_numba, get_hit_miss_counts_numba, get_balancing_weights_numba
# # except:
# #     raise ImportError("Numba has not been installed correctly!")




# class HKOEvaluation_official(object):
#     def __init__(self, seq_len,
#                   no_ssim=True, threholds=None,
#                   metrics_list=['csi', 'csi-4-avg', 'csi-16-avg',
#                                 'csi-4-max', 'csi-16-max'],
#                   eps=1e-4):
#         self.metrics_list = metrics_list
#         self.eps=eps
#         self._thresholds = np.array([0.5, 2, 5, 10, 30]) if threholds is None else threholds
#         self.g_thresholds = [self.rainfall_to_pixel(threshold) for threshold in self._thresholds]
#         self._seq_len = seq_len
#         self._no_ssim = no_ssim
#         self._exclude_mask = 1-self.get_exclude_mask()
#         self._total_batch_num = 0
#         self.begin()

#     def _threshold(self, target, pred ,T):
#         t = target >= T
#         p = pred >= T
#         is_nan = np.logical_or(np.isnan(target),
#                                 np.isnan(pred))
#         t[is_nan] = 0.0
#         p[is_nan] = 0.0
#         return t, p
        
#     def get_exclude_mask(self):
#         with np.load('/mnt/cache/gongjunchao/workdir/empty/radar-forecasting/nwp/datasets/hko/data_provider/mask_dat.npz') as dat:
#             exclude_mask = dat['exclude_mask'][:]
#             return exclude_mask
        
#     def rainfall_to_pixel(self, rainfall_intensity, a=58.53, b=1.56):
#         dBR = np.log10(rainfall_intensity) * 10.0
#         # dBZ = 10b log(R) +10log(a)
#         dBZ = dBR * b + 10.0 * np.log10(a)
#         pixel_vals = (dBZ + 10.0) / 70.0
#         return pixel_vals
        
#     def calc_seq_hits_misses_fas(self, pred, target, thresholds, mask):
#         # _exclude_mask = 1-get_exclude_mask()[np.newaxis,:,:].repeat(18, axis=0)
#         t, p = self._threshold(target, pred, thresholds)
#         # print(self.hits_misses_fas_reduce_dims)
#         hits = np.sum(mask * t * p, axis=(2, 3)).astype(int)
#         misses = np.sum(mask * t * (1 - p), axis=(2, 3)).astype(int)
#         fas = np.sum(mask * (1 - t) * p, axis=(2, 3)).astype(int)
#         # hits = torch.sum(t * p, dim=[0,1,2,3]).int()
#         # misses = torch.sum(t * (1 - p), dim=[0,1,2,3]).int()
#         # fas = torch.sum((1 - t) * p, dim=[0,1,2,3]).int()
#         return hits, misses, fas
        
#     def begin(self):
#         self._total_hits = np.zeros((self._seq_len, len(self._thresholds)), dtype=int)
#         self._total_misses = np.zeros((self._seq_len, len(self._thresholds)),  dtype=int)
#         self._total_false_alarms = np.zeros((self._seq_len, len(self._thresholds)), dtype=int)
#         # self._total_correct_negatives = np.zeros((self._seq_len, len(self._thresholds)),
#         #                                          dtype=int)
#         self._mse = np.zeros((self._seq_len, ), dtype=np.float32)
#         self._mae = np.zeros((self._seq_len, ), dtype=np.float32)
#         # self._balanced_mse = np.zeros((self._seq_len, ), dtype=np.float32)
#         # self._balanced_mae = np.zeros((self._seq_len,), dtype=np.float32)
#         # self._gdl = np.zeros((self._seq_len,), dtype=np.float32)
#         self._ssim = np.zeros((self._seq_len,), dtype=np.float32)
#         self._datetime_dict = {}
#         self._total_batch_num = 0
#         self._exclude_mask = 1-self.get_exclude_mask()[np.newaxis,:,:]

#     def reset(self):
#         self._total_hits[:] = 0
#         self._total_misses[:] = 0
#         self._total_false_alarms[:] = 0
#         # self._total_correct_negatives[:] = 0
#         self._mse[:] = 0
#         self._mae[:] = 0
#         # self._gdl[:] = 0
#         # self._ssim[:] = 0
#         self._total_batch_num = 0
#         # self._balanced_mse[:] = 0
#         # self._balanced_mae[:] = 0

#     def update(self, gt, pred, mask, start_datetimes=None):
#         """

#         Parameters
#         ----------
#         gt : np.ndarray
#         pred : np.ndarray
#         mask : np.ndarray
#             0 indicates not use and 1 indicates that the location will be taken into account
#         start_datetimes : list
#             The starting datetimes of all the testing instances

#         Returns
#         -------

#         """
#         if start_datetimes is not None:
#             batch_size = len(start_datetimes)
#             assert gt.shape[1] == batch_size
#         else:
#             batch_size = gt.shape[0]
#         #assert gt.shape[0] == self._seq_len
#         #assert gt.shape == pred.shape
#         #assert gt.shape == mask.shape
#         # print(batch_size)
#         self._total_batch_num += batch_size
#         # print(pred.shape)
#         # print(gt.shape)
#         #TODO Save all the mse, mae, gdl, hits, misses, false_alarms and correct_negatives
#         mse = (mask * np.square(pred - gt)).sum(axis=(2, 3))
#         mae = (mask * np.abs(pred - gt)).sum(axis=(2, 3))
#         # weights = get_balancing_weights_numba(data=gt, mask=mask,
#         #                                       base_balancing_weights=(1, 1, 2, 5, 10, 30),
#         #                                       thresholds=self._thresholds)
#         # ## <2, <5, ... 不同权值的 MSE.
#         # # S*B*1*H*W
#         # balanced_mse = (weights * np.square(pred - gt)).sum(axis=(2, 3, 4))
#         # balanced_mae = (weights * np.abs(pred - gt)).sum(axis=(2, 3, 4))
#         # gdl = get_GDL_numba(prediction=pred, truth=gt, mask=mask)
#         self._mse += mse.sum(axis=0)
#         # print(self._mse)
#         self._mae += mae.sum(axis=0)
#         # self._balanced_mse += balanced_mse.sum(axis=0)
#         # self._balanced_mae += balanced_mae.sum(axis=0)
#         # self._gdl += gdl.sum(axis=0)
#         if not self._no_ssim:
#             raise NotImplementedError
#             # self._ssim += get_SSIM(prediction=pred, truth=gt)
#         for i, threshold in enumerate(self.g_thresholds):
#             # threshold = threshold / 255.0
#             # print('threshold: {}'.format(threshold))
#             hits, misses, false_alarms = self.calc_seq_hits_misses_fas(pred=pred, target=gt, thresholds=threshold, mask=mask)
#             self._total_hits[:,i] += hits.sum(axis=0)
#             # print(self._total_hits)
#             self._total_misses[:,i] += misses.sum(axis=0)
#             self._total_false_alarms[:,i] += false_alarms.sum(axis=0)
#         # self._total_correct_negatives += correct_negatives.sum(axis=0)



#     def calculate_f1_score(self):
#         '''
#         计算每个分段的 precision, recall 和 f1 score.
#         :return:
#         '''
#         # a: TP, b: TN, c: FP, d: FN
#         a = self._total_hits.astype(np.float64)
#         b = self._total_false_alarms.astype(np.float64)
#         c = self._total_misses.astype(np.float64)
#         d = self._total_correct_negatives.astype(np.float64)
#         precision = a / (a + c)
#         recall = a / (a + d)
#         return precision, recall, (2*precision*recall)/(precision+recall)

#     def compute(self):
#         """The following measurements will be used to measure the score of the forecaster

#         See Also
#         [Weather and Forecasting 2010] Equitability Revisited: Why the "Equitable Threat Score" Is Not Equitable
#         http://www.wxonline.info/topics/verif2.html

#         We will denote
#         (a b    (hits       false alarms
#          c d) =  misses   correct negatives)

#         We will report the
#         POD = a / (a + c)
#         FAR = b / (a + b)
#         CSI = a / (a + b + c)
#         Heidke Skill Score (HSS) = 2(ad - bc) / ((a+c) (c+d) + (a+b)(b+d))
#         Gilbert Skill Score (GSS) = HSS / (2 - HSS), also known as the Equitable Threat Score
#             HSS = 2 * GSS / (GSS + 1)
#         MSE = mask * (pred - gt) **2
#         MAE = mask * abs(pred - gt)
#         GDL = valid_mask_h * abs(gd_h(pred) - gd_h(gt)) + valid_mask_w * abs(gd_w(pred) - gd_w(gt))
#         Returns
#         -------

#         """
#         a = self._total_hits.astype(np.float64)
#         b = self._total_false_alarms.astype(np.float64)
#         c = self._total_misses.astype(np.float64)
#         # d = self._total_correct_negatives.astype(np.float64)
#         pod = a / (a + c + 0.01*np.ones_like(a)) if (a+c == 0).any() else a / (a + c)
#         far = a / (a + b + 0.01*np.ones_like(a)) if (a+b == 0).any() else a / (a + b)
#         csi = a / (a + b + c + 0.01*np.ones_like(a)) if (a+b+c == 0).any() else a / (a + b + c)
#         # n = a + b + c + d
#         # aref = (a + b) / n * (a + c)
#         # gss = (a - aref) / (a + b + c - aref + 0.01*np.ones_like(a)) if\
#         #     (a+b+c == 0).any() else (a - aref) / (a + b + c - aref)
#         # hss = 2 * gss / (gss + 1)
#         mse = self._mse / self._total_batch_num
#         mae = self._mae / self._total_batch_num
#         # balanced_mse = self._balanced_mse / self._total_batch_num
#         # balanced_mae = self._balanced_mae / self._total_batch_num
#         # gdl = self._gdl / self._total_batch_num
#         if not self._no_ssim:
#             raise NotImplementedError
#             # ssim = self._ssim / self._total_batch_num
#         # return pod, far, csi, hss, gss, mse, mae, gdl
#         # return pod, far, csi, hss, gss, mse, mae, balanced_mse, balanced_mae, gdl
#         # return pod, far, csi, hss, gss, mse, mae, balanced_mse, balanced_mae
#         # return csi, mse, mae
#         ## TODO: add other metrics ##
#         metrics = {'csi': csi}
#         ret = {}
#         ret['avg'] = {}
#         for threshold in self._thresholds:
#             ret[threshold] = {}

#         for metric in metrics:
#             score_avg = 0
#             for i, threshold in enumerate(self._thresholds):
#                 ret[threshold][metric] = metrics[metric][:,i].mean()
#                 score_avg += metrics[metric][:,i].mean()
#             ret['avg'][metric] = score_avg / len(self._thresholds)
#         return ret



class HKOEvaluation_official(object):
    def __init__(self, seq_len,
                  no_ssim=True, threholds=None, mode='0',
                  layout = 'NTCHW',
                  preprocess_type = 'hko7', dist_eval=False,
                  metrics_list=['csi', 'csi-4-avg', 'csi-16-avg',
                                'csi-4-max', 'csi-16-max'],
                  eps=1e-4):
        self.metrics_list = metrics_list
        self.eps=eps
        self.layout = layout
        self.preprocess_type = preprocess_type
        self.dist_eval = dist_eval

        self._thresholds = np.array([0.5, 2, 5, 10, 30]) if threholds is None else threholds
        self.g_thresholds = [self.rainfall_to_pixel(threshold) for threshold in self._thresholds]
        self.threshold_list = self.g_thresholds

        self._seq_len = seq_len
        self._no_ssim = no_ssim
        self._exclude_mask = torch.tensor(1-self.get_exclude_mask())
        # self._total_batch_num = 0
        # self.begin()

        self.mode = mode
        
        if mode in ("0", ):
            self.keep_seq_len_dim = False
            state_shape = (len(self.threshold_list), )
        elif mode in ("1", "2"):
            self.keep_seq_len_dim = True
            assert isinstance(self.seq_len, int), "seq_len must be provided when we need to keep seq_len dim."
            state_shape = (len(self.threshold_list), self.seq_len)
        else:
            raise NotImplementedError(f"mode {mode} not supported!")
        
        self.hits = torch.zeros(state_shape)
        self.misses = torch.zeros(state_shape)
        self.fas = torch.zeros(state_shape)

        ## pooling csi ##
        self.hits_avg_pool_4 = torch.zeros(state_shape)
        self.misses_avg_pool_4 = torch.zeros(state_shape)
        self.fas_avg_pool_4 = torch.zeros(state_shape)

        self.hits_max_pool_4 = torch.zeros(state_shape)
        self.misses_max_pool_4 = torch.zeros(state_shape)
        self.fas_max_pool_4 = torch.zeros(state_shape)

        self.hits_avg_pool_16 = torch.zeros(state_shape)
        self.misses_avg_pool_16 = torch.zeros(state_shape)
        self.fas_avg_pool_16 = torch.zeros(state_shape)

        self.hits_max_pool_16 = torch.zeros(state_shape)
        self.misses_max_pool_16 = torch.zeros(state_shape)
        self.fas_max_pool_16 = torch.zeros(state_shape)


    def pod(self, hits, misses, fas, eps):
        return hits / (hits + misses + eps)

    def sucr(self, hits, misses, fas, eps):
        return hits / (hits + fas + eps)

    def csi(self, hits, misses, fas, eps):
        return hits / (hits + misses + fas + eps)

    def bias(self, hits, misses, fas, eps):
        bias = (hits + fas) / (hits + misses + eps)
        logbias = torch.pow(bias / torch.log(torch.tensor(2.0)), 2.0)
        return logbias
    
    def synchronize_between_processes(self):
        if not is_dist_avail_and_initialized():
            return
        dist.barrier()
        dist.all_reduce(self.hits)
        dist.all_reduce(self.misses)
        dist.all_reduce(self.fas)
        ### avg 4 ###
        dist.all_reduce(self.hits_avg_pool_4)
        dist.all_reduce(self.misses_avg_pool_4)
        dist.all_reduce(self.fas_avg_pool_4)
        ### max 4 ###
        dist.all_reduce(self.hits_max_pool_4)
        dist.all_reduce(self.misses_max_pool_4)
        dist.all_reduce(self.fas_max_pool_4)
        ### avg 16 ###
        dist.all_reduce(self.hits_avg_pool_16)
        dist.all_reduce(self.misses_avg_pool_16)
        dist.all_reduce(self.fas_avg_pool_16)
        ### max 16 ###
        dist.all_reduce(self.hits_max_pool_16)
        dist.all_reduce(self.misses_max_pool_16)
        dist.all_reduce(self.fas_max_pool_16)
    
    @property
    def hits_misses_fas_reduce_dims(self):
        if not hasattr(self, "_hits_misses_fas_reduce_dims"):
            seq_dim = self.layout.find('T')
            self._hits_misses_fas_reduce_dims = list(range(len(self.layout)))
            if self.keep_seq_len_dim:
                self._hits_misses_fas_reduce_dims.pop(seq_dim)
        return self._hits_misses_fas_reduce_dims
    
    def preprocess(self, pred, target):
        if self.preprocess_type == "hko7":
            pred = pred.detach() / (1. / 255.) 
            target = target.detach() / (1. / 255.)
        else:
            raise NotImplementedError
        return pred, target
    
    def preprocess_pool(self, pred, target, pool_size=4, type='avg'):
        pred = pred.detach() / (1. / 255.)
        target = target.detach() / (1. / 255.)
        b, t, _, _, _ = pred.shape
        pred = rearrange(pred, 'b t c h w -> (b t) c h w')
        target = rearrange(target, 'b t c h w -> (b t) c h w')
        if type == 'avg':
            pred = F.avg_pool2d(pred, kernel_size=pool_size, stride=pool_size)
            target = F.avg_pool2d(target, kernel_size=pool_size, stride=pool_size)
        elif type == 'max':
            pred = F.max_pool2d(pred, kernel_size=pool_size, stride=pool_size)
            target = F.max_pool2d(target, kernel_size=pool_size, stride=pool_size)
        pred = rearrange(pred, '(b t) c h w -> b t c h w', b=b)
        target = rearrange(target, '(b t) c h w -> b t c h w', b=b)
        return pred, target
    
    def calc_seq_hits_misses_fas(self, pred, target, threshold):
        if self._exclude_mask.device != pred.device:
            self._exclude_mask = self._exclude_mask.to(pred.device)
        mask = self._exclude_mask ## (h, w)
        with torch.no_grad():
            t, p = _threshold(target, pred, threshold)
            mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(pred.shape[-2], pred.shape[-1]), mode='nearest').squeeze(0).squeeze(0)
            hits = torch.sum(mask*t * p, dim=self.hits_misses_fas_reduce_dims).int()
            misses = torch.sum(mask*t * (1 - p), dim=self.hits_misses_fas_reduce_dims).int()
            fas = torch.sum(mask*(1 - t) * p, dim=self.hits_misses_fas_reduce_dims).int()
        return hits, misses, fas
    
    
    # def _threshold(self, target, pred ,T):
    #     t = target >= T
    #     p = pred >= T
    #     is_nan = np.logical_or(np.isnan(target),
    #                             np.isnan(pred))
    #     t[is_nan] = 0.0
    #     p[is_nan] = 0.0
    #     return t, p
        
    def get_exclude_mask(self):
        with np.load('/mnt/cache/gongjunchao/workdir/empty/radar-forecasting/nwp/datasets/hko/data_provider/mask_dat.npz') as dat:
            exclude_mask = dat['exclude_mask'][:]
            return exclude_mask
        
    def rainfall_to_pixel(self, rainfall_intensity, a=58.53, b=1.56):
        dBR = np.log10(rainfall_intensity) * 10.0
        # dBZ = 10b log(R) +10log(a)
        dBZ = dBR * b + 10.0 * np.log10(a)
        pixel_vals = (dBZ + 10.0) / 70.0
        return pixel_vals * 255.0
        
    @torch.no_grad()
    def update(self, pred, target):
        ## pool 1 ##
        self.hits = self.hits.to(pred.device)
        self.misses = self.misses.to(pred.device)
        self.fas = self.fas.to(pred.device)
        _pred, _target = self.preprocess(pred, target)
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            self.hits[i] += hits
            self.misses[i] += misses
            self.fas[i] += fas
        ## max pool 4 ##
        self.hits_max_pool_4 = self.hits_max_pool_4.to(pred.device)
        self.misses_max_pool_4 = self.misses_max_pool_4.to(pred.device)
        self.fas_max_pool_4 = self.fas_max_pool_4.to(pred.device)
        _pred, _target = self.preprocess_pool(pred, target, pool_size=4, type='max')
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            self.hits_max_pool_4[i] += hits
            self.misses_max_pool_4[i] += misses
            self.fas_max_pool_4[i] += fas 
        ## max pool 16 ##
        self.hits_max_pool_16 = self.hits_max_pool_16.to(pred.device)
        self.misses_max_pool_16 = self.misses_max_pool_16.to(pred.device)
        self.fas_max_pool_16 = self.fas_max_pool_16.to(pred.device)
        _pred, _target = self.preprocess_pool(pred, target, pool_size=16, type='max')
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            self.hits_max_pool_16[i] += hits
            self.misses_max_pool_16[i] += misses
            self.fas_max_pool_16[i] += fas 
        ## avg pool 4 ##
        self.hits_avg_pool_4 = self.hits_avg_pool_4.to(pred.device)
        self.misses_avg_pool_4 = self.misses_avg_pool_4.to(pred.device)
        self.fas_avg_pool_4 = self.fas_avg_pool_4.to(pred.device)
        _pred, _target = self.preprocess_pool(pred, target, pool_size=4, type='avg')
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            self.hits_avg_pool_4[i] += hits
            self.misses_avg_pool_4[i] += misses
            self.fas_avg_pool_4[i] += fas 
        ## avg pool 16 ##
        self.hits_avg_pool_16 = self.hits_avg_pool_16.to(pred.device)
        self.misses_avg_pool_16 = self.misses_avg_pool_16.to(pred.device)
        self.fas_avg_pool_16 = self.fas_avg_pool_16.to(pred.device)
        _pred, _target = self.preprocess_pool(pred, target, pool_size=16, type='avg')
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            self.hits_avg_pool_16[i] += hits
            self.misses_avg_pool_16[i] += misses
            self.fas_avg_pool_16[i] += fas 

    def _get_hits_misses_fas(self, metric_name):
        if metric_name.endswith('-4-avg'):
            hits = self.hits_avg_pool_4
            misses = self.misses_avg_pool_4
            fas = self.fas_avg_pool_4
        elif metric_name.endswith('-16-avg'):
            hits = self.hits_avg_pool_16
            misses = self.misses_avg_pool_16
            fas = self.fas_avg_pool_16
        elif metric_name.endswith('-4-max'):
            hits = self.hits_max_pool_4
            misses = self.misses_max_pool_4
            fas = self.fas_max_pool_4
        elif metric_name.endswith('-16-max'):
            hits = self.hits_max_pool_16
            misses = self.misses_max_pool_16
            fas = self.fas_max_pool_16
        else:
            hits = self.hits
            misses = self.misses
            fas = self.fas
        return [hits, misses, fas]

    @torch.no_grad()
    def compute(self):
        if self.dist_eval:
            self.synchronize_between_processes()
        
        metrics_dict = {'pod': self.pod,
                        'csi': self.csi,
                        'csi-4-avg': self.csi, 
                        'csi-16-avg': self.csi,
                        'csi-4-max': self.csi, 
                        'csi-16-max': self.csi,
                        'sucr': self.sucr,
                        'bias': self.bias}
        ret = {}
        for threshold in self.threshold_list:
            ret[threshold] = {}
        ret["avg"] = {}

        for metrics in self.metrics_list:
            if self.keep_seq_len_dim:
                score_avg = np.zeros((self.seq_len, ))
            else:
                score_avg = 0
            hits, misses, fas = self._get_hits_misses_fas(metrics)
            # scores = metrics_dict[metrics](self.hits, self.misses, self.fas, self.eps)
            scores = metrics_dict[metrics](hits, misses, fas, self.eps)
            scores = scores.detach().cpu().numpy()
            for i, threshold in enumerate(self.threshold_list):
                if self.keep_seq_len_dim:
                    score = scores[i]  # shape = (seq_len, )
                else:
                    score = scores[i].item()  # shape = (1, )
                if self.mode in ("0", "1"):
                    ret[threshold][metrics] = score
                elif self.mode in ("2", ):
                    ret[threshold][metrics] = np.mean(score).item()
                else:
                    raise NotImplementedError
                score_avg += score
            score_avg /= len(self.threshold_list)
            if self.mode in ("0", "1"):
                ret["avg"][metrics] = score_avg
            elif self.mode in ("2",):
                ret["avg"][metrics] = np.mean(score_avg).item()
            else:
                raise NotImplementedError
        return ret

    @torch.no_grad()
    def get_single_frame_metrics(self, target, pred, metrics=['ssim', 'psnr', ]): #'cspr', 'cspr-4-avg', 'cspr-16-avg', 'cspr-4-max', 'cspr-16-max'
        metric_funcs = {
            'ssim': cal_SSIM,
            'psnr': cal_PSNR
        }
        metrics_dict = {}
        for metric in metrics:
            metric_fun = metric_funcs[metric]
            metrics_dict[metric] = metric_fun(gt=target*255., pred=pred*255., is_img=False)
        return metrics_dict
    
    @torch.no_grad()
    def get_crps(self, target, pred):
        """
        pred: (b, t, c, h, w)/(b, n, t, c, h, w)
        target: (b, t, c, h, w)
        """
        if len(pred.shape) == 5:
            pred = pred.unsqueeze(1)
        crps = cal_CRPS(gt=target, pred=pred, type='none')
        crps_avg_4 = cal_CRPS(gt=target, pred=pred, type='avg', scale=4)
        crps_avg_16 = cal_CRPS(gt=target, pred=pred, type='avg', scale=16)
        crps_max_4 = cal_CRPS(gt=target, pred=pred, type='max', scale=4)
        crps_max_16 = cal_CRPS(gt=target, pred=pred, type='max', scale=16)
        crps_dict = {
            'crps': crps,
            'crps_avg_4': crps_avg_4,
            'crps_avg_16': crps_avg_16,
            'crps_max_4': crps_max_4,
            'crps_max_16': crps_max_16
        }
        return crps_dict

    def reset(self):
        self.hits = self.hits*0
        self.misses = self.misses*0
        self.fas = self.fas*0

        self.hits_avg_pool_4 *= 0
        self.hits_avg_pool_16 *= 0
        self.hits_max_pool_4 *= 0
        self.hits_max_pool_16 *= 0

        self.misses_avg_pool_4 *= 0
        self.misses_avg_pool_16 *= 0
        self.misses_max_pool_4 *= 0
        self.misses_max_pool_16 *= 0
 
        self.fas_avg_pool_4 *= 0
        self.fas_avg_pool_16 *= 0
        self.fas_max_pool_4  *= 0
        self.fas_max_pool_16  *= 0
    


class SEVIRSkillScore(object):
    def __init__(self,
                 layout='NHWT',
                 mode='0',
                 seq_len=None,
                 preprocess_type='sevir',
                 threshold_list=[16, 74, 133, 160, 181, 219],
                 metrics_list=['csi', 'csi-4-avg', 'csi-16-avg',
                               'csi-4-max', 'csi-16-max', 'bias',
                                'sucr', 'pod', 'hss'], #['csi', 'bias', 'sucr', 'pod'],
                 dist_eval=False,
                #  device='cuda',
                 eps=1e-4,):
        self.layout = layout
        self.preprocess_type = preprocess_type
        self.threshold_list = threshold_list
        self.metrics_list = metrics_list
        self.eps = eps
        self.mode = mode
        self.seq_len = seq_len
        
        self.dist_eval = dist_eval
        # self.device = device
        
        if mode in ("0", ):
            self.keep_seq_len_dim = False
            state_shape = (len(self.threshold_list), )
        elif mode in ("1", "2"):
            self.keep_seq_len_dim = True
            assert isinstance(self.seq_len, int), "seq_len must be provided when we need to keep seq_len dim."
            state_shape = (len(self.threshold_list), self.seq_len)
        else:
            raise NotImplementedError(f"mode {mode} not supported!")
        
        self.hits = torch.zeros(state_shape)
        self.misses = torch.zeros(state_shape)
        self.fas = torch.zeros(state_shape)
        self.cor = torch.zeros(state_shape)

        ## pooling csi ##
        self.hits_avg_pool_4 = torch.zeros(state_shape)
        self.misses_avg_pool_4 = torch.zeros(state_shape)
        self.fas_avg_pool_4 = torch.zeros(state_shape)

        self.hits_max_pool_4 = torch.zeros(state_shape)
        self.misses_max_pool_4 = torch.zeros(state_shape)
        self.fas_max_pool_4 = torch.zeros(state_shape)

        self.hits_avg_pool_16 = torch.zeros(state_shape)
        self.misses_avg_pool_16 = torch.zeros(state_shape)
        self.fas_avg_pool_16 = torch.zeros(state_shape)

        self.hits_max_pool_16 = torch.zeros(state_shape)
        self.misses_max_pool_16 = torch.zeros(state_shape)
        self.fas_max_pool_16 = torch.zeros(state_shape)



    def pod(self, hits, misses, fas, eps):
        return hits / (hits + misses + eps)

    def sucr(self, hits, misses, fas, eps):
        return hits / (hits + fas + eps)

    def csi(self, hits, misses, fas, eps):
        return hits / (hits + misses + fas + eps)

    def bias(self, hits, misses, fas, eps):
        bias = (hits + fas) / (hits + misses + eps)
        logbias = torch.pow(bias / torch.log(torch.tensor(2.0)), 2.0)
        return logbias
    
    def hss(self, hits, misses, fas, cor, eps):
        hss = 2 * (hits * cor - misses * fas) / ((hits + misses) * (misses + cor) + (hits + fas) * (fas + cor) + eps)
        return hss


    def synchronize_between_processes(self):
        if not is_dist_avail_and_initialized():
            return
        dist.barrier()
        dist.all_reduce(self.hits)
        dist.all_reduce(self.misses)
        dist.all_reduce(self.fas)
        ### avg 4 ###
        dist.all_reduce(self.hits_avg_pool_4)
        dist.all_reduce(self.misses_avg_pool_4)
        dist.all_reduce(self.fas_avg_pool_4)
        ### max 4 ###
        dist.all_reduce(self.hits_max_pool_4)
        dist.all_reduce(self.misses_max_pool_4)
        dist.all_reduce(self.fas_max_pool_4)
        ### avg 16 ###
        dist.all_reduce(self.hits_avg_pool_16)
        dist.all_reduce(self.misses_avg_pool_16)
        dist.all_reduce(self.fas_avg_pool_16)
        ### max 16 ###
        dist.all_reduce(self.hits_max_pool_16)
        dist.all_reduce(self.misses_max_pool_16)
        dist.all_reduce(self.fas_max_pool_16)

    @property
    def hits_misses_fas_reduce_dims(self):
        if not hasattr(self, "_hits_misses_fas_reduce_dims"):
            seq_dim = self.layout.find('T')
            self._hits_misses_fas_reduce_dims = list(range(len(self.layout)))
            if self.keep_seq_len_dim:
                self._hits_misses_fas_reduce_dims.pop(seq_dim)
        return self._hits_misses_fas_reduce_dims

    def preprocess(self, pred, target):
        if self.preprocess_type == "sevir":
            pred = pred.detach() / (1. / 255.)
            target = target.detach() / (1. / 255.)
        elif self.preprocess_type == "meteonet":
            pred = pred.detach() / (1. / 70.)
            target = target.detach() / (1. / 70.)
        else:
            raise NotImplementedError
        return pred, target

    def preprocess_pool(self, pred, target, pool_size=4, type='avg'):
        if self.preprocess_type == "sevir":
            pred = pred.detach() / (1. / 255.)
            target = target.detach() / (1. / 255.)
        elif self.preprocess_type == "meteonet":
            pred = pred.detach() / (1. / 70.)
            target = target.detach() / (1. / 70.)
        b, t, _, _, _ = pred.shape
        pred = rearrange(pred, 'b t c h w -> (b t) c h w')
        target = rearrange(target, 'b t c h w -> (b t) c h w')
        if type == 'avg':
            pred = F.avg_pool2d(pred, kernel_size=pool_size, stride=pool_size)
            target = F.avg_pool2d(target, kernel_size=pool_size, stride=pool_size)
        elif type == 'max':
            pred = F.max_pool2d(pred, kernel_size=pool_size, stride=pool_size)
            target = F.max_pool2d(target, kernel_size=pool_size, stride=pool_size)
        pred = rearrange(pred, '(b t) c h w -> b t c h w', b=b)
        target = rearrange(target, '(b t) c h w -> b t c h w', b=b)
        return pred, target


    def calc_seq_hits_misses_fas(self, pred, target, threshold):
        with torch.no_grad():
            t, p = _threshold(target, pred, threshold)
            hits = torch.sum(t * p, dim=self.hits_misses_fas_reduce_dims).int()
            misses = torch.sum(t * (1 - p), dim=self.hits_misses_fas_reduce_dims).int()
            fas = torch.sum((1 - t) * p, dim=self.hits_misses_fas_reduce_dims).int()
            cor = torch.sum((1 - t) * (1 - p), dim=self.hits_misses_fas_reduce_dims).int()
        return hits, misses, fas, cor
    
    @torch.no_grad()
    def update(self, pred, target):
        ## pool 1 ##
        self.hits = self.hits.to(pred.device)
        self.misses = self.misses.to(pred.device)
        self.fas = self.fas.to(pred.device)
        self.cor = self.cor.to(pred.device)
        _pred, _target = self.preprocess(pred, target)
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            self.hits[i] += hits
            self.misses[i] += misses
            self.fas[i] += fas
            self.cor[i] += cor
        ## max pool 4 ##
        self.hits_max_pool_4 = self.hits_max_pool_4.to(pred.device)
        self.misses_max_pool_4 = self.misses_max_pool_4.to(pred.device)
        self.fas_max_pool_4 = self.fas_max_pool_4.to(pred.device)
        _pred, _target = self.preprocess_pool(pred, target, pool_size=4, type='max')
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            self.hits_max_pool_4[i] += hits
            self.misses_max_pool_4[i] += misses
            self.fas_max_pool_4[i] += fas 
        ## max pool 16 ##
        self.hits_max_pool_16 = self.hits_max_pool_16.to(pred.device)
        self.misses_max_pool_16 = self.misses_max_pool_16.to(pred.device)
        self.fas_max_pool_16 = self.fas_max_pool_16.to(pred.device)
        _pred, _target = self.preprocess_pool(pred, target, pool_size=16, type='max')
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            self.hits_max_pool_16[i] += hits
            self.misses_max_pool_16[i] += misses
            self.fas_max_pool_16[i] += fas 
        ## avg pool 4 ##
        self.hits_avg_pool_4 = self.hits_avg_pool_4.to(pred.device)
        self.misses_avg_pool_4 = self.misses_avg_pool_4.to(pred.device)
        self.fas_avg_pool_4 = self.fas_avg_pool_4.to(pred.device)
        _pred, _target = self.preprocess_pool(pred, target, pool_size=4, type='avg')
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            self.hits_avg_pool_4[i] += hits
            self.misses_avg_pool_4[i] += misses
            self.fas_avg_pool_4[i] += fas 
        ## avg pool 16 ##
        self.hits_avg_pool_16 = self.hits_avg_pool_16.to(pred.device)
        self.misses_avg_pool_16 = self.misses_avg_pool_16.to(pred.device)
        self.fas_avg_pool_16 = self.fas_avg_pool_16.to(pred.device)
        _pred, _target = self.preprocess_pool(pred, target, pool_size=16, type='avg')
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            self.hits_avg_pool_16[i] += hits
            self.misses_avg_pool_16[i] += misses
            self.fas_avg_pool_16[i] += fas 

    def _get_hits_misses_fas(self, metric_name):
        if metric_name.endswith('-4-avg'):
            hits = self.hits_avg_pool_4
            misses = self.misses_avg_pool_4
            fas = self.fas_avg_pool_4
        elif metric_name.endswith('-16-avg'):
            hits = self.hits_avg_pool_16
            misses = self.misses_avg_pool_16
            fas = self.fas_avg_pool_16
        elif metric_name.endswith('-4-max'):
            hits = self.hits_max_pool_4
            misses = self.misses_max_pool_4
            fas = self.fas_max_pool_4
        elif metric_name.endswith('-16-max'):
            hits = self.hits_max_pool_16
            misses = self.misses_max_pool_16
            fas = self.fas_max_pool_16
        else:
            hits = self.hits
            misses = self.misses
            fas = self.fas
        return [hits, misses, fas]
    
    def _get_correct_negtives(self):
        return self.cor
    
    @torch.no_grad()
    def compute(self):
        if self.dist_eval:
            self.synchronize_between_processes()
        
        metrics_dict = {'pod': self.pod,
                        'csi': self.csi,
                        'csi-4-avg': self.csi, 
                        'csi-16-avg': self.csi,
                        'csi-4-max': self.csi, 
                        'csi-16-max': self.csi,
                        'sucr': self.sucr,
                        'bias': self.bias,
                        'hss': self.hss}
        ret = {}
        for threshold in self.threshold_list:
            ret[threshold] = {}
        ret["avg"] = {}
        
        for metrics in self.metrics_list:
            if self.keep_seq_len_dim:
                score_avg = np.zeros((self.seq_len, ))
            else:
                score_avg = 0
            hits, misses, fas = self._get_hits_misses_fas(metrics)
            # scores = metrics_dict[metrics](self.hits, self.misses, self.fas, self.eps)
            if metrics != 'hss':
                scores = metrics_dict[metrics](hits, misses, fas, self.eps)
            else:
                cor = self._get_correct_negtives()
                scores = metrics_dict[metrics](hits, misses, fas, cor, self.eps)
            scores = scores.detach().cpu().numpy()
            for i, threshold in enumerate(self.threshold_list):
                if self.keep_seq_len_dim:
                    score = scores[i]  # shape = (seq_len, )
                else:
                    score = scores[i].item()  # shape = (1, )
                if self.mode in ("0", "1"):
                    ret[threshold][metrics] = score
                elif self.mode in ("2", ):
                    ret[threshold][metrics] = np.mean(score).item()
                else:
                    raise NotImplementedError
                score_avg += score
            score_avg /= len(self.threshold_list)
            if self.mode in ("0", "1"):
                ret["avg"][metrics] = score_avg
            elif self.mode in ("2",):
                ret["avg"][metrics] = np.mean(score_avg).item()
            else:
                raise NotImplementedError
        return ret

    @torch.no_grad()
    def get_single_frame_metrics(self, target, pred, metrics=['ssim', 'psnr', ]): #'cspr', 'cspr-4-avg', 'cspr-16-avg', 'cspr-4-max', 'cspr-16-max'
        metric_funcs = {
            'ssim': cal_SSIM,
            'psnr': cal_PSNR
        }
        metrics_dict = {}
        for metric in metrics:
            metric_fun = metric_funcs[metric]
            metrics_dict[metric] = metric_fun(gt=target*255., pred=pred*255., is_img=False)
        return metrics_dict
    
    @torch.no_grad()
    def get_crps(self, target, pred):
        """
        pred: (b, t, c, h, w)/(b, n, t, c, h, w)
        target: (b, t, c, h, w)
        """
        if len(pred.shape) == 5:
            pred = pred.unsqueeze(1)
        crps = cal_CRPS(gt=target, pred=pred, type='none')
        crps_avg_4 = cal_CRPS(gt=target, pred=pred, type='avg', scale=4)
        crps_avg_16 = cal_CRPS(gt=target, pred=pred, type='avg', scale=16)
        crps_max_4 = cal_CRPS(gt=target, pred=pred, type='max', scale=4)
        crps_max_16 = cal_CRPS(gt=target, pred=pred, type='max', scale=16)
        crps_dict = {
            'crps': crps,
            'crps_avg_4': crps_avg_4,
            'crps_avg_16': crps_avg_16,
            'crps_max_4': crps_max_4,
            'crps_max_16': crps_max_16
        }
        return crps_dict



    def reset(self):
        self.hits = self.hits*0
        self.misses = self.misses*0
        self.fas = self.fas*0

        self.hits_avg_pool_4 *= 0
        self.hits_avg_pool_16 *= 0
        self.hits_max_pool_4 *= 0
        self.hits_max_pool_16 *= 0

        self.misses_avg_pool_4 *= 0
        self.misses_avg_pool_16 *= 0
        self.misses_max_pool_4 *= 0
        self.misses_max_pool_16 *= 0
 
        self.fas_avg_pool_4 *= 0
        self.fas_avg_pool_16 *= 0
        self.fas_max_pool_4  *= 0
        self.fas_max_pool_16  *= 0


@torch.no_grad()
class cal_FVD:
    def __init__(self, use_gpu=False, resize_crop=False):
        '''
        iter_cal=True, gt.shape=pred.shape=[nb b t c h w]
        iter_cal=Fasle, gt.shape=pred.shape=[n t c h w]
        '''
        
        self.use_gpu = use_gpu
        self.resize_crop = resize_crop
        # detector_url = 'https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1'
        self.detector = torch.jit.load("/mnt/cache/gongjunchao/workdir/radar_forecasting/utils/fvd/i3d_torchscript.pt").eval()
        if torch.cuda.is_available() and self.use_gpu:
            self.detector = self.detector.cuda()
        self.feats = []
    
    def preprocess(self, video):
        """
        video: (b, t, c, h, w) in [0, 1]
        this function transform the domain to [-1, 1] 
        """
        video = video * 2 - 1
        return video

    @torch.no_grad()
    def __call__(self, videos_real, videos_fake):
        feats_fake = []
        feats_real = []
        detector_kwargs = dict(rescale=False, resize=False, return_features=True)
        
        videos_fake = einops.rearrange(
            self.bilinear_interpolation(videos_fake), 'n t c h w -> n c t h w'
        )
        videos_real = einops.rearrange(
            self.bilinear_interpolation(videos_real), 'n t c h w -> n c t h w'
        )
        if torch.cuda.is_available() and self.use_gpu:
            videos_fake, videos_real = videos_fake.cuda(), videos_real.cuda()
        # print(videos_fake.shape, videos_real.shape)
        # videos_fake = videos_fake.repeat(1, 1, 10, 1, 1)
        # videos_real = videos_real.repeat(1, 1, 10, 1, 1)
        feats_fake = self.detector(videos_fake, **detector_kwargs).cpu()
        feats_real = self.detector(videos_real, **detector_kwargs).cpu()
        self.feats.append(torch.stack([feats_fake, feats_real], dim=0))
        return
    
    def update(self, videos_real, videos_fake):
        self(videos_real=videos_real, videos_fake=videos_fake)
        return

    def _reset(self):
        self.feats = []

    def compute(self):
        feats = torch.cat(self.feats, dim=1)
        fake_feats = feats[0]
        real_feats = feats[1]
        fvd = self._cal_FVD(feats_fake=fake_feats, feats_real=real_feats)
        return fvd

    def bilinear_interpolation(self, image):
        N, T, C, H, W = image.shape
        def my_resize(img):
            img = img.view(-1, C, H, W)
            img = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
            img = img.view(N, T, C, 224, 224)  
            return img
        def my_resize_crop(img):
            img = img.view(-1, C, H, W)
            if H<W:
                img = F.interpolate(img, size=(224, int(W*224/H)), mode='bilinear', align_corners=False)
                img = img.view(N, T, C, 224, int(W*224/H))  
            else:   # W<=H
                img = F.interpolate(img, size=(int(H*224/W), 224), mode='bilinear', align_corners=False)
                img = img.view(N, T, C, int(H*224/W), 224)  
            return center_crop(img, (224, 224))
        if H == W and H < 224:
            return my_resize(img=image)
        elif self.resize_crop:
            return my_resize_crop(img=image)
        else: 
            return my_resize(img=image)

    def _cal_FVD(self, feats_fake, feats_real):
        def compute_fvd(feats_fake, feats_real):
            mu_gen, sigma_gen = compute_stats(feats_fake)
            mu_real, sigma_real = compute_stats(feats_real)
            m = np.square(mu_gen - mu_real).sum()
            s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False)
            fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
            return float(fid)

        def compute_stats(feats):
            feats = feats.reshape(-1, feats.shape[-1])
            mu = feats.mean(axis=0)
            sigma = np.cov(feats, rowvar=False)
            return mu, sigma
        return compute_fvd(feats_fake, feats_real)
    
if __name__ == "__main__":
    eval_metrics = SEVIRSkillScore(layout='NTCHW', seq_len=12, mode='1')

                #  layout: str = "NHWT",
                #  mode: str = "0",
                #  seq_len: Optional[int] = None,
                #  preprocess_type: str = "sevir",
                #  threshold_list: Sequence[int] = (16, 74, 133, 160, 181, 219),
                #  metrics_list: Sequence[str] = ("csi", "bias", "sucr", "pod"),
                #  eps: float = 1e-4,
                #  dist_sync_on_step: bool = False,
                #  ):
    data_dict= {}
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    import numpy as np
    ## b, t, c, h, w
    torch.manual_seed(0)
    data_dict['pred'] = torch.randn(3, 12, 1, 480, 480).to(device)
    data_dict['gt'] = torch.randn(3, 12, 1, 480, 480).to(device)
    ## sevir metrics compute ##
    eval_metrics.update(pred=data_dict["pred"], target=data_dict['gt'])
    losses = eval_metrics.compute()
    single_frame_dict = eval_metrics.get_single_frame_metrics(target=data_dict['gt'], pred=data_dict['pred'])
    crps_dict = eval_metrics.get_crps(target=data_dict['gt'], pred=data_dict['pred'])
    import pdb; pdb.set_trace()

    # fvd compute ##
    # fvd_computer = cal_FVD()
    # fvd_computer.update(data_dict['pred'].repeat(1, 1, 3, 1, 1), data_dict['gt'].repeat(1, 1, 3, 1, 1))
    # fvd_computer.update(data_dict['pred'].repeat(1, 1, 3, 1, 1), data_dict['gt'].repeat(1, 1, 3, 1, 1))
    # fvd = fvd_computer.compute()

    
    # _ = torch.manual_seed(123)
    # fvd_computer = cal_FVD()
    # # generate two slightly overlapping image intensity distributions
    # data_dict['gt'] = (torch.randint(0, 200, (100, 10, 3, 224, 224), dtype=torch.uint8) / 255.0 - 1) * 2
    # data_dict['pred'] = (torch.randint(100, 255, (100, 10, 3, 224, 224), dtype=torch.uint8) / 255.0 - 1) * 2
    # fvd_computer.update(data_dict['pred'], data_dict['gt'])
    # fvd = fvd_computer.compute()
    # import pdb; pdb.set_trace()

## srun -p ai4earth --kill-on-bad-exit=1 --quotatype=auto --gres=gpu:1 python -u metrics.py ##