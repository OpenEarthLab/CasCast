import math
import torch
import numpy as np
import properscoring as ps
import torch.nn.functional as F
from scipy.stats import norm

class Evaluator:
    def __init__(
        self,
        thresholds: list=None,
        pooling_scales: list=None,
        parser=None
        ):

        self.thresholds = thresholds
        self.pooling_scales = pooling_scales
        self.parser = parser
        
        self.init_metrics()
    
    def init_metrics(self):
        self.CSI = dict.fromkeys(self.thresholds, None)
        self.POD = dict.fromkeys(self.thresholds, None)
        self.FAR = dict.fromkeys(self.thresholds, None)
        self.CRPS_avg = dict.fromkeys(self.pooling_scales, None)
        self.CRPS_max = dict.fromkeys(self.pooling_scales, None)
        self.MSE = []
        self.MAE = []
        self.RMSE = []
    
    def cal_TP(self, pred=None, target=None, th=None):
        return torch.where(torch.logical_and(pred>=th, target>=th), 1, 0).sum(dim=(-1, -2))

    def cal_TN(self, pred=None, target=None, th=None):
        return torch.where(torch.logical_and(pred<th, target<th), 1, 0).sum(dim=(-1, -2))

    def cal_FP(self, pred=None, target=None, th=None):
        return torch.where(torch.logical_and(pred>=th, target<th), 1, 0).sum(dim=(-1, -2))

    def cal_FN(self, pred=None, target=None, th=None):
        return torch.where(torch.logical_and(pred<th, target>=th), 1, 0).sum(dim=(-1, -2))

    def cal_CSI(self, pred=None, target=None, th=None, TP=None, FP=None, FN=None):
        """
        Critical Success Index = TP / (TP + FP + FN)
        """
        if TP is None and FP is None and FN is None:
            TP = self.cal_TP(pred=pred, target=target, th=th)
            FP = self.cal_FP(pred=pred, target=target, th=th)
            FN = self.cal_FN(pred=pred, target=target, th=th)
        return (TP / (TP + FP + FN)).mean(dim=0)

    def cal_POD(self, pred=None, target=None, th=None, TP=None, FN=None):
        """
        Probability of Detection = TP / (TP + FN)
        """
        if TP is None and FN is None:
            TP = self.cal_TP(pred=pred, target=target, th=th)
            FN = self.cal_FN(pred=pred, target=target, th=th)
        return (TP / (TP + FN)).mean(dim=0)

    def cal_FAR(self, pred=None, target=None, th=None, FP=None, TP=None):
        """
        False Alarm Rate = FP / (FP + TP)
        """
        if FP is None and TP is None:
            TP = self.cal_TP(pred=pred, target=target, th=th)
            FP = self.cal_FP(pred=pred, target=target, th=th)
        return (FP / (FP + TP)).mean(dim=0)

    def cal_CRPS(self, pred=None, target=None):
        """
        Continuous Ranked Probability Score
        This function uses the library written by researchers at The Climate Corporation.
        The original authors include Leon Barrett, Stephan Hoyer, Alex Kleeman and Drew O'Kane.
        """

        target_cdf = norm.cdf(x=target.detach().cpu().numpy(), loc=0, scale=1)
        pred_cdf = norm.cdf(x=pred.detach().cpu().numpy(), loc=0, scale=1)
        forecast_score = ps.crps_ensemble(target_cdf, pred_cdf).mean(axis=(0, -1, -2))
        return forecast_score

    def cal_MSE(self, pred=None, target=None):
        """
        Mean Squared Error = E[(pred - target) ** 2]
        """
        return torch.mean(((pred - target) ** 2), dim=(0, -1, -2))

    def cal_MAE(self, pred=None, target=None):
        """
        Mean Absolute Error = E[|(pred - target)|]
        """
        return torch.mean(torch.abs((pred - target)), dim=(0, -1, -2))

    def cal_RMSE(self, pred=None, target=None):
        """
        Root Mean-Square Error = { E[(pred - target) ** 2] } ** 0.5
        """
        return torch.sqrt(self.cal_MSE(pred, target))

    def calculate_all(self, pred, target):
        """
        Evaluate prediction in different thresholds

        pred: torch.array or torch.tensor
        target: torch.array or torch.tensor
        ranges: list of numbers
        """
        if torch.is_tensor(pred):
            pred = pred.detach()#.cpu().numpy()
        if torch.is_tensor(target):
            target = target.detach()#.cpu().numpy()

        check_shape = lambda x: x.unsqueeze(0) if len(x.shape) == 3 else x
        pred = check_shape(pred)
        target = check_shape(target)

        if isinstance(pred, torch.Tensor):
            pred = torch.nan_to_num(pred, nan=0)
        if isinstance(target, torch.Tensor):
            target = torch.nan_to_num(target, nan=0)

        if self.parser is not None:
            pred = self.parser(pred)
            target = self.parser(target)

        for scale in self.pooling_scales:
            self.CRPS_avg[scale] = self.cal_CRPS(
                pred=F.avg_pool2d(pred, kernel_size=scale),
                target=F.avg_pool2d(target, kernel_size=scale))
            self.CRPS_max[scale] = self.cal_CRPS(
                pred=F.max_pool2d(pred, kernel_size=scale),
                target=F.max_pool2d(target, kernel_size=scale))

        for th in self.thresholds:
            TP = self.cal_TP(pred=pred, target=target, th=th)
            TN = self.cal_TN(pred=pred, target=target, th=th)
            FP = self.cal_FP(pred=pred, target=target, th=th)
            FN = self.cal_FN(pred=pred, target=target, th=th)

            self.CSI[th] = self.cal_CSI(TP=TP, FP=FP, FN=FN)
            self.POD[th] = self.cal_POD(TP=TP, FN=FN)
            self.FAR[th] = self.cal_FAR(TP=TP, FP=FP)

        self.MSE = self.cal_MSE(pred=pred, target=target)
        self.MAE = self.cal_MAE(pred=pred, target=target)
        self.RMSE = self.cal_RMSE(pred=pred, target=target)

if __name__ == "__main__":
    evaluation = Evaluator([0.3, 0.5, 0.7], [1, 4, 16], norm=False)
    x = torch.randn(16, 6, 256, 256)
    y = torch.randn(16, 6, 256, 256)
    evaluation.calculate_all(x, y)
    print(evaluation.CRPS_avg)
