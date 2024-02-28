import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class PixelWiseRegularizer(nn.Module):
    """
    Weighted loss function.
    Args:
        cfg: config format tranformed by config.py
        magnitude: int, weight of the summed loss value
        min_value: int, lower boundary of input tensor
        max_value: int, highr boundary of output tensor
    Return:
        torch.tensor
    """
    def __init__(
        self,
        cfg=None, 
        magnitude: int=20,
        min_value: int=-9,
        max_value: int=60
        ):
        super().__init__()
        if cfg is not None:
            self.magnitude = cfg.LAMBDA
            self.min_value = cfg.MIN_VALUE
            self.max_value = cfg.MAX_VALUE
        else:
            self.magnitude = magnitude
            self.min_value = min_value
            self.max_value = max_value
    
    def forward(self, pred, target):
        weight = torch.clamp(target, min=self.min_value, max=self.max_value)
        difference = torch.abs(pred - target)
        loss = difference * weight
        return self.magnitude * loss.mean()

def HingeLoss(pred, sign, margin=1.):
    loss = F.relu(margin - sign * pred)
    return loss.mean()

def HingeLossG(pred):
    return -torch.mean(pred)

if __name__ == "__main__":
    
    pass