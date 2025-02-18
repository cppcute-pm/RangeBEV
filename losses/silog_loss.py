import torch
from torch import nn


class SiLogLoss(nn.Module):
    def __init__(self, cfgs):
        super(SiLogLoss, self).__init__()
        self.lambd = cfgs.lambd
        self.min_depth = cfgs.min_depth
        self.max_depth = cfgs.max_depth

    def forward(self, pred, target):
        valid_mask = (target >= self.min_depth) & (target <= self.max_depth)
        valid_mask = valid_mask.detach()
        pred = torch.clamp(pred, min=self.min_depth, max=self.max_depth)
        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                          self.lambd * torch.pow(diff_log.mean(), 2))

        return loss