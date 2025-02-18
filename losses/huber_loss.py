import numpy as np
import torch
from pytorch_metric_learning import losses, miners, reducers
from pytorch_metric_learning.distances import LpDistance
import torch.nn as nn
import torch.nn.functional as F
    
class HuberLoss(nn.Module):

    def __init__(self, cfgs) -> None:
        # got base margin and change the margin according to the overlap_ratio difference
        super(HuberLoss, self).__init__()
        self.beta = cfgs.beta
        self.reduction = cfgs.reduction # just 'mean'
        self.smoothl1loss = nn.SmoothL1Loss(reduction=self.reduction, beta=self.beta)
        self.lamda = cfgs.lamda
    
    def forward(self, feat_dist_matrix, positive_masks, true_dist_matrix):
        # calculate the distance between embeddings1 and embeddings2
        feat_dist_positives = torch.masked_select(feat_dist_matrix, positive_masks)
        true_dist_positives = torch.masked_select(true_dist_matrix, positive_masks)
        feat_dist_positives_lamda = feat_dist_positives * self.lamda
        smoothl1loss = self.smoothl1loss(feat_dist_positives_lamda, true_dist_positives)
        huberloss = smoothl1loss * self.beta
        
        return huberloss