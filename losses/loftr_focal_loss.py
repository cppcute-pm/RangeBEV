import torch
from torch import nn
import torch.nn.functional as F


class LoftrFocalLoss(nn.Module):
    def __init__(self, cfgs):
        super(LoftrFocalLoss, self).__init__()
        self.alpha = cfgs.alpha
        self.gamma = cfgs.gamma
        self.use_negative = cfgs.use_negative

    def forward(self, embeddings1, embeddings2):
        N = embeddings1.shape[0]
        embeddings1 = F.normalize(embeddings1, p=2, dim=-1) # (N, D)
        embeddings2 = F.normalize(embeddings2, p=2, dim=-1) # (N, D)
        conf_pos = torch.mul(embeddings1, embeddings2).sum(dim=-1, keepdim=False) # (N,)
        conf_pos = torch.clamp(conf_pos, min=1e-6, max=1 - 1e-6)
        loss_pos = -self.alpha * torch.pow(1 - conf_pos, self.gamma) * (conf_pos).log()
        if not self.use_negative:
            return loss_pos.mean()
        else:
            eye_mat = torch.eye(N, dtype=torch.bool, device=embeddings1.device) # (N, N)
            conf_all = torch.matmul(embeddings1, embeddings2.permute(1, 0)) # (N, N)
            conf_neg = conf_all[~eye_mat] # (N*(N-1))
            conf_neg = torch.clamp(conf_neg, min=1e-6, max=1 - 1e-6)
            loss_neg = -self.alpha * torch.pow(conf_neg, self.gamma) * (1 - conf_neg).log()
            return loss_pos.mean() + loss_neg.mean()
  