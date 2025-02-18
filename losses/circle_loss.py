import torch
import torch.nn as nn
import torch.nn.functional as F


class CircleLoss:
    """
    Circle loss for positive and negative matrix.

    Args:
    m:  The relaxation factor that controls the radious of the decision boundary.
    gamma: The scale factor that determines the largest scale of each similarity score.

    According to the paper, the suggested default values of m and gamma are:

    Face Recognition: m = 0.25, gamma = 256
    Person Reidentification: m = 0.25, gamma = 128
    Fine-grained Image Retrieval: m = 0.4, gamma = 80

    By default, we set m = 0.4 and gamma = 80
    """

    def __init__(self, cfgs):

        self.m = cfgs.m
        self.gamma = cfgs.gamma
        self.soft_plus = torch.nn.Softplus(beta=1)
        self.op = 1 + self.m
        self.on = -self.m
        self.delta_p = 1 - self.m
        self.delta_n = self.m

    def __call__(self, embeddings1, embeddings2, pos_mask, neg_mask):
        embeddings1 = F.normalize(embeddings1, p=2, dim=1)
        embeddings2 = F.normalize(embeddings2, p=2, dim=1)
        sim_mat = torch.matmul(embeddings1, embeddings2.transpose(0, 1))
        pos_mat = torch.zeros_like(sim_mat)
        neg_mat = torch.zeros_like(sim_mat)

        pos_mat[pos_mask] = (
            -self.gamma
            * torch.relu(self.op - sim_mat[pos_mask].detach())
            * (sim_mat[pos_mask] - self.delta_p)
        )
        neg_mat[neg_mask] = (
            self.gamma
            * torch.relu(sim_mat[neg_mask].detach() - self.on)
            * (sim_mat[neg_mask] - self.delta_n)
        )

        neg_mat = torch.logsumexp(neg_mat, 1, keepdim=True)
        pos_mat = pos_mat + neg_mat

        a = torch.sum(neg_mask, dim=1)
        zero_rows = torch.eq(a, 0)
        length0 = torch.count_nonzero(pos_mask, dim=1) # (N,)
        length = torch.sum(length0) # (1,)
        length0[~zero_rows] = 0
        pos_mat[zero_rows] = 0

        length = length - torch.sum(length0)
        length = torch.clamp(length, min=1.0)
        loss = torch.sum(F.softplus(pos_mat[pos_mask])) / length

        return loss

# class CircleLoss_v2:
#     def __init__(self, m: float, gamma: float) -> None:
#         super(CircleLoss, self).__init__()
#         self.m = m
#         self.gamma = gamma
#         self.soft_plus = nn.Softplus()

#     def forward(self, embeddings1, embeddings2, pos_mask, neg_mask):
#         sim_mat = torch.matmul(embeddings1, embeddings2.transpose(0, 1), p=2.0)
#         sp = sim_mat[pos_mask]
#         sn = sim_mat[neg_mask]
#         ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
#         an = torch.clamp_min(sn.detach() + self.m, min=0.)

#         delta_p = 1 - self.m
#         delta_n = self.m

#         logit_p = - ap * (sp - delta_p) * self.gamma
#         logit_n = an * (sn - delta_n) * self.gamma

#         loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

#         return loss


class CircleLoss_v2:
    """
    Circle loss for positive and negative matrix.

    Args:
    m:  The relaxation factor that controls the radious of the decision boundary.
    gamma: The scale factor that determines the largest scale of each similarity score.

    According to the paper, the suggested default values of m and gamma are:

    Face Recognition: m = 0.25, gamma = 256
    Person Reidentification: m = 0.25, gamma = 128
    Fine-grained Image Retrieval: m = 0.4, gamma = 80

    By default, we set m = 0.4 and gamma = 80
    """

    def __init__(self, cfgs):

        self.m = cfgs.m
        self.gamma = cfgs.gamma
        self.soft_plus = torch.nn.Softplus(beta=1)
        self.op = 1 + self.m
        self.on = -self.m
        self.delta_p = 1 - self.m
        self.delta_n = self.m

    def __call__(self, embeddings1, embeddings2, pos_mask, neg_mask):
        N, M1, M2 = pos_mask.shape
        device = embeddings1.device
        embeddings1 = F.normalize(embeddings1, p=2, dim=-1) # (N, M1, D)
        embeddings2 = F.normalize(embeddings2, p=2, dim=-1) # (N, M2, D)
        sim_mat = torch.matmul(embeddings1, embeddings2.permute(0, 2, 1)) # (N, M1, M2)
        pos_mat = torch.zeros_like(sim_mat) # (N, M1, M2)
        neg_mat = torch.zeros_like(sim_mat) # (N, M1, M2)

        pos_mat[pos_mask] = (
            -self.gamma
            * torch.relu(self.op - sim_mat[pos_mask].detach())
            * (sim_mat[pos_mask] - self.delta_p)
        )
        neg_mat[neg_mask] = (
            self.gamma
            * torch.relu(sim_mat[neg_mask].detach() - self.on)
            * (sim_mat[neg_mask] - self.delta_n)
        )

        neg_mat = torch.logsumexp(neg_mat, -1, keepdim=True) # (N, M1, 1)
        pos_mat = pos_mat + neg_mat # (N, M1, M2)

        a = torch.sum(neg_mask, dim=-1) # (N, M1)
        zero_rows = torch.eq(a, 0) # (N, M1)
        length0 = torch.count_nonzero(pos_mask, dim=-1) # (N, M1)
        length = torch.sum(length0, dim=-1) # (N,)
        length0[~zero_rows] = 0 # (N, M1)
        pos_mat[zero_rows.unsqueeze(-1).expand(-1, -1, M2)] = 0 # (N, M1, M2)

        length = length - torch.sum(length0, dim=-1) # (N,)
        length = torch.clamp(length, min=1.0) # (N,)
        temp = F.softplus(pos_mat[pos_mask])
        new_pos_mat = torch.zeros(pos_mat.shape, device=device, dtype=temp.dtype)
        new_pos_mat[pos_mask] = temp
        loss_temp1 = torch.sum(new_pos_mat, dim=-1) # (N, M1)
        loss_temp2 = torch.sum(loss_temp1, dim=-1) # (N,)
        loss_temp3 = loss_temp2 / length # (N,)
        length1 = torch.count_nonzero(length) # (1,) 
        length1 = torch.clamp(length1, min=1.0) # (1,)
        loss = torch.sum(loss_temp3) / length1 # (1,)

        return loss