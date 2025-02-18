import torch.nn as nn
import torch
import torch.nn.functional as F


def sinkhorn_knopp(matrix, epsilon=1e-6, max_iterations=1000):
    # 确保输入矩阵是浮点数类型
    matrix = matrix.float()

    # 获取矩阵的形状
    rows, cols = matrix.shape
    
    # 设置初始的行和列缩放向量
    r = torch.ones(rows, dtype=torch.float32)
    c = torch.ones(cols, dtype=torch.float32)
    
    # 迭代Sinkhorn-Knopp算法
    for _ in range(max_iterations):
        # 记录当前矩阵以便后续检查收敛性
        prev_matrix = matrix.clone()
        
        # 行归一化
        row_sums = torch.sum(matrix, dim=1, keepdim=True)
        row_sums[row_sums == 0] = 1  # 防止除以零
        matrix = matrix / row_sums
        
        # 列归一化
        col_sums = torch.sum(matrix, dim=0, keepdim=True)
        col_sums[col_sums == 0] = 1  # 防止除以零
        matrix = matrix / col_sums
        
        # 检查收敛性
        if torch.allclose(matrix, prev_matrix, atol=epsilon):
            break
    
    return matrix

def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    '''
    Perform Sinkhorn Normalization in Log-space for stability
    :param Z:
    :param log_mu:
    :param log_nu:
    :param iters:
    :return:
    '''
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)

    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, bins0=None, bins1=None, alpha=None, iters=100):
    """
    Perform Differentiable Optimal Transport in Log-space for stability
    :param scores:
    :param alpha:
    :param iters:
    :return:
    """

    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m * one).to(scores), (n * one).to(scores)

    if bins0 is None:
        bins0 = alpha.expand(b, m, 1)
    if bins1 is None:
        bins1 = alpha.expand(b, 1, n)
    
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm #multiply probabilities by M + N
    return Z

# class CFI2P_loss(nn.Module):

#     def __init__(self, cfgs):
#         super(CFI2P_loss, self).__init__()
#         self.bin_score = nn.Parameter(torch.tensor(1.))
#         self.sinkhorn_iters = cfgs.sinkhorn_iters

#     def optimal_transport(self, scores):
#         scores = scores.unsqueeze(0)
#         scores = log_optimal_transport(scores, None, None, self.bin_score, iters=self.sinkhorn_iters)
#         scores = scores.squeeze(0)
#         return scores
    
    # def forward(self, embeddings1, embeddings2, overlap_ratio_matrix):

    #     # because we select the correspondence among the batch, so need to normalize the overlap ratio matrix
    #     overlap_ratio_matrix = sinkhorn_knopp(overlap_ratio_matrix)
    #     # construct the ground truth transport matrix
    #     row_overlap_ratio = 1.0 - torch.sum(overlap_ratio_matrix, dim=0, keepdim=True)
    #     col_overlap_ratio = 1.0 - torch.sum(overlap_ratio_matrix, dim=1, keepdim=True)

    #     overlap_ratio_matrix = torch.cat([overlap_ratio_matrix, row_overlap_ratio], dim=0)
    #     zeros_0_0 = torch.tensor([[0.0]]).cuda()
    #     col_overlap_ratio_plus = torch.cat([col_overlap_ratio, zeros_0_0], dim=0)
    #     scores_gt = torch.cat([overlap_ratio_matrix, col_overlap_ratio_plus], dim=1)

    #     # caculate the scores matrix
    #     # embeddings1 = F.normalize(embeddings1, p=2.0, dim=-1)
    #     # embeddings2 = F.normalize(embeddings2, p=2.0, dim=-1)
    #     ebd_dim = embeddings1.shape[-1]
    #     scores = torch.matmul(embeddings1, embeddings2.t())
    #     scores = scores / (ebd_dim ** 0.5)  # like the LoFTR paper and CFI2P paper
    #     scores = self.optimal_transport(scores)

    #     # caculate the matching loss
    #     y = scores_gt * scores
    #     loss = torch.sum(-y) / torch.sum(scores_gt)
    #     return loss
class CFI2P_loss(nn.Module):

    def __init__(self, cfgs):
        super(CFI2P_loss, self).__init__()
        self.temperature = cfgs.temperature
        self.type = cfgs.type

    def forward(self, embeddings1, embeddings2, overlap_ratio_matrix):

        # construct the ground truth transport matrix
        overlap_ratio_matrix = F.softmax(overlap_ratio_matrix, 0) * F.softmax(overlap_ratio_matrix, 1)
        scores_gt = overlap_ratio_matrix

        # caculate the scores matrix
        if self.type == 'normalize':
            embeddings1 = F.normalize(embeddings1, p=2.0, dim=-1)
            embeddings2 = F.normalize(embeddings2, p=2.0, dim=-1)
            scores = torch.matmul(embeddings1, embeddings2.t())
            scores = F.softmax(scores, 0) * F.softmax(scores, 1)
        elif self.type == 'temperature':
            scores = torch.matmul(embeddings1, embeddings2.t()) / self.temperature
            scores = F.softmax(scores, 0) * F.softmax(scores, 1)
        elif self.type == 'mixed':
            embeddings1 = F.normalize(embeddings1, p=2.0, dim=-1)
            embeddings2 = F.normalize(embeddings2, p=2.0, dim=-1)
            scores = torch.matmul(embeddings1, embeddings2.t()) / self.temperature
            scores = F.softmax(scores, 0) * F.softmax(scores, 1)

        # caculate the matching loss
        y = scores_gt * torch.log(scores)
        loss = torch.sum(-y) / torch.sum(scores_gt)
        return loss
