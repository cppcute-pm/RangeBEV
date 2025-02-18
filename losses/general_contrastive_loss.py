import torch


class GeneralContrastiveLoss(torch.nn.Module):

    def __init__(self, cfgs):
        super(GeneralContrastiveLoss, self).__init__()
        self.margin = cfgs.margin # 0.5

    def forward(self, embeddings1, embeddings2, overlap_ratio):
        gt = overlap_ratio
        D = torch.cdist(embeddings1, embeddings2, 2.0)
        loss = gt * 0.5 * torch.pow(D, 2) + (1 - gt) * 0.5 * torch.pow(torch.clamp(self.margin - D, min=0.0), 2)
        return torch.mean(loss)