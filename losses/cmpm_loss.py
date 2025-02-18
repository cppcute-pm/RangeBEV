import torch.nn as nn
import torch.nn.functional as F
import torch


class CmpmLoss(nn.Module):
    
    def __init__(self, cfgs):
        super(CmpmLoss, self).__init__()
        self.epsilon = cfgs.epsilon
        self.SDM_logit_scale = cfgs.SDM_logit_scale
        self.avg_sim_info = cfgs.avg_sim_info

    def forward(self, embeddings1, embeddings2, positives_mask):
        """
        Cross-Modal Projection Matching Loss(CMPM)
        :param embeddings1: Tensor with dtype torch.float32
        :param embeddings2: Tensor with dtype torch.float32
        :param labels: Tensor with dtype torch.int32
        :return:
            i2t_loss: cmpm loss for image projected to text
            t2i_loss: cmpm loss for text projected to image
            pos_avg_sim: average cosine-similarity for positive pairs
            neg_avg_sim: averate cosine-similarity for negative pairs
        """

        diag = torch.diag(positives_mask)
        assert torch.all(diag), f'positives_mask diagonal is not all 1'
        labels_mat_norm = F.normalize(positives_mask.type(torch.float64), dim=1, p=2.0)

        image_norm = F.normalize(embeddings1, dim=1, p=2.0)
        text_norm = F.normalize(embeddings2, dim=1, p=2.0)
        image_proj_text = torch.matmul(embeddings1, text_norm.t()) * self.SDM_logit_scale # (B, B)
        text_proj_image = torch.matmul(embeddings2, image_norm.t()) * self.SDM_logit_scale # (B, B)

        i2t_pred = F.softmax(image_proj_text, dim=1)
        # i2t_loss = i2t_pred * torch.log((i2t_pred + self.epsilon)/ (labels_mask_norm + self.epsilon))
        i2t_loss = i2t_pred * (
            F.log_softmax(image_proj_text, dim=1)
            - torch.log(labels_mat_norm + self.epsilon)
        )

        t2i_pred = F.softmax(text_proj_image, dim=1)
        # t2i_loss = t2i_pred * torch.log((t2i_pred + self.epsilon)/ (labels_mask_norm + self.epsilon))
        t2i_loss = t2i_pred * (
            F.log_softmax(text_proj_image, dim=1)
            - torch.log(labels_mat_norm + self.epsilon)
        )

        cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(
            torch.sum(t2i_loss, dim=1)
        )

        stat = {}
        if self.avg_sim_info:
            sim_cos = torch.matmul(image_norm, text_norm.t())
            stat['pos_avg_sim'] = torch.mean(torch.masked_select(sim_cos, positives_mask > 0))
            stat['neg_avg_sim'] = torch.mean(torch.masked_select(sim_cos, positives_mask == 0))
   
        return cmpm_loss, stat