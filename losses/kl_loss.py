from info_nce import info_nce, normalize, transpose
import torch.nn as nn
import torch
import torch.nn.functional as F

class KLLoss(nn.Module):
    def __init__(self, cfgs):
        super(KLLoss, self).__init__()
        self.temperature = cfgs.temperature

    def forward(self, 
                embeddings1, 
                embeddings2, 
                positive_key1, 
                positive_key2, 
                negative_keys1, 
                negative_keys2, 
                positives_mask, 
                negatives_mask):
        B, DB = negatives_mask.shape
        device = embeddings1.device
        negative_inuse = negatives_mask.type(torch.LongTensor) # (B, DB)
        negative_count = torch.count_nonzero(negative_inuse, dim=-1).to(device) # (B)
        negative_zero_mask = torch.le(negative_count, 0)
        negative_count.masked_fill_(negative_zero_mask, DB)
        rand_base = torch.randint(0, 1000000, (B, DB), device=device)
        rand_base_1 = (torch.fmod(rand_base, negative_count.unsqueeze(-1))).type(torch.int64)
        rand_base_2 = torch.arange(0, DB, device=device).unsqueeze(0).expand(B, -1)
        rand_base_2 = rand_base_2.where(torch.lt(rand_base_2, negative_count.unsqueeze(-1)), rand_base_1)
        _, indices = torch.sort(negative_inuse, dim=-1, descending=True)
        rand_indices = torch.gather(indices.to(device), 1, rand_base_2) # (B, DB)

        negatives_keys1 = negative_keys1[rand_indices, :] # (B, DB, D)
        positive_logit1_1, negative_logit1_1 = logit_caculation(query=embeddings1, 
                                            positive_key=positive_key1, 
                                            negative_keys=negatives_keys1)
        positive_logit1_1 = positive_logit1_1 / self.temperature # (B,)
        negative_logit1_1 = negative_logit1_1 / self.temperature # (B, DB)
        logit1_1 = torch.cat([positive_logit1_1.unsqueeze(1), negative_logit1_1], dim=1) # (B, DB+1)
        logit1_1_softmax = F.softmax(logit1_1, dim=-1)
        logit1_1_softmax_log = -torch.log(torch.clamp(logit1_1_softmax, min=1e-6, max=1-1e-6))
        prob1_1 = logit1_1_softmax_log[:, 0]

        positive_logit1_2, negative_logit1_2 = logit_caculation(query=positive_key1, 
                                            positive_key=positive_key1, 
                                            negative_keys=negatives_keys1)
        positive_logit1_2 = positive_logit1_2 / self.temperature
        negative_logit1_2 = negative_logit1_2 / self.temperature
        logit1_2 = torch.cat([positive_logit1_2.unsqueeze(1), negative_logit1_2], dim=1)
        logit1_2_softmax = F.softmax(logit1_2, dim=-1)
        logit1_2_softmax_log = -torch.log(torch.clamp(logit1_2_softmax, min=1e-6, max=1-1e-6))
        prob1_2 = logit1_2_softmax_log[:, 0]

        P1 = prob1_1 / prob1_1.sum(dim=0, keepdim=True) # (B,)
        Q1 = prob1_2 / prob1_2.sum(dim=0, keepdim=True) # (B,)

        log_P1 = torch.log(P1 + 1e-6) # (B,)
        log_Q1 = torch.log(Q1 + 1e-6) # (B,)
        # KL(Q1 || P1)
        kl_div1 = torch.sum(Q1 * (log_Q1 - log_P1)) 

        negative_inuse = negatives_mask.type(torch.LongTensor) # (B, DB)
        negative_count = torch.count_nonzero(negative_inuse, dim=-1).to(device) # (B)
        negative_zero_mask = torch.le(negative_count, 0)
        negative_count.masked_fill_(negative_zero_mask, DB)
        rand_base = torch.randint(0, 1000000, (B, DB), device=device)
        rand_base_1 = (torch.fmod(rand_base, negative_count.unsqueeze(-1))).type(torch.int64)
        rand_base_2 = torch.arange(0, DB, device=device).unsqueeze(0).expand(B, -1)
        rand_base_2 = rand_base_2.where(torch.lt(rand_base_2, negative_count.unsqueeze(-1)), rand_base_1)
        _, indices = torch.sort(negative_inuse, dim=-1, descending=True)
        rand_indices = torch.gather(indices.to(device), 1, rand_base_2) # (B, DB)

        negatives_keys2 = negative_keys2[rand_indices, :] # (B, DB, D)
        positive_logit2_1, negative_logit2_1 = logit_caculation(query=embeddings2, 
                                            positive_key=positive_key2, 
                                            negative_keys=negatives_keys2)
        positive_logit2_1 = positive_logit2_1 / self.temperature
        negative_logit2_1 = negative_logit2_1 / self.temperature
        logit2_1 = torch.cat([positive_logit2_1.unsqueeze(1), negative_logit2_1], dim=1) # (B, DB+1)
        logit2_1_softmax = F.softmax(logit2_1, dim=-1)
        logit2_1_softmax_log = -torch.log(torch.clamp(logit2_1_softmax, min=1e-6, max=1-1e-6))
        prob2_1 = logit2_1_softmax_log[:, 0] # (B,)

        positive_logit2_2, negative_logit2_2 = logit_caculation(query=positive_key2, 
                                            positive_key=positive_key2, 
                                            negative_keys=negatives_keys2)
        positive_logit2_2 = positive_logit2_2 / self.temperature
        negative_logit2_2 = negative_logit2_2 / self.temperature
        logit2_2 = torch.cat([positive_logit2_2.unsqueeze(1), negative_logit2_2], dim=1) # (B, DB+1)
        logit2_2_softmax = F.softmax(logit2_2, dim=-1)
        logit2_2_softmax_log = -torch.log(torch.clamp(logit2_2_softmax, min=1e-6, max=1-1e-6))
        prob2_2 = logit2_2_softmax_log[:, 0] # (B,)

        P2 = prob2_1 / prob2_1.sum(dim=0, keepdim=True)
        Q2 = prob2_2 / prob2_2.sum(dim=0, keepdim=True)

        log_P2 = torch.log(P2 + 1e-6)
        log_Q2 = torch.log(Q2 + 1e-6)

        # KL(Q2 || P2)
        kl_div2 = torch.sum(Q2 * (log_Q2 - log_P2))

        return 0.5 * (kl_div1 + kl_div2)

def logit_caculation(query, positive_key, negative_keys, temperature=0.1):

    # Normalize to unit vectors
    query = F.normalize(query, dim=-1)
    positive_key = F.normalize(positive_key, dim=-1)
    negative_keys = F.normalize(negative_keys, dim=-1)

    # Cosine between positive pairs
    positive_logit = torch.sum(query * positive_key, dim=1, keepdim=False) # (q_num,)

    query = query.unsqueeze(1)
    negative_logits = query @ transpose(negative_keys) # (q_num, 1, n_num)
    negative_logits = negative_logits.squeeze(1) # (q_num, n_num)

    return positive_logit, negative_logits