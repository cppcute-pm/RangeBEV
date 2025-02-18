from info_nce import info_nce, normalize, transpose
import torch.nn as nn
import torch
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    def __init__(self, cfgs):
        super(InfoNCELoss, self).__init__()
        self.temperature = cfgs.temperature
        self.reduction = cfgs.reduction
        self.negative_mode = cfgs.negative_mode # 1: no negative filter, 2: none positives_mask as negative, 3: negatives_mask as negative
        self.positive_mode = cfgs.positive_mode # 1: itself, 2: random positive_sample except itself, 3: random positive_sample including itself
        self.distance_mode = cfgs.distance_mode # 1: p2 norm, 2: cosine similarity
        assert self.distance_mode == 2, 'do you transform the L2 distance to any kind of similarity?'

    def forward(self, embeddings1, embeddings2, positives_mask, negatives_mask):
        B, _ = embeddings1.shape
        device = embeddings1.device
        if self.positive_mode == 1:
            positive_key=embeddings2
        else:
            if self.positive_mode == 2:
                real_positives_mask = torch.logical_and(~torch.eye(n=B, dtype=torch.long, device=device), positives_mask)
            elif self.positive_mode == 3:
                real_positives_mask = positives_mask.type(torch.LongTensor)
            positive_count = torch.count_nonzero(real_positives_mask, dim=-1).to(device) # (B)
            rand_base = torch.randint(0, B, (B,), device=device)
            rand_base_1 = (torch.fmod(rand_base, positive_count)).type(torch.int64)
            _, indices = torch.sort(real_positives_mask.type(torch.LongTensor), dim=-1, descending=True)
            batch_indices = torch.arange(0, B, device=device)
            rand_indices = indices[batch_indices, rand_base_1] # (B)
            positive_key = embeddings2[rand_indices, :] # (B, D)
        if self.negative_mode == 1:
            if self.distance_mode == 1:
                return info_nce_v1dot5(query=embeddings1, 
                            positive_key=positive_key, 
                            negative_keys=None, 
                            temperature=self.temperature, 
                            reduction=self.reduction, 
                            negative_mode=None)
            else:
                return info_nce(query=embeddings1, 
                                positive_key=positive_key, 
                                negative_keys=None, 
                                temperature=self.temperature, 
                                reduction=self.reduction, 
                                negative_mode=None)
        else:
            negative_inuse = negatives_mask.type(torch.LongTensor) if self.negative_mode == 3 else (~positives_mask).type(torch.LongTensor) # (B, B)
            negative_count = torch.count_nonzero(negative_inuse, dim=-1).to(device) # (B)
            negative_zero_mask = torch.le(negative_count, 0)
            negative_count.masked_fill_(negative_zero_mask, B)
            rand_base = torch.randint(0, 10000, (B, B), device=device)
            rand_base_1 = (torch.fmod(rand_base, negative_count.unsqueeze(-1))).type(torch.int64)
            rand_base_2 = torch.arange(0, B, device=device).unsqueeze(0).expand(B, -1)
            rand_base_2 = rand_base_2.where(torch.lt(rand_base_2, negative_count.unsqueeze(-1)), rand_base_1)
            _, indices = torch.sort(negative_inuse, dim=-1, descending=True)
            rand_indices = torch.gather(indices.to(device), 1, rand_base_2) # (B, B)
            negatives_keys = embeddings2[rand_indices, :] # (B, B, D)
            if self.distance_mode == 1:
                return info_nce_v1dot5(query=embeddings1, 
                            positive_key=positive_key, 
                            negative_keys=negatives_keys, 
                            temperature=self.temperature, 
                            reduction=self.reduction, 
                            negative_mode='paired')
            else:
                return info_nce(query=embeddings1, 
                            positive_key=positive_key, 
                            negative_keys=negatives_keys, 
                            temperature=self.temperature, 
                            reduction=self.reduction, 
                            negative_mode='paired')

class MBInfoNCELoss(nn.Module):
    def __init__(self, cfgs):
        super(MBInfoNCELoss, self).__init__()
        self.temperature = cfgs.temperature
        self.reduction = cfgs.reduction
        self.distance_mode = cfgs.distance_mode # 1: p2 norm, 2: cosine similarity
        assert self.distance_mode == 2, 'do you transform the L2 distance to any kind of similarity?'

    def forward(self, embeddings, positive_key, negative_keys, positives_mask, negatives_mask):
        B, DB = negatives_mask.shape
        device = embeddings.device
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
        negatives_keys = negative_keys[rand_indices, :] # (B, DB, D)
        if self.distance_mode == 1:
            return info_nce_v1dot5(query=embeddings, 
                        positive_key=positive_key, 
                        negative_keys=negatives_keys, 
                        temperature=self.temperature, 
                        reduction=self.reduction, 
                        negative_mode='paired')
        else:
            return info_nce(query=embeddings, 
                        positive_key=positive_key, 
                        negative_keys=negatives_keys, 
                        temperature=self.temperature, 
                        reduction=self.reduction, 
                        negative_mode='paired')

class InfoNCELoss_v2(nn.Module):
    def __init__(self, cfgs):
        super(InfoNCELoss_v2, self).__init__()
        self.temperature = cfgs.temperature
        self.reduction = cfgs.reduction
        self.negative_mode = cfgs.negative_mode # 1: no negative filter, 2: none positives_mask as negative, 3: negatives_mask as negative
        self.positive_mode = cfgs.positive_mode # 1: itself, 2: random positive_sample except itself, 3: random positive_sample including itself
        self.distance_mode = cfgs.distance_mode # 1: p2 norm, 2: cosine similarity
        self.positive_overlap_margin = cfgs.positive_overlap_margin
        self.negative_overlap_margin = cfgs.negative_overlap_margin
        assert self.distance_mode == 2, 'do you transform the L2 distance to any kind of similarity?'
        assert self.negative_overlap_margin < self.positive_overlap_margin, 'Negative overlap margin should be smaller than positive overlap margin'

    def forward(self, embeddings1, embeddings2, overlap_ratio):
        positives_mask = torch.gt(overlap_ratio, self.positive_overlap_margin)
        negatives_mask = torch.le(overlap_ratio, self.negative_overlap_margin)
        B, _ = embeddings1.shape
        device = embeddings1.device
        if self.positive_mode == 1:
            positive_key=embeddings2
            if self.negative_mode == 1:
                positive_overlap_ratio_vet = overlap_ratio # (B, B)
            else:
                positive_overlap_ratio_vet = overlap_ratio.diagonal() # (B)
        else:
            if self.positive_mode == 2:
                real_positives_mask = torch.logical_and(~torch.eye(n=B, dtype=torch.long, device=device), positives_mask)
            elif self.positive_mode == 3:
                real_positives_mask = positives_mask.type(torch.LongTensor)
            positive_count = torch.count_nonzero(real_positives_mask, dim=-1).to(device) # (B)
            rand_base = torch.randint(0, B, (B,), device=device)
            rand_base_1 = (torch.fmod(rand_base, positive_count)).type(torch.int64)
            _, indices = torch.sort(real_positives_mask.type(torch.LongTensor), dim=-1, descending=True)
            batch_indices = torch.arange(0, B, device=device)
            rand_indices = indices[batch_indices, rand_base_1] # (B)
            positive_key = embeddings2[rand_indices, :] # (B, D)
            if self.negative_mode == 1:
                batch_indices = batch_indices.unsqueeze(-1).expand(-1, B)
                rand_indices = rand_indices.unsqueeze(0).expand(B, -1)
                positive_overlap_ratio_vet = overlap_ratio[batch_indices, rand_indices]
            else:
                positive_overlap_ratio_vet = overlap_ratio[batch_indices, rand_indices] # (B)
        if self.negative_mode == 1:
            if self.distance_mode == 1:
                return info_nce_v3(query=embeddings1, 
                            positive_key=positive_key, 
                            negative_keys=None, 
                            temperature=self.temperature, 
                            reduction=self.reduction, 
                            negative_mode=None,
                            positive_overlap_ratio_vet=positive_overlap_ratio_vet,
                            negative_overlap_ratio_vet=None)
            else:
                return info_nce_v2(query=embeddings1, 
                            positive_key=positive_key, 
                            negative_keys=None, 
                            temperature=self.temperature, 
                            reduction=self.reduction, 
                            negative_mode=None,
                            positive_overlap_ratio_vet=positive_overlap_ratio_vet,
                            negative_overlap_ratio_vet=None)
        else:
            negative_inuse = negatives_mask.type(torch.LongTensor) if self.negative_mode == 3 else (~positives_mask).type(torch.LongTensor) # (B, B)
            negative_count = torch.count_nonzero(negative_inuse, dim=-1).to(device) # (B)
            negative_zero_mask = torch.le(negative_count, 0)
            negative_count.masked_fill_(negative_zero_mask, B)
            rand_base = torch.randint(0, 10000, (B, B), device=device)
            rand_base_1 = (torch.fmod(rand_base, negative_count.unsqueeze(-1))).type(torch.int64)
            rand_base_2 = torch.arange(0, B, device=device).unsqueeze(0).expand(B, -1)
            rand_base_2 = rand_base_2.where(torch.lt(rand_base_2, negative_count.unsqueeze(-1)), rand_base_1)
            _, indices = torch.sort(negative_inuse, dim=-1, descending=True)
            rand_indices = torch.gather(indices.to(device), 1, rand_base_2) # (B, B)
            negatives_keys = embeddings2[rand_indices, :] # (B, B, D)
            batch_indices = torch.arange(0, B, device=device).unsqueeze(-1).expand(-1, B)
            negative_overlap_ratio_vet = overlap_ratio[batch_indices, rand_indices] # (B, B)
            if self.distance_mode == 1:
                return info_nce_v3(query=embeddings1, 
                            positive_key=positive_key, 
                            negative_keys=negatives_keys, 
                            temperature=self.temperature, 
                            reduction=self.reduction, 
                            negative_mode='paired',
                            positive_overlap_ratio_vet=positive_overlap_ratio_vet,
                            negative_overlap_ratio_vet=negative_overlap_ratio_vet)
            else:
                return info_nce_v2(query=embeddings1, 
                            positive_key=positive_key, 
                            negative_keys=negatives_keys, 
                            temperature=self.temperature, 
                            reduction=self.reduction, 
                            negative_mode='paired',
                            positive_overlap_ratio_vet=positive_overlap_ratio_vet,
                            negative_overlap_ratio_vet=negative_overlap_ratio_vet)

def info_nce_v1dot5(query, 
                positive_key, 
                negative_keys=None, 
                temperature=0.1, 
                reduction='mean', 
                negative_mode='unpaired',):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = F.pairwise_distance(query, positive_key, p=2.0, eps=1e-06, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = torch.dist(query, negative_keys, p=2.0)

        elif negative_mode == 'paired':
            B, D = query.shape
            query = query.expand(B * B, -1)
            negative_keys_used = negative_keys.view(B * B, D)
            negative_logits = F.pairwise_distance(query, negative_keys_used, p=2.0, eps=1e-06, keepdim=False)
            negative_logits = negative_logits.view(B, B)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = torch.cdist(query, positive_key, p=2.0)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)

def info_nce_v2(query, 
                positive_key, 
                negative_keys=None, 
                temperature=0.1, 
                reduction='mean', 
                negative_mode='unpaired', 
                positive_overlap_ratio_vet=None, 
                negative_overlap_ratio_vet=None):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')


    negative_overlap_ratio_vet = 1.0 - negative_overlap_ratio_vet


    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True) * positive_overlap_ratio_vet.unsqueeze(-1)

        if negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)
            negative_logits = negative_logits * negative_overlap_ratio_vet

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)
        logits *= positive_overlap_ratio_vet

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)

def info_nce_v3(query, 
                positive_key, 
                negative_keys=None, 
                temperature=0.1, 
                reduction='mean', 
                negative_mode='unpaired', 
                positive_overlap_ratio_vet=None, 
                negative_overlap_ratio_vet=None):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = F.pairwise_distance(query, positive_key, p=2.0, eps=1e-06, keepdim=True) * positive_overlap_ratio_vet.unsequeeze(-1)

        if negative_mode == 'paired':
            B, D = query.shape
            query = query.expand(B * B, -1)
            negative_keys_used = negative_keys.view(B * B, D)
            negative_logits = F.pairwise_distance(query, negative_keys_used, p=2.0, eps=1e-06, keepdim=False)
            negative_logits = negative_logits.view(B, B)
            negative_logits = negative_logits * negative_overlap_ratio_vet

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = torch.cdist(query, positive_key, p=2.0)
        logits *= positive_overlap_ratio_vet

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def info_nce_v4(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)