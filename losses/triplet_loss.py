import numpy as np
import torch
from pytorch_metric_learning import losses, miners, reducers
from pytorch_metric_learning.distances import LpDistance
import torch.nn as nn
import torch.nn.functional as F

def get_max_per_row(mat, mask):
    non_zero_rows = torch.any(mask, dim=1)
    mat_masked = mat.clone()
    mat_masked[~mask] = 0
    return torch.max(mat_masked, dim=1), non_zero_rows

def get_min_per_row(mat, mask):
    non_inf_rows = torch.any(mask, dim=1)
    mat_masked = mat.clone()
    mat_masked[~mask] = float('inf')
    return torch.min(mat_masked, dim=1), non_inf_rows

class HardTripletMinerWithMasks:
    # Hard triplet miner
    def __init__(self, distance):
        self.distance = distance
        # Stats
        self.max_pos_pair_dist = None
        self.max_neg_pair_dist = None
        self.mean_pos_pair_dist = None
        self.mean_neg_pair_dist = None
        self.min_pos_pair_dist = None
        self.min_neg_pair_dist = None

    def __call__(self, embeddings1, embeddings2, positives_mask, negatives_mask):
        assert embeddings1.dim() == 2 and embeddings2.dim() == 2
        d_embeddings1 = embeddings1.detach()
        d_embeddings2 = embeddings2.detach()
        with torch.no_grad():
            hard_triplets = self.mine(d_embeddings1, d_embeddings2, positives_mask, negatives_mask)
        return hard_triplets

    def mine(self, embeddings1, embeddings2, positives_mask, negatives_mask):
        # Based on pytorch-metric-learning implementation
        dist_mat = self.distance(embeddings1, embeddings2)
        (hardest_positive_dist, hardest_positive_indices), a1p_keep = get_max_per_row(dist_mat, positives_mask)
        (hardest_negative_dist, hardest_negative_indices), a2n_keep = get_min_per_row(dist_mat, negatives_mask)
        a_keep_idx = torch.where(a1p_keep & a2n_keep)
        a = torch.arange(dist_mat.size(0)).to(hardest_positive_indices.device)[a_keep_idx]
        p = hardest_positive_indices[a_keep_idx]
        n = hardest_negative_indices[a_keep_idx]
        self.max_pos_pair_dist = torch.max(hardest_positive_dist).item()
        self.max_neg_pair_dist = torch.max(hardest_negative_dist).item()
        self.mean_pos_pair_dist = torch.mean(hardest_positive_dist).item()
        self.mean_neg_pair_dist = torch.mean(hardest_negative_dist).item()
        self.min_pos_pair_dist = torch.min(hardest_positive_dist).item()
        self.min_neg_pair_dist = torch.min(hardest_negative_dist).item()
        return a, p, n


class TripletLoss(nn.Module):
    def __init__(self, cfgs):
        super(TripletLoss, self).__init__()
        self.margin = cfgs.margin
        self.distance = LpDistance(normalize_embeddings=cfgs.normalize_embeddings, collect_stats=True, power=1, p=2) #欧几里得范数
        self.hard_mining = cfgs.hard_mining
        self.pair_dist_info = cfgs.pair_dist_info
        if self.hard_mining:
            self.miner = HardTripletMinerWithMasks(self.distance)
        # We use triplet loss with Euclidean distance
        reducer_fn = reducers.AvgNonZeroReducer(collect_stats=True)
        #reducer decides how the values of many loss are computed into one single value.
        #AvgNonZeroReducer means take average and greater than zero
        self.loss_fn = losses.TripletMarginLoss(margin=self.margin, swap=True, distance=self.distance,
                                                reducer=reducer_fn, collect_stats=True)
        #swap means if pos-neg dis violate the margin more the anc-neg dis,then use it,why use swap?

    def forward(self, embeddings1, embeddings2, positives_mask, negatives_mask):
        stats = None
        if not self.hard_mining:
            triplet_mask = torch.logical_and(positives_mask.unsqueeze(-1), negatives_mask.unsqueeze(-2))
            indices_tuple = torch.nonzero(triplet_mask, as_tuple=True)
        else:
            indices_tuple = self.miner(embeddings1, embeddings2, positives_mask, negatives_mask)

        loss = self.loss_fn(embeddings=embeddings1, indices_tuple=indices_tuple, ref_emb=embeddings2)
        if torch.isnan(loss):
            loss = torch.tensor(0.0, requires_grad=True, device=embeddings1.device)
        if self.pair_dist_info and self.hard_mining:
            stats = {'max_pos_pair_dist': self.miner.max_pos_pair_dist,
                     'max_neg_pair_dist': self.miner.max_neg_pair_dist,
                     'mean_pos_pair_dist': self.miner.mean_pos_pair_dist,
                     'mean_neg_pair_dist': self.miner.mean_neg_pair_dist,
                     'min_pos_pair_dist': self.miner.min_pos_pair_dist,
                     'min_neg_pair_dist': self.miner.min_neg_pair_dist}

        return loss, stats
    
class TripletLoss_v2(nn.Module):

    def __init__(self, cfgs) -> None:
        # got base margin and change the margin according to the overlap_ratio difference
        super(TripletLoss_v2, self).__init__()
        self.base_margin = cfgs.base_margin
        self.normalize_embeddings = cfgs.normalize_embeddings
        self.positive_overlap_ratio = cfgs.positive_overlap_ratio
        self.negative_overlap_ratio = cfgs.negative_overlap_ratio
        self.delta_overlap_ratio_margin = cfgs.delta_overlap_ratio
        self.tuple_formtype = cfgs.tuple_formtype
        self.choose_nonzero = False
        if 'choose_nonzero' in cfgs.keys():
            self.choose_nonzero = cfgs.choose_nonzero
        assert self.negative_overlap_ratio < self.positive_overlap_ratio, "negative_overlap_ratio should be smaller than positive_overlap_ratio"
    
    def forward(self, embeddings1, embeddings2, overlap_ratio):
        # calculate the distance between embeddings1 and embeddings2
        if self.normalize_embeddings:
            embeddings1 = F.normalize(embeddings1, p=2, dim=1)
            embeddings2 = F.normalize(embeddings2, p=2, dim=1)
        distance = torch.cdist(embeddings1, embeddings2, p=2.0) # (B, B)

        if self.tuple_formtype == 'relative_delta':
            delta_overlap_ratio = torch.sub(overlap_ratio.unsqueeze(2), overlap_ratio.unsqueeze(1)) # (B, B, B)
            indices_tuple = torch.nonzero(torch.gt(delta_overlap_ratio, self.delta_overlap_ratio_margin), as_tuple=True)
        elif self.tuple_formtype == 'absolute_threshold':
            positives_mask = torch.gt(overlap_ratio, self.positive_overlap_ratio)
            negatives_mask = torch.le(overlap_ratio, self.negative_overlap_ratio)
            triplet_mask = torch.logical_and(positives_mask.unsqueeze(-1), negatives_mask.unsqueeze(-2))
            indices_tuple = torch.nonzero(triplet_mask, as_tuple=True)
        # calculate the margin according to the overlap_ratio

        if indices_tuple[0].shape[0] == 0:
            return torch.tensor(0.0, requires_grad=True, device=embeddings1.device)
        
        positive_overlap_ratio = overlap_ratio[indices_tuple[0], indices_tuple[1]]
        negative_overlap_ratio = overlap_ratio[indices_tuple[0], indices_tuple[2]]
        margins = self.base_margin * (positive_overlap_ratio - negative_overlap_ratio)
        # calculate the loss
        loss = F.relu(distance[indices_tuple[0], indices_tuple[1]] - distance[indices_tuple[0], indices_tuple[2]] + margins)
        if not self.choose_nonzero:
            loss = torch.mean(loss)
        else:
            nonzero_num = torch.count_nonzero(loss)
            if nonzero_num == 0:
                loss = torch.mean(loss)
            else:
                loss = torch.sum(loss) / nonzero_num
        return loss

class TripletLoss_v3(nn.Module):

    def __init__(self, cfgs) -> None:
        # got base margin and change the margin according to the overlap_ratio difference
        super(TripletLoss_v3, self).__init__()
        self.base_margin = cfgs.base_margin
        self.normalize_embeddings = cfgs.normalize_embeddings
        self.positive_overlap_ratio = cfgs.positive_overlap_ratio
        self.negative_overlap_ratio = cfgs.negative_overlap_ratio
        self.delta_overlap_ratio_margin = cfgs.delta_overlap_ratio
        self.tuple_formtype = cfgs.tuple_formtype
        assert self.negative_overlap_ratio < self.positive_overlap_ratio, "negative_overlap_ratio should be smaller than positive_overlap_ratio"
    
    def forward(self, embeddings1, embeddings2, overlap_ratio):
        # calculate the distance between embeddings1 and embeddings2
        if self.normalize_embeddings:
            embeddings1 = F.normalize(embeddings1, p=2, dim=1)
            embeddings2 = F.normalize(embeddings2, p=2, dim=1)
        distance = torch.cdist(embeddings1, embeddings2, p=2.0)

        if self.tuple_formtype == 'relative_delta':
            delta_overlap_ratio = torch.sub(overlap_ratio.unsqueeze(2), overlap_ratio.unsqueeze(1)) # (B, B, B)
            indices_tuple = torch.nonzero(torch.gt(delta_overlap_ratio, self.delta_overlap_ratio_margin), as_tuple=True)
        elif self.tuple_formtype == 'absolute_threshold':
            positives_mask = torch.gt(overlap_ratio, self.positive_overlap_ratio)
            negatives_mask = torch.le(overlap_ratio, self.negative_overlap_ratio)
            triplet_mask = torch.logical_and(positives_mask.unsqueeze(-1), negatives_mask.unsqueeze(-2))
            indices_tuple = torch.nonzero(triplet_mask, as_tuple=True)
        # calculate the multiplier according to the overlap_ratio
        positive_overlap_ratio = overlap_ratio[indices_tuple[0], indices_tuple[1]]
        negative_overlap_ratio = overlap_ratio[indices_tuple[0], indices_tuple[2]]
        positive_multiplier = (1 - positive_overlap_ratio) * 10
        negative_multiplier = (1 - negative_overlap_ratio) * 10
        positive_dist = distance[indices_tuple[0], indices_tuple[1]]
        negative_dist = distance[indices_tuple[0], indices_tuple[2]]
        # calculate the loss
        loss = F.relu(positive_dist * positive_multiplier - negative_dist * negative_multiplier + self.base_margin)
        loss = torch.mean(loss)
        return loss


class TripletLoss_v4(nn.Module):

    def __init__(self, cfgs) -> None:
        # got base margin and change the margin according to the overlap_ratio difference
        super(TripletLoss_v4, self).__init__()
        self.base_margin = cfgs.base_margin
        self.normalize_embeddings = cfgs.normalize_embeddings
        self.positive_overlap_ratio = cfgs.positive_overlap_ratio
        self.negative_overlap_ratio = cfgs.negative_overlap_ratio
        self.delta_overlap_ratio_margin = cfgs.delta_overlap_ratio
        self.tuple_formtype = cfgs.tuple_formtype
        self.choose_nonzero = False
        if 'choose_nonzero' in cfgs.keys():
            self.choose_nonzero = cfgs.choose_nonzero
        assert self.negative_overlap_ratio < self.positive_overlap_ratio, "negative_overlap_ratio should be smaller than positive_overlap_ratio"
        if cfgs.p_type == 'learnable':
            self.p = nn.Parameter(torch.ones(1) * cfgs.p)
        elif cfgs.p_type == 'fixed':
            self.p = cfgs.p
        else:
            raise ValueError('p_type should be learnable or fixed')
        self.weights = cfgs.weights
    
    def forward(self, embeddings1, embeddings2, overlap_ratio):
        # calculate the distance between embeddings1 and embeddings2
        if self.normalize_embeddings:
            embeddings1 = F.normalize(embeddings1, p=2, dim=1)
            embeddings2 = F.normalize(embeddings2, p=2, dim=1)
        distance = torch.cdist(embeddings1, embeddings2, p=2.0) # (B, B)

        if self.tuple_formtype == 'relative_delta':
            delta_overlap_ratio = torch.sub(overlap_ratio.unsqueeze(2), overlap_ratio.unsqueeze(1)) # (B, B, B)
            indices_tuple = torch.nonzero(torch.gt(delta_overlap_ratio, self.delta_overlap_ratio_margin), as_tuple=True)
        elif self.tuple_formtype == 'absolute_threshold':
            positives_mask = torch.gt(overlap_ratio, self.positive_overlap_ratio)
            negatives_mask = torch.le(overlap_ratio, self.negative_overlap_ratio)
            triplet_mask = torch.logical_and(positives_mask.unsqueeze(-1), negatives_mask.unsqueeze(-2))
            indices_tuple = torch.nonzero(triplet_mask, as_tuple=True)
        # calculate the margin according to the overlap_ratio
        positive_overlap_ratio = overlap_ratio[indices_tuple[0], indices_tuple[1]]
        negative_overlap_ratio = overlap_ratio[indices_tuple[0], indices_tuple[2]]
        margins = self.base_margin * (positive_overlap_ratio - negative_overlap_ratio)
        # calculate the loss
        triplet_loss_without_relu = distance[indices_tuple[0], indices_tuple[1]] - distance[indices_tuple[0], indices_tuple[2]] + margins
        triplet_loss_weight = torch.full_like(triplet_loss_without_relu, self.weights[0])
        triplet_loss_weight[triplet_loss_without_relu < 0] = self.weights[1]
        loss = (triplet_loss_weight * torch.abs(triplet_loss_without_relu)).pow(self.p)
        if not self.choose_nonzero:
            loss = torch.mean(loss)
        else:
            nonzero_num = torch.count_nonzero(loss)
            if nonzero_num == 0:
                loss = torch.mean(loss)
            else:
                loss = torch.sum(loss) / nonzero_num
        return loss