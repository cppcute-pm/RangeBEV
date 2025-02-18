import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
    
class AngleLoss(nn.Module):

    def __init__(self, cfgs) -> None:
        # got base margin and change the margin according to the overlap_ratio difference
        super(AngleLoss, self).__init__()
        self.angle_loss_type = cfgs.angle_loss_type
        if 'feat_angle_dist_scale' not in cfgs.keys():
            self.feat_angle_dist_scale = 1.0
        else:
            self.feat_angle_dist_scale = cfgs.feat_angle_dist_scale
        if 'pair_type' not in cfgs.keys():
            self.pair_type = 'positive'
        else:
            self.pair_type = cfgs.pair_type
    
    def forward(self, feat_angle_dist_matrix, positive_masks, true_angle_dist_matrix):
        # calculate the distance between embeddings1 and embeddings2

        if self.pair_type == 'positive':
            positive_indices = torch.nonzero(positive_masks, as_tuple=False) # (N, 2)
            feat_angle_dist_positives = feat_angle_dist_matrix[positive_indices[:, 0], positive_indices[:, 1], :] # (N, 2)
            true_angle_dist_positives = true_angle_dist_matrix[positive_indices[:, 0], positive_indices[:, 1], :] # (N, 2)
            feat_angle_dist_positives = feat_angle_dist_positives * self.feat_angle_dist_scale
        elif self.pair_type == 'all':
            feat_angle_dist_positives = feat_angle_dist_matrix.reshape(-1, 2) # (N, 2)
            true_angle_dist_positives = true_angle_dist_matrix.reshape(-1, 2) # (N, 2)
            feat_angle_dist_positives = feat_angle_dist_positives * self.feat_angle_dist_scale
        else:
            raise NotImplementedError
        if self.angle_loss_type == 'mse':
            loss = F.pairwise_distance(feat_angle_dist_positives, true_angle_dist_positives, p=2.0)
            loss = torch.mean(loss)
            loss = loss * 0.5
        elif self.angle_loss_type == 'cos_sim':
            feat_angle_dist_positives_normalized = F.normalize(feat_angle_dist_positives, p=2, dim=1) # (N, 2)
            true_angle_dist_positives_normalized = F.normalize(true_angle_dist_positives, p=2, dim=1)   # (N, 2)
            loss = 1.0 - torch.sum(feat_angle_dist_positives_normalized * true_angle_dist_positives_normalized, dim=1) # (N, )
            loss = torch.mean(loss)
        else:
            raise NotImplementedError
        
        return loss


class AngleLossV2(nn.Module):

    def __init__(self, cfgs) -> None:
        # got base margin and change the margin according to the overlap_ratio difference
        super(AngleLossV2, self).__init__()
        if 'triplet_type' in cfgs.keys():
            self.triplet_type = cfgs.triplet_type
        else:
            self.triplet_type = 'between_positives'
    def forward(self, feat_angle_dist_matrix, positive_masks, true_angle_dist_matrix):
        # calculate the distance between embeddings1 and embeddings2

        if self.triplet_type == 'between_positives':
            triplet_mask = torch.logical_and(positive_masks.unsqueeze(-1), positive_masks.unsqueeze(-2))
        elif self.triplet_type == 'between_all':
            N = positive_masks.shape[0]
            triplet_mask = torch.ones((N , N, N), device=positive_masks.device, dtype=torch.bool)
        else:
            raise NotImplementedError
        indices_tuple = torch.nonzero(triplet_mask, as_tuple=False) # (N, 3)
        indices_tuple_mask1 = indices_tuple[:, 0] == indices_tuple[:, 1] # (N, )
        indices_tuple_mask2 = indices_tuple[:, 0] == indices_tuple[:, 2] # (N, )
        indices_tuple_mask3 = indices_tuple[:, 1] == indices_tuple[:, 2] # (N, )
        indices_tuple_mask = torch.logical_or(torch.logical_or(indices_tuple_mask1, indices_tuple_mask2), indices_tuple_mask3)
        indices_tuple = indices_tuple[~indices_tuple_mask, :] # (M, 3)

        if indices_tuple.shape[0] == 0:
            loss = torch.tensor(0.0).cuda()
        else:
            feat_angle_dist_positives_1 = feat_angle_dist_matrix[indices_tuple[:, 0], indices_tuple[:, 1], :] # (N, 2)
            feat_angle_dist_positives_1_normalized = F.normalize(feat_angle_dist_positives_1, p=2, dim=1, eps=1e-6)
            feat_angle_dist_positives_2 = feat_angle_dist_matrix[indices_tuple[:, 0], indices_tuple[:, 2], :] # (N, 2)
            feat_angle_dist_positives_2_normalized = F.normalize(feat_angle_dist_positives_2, p=2, dim=1, eps=1e-6)
            feat_angle_dist_positives_cossim = torch.sum(feat_angle_dist_positives_1_normalized * feat_angle_dist_positives_2_normalized, dim=1) # (N, )
            true_angle_dist_positives_1 = true_angle_dist_matrix[indices_tuple[:, 0], indices_tuple[:, 1], :] # (N, 2)
            true_angle_dist_positives_1_normalized = F.normalize(true_angle_dist_positives_1, p=2, dim=1, eps=1e-6)
            true_angle_dist_positives_2 = true_angle_dist_matrix[indices_tuple[:, 0], indices_tuple[:, 2], :] # (N, 2)
            true_angle_dist_positives_2_normalized = F.normalize(true_angle_dist_positives_2, p=2, dim=1, eps=1e-6)
            true_angle_dist_positives_cossim = torch.sum(true_angle_dist_positives_1_normalized * true_angle_dist_positives_2_normalized, dim=1) # (N, )

            loss = F.pairwise_distance(feat_angle_dist_positives_cossim.unsqueeze(-1), true_angle_dist_positives_cossim, p=2.0)
            loss = torch.mean(loss)
            loss = loss * 0.5
        
        return loss