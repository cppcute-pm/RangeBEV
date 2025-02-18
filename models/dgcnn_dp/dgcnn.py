import torch.nn as nn
import torch
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def index_aggregate_features(idxs, features):
    """
    Input:
        idxs: input index data, [B, N, K]
        features: input points data, [B, M, C]
    Return:
        new_features:, indexed points data, [B, N, C]
    """
    device = idxs.device
    B, N, K = idxs.shape
    _, C, M = features.shape
    idxs = idxs.type(torch.long)
    batch_indices = torch.arange(B, dtype=torch.long, device=device).unsqueeze(1).unsqueeze(2).expand_as(idxs) # [B, N, K]
    new_features = features[batch_indices, :, idxs] # [B, N, K, C]
    new_features = (new_features.mean(dim=2, keepdim=False)).permute(0, 2, 1) # [B, C, N]
    return new_features

@torch.no_grad()
def KNN(support, query, k):
    """
    Args:
        support ([tensor]): [B, N, C]
        query ([tensor]): [B, M, C]
    Returns:
        [int]: neighbor idx. [B, M, K]
    """
    dist = torch.cdist(support, query)
    k_dist = dist.topk(k=k, dim=1, largest=False)
    return k_dist.values, k_dist.indices.transpose(1, 2).contiguous().int()

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    # device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


class Model(nn.Module):
    # the code resource is https://github.com/WangYueFt/dgcnn/tree/f765b469a67730658ba554e97dc11723a7bab628/pytorch

    def __init__(self, k, emb_dims, dropout, output_channels=40):
        super(Model, self).__init__()
        self.k = k

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(emb_dims),
                                   nn.LeakyReLU(negative_slope=0.2))
        # self.linear1 = nn.Linear(emb_dims * 2, 512, bias=False)
        # self.bn6 = nn.BatchNorm1d(512)
        # self.dp1 = nn.Dropout(p=dropout)
        # self.linear2 = nn.Linear(512, 256)
        # self.bn7 = nn.BatchNorm1d(256)
        # self.dp2 = nn.Dropout(p=dropout)
        # self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)

        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x



class DGCNN(Model):

    def __init__(self, cfgs, out_dim):
        super(DGCNN, self).__init__(k=cfgs.k, 
                                    emb_dims=cfgs.emb_dims, 
                                    dropout=cfgs.dropout, 
                                    output_channels=out_dim)
        
        self.layer_dims = [64, 64 ,128, 256, cfgs.emb_dims]
        self.f_layer = cfgs.f_layer
        self.c_layer = cfgs.c_layer
        self.reducer = cfgs.reducer
        self.out_k = cfgs.out_k
        self.out_dim = out_dim
        self.f_fc = nn.Sequential(
            nn.Conv1d(self.layer_dims[self.f_layer - 1], self.out_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(self.out_dim),
            )
        self.c_fc = nn.Sequential(
            nn.Conv1d(self.layer_dims[self.c_layer - 1], self.out_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(self.out_dim),
            )

    
    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(1)
        points = []
        points.append(x.detach().clone())
        x = x.permute(0, 2, 1)

        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        num_points = num_points // self.reducer[0]
        fps_idx = pointnet2_utils.furthest_point_sample(points[0], num_points).long()  # [B, n1point]
        points.append(index_points(points[0], fps_idx))  # [B, npoint, 3]
        if self.f_layer == 1:
            _, dp_idxs = KNN(points[0], points[-1], self.out_k)
            f_ebds = index_aggregate_features(dp_idxs, x1)
            f_ebds_output = self.f_fc(f_ebds)
            f_points = points[-1]
            f_mask_vets = torch.ones_like(f_ebds_output[:, 0, :], dtype= torch.bool)
        if self.c_layer == 1:
            _, dp_idxs = KNN(points[0], points[-1], self.out_k)
            c_ebds = index_aggregate_features(dp_idxs, x1)
            c_ebds_output = self.c_fc(c_ebds)
            c_points = points[-1]
            c_mask_vets = torch.ones_like(c_ebds_output[:, 0, :], dtype= torch.bool)
            return f_ebds_output, c_ebds_output, f_points, c_points, f_mask_vets, c_mask_vets 

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]
        num_points = num_points // self.reducer[1]
        fps_idx = pointnet2_utils.furthest_point_sample(points[0], num_points).long()  # [B, n1point]
        points.append(index_points(points[0], fps_idx))  # [B, npoint, 3]
        if self.f_layer == 2:
            _, dp_idxs = KNN(points[0], points[-1], self.out_k)
            f_ebds = index_aggregate_features(dp_idxs, x2)
            f_ebds_output = self.f_fc(f_ebds)
            f_points = points[-1]
            f_mask_vets = torch.ones_like(f_ebds_output[:, 0, :], dtype= torch.bool)
        if self.c_layer == 2:
            _, dp_idxs = KNN(points[0], points[-1], self.out_k)
            c_ebds = index_aggregate_features(dp_idxs, x2)
            c_ebds_output = self.c_fc(c_ebds)
            c_points = points[-1]
            c_mask_vets = torch.ones_like(c_ebds_output[:, 0, :], dtype= torch.bool)
            return f_ebds_output, c_ebds_output, f_points, c_points, f_mask_vets, c_mask_vets 

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        num_points = num_points // self.reducer[2]
        fps_idx = pointnet2_utils.furthest_point_sample(points[0], num_points).long()  # [B, n1point]
        points.append(index_points(points[0], fps_idx))  # [B, npoint, 3]
        if self.f_layer == 3:
            _, dp_idxs = KNN(points[0], points[-1], self.out_k)
            f_ebds = index_aggregate_features(dp_idxs, x3)
            f_ebds_output = self.f_fc(f_ebds)
            f_points = points[-1]
            f_mask_vets = torch.ones_like(f_ebds_output[:, 0, :], dtype= torch.bool)
        if self.c_layer == 3:
            _, dp_idxs = KNN(points[0], points[-1], self.out_k)
            c_ebds = index_aggregate_features(dp_idxs, x3)
            c_ebds_output = self.c_fc(c_ebds)
            c_points = points[-1]
            c_mask_vets = torch.ones_like(c_ebds_output[:, 0, :], dtype= torch.bool)
            return f_ebds_output, c_ebds_output, f_points, c_points, f_mask_vets, c_mask_vets 

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]
        num_points = num_points // self.reducer[3]
        fps_idx = pointnet2_utils.furthest_point_sample(points[0], num_points).long()  # [B, n1point]
        points.append(index_points(points[0], fps_idx))  # [B, npoint, 3]
        if self.f_layer == 4:
            _, dp_idxs = KNN(points[0], points[-1], self.out_k)
            f_ebds = index_aggregate_features(dp_idxs, x4)
            f_ebds_output = self.f_fc(f_ebds)
            f_points = points[-1]
            f_mask_vets = torch.ones_like(f_ebds_output[:, 0, :], dtype= torch.bool)
        if self.c_layer == 4:
            _, dp_idxs = KNN(points[0], points[-1], self.out_k)
            c_ebds = index_aggregate_features(dp_idxs, x4)
            c_ebds_output = self.c_fc(c_ebds)
            c_points = points[-1]
            c_mask_vets = torch.ones_like(c_ebds_output[:, 0, :] ,dtype= torch.bool)
            return f_ebds_output, c_ebds_output, f_points, c_points, f_mask_vets, c_mask_vets 

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        num_points = num_points // self.reducer[4]
        fps_idx = pointnet2_utils.furthest_point_sample(points[0], num_points).long()  # [B, n1point]
        points.append(index_points(points[0], fps_idx))  # [B, npoint, 3]
        if self.f_layer == 5:
            _, dp_idxs = KNN(points[0], points[-1], self.out_k)
            f_ebds = index_aggregate_features(dp_idxs, x)
            f_ebds_output = self.f_fc(f_ebds)
            f_points = points[-1]
            f_mask_vets = torch.ones_like(f_ebds_output[:, 0, :], dtype= torch.bool)
        if self.c_layer == 5:
            _, dp_idxs = KNN(points[0], points[-1], self.out_k)
            c_ebds = index_aggregate_features(dp_idxs, x)
            c_ebds_output = self.c_fc(c_ebds)
            c_points = points[-1]
            c_mask_vets = torch.ones_like(c_ebds_output[:, 0, :], dtype= torch.bool)
            return f_ebds_output, c_ebds_output, f_points, c_points, f_mask_vets, c_mask_vets 

        # x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        # x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        # x = torch.cat((x1, x2), 1)

        # x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        # x = self.dp1(x)
        # x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        # x = self.dp2(x)
        # x = self.linear3(x)
        # return x

class DGCNNv2(Model):

    def __init__(self, cfgs, out_dim):
        super(DGCNNv2, self).__init__(k=cfgs.k, 
                                    emb_dims=cfgs.emb_dims, 
                                    dropout=cfgs.dropout, 
                                    output_channels=out_dim)
        
        self.layer_dims = [64, 64 ,128, 256, cfgs.emb_dims]
        self.layer_1 = cfgs.layer_1
        self.layer_2 = cfgs.layer_2
        self.layer_3 = cfgs.layer_3
        self.reducer = cfgs.reducer
        self.out_k = cfgs.out_k
        self.out_dim = out_dim
        self.layer_fc_1 = nn.Sequential(
            nn.Conv1d(self.layer_dims[self.layer_1 - 1], self.out_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(self.out_dim),
            )
        self.layer_fc_2 = nn.Sequential(
            nn.Conv1d(self.layer_dims[self.layer_2 - 1], self.out_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(self.out_dim),
            )
        self.layer_fc_3 = nn.Sequential(
            nn.Conv1d(self.layer_dims[self.layer_3 - 1], self.out_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(self.out_dim),
            )

    
    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(1)
        points = []
        points.append(x.detach().clone())
        x = x.permute(0, 2, 1)

        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        num_points = num_points // self.reducer[0]
        fps_idx = pointnet2_utils.furthest_point_sample(points[0], num_points).long()  # [B, n1point]
        points.append(index_points(points[0], fps_idx))  # [B, npoint, 3]
        if self.layer_1 == 1:
            _, dp_idxs = KNN(points[0], points[-1], self.out_k)
            ebds_1 = index_aggregate_features(dp_idxs, x1)
            ebds_1_output = self.layer_fc_1(ebds_1)
            points_1 = points[-1]
            mask_1_vets = torch.ones_like(ebds_1_output[:, 0, :], dtype= torch.bool)
        if self.layer_2 == 1:
            _, dp_idxs = KNN(points[0], points[-1], self.out_k)
            ebds_2 = index_aggregate_features(dp_idxs, x1)
            ebds_2_output = self.layer_fc_2(ebds_2)
            points_2 = points[-1]
            mask_2_vets = torch.ones_like(ebds_2_output[:, 0, :], dtype= torch.bool)
        if self.layer_3 == 1:
            _, dp_idxs = KNN(points[0], points[-1], self.out_k)
            ebds_3 = index_aggregate_features(dp_idxs, x1)
            ebds_3_output = self.layer_fc_3(ebds_3)
            points_3 = points[-1]
            mask_3_vets = torch.ones_like(ebds_3_output[:, 0, :], dtype= torch.bool)
            return ebds_1_output, ebds_2_output, ebds_3_output, points_1, points_2, points_3, mask_1_vets, mask_2_vets, mask_3_vets 

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]
        num_points = num_points // self.reducer[1]
        fps_idx = pointnet2_utils.furthest_point_sample(points[0], num_points).long()  # [B, n1point]
        points.append(index_points(points[0], fps_idx))  # [B, npoint, 3]
        if self.layer_1 == 2:
            _, dp_idxs = KNN(points[0], points[-1], self.out_k)
            ebds_1 = index_aggregate_features(dp_idxs, x2)
            ebds_1_output = self.layer_fc_1(ebds_1)
            points_1 = points[-1]
            mask_1_vets = torch.ones_like(ebds_1_output[:, 0, :], dtype= torch.bool)
        if self.layer_2 == 2:
            _, dp_idxs = KNN(points[0], points[-1], self.out_k)
            ebds_2 = index_aggregate_features(dp_idxs, x2)
            ebds_2_output = self.layer_fc_2(ebds_2)
            points_2 = points[-1]
            mask_2_vets = torch.ones_like(ebds_2_output[:, 0, :], dtype= torch.bool)
        if self.layer_3 == 2:
            _, dp_idxs = KNN(points[0], points[-1], self.out_k)
            ebds_3 = index_aggregate_features(dp_idxs, x2)
            ebds_3_output = self.layer_fc_3(ebds_3)
            points_3 = points[-1]
            mask_3_vets = torch.ones_like(ebds_3_output[:, 0, :], dtype= torch.bool)
            return ebds_1_output, ebds_2_output, ebds_3_output, points_1, points_2, points_3, mask_1_vets, mask_2_vets, mask_3_vets 
        

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        num_points = num_points // self.reducer[2]
        fps_idx = pointnet2_utils.furthest_point_sample(points[0], num_points).long()  # [B, n1point]
        points.append(index_points(points[0], fps_idx))  # [B, npoint, 3]
        if self.layer_1 == 3:
            _, dp_idxs = KNN(points[0], points[-1], self.out_k)
            ebds_1 = index_aggregate_features(dp_idxs, x3)
            ebds_1_output = self.layer_fc_1(ebds_1)
            points_1 = points[-1]
            mask_1_vets = torch.ones_like(ebds_1_output[:, 0, :], dtype= torch.bool)
        if self.layer_2 == 3:
            _, dp_idxs = KNN(points[0], points[-1], self.out_k)
            ebds_2 = index_aggregate_features(dp_idxs, x3)
            ebds_2_output = self.layer_fc_2(ebds_2)
            points_2 = points[-1]
            mask_2_vets = torch.ones_like(ebds_2_output[:, 0, :], dtype= torch.bool)
        if self.layer_3 == 3:
            _, dp_idxs = KNN(points[0], points[-1], self.out_k)
            ebds_3 = index_aggregate_features(dp_idxs, x3)
            ebds_3_output = self.layer_fc_3(ebds_3)
            points_3 = points[-1]
            mask_3_vets = torch.ones_like(ebds_3_output[:, 0, :], dtype= torch.bool)
            return ebds_1_output, ebds_2_output, ebds_3_output, points_1, points_2, points_3, mask_1_vets, mask_2_vets, mask_3_vets 

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]
        num_points = num_points // self.reducer[3]
        fps_idx = pointnet2_utils.furthest_point_sample(points[0], num_points).long()  # [B, n1point]
        points.append(index_points(points[0], fps_idx))  # [B, npoint, 3]
        if self.layer_1 == 4:
            _, dp_idxs = KNN(points[0], points[-1], self.out_k)
            ebds_1 = index_aggregate_features(dp_idxs, x4)
            ebds_1_output = self.layer_fc_1(ebds_1)
            points_1 = points[-1]
            mask_1_vets = torch.ones_like(ebds_1_output[:, 0, :], dtype= torch.bool)
        if self.layer_2 == 4:
            _, dp_idxs = KNN(points[0], points[-1], self.out_k)
            ebds_2 = index_aggregate_features(dp_idxs, x4)
            ebds_2_output = self.layer_fc_2(ebds_2)
            points_2 = points[-1]
            mask_2_vets = torch.ones_like(ebds_2_output[:, 0, :], dtype= torch.bool)
        if self.layer_3 == 4:
            _, dp_idxs = KNN(points[0], points[-1], self.out_k)
            ebds_3 = index_aggregate_features(dp_idxs, x4)
            ebds_3_output = self.layer_fc_3(ebds_3)
            points_3 = points[-1]
            mask_3_vets = torch.ones_like(ebds_3_output[:, 0, :], dtype= torch.bool)
            return ebds_1_output, ebds_2_output, ebds_3_output, points_1, points_2, points_3, mask_1_vets, mask_2_vets, mask_3_vets 

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        num_points = num_points // self.reducer[4]
        fps_idx = pointnet2_utils.furthest_point_sample(points[0], num_points).long()  # [B, n1point]
        points.append(index_points(points[0], fps_idx))  # [B, npoint, 3]
        if self.layer_1 == 5:
            _, dp_idxs = KNN(points[0], points[-1], self.out_k)
            ebds_1 = index_aggregate_features(dp_idxs, x)
            ebds_1_output = self.layer_fc_1(ebds_1)
            points_1 = points[-1]
            mask_1_vets = torch.ones_like(ebds_1_output[:, 0, :], dtype= torch.bool)
        if self.layer_2 == 5:
            _, dp_idxs = KNN(points[0], points[-1], self.out_k)
            ebds_2 = index_aggregate_features(dp_idxs, x)
            ebds_2_output = self.layer_fc_2(ebds_2)
            points_2 = points[-1]
            mask_2_vets = torch.ones_like(ebds_2_output[:, 0, :], dtype= torch.bool)
        if self.layer_3 == 5:
            _, dp_idxs = KNN(points[0], points[-1], self.out_k)
            ebds_3 = index_aggregate_features(dp_idxs, x)
            ebds_3_output = self.layer_fc_3(ebds_3)
            points_3 = points[-1]
            mask_3_vets = torch.ones_like(ebds_3_output[:, 0, :], dtype= torch.bool)
            return ebds_1_output, ebds_2_output, ebds_3_output, points_1, points_2, points_3, mask_1_vets, mask_2_vets, mask_3_vets 