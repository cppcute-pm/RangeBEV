import torch
from pointnet2_ops import pointnet2_utils
import math
import time
from torch import tensor
from torch import Tensor
from pykeops.torch import LazyTensor
from typing import Tuple

@torch.no_grad()
def keops_knn(device, q_points: Tensor, s_points: Tensor, k: int) -> Tuple[Tensor, Tensor]:
    """kNN with PyKeOps.

    Args:
        q_points (Tensor): (*, N, C)
        s_points (Tensor): (*, M, C)
        k (int)

    Returns:
        knn_distance (Tensor): (*, N, k)
        knn_indices (LongTensor): (*, N, k)
    """
    num_batch_dims = q_points.dim() - 2
    xi = LazyTensor(q_points.unsqueeze(-2))  # (*, N, 1, C)
    xj = LazyTensor(s_points.unsqueeze(-3))  # (*, 1, M, C)
    dij = (xi - xj).norm2()  # (*, N, M)
    if device != 'cpu':
        device_id = device.index
        knn_distances, knn_indices = dij.Kmin_argKmin(k, dim=num_batch_dims + 1, device_id=device_id)  # (*, N, K)
    else:
        knn_distances, knn_indices = dij.Kmin_argKmin(k, dim=num_batch_dims + 1)  # (*, N, K)
    return knn_distances, knn_indices

@torch.no_grad()
def generate_pc_index_and_coords_v1(coords_num_list, coords):
    assert coords.shape[1] == coords_num_list[0]
    B = coords.shape[0]
    coords_list = [coords]
    index_list = []
    for i in range(len(coords_num_list) - 1):
        idx = pointnet2_utils.furthest_point_sample(coords_list[-1], coords_num_list[i + 1]).long()
        curr_coords = torch.gather(coords_list[-1], 1, idx.unsqueeze(-1).expand(-1, -1, 3))
        curr_to_pre_dist = torch.cdist(curr_coords, coords_list[-1]) # Produces (B, curr_coords_num, pre_coords_num) tensor
        index = torch.argmin(curr_to_pre_dist, dim=1, keepdim=False) # Produces (B, pre_coords_num) tensor
        coords_list.append(curr_coords)
        index_list.append(index)
    index_list.append(torch.zeros((B, 1), dtype=torch.int64, device=coords.device))

    return index_list, coords_list

# @torch.no_grad()
# def generate_pc_index_and_coords_v2(coords_list):
#     B = coords_list[0].shape[0]
#     device = coords_list[0].device
#     curr_coords_list_length = len(coords_list)
#     index_list = []
#     for i in range(curr_coords_list_length):
#         if i == curr_coords_list_length - 1:
#             idx = pointnet2_utils.furthest_point_sample(coords_list[-1], 1).long()
#             curr_coords = torch.gather(coords_list[-1], 1, idx.unsqueeze(-1).expand(-1, -1, 3))
#             coords_list.append(curr_coords)
#         else:
#             curr_coords = coords_list[i+1]
#         curr_to_pre_dist = torch.cdist(curr_coords, coords_list[i]) # Produces (B, curr_coords_num, pre_coords_num) tensor
#         index = torch.argmin(curr_to_pre_dist, dim=1, keepdim=False) # Produces (B, pre_coords_num) tensor
#         index_list.append(index)
#     index_list.append(torch.zeros((B, 1), dtype=torch.int64, device=device))

#     return index_list, coords_list

@torch.no_grad()
def generate_pc_index_and_coords_v2(coords_list):
    B = coords_list[0].shape[0]
    device = coords_list[0].device
    curr_coords_list_length = len(coords_list)
    index_list = []
    for i in range(curr_coords_list_length):
        if i == curr_coords_list_length - 1:
            idx = pointnet2_utils.furthest_point_sample(coords_list[-1], 1).long()
            curr_coords = torch.gather(coords_list[-1], 1, idx.unsqueeze(-1).expand(-1, -1, 3))
            coords_list.append(curr_coords)
        else:
            curr_coords = coords_list[i+1]
        # curr_to_pre_dist = torch.cdist(curr_coords, coords_list[i]) # Produces (B, curr_coords_num, pre_coords_num) tensor
        # index = torch.argmin(curr_to_pre_dist, dim=1, keepdim=False) # Produces (B, pre_coords_num) tensor
        _, index = keops_knn(device, coords_list[i], curr_coords, 1) # Produces (B, pre_coords_num, 1) tensor
        index = index.squeeze(-1)
        index_list.append(index)
    index_list.append(torch.zeros((B, 1), dtype=torch.int64, device=device))

    return index_list, coords_list

@torch.no_grad()
def generate_pc_index_and_coords_v3(coords_list, diff_res_num_neighbor_list):
    # curr points num is less
    B = coords_list[0].shape[0]
    device = coords_list[0].device
    curr_coords_list_length = len(coords_list)
    index_list = []

    for i in range(curr_coords_list_length):
        if i == curr_coords_list_length - 1:
            idx = pointnet2_utils.furthest_point_sample(coords_list[-1], 1).long()
            curr_coords = torch.gather(coords_list[-1], 1, idx.unsqueeze(-1).expand(-1, -1, 3))
            coords_list.append(curr_coords)
        else:
            curr_coords = coords_list[i+1]
        _, index = keops_knn(device, curr_coords, coords_list[i], diff_res_num_neighbor_list[i]) # Produces (B, curr_coords_num, num_neighbor) tensor
        index_list.append(index)

    return index_list, coords_list

# @torch.no_grad()
# def generate_pc_index_and_coords_v3(coords_list, diff_res_num_neighbor_list):
#     # curr points num is less
#     B = coords_list[0].shape[0]
#     device = coords_list[0].device
#     curr_coords_list_length = len(coords_list)
#     index_list = []

#     for i in range(curr_coords_list_length):
#         if i == curr_coords_list_length - 1:
#             idx = pointnet2_utils.furthest_point_sample(coords_list[-1], 1).long()
#             curr_coords = torch.gather(coords_list[-1], 1, idx.unsqueeze(-1).expand(-1, -1, 3))
#             coords_list.append(curr_coords)
#         else:
#             curr_coords = coords_list[i+1]
#         curr_to_pre_dist = torch.cdist(curr_coords, coords_list[i]) # Produces (B, curr_coords_num, pre_coords_num) tensor
#         index = torch.topk(curr_to_pre_dist, k=diff_res_num_neighbor_list[i], dim=-1, largest=False, sorted=False)[1]
#         index_list.append(index)

#     return index_list, coords_list

# @torch.no_grad()
# def generate_pc_index_and_coords_v3(coords_list, diff_res_num_neighbor_list):
#     # curr points num is less
#     B = coords_list[0].shape[0]
#     device = coords_list[0].device
#     curr_coords_list_length = len(coords_list)
#     index_list = []

#     for i in range(curr_coords_list_length):
#         if i == curr_coords_list_length - 1:
#             idx = pointnet2_utils.furthest_point_sample(coords_list[-1], 1).long()
#             next_coords = torch.gather(coords_list[-1], 1, idx.unsqueeze(-1).expand(-1, -1, 3))
#             coords_list.append(next_coords)
#         else:
#             next_coords = coords_list[i+1]
#         curr_coords = coords_list[i]
#         curr_batch = torch.full((curr_coords.shape[0],), curr_coords.shape[1], dtype=torch.int64, device=device)
#         curr_offset = torch.cumsum(curr_batch, dim=0).int()
#         next_batch = torch.full((next_coords.shape[0],), next_coords.shape[1], dtype=torch.int64, device=device)
#         next_offset = torch.cumsum(next_batch, dim=0).int()
#         idx, _ = pointops.knn_query(diff_res_num_neighbor_list[i], curr_coords.reshape(-1, 3), curr_offset, next_coords.reshape(-1, 3), next_offset) 
#         idx = idx % curr_coords.shape[1]
#         index = idx.view(B, -1, diff_res_num_neighbor_list[i]).type(torch.int64)
#         index_list.append(index)

#     return index_list, coords_list

@torch.no_grad()
def generate_img_index_and_coords_v1(batch_size, resolution_list, device):
    # resolution_list: [[112, 112],[56, 56],[28, 28],[14, 14], [7, 7], [1, 1]]
    img_H, img_W = resolution_list[0]
    img_mesh_list = []
    for i in range(len(resolution_list)):
        curr_img_H, curr_img_W = resolution_list[i]
        curr_img_H_mesh = torch.arange(0, curr_img_H, device=device)
        curr_img_W_mesh = torch.arange(0, curr_img_W, device=device)
        img_2_curr_img_scale_H = img_H * 1.0 / curr_img_H
        img_2_curr_img_scale_W = img_W * 1.0 / curr_img_W
        delta_H = img_2_curr_img_scale_H / 2 - 0.5
        delta_W = img_2_curr_img_scale_W / 2 - 0.5
        curr_img_H_mesh = curr_img_H_mesh * img_2_curr_img_scale_H + delta_H
        curr_img_W_mesh = curr_img_W_mesh * img_2_curr_img_scale_W + delta_W
        curr_img_H_mesh = curr_img_H_mesh.unsqueeze(1).unsqueeze(2).expand(-1, curr_img_W, -1)
        curr_img_W_mesh = curr_img_W_mesh.unsqueeze(0).unsqueeze(2).expand(curr_img_H, -1, -1)
        curr_img_mesh = torch.cat((curr_img_H_mesh, curr_img_W_mesh), dim=-1) # Produces (curr_img_H, curr_img_W, 2)
        curr_img_mesh = curr_img_mesh.flatten(0, 1).unsqueeze(0).expand(batch_size, -1, -1) # Produces (B, curr_img_H * curr_img_W, 2) tensor
        img_mesh_list.append(curr_img_mesh)
    
    index_list = []
    for i in range(len(img_mesh_list) - 1):
        curr_img_2_next_img_dist = torch.cdist(img_mesh_list[i], img_mesh_list[i+1]) # Produces (B, curr_img_H * curr_img_W, next_img_H * next_img_W) tensor
        index = torch.argmin(curr_img_2_next_img_dist, dim=-1, keepdim=False) # Produces (B, curr_img_H * curr_img_W) tensor
        index_list.append(index)
    index_list.append(torch.zeros((batch_size, 1), dtype=torch.int64, device=device))

    return index_list, img_mesh_list

@torch.no_grad()
def generate_img_index_and_knn_and_coords_v1(batch_size, resolution_list, device):
    # resolution_list: [[112, 112],[56, 56],[28, 28],[14, 14], [7, 7], [1, 1]]
    img_H, img_W = resolution_list[0]
    img_mesh_list = []
    for i in range(len(resolution_list)):
        curr_img_H, curr_img_W = resolution_list[i]
        curr_img_H_mesh = torch.arange(0, curr_img_H, device=device)
        curr_img_W_mesh = torch.arange(0, curr_img_W, device=device)
        img_2_curr_img_scale_H = img_H * 1.0 / curr_img_H
        img_2_curr_img_scale_W = img_W * 1.0 / curr_img_W
        delta_H = img_2_curr_img_scale_H / 2 - 0.5
        delta_W = img_2_curr_img_scale_W / 2 - 0.5
        curr_img_H_mesh = curr_img_H_mesh * img_2_curr_img_scale_H + delta_H
        curr_img_W_mesh = curr_img_W_mesh * img_2_curr_img_scale_W + delta_W
        curr_img_H_mesh = curr_img_H_mesh.unsqueeze(1).unsqueeze(2).expand(-1, curr_img_W, -1)
        curr_img_W_mesh = curr_img_W_mesh.unsqueeze(0).unsqueeze(2).expand(curr_img_H, -1, -1)
        curr_img_mesh = torch.cat((curr_img_H_mesh, curr_img_W_mesh), dim=-1) # Produces (curr_img_H, curr_img_W, 2)
        curr_img_mesh = curr_img_mesh.flatten(0, 1).unsqueeze(0).expand(batch_size, -1, -1) # Produces (B, curr_img_H * curr_img_W, 2) tensor
        img_mesh_list.append(curr_img_mesh)
    
    index_list = []
    knn_index_list = []
    for i in range(len(img_mesh_list) - 1):
        curr_img_2_next_img_dist = torch.cdist(img_mesh_list[i], img_mesh_list[i+1])
        index = torch.argmin(curr_img_2_next_img_dist, dim=-1, keepdim=False) # Produces (B, curr_img_H * curr_img_W) tensor
        index_list.append(index)
        num_k = (img_mesh_list[i].shape[0] // img_mesh_list[i+1].shape[0]) * (img_mesh_list[i].shape[1] // img_mesh_list[i+1].shape[1])
        _, knn_index = torch.topk(input=curr_img_2_next_img_dist, 
                     k=num_k, 
                     dim=-2, 
                     largest=False, 
                     sorted=False) # Produces (B, num_k, next_img_H * next_img_W) tensor
        knn_index_list.append(knn_index)
    index_list.append(torch.zeros((batch_size, 1), dtype=torch.int64, device=device))

    return index_list, knn_index_list, img_mesh_list

@torch.no_grad()
def generate_img_index_and_knn_and_coords_v2(batch_size, resolution_list, device, num_neighbor_list):
    # resolution_list: [[112, 112],[56, 56],[28, 28],[14, 14], [7, 7], [1, 1]]
    # num_neighbor_list: [25, 25, 25, 25, 25]
    num_neighbor_list_inuse = num_neighbor_list[::-1]
    img_H, img_W = resolution_list[0]
    for resolution in resolution_list:
        assert resolution[0] == resolution[1], "Only support square image"
    for i in range(len(resolution_list) - 1):
        assert resolution_list[i][0] % resolution_list[i+1][0] == 0, "Each resolution should be divisible by the next resolution"
    for num_neighbor in num_neighbor_list_inuse:
        num_neighbor_sqrt = int(math.sqrt(num_neighbor))
        assert num_neighbor_sqrt * num_neighbor_sqrt == num_neighbor, "Only support square number of neighbors"
    img_mesh_list = []
    for i in range(len(resolution_list)):
        curr_img_H, curr_img_W = resolution_list[i]
        curr_img_H_mesh = torch.arange(0, curr_img_H, device=device)
        curr_img_W_mesh = torch.arange(0, curr_img_W, device=device)
        img_2_curr_img_scale_H = img_H * 1.0 / curr_img_H
        img_2_curr_img_scale_W = img_W * 1.0 / curr_img_W
        delta_H = img_2_curr_img_scale_H / 2 - 0.5
        delta_W = img_2_curr_img_scale_W / 2 - 0.5
        curr_img_H_mesh = curr_img_H_mesh * img_2_curr_img_scale_H + delta_H
        curr_img_W_mesh = curr_img_W_mesh * img_2_curr_img_scale_W + delta_W
        curr_img_H_mesh = curr_img_H_mesh.unsqueeze(1).unsqueeze(2).expand(-1, curr_img_W, -1)
        curr_img_W_mesh = curr_img_W_mesh.unsqueeze(0).unsqueeze(2).expand(curr_img_H, -1, -1)
        curr_img_mesh = torch.cat((curr_img_H_mesh, curr_img_W_mesh), dim=-1) # Produces (curr_img_H, curr_img_W, 2)
        curr_img_mesh = curr_img_mesh.flatten(0, 1).unsqueeze(0).expand(batch_size, -1, -1) # Produces (B, curr_img_H * curr_img_W, 2) tensor
        img_mesh_list.append(curr_img_mesh)
    
    index_list = []
    knn_index_list_1 = []
    knn_index_list_2 = []
    device_id = device.index
    for i in range(len(img_mesh_list) - 1):
        curr_img_mesh_lazy = LazyTensor(img_mesh_list[i][0:1, ...].unsqueeze(-2)) # (1, curr_img_H * curr_img_W, 1, 2)
        next_img_mesh_lazy = LazyTensor(img_mesh_list[i+1][0:1, ...].unsqueeze(-3)) # (1, 1, next_img_H * next_img_W, 2)
        curr_img_2_next_img_dist_lazy = (curr_img_mesh_lazy - next_img_mesh_lazy).norm2() # (1, curr_img_H * curr_img_W, next_img_H * next_img_W)
        _, index = curr_img_2_next_img_dist_lazy.argKmin(1, dim=2, device_id=device_id) # Produces (1, curr_img_H * curr_img_W, 1)
        index_list.append(index.squeeze(-1).expand(batch_size, -1)) # Produces (B, curr_img_H * curr_img_W) tensor
        num_k_1 = img_mesh_list[i].shape[1] // img_mesh_list[i+1].shape[1]
        _, knn_index_1 = curr_img_2_next_img_dist_lazy.argKmin(num_k_1, dim=1, device_id=device_id) # Produces (1, num_k_1, next_img_H * next_img_W) tensor
        knn_index_list_1.append(knn_index_1.expand(batch_size, -1, -1)) # Produces (B, num_k_1, next_img_H * next_img_W) tensor
        num_k_2 = (int(math.sqrt(img_mesh_list[i].shape[1] // img_mesh_list[i+1].shape[1])) - 1 + int(math.sqrt(num_neighbor_list_inuse[i])))**2
        num_k_2 = min(num_k_2, img_mesh_list[i].shape[1])
        _, knn_index_2 = curr_img_2_next_img_dist_lazy.argKmin(num_k_2, dim=1, device_id=device_id) # Produces (1, num_k_2, next_img_H * next_img_W) tensor
        knn_index_list_2.append(knn_index_2.expand(batch_size, -1, -1)) # Produces (B, num_k_2, next_img_H * next_img_W) tensor
    index_list.append(torch.zeros((batch_size, 1), dtype=torch.int64, device=device))

    return index_list, knn_index_list_1, knn_index_list_2, img_mesh_list

@torch.no_grad()
def generate_img_index_and_knn_and_coords_v3(batch_size, resolution_list, device):
    # resolution_list: [[112, 112],[56, 56],[28, 28],[14, 14], [7, 7], [1, 1]]
    img_H, img_W = resolution_list[0]
    for resolution in resolution_list:
        assert resolution[0] == resolution[1], "Only support square image"
    for i in range(len(resolution_list) - 1):
        assert resolution_list[i][0] % resolution_list[i+1][0] == 0, "Each resolution should be divisible by the next resolution"
    img_mesh_list = []
    for i in range(len(resolution_list)):
        curr_img_H, curr_img_W = resolution_list[i]
        curr_img_H_mesh = torch.arange(0, curr_img_H, device=device)
        curr_img_W_mesh = torch.arange(0, curr_img_W, device=device)
        img_2_curr_img_scale_H = img_H * 1.0 / curr_img_H
        img_2_curr_img_scale_W = img_W * 1.0 / curr_img_W
        delta_H = img_2_curr_img_scale_H / 2 - 0.5
        delta_W = img_2_curr_img_scale_W / 2 - 0.5
        curr_img_H_mesh = curr_img_H_mesh * img_2_curr_img_scale_H + delta_H
        curr_img_W_mesh = curr_img_W_mesh * img_2_curr_img_scale_W + delta_W
        curr_img_H_mesh = curr_img_H_mesh.unsqueeze(1).unsqueeze(2).expand(-1, curr_img_W, -1)
        curr_img_W_mesh = curr_img_W_mesh.unsqueeze(0).unsqueeze(2).expand(curr_img_H, -1, -1)
        curr_img_mesh = torch.cat((curr_img_H_mesh, curr_img_W_mesh), dim=-1) # Produces (curr_img_H, curr_img_W, 2)
        curr_img_mesh = curr_img_mesh.flatten(0, 1).unsqueeze(0).expand(batch_size, -1, -1) # Produces (B, curr_img_H * curr_img_W, 2) tensor
        img_mesh_list.append(curr_img_mesh)
    
    index_list = []
    knn_index_list = []
    device_id = device.index
    for i in range(len(img_mesh_list) - 1):
        curr_img_mesh_lazy = LazyTensor(img_mesh_list[i][0:1, ...].contiguous().unsqueeze(-2)) # (1, curr_img_H * curr_img_W, 1, 2)
        next_img_mesh_lazy = LazyTensor(img_mesh_list[i+1][0:1, ...].contiguous().unsqueeze(-3)) # (1, 1, next_img_H * next_img_W, 2)
        curr_img_2_next_img_dist_lazy = (curr_img_mesh_lazy - next_img_mesh_lazy).norm2() # (1, curr_img_H * curr_img_W, next_img_H * next_img_W)
        _, index = curr_img_2_next_img_dist_lazy.Kmin_argKmin(1, dim=2, device_id=device_id) # Produces (1, curr_img_H * curr_img_W, 1)
        index_list.append(index.squeeze(-1).expand(batch_size, -1)) # Produces (B, curr_img_H * curr_img_W) tensor
        num_k = img_mesh_list[i].shape[1] // img_mesh_list[i+1].shape[1]
        
        _, knn_index = keops_knn(device, img_mesh_list[i+1][0:1, ...].contiguous(), img_mesh_list[i][0:1, ...].contiguous(), num_k) # Produces (1, next_img_H * next_img_W, num_k) tensor
        knn_index_list.append(knn_index.expand(batch_size, -1, -1).permute(0, 2, 1)) # Produces (B, num_k, next_img_H * next_img_W) tensor
    index_list.append(torch.zeros((batch_size, 1), dtype=torch.int64, device=device))

    return index_list, knn_index_list, img_mesh_list


@torch.no_grad()
def generate_img_meshgrid(batch_size, img_size, original_img_size, device):
    img_H, img_W = original_img_size

    curr_img_H, curr_img_W = img_size
    curr_img_H_mesh = torch.arange(0, curr_img_H, device=device)
    curr_img_W_mesh = torch.arange(0, curr_img_W, device=device)
    img_2_curr_img_scale_H = img_H * 1.0 / curr_img_H
    img_2_curr_img_scale_W = img_W * 1.0 / curr_img_W
    delta_H = img_2_curr_img_scale_H / 2 - 0.5
    delta_W = img_2_curr_img_scale_W / 2 - 0.5
    curr_img_H_mesh = curr_img_H_mesh * img_2_curr_img_scale_H + delta_H
    curr_img_W_mesh = curr_img_W_mesh * img_2_curr_img_scale_W + delta_W
    curr_img_H_mesh = curr_img_H_mesh.unsqueeze(1).unsqueeze(2).expand(-1, curr_img_W, -1)
    curr_img_W_mesh = curr_img_W_mesh.unsqueeze(0).unsqueeze(2).expand(curr_img_H, -1, -1)
    curr_img_mesh = torch.cat((curr_img_H_mesh, curr_img_W_mesh), dim=-1) # Produces (curr_img_H, curr_img_W, 2)
    curr_img_mesh = curr_img_mesh.flatten(0, 1).unsqueeze(0).expand(batch_size, -1, -1) # Produces (B, curr_img_H * curr_img_W, 2) tensor

    return curr_img_mesh


def inverse_mapping(A, N):
    B = torch.full((A.size(0), N), A.size(1), dtype=torch.long, device=A.device)
    B.scatter_(1, A, torch.arange(A.size(1), device=A.device).unsqueeze(0).expand_as(A))
    return B