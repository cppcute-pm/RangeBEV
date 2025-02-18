import torch
from torch import Tensor
from typing import Tuple
from pykeops.torch import LazyTensor
import torch_scatter
from pointnet2_ops import pointnet2_utils
from torch import tensor
import matplotlib.pyplot as plt
import numpy as np
import pointops
import torch.nn.functional as F
import copy

def camera_matrix_scaling(K: torch.tensor, sx: float, sy: float):
    K_scale = copy.deepcopy(K)
    K_scale[..., 0, 2] *= sx
    K_scale[..., 0, 0] *= sx
    K_scale[..., 1, 2] *= sy
    K_scale[..., 1, 1] *= sy
    return K_scale

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


def knn(
    device,
    q_points: Tensor,
    s_points: Tensor,
    k: int,
    dilation: int = 1,
    distance_limit: float = None,
    return_distance: bool = False,
    remove_nearest: bool = False,
    transposed: bool = False,
    padding_mode: str = "nearest",
    padding_value: float = 1e10,
    squeeze: bool = False,
):
    """Compute the kNNs of the points in `q_points` from the points in `s_points`.

    Use KeOps to accelerate computation.

    Args:
        s_points (Tensor): coordinates of the support points, (*, C, N) or (*, N, C).
        q_points (Tensor): coordinates of the query points, (*, C, M) or (*, M, C).
        k (int): number of nearest neighbors to compute.
        dilation (int): dilation for dilated knn.
        distance_limit (float=None): if further than this radius, the neighbors are ignored according to `padding_mode`.
        return_distance (bool=False): whether return distances.
        remove_nearest (bool=True) whether remove the nearest neighbor (itself).
        transposed (bool=False): if True, the points shape is (*, C, N).
        padding_mode (str='nearest'): the padding mode for neighbors further than distance radius. ('nearest', 'empty').
        padding_value (float=1e10): the value for padding.
        squeeze (bool=False): if True, the distance and the indices are squeezed if k=1.

    Returns:
        knn_distances (Tensor): The distances of the kNNs, (*, M, k).
        knn_indices (LongTensor): The indices of the kNNs, (*, M, k).
    """
    if transposed:
        q_points = q_points.transpose(-1, -2)  # (*, C, N) -> (*, N, C)
        s_points = s_points.transpose(-1, -2)  # (*, C, M) -> (*, M, C)
    q_points = q_points.contiguous()
    s_points = s_points.contiguous()

    num_s_points = s_points.shape[-2]

    dilated_k = (k - 1) * dilation + 1
    if remove_nearest:
        dilated_k += 1
    final_k = min(dilated_k, num_s_points)

    knn_distances, knn_indices = keops_knn(device, q_points, s_points, final_k)  # (*, N, k)

    if remove_nearest:
        knn_distances = knn_distances[..., 1:]
        knn_indices = knn_indices[..., 1:]

    if dilation > 1:
        knn_distances = knn_distances[..., ::dilation]
        knn_indices = knn_indices[..., ::dilation]

    knn_distances = knn_distances.contiguous()
    knn_indices = knn_indices.contiguous()

    if distance_limit is not None:
        assert padding_mode in ["nearest", "empty"]
        knn_masks = torch.ge(knn_distances, distance_limit)
        if padding_mode == "nearest":
            knn_distances = torch.where(knn_masks, knn_distances[..., :1], knn_distances)
            knn_indices = torch.where(knn_masks, knn_indices[..., :1], knn_indices)
        else:
            knn_distances[knn_masks] = padding_value
            knn_indices[knn_masks] = num_s_points

    if squeeze and k == 1:
        knn_distances = knn_distances.squeeze(-1)
        knn_indices = knn_indices.squeeze(-1)

    if return_distance:
        return knn_distances, knn_indices

    return knn_indices


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

@torch.no_grad()
def generate_overlap_ratio(device, result, model_cfgs, dataset_cfgs, dataset=None):
    
    overlap_ratio = None
    overlap_ratio_list = []
    pose_dist_flag = dataset_cfgs.use_overlap_ratio and (dataset_cfgs.overlap_ratio_type == 'pose_dist_sim' or dataset_cfgs.overlap_ratio_type == 'all_sim_mixed')
    project_flag = dataset_cfgs.use_overlap_ratio and (dataset_cfgs.overlap_ratio_type == 'project_sim' or dataset_cfgs.overlap_ratio_type == 'all_sim_mixed')
    point_knn_flag = dataset_cfgs.use_overlap_ratio and (dataset_cfgs.overlap_ratio_type == 'point_knn_sim' or dataset_cfgs.overlap_ratio_type == 'all_sim_mixed')
    area_overlap_ratio_flag = dataset_cfgs.use_overlap_ratio and dataset_cfgs.overlap_ratio_type == 'area_overlap'
    exp_dist_flag = dataset_cfgs.use_overlap_ratio and dataset_cfgs.overlap_ratio_type == 'exp_dist'
    exp_dist_v2_flag = dataset_cfgs.use_overlap_ratio and dataset_cfgs.overlap_ratio_type == 'exp_dist_v2'
    if pose_dist_flag:
        if isinstance(result['labels'], list):
            curr_seq_ID = result['labels'][0][0]
            for seq_ID, _ in result['labels']:
                if seq_ID != curr_seq_ID:
                    raise ValueError('The seq_ID is not consistent.')
            labels = torch.tensor([frame_ID for seq_ID, frame_ID in result['labels']], dtype=torch.int64)

            if 'dist_caculation_type' in dataset_cfgs.keys() and dataset_cfgs.dist_caculation_type == 'all_coords_L2_mean':
                query_UTM_coord = dataset.UTM_coord_tensor[curr_seq_ID][:, labels, :]
                query_UTM_coord = query_UTM_coord.to(device)
                query_2_query_dist = torch.cdist(query_UTM_coord, query_UTM_coord, p=2.0).mean(dim=0, keepdim=False) # (B, B)
            else:
                query_UTM_coord = dataset.UTM_coord_tensor[curr_seq_ID][labels]
                query_UTM_coord = query_UTM_coord.to(device)
                query_2_query_dist = torch.cdist(query_UTM_coord, query_UTM_coord, p=2.0) # (B, B)
            
            if 'overlap_ratio_cfgs' in dataset_cfgs.keys() and dataset_cfgs.overlap_ratio_cfgs.reverse:
                query_UTM_coord_reverse = dataset.UTM_coord_tensor_reverse[curr_seq_ID][:, labels, :]
                query_UTM_coord_reverse = query_UTM_coord_reverse.to(device)
                query_2_query_dist_reverse = torch.cdist(query_UTM_coord, query_UTM_coord_reverse, p=2.0).mean(dim=0, keepdim=False) # (B, B)
                query_2_query_dist = torch.minimum(query_2_query_dist, query_2_query_dist_reverse)
            
            zero_mask = torch.ge(query_2_query_dist, dataset_cfgs.pose_dist_threshold)
            overlap_ratio = (dataset_cfgs.pose_dist_threshold - query_2_query_dist) * 1.0 / dataset_cfgs.pose_dist_threshold
            overlap_ratio[zero_mask] = 0.0
            overlap_ratio_list.append(overlap_ratio)

        elif isinstance(result['labels'], np.ndarray):
            labels = torch.tensor(result['labels'], dtype=torch.int64)
            if 'dist_caculation_type' in dataset_cfgs.keys() and dataset_cfgs.dist_caculation_type == 'all_coords_L2_mean':
                query_UTM_coord = dataset.UTM_coord_tensor[:, labels, :]
                query_UTM_coord = query_UTM_coord.to(device)
                query_2_query_dist = torch.cdist(query_UTM_coord, query_UTM_coord, p=2.0).mean(dim=0, keepdim=False) # (B, B)
            else:
                query_UTM_coord = dataset.UTM_coord_tensor[labels]
                query_UTM_coord = query_UTM_coord.to(device)
                query_2_query_dist = torch.cdist(query_UTM_coord, query_UTM_coord, p=2.0) # (B, B)
            zero_mask = torch.ge(query_2_query_dist, dataset_cfgs.pose_dist_threshold)
            overlap_ratio = (dataset_cfgs.pose_dist_threshold - query_2_query_dist) * 1.0 / dataset_cfgs.pose_dist_threshold
            overlap_ratio[zero_mask] = 0.0
            overlap_ratio_list.append(overlap_ratio)
        else:
            raise ValueError('The type of result[\'labels\'] is not supported.')
    
    if exp_dist_flag:
        if isinstance(result['labels'], list):
            curr_seq_ID = result['labels'][0][0]
            labels = torch.tensor([frame_ID for seq_ID, frame_ID in result['labels']], dtype=torch.int64)
            curr_seq_UTM_coord = dataset.UTM_coord_tensor[curr_seq_ID]
            query_UTM_coord = curr_seq_UTM_coord[labels]
            query_UTM_coord = query_UTM_coord.to(device)
            query_2_query_dist = torch.cdist(query_UTM_coord[:, :2], query_UTM_coord[:, :2], p=2.0) # (B, B)
            query_2_query_sim = torch.exp(-dataset.exp_scale * query_2_query_dist)
            overlap_ratio = query_2_query_sim
            overlap_ratio_list.append(query_2_query_sim)
        elif isinstance(result['labels'], np.ndarray): 
            labels = torch.tensor(result['labels'], dtype=torch.int64)
            query_UTM_coord = dataset.UTM_coord_tensor[labels]
            query_UTM_coord = query_UTM_coord.to(device)
            query_2_query_dist = torch.cdist(query_UTM_coord[:, :2], query_UTM_coord[:, :2], p=2.0) # (B, B)
            query_2_query_sim = torch.exp(-dataset.exp_scale * query_2_query_dist)
            overlap_ratio = query_2_query_sim
            overlap_ratio_list.append(query_2_query_sim)
        else:
            raise ValueError('The type of result[\'labels\'] is not supported.')
    
    if exp_dist_v2_flag:
        labels = torch.tensor(result['labels'], dtype=torch.int64)
        query_UTM_coord = dataset.UTM_coord_tensor[labels]
        query_UTM_coord = query_UTM_coord.to(device)
        query_2_query_dist = torch.cdist(query_UTM_coord[:, 2:], query_UTM_coord[:, 2:], p=2.0) # (B, B)
        query_2_query_sim = torch.exp(-dataset.exp_scale * query_2_query_dist)
        overlap_ratio = query_2_query_sim
        overlap_ratio_list.append(query_2_query_sim)

    if area_overlap_ratio_flag:
        if isinstance(result['labels'], list):
            curr_seq_ID = result['labels'][0][0]
            labels = torch.tensor([frame_ID for seq_ID, frame_ID in result['labels']], dtype=torch.int64)
            labels_numpy = labels.numpy()
            curr_seq_area_overlap = dataset.area_overlap[curr_seq_ID]
            overlap_ratio = curr_seq_area_overlap[labels_numpy, :][:, labels_numpy].cuda()
            overlap_ratio_list.append(overlap_ratio)
        elif isinstance(result['labels'], np.ndarray):
            labels_numpy = result['labels']
            overlap_ratio = torch.tensor(dataset.area_overlap[labels_numpy, :][:, labels_numpy], dtype=torch.float32, device=device)
            overlap_ratio_list.append(overlap_ratio)
        else:
            raise ValueError('The type of result[\'labels\'] is not supported.')
    
    if project_flag:
        # v1: don't consider the distance between the pc UTM and the camera UTM
        point_project_threshold = dataset_cfgs.point_project_threshold
        P_camera_lidar = torch.matmul(torch.linalg.inv(result['image_poses'].to(device)).unsqueeze(0), result['cloud_poses_original'].to(device).unsqueeze(1)) # Produces (B, B, 4, 4) tensor
        clouds = result['clouds_original'].to(device) # Produces (B, N, 3) tensor
        N = clouds.shape[1]
        clouds_to_mult = clouds.unsqueeze(1) # Produces (B, 1, N, 3) tensor
        clouds_to_mult = torch.cat([clouds_to_mult, torch.ones_like(clouds_to_mult[..., :1])], dim=-1) # Produces (B, 1, N, 4) tensor
        clouds_in_camera = torch.matmul(clouds_to_mult, P_camera_lidar.permute(0, 1, 3, 2)) # Produces (B, B, N, 4) tensor
        clouds_in_camera = clouds_in_camera[..., :3]

        mask_0 = torch.ge(clouds_in_camera[..., 2], 0.0) # Produces (B, B, N) tensor
        # mask_1 = result['cloud_remove_masks'].unsqueeze(1).expand_as(mask_0).to(device) # Produces (B, B, N) tensor
        original_2_downsampled_indices_inuse = torch.gather(result['cloud_shuffle_indices'].to(device),
                                                            dim=-1,
                                                            index=result["original_2_downsampled_indices"].to(device)) # Produces (B, N) tensor

        mask_1 = torch.gather(result['cloud_remove_masks'].to(device),
                            dim=-1,
                            index=original_2_downsampled_indices_inuse) # Produces (B, N) tensor
        mask_2 = torch.lt(clouds_in_camera[..., 2], point_project_threshold) # Produces (B, B, N) tensor TODO this should be a parameter
        image_intrinscs_to_mult = result['image_intrinscs'].to(device).unsqueeze(0) # Produces (1, B, 3, 3) tensor
        clouds_in_image = torch.matmul(clouds_in_camera, image_intrinscs_to_mult.permute(0, 1, 3, 2)) # Produces (B, B, N, 3) tensor
        clouds_in_plane = clouds_in_image[..., :2] / clouds_in_image[..., 2:] # Produces (B, B, N, 2) tensor
        B, _, img_H, img_W = result['images'].shape
        mask_3 = torch.ge(clouds_in_plane[..., 0], 0.0) & \
           torch.lt(clouds_in_plane[..., 0], float(img_W)) & \
           torch.ge(clouds_in_plane[..., 1], 0.0) & \
           torch.lt(clouds_in_plane[..., 1], float(img_H)) # Produces (B, B, N) tensor
        mask = mask_0 & (~mask_1) & mask_2 & mask_3
        mask_num = torch.count_nonzero(~mask, dim=-1) # Produces (B, B) tensor
        
        overlap_points_num = torch.count_nonzero(mask, dim=-1) # Produces (B, B) tensor
        clouds_in_plane_pixels = torch.floor(clouds_in_plane).type(torch.int64) # Produces (B, B, N, 2) tensor
        clouds_in_plane_pixels_flattened = clouds_in_plane_pixels[:, :, :, 1] * img_W + clouds_in_plane_pixels[:, :, :, 0] # Produces (B, B, N) tensor
        clouds_in_plane_pixels_flattened.masked_fill_(~mask, img_H * img_W - 1) # Produces (B, B, N) tensor
        img_num_pt = torch_scatter.scatter_sum(torch.ones_like(clouds_in_plane_pixels_flattened, dtype=torch.int32), clouds_in_plane_pixels_flattened, dim=-1, dim_size=img_H * img_W) # Produces (B, B, H*W) tensor
        img_num_pt[:, :, -1] -= mask_num # Produces (B, B, H*W) tensor
        img_num_pixel = torch.gt(img_num_pt, 0).type(torch.int32) # Produces (B, B, H*W) tensor
        overlap_pixels_num = torch.count_nonzero(img_num_pixel, dim=-1) # Produces (B, B) tensor 

        # mask2fill = torch.tensor([img_H, img_W], dtype=clouds_in_plane_pixels.dtype, device=clouds_in_plane_pixels.device).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, B, N, -1) # Produces (B, B, N, 2) tensor
        # clouds_in_plane_pixels[mask] = mask2fill[mask] # Produces (B, B, N, 2) tensor
        # image_mesh = torch.zeros((B, B, img_H + 1, img_W + 1), dtype=torch.int64, device=clouds_in_plane_pixels.device) # Produces (B, B, H+1, W+1) tensor
        # image_mesh[clouds_in_plane_pixels[..., 1], clouds_in_plane_pixels[..., 0]] = 1 # Produces (B, B, H+1, W+1) tensor
        # overlap_pixels_num = torch.count_nonzero(image_mesh[..., :-1, :-1], dim=(-1, -2)) # Produces (B, B) tensor
        image_overlap_ratio = overlap_pixels_num * 1.0 / (img_H * img_W) # Produces (B, B) tensor
        pc_overlap_ratio = overlap_points_num * 1.0 / N # Produces (B, B) tensor
        if dataset_cfgs.project_ratio_strategy == 'mean':
            overlap_ratio = (image_overlap_ratio + pc_overlap_ratio) / 2
        elif dataset_cfgs.project_ratio_strategy == 'min':
            overlap_ratio = torch.minimum(image_overlap_ratio, pc_overlap_ratio)
        elif dataset_cfgs.project_ratio_strategy == 'max':
            overlap_ratio = torch.maximum(image_overlap_ratio, pc_overlap_ratio)
        overlap_ratio = (overlap_ratio + overlap_ratio.permute(1, 0)) / 2
        if dataset_cfgs.project_relative_strategy:
            overlap_ratio = overlap_ratio / (torch.max(overlap_ratio) + 1e-6)
        overlap_ratio_list.append(overlap_ratio)
    if point_knn_flag:
        overlap_knn_dis_threshold = dataset_cfgs.overlap_knn_dis_threshold
        P_lidar2_lidar1 = torch.matmul(torch.linalg.inv(result['cloud_poses'].to(device)).unsqueeze(0), result['cloud_poses'].to(device).unsqueeze(1)) # Produces (B, B, 4, 4) tensor
        clouds = torch.stack(result['clouds'], dim=0).to(device) # Produces (B, N, 3) tensor
        N = clouds.shape[1]
        clouds1_in_lidar1 = clouds.unsqueeze(1) # Produces (B, 1, N, 3) tensor
        clouds1_in_lidar1 = torch.cat([clouds1_in_lidar1, torch.ones_like(clouds1_in_lidar1[..., :1])], dim=-1) # Produces (B, 1, N, 4) tensor
        clouds1_in_lidar2 = (torch.matmul(clouds1_in_lidar1, P_lidar2_lidar1.permute(0, 1, 3, 2)))[..., :3] # Produces (B, B, N, 3) tensor
        clouds2_in_lidar2 = clouds.unsqueeze(0) # Produces (1, B, N, 3) tensor
        clouds1to2_knn_indices = knn(
            device=device,
            q_points=clouds1_in_lidar2, 
            s_points=clouds2_in_lidar2, 
            k=1, 
            dilation=1, 
            distance_limit=overlap_knn_dis_threshold, # TODO: need test
            return_distance=False,
            remove_nearest=False,
            transposed=False,
            padding_mode='empty',
            padding_value=1e10,
            squeeze=False) # Produces (B, B, N, 1) tensor
        clouds2to1_knn_indices = knn(
            device=device,
            q_points=clouds2_in_lidar2,
            s_points=clouds1_in_lidar2,
            k=1,
            dilation=1,
            distance_limit=overlap_knn_dis_threshold, # TODO: need test
            return_distance=False,
            remove_nearest=False,
            transposed=False,
            padding_mode='empty',
            padding_value=1e10,
            squeeze=False)

        if dataset_cfgs.point_knn_ratio_strategy == "mutualnn":
            clouds1to2_knn_indices = torch.cat((clouds1to2_knn_indices.squeeze(-1), torch.full_like(clouds1to2_knn_indices[:, :, -1:, 0], N)), dim=-1) # Produces (B, B, N + 1) tensor
            clouds2to1_knn_indices = torch.cat((clouds2to1_knn_indices.squeeze(-1), torch.full_like(clouds2to1_knn_indices[:, :, -1:, 0], N)), dim=-1) # Produces (B, B, N + 1) tensor
            batch_indices_1 = torch.arange(clouds1to2_knn_indices.shape[0], device=device).unsqueeze(1).unsqueeze(2).expand_as(clouds1to2_knn_indices) # Produces (B, B, N + 1) tensor
            batch_indices_2 = torch.arange(clouds1to2_knn_indices.shape[1], device=device).unsqueeze(0).unsqueeze(2).expand_as(clouds1to2_knn_indices) # Produces (B, B, N + 1) tensor
            clouds1to2to1 = clouds2to1_knn_indices[batch_indices_1, batch_indices_2, clouds1to2_knn_indices] # Produces (B, B, N + 1) tensor
            equal_meshgrid = torch.arange(N, device=device).unsqueeze(0).unsqueeze(0).expand(clouds1to2to1.shape[0], clouds1to2to1.shape[1], -1) # Produces (B, B, N) tensor
            overlap_num = torch.count_nonzero(torch.eq(clouds1to2to1[:, :, :-1], equal_meshgrid), dim=-1) # Produces (B, B) tensor
            overlap_ratio = overlap_num * 1.0 / N # Produces (B, B) tensor
        else:
            clouds_1to2_overlap_num = torch.count_nonzero(torch.lt(clouds1to2_knn_indices, N), dim=(-1, -2)) # Produces (B, B) tensor
            clouds_2to1_overlap_num = torch.count_nonzero(torch.lt(clouds2to1_knn_indices, N), dim=(-1, -2)) # Produces (B, B) tensor
            clouds_1to2_overlap_ratio = clouds_1to2_overlap_num * 1.0 / N # Produces (B, B) tensor
            clouds_2to1_overlap_ratio = clouds_2to1_overlap_num * 1.0 / N # Produces (B, B) tensor
            if dataset_cfgs.point_knn_ratio_strategy == 'mean':
                overlap_ratio = (clouds_1to2_overlap_ratio + clouds_2to1_overlap_ratio) / 2
            elif dataset_cfgs.point_knn_ratio_strategy == 'min':
                overlap_ratio = torch.minimum(clouds_1to2_overlap_ratio, clouds_2to1_overlap_ratio)
            elif dataset_cfgs.point_knn_ratio_strategy == 'max':
                overlap_ratio = torch.maximum(clouds_1to2_overlap_ratio, clouds_2to1_overlap_ratio)
        overlap_ratio_list.append(overlap_ratio)
    
    if dataset_cfgs.use_overlap_ratio and dataset_cfgs.overlap_ratio_type == 'all_sim_mixed':
        if dataset_cfgs.all_sim_mixed_type == 'max':
            overlap_ratio, _ = torch.max(torch.stack(overlap_ratio_list, dim=-1), dim=-1, keepdim=False)
        elif dataset_cfgs.all_sim_mixed_type == 'min':
            overlap_ratio, _ = torch.min(torch.stack(overlap_ratio_list, dim=-1), dim=-1, keepdim=False)
        elif dataset_cfgs.all_sim_mixed_type == 'mean':
            overlap_ratio = torch.mean(torch.stack(overlap_ratio_list, dim=-1), dim=-1, keepdim=False)
        elif dataset_cfgs.all_sim_mixed_type == 'weighted_mean':
            overlap_ratio = torch.stack(overlap_ratio_list, dim=-1)
            all_sim_mixed_weight = torch.tensor(dataset_cfgs.all_sim_mixed_weight, dtype=overlap_ratio.dtype, device=overlap_ratio.device).unsqueeze(0).unsqueeze(0)
            overlap_ratio = overlap_ratio * all_sim_mixed_weight
            overlap_ratio = torch.sum(overlap_ratio, dim=-1, keepdim=False)
    return overlap_ratio

@torch.no_grad()
def generate_overlap_matrix(device, result, model_cfgs, dataset_cfgs):
    
    P_camera_lidar = torch.matmul(torch.linalg.inv(result['image_poses']).unsqueeze(0), result['cloud_poses'].unsqueeze(1)) # Produces (B, B, 4, 4) tensor
    P_camera_lidar = P_camera_lidar.to(device)
    clouds = torch.stack(result['clouds'], dim=0) # Produces (B, N, 3) tensor
    clouds = clouds.to(device)
    N = clouds.shape[1]
    clouds_to_mult = clouds.unsqueeze(1) # Produces (B, 1, N, 3) tensor
    clouds_to_mult = torch.cat([clouds_to_mult, torch.ones_like(clouds_to_mult[..., :1])], dim=-1) # Produces (B, 1, N, 4) tensor
    clouds_in_camera = torch.matmul(clouds_to_mult, P_camera_lidar.permute(0, 1, 3, 2)) # Produces (B, B, N, 4) tensor
    clouds_in_camera = clouds_in_camera[..., :3]

    mask_0 = torch.ge(clouds_in_camera[..., 2], 0.0) # Produces (B, B, N) tensor
    mask_1 = result['cloud_remove_masks'].unsqueeze(1).expand_as(mask_0).to(device) # Produces (B, B, N) tensor
    mask_2 = torch.lt(clouds_in_camera[..., 2], 250.0) # Produces (B, B, N) tensor
    image_intrinscs_to_mult = result['image_intrinscs'].unsqueeze(0).to(device) # Produces (1, B, 3, 3) tensor
    clouds_in_image = torch.matmul(clouds_in_camera, image_intrinscs_to_mult.permute(0, 1, 3, 2)) # Produces (B, B, N, 3) tensor
    clouds_in_plane = clouds_in_image[..., :2] / clouds_in_image[..., 2:] # Produces (B, B, N, 2) tensor
    B, _, img_H, img_W = result['images'].shape
    mask_3 = torch.ge(clouds_in_plane[..., 0], 0.0) & \
           torch.lt(clouds_in_plane[..., 0], float(img_W)) & \
           torch.ge(clouds_in_plane[..., 1], 0.0) & \
           torch.lt(clouds_in_plane[..., 1], float(img_H)) # Produces (B, B, N) tensor
    
    # need more mask to filter out the points that are not in the image
    # 1、the max distance between the points and the center point of current camera frame : 100、150、200、250、300？
    mask = mask_0 & (~mask_1) & mask_2 & mask_3
    img_H1 = 64
    img_W1 = 64
    img_H2 = 32
    img_W2 = 32
    img_H3 = 16
    img_W3 = 16
    num_points1 = 64
    num_points2 = 32
    num_points3 = 16
    mask_num = torch.count_nonzero(~mask, dim=-1) # Produces (B, B) tensor
    
    

    # for i in range(B):
    #     for j in range(B):
    #         color = clouds_in_camera[i, j, :, 2]
    #         curr_mask = mask_0[i, j, :] & mask_2[i, j, :]
    #         color = color[curr_mask]
    #         # color[~mask_0[i, j, :]] = -500.0
    #         uv = clouds_in_plane[i, j, :, :]
    #         uv = uv[curr_mask, :]
    #         fig = plt.figure(figsize=(30.00, 30.00), dpi=100)
    #         ax = fig.add_subplot()
    #         ax.imshow(result['images'][j, :].permute(1, 2, 0))
    #         # ax.set_xlim(-500, 1000)
    #         # ax.set_ylim(1000, -500)
    #         ax.set_xlim(-250, 2500)
    #         ax.set_ylim(2500, -250)
    #         ax.scatter(uv[:, 0], uv[:, 1], c=color, marker=',', s=3, edgecolors='none', alpha=0.7, cmap='jet')
    #         ax.set_axis_off()
    #         plt.savefig(f'/home/test5/code_project/visualization/heihei_boreas_crop_{i}_{j}.jpg', bbox_inches='tight', pad_inches=0, dpi=200)


    clouds_in_plane_pixels = torch.floor(clouds_in_plane).type(torch.int64) # Produces (B, B, N, 2) tensor
    clouds_in_plane_pixels_flattened = clouds_in_plane_pixels[:, :, :, 1] * img_W + clouds_in_plane_pixels[:, :, :, 0] # Produces (B, B, N) tensor
    clouds_in_plane_pixels_flattened.masked_fill_(~mask, img_H * img_W - 1) # Produces (B, B, N) tensor
    img_num_pt = torch_scatter.scatter_sum(torch.ones_like(clouds_in_plane_pixels_flattened, dtype=torch.int32), clouds_in_plane_pixels_flattened, dim=-1, dim_size=img_H * img_W) # Produces (B, B, H*W) tensor
    img_num_pt[:, :, -1] -= mask_num # Produces (B, B, H*W) tensor
    img_num_pixel = torch.gt(img_num_pt, 0).type(torch.int32) # Produces (B, B, H*W) tensor    

    fps_idx = pointnet2_utils.furthest_point_sample(clouds, num_points1).long()  # [B, num_points1]
    node1 = index_points(clouds, fps_idx)  # [B, num_points1, 3]
    clouds2node1_dis = torch.cdist(clouds, node1) # Produces (B, N, num_points1) tensor
    _, clouds2node1_idx = torch.topk(clouds2node1_dis, k=1, dim=2, largest=False, sorted=False) # Produces (B, N, 1) tensor
    clouds2node1_idx = clouds2node1_idx.squeeze(-1) # Produces (B, N) tensor
    clouds2node1_idx_expand = clouds2node1_idx.unsqueeze(1).expand(-1, B, -1) # Produces (B, B, N) tensor
    node1_num_pt = torch_scatter.scatter_sum(torch.ones_like(clouds2node1_idx, dtype=torch.int32), clouds2node1_idx, dim=-1, dim_size=num_points1) # Produces (B, num_points1) tensor
    clouds2node1_idx_masked = clouds2node1_idx_expand.masked_fill(~mask, num_points1 - 1) # Produces (B, B, N) tensor
    node1_pixel_map = clouds_in_plane_pixels_flattened * num_points1 + clouds2node1_idx_masked # Produces (B, B, N) tensor 
    node1_pixel_map_num = torch_scatter.scatter_sum(torch.ones_like(node1_pixel_map, dtype=torch.int32), node1_pixel_map, dim=-1, dim_size=num_points1 * img_H * img_W) # Produces (B, B, img_H * img_W * num_points1) tensor
    node1_pixel_map_num[:, :, -1] -= mask_num # Produces (B, B, img_H * img_W * num_points1) tensor
    node1_image_map_num = node1_pixel_map_num.reshape(B, B, img_H * img_W, num_points1) # Produces (B, B, img_H * img_W, num_points1) tensor
    node1_image_map_pixel_num = torch.gt(node1_image_map_num, 0).type(torch.int32) # Produces (B, B, img_H * img_W, num_points1) tensor


    img_H_mesh = torch.arange(0, img_H, device=device).type(torch.float32)
    img_W_mesh = torch.arange(0, img_W, device=device).type(torch.float32)
    img_H_mesh = img_H_mesh.unsqueeze(1).unsqueeze(2).expand(-1, img_W, -1)
    img_W_mesh = img_W_mesh.unsqueeze(0).unsqueeze(2).expand(img_H, -1, -1)
    img_mesh = torch.cat((img_H_mesh, img_W_mesh), dim=-1) # Produces (img_H, img_W, 2)
    img_mesh = img_mesh.flatten(0, 1) # Produces (img_H * img_W, 2) tensor
    img1_H_mesh = torch.arange(0, img_H1, device=device)
    img1_W_mesh = torch.arange(0, img_W1, device=device)
    ori_2_1_scale_H = img_H * 1.0 / img_H1
    ori_2_1_scale_W = img_W * 1.0 / img_W1
    delta_H = ori_2_1_scale_H / 2 - 0.5
    delta_W = ori_2_1_scale_W / 2 - 0.5
    img1_H_mesh = img1_H_mesh * ori_2_1_scale_H + delta_H
    img1_W_mesh = img1_W_mesh * ori_2_1_scale_W + delta_W
    img1_H_mesh = img1_H_mesh.unsqueeze(1).unsqueeze(2).expand(-1, img_W1, -1) # Produces (img_H1, img_W1, 1) tensor
    img1_W_mesh = img1_W_mesh.unsqueeze(0).unsqueeze(2).expand(img_H1, -1, -1) # Produces (img_H1, img_W1, 1) tensor
    img_1_mesh = torch.cat((img1_H_mesh, img1_W_mesh), dim=-1) # Produces (img_H1, img_W1, 2) tensor
    img_1_mesh = img_1_mesh.flatten(0, 1) # Produces (img_H1 * img_W1, 2) tensor
    img2_H_mesh = torch.arange(0, img_H2, device=device)
    img2_W_mesh = torch.arange(0, img_W2, device=device)
    ori_2_2_scale_H = img_H * 1.0 / img_H2
    ori_2_2_scale_W = img_W * 1.0 / img_W2
    delta_H = ori_2_2_scale_H / 2 - 0.5
    delta_W = ori_2_2_scale_W / 2 - 0.5
    img2_H_mesh = img2_H_mesh * ori_2_2_scale_H + delta_H
    img2_W_mesh = img2_W_mesh * ori_2_2_scale_W + delta_W
    img2_H_mesh = img2_H_mesh.unsqueeze(1).unsqueeze(2).expand(-1, img_W2, -1) # Produces (img_H2, img_W2, 1) tensor
    img2_W_mesh = img2_W_mesh.unsqueeze(0).unsqueeze(2).expand(img_H2, -1, -1) # Produces (img_H2, img_W2, 1) tensor
    img_2_mesh = torch.cat((img2_H_mesh, img2_W_mesh), dim=-1) # Produces (img_H2, img_W2, 2) tensor
    img_2_mesh = img_2_mesh.flatten(0, 1)
    img3_H_mesh = torch.arange(0, img_H3, device=device)
    img3_W_mesh = torch.arange(0, img_W3, device=device)
    ori_2_3_scale_H = img_H * 1.0 / img_H3
    ori_2_3_scale_W = img_W * 1.0 / img_W3
    delta_H = ori_2_3_scale_H / 2 - 0.5
    delta_W = ori_2_3_scale_W / 2 - 0.5
    img3_H_mesh = img3_H_mesh * ori_2_3_scale_H + delta_H
    img3_W_mesh = img3_W_mesh * ori_2_3_scale_W + delta_W
    img3_H_mesh = img3_H_mesh.unsqueeze(1).unsqueeze(2).expand(-1, img_W3, -1) # Produces (img_H3, img_W3, 1) tensor
    img3_W_mesh = img3_W_mesh.unsqueeze(0).unsqueeze(2).expand(img_H3, -1, -1) # Produces (img_H3, img_W3, 1) tensor
    img_3_mesh = torch.cat((img3_H_mesh, img3_W_mesh), dim=-1) # Produces (img_H3, img_W3, 2) tensor
    img_3_mesh = img_3_mesh.flatten(0, 1)
    orimg2img1_dis = torch.cdist(img_mesh, img_1_mesh) # Produces (img_H * img_W, img_H1 * img_W1) tensor
    _, orimg2img1_idx = torch.topk(orimg2img1_dis, k=1, dim=1, largest=False, sorted=False) # Produces (img_H * img_W, 1) tensor
    orimg2img1_idx = orimg2img1_idx.squeeze(-1).unsqueeze(0).unsqueeze(0).expand(B, B, -1) # Produces (B, B, img_H * img_W) tensor
    img1_2_img2_dis = torch.cdist(img_1_mesh, img_2_mesh) # Produces (img_H1 * img_W1, img_H2 * img_W2) tensor
    _, img1_2_img2_idx = torch.topk(img1_2_img2_dis, k=1, dim=1, largest=False, sorted=False) # Produces (img_H1 * img_W1, 1) tensor
    img1_2_img2_idx = img1_2_img2_idx.squeeze(-1).unsqueeze(0).unsqueeze(0).expand(B, B, -1) # Produces (B, B, img_H1 * img_W1) tensor
    img2_2_img3_dis = torch.cdist(img_2_mesh, img_3_mesh) # Produces (img_H2 * img_W2, img_H3 * img_W3) tensor
    _, img2_2_img3_idx = torch.topk(img2_2_img3_dis, k=1, dim=1, largest=False, sorted=False) # Produces (img_H2 * img_W2, 1) tensor
    img2_2_img3_idx = img2_2_img3_idx.squeeze(-1).unsqueeze(0).unsqueeze(0).expand(B, B, -1) # Produces (B, B, img_H2 * img_W2) tensor

    fps_idx = pointnet2_utils.furthest_point_sample(node1, num_points2).long()  # [B, num_points2]
    node2 = index_points(node1, fps_idx)  # [B, num_points2, 3]
    node12node2_dis = torch.cdist(node1, node2) # Produces (B, num_points1, num_points2) tensor
    _, node12node2_idx = torch.topk(node12node2_dis, k=1, dim=2, largest=False, sorted=False) # Produces (B, num_points1, 1) tensor
    node12node2_idx = node12node2_idx.squeeze(-1) # Produces (B, num_points1) tensor
    fps_idx = pointnet2_utils.furthest_point_sample(node2, num_points3).long()  # [B, num_points3]
    node3 = index_points(node2, fps_idx)  # [B, num_points3, 3]
    node23node3_dis = torch.cdist(node2, node3) # Produces (B, num_points2, num_points3) tensor
    _, node23node3_idx = torch.topk(node23node3_dis, k=1, dim=2, largest=False, sorted=False) # Produces (B, num_points2, 1) tensor
    node23node3_idx = node23node3_idx.squeeze(-1) # Produces (B, num_points2) tensor

    img1_num_pt = torch_scatter.scatter_sum(img_num_pt, orimg2img1_idx, dim=2, dim_size=img_H1 * img_W1) # Produces (B, B, img_H1 * img_W1) tensor
    img2_mum_pt = torch_scatter.scatter_sum(img1_num_pt, img1_2_img2_idx, dim=2, dim_size=img_H2 * img_W2) # Produces (B, B, img_H2 * img_W2) tensor
    img3_num_pt = torch_scatter.scatter_sum(img2_mum_pt, img2_2_img3_idx, dim=2, dim_size=img_H3 * img_W3) # Produces (B, B, img_H3 * img_W3) tensor
    node2_num_pt = torch_scatter.scatter_sum(node1_num_pt, node12node2_idx, dim=1, dim_size=num_points2) # Produces (B, num_points2) tensor
    node3_num_pt = torch_scatter.scatter_sum(node2_num_pt, node23node3_idx, dim=1, dim_size=num_points3) # Produces (B, num_points3) tensor
    img1_num_pixel = torch_scatter.scatter_sum(img_num_pixel, orimg2img1_idx, dim=2, dim_size=img_H1 * img_W1) # Produces (B, B, img_H1 * img_W1) tensor
    img2_num_pixel = torch_scatter.scatter_sum(img1_num_pixel, img1_2_img2_idx, dim=2, dim_size=img_H2 * img_W2) # Produces (B, B, img_H2 * img_W2) tensor
    img3_num_pixel = torch_scatter.scatter_sum(img2_num_pixel, img2_2_img3_idx, dim=2, dim_size=img_H3 * img_W3) # Produces (B, B, img_H3 * img_W3) tensor

    node1_img1_map_num = torch_scatter.scatter_sum(node1_image_map_num, orimg2img1_idx.unsqueeze(-1), dim=2, dim_size=img_H1 * img_W1) # Produces (B, B, img_H1 * img_W1, num_points1) tensor
    node2_img1_map_num = torch_scatter.scatter_sum(node1_img1_map_num, node12node2_idx.unsqueeze(1).unsqueeze(2), dim=3, dim_size=num_points2) # Produces (B, B, img_H1 * img_W1, num_points2) tensor
    node2_img2_map_num = torch_scatter.scatter_sum(node2_img1_map_num, img1_2_img2_idx.unsqueeze(-1), dim=2, dim_size=img_H2 * img_H2) # Produces (B, B, img_H2 * img_W2, num_points2) tensor
    node3_img2_map_num = torch_scatter.scatter_sum(node2_img2_map_num, node23node3_idx.unsqueeze(1).unsqueeze(2), dim=3, dim_size=num_points3) # Produces (B, B, img_H2 * img_W2, num_points3) tensor
    node3_img3_map_num = torch_scatter.scatter_sum(node3_img2_map_num, img2_2_img3_idx.unsqueeze(-1), dim=2, dim_size=img_H3 * img_W3) # Produces (B, B, img_H3 * img_W3, num_points3) tensor

    node1_img1_map_pixel_num = torch_scatter.scatter_sum(node1_image_map_pixel_num, orimg2img1_idx.unsqueeze(-1), dim=2, dim_size=img_H1 * img_W1) # Produces (B, B, img_H1 * img_W1, num_points1) tensor
    node2_img1_map_pixel_num = torch_scatter.scatter_sum(node1_img1_map_pixel_num, node12node2_idx.unsqueeze(1).unsqueeze(2), dim=3, dim_size=num_points2) # Produces (B, B, img_H1 * img_W1, num_points2) tensor
    node2_img2_map_pixel_num = torch_scatter.scatter_sum(node2_img1_map_pixel_num, img1_2_img2_idx.unsqueeze(-1), dim=2, dim_size=img_H2 * img_W2) # Produces (B, B, img_H2 * img_W2, num_points2) tensor
    node3_img2_map_pixel_num = torch_scatter.scatter_sum(node2_img2_map_pixel_num, node23node3_idx.unsqueeze(1).unsqueeze(2), dim=3, dim_size=num_points3) # Produces (B, B, img_H2 * img_W2, num_points3) tensor
    node3_img3_map_pixel_num = torch_scatter.scatter_sum(node3_img2_map_pixel_num, img2_2_img3_idx.unsqueeze(-1), dim=2, dim_size=img_H3 * img_W3) # Produces (B, B, img_H3 * img_W3, num_points3) tensor

@torch.no_grad()
def generate_original_pc_correspondence(device, result, dataset_cfgs):

    P_camera_lidar = torch.matmul(torch.linalg.inv(result['image_poses']).unsqueeze(0).to(device), result['cloud_poses_original'].unsqueeze(1).to(device)) # Produces (B, B, 4, 4) tensor
    clouds = result['clouds_original'].to(device)
    N = clouds.shape[1]
    clouds_to_mult = clouds.unsqueeze(1) # Produces (B, 1, N, 3) tensor
    clouds_to_mult = torch.cat([clouds_to_mult, torch.ones_like(clouds_to_mult[..., :1])], dim=-1) # Produces (B, 1, N, 4) tensor
    clouds_in_camera = torch.matmul(clouds_to_mult, P_camera_lidar.permute(0, 1, 3, 2)) # Produces (B, B, N, 4) tensor
    clouds_in_camera = clouds_in_camera[..., :3]

    mask_0 = torch.ge(clouds_in_camera[..., 2], 0.0) # Produces (B, B, N) tensor
    # mask_1 = result['cloud_remove_masks'].unsqueeze(1).expand_as(mask_0).to(device) # Produces (B, B, num_downsampled) tensor
    mask_1 = torch.lt(clouds_in_camera[..., 2], dataset_cfgs.correspondence_point_project_threshold) # Produces (B, B, N) tensor TODO: this should be treated as a super parameter
    image_intrinscs_to_mult = result['image_intrinscs'].unsqueeze(0).to(device) # Produces (1, B, 3, 3) tensor
    clouds_in_image = torch.matmul(clouds_in_camera, image_intrinscs_to_mult.permute(0, 1, 3, 2)) # Produces (B, B, N, 3) tensor
    clouds_in_plane = clouds_in_image[..., :2] / clouds_in_image[..., 2:] # Produces (B, B, N, 2) tensor
    B, _, img_H, img_W = result['images'].shape

    mask_2 = torch.ge(clouds_in_plane[..., 0], 0.0) & \
           torch.lt(clouds_in_plane[..., 0], img_W * 1.0) & \
           torch.ge(clouds_in_plane[..., 1], 0.0) & \
           torch.lt(clouds_in_plane[..., 1], img_H * 1.0) # Produces (B, B, N) tensor
    
    # need more mask to filter out the points that are not in the image
    # 1、the max distance between the points and the center point of current camera frame : 100、150、200、250、300？
    mask = mask_0 & mask_1 & mask_2 # Produces (B, B, N) tensor

    original_2_downsampled_indices_inuse = torch.gather(result['cloud_shuffle_indices'].to(device),
                                                        dim=-1,
                                                        index=result["original_2_downsampled_indices"].to(device)) # Produces (B, N) tensor
    original_cloud_remove_masks = torch.gather(input=result["cloud_remove_masks"].to(device),
                                                dim=-1,
                                                index=original_2_downsampled_indices_inuse) # Produces (B, N) tensor

    clouds_in_plane_pixels = torch.floor(clouds_in_plane).type(torch.float32) # Produces (B, B, N, 2) tensor
    clouds_in_plane_pixels[..., 0] = torch.clamp(clouds_in_plane_pixels[..., 0], min=0.0, max=img_W * 1.0 - 1)
    clouds_in_plane_pixels[..., 1] = torch.clamp(clouds_in_plane_pixels[..., 1], min=0.0, max=img_H * 1.0 - 1)
    clouds_in_plane_pixels_flattened = clouds_in_plane_pixels[:, :, :, 1] * img_W + clouds_in_plane_pixels[:, :, :, 0] # Produces (B, B, N) tensor
    
    original_pc_2_many_1 = torch.cat((original_2_downsampled_indices_inuse.unsqueeze(1).unsqueeze(-1).expand(-1, B, -1, -1).type(torch.int64),
                                    clouds_in_plane_pixels_flattened.unsqueeze(-1).type(torch.int64)), dim=-1) # Produces (B, B, N, 2) tensor
    original_pc_2_many_2 = torch.cat((mask.unsqueeze(-1), 
                                      ~original_cloud_remove_masks.unsqueeze(1).unsqueeze(-1).expand(-1, B, -1, -1)), dim=-1) # Produces (B, B, N, 2) tensor

    if "save_for_visualization" in dataset_cfgs.keys() and dataset_cfgs.save_for_visualization:
        if result['bn'] in dataset_cfgs.visualization_batches:
            curr_bn = result['bn']
            for i in range(len(result['labels'])):
                label_i = result['labels'][i]
                file_name_2 = f'/home/pengjianyi/code_projects/visualization_open3d/train_batch{curr_bn}_{label_i}_pc_original.npy'
                curr_pc_original = clouds[i, :, :].cpu().numpy()
                np.save(file_name_2, curr_pc_original)
                for j in range(len(result['labels'])):
                    label_j = result['labels'][j]
                    file_name_1 = f'/home/pengjianyi/code_projects/visualization_open3d/train_batch{curr_bn}_{label_i}_2_{label_j}_original_pc_2_many_2.npy'
                    curr_original_pc_2_many_2 = original_pc_2_many_2[i, j, :, :].cpu().numpy()
                    np.save(file_name_1, curr_original_pc_2_many_2)
                

    return original_pc_2_many_1, original_pc_2_many_2
    
    # clouds_in_plane[..., 0].clamp_(min=0.0, max=img_W * 1.0 - 1e-9)
    # clouds_in_plane[..., 1].clamp_(min=0.0, max=img_H * 1.0 - 1e-9)
    # original_pc_2_many_3 = clouds_in_plane # Produces (B, B, N, 2) tensor

    # return original_pc_2_many_1, original_pc_2_many_2, original_pc_2_many_3




# co-relate the relations on in pairs 
@torch.no_grad()
def generate_original_pc_correspondence_v2(device, result, dataset_cfgs):

    P_camera_lidar = torch.matmul(torch.linalg.inv(result['image_poses']).to(device), result['cloud_poses_original'].to(device)) # Produces (B, 4, 4) tensor
    clouds = result['clouds_original'].to(device)
    clouds_to_mult = clouds # Produces (B, N, 3) tensor
    clouds_to_mult = torch.cat([clouds_to_mult, torch.ones_like(clouds_to_mult[..., :1])], dim=-1) # Produces (B, N, 4) tensor
    clouds_in_camera = torch.matmul(clouds_to_mult, P_camera_lidar.permute(0, 2, 1)) # Produces (B, N, 4) tensor
    clouds_in_camera = clouds_in_camera[..., :3]

    mask_0 = torch.ge(clouds_in_camera[..., 2], 0.0) # Produces (B, N) tensor
    mask_1 = torch.lt(clouds_in_camera[..., 2], dataset_cfgs.correspondence_point_project_threshold) # Produces (B, N)
    image_intrinscs_to_mult = result['image_intrinscs'].to(device) # Produces (B, 3, 3) tensor
    clouds_in_image = torch.matmul(clouds_in_camera, image_intrinscs_to_mult.permute(0, 2, 1)) # Produces (B, N, 3) tensor
    clouds_in_plane = clouds_in_image[..., :2] / clouds_in_image[..., 2:] # Produces (B, N, 2) tensor
    B, _, img_H, img_W = result['images'].shape

    mask_2 = torch.ge(clouds_in_plane[..., 0], 0.0) & \
           torch.lt(clouds_in_plane[..., 0], img_W * 1.0) & \
           torch.ge(clouds_in_plane[..., 1], 0.0) & \
           torch.lt(clouds_in_plane[..., 1], img_H * 1.0) # Produces (B, N) tensor
    
    mask = mask_0 & mask_1 & mask_2 # Produces (B, N) tensor

    original_2_downsampled_indices_inuse = torch.gather(result['cloud_shuffle_indices'].to(device),
                                                        dim=-1,
                                                        index=result["original_2_downsampled_indices"].to(device)) # Produces (B, N) tensor
    original_cloud_remove_masks = torch.gather(input=result["cloud_remove_masks"].to(device),
                                                dim=-1,
                                                index=original_2_downsampled_indices_inuse) # Produces (B, N) tensor

    clouds_in_plane_pixels = torch.floor(clouds_in_plane).type(torch.float32) # Produces (B, N, 2) tensor
    clouds_in_plane_pixels[..., 0] = torch.clamp(clouds_in_plane_pixels[..., 0], min=0.0, max=img_W * 1.0 - 1)
    clouds_in_plane_pixels[..., 1] = torch.clamp(clouds_in_plane_pixels[..., 1], min=0.0, max=img_H * 1.0 - 1)
    clouds_in_plane_pixels_flattened = clouds_in_plane_pixels[:, :, 1] * img_W + clouds_in_plane_pixels[:, :, 0] # Produces (B, N) tensor
    
    original_pc_2_many_1 = torch.cat((original_2_downsampled_indices_inuse.unsqueeze(-1).type(torch.int64),
                                    clouds_in_plane_pixels_flattened.unsqueeze(-1).type(torch.int64)), dim=-1) # Produces (B, N, 2) tensor
    original_pc_2_many_2 = torch.cat((mask.unsqueeze(-1), 
                                      ~original_cloud_remove_masks.unsqueeze(-1)), dim=-1) # Produces (B, N, 2) tensor

                

    return original_pc_2_many_1, original_pc_2_many_2

@torch.no_grad()
def generate_original_pc_correspondence_v3(device, result, dataset_cfgs):

    original_2_downsampled_indices_inuse = torch.gather(result['cloud_shuffle_indices'].to(device),
                                                        dim=-1,
                                                        index=result["original_2_downsampled_indices"].to(device)) # Produces (B, N) tensor
    original_cloud_remove_masks = torch.gather(input=result["cloud_remove_masks"].to(device),
                                                dim=-1,
                                                index=original_2_downsampled_indices_inuse) # Produces (B, N) tensor
    return (result['cloud_poses'].to(device), # (B, N, 4)
            ~original_cloud_remove_masks, # (B, N)
            result['clouds_original'].to(device), # (B, original_points_num, 3)
            result['cloud_poses_original'].to(device), # (B, 4, 4)
            )

    
@torch.no_grad()
def generate_pixel_point_correspondence(device, result, dataset_cfgs):

    if "use_pixel_distance_correspondence" not in dataset_cfgs.keys() or not dataset_cfgs.use_pixel_distance_correspondence:
        return None, None, None, None, None

    P_camera_lidar = torch.matmul(torch.linalg.inv(result['image_poses']).unsqueeze(0).to(device), result['cloud_poses_original'].unsqueeze(1).to(device)) # Produces (B, B, 4, 4) tensor
    clouds = result['clouds_original'].to(device)
    clouds_to_mult = clouds.unsqueeze(1) # Produces (B, 1, N, 3) tensor
    clouds_to_mult = torch.cat([clouds_to_mult, torch.ones_like(clouds_to_mult[..., :1])], dim=-1) # Produces (B, 1, N, 4) tensor
    clouds_in_camera = torch.matmul(clouds_to_mult, P_camera_lidar.permute(0, 1, 3, 2)) # Produces (B, B, N, 4) tensor
    clouds_in_camera = clouds_in_camera[..., :3]

    mask_0 = torch.ge(clouds_in_camera[..., 2], 0.0) # Produces (B, B, N) tensor
    image_intrinscs_to_mult = result['image_intrinscs'].unsqueeze(0).to(device) # Produces (1, B, 3, 3) tensor



    # TODO: when the feature resolution changed, this should be changed too
    image_intrinscs_to_mult = camera_matrix_scaling(image_intrinscs_to_mult, 0.5, 0.5)
    img_H = 112
    img_W = 112




    clouds_in_image = torch.matmul(clouds_in_camera, image_intrinscs_to_mult.permute(0, 1, 3, 2)) # Produces (B, B, N, 3) tensor
    clouds_in_plane = clouds_in_image[..., :2] / clouds_in_image[..., 2:] # Produces (B, B, N, 2) tensor
    B = result['images'].shape[0]

    mask_2 = torch.ge(clouds_in_plane[..., 0], 0.0) & \
           torch.lt(clouds_in_plane[..., 0], img_W * 1.0) & \
           torch.ge(clouds_in_plane[..., 1], 0.0) & \
           torch.lt(clouds_in_plane[..., 1], img_H * 1.0) # Produces (B, B, N) tensor
    
    original_2_downsampled_indices_inuse = torch.gather(result['cloud_shuffle_indices'].to(device),
                                                        dim=-1,
                                                        index=result["original_2_downsampled_indices"].to(device)) # Produces (B, N) tensor
    original_cloud_remove_masks = torch.gather(input=result["cloud_remove_masks"].to(device),
                                                dim=-1,
                                                index=original_2_downsampled_indices_inuse) # Produces (B, N) tensor
    
    mask1 = original_cloud_remove_masks.unsqueeze(1).expand(-1, B, -1) # Produces (B, B, N) tensor

    # need more mask to filter out the points that are not in the image
    # 1、the max distance between the points and the center point of current camera frame : 100、150、200、250、300？
    mask = mask_0 & ~(mask1) & mask_2 # Produces (B, B, N) tensor

    clouds_in_plane_pixels = torch.floor(clouds_in_plane) # Produces (B, B, N, 2) tensor
    clouds_in_plane_pixels[..., 0] = torch.clamp(clouds_in_plane_pixels[..., 0], min=0, max=img_W - 1)
    clouds_in_plane_pixels[..., 1] = torch.clamp(clouds_in_plane_pixels[..., 1], min=0, max=img_H - 1)
    original_pc_2_cimg = torch.cat((clouds_in_plane_pixels, mask.unsqueeze(-1)), dim=-1) # Produces (B, B, N, 3) tensor
    original_pc_2_cimg_unique, original_pc_2_cimg_num = my_unique_v2(original_pc_2_cimg, 
                                                                     torch.ones_like(original_pc_2_cimg[..., 0]),
                                                                    (img_W, img_H, 2)) # Produces (huge_num, 5), (huge_num)
    
    original_pc_2_cimg_mask = torch.eq(original_pc_2_cimg_unique[..., -1], 1) # produce (huge_num)
    original_pc_2_cimg_unique = original_pc_2_cimg_unique[original_pc_2_cimg_mask, :-1] # produce (huge_num_1, 4)
    original_pc_2_cimg_num = original_pc_2_cimg_num[original_pc_2_cimg_mask] # produce (huge_num_1)

    original_pc_2_cimg_mask_2 = torch.gt(original_pc_2_cimg_num, dataset_cfgs.pixel_candidate_threshold) # produce (huge_num_1)
    original_pc_2_cimg_unique = original_pc_2_cimg_unique[original_pc_2_cimg_mask_2, :] # produce (huge_num_2, 4)
    original_pc_2_cimg_num = original_pc_2_cimg_num[original_pc_2_cimg_mask_2] # produce (huge_num_2)

    clouds_ds = torch.stack(result['clouds'], dim=0).to(device) # Produces (B, 4096, 3) tensor
    P_camera_lidar_ds = torch.matmul(torch.linalg.inv(result['image_poses']).unsqueeze(0).to(device), result['cloud_poses'].unsqueeze(1).to(device)) # Produces (B, B, 4, 4) tensor
    clouds_ds_to_mult = clouds_ds.unsqueeze(1) # Produces (B, 1, 4096, 3) tensor
    clouds_ds_to_mult = torch.cat([clouds_ds_to_mult, torch.ones_like(clouds_ds_to_mult[..., :1])], dim=-1) # Produces (B, 1, 4096, 4) tensor
    clouds_ds_in_camera = torch.matmul(clouds_ds_to_mult, P_camera_lidar_ds.permute(0, 1, 3, 2)) # Produces (B, B, 4096, 4) tensor
    clouds_ds_in_camera = clouds_ds_in_camera[..., :3]

    mask_ds_0 = torch.ge(clouds_ds_in_camera[..., 2], 0.0) # Produces (B, B, 4096) tensor
    clouds_ds_in_image = torch.matmul(clouds_ds_in_camera, image_intrinscs_to_mult.permute(0, 1, 3, 2)) # Produces (B, B, 4096, 3) tensor
    clouds_ds_in_plane = clouds_ds_in_image[..., :2] / clouds_ds_in_image[..., 2:] # Produces (B, B, 4096, 2) tensor

    mask_ds_1 = result['cloud_remove_masks'].to(device)

    mask_ds_2 = torch.ge(clouds_ds_in_plane[..., 0], 0.0) & \
           torch.lt(clouds_ds_in_plane[..., 0], img_W * 1.0) & \
           torch.ge(clouds_ds_in_plane[..., 1], 0.0) & \
           torch.lt(clouds_ds_in_plane[..., 1], img_H * 1.0) # Produces (B, B, 4096) tensor
    
    # need more mask to filter out the points that are not in the image
    # 1、the max distance between the points and the center point of current camera frame : 100、150、200、250、300？
    mask_ds = mask_ds_0 & (~mask_ds_1) & mask_ds_2 # Produces (B, B, 4096) tensor

    clouds_ds_in_plane_pixels = torch.floor(clouds_ds_in_plane) # Produces (B, B, 4096, 2) tensor
    clouds_ds_in_plane_pixels[..., 0] = torch.clamp(clouds_ds_in_plane_pixels[..., 0], min=0, max=img_W - 1)
    clouds_ds_in_plane_pixels[..., 1] = torch.clamp(clouds_ds_in_plane_pixels[..., 1], min=0, max=img_H - 1)
    clouds_ds_in_plane_pixels[~mask_ds.unsqueeze(-1).expand(-1, -1, -1, 2)] = 1e6 # Produces (B, B, 4096, 2) tensor
    


    if dataset_cfgs.pixel_selection_method == "random_among_batch": # a easier way, but hard to ensure every image has enough selected pixels
        random_upon_pixel_num_indices = torch.nonzero(torch.gt(original_pc_2_cimg_num, dataset_cfgs.random_upon_pixel_num))[:, 0] # produce (cfgs.random_upon_pixel_num, )
        random_upon_pixel_num_length = random_upon_pixel_num_indices.shape[0]
        random_upon_pixel_num_length_randperm = torch.randperm(random_upon_pixel_num_length, dtype=torch.int64, device=device)
        pixel_selection_num_all = min(dataset_cfgs.pixel_selection_num_all, random_upon_pixel_num_length)
        random_upon_pixel_num_length_selected_indices = random_upon_pixel_num_length_randperm[:pixel_selection_num_all]
        original_pc_2_cimg_unique_selected_indices = random_upon_pixel_num_indices[random_upon_pixel_num_length_selected_indices] # produce (cfgs.pixel_selection_num_all, )
        original_pc_2_cimg_unique_selected = original_pc_2_cimg_unique[original_pc_2_cimg_unique_selected_indices, :] # produce (pixel_selection_num_all, 4)
        # original_pc_2_cimg_unique_selected = torch.unique(original_pc_2_cimg_unique_selected, dim=0, sorted=True)
        # original_pc_2_cimg_unique_batch = original_pc_2_cimg_unique_selected[:, 0] * B + original_pc_2_cimg_unique_selected[:, 1]
        # original_pc_2_cimg_unique_batch_num = torch.bincount(original_pc_2_cimg_unique_batch, minlength=B * B)
        # original_pc_2_cimg_unique_batch_cumsum = torch.cumsum(original_pc_2_cimg_unique_batch_num, dim=0) # produce (B * B)
        # clouds_ds_in_plane_pixels_batch_num = torch.full((B * B,), 4096, dtype=torch.int64, device=device)
        # clouds_ds_in_plane_pixels_batch_cumsum = torch.cumsum(clouds_ds_in_plane_pixels_batch_num, dim=0) # produce (B * B)
        # knn_idxs
        clouds_ds_to_plane_pixels_selected = clouds_ds_in_plane_pixels[original_pc_2_cimg_unique_selected[:, 0], original_pc_2_cimg_unique_selected[:, 1], :, :]  # (cfgs.pixel_selection_num_all, 4096, 2)
        clouds_ds_to_pixels_selected_dist = F.pairwise_distance(clouds_ds_to_plane_pixels_selected.flatten(0, 1), 
                                                        original_pc_2_cimg_unique_selected[:, 2:].unsqueeze(1).expand(-1, 4096, -1).flatten(0, 1), 
                                                        p=2.0,
                                                        eps=1e-06) # produce (cfgs.pixel_selection_num_all * 4096)
        clouds_ds_to_pixels_selected_dist = clouds_ds_to_pixels_selected_dist.reshape(pixel_selection_num_all, 4096)
        _, points_selected_indices = torch.topk(clouds_ds_to_pixels_selected_dist,
                                                               k=dataset_cfgs.points_selection_num,
                                                               dim=-1,
                                                               largest=False,
                                                               sorted=False) # produce (cfgs.pixel_selection_num_all, cfgs.points_selection_num)
        points_selected_indices = torch.stack([original_pc_2_cimg_unique_selected[:, :1].expand(-1, dataset_cfgs.points_selection_num), points_selected_indices], dim=-1) # (cfgs.pixel_selection_num_all, cfgs.points_selection_num, 2)
        points_selected_indices = points_selected_indices.flatten(0, 1) # (cfgs.pixel_selection_num_all * cfgs.points_selection_num, 2)
        points_selected_plane_pixels = clouds_ds_in_plane_pixels[points_selected_indices[:, 0], :, points_selected_indices[:, 1], :] # (cfgs.pixel_selection_num_all * cfgs.points_selection_num, B, 2)
        points_selected_plane_pixels = points_selected_plane_pixels[:, original_pc_2_cimg_unique_selected[:, 1], :] # (cfgs.pixel_selection_num_all * cfgs.points_selection_num, cfgs.pixel_selection_num_all, 2)
        points_selected_plane_pixels = points_selected_plane_pixels.permute(1, 0, 2).flatten(0, 1) #(cfgs.pixel_selection_num_all * (cfgs.pixel_selection_num_all * cfgs.points_selection_num), 2)
        pixels_selected_pixels_temp = original_pc_2_cimg_unique_selected[:, 2:].unsqueeze(1).expand(-1, pixel_selection_num_all * dataset_cfgs.points_selection_num, -1).flatten(0, 1) # (cfgs.pixel_selection_num_all * (cfgs.pixel_selection_num_all * cfgs.points_selection_num), 2)
        pixels_selected_to_points_selected_dist = F.pairwise_distance(points_selected_plane_pixels, 
                                                        pixels_selected_pixels_temp, 
                                                        p=2.0,
                                                        eps=1e-06) # produce (cfgs.pixel_selection_num_all * (cfgs.pixel_selection_num_all * cfgs.points_selection_num))
        pixels_selected_to_points_selected_dist = pixels_selected_to_points_selected_dist.reshape(pixel_selection_num_all, pixel_selection_num_all * dataset_cfgs.points_selection_num)
        positive_mask = torch.lt(pixels_selected_to_points_selected_dist, dataset_cfgs.pair_positive_dist) # produce (cfgs.pixel_selection_num_all, cfgs.pixel_selection_num_all * cfgs.points_selection_num)
        negative_mask = torch.gt(pixels_selected_to_points_selected_dist, dataset_cfgs.pair_negative_dist) # produce (cfgs.pixel_selection_num_all, cfgs.pixel_selection_num_all * cfgs.points_selection_num)
        pixels_selected_indices = original_pc_2_cimg_unique_selected[:, 1:] # produce (cfgs.pixel_selection_num_all, 3)
        return dataset_cfgs.pixel_selection_method, pixels_selected_indices, points_selected_indices, positive_mask, negative_mask
    elif dataset_cfgs.pixel_selection_method == "random_in_pair": # like what the CFI2P did
        original_pc_2_cimg_unique_pixels = torch.unique(original_pc_2_cimg_unique[:, 1:], dim=0, sorted=True) # produce (huge_num_3, 3)
        original_pc_2_cimg_unique_pixels_num = torch.bincount(original_pc_2_cimg_unique_pixels[:, 0].long(), minlength=B) # produce (B,)
        original_pc_2_cimg_unique_pixels_cumsum = torch.cumsum(original_pc_2_cimg_unique_pixels_num, dim=0) # produce (B,)
        original_pc_2_cimg_unique_pixels_cumsum = torch.cat((torch.zeros(1, device=device), original_pc_2_cimg_unique_pixels_cumsum[:-1]), dim=0) # produce (B,)
        pixels_selected_indices_base_1 = original_pc_2_cimg_unique_pixels_cumsum.unsqueeze(1).expand(-1, dataset_cfgs.pixel_selection_num_each_pair) # produce (B, cfgs.pixel_selection_num_each_pair)
        pixels_selected_indices_base_2 = torch.arange(0, dataset_cfgs.pixel_selection_num_each_pair, device=device).unsqueeze(0).expand(B, -1) # produce (B, cfgs.pixel_selection_num_each_pair)
        pixels_selected_indices_qt = torch.div(torch.tensor([dataset_cfgs.pixel_selection_num_each_pair], device=device, dtype=torch.int64).expand(B), original_pc_2_cimg_unique_pixels_num, rounding_mode='floor') # produce (B,)
        pixels_selected_indices_base_2_rmd = torch.fmod(pixels_selected_indices_base_2, 
                                                        original_pc_2_cimg_unique_pixels_num.unsqueeze(-1).expand(-1, dataset_cfgs.pixel_selection_num_each_pair)) # produce (B, cfgs.pixel_selection_num_each_pair)
        pixels_selected_indices_mult = pixels_selected_indices_qt * original_pc_2_cimg_unique_pixels_num # produce (B,)
        pixels_selected_indices_base_rand = torch.randint(0, 1000, (B, dataset_cfgs.pixel_selection_num_each_pair), device=device) # produce (B, cfgs.pixel_selection_num_each_pair)
        pixels_selected_indices_base_rand_rmd = torch.fmod(pixels_selected_indices_base_rand, 
                                                           original_pc_2_cimg_unique_pixels_num.unsqueeze(-1).expand(-1, dataset_cfgs.pixel_selection_num_each_pair)) # produce (B, cfgs.pixel_selection_num_each_pair)
        pixels_selected_indices_base_2 = torch.where(pixels_selected_indices_base_2 < pixels_selected_indices_mult.unsqueeze(-1).expand(-1, dataset_cfgs.pixel_selection_num_each_pair),
                                                     pixels_selected_indices_base_2_rmd,
                                                     pixels_selected_indices_base_rand_rmd) # produce (B, cfgs.pixel_selection_num_each_pair)
        pixels_selected_indices = (pixels_selected_indices_base_1 + pixels_selected_indices_base_2).type(torch.int64) # produce (B, cfgs.pixel_selection_num_each_pair)
        pixels_selected_indices = original_pc_2_cimg_unique_pixels[pixels_selected_indices, 1:]  # (B, cfgs.pixel_selection_num_each_pair, 2)

        clouds_ds_to_pixels_selected_dist = torch.cdist(x1=clouds_ds_in_plane_pixels.flatten(0, 1).type(torch.float32), 
                                                        x2=pixels_selected_indices.unsqueeze(0).expand(B, -1, -1, -1).flatten(0, 1).type(torch.float32), 
                                                        p=2.0) # produce (B * B, 4096, cfgs.pixel_selection_num_each_pair)
        _, clouds_ds_to_pixels_selected_topk_indices = torch.topk(clouds_ds_to_pixels_selected_dist,
                                                               k=dataset_cfgs.points_selection_num,
                                                               dim=1,
                                                               largest=False,
                                                               sorted=False) # produce (B * B, cfgs.points_selection_num, cfgs.pixel_selection_num_each_pair)
        points_selected_indices = clouds_ds_to_pixels_selected_topk_indices.flatten(1, 2) # (B * B, cfgs.points_selection_num * cfgs.pixel_selection_num_each_pair)
        points_selected_to_pixels_selected_dist = torch.gather(clouds_ds_to_pixels_selected_dist,
                                                           dim=1,
                                                           index=points_selected_indices.unsqueeze(-1).expand(-1, -1, dataset_cfgs.pixel_selection_num_each_pair)) # produce (B * B, cfgs.points_selection_num * cfgs.pixel_selection_num_each_pair, cfgs.pixel_selection_num_each_pair)
        positive_mask = torch.lt(points_selected_to_pixels_selected_dist, dataset_cfgs.pair_positive_dist) # produce (B * B, cfgs.points_selection_num * cfgs.pixel_selection_num_each_pair, cfgs.pixel_selection_num_each_pair)
        negative_mask = torch.gt(points_selected_to_pixels_selected_dist, dataset_cfgs.pair_negative_dist) # produce (B * B, cfgs.points_selection_num * cfgs.pixel_selection_num_each_pair, cfgs.pixel_selection_num_each_pair)
        return dataset_cfgs.pixel_selection_method, pixels_selected_indices, points_selected_indices, positive_mask, negative_mask
    else:
        raise NotImplementedError
                

@torch.no_grad()
def my_unique(x: tensor, x_num: tensor):
    """
    when the x is (16, 16, 40960, 3) and the k is 2, the unique_x takes 200 MB GPU memory mostly
    this function means we can select very high resolution corresponding pairs without choosing a small area first
    or compute a very memory_cost matrix
    it's worth noted that the CFI2P choose the high confident proxy first and choose the high confident points for every proxy,
    then choose the high confident patch for every proxy and got every pixel in this patch
    TODO: need to test the time and memory consumed by the torch.unique function
    Args:
        x: tensor of shape (dim1, dim2, ..., dimk, n, m)
        x_num: tensor of shape (dim1, dim2, ..., dimk, n)
    Returns:
        unique_x: tensor of shape (huge_num, k+m),
        unique_x_num: tensor of shape (huge_num)
    """
    device = x.device
    x_shape = x.shape
    indices = x_shape[:-2]
    coordinate_list = []
    for i in range(len(indices)):
        coordinate_list.append(torch.arange(0, indices[i], device=device))
    meshgrid_list = torch.meshgrid(*coordinate_list) # Produces (k,) list, the i th element is a (dim1, dim2, ..., dimk) tensor
    x_flattened = x.view(-1, x_shape[-1]) # produce (dim1 * dim2 * ... * dimk * n, m) tensor
    meshgrid_v1_list = []
    for i in range(len(meshgrid_list)):
        meshgrid_v1_list.append(meshgrid_list[i].reshape(-1).repeat_interleave(x_shape[-2]).unsqueeze(-1)) # produce (dim1 * dim2 * ... * dimk * n, 1) tensor

    x_v1 = torch.cat((*meshgrid_v1_list, x_flattened), dim=-1) # produce (dim1 * dim2 * ... * dimk * n, k+m) tensor
    unique_x, x_v2_indices = torch.unique(x_v1, dim=0, 
                                        sorted=True, 
                                        return_inverse=True, 
                                        return_counts=False) # TODO: need test the time spend、the real memory cost
                                                            # produce(huge_num, k+m), (dim1 * dim2 * ... * dimk * n)tensor
    x_num_flattened = x_num.view(-1)
    unique_x_num = torch_scatter.scatter_sum(x_num_flattened, x_v2_indices, dim=-1) # produce (huge_num) tensor
    return unique_x, unique_x_num

@torch.no_grad()
def my_unique_v2(x, x_num, dim_info):
    """
    when the x is (16, 16, 40960, 3) and the k is 2, the unique_x takes 200 MB GPU memory mostly
    this function means we can select very high resolution corresponding pairs without choosing a small area first
    or compute a very memory_cost matrix
    it's worth noted that the CFI2P choose the high confident proxy first and choose the high confident points for every proxy,
    then choose the high confident patch for every proxy and got every pixel in this patch
    TODO: need to test the time and memory consumed by the torch.unique function
    Args:
        x: tensor of shape (dim1, dim2, ..., dimk, n, m)
        x_num: tensor of shape (dim1, dim2, ..., dimk, n)
        dim_info: list of shape (m)
    Returns:
        unique_x: tensor of shape (huge_num, k+m),
        unique_x_num: tensor of shape (huge_num)
    """
    device = x.device
    x_shape = x.shape
    indices = x_shape[:-2]
    coordinate_list = []
    for i in range(len(indices)):
        coordinate_list.append(torch.arange(0, indices[i], device=device))
    meshgrid_list = torch.meshgrid(*coordinate_list) # Produces (k,) list, the i th element is a (dim1, dim2, ..., dimk) tensor
    x_flattened = x.view(-1, x_shape[-1]) # produce (dim1 * dim2 * ... * dimk * n, m) tensor
    meshgrid_v1_list = []
    for i in range(len(meshgrid_list)):
        meshgrid_v1_list.append(meshgrid_list[i].reshape(-1).repeat_interleave(x_shape[-2]).unsqueeze(-1)) # produce (dim1 * dim2 * ... * dimk * n, 1) tensor

    x_v1 = torch.cat((*meshgrid_v1_list, x_flattened), dim=-1) # produce (dim1 * dim2 * ... * dimk * n, k+m) tensor
    x_num_flattened = x_num.view(-1)
    x_v1_sparse = torch.sparse_coo_tensor(x_v1.transpose(0, 1), x_num_flattened, indices + dim_info) # produce (dim1, dim2, ..., dimk, dim_info1, dim_info2, ..., dim_infom) tensor
    x_v1_sparse = x_v1_sparse.coalesce() # produce (dim1, dim2, ..., dimk, dim_info1, dim_info2, ..., dim_infom) tensor
    unique_x = x_v1_sparse.indices().transpose(0, 1) # produce (huge_num, k+m) tensor
    unique_x_num = x_v1_sparse.values() # produce (huge_num) tensor
    return unique_x, unique_x_num

@torch.no_grad()
def generate_masks(device, data_input, data, dataloader, dataset_cfgs):
    if 'labels' in data.keys():
        
        if 'generate_overlap_mask' in dataset_cfgs.keys() and dataset_cfgs.generate_overlap_mask:
            if isinstance(data['labels'], list):
                if dataset_cfgs.overlap_ratio_type == 'pos_vec_vet':
                    curr_seq_ID = data['labels'][0][0]
                    labels = torch.tensor([frame_ID for seq_ID, frame_ID in data['labels']], dtype=torch.int64)
                    data_input['positives_mask'] = dataloader.dataset.get_sp_positives_1(labels, curr_seq_ID)
                    data_input['negatives_mask'] = ~data_input['positives_mask']
                else:
                    data_input['negatives_mask'] = torch.le(data_input['overlap_ratio'], dataset_cfgs.negative_mask_overlap_threshld)
                    data_input['positives_mask'] = torch.ge(data_input['overlap_ratio'], dataset_cfgs.positive_mask_overlap_threshld)
            elif isinstance(data['labels'], np.ndarray):
                if dataset_cfgs.overlap_ratio_type == 'pos_vec_vet':
                    labels_tensor = torch.tensor(data['labels'], dtype=torch.int64)
                    data_input['positives_mask'] = dataloader.dataset.get_sp_positives_1(labels_tensor)
                    data_input['negatives_mask'] = ~data_input['positives_mask']
                else:
                    data_input['negatives_mask'] = torch.le(data_input['overlap_ratio'], dataset_cfgs.negative_mask_overlap_threshld)
                    data_input['positives_mask'] = torch.ge(data_input['overlap_ratio'], dataset_cfgs.positive_mask_overlap_threshld)
            else:
                raise NotImplementedError
        else:
            if not dataset_cfgs.use_overlap_ratio:
                # 2、rely on the torch.cdist
                if isinstance(data['labels'], list):
                    curr_seq_ID = data['labels'][0][0]
                    for seq_ID, _ in data['labels']:
                        if seq_ID != curr_seq_ID:
                            raise ValueError('The seq_ID is not consistent.')
                    labels = torch.tensor([frame_ID for seq_ID, frame_ID in data['labels']], dtype=torch.int64)

                    if 'dist_caculation_type' in dataset_cfgs.keys() and dataset_cfgs.dist_caculation_type == 'all_coords_L2_mean':
                        query_UTM_coord = dataloader.dataset.UTM_coord_tensor[curr_seq_ID][:, labels, :]
                        query_UTM_coord = query_UTM_coord.to(device)
                        query_2_query_dist = torch.cdist(query_UTM_coord, query_UTM_coord, p=2.0).mean(dim=0, keepdim=False) # (B, B)
                    else:
                        query_UTM_coord = dataloader.dataset.UTM_coord_tensor[curr_seq_ID][labels]
                        query_UTM_coord = query_UTM_coord.to(device)
                        query_2_query_dist = torch.cdist(query_UTM_coord, query_UTM_coord, p=2.0) # (B, B)
                elif isinstance(data['labels'], np.ndarray):
                    labels = torch.tensor(data['labels'], dtype=torch.int64, device=device)
                    if 'dist_caculation_type' in dataset_cfgs.keys() and dataset_cfgs.dist_caculation_type == 'all_coords_L2_mean':
                        query_UTM_coord = dataloader.dataset.UTM_coord_tensor[:, labels, :]
                        query_UTM_coord = query_UTM_coord.to(device)
                        query_2_query_dist = torch.cdist(query_UTM_coord, query_UTM_coord, p=2.0).mean(dim=0, keepdim=False) # (B, B)
                    else: 
                        query_UTM_coord = dataloader.dataset.UTM_coord_tensor[labels]
                        query_UTM_coord = query_UTM_coord.to(device)
                        query_2_query_dist = torch.cdist(query_UTM_coord, query_UTM_coord, p=2.0) # (B, B)
                else:
                    raise NotImplementedError
                data_input['positives_mask'] = torch.le(query_2_query_dist, dataset_cfgs.positive_distance)
                data_input['negatives_mask'] = torch.gt(query_2_query_dist, dataset_cfgs.non_negative_distance)
                if 'use_true_neighbors_mask' in dataset_cfgs.keys() and dataset_cfgs.use_true_neighbors_mask:
                    data_input['true_neighbors_mask'] = torch.le(query_2_query_dist, dataset_cfgs.true_neighbour_dist)
        
        if 'use_coords_information' in dataset_cfgs.keys() and dataset_cfgs.use_coords_information:
            data_input['coords'] = dataloader.dataset.UTM_coord_tensor[labels].cuda()
        
        if 'use_memory_bank' in dataset_cfgs.keys() and dataset_cfgs.use_memory_bank:
            data_input['labels'] = data['labels']

@torch.no_grad()
def generate_UTM_overlap_ratio(device, data, dataloader, cfgs):
    labels = torch.tensor(data['labels'], dtype=torch.int64)
    UTM_coords = dataloader.dataset.UTM_coord_tensor[labels].to(device)
    dist_mat = torch.cdist(UTM_coords, UTM_coords, p=2.0)
    dist_mat = torch.clamp(dist_mat, max=cfgs.UTM_overlap_ratio_max_dist)
    overlap_ratio = 1.0 - dist_mat / cfgs.UTM_overlap_ratio_max_dist
    return overlap_ratio

@torch.no_grad()
def process_labels(device, data_input, data, dataset, dataset_cfgs):
    if hasattr(dataset, 'use_semantic_label') and dataset.use_semantic_label:
        data_input['cityscapes_label_in_semanticKitti_label_list'] = dataset.label_correspondence_table
        data_input['img_semantic_label'] = data['img_semantic_labels'].to(device)
        data_input['img_ccl_cluster_label'] = data['img_ccl_cluster_labels'].to(device)
        pc_semantic_label_temp = data['pc_semantic_labels'].to(device)
        pc_dbscan_cluster_label_temp = data['pc_dbscan_cluster_labels'].to(device)
        cloud_remove_masks_temp = data['cloud_remove_masks'].to(device)
        cloud_shuffle_indices_temp = data['cloud_shuffle_indices'].to(device)
        cloud_shuffle_indices_temp = torch.argsort(cloud_shuffle_indices_temp, dim=-1)
        pc_semantic_label_shuffled_temp = torch.gather(pc_semantic_label_temp,
                                                  dim=-1,
                                                  index=cloud_shuffle_indices_temp.unsqueeze(1)) # Produces (B, 1, N) tensor
        pc_dbscan_cluster_label_shuffled_temp = torch.gather(pc_dbscan_cluster_label_temp,
                                                  dim=-1,
                                                  index=cloud_shuffle_indices_temp.unsqueeze(1)) # Produces (B, 1, N) tensor
        pc_semantic_label_shuffled_temp[cloud_remove_masks_temp.unsqueeze(1)] = -1
        pc_dbscan_cluster_label_shuffled_temp[cloud_remove_masks_temp.unsqueeze(1)] = -1
        data_input['pc_semantic_label'] = pc_semantic_label_shuffled_temp
        data_input['pc_dbscan_cluster_label'] = pc_dbscan_cluster_label_shuffled_temp

@torch.no_grad()
def generate_pixel_point_correspondence_v2(device, result, dataset_cfgs):

    if 'pixel_selection_method' not in dataset_cfgs.keys():
        return None, None, None, None
    B = result['images'].shape[0]
    range_img_H, range_img_W = result['range_to_pc_original_idxs'].shape[1:]
    range_to_pc_original_idxs_flatten = result['range_to_pc_original_idxs'].cuda().flatten(1).type(torch.int64) # (B, range_img_H * range_img_W)
    N = result['clouds_original'].shape[1]
    pc_original_to_range_idx = torch.full((B, N), -1, dtype=torch.int64, device=device)
    valid_mask = range_to_pc_original_idxs_flatten >= 0 # (B, range_img_H * range_img_W)
    value = torch.nonzero(valid_mask, as_tuple=False) # (temp_N, 2)
    indices = range_to_pc_original_idxs_flatten[value[:, 0], value[:, 1]] # (temp_N,)
    # batch_indices = torch.arange(0, B, device=device).repeat_interleave(valid_mask.sum(dim=-1)) # (temp_N,)
    pc_original_to_range_idx[value[:, 0], indices] = value[:, 1] # (B, N)

    pc_original_to_range_idx_mask = pc_original_to_range_idx == -1 # (B, N)
    pc_original_to_range_idx[pc_original_to_range_idx_mask] = range_img_H * range_img_W - 1 # (B, N)
    pc_original_to_range_idx_mask_num = torch.count_nonzero(pc_original_to_range_idx_mask, dim=-1) # (B,)


    range_img_H_mesh = torch.arange(0, range_img_H, device=device).type(torch.float32)
    range_img_W_mesh = torch.arange(0, range_img_W, device=device).type(torch.float32)
    range_img_H_mesh = range_img_H_mesh.unsqueeze(1).unsqueeze(2).expand(-1, range_img_W, -1)
    range_img_W_mesh = range_img_W_mesh.unsqueeze(0).unsqueeze(2).expand(range_img_H, -1, -1)
    range_img_mesh = torch.cat((range_img_H_mesh, range_img_W_mesh), dim=-1) # Produces (range_img_H, range_img_W, 2)
    range_img_mesh = range_img_mesh.flatten(0, 1) # Produces (range_img_H * range_img_W, 2) tensor
    c_range_img_H = range_img_H // 2
    c_range_img_W = range_img_W // 2
    c_range_img_H_mesh = torch.arange(0, c_range_img_H, device=device)
    c_range_img_W_mesh = torch.arange(0, c_range_img_W, device=device)
    img_2_c_range_img_scale_H = range_img_H * 1.0 / c_range_img_H
    img_2_c_range_img_scale_W = range_img_W * 1.0 / c_range_img_W
    delta_H = img_2_c_range_img_scale_H / 2 - 0.5
    delta_W = img_2_c_range_img_scale_W / 2 - 0.5
    c_range_img_H_mesh = c_range_img_H_mesh * img_2_c_range_img_scale_H + delta_H
    c_range_img_W_mesh = c_range_img_W_mesh * img_2_c_range_img_scale_W + delta_W
    c_range_img_H_mesh = c_range_img_H_mesh.unsqueeze(1).unsqueeze(2).expand(-1, c_range_img_W, -1) # Produces (c_range_img_H, c_range_img_W, 1) tensor
    c_range_img_W_mesh = c_range_img_W_mesh.unsqueeze(0).unsqueeze(2).expand(c_range_img_H, -1, -1) # Produces (c_range_img_H, c_range_img_W, 1) tensor
    c_range_img_mesh = torch.cat((c_range_img_H_mesh, c_range_img_W_mesh), dim=-1) # Produces (c_range_img_H, c_range_img_W, 2) tensor
    c_range_img_mesh = c_range_img_mesh.flatten(0, 1) # Produces (c_range_img_H * c_range_img_W, 2) tensor
    _, range_img_2_c_range_img_idx = keops_knn(device, range_img_mesh, c_range_img_mesh, 1) # Produces (range_img_H * range_img_W, 1) tensor
    range_img_2_c_range_img_idx = range_img_2_c_range_img_idx.squeeze(-1).unsqueeze(0).expand(B, -1) # Produces (B, range_img_H * range_img_W) tensor
    original_pc_2_c_range_img = torch.gather(input=range_img_2_c_range_img_idx,
                                            dim=-1,
                                            index=pc_original_to_range_idx) # Produces (B, N)
    
    original_pc_2_c_range_img[pc_original_to_range_idx_mask] = c_range_img_H * c_range_img_W - 1
    
    downsampled_pc = torch_scatter.scatter_mean(result['clouds_original'].to(device),
                                                dim=1,
                                                index=original_pc_2_c_range_img.unsqueeze(-1).expand(-1, -1, 3),
                                                dim_size=c_range_img_H * c_range_img_W) # Produces (B, c_range_img_H * c_range_img_W, 3) tensor
    downsampled_pc_num = torch_scatter.scatter_sum(torch.ones_like(result['clouds_original'][:, :, 0], device=device),
                                                    dim=-1,
                                                    index=original_pc_2_c_range_img,
                                                    dim_size=c_range_img_H * c_range_img_W) # Produces (B, c_range_img_H * c_range_img_W) tensor
    
    last_fake_num = downsampled_pc_num[:, c_range_img_H * c_range_img_W - 1] # Produces (B,) tensor
    last_real_num = last_fake_num - pc_original_to_range_idx_mask_num # Produces (B,) tensor
    last_real_num_div = torch.clamp(last_real_num, min=1) # Produces (B,) tensor
    last_fake_coords_mean = downsampled_pc[:, c_range_img_H * c_range_img_W - 1] # Produces (B, 3) tensor
    last_fake_coords_sum = last_fake_coords_mean * last_fake_num.unsqueeze(-1) # Produces (B, 3) tensor
    clouds_false_original = result['clouds_original'].detach().clone().cuda()
    pc_original_to_range_idx_mask_idx = torch.nonzero(~pc_original_to_range_idx_mask, as_tuple=False) # Produces (temp_M, 2) tensor
    clouds_false_original[pc_original_to_range_idx_mask_idx[:, 0], pc_original_to_range_idx_mask_idx[:, 1]] = 0.0 # Produces (B, N, 3) tensor
    last_real_coords_sum = last_fake_coords_sum - torch.sum(clouds_false_original, dim=1, keepdim=False) # Produces (B, 3) tensor
    last_real_coords_mean = last_real_coords_sum / last_real_num_div.unsqueeze(-1) # Produces (B, 3) tensor

    downsampled_pc_num[:, c_range_img_H * c_range_img_W - 1] = last_real_num # Produces (B, c_range_img_H * c_range_img_W) tensor
    downsampled_pc[:, c_range_img_H * c_range_img_W - 1] = last_real_coords_mean # Produces (B, c_range_img_H * c_range_img_W, 3) tensor

    downsampled_pc_mask = downsampled_pc_num <= 0 # Produces (B, c_range_img_H * c_range_img_W) tensor

    P_camera_lidar = torch.matmul(torch.linalg.inv(result['image_poses']).to(device), result['cloud_poses_original'].to(device)) # Produces (B, 4, 4) tensor
    clouds = downsampled_pc # Produces (B, c_range_img_H * c_range_img_W, 3) tensor
    clouds_to_mult = clouds # Produces (B, c_range_img_H * c_range_img_W, 3) tensor
    clouds_to_mult = torch.cat([clouds_to_mult, torch.ones_like(clouds_to_mult[..., :1])], dim=-1) # Produces (B, c_range_img_H * c_range_img_W, 4) tensor
    clouds_in_camera = torch.matmul(clouds_to_mult, P_camera_lidar.permute(0, 2, 1)) # Produces (B, c_range_img_H * c_range_img_W, 4) tensor
    clouds_in_camera = clouds_in_camera[..., :3]

    mask_0 = torch.ge(clouds_in_camera[..., 2], 0.0) # Produces (B, c_range_img_H * c_range_img_W) tensor
    image_intrinscs_to_mult = result['image_intrinscs'].to(device) # Produces (B, 3, 3) tensor
    # TODO: when the feature resolution changed, this should be changed too
    image_intrinscs_to_mult = camera_matrix_scaling(image_intrinscs_to_mult, 0.5, 0.5)
    img_H = result['images'].shape[2] // 2
    img_W = result['images'].shape[3] // 2


    clouds_in_image = torch.matmul(clouds_in_camera, image_intrinscs_to_mult.permute(0, 2, 1)) # Produces (B, c_range_img_H * c_range_img_W, 3) tensor
    clouds_in_plane = clouds_in_image[..., :2] / clouds_in_image[..., 2:] # Produces (B, c_range_img_H * c_range_img_W, 2) tensor

    mask_2 = torch.ge(clouds_in_plane[..., 0], 0.0) & \
           torch.lt(clouds_in_plane[..., 0], img_W * 1.0) & \
           torch.ge(clouds_in_plane[..., 1], 0.0) & \
           torch.lt(clouds_in_plane[..., 1], img_H * 1.0) # Produces (B, c_range_img_H * c_range_img_W,) tensor
    
    mask1 = downsampled_pc_mask # Produces (B, c_range_img_H * c_range_img_W) tensor
    mask = mask_0 & ~(mask1) & mask_2 # Produces (B, c_range_img_H * c_range_img_W) tensor

    clouds_in_plane_pixels = torch.floor(clouds_in_plane) # Produces (B, c_range_img_H * c_range_img_W, 2) tensor
    clouds_in_plane_pixels[..., 0] = torch.clamp(clouds_in_plane_pixels[..., 0], min=0, max=img_W - 1)
    clouds_in_plane_pixels[..., 1] = torch.clamp(clouds_in_plane_pixels[..., 1], min=0, max=img_H - 1)
    original_pc_2_cimg = torch.cat((clouds_in_plane_pixels, mask.unsqueeze(-1)), dim=-1) # Produces (B, c_range_img_H * c_range_img_W, 3) tensor
    original_pc_2_cimg_unique, original_pc_2_cimg_num = my_unique_v2(original_pc_2_cimg, 
                                                                     torch.ones_like(original_pc_2_cimg[..., 0]),
                                                                    (img_W, img_H, 2)) # Produces (huge_num, 4), (huge_num)
    
    original_pc_2_cimg_mask = torch.eq(original_pc_2_cimg_unique[..., -1], 1) # produce (huge_num)
    original_pc_2_cimg_unique = original_pc_2_cimg_unique[original_pc_2_cimg_mask, :-1] # produce (huge_num_1, 3)
    original_pc_2_cimg_num = original_pc_2_cimg_num[original_pc_2_cimg_mask] # produce (huge_num_1)

    clouds_ds_in_plane_pixels = clouds_in_plane_pixels
    clouds_ds_in_plane_pixels[~mask.unsqueeze(-1).expand(-1, -1, 2)] = 1e6 # Produces (B, c_range_img_H * c_range_img_W, 2) tensor

    original_pc_2_cimg_unique_pixels = torch.unique(original_pc_2_cimg_unique, dim=0, sorted=True) # produce (huge_num_3, 3)
    original_pc_2_cimg_unique_pixels_num = torch.bincount(original_pc_2_cimg_unique_pixels[:, 0].long(), minlength=B) # produce (B,)

    # print(original_pc_2_cimg_unique_pixels_num)

    original_pc_2_cimg_unique_pixels_cumsum = torch.cumsum(original_pc_2_cimg_unique_pixels_num, dim=0) # produce (B,)
    original_pc_2_cimg_unique_pixels_cumsum = torch.cat((torch.zeros(1, device=device), original_pc_2_cimg_unique_pixels_cumsum[:-1]), dim=0) # produce (B,)
    pixels_selected_indices_base_1 = original_pc_2_cimg_unique_pixels_cumsum.unsqueeze(1).expand(-1, dataset_cfgs.pixel_selection_num_each_pair) # produce (B, cfgs.pixel_selection_num_each_pair)
    pixels_selected_indices_base_2 = torch.arange(0, dataset_cfgs.pixel_selection_num_each_pair, device=device).unsqueeze(0).expand(B, -1) # produce (B, cfgs.pixel_selection_num_each_pair)
    pixels_selected_indices_qt = torch.div(torch.tensor([dataset_cfgs.pixel_selection_num_each_pair], device=device, dtype=torch.int64).expand(B), original_pc_2_cimg_unique_pixels_num, rounding_mode='floor') # produce (B,)
    pixels_selected_indices_base_2_rmd = torch.fmod(pixels_selected_indices_base_2, 
                                                    original_pc_2_cimg_unique_pixels_num.unsqueeze(-1).expand(-1, dataset_cfgs.pixel_selection_num_each_pair)) # produce (B, cfgs.pixel_selection_num_each_pair)
    pixels_selected_indices_mult = pixels_selected_indices_qt * original_pc_2_cimg_unique_pixels_num # produce (B,)
    pixels_selected_indices_base_rand = torch.randint(0, 1000, (B, dataset_cfgs.pixel_selection_num_each_pair), device=device) # produce (B, cfgs.pixel_selection_num_each_pair)
    pixels_selected_indices_base_rand_rmd = torch.fmod(pixels_selected_indices_base_rand, 
                                                        original_pc_2_cimg_unique_pixels_num.unsqueeze(-1).expand(-1, dataset_cfgs.pixel_selection_num_each_pair)) # produce (B, cfgs.pixel_selection_num_each_pair)
    pixels_selected_indices_base_2 = torch.where(pixels_selected_indices_base_2 < pixels_selected_indices_mult.unsqueeze(-1).expand(-1, dataset_cfgs.pixel_selection_num_each_pair),
                                                    pixels_selected_indices_base_2_rmd,
                                                    pixels_selected_indices_base_rand_rmd) # produce (B, cfgs.pixel_selection_num_each_pair)
    pixels_selected_indices = (pixels_selected_indices_base_1 + pixels_selected_indices_base_2).type(torch.int64) # produce (B, cfgs.pixel_selection_num_each_pair)
    pixels_selected_indices = original_pc_2_cimg_unique_pixels[pixels_selected_indices, 1:]  # (B, cfgs.pixel_selection_num_each_pair, 2)

    clouds_ds_to_pixels_selected_dist = torch.cdist(x1=clouds_ds_in_plane_pixels.type(torch.float32), 
                                                    x2=pixels_selected_indices.type(torch.float32), 
                                                    p=2.0) # produce (B, c_range_img_H * c_range_img_W, cfgs.pixel_selection_num_each_pair)
    
    if dataset_cfgs.pixel_selection_method == "topk_points_for_every_pixel":
        _, clouds_ds_to_pixels_selected_topk_indices = torch.topk(clouds_ds_to_pixels_selected_dist,
                                                                k=dataset_cfgs.points_selection_num,
                                                                dim=1,
                                                                largest=False,
                                                                sorted=False) # produce (B, cfgs.points_selection_num, cfgs.pixel_selection_num_each_pair)
        points_selected_indices = clouds_ds_to_pixels_selected_topk_indices.flatten(1, 2) # (B, cfgs.points_selection_num * cfgs.pixel_selection_num_each_pair)
        points_selected_to_pixels_selected_dist = torch.gather(clouds_ds_to_pixels_selected_dist,
                                                            dim=1,
                                                            index=points_selected_indices.unsqueeze(-1).expand(-1, -1, dataset_cfgs.pixel_selection_num_each_pair)) # produce (B, cfgs.points_selection_num * cfgs.pixel_selection_num_each_pair, cfgs.pixel_selection_num_each_pair)
        positive_mask = torch.lt(points_selected_to_pixels_selected_dist, dataset_cfgs.pair_positive_dist) # produce (B, cfgs.points_selection_num * cfgs.pixel_selection_num_each_pair, cfgs.pixel_selection_num_each_pair)
        negative_mask = torch.gt(points_selected_to_pixels_selected_dist, dataset_cfgs.pair_negative_dist) # produce (B, cfgs.points_selection_num * cfgs.pixel_selection_num_each_pair, cfgs.pixel_selection_num_each_pair)
    elif dataset_cfgs.pixel_selection_method == "all_the_candidate_points":
        points_selected_indices = torch.arange(0, c_range_img_H * c_range_img_W, device=device, dtype=torch.int64).unsqueeze(0).expand(B, -1) # produce (B, c_range_img_H * c_range_img_W)
        positive_mask = torch.lt(clouds_ds_to_pixels_selected_dist, dataset_cfgs.pair_positive_dist) # produce (B, c_range_img_H * c_range_img_W, cfgs.pixel_selection_num_each_pair)
        negative_mask = torch.gt(clouds_ds_to_pixels_selected_dist, dataset_cfgs.pair_negative_dist) # produce (B, c_range_img_H * c_range_img_W, cfgs.pixel_selection_num_each_pair)
    else: 
        raise NotImplementedError
    return pixels_selected_indices, points_selected_indices, positive_mask, negative_mask