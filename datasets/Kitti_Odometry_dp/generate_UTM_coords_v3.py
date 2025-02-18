import os
import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import random
from my_pykitti_odometry import my_odometry
import open3d as o3d
from torch import Tensor
from pykeops.torch import LazyTensor
from typing import Tuple
import torch

# pointcloud mnn of KITTI dataset

# use the LiDAR coordinate system as the world coordinate system

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

data_root = '/DATA1/pengjianyi'
dataset_root = os.path.join(data_root, 'KITTI/dataset')
pose_root = os.path.join(data_root, 'semanticKITTI/dataset')
tool_path = os.path.join(data_root, 'KITTI', 'my_tool')
sequence_list = ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
pc_inuse_root = os.path.join(data_root, 'KITTI/16384_to_4096_cliped_fov')
device = torch.device('cuda:0')
torch.cuda.set_device(device)


all_train_pointcloud_coords = {}
for seq_ID in sequence_list:
    curr_sequence = my_odometry(sequence=seq_ID, base_path=dataset_root, pose_path=pose_root)
    pointcloud_coords = []
    for id in range(len(curr_sequence.timestamps)):

        T_first_cam0_curr_cam0 = curr_sequence.poses[id]
        curr_calib = curr_sequence.calib
        T_cam0_LiDAR = curr_calib['T_ego_LiDAR']

        T_first_cam0_curr_LiDAR = np.matmul(T_first_cam0_curr_cam0, T_cam0_LiDAR).astype(np.float32)
        T_first_LiDAR_curr_LiDAR = np.matmul(np.linalg.inv(T_cam0_LiDAR), T_first_cam0_curr_LiDAR)

        file_name = str(id).zfill(6)
        file_path = os.path.join(pc_inuse_root, seq_ID, 'velodyne', file_name + '_2.npy')
        pc_cliped = np.load(file_path)
        pc_cliped_to_mul = np.concatenate([pc_cliped, np.ones((pc_cliped.shape[0], 1))], axis=1) # (N, 4)
        pc_cliped_mul = np.matmul(pc_cliped_to_mul, T_first_LiDAR_curr_LiDAR.T) # (N, 4)
        pc_in_global = pc_cliped_mul[:, :3]

        pointcloud_coords.append(pc_in_global)
    train_pointcloud_coords = np.stack(pointcloud_coords, axis=0)
    all_train_pointcloud_coords[seq_ID] = train_pointcloud_coords
    print(f'visit {seq_ID} done!')

all_overlap_ratio_dict = {}
step = 128
overlap_knn_dis_threshold = 0.9
overlap_knn_dis_threshold_str = str(overlap_knn_dis_threshold).replace('.', 'p')
N = 4096

with torch.no_grad():
    for seq_ID, train_pointcloud_coords in all_train_pointcloud_coords.items():
        curr_seq_overlap_ratio = []
        for idx_1 in range(0, train_pointcloud_coords.shape[0], step):
            idx_1_start = idx_1
            idx_1_end = min(idx_1_start + step, train_pointcloud_coords.shape[0])
            pc_1_ndarray = train_pointcloud_coords[idx_1_start:idx_1_end] # Produces (B1, N, 3) ndarray
            pc_1_tensor = torch.tensor(pc_1_ndarray, dtype=torch.float32, device=device) # Produces (B1, N, 3) tensor
            B1 = pc_1_tensor.shape[0]

            curr_seq_idx1_overlap_ratio = []
            for idx_2 in range(0, train_pointcloud_coords.shape[0], step):
                idx_2_start = idx_2
                idx_2_end = min(idx_2_start + step, train_pointcloud_coords.shape[0])
                pc_2_ndarray = train_pointcloud_coords[idx_2_start:idx_2_end]
                pc_2_tensor = torch.tensor(pc_2_ndarray, dtype=torch.float32, device=device)
                B2 = pc_2_tensor.shape[0]

                clouds1to2_knn_indices = knn(
                    device=device,
                    q_points=pc_1_tensor.unsqueeze(1).expand(-1, B2, -1, -1),
                    s_points=pc_2_tensor.unsqueeze(0).expand(B1, -1, -1, -1),
                    k=1, 
                    dilation=1, 
                    distance_limit=overlap_knn_dis_threshold, 
                    return_distance=False,
                    remove_nearest=False,
                    transposed=False,
                    padding_mode='empty',
                    padding_value=1e10,
                    squeeze=False) # Produces (B1, B2, N, 1) tensor
                clouds2to1_knn_indices = knn(
                    device=device,
                    q_points=pc_2_tensor.unsqueeze(1).expand(-1, B1, -1, -1),
                    s_points=pc_1_tensor.unsqueeze(0).expand(B2, -1, -1, -1),
                    k=1,
                    dilation=1,
                    distance_limit=overlap_knn_dis_threshold,
                    return_distance=False,
                    remove_nearest=False,
                    transposed=False,
                    padding_mode='empty',
                    padding_value=1e10,
                    squeeze=False) # Produces (B2, B1, N, 1) tensor
                clouds_1to2_overlap_num = torch.count_nonzero(torch.lt(clouds1to2_knn_indices, N), dim=(-1, -2)) # Produces (B1, B2) tensor
                clouds_2to1_overlap_num = torch.count_nonzero(torch.lt(clouds2to1_knn_indices, N), dim=(-1, -2)) # Produces (B2, B1) tensor
                clouds_1to2_overlap_ratio = clouds_1to2_overlap_num * 1.0 / N # Produces (B1, B2) tensor
                clouds_2to1_overlap_ratio = clouds_2to1_overlap_num * 1.0 / N # Produces (B2, B1) tensor
                overlap_ratio = torch.maximum(clouds_1to2_overlap_ratio, clouds_2to1_overlap_ratio.T) # Produces (B1, B2) tensor
                overlap_ratio_ndarray = overlap_ratio.cpu().numpy() # Produces (B1, B2) ndarray
                curr_seq_idx1_overlap_ratio.append(overlap_ratio_ndarray)
            curr_seq_overlap_ratio.append(np.concatenate(curr_seq_idx1_overlap_ratio, axis=1))
        all_overlap_ratio_dict[seq_ID] = np.concatenate(curr_seq_overlap_ratio, axis=0)

train_overlap_save_path = os.path.join(data_root, 'KITTI', 'my_tool', f'train_point_cloud_mnn_{overlap_knn_dis_threshold_str}.pkl')
if os.path.exists(train_overlap_save_path):
    os.remove(train_overlap_save_path)
with open(train_overlap_save_path, 'wb') as f:
    pickle.dump(all_overlap_ratio_dict, f)
print(f'save train_overlap_ratio done!')