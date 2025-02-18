import os
import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import KDTree
from mini_boreas import BoreasDataset_U
import matplotlib.pyplot as plt
import random
from shapely.geometry import Polygon, Point
from torch import Tensor
import torch
from pykeops.torch import LazyTensor
from typing import Tuple

# this version is for pointcloud mnn

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

# different from the v3, the v4 is to sample the coord equally in the round 
P1 = {'x1': -150.0, 'x2': -50.0, 'y1': 0.0, 'y2': 100.0}
P2 = {'x1': -420.0, 'x2': -380.0, 'y1': 950.0, 'y2': 1100.0}
P3 = {'x1': -1200.0, 'x2': -1100.0, 'y1': 950.0, 'y2': 1050.0}
P4 = {'x1': -950.0, 'x2': -830.0, 'y1': 1950.0, 'y2': 2100.0}

P_val = []
P_test = [P1, P2, P3, P4]

device = torch.device('cuda:1')
torch.cuda.set_device(device)
# x = torch.cuda.FloatTensor(256, 1024, 16 * 1024)
# del x

def check_in_specified_set(x_poses, y_poses, rect_areas):
    test_flags = np.zeros(x_poses.shape[0], dtype=np.bool_)
    for rect_area in rect_areas:
        flags1 = x_poses > rect_area['x1']
        flags2 = x_poses < rect_area['x2']
        flags3 = y_poses > rect_area['y1']
        flags4 = y_poses < rect_area['y2']
        test_flags |= flags1 * flags2 * flags3 * flags4
    return test_flags

data_root = '/DATA5/pengjianyi'
dataset_root = os.path.join(data_root, 'Boreas_minuse')
dataset = BoreasDataset_U(dataset_root)
minuse_lidar_idxs_path = os.path.join(dataset_root, 'my_tool', 'minuse_lidar_idxs.pickle')
minuse_lidar = pickle.load(open(minuse_lidar_idxs_path, 'rb'))

pc_dir_name = "Boreas_minuse_40960_to_4096_cliped_fov"
pc_path = os.path.join(data_root, pc_dir_name)
lidar_2_image_idx_path = os.path.join(dataset_root, 'my_tool/lidar2image.pickle')
lidar_2_image_idx = pickle.load(open(lidar_2_image_idx_path, 'rb'))


all_train_pointcloud_list = []
overlap_knn_dis_threshold = 0.8
N = 4096


for sequence in dataset.sequences:
    lidar_pose = []
    curr_pointcloud_list = []
    for lidar_id in minuse_lidar[sequence.ID]:
        curr_lidar_pose = []
        curr_lidar = sequence.lidar_frames[lidar_id]

        lidar_pre_path = curr_lidar.path
        seq_ID, lidar_dir, pc_file_name = lidar_pre_path.split('/')[-3:]
        lidar_curr_path_prefix = os.path.join(pc_path, seq_ID, lidar_dir, pc_file_name.split('.')[0])
        pc = np.load(lidar_curr_path_prefix + '_1.npy')
        pc_pose = curr_lidar.pose.astype(np.float32)
        pc_to_mult = np.concatenate([pc, np.ones_like(pc[:, -1:])], axis=-1) # shape: (N, 4)
        pc_INS = np.dot(pc_to_mult, pc_pose.T) # shape: (N, 4)
        pc_INS = pc_INS[:, :3] # shape: (N, 3)


        # curr_image_idxs = lidar_2_image_idx[sequence.ID][str(lidar_id)]
        # curr_image_idx = curr_image_idxs[0]
        # curr_image_frame = sequence.camera_frames[curr_image_idx]
        # curr_image_pose = curr_image_frame.pose.astype(np.float32)
        
        curr_lidar_pose_x = curr_lidar.pose[0, 3].astype(np.float32)
        curr_lidar_pose_y = curr_lidar.pose[1, 3].astype(np.float32)
        curr_lidar_pose.append(curr_lidar_pose_x)
        curr_lidar_pose.append(curr_lidar_pose_y)

        lidar_pose.append(curr_lidar_pose)
        curr_pointcloud_list.append(pc_INS)
    
    UTM_coords = np.array(lidar_pose, dtype=np.float32)
    test_flags = check_in_specified_set(UTM_coords[:, 0], UTM_coords[:, 1], P_test)
    test_indices = np.nonzero(test_flags)[0]
    curr_train_pointcloud_list = []
    for i in range(len(curr_pointcloud_list)):
        if i in test_indices:
            continue
        curr_train_pointcloud_list.append(curr_pointcloud_list[i])
    all_train_pointcloud_list += curr_train_pointcloud_list
    print(f'visit {sequence.ID} done!')

all_overlap_ratio_list = []
all_train_pointcloud_list_length = len(all_train_pointcloud_list)
step = 288


with torch.no_grad():
    for idx_1 in range(0, all_train_pointcloud_list_length, step):
        curr_overlap_ratio_list = []
        idx_1_start = idx_1
        idx_1_end = min(idx_1_start + step, all_train_pointcloud_list_length)
        pc_1_ndarray = np.stack(all_train_pointcloud_list[idx_1_start:idx_1_end], axis=0) # Produces (B1, N, 3) ndarray
        pc_1_tensor = torch.tensor(pc_1_ndarray, dtype=torch.float32, device=device) # Produces (B1, N, 3) tensor
        B1 = pc_1_tensor.shape[0]
        for idx_2 in range(0, all_train_pointcloud_list_length, step):
            idx_2_start = idx_2
            idx_2_end = min(idx_2_start + step, all_train_pointcloud_list_length)
            pc_2_ndarray = np.stack(all_train_pointcloud_list[idx_2_start:idx_2_end], axis=0)
            pc_2_tensor = torch.tensor(pc_2_ndarray, dtype=torch.float32, device=device) # Produces (B2, N, 3) tensor
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
            curr_overlap_ratio_list.append(overlap_ratio_ndarray)

            for i in range(B1):
                for j in range(B2):
                    pc_1_x = pc_1_tensor[i, :, 0]
                    pc_1_y = pc_1_tensor[i, :, 1]
                    pc_2_x = pc_2_tensor[j, :, 0]
                    pc_2_y = pc_2_tensor[j, :, 1]
                    plt.scatter(pc_1_x.to('cpu').numpy(), pc_1_y.to('cpu').numpy(), s=float(1/10), color='red')
                    plt.scatter(pc_2_x.to('cpu').numpy(), pc_2_y.to('cpu').numpy(), s=float(1/10), color='blue')
                    plt.xlabel('Longitude (x)')
                    plt.ylabel('Latitude (y)')
                    plt.title(f'pc mnn overlap {overlap_ratio_ndarray[i, j]}')
                    plt.savefig(f'/home/pengjianyi/code_projects/vis1018/pc_mnn_{idx_1}_{idx_2}_{i}_{j}.png', bbox_inches='tight', pad_inches=0, dpi=200)
                    plt.close()

            print(f'{idx_1} to {idx_2} done!')
        curr_overlap_ratio_ndarray = np.concatenate(curr_overlap_ratio_list, axis=1) # Produces (B1, all_train_pointcloud_list_length) ndarray
        all_overlap_ratio_list.append(curr_overlap_ratio_ndarray)

all_overlap_ratio = np.concatenate(all_overlap_ratio_list, axis=0)

train_overlap_save_path = os.path.join(dataset_root, 'my_tool', f'train_point_cloud_mnn.npy')
np.save(train_overlap_save_path, all_overlap_ratio)
print(f'save {dataset_root}/{train_overlap_save_path} done!')