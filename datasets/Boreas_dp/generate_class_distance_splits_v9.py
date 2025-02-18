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
import torch_scatter

# this version is for point cloud projection overlap


# different from the v3, the v4 is to sample the coord equally in the round 
P1 = {'x1': -150.0, 'x2': -50.0, 'y1': 0.0, 'y2': 100.0}
P2 = {'x1': -420.0, 'x2': -380.0, 'y1': 950.0, 'y2': 1100.0}
P3 = {'x1': -1200.0, 'x2': -1100.0, 'y1': 950.0, 'y2': 1050.0}
P4 = {'x1': -950.0, 'x2': -830.0, 'y1': 1950.0, 'y2': 2100.0}

P_val = []
P_test = [P1, P2, P3, P4]

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
image_dir_name = 'Boreas_224x224_image'
image_path = os.path.join(data_root, image_dir_name)
lidar_2_image_idx_path = os.path.join(dataset_root, 'my_tool/lidar2image.pickle')
lidar_2_image_idx = pickle.load(open(lidar_2_image_idx_path, 'rb'))
device = torch.device('cuda:1')
point_project_threshold = 110.0

all_train_pointcloud_list = []
all_train_pointcloud_pose_list = []
all_train_image_pose_list = []
all_train_image_intrinsics_list = []

for sequence in dataset.sequences:
    
    lidar_pose = []
    curr_pointcloud_list = []
    curr_pointcloud_pose_list = []
    curr_image_pose_list = []
    curr_image_intrinsics_list = []

    for lidar_id in minuse_lidar[sequence.ID]:
        curr_lidar_pose = []
        curr_lidar = sequence.lidar_frames[lidar_id]

        lidar_pre_path = curr_lidar.path
        seq_ID, lidar_dir, pc_file_name = lidar_pre_path.split('/')[-3:]
        lidar_curr_path_prefix = os.path.join(pc_path, seq_ID, lidar_dir, pc_file_name.split('.')[0])
        pc = np.load(lidar_curr_path_prefix + '_2.npy')
        curr_pointcloud_list.append(pc)
        pc_pose = curr_lidar.pose.astype(np.float32)
        curr_pointcloud_pose_list.append(pc_pose)

        curr_image_idxs = lidar_2_image_idx[sequence.ID][str(lidar_id)]
        curr_image_idx = curr_image_idxs[0]
        curr_image_frame = sequence.camera_frames[curr_image_idx]
        curr_image_pose = curr_image_frame.pose.astype(np.float32)
        curr_image_pose_list.append(curr_image_pose)
        P0_path = os.path.join(image_path, seq_ID, 'calib', 'intrinsics.npy')
        curr_P0 = np.load(P0_path).astype(np.float32)
        curr_image_intrinsics_list.append(curr_P0)
        
        curr_lidar_pose_x = curr_lidar.pose[0, 3].astype(np.float32)
        curr_lidar_pose_y = curr_lidar.pose[1, 3].astype(np.float32)
        curr_lidar_pose.append(curr_lidar_pose_x)
        curr_lidar_pose.append(curr_lidar_pose_y)

        lidar_pose.append(curr_lidar_pose)
    
    UTM_coords = np.array(lidar_pose, dtype=np.float32)
    test_flags = check_in_specified_set(UTM_coords[:, 0], UTM_coords[:, 1], P_test)
    test_indices = np.nonzero(test_flags)[0]
    curr_train_pointcloud_list = []
    curr_train_pointcloud_pose_list = []
    curr_train_image_pose_list = []
    curr_train_image_intrinsics_list = []
    for i in range(len(curr_pointcloud_list)):
        if i in test_indices:
            continue
        curr_train_pointcloud_list.append(curr_pointcloud_list[i])
        curr_train_pointcloud_pose_list.append(curr_pointcloud_pose_list[i])
        curr_train_image_pose_list.append(curr_image_pose_list[i])
        curr_train_image_intrinsics_list.append(curr_image_intrinsics_list[i])
    all_train_pointcloud_list += curr_train_pointcloud_list
    all_train_pointcloud_pose_list += curr_train_pointcloud_pose_list
    all_train_image_pose_list += curr_train_image_pose_list
    all_train_image_intrinsics_list += curr_train_image_intrinsics_list
    print(f'visit {sequence.ID} done!')

assert len(all_train_pointcloud_list) == len(all_train_pointcloud_pose_list) == len(all_train_image_pose_list) == len(all_train_image_intrinsics_list)
all_overlap_ratio_list = []
all_train_pointcloud_list_length = len(all_train_pointcloud_list)
step = 48

img_H = 224
img_W = 224
N = 40960


for idx_pc in range(0, all_train_pointcloud_list_length, step):
    curr_overlap_ratio_list = []
    idx_pc_start = idx_pc
    idx_pc_end = min(idx_pc_start + step, all_train_pointcloud_list_length)
    pc_ndarray = np.stack(all_train_pointcloud_list[idx_pc_start:idx_pc_end], axis=0) # Produces (B1, N, 3) ndarray
    pc_tensor = torch.tensor(pc_ndarray, dtype=torch.float32, device=device) # Produces (B1, N, 3) tensor
    pc_pose_ndarray = np.stack(all_train_pointcloud_pose_list[idx_pc_start:idx_pc_end], axis=0) # Produces (B1, 4, 4) ndarray
    pc_pose_tensor = torch.tensor(pc_pose_ndarray, dtype=torch.float32, device=device) # Produces (B1, 4, 4) tensor
    B1 = pc_tensor.shape[0]
    for idx_img in range(0, all_train_pointcloud_list_length, step):
        idx_img_start = idx_img
        idx_img_end = min(idx_img_start + step, all_train_pointcloud_list_length)
        img_pose_ndarray = np.stack(all_train_image_pose_list[idx_img_start:idx_img_end], axis=0) # Produces (B2, 4, 4) ndarray
        img_pose_tensor = torch.tensor(img_pose_ndarray, dtype=torch.float32, device=device) # Produces (B2, 4, 4) tensor
        img_intrinsics_ndarray = np.stack(all_train_image_intrinsics_list[idx_img_start:idx_img_end], axis=0) # Produces (B2, 3, 3) ndarray
        img_intrinsics_tensor = torch.tensor(img_intrinsics_ndarray, dtype=torch.float32, device=device) # Produces (B2, 3, 3) tensor
        B2 = img_pose_tensor.shape[0]

        P_camera_lidar = torch.matmul(torch.linalg.inv(img_pose_tensor.unsqueeze(0)), pc_pose_tensor.unsqueeze(1)) # Produces (B1, B2, 4, 4) tensor
        clouds_to_mult = pc_tensor.unsqueeze(1) # Produces (B1, 1, N, 3) tensor
        clouds_to_mult = torch.cat([clouds_to_mult, torch.ones_like(clouds_to_mult[..., :1])], dim=-1) # Produces (B1, 1, N, 4) tensor
        clouds_in_camera = torch.matmul(clouds_to_mult, P_camera_lidar.permute(0, 1, 3, 2)) # Produces (B1, B2, N, 4) tensor
        clouds_in_camera = clouds_in_camera[..., :3] # Produces (B1, B2, N, 3) tensor
        mask_0 = torch.ge(clouds_in_camera[..., 2], 0.0) # Produces (B1, B2, N) tensor
        mask_1 = torch.lt(clouds_in_camera[..., 2], point_project_threshold) # Produces (B1, B2, N) tensor
        image_intrinscs_to_mult = img_intrinsics_tensor.unsqueeze(0) # Produces (1, B2, 3, 3) tensor
        clouds_in_image = torch.matmul(clouds_in_camera, image_intrinscs_to_mult.permute(0, 1, 3, 2)) # Produces (B1, B2, N, 3) tensor
        clouds_in_plane = clouds_in_image[..., :2] / clouds_in_image[..., 2:] # Produces (B1, B2, N, 2) tensor

        mask_2 = torch.ge(clouds_in_plane[..., 0], 0.0) & \
           torch.lt(clouds_in_plane[..., 0], float(img_W)) & \
           torch.ge(clouds_in_plane[..., 1], 0.0) & \
           torch.lt(clouds_in_plane[..., 1], float(img_H)) # Produces (B1, B2, N) tensor
        
        mask = mask_0 & mask_1 & mask_2 # Produces (B1, B2, N) tensor
        mask_num = torch.count_nonzero(~mask, dim=-1) # Produces (B1, B2) tensor
        
        overlap_points_num = torch.count_nonzero(mask, dim=-1) # Produces (B1, B2) tensor
        clouds_in_plane_pixels = torch.floor(clouds_in_plane).type(torch.int64) # Produces (B1, B2, N, 2) tensor
        clouds_in_plane_pixels_flattened = clouds_in_plane_pixels[:, :, :, 1] * img_W + clouds_in_plane_pixels[:, :, :, 0] # Produces (B1, B2, N) tensor
        clouds_in_plane_pixels_flattened.masked_fill_(~mask, img_H * img_W - 1) # Produces (B1, B2, N) tensor
        img_num_pt = torch_scatter.scatter_sum(torch.ones_like(clouds_in_plane_pixels_flattened, dtype=torch.int32), clouds_in_plane_pixels_flattened, dim=-1, dim_size=img_H * img_W) # Produces (B1, B2 H*W) tensor
        img_num_pt[:, :, -1] -= mask_num # Produces (B1, B2, H*W) tensor
        img_num_pixel = torch.gt(img_num_pt, 0).type(torch.int32) # Produces (B1, B2, H*W) tensor
        overlap_pixels_num = torch.count_nonzero(img_num_pixel, dim=-1) # Produces (B1, B2) tensor 

        image_overlap_ratio = overlap_pixels_num * 1.0 / (img_H * img_W) # Produces (B1, B2) tensor
        pc_overlap_ratio = overlap_points_num * 1.0 / N # Produces (B1, B2) tensor
        overlap_ratio = torch.minimum(image_overlap_ratio, pc_overlap_ratio)
        overlap_ratio = overlap_ratio / (torch.max(overlap_ratio) + 1e-6)
        overlap_ratio_ndarray = overlap_ratio.cpu().numpy() # Produces (B1, B2) ndarray
        curr_overlap_ratio_list.append(overlap_ratio_ndarray)

        print(f'{idx_pc} to {idx_img} done!')
        
    curr_overlap_ratio_ndarray = np.concatenate(curr_overlap_ratio_list, axis=1) # Produces (B1, all_train_pointcloud_list_length) ndarray
    all_overlap_ratio_list.append(curr_overlap_ratio_ndarray)

all_overlap_ratio = np.concatenate(all_overlap_ratio_list, axis=0)
all_overlap_ratio = (all_overlap_ratio + all_overlap_ratio.T) / 2.0
train_overlap_save_path = os.path.join(dataset_root, 'my_tool', f'train_point_cloud_projection_overlap.npy')
np.save(train_overlap_save_path, all_overlap_ratio)
print(f'save {dataset_root}/{train_overlap_save_path} done!')