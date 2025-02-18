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
import torch_scatter

# pointcloud mnn of KITTI dataset

# use the LiDAR coordinate system as the world coordinate system

data_root = '/DATA1/pengjianyi'
dataset_root = os.path.join(data_root, 'KITTI/dataset')
pose_root = os.path.join(data_root, 'semanticKITTI/dataset')
tool_path = os.path.join(data_root, 'KITTI', 'my_tool')
sequence_list = ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
pc_inuse_root = os.path.join(data_root, 'KITTI/16384_to_4096_cliped_fov')
image_inuse_root = os.path.join(data_root, 'KITTI/768x128_image')
device = torch.device('cuda:1')
torch.cuda.set_device(device)


all_train_pointcloud_dict = {}
all_train_pointcloud_pose_dict = {}
all_train_image_pose_dict = {}
all_train_image_intrinsics_dict = {}

for seq_ID in sequence_list:
    curr_sequence = my_odometry(sequence=seq_ID, base_path=dataset_root, pose_path=pose_root)
    pointcloud_list = []
    pointcloud_pose_list = []
    image_pose_list = []
    image_intrinsics_list = []
    for id in range(len(curr_sequence.timestamps)):

        T_first_cam0_curr_cam0 = curr_sequence.poses[id]
        curr_calib = curr_sequence.calib
        T_cam0_LiDAR = curr_calib['T_ego_LiDAR']
        T_cam0_cam2 = curr_calib['T_ego_cam2']

        T_first_cam0_curr_LiDAR = np.matmul(T_first_cam0_curr_cam0, T_cam0_LiDAR).astype(np.float32)
        T_first_cam0_curr_cam2 = np.matmul(T_first_cam0_curr_cam0, T_cam0_cam2).astype(np.float32)

        file_name = str(id).zfill(6)
        pc_file_path = os.path.join(pc_inuse_root, seq_ID, 'velodyne', file_name + '_1.npy')
        pc_cliped = np.load(pc_file_path)

        pointcloud_list.append(pc_cliped)
        pointcloud_pose_list.append(T_first_cam0_curr_LiDAR)

        image_pose_list.append(T_first_cam0_curr_cam2)

        image_intrinsics_path = os.path.join(image_inuse_root, seq_ID, 'image_2_intrinsic', file_name + '.npy')
        cam2_K = np.load(image_intrinsics_path)
        image_intrinsics_list.append(cam2_K)
    
    all_train_pointcloud_dict[seq_ID] = np.stack(pointcloud_list, axis=0)
    all_train_pointcloud_pose_dict[seq_ID] = np.stack(pointcloud_pose_list, axis=0)
    all_train_image_pose_dict[seq_ID] = np.stack(image_pose_list, axis=0)
    all_train_image_intrinsics_dict[seq_ID] = np.stack(image_intrinsics_list, axis=0)
    print(f'visit {seq_ID} done!')

all_overlap_ratio_dict = {}
step = 48
point_project_threshold = 50.0
img_H = 224
img_W = 224
N = 40960

with torch.no_grad():
    for seq_ID, train_pointcloud_coords in all_train_pointcloud_dict.items():
        curr_seq_overlap_ratio = []
        train_pointcloud_pose = all_train_pointcloud_pose_dict[seq_ID]
        train_image_pose = all_train_image_pose_dict[seq_ID]
        train_image_intrinsics = all_train_image_intrinsics_dict[seq_ID]
        for idx_pc in range(0, train_pointcloud_coords.shape[0], step):
            idx_pc_start = idx_pc
            idx_pc_end = min(idx_pc_start + step, train_pointcloud_coords.shape[0])
            pc_ndarray = train_pointcloud_coords[idx_pc_start:idx_pc_end] # Produces (B1, N, 3) ndarray
            pc_tensor = torch.tensor(pc_ndarray, dtype=torch.float32, device=device) # Produces (B1, N, 3) tensor
            pc_pose_ndarray = np.stack(train_pointcloud_pose[idx_pc_start:idx_pc_end], axis=0) # Produces (B1, 4, 4) ndarray
            pc_pose_tensor = torch.tensor(pc_pose_ndarray, dtype=torch.float32, device=device) # Produces (B1, 4, 4) tensor
            B1 = pc_tensor.shape[0]

            curr_seq_idx1_overlap_ratio = []
            for idx_img in range(0, train_pointcloud_coords.shape[0], step):
                idx_img_start = idx_img
                idx_img_end = min(idx_img_start + step, train_pointcloud_coords.shape[0])
                img_pose_ndarray = train_image_pose[idx_img_start:idx_img_end] # Produces (B2, 4, 4) ndarray
                img_pose_tensor = torch.tensor(img_pose_ndarray, dtype=torch.float32, device=device) # Produces (B2, 4, 4) tensor
                img_intrinsics_ndarray = train_image_intrinsics[idx_img_start:idx_img_end] # Produces (B2, 3, 3) ndarray
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

                curr_seq_idx1_overlap_ratio.append(overlap_ratio_ndarray)
            curr_seq_overlap_ratio.append(np.concatenate(curr_seq_idx1_overlap_ratio, axis=1))
        all_overlap_ratio_dict[seq_ID] = np.concatenate(curr_seq_overlap_ratio, axis=0)

train_overlap_save_path = os.path.join(dataset_root, 'my_tool', f'train_point_cloud_projection_overlap_{int(point_project_threshold)}.pkl')
if os.path.exists(train_overlap_save_path):
    os.remove(train_overlap_save_path)
with open(train_overlap_save_path, 'wb') as f:
    pickle.dump(all_overlap_ratio_dict, f)
print(f'save train_overlap_ratio done!')