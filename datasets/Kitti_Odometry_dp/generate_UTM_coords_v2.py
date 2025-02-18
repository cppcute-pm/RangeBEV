import os
import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import random
from my_pykitti_odometry import my_odometry
import open3d as o3d


# pos_vec_vet and exp_dist of KITTI dataset

# use the LiDAR coordinate system as the world coordinate system

data_root = '/DATA1/pengjianyi'
dataset_root = os.path.join(data_root, 'KITTI/dataset')
pose_root = os.path.join(data_root, 'semanticKITTI/dataset')
tool_path = os.path.join(data_root, 'KITTI', 'my_tool')
sequence_list = ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
pc_inuse_root = os.path.join(data_root, 'KITTI/16384_to_4096_cliped_fov')
ahead_distance = 3.0

all_train_UTM_coords = {}
for seq_ID in sequence_list:
    curr_sequence = my_odometry(sequence=seq_ID, base_path=dataset_root, pose_path=pose_root)
    UTM_coords = []
    for id in range(len(curr_sequence.timestamps)):
        curr_UTM_coords = []

        T_first_cam0_curr_cam0 = curr_sequence.poses[id]
        curr_calib = curr_sequence.calib
        T_cam0_LiDAR = curr_calib['T_ego_LiDAR']

        T_first_cam0_curr_LiDAR = np.matmul(T_first_cam0_curr_cam0, T_cam0_LiDAR).astype(np.float32)
        T_first_LiDAR_curr_LiDAR = np.matmul(np.linalg.inv(T_cam0_LiDAR), T_first_cam0_curr_LiDAR)

        alpha = np.arctan2(T_first_LiDAR_curr_LiDAR[1, 0], T_first_LiDAR_curr_LiDAR[0, 0])

        ahead_pos_x = T_first_LiDAR_curr_LiDAR[0, 3].astype(np.float32) + ahead_distance * np.cos(alpha)
        ahead_pos_y = T_first_LiDAR_curr_LiDAR[1, 3].astype(np.float32) + ahead_distance * np.sin(alpha)

        curr_UTM_coords.append(ahead_pos_x.astype(np.float32))
        curr_UTM_coords.append(ahead_pos_y.astype(np.float32))
        curr_UTM_coords.append(T_first_LiDAR_curr_LiDAR[0, 3].astype(np.float32))
        curr_UTM_coords.append(T_first_LiDAR_curr_LiDAR[1, 3].astype(np.float32))
        UTM_coords.append(curr_UTM_coords)
    train_UTM_coords = np.array(UTM_coords, dtype=np.float32)
    all_train_UTM_coords[seq_ID] = train_UTM_coords
    print(f'visit {seq_ID} done!')

train_save_path = os.path.join(tool_path, f'train_UTM_coords_v2_ahead_dist_{int(ahead_distance)}m.pkl')
with open(train_save_path, 'wb') as f:
    pickle.dump(all_train_UTM_coords, f)
print(f'save {dataset_root}/{train_save_path} done!')