import os
import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import random
from my_pykitti_odometry import my_odometry
import open3d as o3d


# points average distance of KITTI dataset

# use the LiDAR coordinate system as the world coordinate system

data_root = '/DATA1/pengjianyi'
dataset_root = os.path.join(data_root, 'KITTI/dataset')
pose_root = os.path.join(data_root, 'semanticKITTI/dataset')
tool_path = os.path.join(data_root, 'KITTI', 'my_tool')
sequence_list = ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
pc_inuse_root = os.path.join(data_root, 'KITTI/16384_to_4096_cliped_fov')

the_radio = 15.0 # the radio of the sector
the_angle = 90.0 / 180.0 * np.pi # the total angle of the sector, and the 81.0 is the HFoV of the camera
the_radio_split = 16
the_base_angel_points_num = 2

all_train_UTM_coords = {}
for seq_ID in sequence_list:
    curr_sequence = my_odometry(sequence=seq_ID, base_path=dataset_root, pose_path=pose_root)
    lidar_pose = []
    UTM_coords = []
    for id in range(len(curr_sequence.timestamps)):
        curr_UTM_coords = []

        T_first_cam0_curr_cam0 = curr_sequence.poses[id]
        curr_calib = curr_sequence.calib
        T_cam0_LiDAR = curr_calib['T_ego_LiDAR']

        T_first_cam0_curr_LiDAR = np.matmul(T_first_cam0_curr_cam0, T_cam0_LiDAR).astype(np.float32)
        T_first_LiDAR_curr_LiDAR = np.matmul(np.linalg.inv(T_cam0_LiDAR), T_first_cam0_curr_LiDAR)

        alpha = np.arctan2(T_first_LiDAR_curr_LiDAR[1, 0], T_first_LiDAR_curr_LiDAR[0, 0])
        curr_UTM_coords.append(T_first_LiDAR_curr_LiDAR[0, 3])
        curr_UTM_coords.append(T_first_LiDAR_curr_LiDAR[1, 3])

        # filename_preffix = str(id).zfill(6)
        # curr_pc_inuse_path = os.path.join(pc_inuse_root, seq_ID, 'velodyne', filename_preffix + '_2.npy')
        # pc = np.load(curr_pc_inuse_path)
        # pc = pc[:, :3].astype(np.float32) # shape: (N, 3)
        # pc_to_mult = np.concatenate([pc, np.ones_like(pc[:, -1:])], axis=-1) # shape: (N, 4)
        # pc_first_LiDAR = np.dot(pc_to_mult, T_first_LiDAR_curr_LiDAR.T) # shape: (N, 4)
        # pc_first_LiDAR = pc_first_LiDAR[:, :3] # shape: (N, 3)


        for i in range(the_radio_split):
            the_angle_split = (i + 1) * the_base_angel_points_num
            for j in range(the_angle_split):
                curr_angle = alpha - the_angle / 2 + j * the_angle / (the_angle_split - 1)
                curr_radio = the_radio * (i + 1) / the_radio_split
                curr_x = T_first_LiDAR_curr_LiDAR[0, 3] + curr_radio * np.cos(curr_angle)
                curr_y = T_first_LiDAR_curr_LiDAR[1, 3] + curr_radio * np.sin(curr_angle)
                curr_x = curr_x.astype(np.float32)
                curr_y = curr_y.astype(np.float32)
                curr_UTM_coords.append(curr_x)
                curr_UTM_coords.append(curr_y)
        # pose_x = curr_UTM_coords[0::2]
        # pose_y = curr_UTM_coords[1::2]
        # lidar_point_x = pc_first_LiDAR[:, 0]
        # lidar_point_y = pc_first_LiDAR[:, 1]
        # plt.scatter(pose_x, pose_y, s=float(1/10), color='red')
        # plt.scatter(lidar_point_x, lidar_point_y, s=float(1/10), color='blue')
        # plt.xlabel('Longitude (x)')
        # plt.ylabel('Latitude (y)')
        # plt.title('Lidar global pose')
        # plt.savefig(f'/home/pengjianyi/code_projects/vis1027/lidar_pose_{seq_ID}_{id}.png', bbox_inches='tight', pad_inches=0, dpi=200)
        # plt.close()
        UTM_coords.append(curr_UTM_coords)
    train_UTM_coords = np.array(UTM_coords, dtype=np.float32)

    all_train_UTM_coords[seq_ID] = train_UTM_coords
    print(f'visit {seq_ID} done!')

the_radio_str = str(the_radio).replace('.', 'p')
train_save_path = os.path.join(tool_path, f'train_UTM_coords_v1_{the_radio_str}m.pkl')
with open(train_save_path, 'wb') as f:
    pickle.dump(all_train_UTM_coords, f)
print(f'save {dataset_root}/{train_save_path} done!')