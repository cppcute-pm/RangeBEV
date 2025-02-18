import os
import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import random
from my_pykitti_odometry import my_odometry
import open3d as o3d
from shapely.geometry import Polygon


# sector area overlap of KITTI dataset
def create_sector(center, radius, start_angle_rad, end_angle_rad, num_points=500):
    # Convert degrees to radians
    
    # Generate points along the arc of the sector
    arc_points = [
        (center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle))
        for angle in np.linspace(start_angle_rad, end_angle_rad, num_points)
    ]
    
    # Add the center point to close the sector
    points = [center] + arc_points + [center]

    # points_to_vis = np.array(points)
    # plt.scatter(points_to_vis[:, 0], points_to_vis[:, 1], s=float(1/10), color='green')
    # plt.xlabel('Longitude (x)')
    # plt.ylabel('Latitude (y)')
    # plt.title('polygon outlier points')
    # plt.savefig(f'/home/pengjianyi/code_projects/vis1015/lidar_pose.png', bbox_inches='tight', pad_inches=0, dpi=200)
    # plt.close()
    
    # Create a Polygon representing the sector
    sector = Polygon(points)
    return sector, points


# use the LiDAR coordinate system as the world coordinate system

data_root = '/DATA1/pengjianyi'
dataset_root = os.path.join(data_root, 'KITTI/dataset')
pose_root = os.path.join(data_root, 'semanticKITTI/dataset')
tool_path = os.path.join(data_root, 'KITTI', 'my_tool')
sequence_list = ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
the_radio = 10.0
the_angle = 90.0 / 180.0 * np.pi # the total angle of the sector, and the 90.0 is the HFoV of the camera

all_train_sector_dict = {}
all_train_pc_outlier_dict = {}
num_points = 100
for seq_ID in sequence_list:
    lidar_dir = os.path.join(data_root, 'KITTI/16384_to_4096_cliped_fov', seq_ID, 'velodyne')
    curr_sequence = my_odometry(sequence=seq_ID, base_path=dataset_root, pose_path=pose_root)
    sector_list = []
    pc_outlier_list = []
    for id in range(len(curr_sequence.timestamps)):

        T_first_cam0_curr_cam0 = curr_sequence.poses[id]
        curr_calib = curr_sequence.calib
        T_cam0_LiDAR = curr_calib['T_ego_LiDAR']

        T_first_cam0_curr_LiDAR = np.matmul(T_first_cam0_curr_cam0, T_cam0_LiDAR).astype(np.float32)
        T_first_LiDAR_curr_LiDAR = np.matmul(np.linalg.inv(T_cam0_LiDAR), T_first_cam0_curr_LiDAR)

        file_name = str(id).zfill(6)
        lidar_path = os.path.join(lidar_dir, file_name + '_1.npy')
        pc = np.load(lidar_path)
        pc_to_mult = np.concatenate((pc, np.ones((pc.shape[0], 1))), axis=1)
        pc_mult = np.matmul(pc_to_mult, T_first_LiDAR_curr_LiDAR.T)
        pc_INS = pc_mult[:, :3]

        alpha = np.arctan2(T_first_LiDAR_curr_LiDAR[1, 0], T_first_LiDAR_curr_LiDAR[0, 0])
        start_angle_rad = alpha - the_angle / 2
        end_angle_rad = alpha + the_angle / 2
        center = [T_first_LiDAR_curr_LiDAR[0, 3], T_first_LiDAR_curr_LiDAR[1, 3]]
        curr_sector, points_list = create_sector(center, the_radio, start_angle_rad, end_angle_rad, num_points=num_points)

        points_array = np.array(points_list) # (num_points, 2)
        # lidar_pose_x = points_array[:, 0]
        # lidar_pose_y = points_array[:, 1]
        # lidar_point_x = pc_INS[:, 0]
        # lidar_point_y = pc_INS[:, 1]
        # plt.scatter(lidar_pose_x, lidar_pose_y, s=float(1/10), color='red')
        # plt.scatter(lidar_point_x, lidar_point_y, s=float(1/10), color='blue')
        # plt.xlabel('Longitude (x)')
        # plt.ylabel('Latitude (y)')
        # plt.title('Lidar global pose')
        # plt.savefig(f'/home/pengjianyi/code_projects/vis1209/lidar_pose_{seq_ID}_{file_name}.png', bbox_inches='tight', pad_inches=0, dpi=200)
        # plt.close()

        sector_list.append(curr_sector)
        pc_outlier_list.append(points_array)
    all_train_sector_dict[seq_ID] = sector_list
    all_train_pc_outlier_dict[seq_ID] = pc_outlier_list
    print(f'visit {seq_ID} done!')

all_overlap_ratio_dict = {}
for curr_seq_ID, curr_seq_sector_list in all_train_sector_dict.items():
    curr_seq_pc_outlier_list = all_train_pc_outlier_dict[curr_seq_ID]
    idx1_overlap_ratio_list = []
    for idx_1, sector_1 in enumerate(curr_seq_sector_list):
        pc_outlier_1 = curr_seq_pc_outlier_list[idx_1]
        idx2_overlap_ratio_list = []
        for idx_2, sector_2 in enumerate(curr_seq_sector_list):
            overlap_area = sector_1.intersection(sector_2).area
            overlap_ratio_1_to_2 = overlap_area / (sector_1.area + sector_2.area) * 2
            idx2_overlap_ratio_list.append(overlap_ratio_1_to_2)

            # pc_outlier_2 = curr_seq_pc_outlier_list[idx_2]
            # pc_outlier_1_x = pc_outlier_1[:, 0]
            # pc_outlier_1_y = pc_outlier_1[:, 1]
            # pc_outlier_2_x = pc_outlier_2[:, 0]
            # pc_outlier_2_y = pc_outlier_2[:, 1]
            # plt.scatter(pc_outlier_1_x, pc_outlier_1_y, s=float(1/10), color='red')
            # plt.scatter(pc_outlier_2_x, pc_outlier_2_y, s=float(1/10), color='blue')
            # plt.xlabel('Longitude (x)')
            # plt.ylabel('Latitude (y)')
            # plt.title(f'points outlier {idx_1} to {idx_2} overlap ratio: {overlap_ratio_1_to_2}')
            # plt.savefig(f'/home/pengjianyi/code_projects/vis1209/points_outlier_seq_{curr_seq_ID}_{idx_1}_{idx_2}.png', bbox_inches='tight', pad_inches=0, dpi=200)
            # plt.close()

            print(f'visit {idx_1} to {idx_2} done!')
        idx2_overlap_ratio_array = np.array(idx2_overlap_ratio_list)
        idx1_overlap_ratio_list.append(idx2_overlap_ratio_array)
    idx1_overlap_ratio_array = np.stack(idx1_overlap_ratio_list, axis=0)
    all_overlap_ratio_dict[curr_seq_ID] = idx1_overlap_ratio_array

train_overlap_save_path = os.path.join(tool_path, f'train_area_overlap_{int(the_radio)}m.pkl')
if os.path.exists(train_overlap_save_path):
    os.remove(train_overlap_save_path)
with open(train_overlap_save_path, 'wb') as f:
    pickle.dump(all_overlap_ratio_dict, f)
print(f'save train overlap ratio done!')