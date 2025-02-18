import os
import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import KDTree
from mini_boreas import BoreasDataset_U
import matplotlib.pyplot as plt
import random
from shapely.geometry import Polygon, Point

# this version is for area overlap


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
the_radio = 35.0 # the radio of the sector
the_angle = 81.0 / 180.0 * np.pi # the total angle of the sector, and the 81.0 is the HFoV of the camera

pc_dir_name = "Boreas_minuse_40960_to_4096_cliped_fov"
pc_path = os.path.join(data_root, pc_dir_name)
lidar_2_image_idx_path = os.path.join(dataset_root, 'my_tool/lidar2image.pickle')
lidar_2_image_idx = pickle.load(open(lidar_2_image_idx_path, 'rb'))

num_points = 100

all_train_sector_list = []
all_train_pc_outlier_list = []

for sequence in dataset.sequences:
    lidar_pose = []
    sector_list = []
    pc_outlier_list = []
    for lidar_id in minuse_lidar[sequence.ID]:
        curr_lidar_pose = []
        curr_lidar = sequence.lidar_frames[lidar_id]

        # lidar_pre_path = curr_lidar.path
        # seq_ID, lidar_dir, pc_file_name = lidar_pre_path.split('/')[-3:]
        # lidar_curr_path_prefix = os.path.join(pc_path, seq_ID, lidar_dir, pc_file_name.split('.')[0])
        # pc = np.load(lidar_curr_path_prefix + '_1.npy')
        # pc_pose = curr_lidar.pose.astype(np.float32)
        # pc_to_mult = np.concatenate([pc, np.ones_like(pc[:, -1:])], axis=-1) # shape: (N, 4)
        # pc_INS = np.dot(pc_to_mult, pc_pose.T) # shape: (N, 4)
        # pc_INS = pc_INS[:, :3] # shape: (N, 3)

        curr_image_idxs = lidar_2_image_idx[sequence.ID][str(lidar_id)]
        curr_image_idx = curr_image_idxs[0]
        curr_image_frame = sequence.camera_frames[curr_image_idx]
        curr_image_pose = curr_image_frame.pose.astype(np.float32)
        alpha = np.arctan2(curr_image_pose[1, 2].astype(np.float32), curr_image_pose[0, 2].astype(np.float32))
        
        curr_lidar_pose_x = curr_lidar.pose[0, 3].astype(np.float32)
        curr_lidar_pose_y = curr_lidar.pose[1, 3].astype(np.float32)
        curr_lidar_pose.append(curr_lidar_pose_x)
        curr_lidar_pose.append(curr_lidar_pose_y)

        start_angle_rad = alpha - the_angle / 2
        end_angle_rad = alpha + the_angle / 2
        center = [curr_image_pose[0, 3], curr_image_pose[1, 3]]
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
        # plt.savefig(f'/home/pengjianyi/code_projects/vis1017/lidar_pose_{sequence.ID}_{lidar_id}.png', bbox_inches='tight', pad_inches=0, dpi=200)
        # plt.close()

        lidar_pose.append(curr_lidar_pose)
        sector_list.append(curr_sector)
        pc_outlier_list.append(points_array)
    
    UTM_coords = np.array(lidar_pose, dtype=np.float32)
    test_flags = check_in_specified_set(UTM_coords[:, 0], UTM_coords[:, 1], P_test)
    test_indices = np.nonzero(test_flags)[0]
    curr_train_sector_list = []
    curr_train_pc_outlier_list = []
    for i in range(len(sector_list)):
        if i in test_indices:
            continue
        curr_train_sector_list.append(sector_list[i])
        curr_train_pc_outlier_list.append(pc_outlier_list[i])
    all_train_sector_list += curr_train_sector_list
    all_train_pc_outlier_list += curr_train_pc_outlier_list
    print(f'visit {sequence.ID} done!')

all_overlap_ratio_list = []
for idx_1, sector_1 in enumerate(all_train_sector_list):
    curr_overlap_ratio_list = []
    pc_outlier_1 = all_train_pc_outlier_list[idx_1]
    for idx_2, sector_2 in enumerate(all_train_sector_list):
        overlap_area = sector_1.intersection(sector_2).area
        overlap_ratio_1_to_2 = overlap_area / (sector_1.area + sector_2.area) * 2
        curr_overlap_ratio_list.append(overlap_ratio_1_to_2)

        pc_outlier_2 = all_train_pc_outlier_list[idx_2]


        # pc_outlier_1_x = pc_outlier_1[:, 0]
        # pc_outlier_1_y = pc_outlier_1[:, 1]
        # pc_outlier_2_x = pc_outlier_2[:, 0]
        # pc_outlier_2_y = pc_outlier_2[:, 1]
        # plt.scatter(pc_outlier_1_x, pc_outlier_1_y, s=float(1/10), color='red')
        # plt.scatter(pc_outlier_2_x, pc_outlier_2_y, s=float(1/10), color='blue')
        # plt.xlabel('Longitude (x)')
        # plt.ylabel('Latitude (y)')
        # plt.title(f'points outlier {idx_1} to {idx_2} overlap ratio: {overlap_ratio_1_to_2}')
        # plt.savefig(f'/home/pengjianyi/code_projects/vis1018/points_outlier_{idx_1}_{idx_2}.png', bbox_inches='tight', pad_inches=0, dpi=200)
        # plt.close()

        print(f'visit {idx_1} to {idx_2} done!')

    all_overlap_ratio_list.append(curr_overlap_ratio_list)

all_overlap_ratio = np.array(all_overlap_ratio_list, dtype=np.float32)

assert all_overlap_ratio.shape[0] == all_overlap_ratio.shape[1]

train_overlap_save_path = os.path.join(dataset_root, 'my_tool', f'train_area_overlap_{int(the_radio)}m.npy')
np.save(train_overlap_save_path, all_overlap_ratio)
print(f'save {dataset_root}/{train_overlap_save_path} done!')