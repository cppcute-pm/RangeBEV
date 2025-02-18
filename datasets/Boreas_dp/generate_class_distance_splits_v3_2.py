import os
import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import KDTree
from mini_boreas import BoreasDataset_U
import matplotlib.pyplot as plt
import random

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
the_radio = 30.0 # the radio of the sector
the_angle = 81.0 / 180.0 * np.pi # the total angle of the sector, and the 81.0 is the HFoV of the camera
the_radio_split = 16
# additional_num_per_ratio_dict = {'type_name': 'type_1',
#                                  'num_list': [1, 1, 1, 1,
#                                             2, 2, 2, 2,
#                                             3, 3, 3, 3,
#                                             4, 4, 4, 4], 
#                             }

# additional_num_per_ratio_dict = {'type_name': 'type_2',
#                                  'num_list': [1, 1, 1, 1,
#                                             4, 4, 4, 4,
#                                             6, 6, 6, 6,
#                                             8, 8, 8, 8],}





# additional_num_per_ratio_dict = {'type_name': 'type_3',
#                                  'num_list': [1, 1, 1, 1,
#                                             4, 4, 4, 4,
#                                             8, 8, 8, 8,
#                                             12, 12, 12, 12],}





# additional_num_per_ratio_dict = {'type_name': 'type_4',
#                                  'num_list': [1, 1, 1, 1,
#                                             6, 6, 6, 6,
#                                             11, 11, 11, 11,
#                                             16, 16, 16, 16],}

the_base_angel_points_num = 2

all_train_UTM_coords = []

pc_dir_name = "Boreas_minuse_40960_to_4096_cliped_fov"
pc_path = os.path.join(data_root, pc_dir_name)
lidar_2_image_idx_path = os.path.join(dataset_root, 'my_tool/lidar2image.pickle')
lidar_2_image_idx = pickle.load(open(lidar_2_image_idx_path, 'rb'))



for sequence in dataset.sequences:
    lidar_pose = []
    lidar_ids = []
    for lidar_id in minuse_lidar[sequence.ID]:
        curr_lidar_pose = []
        lidar_ids.append(lidar_id)
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
        alpha = np.arctan2(curr_image_pose[1, 2], curr_image_pose[0, 2])
        
        curr_lidar_pose_x = curr_lidar.pose[0, 3].astype(np.float32)
        curr_lidar_pose_y = curr_lidar.pose[1, 3].astype(np.float32)
        curr_lidar_pose.append(curr_lidar_pose_x)
        curr_lidar_pose.append(curr_lidar_pose_y)
        for i in range(the_radio_split):
            the_angle_split = (i + 1) * the_base_angel_points_num
            for k in range(additional_num_per_ratio_dict['num_list'][i]):
                for j in range(the_angle_split):
                    curr_angle = alpha - the_angle / 2 + j * the_angle / (the_angle_split - 1)
                    curr_radio = the_radio * (i + 1) / the_radio_split
                    curr_x = curr_lidar_pose_x + curr_radio * np.cos(curr_angle)
                    curr_y = curr_lidar_pose_y + curr_radio * np.sin(curr_angle)
                    curr_x = curr_x.astype(np.float32)
                    curr_y = curr_y.astype(np.float32)
                    curr_lidar_pose.append(curr_x)
                    curr_lidar_pose.append(curr_y)
        # lidar_pose_x = curr_lidar_pose[0::2]
        # lidar_pose_y = curr_lidar_pose[1::2]
        # lidar_point_x = pc_INS[:, 0]
        # lidar_point_y = pc_INS[:, 1]
        # plt.scatter(lidar_pose_x, lidar_pose_y, s=float(1/10), color='red')
        # plt.scatter(lidar_point_x, lidar_point_y, s=float(1/10), color='blue')
        # plt.xlabel('Longitude (x)')
        # plt.ylabel('Latitude (y)')
        # plt.title('Lidar global pose')
        # plt.savefig(f'/home/test5/code_project/visualization0701/lidar_pose_{sequence.ID}_{lidar_id}.png', bbox_inches='tight', pad_inches=0, dpi=200)
        # plt.close()
        lidar_pose.append(curr_lidar_pose)
    UTM_coords = np.array(lidar_pose, dtype=np.float32)
    lidar_ids = np.array(lidar_ids)
    test_flags = check_in_specified_set(UTM_coords[:, 0], UTM_coords[:, 1], P_test)
    train_UTM_coords = UTM_coords[~test_flags]

    all_train_UTM_coords.append(train_UTM_coords)
    print(f'visit {sequence.ID} done!')

all_train_UTM_coords = np.concatenate(all_train_UTM_coords, axis=0)
type_name = additional_num_per_ratio_dict['type_name']
train_save_path = os.path.join(dataset_root, 'my_tool', f'train_UTM_coords_v3_1_radio_{int(the_radio)}m_additional_num_per_radio_type_{type_name}.npy')
np.save(train_save_path, all_train_UTM_coords)
print(f'save {dataset_root}/{train_save_path} done!')