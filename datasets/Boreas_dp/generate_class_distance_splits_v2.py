import os
import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import KDTree
from mini_boreas import BoreasDataset_U

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

class_bound = 20.0
levels_postive_distance = range(10, 160, 10)
positive_distance = 10
non_negative_distance = 50
true_neighbor_distance = 25
dataset_root = '/media/group2/data/pengjianyi/Boreas_minuse'
dataset = BoreasDataset_U(dataset_root)
minuse_lidar_idxs_path = os.path.join(dataset_root, 'my_tool', 'minuse_lidar_idxs.pickle')
minuse_lidar = pickle.load(open(minuse_lidar_idxs_path, 'rb'))

max_x_poses_class_index = 0
max_y_poses_class_index = 0
min_y_poses_class_index = 0
min_x_poses_class_index = 0
all_train_dict = {}
all_test_dict = {}
all_train_x_poses_class = []
all_train_y_poses_class = []
all_test_x_poses_class = []
all_test_y_poses_class = []
all_train_UTM_coords = []
all_test_UTM_coords = []

start_flag = True

for sequence in dataset.sequences:
    lidar_pose_x = []
    lidar_pose_y = []
    lidar_ids = []
    for lidar_id in minuse_lidar[sequence.ID]:
        lidar_ids.append(lidar_id)
        curr_lidar = sequence.lidar_frames[lidar_id]
        curr_lidar_pose_x = curr_lidar.pose[0, 3]
        curr_lidar_pose_y = curr_lidar.pose[1, 3]
        lidar_pose_x.append(curr_lidar_pose_x)
        lidar_pose_y.append(curr_lidar_pose_y)
    lidar_ids = np.array(lidar_ids)
    lidar_pose_x = np.array(lidar_pose_x)
    lidar_pose_y = np.array(lidar_pose_y)
    x_poses_classes = np.ceil(lidar_pose_x / class_bound)
    y_poses_classes = np.ceil(lidar_pose_y / class_bound)
    UTM_coords = np.stack((lidar_pose_x, lidar_pose_y), axis=-1)
    if not start_flag:
        min_y_poses_class_index = min(min_y_poses_class_index, np.min(y_poses_classes))
        min_x_poses_class_index = min(min_x_poses_class_index, np.min(x_poses_classes))
    else:
        start_flag = False
        min_y_poses_class_index = np.min(y_poses_classes)
        min_x_poses_class_index = np.min(x_poses_classes)
    max_y_poses_class_index = max(max_y_poses_class_index, np.max(y_poses_classes))
    max_x_poses_class_index = max(max_x_poses_class_index, np.max(x_poses_classes))
    test_flags = check_in_specified_set(lidar_pose_x, lidar_pose_y, P_test)
    test_list = lidar_ids[test_flags]
    train_list = lidar_ids[~test_flags]

    test_x_poses_classes = x_poses_classes[test_flags]
    test_y_poses_classes = y_poses_classes[test_flags]
    test_UTM_coords = UTM_coords[test_flags]
    train_x_poses_classes = x_poses_classes[~test_flags]
    train_y_poses_classes = y_poses_classes[~test_flags]
    train_UTM_coords = UTM_coords[~test_flags]

    all_train_dict[sequence.ID] = list(train_list)
    all_test_dict[sequence.ID] = list(test_list)
    all_train_UTM_coords.append(train_UTM_coords)
    all_test_UTM_coords.append(test_UTM_coords)

    print(f'visit {sequence.ID} done!')

all_train_UTM_coords = np.concatenate(all_train_UTM_coords, axis=0)
all_test_UTM_coords = np.concatenate(all_test_UTM_coords, axis=0)
train_minuse_lidar_path = os.path.join(dataset_root, 'my_tool', 'train_minuse_lidar_idx.pickle')
if os.path.exists(train_minuse_lidar_path):
    os.remove(train_minuse_lidar_path)
pickle.dump(all_train_dict, open(train_minuse_lidar_path, 'wb'))
test_minuse_lidar_path = os.path.join(dataset_root, 'my_tool', 'test_minuse_lidar_idx.pickle')
if os.path.exists(test_minuse_lidar_path):
    os.remove(test_minuse_lidar_path)
pickle.dump(all_test_dict, open(test_minuse_lidar_path, 'wb'))
train_save_path = os.path.join(dataset_root, 'my_tool', 'train_UTM_coords.npy')
np.save(train_save_path, all_train_UTM_coords)
test_save_path = os.path.join(dataset_root, 'my_tool', 'test_UTM_coords.npy')
np.save(test_save_path, all_test_UTM_coords)