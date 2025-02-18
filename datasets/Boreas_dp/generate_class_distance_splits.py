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
all_train_list = []
all_test_list = []
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

    all_train_list.append(train_list)
    all_test_list.append(test_list)
    all_train_x_poses_class.append(train_x_poses_classes)
    all_train_y_poses_class.append(train_y_poses_classes)
    all_test_x_poses_class.append(test_x_poses_classes)
    all_test_y_poses_class.append(test_y_poses_classes)
    all_train_UTM_coords.append(train_UTM_coords)
    all_test_UTM_coords.append(test_UTM_coords)

    print(f'visit {sequence.ID} done!')
train_traversal_cumsum = np.cumsum([len(x) for x in all_train_list])
test_traversal_cumsum = np.cumsum([len(x) for x in all_test_list])
all_train_UTM_coords = np.concatenate(all_train_UTM_coords, axis=0)
all_test_UTM_coords = np.concatenate(all_test_UTM_coords, axis=0)


test_tree = KDTree(all_test_UTM_coords)
test_neighbor_idxs = test_tree.query_radius(all_test_UTM_coords, r=true_neighbor_distance)
train_tree = KDTree(all_train_UTM_coords)
levels_positive_idxs = []
train_positive_idxs = train_tree.query_radius(all_train_UTM_coords, r=positive_distance)
train_non_negative_idxs = train_tree.query_radius(all_train_UTM_coords, r=non_negative_distance)
for curr_level_positive_distance in levels_postive_distance:
    levels_positive_idxs.append(train_tree.query_radius(all_train_UTM_coords, r=curr_level_positive_distance))

x_poses_class_num = max_x_poses_class_index - min_x_poses_class_index + 1
y_poses_class_num = max_y_poses_class_index - min_y_poses_class_index + 1

class_num_path = os.path.join(dataset_root, 'my_tool', 'class_split_num_v3.pickle')
if os.path.exists(class_num_path):
    os.remove(class_num_path)
class_num_info = {'x_poses_class_num': int(x_poses_class_num), 
                  'y_poses_class_num': int(y_poses_class_num), 
                  'total_class_num': int(x_poses_class_num * y_poses_class_num)}
with open(class_num_path, 'wb') as handle:
    pickle.dump(class_num_info, handle, protocol=pickle.HIGHEST_PROTOCOL)
print(f'successfully write class_num_info!')

print(f'total_class_num: {int(x_poses_class_num * y_poses_class_num)}')
train_queries = []
test_queries = []
for i, sequence in enumerate(dataset.sequences):
    curr_train_x_poses_class = all_train_x_poses_class[i] - min_x_poses_class_index
    curr_train_y_poses_class = all_train_y_poses_class[i] - min_y_poses_class_index
    curr_test_x_poses_class = all_test_x_poses_class[i] - min_x_poses_class_index
    curr_test_y_poses_class = all_test_y_poses_class[i] - min_y_poses_class_index
    for j, id in enumerate(all_train_list[i]):
        curr_train_query = {'sequence_ID': sequence.ID, \
                            'idx': int(id), \
                            'class': int(curr_train_x_poses_class[j] * y_poses_class_num + curr_train_y_poses_class[j])}
        curr_train_query['neighbor_classes'] = []
        for k in range(-1, 2, 1):
            for l in range(-1, 2, 1):
                if curr_train_x_poses_class[j] + k >= 0 and curr_train_x_poses_class[j] + k < x_poses_class_num and curr_train_y_poses_class[j] + l >= 0 and curr_train_y_poses_class[j] + l < y_poses_class_num:
                    curr_train_query['neighbor_classes'].append(int((curr_train_x_poses_class[j] + k) * y_poses_class_num + curr_train_y_poses_class[j] + l))
        curr_train_query['neighbor_classes'].remove(int(curr_train_x_poses_class[j] * y_poses_class_num + curr_train_y_poses_class[j]))

        if i == 0:
            total_idx = j
        else:
            total_idx = train_traversal_cumsum[i-1] + j
        curr_train_query['level_positives'] = {}
        for k in range(len(levels_positive_idxs)):
            if k < len(levels_positive_idxs) - 1:
                curr_level_positive_idxs = levels_positive_idxs[len(levels_positive_idxs) - k - 1]
                curr_level_positive_idxs = np.sort(curr_level_positive_idxs[total_idx])
                higher_level_positive_idxs = (levels_positive_idxs[len(levels_positive_idxs) - k - 2])[total_idx]
                delete_idxs = np.searchsorted(curr_level_positive_idxs, higher_level_positive_idxs)
                level_positive_idxs_inuse = np.delete(curr_level_positive_idxs, delete_idxs)
            else:
                level_positive_idxs_inuse = np.sort((levels_positive_idxs[len(levels_positive_idxs) - k - 1])[total_idx])
            curr_train_query['level_positives'][f'level_{k+1}'] = sorted(list(level_positive_idxs_inuse))

        curr_train_query['positives'] = list(np.sort(train_positive_idxs[total_idx]))
        curr_train_query['non_negatives'] = list(np.sort(train_non_negative_idxs[total_idx]))
        train_queries.append(curr_train_query)
        print(f'add the file {str(id)} to train queries!')
    for j, id in enumerate(all_test_list[i]):
        curr_test_query = {'sequence_ID': sequence.ID, \
                           'idx': int(id), \
                            'class': int(curr_test_x_poses_class[j] * y_poses_class_num + curr_test_y_poses_class[j])}
        curr_test_query['neighbor_classes'] = []
        for k in range(-1, 2, 1):
            for l in range(-1, 2, 1):
                if curr_test_x_poses_class[j] + k >= 0 and curr_test_x_poses_class[j] + k < x_poses_class_num and curr_test_y_poses_class[j] + l >= 0 and curr_test_y_poses_class[j] + l < y_poses_class_num:
                    curr_test_query['neighbor_classes'].append(int((curr_test_x_poses_class[j] + k) * y_poses_class_num + curr_test_y_poses_class[j] + l))
        curr_test_query['neighbor_classes'].remove(int(curr_test_x_poses_class[j] * y_poses_class_num + curr_test_y_poses_class[j]))

        if i == 0:
            total_idx = j
        else:
            total_idx = test_traversal_cumsum[i-1] + j
        curr_test_query['true_neighbors'] = list(np.sort(test_neighbor_idxs[total_idx]))

        test_queries.append(curr_test_query)
        print(f'add the file {str(id)} to test queries!')

file_path = os.path.join(dataset_root, 'class_split_train_queries_v3.pickle')
if os.path.exists(file_path):
    os.remove(file_path)
with open(file_path, 'wb') as handle:
    pickle.dump(train_queries, handle, protocol=pickle.HIGHEST_PROTOCOL)
print(f'successfully write {len(train_queries)} train queries!')
file_path = os.path.join(dataset_root, 'class_split_test_queries_v3.pickle')
if os.path.exists(file_path):
    os.remove(file_path)
with open(file_path, 'wb') as handle:
    pickle.dump(test_queries, handle, protocol=pickle.HIGHEST_PROTOCOL)
print(f'successfully write {len(test_queries)} test queries!')
print('Done!')