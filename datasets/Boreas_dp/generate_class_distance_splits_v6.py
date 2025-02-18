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

dataset_root = '/DATA5/Boreas_minuse'
dataset = BoreasDataset_U(dataset_root)
minuse_lidar_idxs_path = os.path.join(dataset_root, 'my_tool', 'minuse_lidar_idxs.pickle')
minuse_lidar = pickle.load(open(minuse_lidar_idxs_path, 'rb'))

all_id_list = []
all_UTM_coords = []
all_test_flags = []

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
    UTM_coords = np.stack((lidar_pose_x, lidar_pose_y), axis=-1)
    test_flags = check_in_specified_set(lidar_pose_x, lidar_pose_y, P_test)

    all_id_list.append(list(lidar_ids))
    all_UTM_coords.append(UTM_coords)
    all_test_flags.append(test_flags)

    print(f'visit {sequence.ID} done!')

all_UTM_coords = np.concatenate(all_UTM_coords, axis=0)

all_queries = []
for i, sequence in enumerate(dataset.sequences):
    for j, id in enumerate(all_id_list[i]):
        curr_train_query = {'sequence_ID': sequence.ID, \
                            'idx': int(id), \
                            'is_test': all_test_flags[i][j]}
        all_queries.append(curr_train_query)
        print(f'add the file {str(id)} to all queries!')

all_test_queries_file_path = os.path.join(dataset_root, 'my_tool', 'all_test_queries.pickle')
if os.path.exists(all_test_queries_file_path):
    os.remove(all_test_queries_file_path)
with open(all_test_queries_file_path, 'wb') as handle:
    pickle.dump(all_queries, handle, protocol=pickle.HIGHEST_PROTOCOL)
all_test_UTM_coords_file_path = os.path.join(dataset_root, 'my_tool', 'all_test_UTM_coords.npy')
np.save(all_test_UTM_coords_file_path, all_UTM_coords)
print('save all test queries and UTM coords!')