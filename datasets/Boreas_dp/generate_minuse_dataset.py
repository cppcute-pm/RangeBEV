import os.path as osp
import numpy as np
import os
import pickle
from pyboreas.data.splits import loc_train
from mini_boreas import BoreasDataset_U
    
every_x_meter = 10.0 # 10m
data_root = '/media/data/pengjianyi/Boreas'
ds = BoreasDataset_U(root=data_root,
                     split=loc_train,
                     verbose=True)
print("initalize the dataset: done")

minuse_lidar_idxs = {}
for curr_sequence in ds.sequences:
    minuse_lidar_idxs[curr_sequence.ID] = []
    lidar_iter = curr_sequence.get_lidar_iter()
    for idx, lidar in enumerate(lidar_iter):
        if idx == 0:
            minuse_lidar_idxs[curr_sequence.ID].append(idx)
            _prev_pose = lidar.pose
            _prev_idx = idx
        else:
            distance = np.linalg.norm(lidar.pose[:3, 3] - _prev_pose[:3, 3])
            if distance >= every_x_meter:
                minuse_lidar_idxs[curr_sequence.ID].append(idx)
                _prev_pose = lidar.pose
                delta_idx = idx - _prev_idx
                _prev_idx = idx
                print(f'traversal: {curr_sequence.ID}  idx: {idx}  distance: {distance}  delta_idx: {delta_idx}')

print("minuse_lidar_idxs: done")
print('start saving the file')

new_tool_path = osp.join(data_root, 'my_tool')
if not osp.exists(new_tool_path):
    os.makedirs(new_tool_path)
save_path = osp.join(new_tool_path, 'minuse_lidar_idxs.pickle')
with open(save_path, 'wb') as handle:
    pickle.dump(minuse_lidar_idxs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
print(f'save the file to {save_path} done')
