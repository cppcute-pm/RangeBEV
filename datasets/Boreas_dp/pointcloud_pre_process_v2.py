import open3d as o3d
import numpy as np
from mini_boreas import BoreasDataset_U
import os
import pickle
from pyboreas.utils.utils import load_lidar
from pointcloud_process import FPS_downsample, voxel_downsample
from pointnet2_ops import pointnet2_utils
from multiprocessing import Pool
import torch

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def process_sequence(seq_list):
    seq_ID = seq_list[0]
    lidar_idxs = seq_list[1]
    print(f"enter {seq_ID}")
    curr_seq = dataset.get_seq_from_ID(seq_ID)
    target_seq_path = os.path.join(target_path, str(seq_ID))
    os.makedirs(target_seq_path, exist_ok=True)
    target_seq_lidar_path = os.path.join(target_seq_path, "lidar")
    os.makedirs(target_seq_lidar_path, exist_ok=True)
    for lidar_idx in lidar_idxs:
        curr_lidar_frame = curr_seq.lidar_frames[int(lidar_idx)]
        filename = curr_lidar_frame.path.split("/")[-1].split(".")[0]
        save_path_4096 = os.path.join(target_seq_lidar_path, filename + "_1.npy")
        save_path_163840 = os.path.join(target_seq_lidar_path, filename + "_2.npy")
        save_path_idx = os.path.join(target_seq_lidar_path, filename + "_3.npy")
        if os.path.exists(save_path_idx):
            print(f"pass the {lidar_idx} in {seq_ID}")
            continue
        curr_lidar_frame.load_data()
        curr_lidar_frame.remove_motion(curr_lidar_frame.body_rate)
        pc = curr_lidar_frame.points[:, :3].astype(np.float32)

        # pc_tensor = torch.tensor(pc).to(device)
        # fps_idx = pointnet2_utils.furthest_point_sample(pc_tensor, 163840).long() 
        # pc1_tensor = index_points(pc_tensor, fps_idx)  
        # fps_idx = pointnet2_utils.furthest_point_sample(pc1_tensor, 4096).long()
        # pc2_tensor = index_points(pc1_tensor, fps_idx)
        # pc1 = pc1_tensor.to('cpu').numpy()
        # pc2 = pc2_tensor.to('cpu').numpy()
        curr_voxel_size = voxel_size

        while True:
            pc1 = voxel_downsample(pc, curr_voxel_size)
            if pc1.shape[0] > 163840 * 1.01:
                curr_voxel_size *= 1.1
            elif pc1.shape[0] < 163840:
                curr_voxel_size *= 0.9
            else:
                break
        indices = np.random.permutation(163840)
        pc1 = pc1[indices, :]


        pc2 = FPS_downsample(pc1, 4096)
        pc1_tensor = torch.tensor(pc1)
        pc2_tensor = torch.tensor(pc2)


        pc1_2_pc2_dis = torch.cdist(pc1_tensor, pc2_tensor) # Produces (2 * 81920, 4096) tensor
        _, node12node2_idx_tensor = torch.topk(pc1_2_pc2_dis, k=1, dim=1, largest=False, sorted=False) # Produces (2 * 81920) tensor
        node12node2_idx = node12node2_idx_tensor.squeeze(-1).to('cpu').numpy()
        np.save(save_path_4096, pc2)
        np.save(save_path_163840, pc1)
        np.save(save_path_idx, node12node2_idx)
        print(f"saved {lidar_idx} in {seq_ID}")

if __name__ == "__main__":
    device = torch.device("cuda:2")
    dataset_root = "/DATA5"
    raw_name = "Boreas_minuse"
    target_name = "Boreas_minuse_163840_to_4096"
    dataset_path = os.path.join(dataset_root, raw_name)
    target_path = os.path.join(dataset_root, target_name)
    os.makedirs(target_path, exist_ok=True)
    dataset = BoreasDataset_U(root=dataset_path, verbose=True)
    minuse_lidar_path = os.path.join(dataset_path, "my_tool", "minuse_lidar_idxs.pickle")
    minuse_lidar = pickle.load(open(minuse_lidar_path, "rb"))
    sequence_list = []
    voxel_size = 0.04
    for seq_ID, lidar_idxs in minuse_lidar.items():
        curr_list = [seq_ID, lidar_idxs]
        sequence_list.append(curr_list)
    
    # for seq_list in sequence_list:
    #     process_sequence(seq_list)
    with Pool() as p:
        p.map(process_sequence, sequence_list)

        
