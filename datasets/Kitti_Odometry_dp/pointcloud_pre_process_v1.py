import open3d as o3d
import numpy as np
import os
import pickle
from pyboreas.utils.utils import load_lidar
from pointcloud_process import FPS_downsample, voxel_downsample
from pointnet2_ops import pointnet2_utils
import torch
import copy
import argparse
from my_pykitti_odometry import my_odometry

# import torch.multiprocessing as mp

all_seq_ID_list = ['00', '01', '02', '03', '04', 
                     '05', '06', '07', '08', '09', 
                     '10', '11', '12', '13', '14',
                     '15', '16', '17', '18', '19',
                     '20', '21']

def parse_args():
    parser = argparse.ArgumentParser(description='generate pointcloud')
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to data storing file')
    parser.add_argument(
        '--part_num',
        type=int,
        help='Number of parts to split the data')
    parser.add_argument(
        '--local_rank',
        type=int,
        default=0)

    parser.set_defaults(debug=False)

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

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

def process_sequence(seq_ID):
    print(f"enter {seq_ID}")
    curr_seq = my_odometry(sequence=seq_ID, 
                           base_path=dataset_path, 
                           pose_path=os.path.join(args.data_path, "semanticKITTI", raw_name))
    target_seq_path = os.path.join(target_path, seq_ID)
    os.makedirs(target_seq_path, exist_ok=True)
    target_seq_lidar_path = os.path.join(target_seq_path, "velodyne")
    os.makedirs(target_seq_lidar_path, exist_ok=True)

    curr_calib = curr_seq.calib
    curr_T_ego_cam2 = curr_calib['T_ego_cam2']
    curr_T_ego_LiDAR = curr_calib['T_ego_LiDAR']
    T_cam2_LiDAR = np.linalg.inv(curr_T_ego_cam2) @ curr_T_ego_LiDAR
    curr_cam2_K = curr_calib['cam2_K']
    img_H = 370
    img_W = 1226

    for i in range(len(curr_seq.timestamps)):
        filename = str(i).zfill(6)
        save_path_1 = os.path.join(target_seq_lidar_path, filename + "_1.npy")
        save_path_2 = os.path.join(target_seq_lidar_path, filename + "_2.npy")
        save_path_idx = os.path.join(target_seq_lidar_path, filename + "_3.npy")
        if os.path.exists(save_path_idx):
            print(f"pass the {filename} in {seq_ID}")
            continue

        pc = curr_seq.get_velo(i)
        pc = pc[:, :3].astype(np.float32) # shape: (N, 3)
        pc_to_mult = np.concatenate([pc, np.ones((pc.shape[0], 1))], axis=1) # (N, 4)
        pc_in_camera = pc_to_mult @ T_cam2_LiDAR.T # (N, 4)
        pc_in_camera = pc_in_camera[..., :3]
        mask_1 = pc_in_camera[:, 2] >= 0.0
        pc_in_image = pc_in_camera @ curr_cam2_K.T # (N, 3)
        pc_in_image = pc_in_image[:, :2] / (pc_in_image[:, 2:] * 1.0)
        mask_2 = (pc_in_image[..., 0] >= 0.0) \
                  & (pc_in_image[..., 0] < float(img_W)) \
                  & (pc_in_image[..., 1] >= 0.0) \
                  & (pc_in_image[..., 1] < float(img_H))
        pc_mask = mask_1 & mask_2
        pc = pc[pc_mask, :]

        curr_voxel_size_high_border = copy.deepcopy(voxel_size_high_border)
        curr_voxel_size_low_border = copy.deepcopy(voxel_size_low_border)
        CTR = 0
        # print('get in the loop')
        if pc.shape[0] < target_points_num_1:
            random_indices = np.random.permutation(pc.shape[0])
            pc_append_num = target_points_num_1 - pc.shape[0]
            pc_append = pc[random_indices[:pc_append_num], :]
            pc1 = np.concatenate([pc, pc_append], axis=0)
        elif pc.shape[0] <= target_points_num_1 * 1.001:
            random_indices = np.random.permutation(pc.shape[0])
            pc1 = pc[random_indices[:target_points_num_1], :]
        else:
            while True:
                curr_voxel_size = (curr_voxel_size_high_border + curr_voxel_size_low_border) / 2
                pc1 = voxel_downsample(pc, curr_voxel_size)
                if pc1.shape[0] > target_points_num_1 * 1.001:
                    curr_voxel_size_low_border = curr_voxel_size
                elif pc1.shape[0] < target_points_num_1:
                    curr_voxel_size_high_border = curr_voxel_size
                else:
                    break
                CTR += 1
            indices = np.random.permutation(pc1.shape[0])
            print(f"CTR is {CTR}")
            pc1 = pc1[indices[:target_points_num_1], :]
        # print('out the loop')
        pc1_tensor = torch.tensor(pc1, device=device)

        # pc2 = FPS_downsample(pc1, 4096)
        # pc2_tensor = torch.tensor(pc2)
        
        fps_idx = pointnet2_utils.furthest_point_sample(pc1_tensor.unsqueeze(0), target_points_num_2).long()
        pc2_tensor = index_points(pc1_tensor.unsqueeze(0), fps_idx).squeeze(0)
        pc2 = pc2_tensor.to('cpu').numpy()
        


        pc1_2_pc2_dis = torch.cdist(pc1_tensor, pc2_tensor) # Produces (target_points_num_1, target_points_num_2) tensor
        _, node12node2_idx_tensor = torch.topk(pc1_2_pc2_dis, k=1, dim=1, largest=False, sorted=False) # Produces (target_points_num_1) tensor
        node12node2_idx = node12node2_idx_tensor.squeeze(-1).to('cpu').numpy()
        np.save(save_path_2, pc2)
        np.save(save_path_1, pc1)
        np.save(save_path_idx, node12node2_idx)
        print(f"saved {filename} in {seq_ID}")

    


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(f"cuda:{args.local_rank}")
    torch.cuda.set_device(device)
    dataset_root = args.data_path + "/KITTI"
    raw_name = "dataset"
    target_points_num_1 = 16384
    target_points_num_2 = 4096
    target_name = f"{target_points_num_1}_to_{target_points_num_2}_cliped_fov"
    dataset_path = os.path.join(dataset_root, raw_name)
    target_path = os.path.join(dataset_root, target_name)
    os.makedirs(target_path, exist_ok=True)

    voxel_size = 0.020 # to tune
    voxel_size_high_border = 0.2 # to tune
    voxel_size_low_border = 0.001 # to tune

    seq_start = args.part_num * 11
    seq_end = min((args.part_num + 1) * 11, len(all_seq_ID_list))
    seq_ID_list_inuse = all_seq_ID_list[seq_start : seq_end]
    for seq_ID in seq_ID_list_inuse:
        process_sequence(seq_ID)