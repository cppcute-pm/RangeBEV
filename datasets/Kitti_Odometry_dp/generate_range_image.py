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
from range_projection import range_projection, range_projection_v2

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
                           base_path=os.path.join(dataset_root, 'dataset'), 
                           pose_path=os.path.join(args.data_path, "semanticKITTI", raw_name))
    target_seq_path = os.path.join(target_path, seq_ID)
    os.makedirs(target_seq_path, exist_ok=True)
    target_seq_lidar_path = os.path.join(target_seq_path, "velodyne")
    os.makedirs(target_seq_lidar_path, exist_ok=True)
    source_seq_lidar_path = os.path.join(dataset_path, seq_ID, "velodyne")

    for i in range(len(curr_seq.timestamps)):
        filename = str(i).zfill(6)
        save_path = os.path.join(target_seq_lidar_path, filename + "_1.npy")
        save_path_idx = os.path.join(target_seq_lidar_path, filename + "_2.npy")
        if os.path.exists(save_path_idx):
            print(f"pass the {filename} in {seq_ID}")
            continue
        source_pc_path = os.path.join(source_seq_lidar_path, filename + "_1.npy")

        source_pc = np.load(source_pc_path) # shape: (N, 3)
        source_vertex = np.concatenate([source_pc, np.ones_like(source_pc[:, 0:1])], axis=1) # (N, 4)

        # proj_range, _, _, proj_idx = range_projection(source_vertex)
        proj_range, proj_idx = range_projection_v2(source_vertex)
        W_clip_start = int((900 - 224) / 2)
        W_clip_end = int(W_clip_start + 224)
        proj_range_cliped = proj_range[:, W_clip_start:W_clip_end] # (64, 224)
        # proj_vertex_cliped = proj_vertex[:, W_clip_start:W_clip_end] # (64, 224, 4)
        # proj_intensity_cliped = proj_intensity[:, W_clip_start:W_clip_end] # (64, 224)
        proj_idx_cliped = proj_idx[:, W_clip_start:W_clip_end] # (64, 224)

        assert proj_range_cliped.shape == (64, 224)

        np.save(save_path, proj_range_cliped)
        np.save(save_path_idx, proj_idx_cliped)
        print(f"saved {filename} in {seq_ID}")

    


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(f"cuda:{args.local_rank}")
    torch.cuda.set_device(device)
    dataset_root = args.data_path + "/KITTI"
    raw_name = "16384_to_4096_cliped_fov"
    target_name = f"16384_to_4096_cliped_fov_range_image"
    dataset_path = os.path.join(dataset_root, raw_name)
    target_path = os.path.join(dataset_root, target_name)
    os.makedirs(target_path, exist_ok=True)

    seq_start = args.part_num * 22
    seq_end = min((args.part_num + 1) * 22, len(all_seq_ID_list))
    seq_ID_list_inuse = all_seq_ID_list[seq_start : seq_end]
    for seq_ID in seq_ID_list_inuse:
        process_sequence(seq_ID)