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
import copy
import argparse
# import torch.multiprocessing as mp
import cv2

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
        save_path_40960 = os.path.join(target_seq_lidar_path, filename + "_2.npy")
        save_path_idx = os.path.join(target_seq_lidar_path, filename + "_3.npy")
        if os.path.exists(save_path_idx):
            print(f"pass the {lidar_idx} in {seq_ID}")
            continue
        curr_lidar_frame.load_data()
        curr_lidar_frame.remove_motion(curr_lidar_frame.body_rate)
        pc = curr_lidar_frame.points[:, :3].astype(np.float32)
        pc_pose = curr_lidar_frame.pose.astype(np.float32)
        image_id = lidar2image[seq_ID][str(lidar_idx)][0]
        curr_image_frame = curr_seq.camera_frames[image_id]



        curr_P0 = curr_seq.calib.P0.astype(np.float32)
        curr_P0 = curr_P0[:3, :3]
        image_pose = curr_image_frame.pose.astype(np.float32)
        img_H = 2048
        img_W = 2448
        P_camera_lidar = np.linalg.inv(image_pose) @ pc_pose # (4, 4)
        pc_to_mult = np.concatenate([pc, np.ones((pc.shape[0], 1))], axis=1) # (N, 4)
        pc_in_camera = pc_to_mult @ P_camera_lidar.T # (N, 4)
        pc_in_camera = pc_in_camera[..., :3]
        mask_1 = pc_in_camera[:, 2] >= 0.0
        pc_in_image = pc_in_camera @ curr_P0.T # (N, 3)
        pc_in_image = pc_in_image[:, :2] / (pc_in_image[:, 2:] * 1.0)
        mask_2 = (pc_in_image[..., 0] >= 0.0) \
                  & (pc_in_image[..., 0] < float(img_W)) \
                  & (pc_in_image[..., 1] >= 0.0) \
                  & (pc_in_image[..., 1] < float(img_H))
        pc_mask = mask_1 & mask_2
        pc = pc[pc_mask, :]

        # pc_tensor = torch.tensor(pc).to(device)
        # fps_idx = pointnet2_utils.furthest_point_sample(pc_tensor, 163840).long() 
        # pc1_tensor = index_points(pc_tensor, fps_idx)  
        # fps_idx = pointnet2_utils.furthest_point_sample(pc1_tensor, 4096).long()
        # pc2_tensor = index_points(pc1_tensor, fps_idx)
        # pc1 = pc1_tensor.to('cpu').numpy()
        # pc2 = pc2_tensor.to('cpu').numpy()
        curr_voxel_size_high_border = copy.deepcopy(voxel_size_high_border)
        curr_voxel_size_low_border = copy.deepcopy(voxel_size_low_border)
        CTR = 0
        # print('get in the loop')
        if pc.shape[0] < 40960:
            random_indices = np.random.permutation(pc.shape[0])
            pc_append_num = 40960 - pc.shape[0]
            pc_append = pc[random_indices[:pc_append_num], :]
            pc1 = np.concatenate([pc, pc_append], axis=0)
        elif pc.shape[0] <= 40960 * 1.001:
            random_indices = np.random.permutation(pc.shape[0])
            pc1 = pc[random_indices[:40960], :]
        else:
            while True:
                curr_voxel_size = (curr_voxel_size_high_border + curr_voxel_size_low_border) / 2
                pc1 = voxel_downsample(pc, curr_voxel_size)
                if pc1.shape[0] > 40960 * 1.001:
                    curr_voxel_size_low_border = curr_voxel_size
                elif pc1.shape[0] < 40960:
                    curr_voxel_size_high_border = curr_voxel_size
                else:
                    break
                CTR += 1
            indices = np.random.permutation(pc1.shape[0])
            pc1 = pc1[indices[:40960], :]
        # print('out the loop')
        pc1_tensor = torch.tensor(pc1, device=device)

        # pc2 = FPS_downsample(pc1, 4096)
        # pc2_tensor = torch.tensor(pc2)
        
        fps_idx = pointnet2_utils.furthest_point_sample(pc1_tensor.unsqueeze(0), 4096).long()
        pc2_tensor = index_points(pc1_tensor.unsqueeze(0), fps_idx).squeeze(0)
        pc2 = pc2_tensor.to('cpu').numpy()
        


        pc1_2_pc2_dis = torch.cdist(pc1_tensor, pc2_tensor) # Produces (40960, 4096) tensor
        _, node12node2_idx_tensor = torch.topk(pc1_2_pc2_dis, k=1, dim=1, largest=False, sorted=False) # Produces (2 * 81920) tensor
        node12node2_idx = node12node2_idx_tensor.squeeze(-1).to('cpu').numpy()
        np.save(save_path_4096, pc2)
        np.save(save_path_40960, pc1)
        np.save(save_path_idx, node12node2_idx)
        print(f"saved {lidar_idx} in {seq_ID}")

if __name__ == "__main__":
    args = parse_args()
    device = torch.device(f"cuda:{args.local_rank}")
    torch.cuda.set_device(device)
    dataset_root = args.data_path
    raw_name = "Boreas_minuse"
    target_name = "Boreas_minuse_40960_to_4096_cliped_fov"
    dataset_path = os.path.join(dataset_root, raw_name)
    target_path = os.path.join(dataset_root, target_name)
    os.makedirs(target_path, exist_ok=True)
    dataset = BoreasDataset_U(root=dataset_path, verbose=True)
    minuse_lidar_path = os.path.join(dataset_path, "my_tool", "minuse_lidar_idxs.pickle")
    minuse_lidar = pickle.load(open(minuse_lidar_path, "rb"))
    lidar2image_path = os.path.join(dataset_path, 'my_tool', 'lidar2image.pickle')
    lidar2image = pickle.load(open(lidar2image_path, "rb"))
    sequence_list = []
    voxel_size = 0.022
    voxel_size_high_border = 0.2
    voxel_size_low_border = 0.001
    for seq_ID, lidar_idxs in minuse_lidar.items():
        curr_list = [seq_ID, lidar_idxs]
        sequence_list.append(curr_list)

    sequence_list = sequence_list[args.part_num * 4 : (args.part_num + 1) * 4]
    for seq_list in sequence_list:
        process_sequence(seq_list)
    # with Pool() as p:
    #     p.map(process_sequence, sequence_list)

        
