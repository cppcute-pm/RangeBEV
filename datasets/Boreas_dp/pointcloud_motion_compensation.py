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
from pyboreas.utils.utils import get_inverse_tf
import matplotlib.pyplot as plt
from pyboreas.data.calib import Calib
from pyboreas.utils.utils import get_transform
from math import sin, cos, atan2, sqrt
import time

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

def get_gt_data_for_traversal(root):
    """Retrieves ground truth applanix data for a given sensor frame
    Args:
        root (str): path to the sequence root
    Returns:
        gt (list): A list of ground truth values from the applanix sensor_poses.scv
    """
    posepath = os.path.join(root, "gps_post_process.csv")
    pose_list = []
    with open(posepath, "r") as f:
        f.readline()  # header
        for line in f:
            pose_list.append([float(x) for x in line.split(",")])
    pose_ndarray = np.array(pose_list)
    return pose_ndarray

def is_sorted_along_axis(arr, axis):
    # 对数组沿给定轴方向上的差值进行计算
    diff = np.diff(arr, axis=axis)
    # 如果所有差值都是正数或零，则数组在该维度上是升序的
    return np.all(diff >= 0)

def so3_to_quaternion(so3):
    """Converts an SO3 rotation matrix to a quaternion

    Args:
        so3: 3x3 rotation matrix

    Returns:
        numpy.ndarray: quaternion [w, x, y, z]

    Raises:
        ValueError: if so3 is not 3x3
    """
    if so3.shape != (3, 3):
        raise ValueError("SO3 matrix must be 3x3")

    R_xx = so3[0, 0]
    R_xy = so3[0, 1]
    R_xz = so3[0, 2]
    R_yx = so3[1, 0]
    R_yy = so3[1, 1]
    R_yz = so3[1, 2]
    R_zx = so3[2, 0]
    R_zy = so3[2, 1]
    R_zz = so3[2, 2]

    try:
        w = sqrt(so3.trace() + 1) / 2
    except(ValueError):
        # w is non-real
        w = 0

    # Due to numerical precision the value passed to `sqrt` may be a negative of the order 1e-15.
    # To avoid this error we clip these values to a minimum value of 0.
    x = sqrt(max(1 + R_xx - R_yy - R_zz, 0)) / 2
    y = sqrt(max(1 + R_yy - R_xx - R_zz, 0)) / 2
    z = sqrt(max(1 + R_zz - R_yy - R_xx, 0)) / 2

    max_index = max(range(4), key=[w, x, y, z].__getitem__)

    if max_index == 0:
        x = (R_zy - R_yz) / (4 * w)
        y = (R_xz - R_zx) / (4 * w)
        z = (R_yx - R_xy) / (4 * w)
    elif max_index == 1:
        w = (R_zy - R_yz) / (4 * x)
        y = (R_xy + R_yx) / (4 * x)
        z = (R_zx + R_xz) / (4 * x)
    elif max_index == 2:
        w = (R_xz - R_zx) / (4 * y)
        x = (R_xy + R_yx) / (4 * y)
        z = (R_yz + R_zy) / (4 * y)
    elif max_index == 3:
        w = (R_yx - R_xy) / (4 * z)
        x = (R_zx + R_xz) / (4 * z)
        y = (R_yz + R_zy) / (4 * z)

    return np.array([w, x, y, z])

def my_interpolate_poses(pose_timestamps, requested_timestamps, abs_quaternions, abs_positions):

    upper_indices = np.minimum(np.searchsorted(pose_timestamps, requested_timestamps, side='left'), len(pose_timestamps) - 1)
    lower_indices = np.maximum(upper_indices - 1, 0)

    assert np.all(pose_timestamps[upper_indices] - pose_timestamps[lower_indices] > 0.0)

    fractions = (requested_timestamps - pose_timestamps[lower_indices]) / \
                (pose_timestamps[upper_indices] - pose_timestamps[lower_indices])

    quaternions_lower = abs_quaternions[:, lower_indices]
    quaternions_upper = abs_quaternions[:, upper_indices]

    d_array = (quaternions_lower * quaternions_upper).sum(0)

    linear_interp_indices = np.nonzero(d_array >= 1)
    sin_interp_indices = np.nonzero(d_array < 1)

    scale0_array = np.zeros(d_array.shape)
    scale1_array = np.zeros(d_array.shape)

    scale0_array[linear_interp_indices] = 1 - fractions[linear_interp_indices]
    scale1_array[linear_interp_indices] = fractions[linear_interp_indices]

    theta_array = np.arccos(np.abs(d_array[sin_interp_indices]))

    scale0_array[sin_interp_indices] = \
        np.sin((1 - fractions[sin_interp_indices]) * theta_array) / np.sin(theta_array)
    scale1_array[sin_interp_indices] = \
        np.sin(fractions[sin_interp_indices] * theta_array) / np.sin(theta_array)

    negative_d_indices = np.nonzero(d_array < 0)
    scale1_array[negative_d_indices] = -scale1_array[negative_d_indices]

    quaternions_interp = np.tile(scale0_array, (4, 1)) * quaternions_lower \
                         + np.tile(scale1_array, (4, 1)) * quaternions_upper

    positions_lower = abs_positions[:, lower_indices]
    positions_upper = abs_positions[:, upper_indices]

    positions_interp = np.multiply(np.tile((1 - fractions), (3, 1)), positions_lower) \
                       + np.multiply(np.tile(fractions, (3, 1)), positions_upper)

    poses_mat = np.zeros((4, 4 * len(requested_timestamps)))

    poses_mat[0, 0::4] = 1 - 2 * np.square(quaternions_interp[2, :]) - \
                         2 * np.square(quaternions_interp[3, :])
    poses_mat[0, 1::4] = 2 * np.multiply(quaternions_interp[1, :], quaternions_interp[2, :]) - \
                         2 * np.multiply(quaternions_interp[3, :], quaternions_interp[0, :])
    poses_mat[0, 2::4] = 2 * np.multiply(quaternions_interp[1, :], quaternions_interp[3, :]) + \
                         2 * np.multiply(quaternions_interp[2, :], quaternions_interp[0, :])

    poses_mat[1, 0::4] = 2 * np.multiply(quaternions_interp[1, :], quaternions_interp[2, :]) \
                         + 2 * np.multiply(quaternions_interp[3, :], quaternions_interp[0, :])
    poses_mat[1, 1::4] = 1 - 2 * np.square(quaternions_interp[1, :]) \
                         - 2 * np.square(quaternions_interp[3, :])
    poses_mat[1, 2::4] = 2 * np.multiply(quaternions_interp[2, :], quaternions_interp[3, :]) - \
                         2 * np.multiply(quaternions_interp[1, :], quaternions_interp[0, :])

    poses_mat[2, 0::4] = 2 * np.multiply(quaternions_interp[1, :], quaternions_interp[3, :]) - \
                         2 * np.multiply(quaternions_interp[2, :], quaternions_interp[0, :])
    poses_mat[2, 1::4] = 2 * np.multiply(quaternions_interp[2, :], quaternions_interp[3, :]) + \
                         2 * np.multiply(quaternions_interp[1, :], quaternions_interp[0, :])
    poses_mat[2, 2::4] = 1 - 2 * np.square(quaternions_interp[1, :]) - \
                         2 * np.square(quaternions_interp[2, :])

    poses_mat[0:3, 3::4] = positions_interp
    poses_mat[3, 3::4] = 1

    poses_out = poses_mat.reshape((4, len(requested_timestamps), 4))
    poses_out = np.transpose(poses_out, (1, 0, 2))

    return poses_out


def process_sequence(seq_list):
    seq_ID = seq_list[0]
    lidar_idxs = seq_list[1]
    print(f"enter {seq_ID}")
    curr_seq = dataset.get_seq_from_ID(seq_ID)
    target_seq_path = os.path.join(target_path, str(seq_ID))
    os.makedirs(target_seq_path, exist_ok=True)
    target_seq_lidar_path = os.path.join(target_seq_path, "lidar")
    os.makedirs(target_seq_lidar_path, exist_ok=True)
    calib_root = os.path.join(dataset_path, str(seq_ID), 'calib')
    calib = Calib(calib_root)
    gt_file_ndarray = get_gt_data_for_traversal(curr_seq.applanix_root) # (gt_num, 18)
    assert is_sorted_along_axis(gt_file_ndarray[:, 0], 0)
    gt_trans_matrix = np.zeros((gt_file_ndarray.shape[0], 4, 4))
    gt_positions = np.zeros((3, gt_file_ndarray.shape[0]))
    gt_quaternions = np.zeros((4, gt_file_ndarray.shape[0]))
    for i in range(gt_file_ndarray.shape[0]):
        gt_trans_matrix[i] = get_transform(gt_file_ndarray[i])
        gt_quaternions[:, i] = so3_to_quaternion(gt_trans_matrix[i][0:3, 0:3])
        gt_positions[:, i] = np.ravel(gt_trans_matrix[i][0:3, 3])
    gt_file_timestamp = gt_file_ndarray[:, 0]
    
    for lidar_idx in lidar_idxs:
        t1 = time.perf_counter()
        curr_lidar_frame = curr_seq.lidar_frames[int(lidar_idx)]
        filename = curr_lidar_frame.path.split("/")[-1].split(".")[0]
        save_path = os.path.join(target_seq_lidar_path, filename + ".npy")

        if os.path.exists(save_path):
            print(f"pass the {lidar_idx} in {seq_ID}")
            continue
        
        curr_lidar_frame.load_data()

        # curr_lidar_frame.remove_motion(curr_lidar_frame.body_rate)
        curr_lidar_frame_points_time = curr_lidar_frame.points[:, -1]
        curr_lidar_frame_points_coords = curr_lidar_frame.points[:, :3] # (N, 3)

        vis_pc_1 = curr_lidar_frame_points_coords

        curr_lidar_frame_points_poses = my_interpolate_poses(gt_file_timestamp, curr_lidar_frame_points_time, gt_quaternions, gt_positions) # (N, 4, 4)
        T_applanix_lidar = calib.T_applanix_lidar.astype(np.float32) # (4, 4)
        curr_lidar_frame_points_coords_to_mul = np.concatenate([curr_lidar_frame_points_coords, np.ones_like(curr_lidar_frame_points_coords[:, :1])], axis=1) # (N, 4)
        curr_lidar_frame_points_coords_in_applanix = np.matmul(curr_lidar_frame_points_coords_to_mul, T_applanix_lidar.T) # (N, 4)
        curr_lidar_frame_points_coords_in_enu = np.matmul(np.expand_dims(curr_lidar_frame_points_coords_in_applanix, axis=1), curr_lidar_frame_points_poses.transpose(0, 2, 1)) # (N, 1, 4)
        curr_lidar_frame_points_coords_in_enu = curr_lidar_frame_points_coords_in_enu.squeeze(1) # (N, 4)

        vis_pc_2 = curr_lidar_frame_points_coords_in_enu[:, :3]


        T_enu_lidar = curr_lidar_frame.pose.astype(np.float32) # (4, 4)
        T_lidar_enu = np.linalg.inv(T_enu_lidar)
        curr_lidar_frame_points_coords_after_mc = np.matmul(curr_lidar_frame_points_coords_in_enu, T_lidar_enu.transpose(1, 0)) # (N, 4)
        curr_lidar_frame_points_coords_after_mc = curr_lidar_frame_points_coords_after_mc[:, :3] # (N, 3)




        image_id = lidar2image[seq_ID][str(lidar_idx)][0]
        curr_image_frame = curr_seq.camera_frames[image_id]
        curr_image_frame.load_data()
        T_enu_camera = curr_image_frame.pose
        T_enu_lidar = curr_lidar_frame.pose
        T_camera_lidar = np.matmul(get_inverse_tf(T_enu_camera), T_enu_lidar)



        
        lidar_curr_frame = copy.deepcopy(curr_lidar_frame)
        lidar_curr_frame.remove_motion(lidar_curr_frame.body_rate)
        lidar_curr_frame.transform(T_camera_lidar)

        # Remove points outside our region of interest
        lidar_curr_frame.passthrough([-75, 75, -20, 10, 0, 40])  # xmin, xmax, ymin, ymax, zmin, zmax

        # Project lidar points onto the camera image, using the projection matrix, P0.
        uv, colors, _ = lidar_curr_frame.project_onto_image(calib.P0)

        # Draw the projection
        fig = plt.figure(figsize=(24.48, 20.48), dpi=100)
        ax = fig.add_subplot()
        ax.imshow(curr_image_frame.img)
        ax.set_xlim(0, 2448)
        ax.set_ylim(2048, 0)
        ax.scatter(uv[:, 0], uv[:, 1], c=colors, marker=',', s=3, edgecolors='none', alpha=0.7, cmap='jet')
        ax.set_axis_off()
        plt.savefig(f'/DATA5/pengjianyi/vis_1023/pc_image_projection_before_mc_{seq_ID}_{int(lidar_idx)}.jpg', bbox_inches='tight', pad_inches=0, dpi=200)
        print('look look')
        plt.close()


        lidar_curr_frame_mc = copy.deepcopy(curr_lidar_frame)
        lidar_curr_frame_mc.points[:, :3] = curr_lidar_frame_points_coords_after_mc
        lidar_curr_frame_mc.transform(T_camera_lidar)

        # Remove points outside our region of interest
        lidar_curr_frame_mc.passthrough([-75, 75, -20, 10, 0, 40])  # xmin, xmax, ymin, ymax, zmin, zmax

        # Project lidar points onto the camera image, using the projection matrix, P0.
        uv_mc, colors_mv, _ = lidar_curr_frame_mc.project_onto_image(calib.P0)

        # Draw the projection
        fig = plt.figure(figsize=(24.48, 20.48), dpi=100)
        ax = fig.add_subplot()
        ax.imshow(curr_image_frame.img)
        ax.set_xlim(0, 2448)
        ax.set_ylim(2048, 0)
        ax.scatter(uv_mc[:, 0], uv_mc[:, 1], c=colors_mv, marker=',', s=3, edgecolors='none', alpha=0.7, cmap='jet')
        ax.set_axis_off()
        plt.savefig(f'/DATA5/pengjianyi/vis_1023/pc_image_projection_after_mc_{seq_ID}_{int(lidar_idx)}.jpg', bbox_inches='tight', pad_inches=0, dpi=200)
        print('look look')
        plt.close()

        t2 = time.perf_counter()
        print(f"cost {t2 - t1} seconds")

        # np.save(save_path, curr_lidar_frame.points[:, :3].astype(np.float32))
        # print(f"saved {lidar_idx} in {seq_ID}")

if __name__ == "__main__":
    args = parse_args()
    device = torch.device(f"cuda:{args.local_rank}")
    torch.cuda.set_device(device)
    dataset_root = args.data_path
    raw_name = "Boreas_minuse"
    target_name = "Boreas_minuse_lidar_mc"
    dataset_path = os.path.join(dataset_root, raw_name)
    target_path = os.path.join(dataset_root, target_name)
    os.makedirs(target_path, exist_ok=True)
    dataset = BoreasDataset_U(root=dataset_path, verbose=True)
    minuse_lidar_path = os.path.join(dataset_path, "my_tool", "minuse_lidar_idxs.pickle")
    minuse_lidar = pickle.load(open(minuse_lidar_path, "rb"))
    lidar2image_path = os.path.join(dataset_path, 'my_tool', 'lidar2image.pickle')
    lidar2image = pickle.load(open(lidar2image_path, "rb"))
    sequence_list = []
    for seq_ID, lidar_idxs in minuse_lidar.items():
        curr_list = [seq_ID, lidar_idxs]
        sequence_list.append(curr_list)

    sequence_list = sequence_list[args.part_num * 4 : (args.part_num + 1) * 4]
    for seq_list in sequence_list:
        process_sequence(seq_list)
    # with Pool() as p:
    #     p.map(process_sequence, sequence_list)

        
