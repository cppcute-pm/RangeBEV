import os.path as osp
import os
from pyboreas.data.sensors import Lidar, Camera
from pathlib import Path
import subprocess
import numpy as np
from pyboreas.utils.utils import get_inverse_tf
from pyboreas.data.calib import Calib
import matplotlib.pyplot as plt
import copy

root = '/media/data/pengjianyi/Boreas'
traversal = 'boreas-2020-11-26-13-58'
camera_poses = os.path.join(root, traversal, 'applanix', 'camera_poses.csv')
lidar_poses = os.path.join(root, traversal, 'applanix', 'lidar_poses.csv')
camera_root = os.path.join(root, traversal, 'camera')
calib_root = os.path.join(root, traversal, 'calib')
calib = Calib(calib_root)
if not os.path.exists(camera_root):
    os.makedirs(camera_root)
lidar_root = os.path.join(root, traversal, 'lidar')
if not os.path.exists(lidar_root):
    os.makedirs(lidar_root)
chosen_lidar = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
Lidar_frames = []
with open(lidar_poses, 'r') as f:
    f.readline()  # header
    for line in f:
        data = line.split(",")
        ts = data[0]
        frame = Lidar(osp.join(lidar_root, ts + '.bin'))
        frame.init_pose(data)
        frame.labelFolder = None
        Lidar_frames.append(frame)

Camera_frames = []
camera_timestamp_micro_reverse = {}
camera_timestamp_micro_list = []
camera_ind = 0
with open(camera_poses, 'r') as f:
    f.readline()  # header
    for line in f:
        data = line.split(",")
        ts = data[0]
        frame = Camera(osp.join(camera_root, ts + '.png'))
        frame.init_pose(data)
        frame.labelFolder = None
        curr_camera_timestamp_micro = frame.timestamp_micro
        camera_timestamp_micro_reverse[curr_camera_timestamp_micro] = camera_ind
        camera_timestamp_micro_list.append(curr_camera_timestamp_micro)
        Camera_frames.append(frame)
        camera_ind += 1

chosen_lidar_frames = []
chosen_lidar_timestamp_micro_list = []
for indice in chosen_lidar:
    curr_lidar = Lidar_frames[indice]
    curr_lidar_path = curr_lidar.path
    if not os.path.exists(curr_lidar_path):
        print(f'lidar file {curr_lidar_path} does not exist, now downloading...')
        download_cmd = f"aws s3 cp s3://boreas/{os.path.join(traversal, curr_lidar.sensType, curr_lidar.frame + '.bin')} {curr_lidar_path} --no-sign-request"
        print(download_cmd)
        subprocess.run(download_cmd, shell=True)
    curr_lidar.load_data()
    chosen_lidar_frames.append(curr_lidar)
    chosen_lidar_timestamp_micro_list.append(curr_lidar.timestamp_micro)

chosen_lidar_timestamp_micro_array = np.array(chosen_lidar_timestamp_micro_list)
camera_timestamp_micro_array = np.array(camera_timestamp_micro_list)
camera_timestamp_micro_array = np.sort(camera_timestamp_micro_array)
indices = np.searchsorted(camera_timestamp_micro_array, chosen_lidar_timestamp_micro_array)
cam_num_per_lidar = 20
cam_interval = 10
indices_base = np.arange(-cam_num_per_lidar * cam_interval, cam_num_per_lidar * cam_interval + 1, cam_interval, dtype=int)
indices = np.expand_dims(indices, axis=-1) + indices_base
indices = np.clip(indices, 0, len(Camera_frames) - 1)
chosen_camera_timestamp_micro_array = camera_timestamp_micro_array[indices]
for i, lidar_frame in enumerate(chosen_lidar_frames):
    chosen_camera_timestamp_micro_lists = list(chosen_camera_timestamp_micro_array[i])
    # Remove motion distortion from pointcloud:
    print('body rate in lidar frame:')
    print(lidar_frame.body_rate)
    lidar_frame.remove_motion(lidar_frame.body_rate)
    for j, micro_ts in enumerate(chosen_camera_timestamp_micro_lists):
        camera_ind = camera_timestamp_micro_reverse[micro_ts]
        camera_frame = Camera_frames[camera_ind]
        curr_camera_path = camera_frame.path
        if not os.path.exists(curr_camera_path):
            print(f'camera file {curr_camera_path} does not exist, now downloading...')
            download_cmd = f"aws s3 cp s3://boreas/{os.path.join(traversal, camera_frame.sensType, camera_frame.frame + '.png')} {curr_camera_path} --no-sign-request"
            print(download_cmd)
            subprocess.run(download_cmd, shell=True)
        camera_frame.load_data()
        # Get the transform from lidar to camera:
        T_enu_camera = camera_frame.pose
        T_enu_lidar = lidar_frame.pose
        T_camera_lidar = np.matmul(get_inverse_tf(T_enu_camera), T_enu_lidar)
        lidar_curr_frame = copy.deepcopy(lidar_frame)
        print('T_camera_lidar:')
        print(T_camera_lidar)
        lidar_curr_frame.transform(T_camera_lidar)

        # Remove points outside our region of interest
        lidar_curr_frame.passthrough([-75, 75, -20, 10, 0, 40])  # xmin, xmax, ymin, ymax, zmin, zmax

        # Project lidar points onto the camera image, using the projection matrix, P0.
        uv, colors, _ = lidar_curr_frame.project_onto_image(calib.P0)

        # Draw the projection
        fig = plt.figure(figsize=(24.48, 20.48), dpi=100)
        ax = fig.add_subplot()
        ax.imshow(camera_frame.img)
        ax.set_xlim(0, 2448)
        ax.set_ylim(2048, 0)
        ax.scatter(uv[:, 0], uv[:, 1], c=colors, marker=',', s=3, edgecolors='none', alpha=0.7, cmap='jet')
        ax.set_axis_off()
        plt.savefig(f'/home/test5/code_project/visualization/haha_boreas_{i}_{j}.jpg', bbox_inches='tight', pad_inches=0, dpi=200)
        print('look look')






