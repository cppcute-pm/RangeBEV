import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from mini_boreas import BoreasDataset_U
import os
import pickle
from pyboreas.utils.utils import load_lidar
import cv2
import torch_scatter
from pyboreas.utils.utils import get_inverse_tf
from ip_basic import fill_in_multiscale
import argparse

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

args = parse_args()
IMAGE_MAX = 1e6
data_path = args.data_path
root_name = 'Boreas_minuse'
root_path = os.path.join(data_path, root_name)
target_name = 'Boreas_minuse_lidar_depth'
target_path = os.path.join(data_path, target_name)
os.makedirs(target_path, exist_ok=True)
dataset = BoreasDataset_U(root_path)
minuse_lidar_path = os.path.join(root_path, 'my_tool', "minuse_lidar_idxs.pickle")
lidar2image_path = os.path.join(root_path, "my_tool", "lidar2image.pickle")
minuse_lidar = pickle.load(open(minuse_lidar_path, 'rb'))
lidar2image = pickle.load(open(lidar2image_path, 'rb'))
max_lidar_depth = 150.0
device = torch.device(f'cuda:{args.local_rank}')
torch.cuda.set_device(device)
threshold = 200.0

seq_ID_list = sorted(list(minuse_lidar.keys()))
seq_ID_list = seq_ID_list[args.part_num * 7 : (args.part_num + 1) * 7]

for seq_ID in seq_ID_list:
    lidar_ids = minuse_lidar[seq_ID]
    curr_seq = dataset.get_seq_from_ID(seq_ID)
    target_seq_path = os.path.join(target_path, str(seq_ID))
    os.makedirs(target_seq_path, exist_ok=True)
    target_seq_lidar_path = os.path.join(target_seq_path, 'lidar')
    os.makedirs(target_seq_lidar_path, exist_ok=True)
    for lidar_id in lidar_ids:
        curr_lidar_frame = curr_seq.lidar_frames[lidar_id]
        seq_ID, lidar_dir, pc_file_name = curr_lidar_frame.path.split('/')[-3:]
        prefix = pc_file_name.split('.')[0]
        target_depth_path = os.path.join(target_seq_lidar_path, prefix + '.png')
        if os.path.exists(target_depth_path):
            print(f'{target_depth_path} exists, passed')
            continue
        curr_lidar_frame.load_data()
        curr_lidar_frame.remove_motion(curr_lidar_frame.body_rate)
        pc = curr_lidar_frame.points[:, :3].astype(np.float32)
        pc_pose = curr_lidar_frame.pose.astype(np.float32)
        image_id = lidar2image[seq_ID][str(lidar_id)][0]
        curr_image_frame = curr_seq.camera_frames[image_id]
        image_pose = curr_image_frame.pose.astype(np.float32)
        image = cv2.imread(curr_image_frame.path)

        # T_enu_camera = curr_image_frame.pose
        # T_enu_lidar = curr_lidar_frame.pose
        # T_camera_lidar = np.matmul(get_inverse_tf(T_enu_camera), T_enu_lidar)
        # curr_lidar_frame.transform(T_camera_lidar)
        # # Remove points outside our region of interest
        # curr_lidar_frame.passthrough([-75, 75, -20, 10, 0, 40])  # xmin, xmax, ymin, ymax, zmin, zmax

        # # Project lidar points onto the camera image, using the projection matrix, P0.
        # uv, colors, _ = curr_lidar_frame.project_onto_image(curr_seq.calib.P0)

        # # Draw the projection
        # fig = plt.figure(figsize=(24.48, 20.48), dpi=100)
        # ax = fig.add_subplot()
        # ax.imshow(image)
        # ax.set_xlim(0, 2448)
        # ax.set_ylim(2048, 0)
        # ax.scatter(uv[:, 0], uv[:, 1], c=colors, marker=',', s=3, edgecolors='none', alpha=0.7, cmap='jet')
        # ax.set_axis_off()
        # plt.savefig(f'/home/pengjianyi/code_projects/haha_boreas_hehe.jpg', bbox_inches='tight', pad_inches=0, dpi=200)
        # print('look look')



        img_H, img_W = image.shape[:2]
        curr_P0 = curr_seq.calib.P0.astype(np.float32)
        curr_P0 = curr_P0[:3, :3]
        P_camera_lidar = np.linalg.inv(image_pose) @ pc_pose # (4, 4)
        pc_to_mult = np.concatenate([pc, np.ones((pc.shape[0], 1))], axis=1) # (N, 4)
        pc_in_camera = pc_to_mult @ P_camera_lidar.T # (N, 4)
        pc_in_camera = pc_in_camera[..., :3]
        mask_1 = pc_in_camera[:, 2] >= 0.0

        mask_0 = pc_in_camera[:, 2] < max_lidar_depth # TODO: big mistake
        mask_1 = mask_1 & mask_0

        pc_in_image = pc_in_camera @ curr_P0.T # (N, 3)
        pc_in_image = pc_in_image[:, :2] / (pc_in_image[:, 2:] * 1.0)
        mask_2 = (pc_in_image[..., 0] >= 0.0) \
                  & (pc_in_image[..., 0] < float(img_W)) \
                  & (pc_in_image[..., 1] >= 0.0) \
                  & (pc_in_image[..., 1] < float(img_H))
        pc_in_image_chosen1 = pc_in_image[mask_1 & mask_2]
        pc_in_image_chosen1_pixel = np.floor(pc_in_image_chosen1, dtype=np.float32).astype(np.int32)
        pc_in_image_chosen1_pixel_map = pc_in_image_chosen1_pixel[:, 1] * img_W + pc_in_image_chosen1_pixel[:, 0]
        pc_in_camera_chosen1_depth = np.linalg.norm(pc_in_camera[mask_1 & mask_2], axis=-1)
        # pc_in_camera_chosen1_depth = np.clip(pc_in_camera_chosen1_depth, 0, max_lidar_depth)
        # pc_in_camera_chosen1_depth = pc_in_camera_chosen1_depth / max_lidar_depth
        pc_in_camera_chosen1_depth = torch.tensor(pc_in_camera_chosen1_depth, dtype=torch.float32, device=device) # (N,)
        pc_in_image_chosen1_pixel_map = torch.tensor(pc_in_image_chosen1_pixel_map, dtype=torch.int64, device=device) # (N,)
        pc_depth_image_0, _ = torch_scatter.scatter_min(pc_in_camera_chosen1_depth, 
                                                pc_in_image_chosen1_pixel_map, 
                                                dim=-1, 
                                                dim_size=img_H * img_W) # (img_H * img_W)
        
        # ① : implement as the original article
        # pc_depth_image = pc_depth_image_0.reshape(img_H, img_W).permute(1, 0) # (img_W, img_H)
        # vis_b = pc_depth_image.permute(1, 0).unsqueeze(-1).cpu().numpy().repeat(3, axis=-1) / max_lidar_depth
        # vis_b = vis_b * 255
        # hole_mask = torch.eq(pc_depth_image, 0.0)
        # hole_indices = torch.nonzero(hole_mask, as_tuple=False) # (N, 2)
        # hole_bincount = torch.bincount(hole_indices[:, 0], minlength=img_W) # (img_W,)
        # compare1 = torch.arange(0, img_H, device=device).unsqueeze(0).repeat(img_W, 1)
        # compare2 = hole_bincount.unsqueeze(1).repeat(1, img_H)
        # mask = torch.ge(compare1, compare2)
        # compare1.masked_fill_(mask, img_H)
        # hole_indices_count = compare1[compare1 != img_H] # (N,)
        # img_W_indices = torch.arange(0, img_W, device=device).unsqueeze(1).repeat(1, img_H)
        # hole_indices_img_W = img_W_indices[compare1 != img_H] # (N,)
        # unhole_indices = torch.nonzero(~hole_mask, as_tuple=False) # (M, 2)
        # unhole_bincount = torch.bincount(unhole_indices[:, 0], minlength=img_W)
        # compare1 = torch.arange(0, img_H, device=device).unsqueeze(0).repeat(img_W, 1)
        # compare2 = unhole_bincount.unsqueeze(1).repeat(1, img_H)
        # mask = torch.ge(compare1, compare2)
        # compare1.masked_fill_(mask, img_H)
        # unhole_indices_count = compare1[compare1 != img_H] # (M,)
        # hole_matrix = torch.full(size=(img_W, img_H), fill_value=img_H, dtype=torch.int64, device=device)
        # unhole_matrix = torch.full(size=(img_W, img_H), fill_value=img_H, dtype=torch.int64, device=device)
        # hole_matrix[hole_indices[:, 0], hole_indices_count] = hole_indices[:, 1]
        # unhole_matrix[unhole_indices[:, 0], unhole_indices_count] = unhole_indices[:, 1]
        # indices_bigger = torch.searchsorted(unhole_matrix, hole_matrix, out_int32=True, right=False) # (img_W, img_H)
        # indices_bigger = torch.clamp(indices_bigger, max=unhole_bincount.unsqueeze(1).repeat(1, img_H) - 1)
        # indices_smaller = torch.clamp(indices_bigger - 1, min=0)
        # hole_indices_bigger = indices_bigger[hole_indices_img_W, hole_indices_count] # (N,)
        # hole_indices_smaller = indices_smaller[hole_indices_img_W, hole_indices_count] # (N,)
        # unhole_indices_bigger = unhole_matrix[hole_indices_img_W, hole_indices_bigger] # (N,)
        # unhole_indices_smaller = unhole_matrix[hole_indices_img_W, hole_indices_smaller] # (N,)
        # unhole_indices_bigger_delta = torch.abs(unhole_indices_bigger - hole_indices[..., 1])
        # unhole_indices_smaller_delta = torch.abs(unhole_indices_smaller - hole_indices[..., 1])
        # pc_depth_image_unhole_bigger = pc_depth_image[hole_indices_img_W, unhole_indices_bigger] # (N,)
        # pc_depth_image_unhole_smaller = pc_depth_image[hole_indices_img_W, unhole_indices_smaller] # (N,)
        # interpolate_depth = (pc_depth_image_unhole_bigger * unhole_indices_smaller_delta + pc_depth_image_unhole_smaller * unhole_indices_bigger_delta) / (unhole_indices_bigger_delta + unhole_indices_smaller_delta)
        # over_threshold_mask = torch.gt(torch.abs(pc_depth_image_unhole_bigger - pc_depth_image_unhole_smaller), threshold)
        # min_depth = torch.minimum(pc_depth_image_unhole_bigger, pc_depth_image_unhole_smaller)
        # final_depth = torch.where(over_threshold_mask, min_depth, interpolate_depth)
        # pc_depth_image[hole_indices[..., 0], hole_indices[..., 1]] = final_depth
        # pc_depth_image = pc_depth_image.permute(1, 0)
        # vis_after_complet_pc_depth_image = pc_depth_image.unsqueeze(-1).cpu().numpy().repeat(3, axis=-1) / float(torch.max(pc_depth_image).to('cpu').numpy())
        # vis_after_complet_pc_depth_image = vis_after_complet_pc_depth_image * 255
        # vis_after_complet_pc_depth_image = np.floor(vis_after_complet_pc_depth_image).astype(np.uint8)
        # cv2.imwrite(f'/home/pengjianyi/code_projects/visualization_depth/boreas_heiheihei_{str(lidar_id)}.jpg', vis_after_complet_pc_depth_image)
        # print("one image success")

        # ②：use the code from SuperFusion
        pc_depth_image_1 = pc_depth_image_0.reshape(img_H, img_W).to('cpu').numpy().astype(np.float32)
        depths_out, process_dict = fill_in_multiscale(depth_map=pc_depth_image_1, max_depth=max_lidar_depth, extrapolate=False, show_process=True)
        depths_out = ((depths_out / max_lidar_depth) * 65535).astype(np.uint16)
        cv2.imwrite(target_depth_path, depths_out)
        print(f'{target_depth_path} success')
        # vis_after_complet_pc_depth_image = np.floor(depths_out).astype(np.uint8)
        # color_depth_map = cv2.applyColorMap(vis_after_complet_pc_depth_image, cv2.COLORMAP_JET)
        # cv2.imwrite(f'/home/pengjianyi/code_projects/visualization_depth/boreas_haihaihai_{str(lidar_id)}.jpg', color_depth_map)
        # print("pause !")
