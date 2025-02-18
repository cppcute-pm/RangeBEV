from my_pykitti_odometry import my_odometry
import torch
import os
import pickle
import numpy as np
import cv2
import open3d as o3d
from open3d import geometry
from open3d import visualization
import matplotlib.pyplot as plt
from PIL import Image
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# camera_params_path_list = [
#     '/home/pengjianyi/code_projects/CoarseFromFine/camera_params/camera_params_global_map_vis_1.json',
#     # '/home/pengjianyi/code_projects/CoarseFromFine/camera_params/camera_params_global_map_vis_2.json',
#     # '/home/pengjianyi/code_projects/CoarseFromFine/camera_params/camera_params_global_map_vis_3.json',
#     # '/home/pengjianyi/code_projects/CoarseFromFine/camera_params/camera_params_global_map_vis_4.json',
#     # '/home/pengjianyi/code_projects/CoarseFromFine/camera_params/camera_params_global_map_vis_5.json',
#     # '/home/pengjianyi/code_projects/CoarseFromFine/camera_params/camera_params_global_map_vis_6.json',
#     # '/home/pengjianyi/code_projects/CoarseFromFine/camera_params/camera_params_global_map_vis_7.json',
#     ]

# camera_params_list = []
# for camera_params_path in camera_params_path_list:
#     camera_params = o3d.io.read_pinhole_camera_parameters(camera_params_path)
#     camera_params_list.append(camera_params)

# point_size = 3.0
# image_shape = (camera_params_list[0].intrinsic.height, camera_params_list[0].intrinsic.width)
# renderer = o3d.visualization.rendering.OffscreenRenderer(width=image_shape[1], 
#                                                        height=image_shape[0])
# renderer.scene.set_background([1, 1, 1, 255]) # Set background to white

# # renderer.scene.set_lighting(profile=renderer.scene.NO_SHADOWS, sun_dir=np.array([[0.1],[0.1],[0.1]]))

# material = o3d.visualization.rendering.MaterialRecord() # Create material
# material.point_size = point_size
# material.shader = "defaultUnlit" # Set shader to defaultUnlit

# vis_id_list = [4297, 2118, 4226, 4340, 4371]

seq_ID = '05'
# device = torch.device("cuda:1")
# torch.cuda.set_device(device)
# dataset_root = os.path.join("/DATA1/pengjianyi", "KITTI")
# raw_name = "dataset/sequences"
# dataset_pc_root = os.path.join(dataset_root, raw_name)
# dataset_pc_seq_path = os.path.join(dataset_pc_root, seq_ID, "velodyne")

# curr_seq = my_odometry(sequence=seq_ID, 
#                         base_path=os.path.join(dataset_root, 'dataset'), 
#                         pose_path=os.path.join("/DATA1/pengjianyi", "semanticKITTI", 'dataset'))
# LiDAR_list = []
# for id in range(len(curr_seq.timestamps)):
#     T_first_cam0_curr_cam0 = curr_seq.poses[id]
#     curr_calib = curr_seq.calib
#     T_cam0_LiDAR = curr_calib['T_ego_LiDAR']
#     T_first_cam0_curr_LiDAR = np.matmul(T_first_cam0_curr_cam0, T_cam0_LiDAR).astype(np.float32)
#     T_first_LiDAR_curr_LiDAR = np.matmul(np.linalg.inv(T_cam0_LiDAR), T_first_cam0_curr_LiDAR)

#     file_name = str(id).zfill(6)
#     curr_LiDAR = np.fromfile(os.path.join(dataset_pc_seq_path, file_name + '.bin'), dtype=np.float32).reshape((-1, 4))
#     curr_LiDAR = curr_LiDAR[:, :3]

#     curr_LiDAR_to_mul = np.concatenate((curr_LiDAR, np.ones((curr_LiDAR.shape[0], 1))), axis=1)
#     curr_LiDAR_mul = np.matmul(curr_LiDAR_to_mul, T_first_LiDAR_curr_LiDAR.T)
#     curr_LiDAR = curr_LiDAR_mul[:, :3]
#     LiDAR_list.append(curr_LiDAR)

# LiDAR_vis = np.concatenate(LiDAR_list, axis=0)
# LiDAR_vis_pcd = geometry.PointCloud()
# LiDAR_vis_pcd.points = o3d.utility.Vector3dVector(LiDAR_vis)
# LiDAR_vis_pcd_down_sampled = LiDAR_vis_pcd.voxel_down_sample(voxel_size=0.25)

# LiDAR_vis_down_sampled = np.asarray(LiDAR_vis_pcd_down_sampled.points)
# LiDAR_save_path = os.path.join('/home/pengjianyi/code_projects/vis1209', f'GM_vis_seq{seq_ID}_voxel_025.npy')
# np.save(LiDAR_save_path, LiDAR_vis_down_sampled)



# LiDAR_save_path = os.path.join('/home/pengjianyi/code_projects/vis1209', f'GM_vis_seq{seq_ID}_voxel_025.npy')
# LiDAR_vis = np.load(LiDAR_save_path)

# LiDAR_vis_tensor = torch.from_numpy(LiDAR_vis).to(device)

# knn_idx_list = []
# for id in vis_id_list:
#     T_first_cam0_curr_cam0 = curr_seq.poses[id]
#     curr_calib = curr_seq.calib
#     T_cam0_LiDAR = curr_calib['T_ego_LiDAR']
#     T_first_cam0_curr_LiDAR = np.matmul(T_first_cam0_curr_cam0, T_cam0_LiDAR).astype(np.float32)
#     T_first_LiDAR_curr_LiDAR = np.matmul(np.linalg.inv(T_cam0_LiDAR), T_first_cam0_curr_LiDAR)

#     file_name = str(id).zfill(6)
#     curr_LiDAR = np.fromfile(os.path.join(dataset_pc_seq_path, file_name + '.bin'), dtype=np.float32).reshape((-1, 4))
#     curr_LiDAR = curr_LiDAR[:, :3]

#     curr_LiDAR_to_mul = np.concatenate((curr_LiDAR, np.ones((curr_LiDAR.shape[0], 1))), axis=1)
#     curr_LiDAR_mul = np.matmul(curr_LiDAR_to_mul, T_first_LiDAR_curr_LiDAR.T)
#     curr_LiDAR = curr_LiDAR_mul[:, :3]

#     curr_LiDAR_mean_pos = np.mean(curr_LiDAR, axis=0) # [3]

#     curr_LiDAR_mean_pos = torch.from_numpy(curr_LiDAR_mean_pos).to(device)

#     curr_cdist = torch.cdist(curr_LiDAR_mean_pos.unsqueeze(0), LiDAR_vis_tensor, p=2.0)

#     _, curr_knn_idx = torch.topk(curr_cdist, dim=1, k=10000, largest=False, sorted=False)

#     knn_idx_list.append(curr_knn_idx.cpu().numpy().squeeze())

# print('points loading done')

# common_colormap = plt.get_cmap("viridis")
# def get_render_image_for_pc_with_label(pc, camera_params_list):

#     pcd_GM_vis = geometry.PointCloud()
#     pcd_GM_vis.points = o3d.utility.Vector3dVector(pc)

#     z_values = pc[:, 2]
#     z_values = (z_values - z_values.min()) / (z_values.max() - z_values.min())

#     print('start drawing color')
#     colors = common_colormap(z_values)[:, :3]

#     # semantic_colors = pc_semantic_label_color_range[pc_semantic_label, :] # [N, 3]
#     GM_vis_colors = colors
#     pcd_GM_vis.colors = o3d.utility.Vector3dVector(GM_vis_colors)

#     GM_render_imgs = []
#     renderer.scene.clear_geometry()

#     print('start add geometry')
#     renderer.scene.add_geometry('pcd', pcd_GM_vis, material)
    
#     for i, camera_params in enumerate(camera_params_list):
#         print('start setup camera')
#         renderer.setup_camera(camera_params.intrinsic, camera_params.extrinsic)
#         print('start render')
#         image_new = np.asarray(renderer.render_to_image())
#         GM_render_imgs.append(image_new)
    
#     return GM_render_imgs

# GM_render_imgs = get_render_image_for_pc_with_label(LiDAR_vis, camera_params_list)
# for i, semantic_render_img in enumerate(GM_render_imgs):
#     if semantic_render_img.dtype == np.float32:
#         curr_render_img = (semantic_render_img * 255.0).astype(np.uint8)
#     else:
#         curr_render_img = semantic_render_img
    
#     output_path_name = 'GM_render.png'
#     output_path = os.path.join('/home/pengjianyi/code_projects/vis1214', output_path_name)
#     Image.fromarray(curr_render_img).save(output_path)


# def get_render_image_for_pc_with_label(pc, camera_params_list, knn_idx):

#     pcd_GM_vis = geometry.PointCloud()
#     pcd_GM_vis.points = o3d.utility.Vector3dVector(pc)

#     z_values = pc[:, 2]
#     z_values = (z_values - z_values.min()) / (z_values.max() - z_values.min())

#     print('start drawing color')
#     colors = common_colormap(z_values)[:, :3]
#     colors[knn_idx, :] = [0.99999, 0, 0]

#     # semantic_colors = pc_semantic_label_color_range[pc_semantic_label, :] # [N, 3]
#     GM_vis_colors = colors
#     pcd_GM_vis.colors = o3d.utility.Vector3dVector(GM_vis_colors)

#     GM_render_imgs = []
#     renderer.scene.clear_geometry()

#     print('start add geometry')
#     renderer.scene.add_geometry('pcd', pcd_GM_vis, material)
    
#     for i, camera_params in enumerate(camera_params_list):
#         print('start setup camera')
#         renderer.setup_camera(camera_params.intrinsic, camera_params.extrinsic)
#         print('start render')
#         image_new = np.asarray(renderer.render_to_image())
#         GM_render_imgs.append(image_new)
    
#     return GM_render_imgs

# for j, knn_idx in enumerate(knn_idx_list):
#     GM_render_imgs = get_render_image_for_pc_with_label(LiDAR_vis, camera_params_list, knn_idx)
#     for i, semantic_render_img in enumerate(GM_render_imgs):
#         if semantic_render_img.dtype == np.float32:
#             curr_render_img = (semantic_render_img * 255.0).astype(np.uint8)
#         else:
#             curr_render_img = semantic_render_img
        
#         output_path_name = f'GM_render_{vis_id_list[j]}.png'
#         output_path = os.path.join('/home/pengjianyi/code_projects/vis1209', output_path_name)
#         Image.fromarray(curr_render_img).save(output_path)





# Load the global LiDAR map


# vis_rank_path = '/DATA5/pengjianyi/vis_rerank/fa6omv6u/vis_rerank.pkl'
# with open(vis_rank_path, 'rb') as f:
#     vis_rank = pickle.load(f)

# for key in vis_rank.keys():
#     curr_seq = key
#     for i in range(vis_rank[curr_seq].shape[0]):
#         if vis_rank[curr_seq][i] == 1:
#             print(curr_seq, i)
#             break

idx_list = [1456]
LiDAR_range_image_dir = os.path.join('/DATA1/pengjianyi/KITTI/16384_to_4096_cliped_fov_range_image', seq_ID, 'velodyne')
for idx in idx_list:
    LiDAR_range_image_path = os.path.join(LiDAR_range_image_dir, str(idx).zfill(6) + '_1.npy')
    LiDAR_range_image = np.load(LiDAR_range_image_path)
    range_img = np.uint8(LiDAR_range_image * 5.1)
    range_img = np.expand_dims(range_img, axis=2) # (64, 224, 1)
    range_img = np.repeat(range_img, 3, axis=2) # (64, 224, 3)
    print(range_img.shape)

# pc_dir = os.path.join('/DATA1/pengjianyi/KITTI/16384_to_4096_cliped_fov', seq_ID, 'velodyne')
# for idx in idx_list:
#     pc_path = os.path.join(pc_dir, str(idx).zfill(6) + '_2.npy')
#     pc_original = np.load(pc_path).astype(np.float32)
#     print(pc_original.shape)

print('pause')