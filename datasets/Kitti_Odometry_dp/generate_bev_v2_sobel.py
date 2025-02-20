import os
import pickle
import numpy as np
import cv2
import sys
import torch
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
import torchvision.transforms as transforms

import open3d as o3d
from my_pykitti_odometry import my_odometry
import torch_scatter
import time
from PIL import Image
import pypatchworkpp

def getBEV(all_points): #N*3
    
    voxel_size_inuse = 0.4
    all_points_pc = o3d.geometry.PointCloud()# pcl.PointCloud()
    all_points_pc.points = o3d.utility.Vector3dVector(all_points)#all_points_pc.from_array(all_points)
    all_points_pc = all_points_pc.voxel_down_sample(voxel_size=voxel_size_inuse) #f = all_points_pc.make_voxel_grid_filter()
    

    all_points = np.asarray(all_points_pc.points)# np.array(all_points_pc.to_list())


    x_min = -25.6
    y_min = 0.0
    x_max = 25.6
    y_max = 51.2

    x_min_ind = np.floor(x_min/0.4).astype(int)
    x_max_ind = np.floor(x_max/0.4).astype(int)
    y_min_ind = np.floor(y_min/0.4).astype(int)
    y_max_ind = np.floor(y_max/0.4).astype(int)

    x_num = x_max_ind-x_min_ind
    y_num = y_max_ind-y_min_ind

    # mat_global_image = np.zeros(( y_num,x_num),dtype=np.float32)
          
    # for i in range(all_points.shape[0]):
    #     x_ind = x_max_ind-np.floor(all_points[i,1]/voxel_size_inuse).astype(int)
    #     y_ind = y_max_ind-np.floor(all_points[i,0]/voxel_size_inuse).astype(int)
    #     if(x_ind>=x_num or y_ind>=y_num or x_ind<0 or y_ind<0):
    #         continue
    #     # if mat_global_image[ y_ind,x_ind]<10:
    #     mat_global_image[ y_ind,x_ind] += 1

    # mat_global_image[mat_global_image<=1] = 0


    x_min_ind_tensor = torch.tensor(x_min_ind, dtype=torch.int64).cuda()
    x_max_ind_tensor = torch.tensor(x_max_ind, dtype=torch.int64).cuda()
    y_min_ind_tensor = torch.tensor(y_min_ind, dtype=torch.int64).cuda()
    y_max_ind_tensor = torch.tensor(y_max_ind, dtype=torch.int64).cuda()
    x_num_tensor = torch.tensor(x_num, dtype=torch.int64).cuda()
    y_num_tensor = torch.tensor(y_num, dtype=torch.int64).cuda()
    all_points_tensor = torch.tensor(all_points, dtype=torch.float32).cuda()
    x_ind_tensor = x_max_ind_tensor - torch.floor(all_points_tensor[:, 1] / voxel_size_inuse).type(torch.int64)
    y_ind_tensor = y_max_ind_tensor - torch.floor(all_points_tensor[:, 0] / voxel_size_inuse).type(torch.int64)
    illegal_mask = (x_ind_tensor >= x_num_tensor) | (y_ind_tensor >= y_num_tensor) | (x_ind_tensor < 0) | (y_ind_tensor < 0)
    x_ind_tensor = x_ind_tensor[~illegal_mask] # (N, )
    y_ind_tensor = y_ind_tensor[~illegal_mask] # (N, )
    ind_tensor = torch.stack((y_ind_tensor, x_ind_tensor), dim=1) # (N, 2)
    ind_tensor_unique, ind_tensor_counts = torch.unique(ind_tensor, 
                                                        sorted=False, 
                                                        return_inverse=False, 
                                                        return_counts=True, 
                                                        dim=0) # (M, 2)
    
    mat_global_image_tensor = torch.zeros((y_num, x_num), dtype=torch.float32).cuda()
    mat_global_image_tensor[ind_tensor_unique[:, 0], ind_tensor_unique[:, 1]] = ind_tensor_counts.float()
    mat_global_image_tensor[mat_global_image_tensor <= 1] = 0
    mat_global_image_ndarray = mat_global_image_tensor.cpu().numpy()
    # assert np.allclose(mat_global_image, mat_global_image_ndarray)

    mat_global_image = mat_global_image_ndarray

    mat_global_image[np.where(mat_global_image>255)]=255
    mat_global_image = mat_global_image/np.max(mat_global_image)*255
    mat_global_image = mat_global_image.astype(np.uint8)
    mat_global_image = np.expand_dims(mat_global_image, axis=-1)
    mat_global_image = np.concatenate((mat_global_image, mat_global_image, mat_global_image), axis=-1) # H*W*3

    return mat_global_image


data_root = '/DATA1/pengjianyi'
dataset_root = os.path.join(data_root, 'KITTI/dataset')
pose_root = os.path.join(data_root, 'semanticKITTI/dataset')
source_image_dir = 'KITTI/768x128_image'
source_image_path = os.path.join(data_root, source_image_dir)
threshold = 17
target_image_dir = f'KITTI/768x128_image_bev_v2_sober_{threshold}'
target_image_path = os.path.join(data_root, target_image_dir)
os.makedirs(target_image_path, exist_ok=True)
sequence_list = sorted(os.listdir(source_image_path))
device = torch.device('cuda:0')
torch.cuda.set_device(device)
max_depth = 80.0 # only for the KITTI
max_z = 5.0
min_z = -5.0

config = get_config('zoedepth', 'eval', 'kitti')
config.pretrained_resource = 'local::/home/pengjianyi/.cache/torch/hub/checkpoints/depth_anything_metric_depth_outdoor.pt'
model = build_model(config).cuda()
model.eval()

for sequence in sequence_list:
    source_image_sequence_path = os.path.join(source_image_path, sequence)
    target_image_sequence_path = os.path.join(target_image_path, sequence)
    os.makedirs(target_image_sequence_path, exist_ok=True)
    source_sequence_camera_path = os.path.join(source_image_sequence_path, 'image_2')
    source_sequence_camera_intrinsics_path = os.path.join(source_image_sequence_path, 'image_2_intrinsic')
    target_sequence_camera_path = os.path.join(target_image_sequence_path, 'image_2')
    os.makedirs(target_sequence_camera_path, exist_ok=True)
    file_name_list = sorted(os.listdir(source_sequence_camera_path))
    curr_sequence = my_odometry(sequence=sequence, base_path=dataset_root, pose_path=pose_root)
    curr_calib = curr_sequence.calib
    T_cam0_cam2 = curr_calib['T_ego_cam2']
    T_cam0_LiDAR = curr_calib['T_ego_LiDAR']
    T_LiDAR_cam2 = np.linalg.inv(T_cam0_LiDAR) @ T_cam0_cam2

    for file_name in file_name_list:
        t0 = time.time()
        target_image_file_path = os.path.join(target_sequence_camera_path, file_name)
        if os.path.exists(target_image_file_path):
           print('skip', target_image_file_path)
           continue

        image_file_path = os.path.join(source_sequence_camera_path, file_name)
        intrinsics_file_path = os.path.join(source_sequence_camera_intrinsics_path, file_name.split('.')[0] + '.npy')
        raw_img = cv2.imread(image_file_path)

        t1 = time.time()

        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        image_tensor = transforms.ToTensor()(raw_img).unsqueeze(0).cuda()
        pred = model(image_tensor, dataset='kitti')
        if isinstance(pred, dict):
            pred = pred.get('metric_depth', pred.get('out'))
        elif isinstance(pred, (list, tuple)):
            pred = pred[-1]
        pred = pred.squeeze().detach().cpu().numpy()
        resized_pred = Image.fromarray(pred).resize((768, 128), Image.NEAREST)
        depth = np.array(resized_pred)

        t2 = time.time()

        image_intrinsics = np.load(intrinsics_file_path)

        x, y = np.meshgrid(np.arange(raw_img.shape[1]), np.arange(raw_img.shape[0]))
        x = (x - image_intrinsics[0, 2]) / image_intrinsics[0, 0]
        y = (y - image_intrinsics[1, 2]) / image_intrinsics[1, 1]
        z = depth
        points_in_cam2 = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
        points_to_mul = np.concatenate((points_in_cam2, np.ones((points_in_cam2.shape[0], 1))), axis=-1) # N*4
        points_mul = np.matmul(points_to_mul, T_LiDAR_cam2.T) # N*4
        raw_points = points_mul[:, :3]
        raw_points = np.reshape(raw_points, (-1, 3))


        # 1 Sobel算子
        depth_uint8 = np.uint8(depth / max_depth * 255)
        sobel_x = cv2.Sobel(depth_uint8, cv2.CV_64F, 1, 0, ksize=3)  # 水平边缘
        sobel_y = cv2.Sobel(depth_uint8, cv2.CV_64F, 0, 1, ksize=3)  # 垂直边缘
        sobel = cv2.magnitude(sobel_x, sobel_y)
        edge_mask = np.uint8(sobel > threshold)
        vis_edge_mask_0 = edge_mask * 255
        vis_edge_mask = vis_edge_mask_0.flatten()

        params = pypatchworkpp.Parameters()
        params.sensor_height = 0.0
        params.verbose = False
        PatchworkPLUSPLUS = pypatchworkpp.patchworkpp(params)
        pc_all_reflection = np.concatenate((raw_points, np.ones((raw_points.shape[0], 1))), axis=-1)
        PatchworkPLUSPLUS.estimateGround(pc_all_reflection)
        # pc_only_ground = PatchworkPLUSPLUS.getGround() # (M, 3) np array
        pc_without_ground = PatchworkPLUSPLUS.getNonground() # (N - M, 3) np array
        pc_without_ground_indices = PatchworkPLUSPLUS.getNongroundIndices() # (N - M,) np array
        raw_points1 = pc_without_ground

        vis_edge_mask_without_ground = vis_edge_mask[pc_without_ground_indices]
        raw_points = raw_points1[vis_edge_mask_without_ground == 0, :]

        points = raw_points[(raw_points[:, 2] > min_z) & (raw_points[:, 2] < max_z)]

        bev_image = getBEV(points)
        cv2.imwrite(target_image_file_path, bev_image)

        print('save', target_image_file_path)


        t3 = time.time()

        print('first', t1-t0, 'second', t2-t1, 'third', t3-t2)