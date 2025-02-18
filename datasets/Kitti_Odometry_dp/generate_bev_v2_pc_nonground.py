import os
import pickle
import numpy as np
import cv2
import sys
import torch
import open3d as o3d
import torch_scatter
import time
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
source_pc_dir = 'KITTI/16384_to_4096_cliped_fov'
source_pc_path = os.path.join(data_root, source_pc_dir)
target_pc_dir = 'KITTI/16384_to_4096_cliped_fov_bev_nonground'
target_pc_path = os.path.join(data_root, target_pc_dir)
os.makedirs(target_pc_path, exist_ok=True)
sequence_list = sorted(os.listdir(source_pc_path))
max_depth = 80.0 # only for the KITTI
max_z = 5.0
min_z = -5.0

for sequence in sequence_list:
    source_pc_sequence_path = os.path.join(source_pc_path, sequence)
    target_pc_sequence_path = os.path.join(target_pc_path, sequence)
    os.makedirs(target_pc_sequence_path, exist_ok=True)
    source_sequence_velodyne_path = os.path.join(source_pc_sequence_path, 'velodyne')
    target_sequence_velodyne_path = os.path.join(target_pc_sequence_path, 'velodyne')
    os.makedirs(target_sequence_velodyne_path, exist_ok=True)
    file_name_list = sorted(os.listdir(source_sequence_velodyne_path))
    file_name_list = [file_name for file_name in file_name_list if file_name.endswith('_1.npy')]
    file_name_list = sorted(file_name_list)

    for file_name in file_name_list:
        t0 = time.time()
        target_pc_file_path = os.path.join(target_sequence_velodyne_path, file_name.split('.')[0] + '.png')
        if os.path.exists(target_pc_file_path):
           print('skip', target_pc_file_path)
           continue

        pc_file_path = os.path.join(source_sequence_velodyne_path, file_name.split('.')[0] + '.npy')
        raw_pc_0 = np.load(pc_file_path)

        params = pypatchworkpp.Parameters()
        params.sensor_height = 0.0
        params.verbose = False
        PatchworkPLUSPLUS = pypatchworkpp.patchworkpp(params)
        pc_all_reflection = np.concatenate((raw_pc_0, np.ones((raw_pc_0.shape[0], 1))), axis=-1)
        PatchworkPLUSPLUS.estimateGround(pc_all_reflection)
        # pc_only_ground = PatchworkPLUSPLUS.getGround() # (M, 3) np array
        pc_without_ground = PatchworkPLUSPLUS.getNonground() # (N - M, 3) np array
        # pc_without_ground_indices = PatchworkPLUSPLUS.getNongroundIndices() # (N - M,) np array
        raw_pc = pc_without_ground


        pc_points = raw_pc[(raw_pc[:, 2] > min_z) & (raw_pc[:, 2] < max_z)]
        bev_pc = getBEV(pc_points)
        cv2.imwrite(target_pc_file_path, bev_pc)
        print('save', target_pc_file_path, time.time() - t0)