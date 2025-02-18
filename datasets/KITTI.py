from torch.utils.data import Dataset
import os
import pickle
import random
import numpy as np
import copy
import cv2
import torch
import faiss
import psutil
import json
from .Kitti_Odometry_dp import my_odometry
from PIL import Image


# 
class kitti(Dataset):

    def __init__(self,
                 data_root:str,
                 pose_root:str,
                 raw_dir_name:str,
                 pc_dir_name:str,
                 image_dir_name:str,
                 coords_filename:str,
                 image_size: list,
                 pc_transform=None,
                 image_transform=None,
                 use_cloud: bool = False,
                 use_image: bool = False,
                 dist_caculation_type: str = 'all_coords_L2',
                 overlap_ratio_type: str = 'points_average_distance',
                 overlap_ratio_cfgs: dict = None,
                 sequence_list: list = ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21'],
                 use_range: bool = False,
                 range_dir_name: str = None,
                 range_transform=None,
                 use_memory_bank: bool = False,
                 use_pc_bev: bool=False,
                 pc_bev_dir_name: str=None,
                 pc_bev_transform=None,
                 use_image_bev: bool=False,
                 image_bev_dir_name: str=None,
                 image_bev_transform=None):
        self.raw_path = os.path.join(data_root, raw_dir_name)
        self.sequence_list = sequence_list
        if pc_dir_name is not None:
            self.pc_path = os.path.join(data_root, pc_dir_name) # TODO: when needed, change every lidar root_path to this
            self.pc_dir_name = pc_dir_name
        else:
            self.pc_path = self.raw_path
            self.pc_dir_name = raw_dir_name
        if image_dir_name is not None:
            self.image_path = os.path.join(data_root, image_dir_name)
            self.image_dir_name = image_dir_name
        else:
            self.image_path = self.raw_path
            self.image_dir_name = raw_dir_name
        if range_dir_name is not None:
            self.range_path = os.path.join(data_root, range_dir_name)
            self.range_dir_name = range_dir_name
        else:
            self.range_path = self.raw_path
            self.range_dir_name = raw_dir_name
        if pc_bev_dir_name is not None:
            self.pc_bev_path = os.path.join(data_root, pc_bev_dir_name)
            self.pc_bev_dir_name = pc_bev_dir_name
        else:
            self.pc_bev_path = self.raw_path
            self.pc_bev_dir_name = raw_dir_name
        if image_bev_dir_name is not None:
            self.image_bev_path = os.path.join(data_root, image_bev_dir_name)
            self.image_bev_dir_name = image_bev_dir_name
        else:
            self.image_bev_path = self.raw_path
            self.image_bev_dir_name = raw_dir_name
        
        if overlap_ratio_type == 'points_average_distance' and (overlap_ratio_cfgs is not None and overlap_ratio_cfgs.reverse):
            with open(os.path.join(data_root, 'my_tool', coords_filename.split('.')[0] + '_reverse.pkl'), 'rb') as f:
                UTM_coord_tensor_reverse = pickle.load(f)
            for seq_ID, seq_UTM_coords in UTM_coord_tensor_reverse.items():
                UTM_coord_tensor_reverse[seq_ID] = torch.tensor(seq_UTM_coords, device='cpu', dtype=torch.float32)
            self.reverse = True
        else:
            self.reverse = False
        
        with open(os.path.join(data_root, 'my_tool', coords_filename), 'rb') as f:
            UTM_coord_tensor = pickle.load(f)
        for seq_ID, seq_UTM_coords in UTM_coord_tensor.items():
            UTM_coord_tensor[seq_ID] = torch.tensor(seq_UTM_coords, device='cpu', dtype=torch.float32)
        self.samples_len = 0

        for seq_ID in sequence_list:
            self.samples_len += len(UTM_coord_tensor[seq_ID])
        
        self.seq_length_dict = {}
        if overlap_ratio_type == 'points_average_distance':
            if dist_caculation_type == 'all_coords_L2':
                for seq_ID, seq_UTM_coords in UTM_coord_tensor.items():
                    sample_size = seq_UTM_coords.shape[0]
                    self.seq_length_dict[seq_ID] = sample_size
            elif dist_caculation_type == 'all_coords_L2_mean':
                for seq_ID, seq_UTM_coords in UTM_coord_tensor.items():
                    curr_seq_UTM_coords = seq_UTM_coords
                    sample_size = curr_seq_UTM_coords.shape[0]
                    UTM_coord_tensor[seq_ID] = curr_seq_UTM_coords.reshape(sample_size, -1, 2).permute(1, 0, 2) # (coord_num, sample_size, 2)
                    self.seq_length_dict[seq_ID] = sample_size
                if overlap_ratio_cfgs is not None and overlap_ratio_cfgs.reverse:
                    for seq_ID, seq_UTM_coords in UTM_coord_tensor_reverse.items():
                        curr_seq_UTM_coords = seq_UTM_coords
                        sample_size = curr_seq_UTM_coords.shape[0]
                        UTM_coord_tensor_reverse[seq_ID] = curr_seq_UTM_coords.reshape(sample_size, -1, 2).permute(1, 0, 2) # (coord_num, sample_size, 2)
            else:
                raise NotImplementedError
        else:
            for seq_ID, seq_UTM_coords in UTM_coord_tensor.items():
                sample_size = seq_UTM_coords.shape[0]
                self.seq_length_dict[seq_ID] = sample_size
        
        if overlap_ratio_type == 'points_average_distance':
            self.UTM_coord_tensor = UTM_coord_tensor
            if overlap_ratio_cfgs is not None and overlap_ratio_cfgs.reverse:
                self.UTM_coord_tensor_reverse = UTM_coord_tensor_reverse
        elif overlap_ratio_type == 'area_overlap':
            self.area_overlap = UTM_coord_tensor
        elif overlap_ratio_type == 'pos_vec_vet':
            self.pos_vec_vet_coords_tensor = UTM_coord_tensor
            self.pos_candidate_distance = overlap_ratio_cfgs['pos_candidate_distance']
            self.pos_final_degree = overlap_ratio_cfgs['pos_final_degree']
        elif overlap_ratio_type == 'exp_dist':
            self.UTM_coord_tensor = UTM_coord_tensor
            self.exp_scale = overlap_ratio_cfgs['exp_scale']
        else:
            raise NotImplementedError
        self.dist_caculation_type = dist_caculation_type

        self.image_size = image_size
        self.pc_transform = pc_transform
        self.image_transform = image_transform
        self.use_cloud = use_cloud
        self.use_image = use_image

        self.range_transform = range_transform
        self.use_range = use_range

        self.pc_bev_transform = pc_bev_transform
        self.use_pc_bev = use_pc_bev

        self.image_bev_transform = image_bev_transform
        self.use_image_bev = use_image_bev

        self.seq_info = {}
        for seq_ID in sequence_list:
            curr_seq_info = my_odometry(sequence=seq_ID, 
                            base_path=self.raw_path, 
                            pose_path=os.path.join(pose_root, raw_dir_name))
            self.seq_info[seq_ID] = curr_seq_info
        
        if use_memory_bank:
            self.memory_bank = {}
            for seq_ID in sequence_list:
                self.memory_bank[seq_ID] = torch.zeros((2, self.seq_length_dict[seq_ID], 256), dtype=torch.float32, device='cpu', requires_grad=False)
    
    def __len__(self):
        return self.samples_len

    def __getitem__(self, idx_list):

        seq_ID = idx_list[0]
        idx = idx_list[1]
        filename = str(idx).zfill(6)
        result = {}
        result['idx_list'] = idx_list
        curr_seq_info = self.seq_info[seq_ID]
        curr_calib = curr_seq_info.calib

        T_first_cam0_curr_cam0 = curr_seq_info.poses[idx]
        T_cam0_LiDAR = curr_calib['T_ego_LiDAR']

        # let's assume the first Lidar frame's coordinate system is the world coordinate system 

        if self.use_cloud:
            if self.pc_dir_name == '16384_to_4096_cliped_fov':
                pc = np.load(os.path.join(self.pc_path, seq_ID, 'velodyne', filename + '_2.npy')).astype(np.float32)
                pc_original = np.load(os.path.join(self.pc_path, seq_ID, 'velodyne', filename + '_1.npy')).astype(np.float32)
                result['cloud_original'] = pc_original
                original_2_downsampled_indices = np.load(os.path.join(self.pc_path, seq_ID, 'velodyne', filename + '_3.npy'))
                result['original_2_downsampled_indices'] = original_2_downsampled_indices
                T_first_cam0_curr_LiDAR = np.matmul(T_first_cam0_curr_cam0, T_cam0_LiDAR).astype(np.float32)
                T_first_LiDAR_curr_LiDAR = np.matmul(np.linalg.inv(T_cam0_LiDAR), T_first_cam0_curr_LiDAR)
                result['cloud_pose_original'] = T_first_LiDAR_curr_LiDAR.astype(np.float32)
            elif self.pc_dir_name == 'dataset':
                raise NotImplementedError
            else:
                raise NotImplementedError
            pc_pose = T_first_LiDAR_curr_LiDAR.astype(np.float32)
            pc, P_random, P_remove_mask, P_shuffle_indices = self.pc_transform(pc)
            P_shuffle_indices = np.argsort(P_shuffle_indices)
            if P_random is not None:
                pc_pose = np.dot(pc_pose, np.linalg.inv(P_random))
            result['cloud'] = pc
            result['cloud_pose'] = pc_pose
            result['cloud_remove_mask'] = P_remove_mask
            result['cloud_shuffle_indices'] = P_shuffle_indices
        
        if self.use_image:
            if self.image_dir_name is not None:
                image_curr_path = os.path.join(self.image_path, seq_ID, 'image_2', filename + '.png')
                image = cv2.imread(image_curr_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                T_cam0_cam2 = curr_calib['T_ego_cam2']
                T_first_cam0_curr_cam2 = np.matmul(T_first_cam0_curr_cam0, T_cam0_cam2).astype(np.float32)
                T_first_LiDAR_curr_cam2 = np.matmul(np.linalg.inv(T_cam0_LiDAR), T_first_cam0_curr_cam2)
                image_pose = T_first_LiDAR_curr_cam2.astype(np.float32)
                curr_P0 = np.load(os.path.join(self.image_path, seq_ID, 'image_2_intrinsic', filename + ".npy")).astype(np.float32)
            else:
                raise NotImplementedError
            image, curr_P0_processed = self.image_transform(image, curr_P0)
            result['image'] = image
            result['P0'] = curr_P0_processed
            result['image_pose'] = image_pose
            result['image_path'] = image_curr_path
        
        if self.use_range:
            if self.range_dir_name == "16384_to_4096_cliped_fov_range_image":
                range_curr_path = os.path.join(self.range_path, seq_ID, 'velodyne', filename + '_1.npy')
                range_img = np.load(range_curr_path) # (64, 200)
                range_to_pc_original_idxs_path = os.path.join(self.range_path, seq_ID, 'velodyne', filename + '_2.npy')
                range_to_pc_original_idxs = np.load(range_to_pc_original_idxs_path) # (64, 224)
                range_img = np.uint8(range_img * 5.1)
                range_img = np.expand_dims(range_img, axis=2) # (64, 224, 1)
                range_img = np.repeat(range_img, 3, axis=2) # (64, 224, 3)
                range_img = self.range_transform(range_img) # don't do resize here
                result['range_img'] = range_img
                result['range_to_pc_original_idxs'] = range_to_pc_original_idxs
            elif self.range_dir_name == '16384_to_4096_cliped_fov_bev':
                pc_bev = Image.open(os.path.join(self.range_path, seq_ID, 'velodyne', filename + '.png'))
                pc_bev = np.array(pc_bev)
                pc_bev = self.range_transform(pc_bev)
                result['range_img'] = pc_bev
            elif self.range_dir_name == '768x128_image_bev':
                image_bev = Image.open(os.path.join(self.range_path, seq_ID, 'image_2', filename + '.png'))
                image_bev = np.array(image_bev)
                image_bev = self.range_transform(image_bev)
                result['range_img'] = image_bev
            else:
                raise NotImplementedError
        
        if self.use_pc_bev:
            if self.pc_bev_dir_name == '16384_to_4096_cliped_fov_bev':
                pc_bev = Image.open(os.path.join(self.pc_bev_path, seq_ID, 'velodyne', filename + '.png'))
                pc_bev = np.array(pc_bev)
                pc_bev = self.pc_bev_transform(pc_bev)
                result['pc_bev'] = pc_bev
            elif self.pc_bev_dir_name == '16384_to_4096_cliped_fov_bev_nonground':
                pc_bev = Image.open(os.path.join(self.pc_bev_path, seq_ID, 'velodyne', filename + '_1.png'))
                pc_bev = np.array(pc_bev)
                pc_bev = self.pc_bev_transform(pc_bev)
                result['pc_bev'] = pc_bev
            else:
                raise NotImplementedError
        
        if self.use_image_bev:
            if (self.image_bev_dir_name == '768x128_image_bev' 
                or self.image_bev_dir_name == '768x128_image_bev_v2_canny_3'
                or self.image_bev_dir_name == '768x128_image_bev_v2_canny_4'
                or self.image_bev_dir_name == '768x128_image_bev_v2_canny_5'
                or self.image_bev_dir_name == '768x128_image_bev_v2_sober_12'
                or self.image_bev_dir_name == '768x128_image_bev_v2_sober_17'
                or self.image_bev_dir_name == '768x128_image_bev_v2_sober_22'
                or self.image_bev_dir_name == '768x128_image_bev_v2_nonground'):
                image_bev = Image.open(os.path.join(self.image_bev_path, seq_ID, 'image_2', filename + '.png'))
                image_bev = np.array(image_bev)
                image_bev = self.image_bev_transform(image_bev)
                result['image_bev'] = image_bev
            else:
                raise NotImplementedError
        
        return result

    def get_level_positives(self, idx_list, interval=30):
        all_masks = []
        curr_seq_UTM_coord_tensor = self.UTM_coord_tensor[idx_list[0]]
        ndx = idx_list[1]
        if self.dist_caculation_type == 'all_coords_L2':
            query_2_database_dist = torch.cdist(curr_seq_UTM_coord_tensor[ndx:ndx+1], curr_seq_UTM_coord_tensor).squeeze(0)
        elif self.dist_caculation_type == 'all_coords_L2_mean':
            query_2_database_dist = torch.cdist(curr_seq_UTM_coord_tensor[:, ndx:ndx+1, :], curr_seq_UTM_coord_tensor).mean(dim=0, keepdim=False).squeeze(0)
        
        if self.reverse:
            curr_seq_UTM_coord_tensor_reverse = self.UTM_coord_tensor_reverse[idx_list[0]]
            query_2_database_dist_reverse = torch.cdist(curr_seq_UTM_coord_tensor[:, ndx:ndx+1, :], curr_seq_UTM_coord_tensor_reverse).mean(dim=0, keepdim=False).squeeze(0)
            query_2_database_dist = torch.minimum(query_2_database_dist, query_2_database_dist_reverse)

        query_2_database_dist_sorted, _ = torch.sort(query_2_database_dist)
        curr_positive_distance = query_2_database_dist_sorted[interval]
        curr_mask = torch.le(query_2_database_dist, curr_positive_distance)
        curr_mask = np.array(curr_mask.cpu(), dtype=np.bool_)
        for i in range(4):
            all_masks.append(curr_mask)
        return all_masks
    
    # area_overlap
    def get_level_positives_v1(self, idx_list, interval=5):
        all_masks = []
        curr_seq_area_overlap = self.area_overlap[idx_list[0]]
        ndx = idx_list[1]
        query_2_database_overlap_ratio = curr_seq_area_overlap[ndx, :].cuda()
        query_2_database_overlap_ratio_sorted, _ = torch.sort(query_2_database_overlap_ratio, descending=True)

        curr_positive_overlap_ratio = query_2_database_overlap_ratio_sorted[interval]
        curr_mask = torch.ge(query_2_database_overlap_ratio, curr_positive_overlap_ratio)
        curr_mask = np.array(curr_mask.cpu(), dtype=np.bool_)
        for i in range(4):
            all_masks.append(curr_mask)
        return all_masks

    # pos_vec_vet
    def get_level_positives_v2(self, idx_list):
        all_masks = []
        curr_pos_vec_vet_coords_tensor = self.pos_vec_vet_coords_tensor[idx_list[0]]
        ndx = idx_list[1]
        query_2_database_dist = torch.cdist(curr_pos_vec_vet_coords_tensor[ndx:ndx+1, :2], curr_pos_vec_vet_coords_tensor[:, :2]) # (1, all_train_sample_size)
        query_2_database_pos_candidate_mask = torch.lt(query_2_database_dist.squeeze(0), self.pos_candidate_distance) # (all_train_sample_size, )
        query_vec = curr_pos_vec_vet_coords_tensor[ndx:ndx+1, :2] - curr_pos_vec_vet_coords_tensor[ndx:ndx+1, 2:] # (1, 2)
        query_vec_normalized = torch.nn.functional.normalize(query_vec, p=2.0, dim=-1) # (1, 2)
        pos_candidate_vec = curr_pos_vec_vet_coords_tensor[query_2_database_pos_candidate_mask][:, :2] - curr_pos_vec_vet_coords_tensor[query_2_database_pos_candidate_mask][:, 2:] # (all_train_sample_size, 2)
        pos_candidate_vec_normalized = torch.nn.functional.normalize(pos_candidate_vec, p=2.0, dim=-1) # (all_train_sample_size, 2)
        query_2_pos_candidate_cos_sim = torch.sum(query_vec_normalized * pos_candidate_vec_normalized, dim=-1) # (all_train_sample_size, )
        query_2_pos_candidate_cos_sim = torch.clamp(query_2_pos_candidate_cos_sim, -1.0, 1.0)
        query_2_pos_candidate_rad = torch.acos(query_2_pos_candidate_cos_sim) # (all_train_sample_size, )
        query_2_pos_candidate_deg = torch.rad2deg(query_2_pos_candidate_rad) # (all_train_sample_size, )
        query_2_final_pos_mask_1 = torch.lt(query_2_pos_candidate_deg, self.pos_final_degree)
        query_2_final_pos_mask_2 = torch.gt(query_2_pos_candidate_deg, -self.pos_final_degree)
        query_2_final_pos_mask = torch.logical_and(query_2_final_pos_mask_1, query_2_final_pos_mask_2)
        query_2_database_pos_candidate_mask_temp = query_2_database_pos_candidate_mask.detach().clone()
        query_2_database_pos_candidate_mask[query_2_database_pos_candidate_mask_temp] = query_2_final_pos_mask
        final_mask = np.array(query_2_database_pos_candidate_mask.cpu(), dtype=np.bool_)
        for i in range(4):
            all_masks.append(final_mask)
        return all_masks

    # exp_dist
    def get_level_positives_v3(self, idx_list, interval=50):
        all_masks = []
        curr_seq_UTM_coord_tensor = self.UTM_coord_tensor[idx_list[0]]
        ndx = idx_list[1]
        query_2_database_dist = torch.cdist(curr_seq_UTM_coord_tensor[ndx:ndx+1, :2], curr_seq_UTM_coord_tensor[:, :2]).squeeze(0)
        query_2_database_sim = torch.exp(-self.exp_scale * query_2_database_dist)
        query_2_database_sim_sorted, _ = torch.sort(query_2_database_sim, descending=True)
        curr_positive_sim = query_2_database_sim_sorted[interval]
        curr_mask = torch.ge(query_2_database_sim, curr_positive_sim)
        curr_mask = np.array(curr_mask.cpu(), dtype=np.bool_)
        for i in range(4):
            all_masks.append(curr_mask)
        return all_masks
    
    # pos_vec_vet batchs
    def get_sp_positives_1(self, ndxs, curr_seq_ID):
        curr_seq_pos_vec_vet_coords_tensor = self.pos_vec_vet_coords_tensor[curr_seq_ID]
        query_2_query_dist = torch.cdist(curr_seq_pos_vec_vet_coords_tensor[ndxs, :2], curr_seq_pos_vec_vet_coords_tensor[ndxs, :2]) # (ndxs_num, ndxs_num)
        query_2_query_pos_candidate_mask_1 = torch.lt(query_2_query_dist, self.pos_candidate_distance) # (ndxs_num, ndxs_num)
        query_vec = curr_seq_pos_vec_vet_coords_tensor[ndxs, :2] - curr_seq_pos_vec_vet_coords_tensor[ndxs, 2:] # (ndxs_num, 2)
        query_vec_normalized = torch.nn.functional.normalize(query_vec, p=2.0, dim=-1) # (ndxs_num, 2)

        query_2_query_cos_sim = torch.sum(query_vec_normalized.unsqueeze(1) * query_vec_normalized.unsqueeze(0), dim=-1) # (ndxs_num, ndxs_num)
        query_2_query_cos_sim = torch.clamp(query_2_query_cos_sim, -1.0, 1.0)
        query_2_query_rad = torch.acos(query_2_query_cos_sim) # (ndxs_num, ndxs_num)
        query_2_query_deg = torch.rad2deg(query_2_query_rad) # (ndxs_num, )
        query_2_query_pos_candidate_mask_2_1 = torch.lt(query_2_query_deg, self.pos_final_degree)
        query_2_query_pos_candidate_mask_2_2 = torch.gt(query_2_query_deg, -self.pos_final_degree)
        query_2_query_pos_candidate_mask_2 = torch.logical_and(query_2_query_pos_candidate_mask_2_1, query_2_query_pos_candidate_mask_2_2)
        final_mask = torch.logical_and(query_2_query_pos_candidate_mask_1, query_2_query_pos_candidate_mask_2)
        return final_mask


class kittiEval(Dataset):

    def __init__(self,
                 data_root:str,
                 raw_dir_name:str,
                 pc_dir_name:str,
                 image_dir_name:str,
                 coords_filename:str,
                 image_size: list,
                 pc_transform=None,
                 image_transform=None,
                 use_cloud: bool = False,
                 use_image: bool = False,
                 sequence_list: list = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10'],
                 true_neighbour_dist: float = 10.0,
                 use_range: bool = False,
                 range_dir_name: str = None,
                 range_transform=None,
                 use_pc_bev: bool=False,
                 pc_bev_dir_name: str=None,
                 pc_bev_transform=None,
                 use_image_bev: bool=False,
                 image_bev_dir_name: str=None,
                 image_bev_transform=None):
        self.raw_path = os.path.join(data_root, raw_dir_name)
        self.sequence_list = sequence_list
        if pc_dir_name is not None:
            self.pc_path = os.path.join(data_root, pc_dir_name) # TODO: when needed, change every lidar root_path to this
            self.pc_dir_name = pc_dir_name
        else:
            self.pc_path = self.raw_path
            self.pc_dir_name = raw_dir_name
        if image_dir_name is not None:
            self.image_path = os.path.join(data_root, image_dir_name)
            self.image_dir_name = image_dir_name
        else:
            self.image_path = self.raw_path
            self.image_dir_name = raw_dir_name
        
        if range_dir_name is not None:
            self.range_path = os.path.join(data_root, range_dir_name)
            self.range_dir_name = range_dir_name
        else:
            self.range_path = self.raw_path
            self.range_dir_name = raw_dir_name
        
        if pc_bev_dir_name is not None:
            self.pc_bev_path = os.path.join(data_root, pc_bev_dir_name)
            self.pc_bev_dir_name = pc_bev_dir_name
        else:
            self.pc_bev_path = self.raw_path
            self.pc_bev_dir_name = raw_dir_name
        if image_bev_dir_name is not None:
            self.image_bev_path = os.path.join(data_root, image_bev_dir_name)
            self.image_bev_dir_name = image_bev_dir_name
        else:
            self.image_bev_path = self.raw_path
            self.image_bev_dir_name = raw_dir_name
        
        with open(os.path.join(data_root, 'my_tool', coords_filename), 'rb') as f:
            UTM_coord_tensor = pickle.load(f)
        self.true_neighbors_matrix = {}
        for seq_ID, seq_UTM_coords in UTM_coord_tensor.items():
            seq_UTM_coords = torch.tensor(seq_UTM_coords, dtype=torch.float32).cuda() # (curr_seq_sample_size, 2)
            curr_dist_matrix = torch.cdist(seq_UTM_coords.unsqueeze(0), seq_UTM_coords.unsqueeze(0), p=2.0).squeeze(0) # (curr_seq_sample_size, curr_seq_sample_size)
            self.true_neighbors_matrix[seq_ID] = torch.lt(curr_dist_matrix, true_neighbour_dist)
        
        self.samples_len = 0
        self.samples_length_cumsum = [0]
        for seq_ID in sequence_list:
            self.samples_len += len(self.true_neighbors_matrix[seq_ID])
            self.samples_length_cumsum.append(self.samples_len)
        
        self.samples_length_cumsum = np.array(self.samples_length_cumsum[:-1])
        
        self.sequence_list = sequence_list
        self.image_size = image_size
        self.pc_transform = pc_transform
        self.image_transform = image_transform
        self.use_cloud = use_cloud
        self.use_image = use_image

        self.pc_bev_transform = pc_bev_transform
        self.use_pc_bev = use_pc_bev

        self.image_bev_transform = image_bev_transform
        self.use_image_bev = use_image_bev

        self.range_transform = range_transform
        self.use_range = use_range

        
    
    def __len__(self):
        return self.samples_len

    def __getitem__(self, idx_list):

        seq_ID = idx_list[0]
        idx = idx_list[1]
        filename = str(idx).zfill(6)
        result = {}
        result['idx_list'] = idx_list

        # let's assume the first Lidar frame's coordinate system is the world coordinate system 

        if self.use_cloud:
            if self.pc_dir_name == '16384_to_4096_cliped_fov':
                pc = np.load(os.path.join(self.pc_path, seq_ID, 'velodyne', filename + '_2.npy'))
            elif self.pc_dir_name == 'dataset':
                raise NotImplementedError
            else:
                raise NotImplementedError
            pc, _, _, _ = self.pc_transform(pc)
            result['cloud'] = pc
        
        if self.use_image:
            if self.image_dir_name is not None:
                image_curr_path = os.path.join(self.image_path, seq_ID, 'image_2', filename + '.png')
                image = cv2.imread(image_curr_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                raise NotImplementedError
            image = self.image_transform(image)
            result['image'] = image
            result['image_path'] = image_curr_path
        
        if self.use_range:
            if self.range_dir_name == "16384_to_4096_cliped_fov_range_image":
                range_curr_path = os.path.join(self.range_path, seq_ID, 'velodyne', filename + '_1.npy')
                range_img = np.load(range_curr_path) # (64, 200)
                range_img = np.uint8(range_img * 5.1)
                range_img = np.expand_dims(range_img, axis=2) # (64, 200, 1)
                range_img = np.repeat(range_img, 3, axis=2) # (64, 200, 3)
                range_img = self.range_transform(range_img) # don't do resize here
                result['range_img'] = range_img
            elif self.range_dir_name == '16384_to_4096_cliped_fov_bev':
                pc_bev = Image.open(os.path.join(self.range_path, seq_ID, 'velodyne', filename + '.png'))
                pc_bev = np.array(pc_bev)
                pc_bev = self.range_transform(pc_bev)
                result['range_img'] = pc_bev
            elif self.range_dir_name == '768x128_image_bev':
                image_bev = Image.open(os.path.join(self.range_path, seq_ID, 'image_2', filename + '.png'))
                image_bev = np.array(image_bev)
                image_bev = self.range_transform(image_bev)
                result['range_img'] = image_bev
            else:
                raise NotImplementedError
        
        if self.use_pc_bev:
            if self.pc_bev_dir_name == '16384_to_4096_cliped_fov_bev':
                pc_bev = Image.open(os.path.join(self.pc_bev_path, seq_ID, 'velodyne', filename + '.png'))
                pc_bev = np.array(pc_bev)
                pc_bev = self.pc_bev_transform(pc_bev)
                result['pc_bev'] = pc_bev
            elif self.pc_bev_dir_name == '16384_to_4096_cliped_fov_bev_nonground':
                pc_bev = Image.open(os.path.join(self.pc_bev_path, seq_ID, 'velodyne', filename + '_1.png'))
                pc_bev = np.array(pc_bev)
                pc_bev = self.pc_bev_transform(pc_bev)
                result['pc_bev'] = pc_bev
            else:
                raise NotImplementedError
        
        if self.use_image_bev:
            if (self.image_bev_dir_name == '768x128_image_bev'
                or self.image_bev_dir_name == '768x128_image_bev_v2_canny_3'
                or self.image_bev_dir_name == '768x128_image_bev_v2_canny_4'
                or self.image_bev_dir_name == '768x128_image_bev_v2_canny_5'
                or self.image_bev_dir_name == '768x128_image_bev_v2_sober_12'
                or self.image_bev_dir_name == '768x128_image_bev_v2_sober_17'
                or self.image_bev_dir_name == '768x128_image_bev_v2_sober_22'
                or self.image_bev_dir_name == '768x128_image_bev_v2_nonground'):
                image_bev = Image.open(os.path.join(self.image_bev_path, seq_ID, 'image_2', filename + '.png'))
                image_bev = np.array(image_bev)
                image_bev = self.image_bev_transform(image_bev)
                result['image_bev'] = image_bev
            else:
                raise NotImplementedError
        
        return result