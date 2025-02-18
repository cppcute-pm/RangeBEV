from .Boreas_dp import BoreasDataset_U, voxel_downsample, FPS_downsample
from torch.utils.data import Dataset
import os
import pickle
import random
import numpy as np
from pyboreas.utils.utils import load_lidar
import copy
import cv2
import torch
import faiss
import psutil
import json

class boreas_v2(Dataset):

    def __init__(self, 
                 data_root: str,
                 raw_dir_name: str, 
                 pc_dir_name: str,
                 rendered_dir_name: str,
                 image_dir_name: str,
                 mask_dir_name: str,
                 tool_name: str,
                 coords_filename: str,
                 minuse_lidar_filename: str,
                 positive_distance: float,
                 non_negative_distance: float,
                 positive_distance_list: list,
                 lidar2image_filename: str,
                 image_size: list,
                 mask_size: list,
                 render_size: list, 
                 pc_transform=None, 
                 pc_preprocess=None,
                 image_transform=None,
                 render_transform=None,
                 mask_transform=None,
                 render_view_num=None,
                 use_cloud: bool = False, 
                 use_render: bool = False,
                 use_image: bool = False,
                 use_mask: bool = False,
                 ratio_strategy: str = 'mean',
                 relative_strategy: bool = True,
                 img_neighbor_num: int = 1,
                 dist_caculation_type: str = 'all_coords_L2',
                 rgb_depth_label_dir_name: str = None,
                 use_rgb_depth_label: bool = False,
                 # below are for the semantic based methods options,
                 use_semantic_label: bool = False,
                 pc_semantic_label_dir_name: str = None,
                 img_semantic_label_dir_name: str = None,
                 use_label_correspondence_table: bool = False,
                 overlap_ratio_type: str = 'points_average_distance',
                 ):
        self.raw_path = os.path.join(data_root, raw_dir_name)
        self.official_dataset = BoreasDataset_U(self.raw_path)
        if pc_dir_name is not None:
            self.pc_path = os.path.join(data_root, pc_dir_name) # TODO: when needed, change every lidar root_path to this
            self.pc_dir_name = pc_dir_name
        else:
            self.pc_path = self.raw_path
            self.pc_dir_name = raw_dir_name
        if rendered_dir_name is not None:
            self.rendered_path = os.path.join(data_root, rendered_dir_name)
            self.rendered_dir_name = rendered_dir_name
        else:
            self.rendered_path = self.raw_path
            self.rendered_dir_name = raw_dir_name
        if image_dir_name is not None:
            self.image_path = os.path.join(data_root, image_dir_name)
            self.image_dir_name = image_dir_name
        else:
            self.image_path = self.raw_path
            self.image_dir_name = raw_dir_name
        if mask_dir_name is not None:
            self.mask_path = os.path.join(data_root, mask_dir_name)
        else:
            self.mask_path = self.raw_path
        
        if rgb_depth_label_dir_name is not None:
            self.rgb_depth_label_path = os.path.join(data_root, rgb_depth_label_dir_name)
            self.rgb_depth_label_dir_name = rgb_depth_label_dir_name
        else:
            self.rgb_depth_label_path = self.raw_path
            self.rgb_depth_label_dir_name = raw_dir_name
        
        if overlap_ratio_type == 'points_average_distance':
            self.UTM_coord_path = os.path.join(self.raw_path, tool_name, coords_filename)
            UTM_coord = np.load(self.UTM_coord_path)
            if dist_caculation_type == 'all_coords_L2':
                self.UTM_coord_tensor = torch.tensor(UTM_coord, device='cpu', dtype=torch.float32)
            elif dist_caculation_type == 'all_coords_L2_mean':
                UTM_coord_tensor = torch.tensor(UTM_coord, device='cpu', dtype=torch.float32)
                sample_size = UTM_coord_tensor.shape[0]
                self.UTM_coord_tensor = UTM_coord_tensor.reshape(sample_size, -1, 2).permute(1, 0, 2) # (coord_num, sample_size, 2)
            else:
                raise NotImplementedError
            self.pose_dist_threshold = positive_distance
        elif overlap_ratio_type == 'area_overlap':
            self.area_overlap_path = os.path.join(self.raw_path, tool_name, coords_filename)
            self.area_overlap = np.load(self.area_overlap_path) # (all_train_sample_size, all_train_sample_size)
        elif overlap_ratio_type == 'pos_vec_vet':
            self.pos_vec_vet_coords_path = os.path.join(self.raw_path, tool_name, coords_filename)
            pos_vec_vet_coords = np.load(self.pos_vec_vet_coords_path) # (all_train_sample_size, 4)
            self.pos_vec_vet_coords_tensor = torch.tensor(pos_vec_vet_coords, device='cpu', dtype=torch.float32)
            self.pos_candidate_distance = positive_distance_list[0]
            self.pos_final_degree = positive_distance_list[1]
        elif overlap_ratio_type == 'exp_dist' or 'exp_dist_v2':
            self.UTM_coord_path = os.path.join(self.raw_path, tool_name, coords_filename)
            UTM_coord = np.load(self.UTM_coord_path)
            self.UTM_coord_tensor = torch.tensor(UTM_coord, device='cpu', dtype=torch.float32)
            self.exp_scale = positive_distance
        else:
            raise NotImplementedError
        
        if use_semantic_label:
            self.use_semantic_label = use_semantic_label
            self.pc_semantic_label_path = os.path.join(data_root, pc_semantic_label_dir_name)
            self.img_semantic_label_path = os.path.join(data_root, img_semantic_label_dir_name)
            self.use_label_correspondence_table = use_label_correspondence_table
            if self.use_label_correspondence_table:
                with open(os.path.join(self.raw_path, tool_name, 'label_correspondence_table.json'), 'r') as f:
                    self.label_correspondence_table = json.load(f)
            else:
                label_correspondence_table_ndarray = np.arange(0, 19, dtype=np.int32) # (19,) is the number of semantic labels of Cityscapes
                label_correspondence_table_ndarray = np.expand_dims(label_correspondence_table_ndarray, axis=1) # (19, 1)
                label_correspondence_table_ndarray = np.repeat(label_correspondence_table_ndarray, 2, axis=1) # (19, 2)
                self.label_correspondence_table = label_correspondence_table_ndarray.tolist()
        else:
            self.use_semantic_label = False
        
        self.dist_caculation_type = dist_caculation_type

        # self.UTM_index = faiss.IndexFlatL2(dist_dim)

        # cpu_UTM_index = faiss.IndexFlatL2(dist_dim)
        # res = faiss.StandardGpuResources()
        # self.UTM_index = faiss.index_cpu_to_gpu(res, torch.cuda.current_device(), cpu_UTM_index)

        # self.UTM_index.add(UTM_coord)

        # pid = os.getpid()
        # p = psutil.Process(pid)
        # mem_info = p.memory_info()
        # print(f"before load minuse_lidar_path, RSS: {mem_info.rss / 1024 / 1024} MB")
        # print(f"before load minuse_lidar_path, VMS: {mem_info.vms / 1024 / 1024} MB")


        minuse_lidar_path = os.path.join(self.raw_path, tool_name, minuse_lidar_filename)
        self.minuse_lidar = pickle.load(open(minuse_lidar_path, "rb"))

        # mem_info = p.memory_info()
        # print(f"after load minuse_lidar_path, RSS: {mem_info.rss / 1024 / 1024} MB")
        # print(f"after load minuse_lidar_path, VMS: {mem_info.vms / 1024 / 1024} MB")

        self.positive_distance = positive_distance
        self.non_negative_distance = non_negative_distance
        self.positive_distance_list = sorted(positive_distance_list)
        traversal_num = []
        self.seq_IDs = []
        self.traversal_idxs = {}
        traversal_idxs_CTR = 0
        for seq_ID, lidar_idxs in self.minuse_lidar.items():
            traversal_num.append(len(lidar_idxs))
            self.seq_IDs.append(seq_ID)
            self.traversal_idxs[seq_ID] = []
            for _ in lidar_idxs:
                self.traversal_idxs[seq_ID].append(traversal_idxs_CTR)
                traversal_idxs_CTR += 1
        traversal_num = np.array(traversal_num, dtype=np.int64)
        self.traversal_cumsum = np.cumsum(traversal_num, axis=-1)
        self.traversal_num = len(self.official_dataset.sequences)
            
        self.lidar2image_path = os.path.join(self.raw_path, tool_name, lidar2image_filename)
        self.lidar2image = pickle.load(open(self.lidar2image_path, 'rb'))
        query_path = "/media/group2/data/pengjianyi/Boreas_minuse/my_tool/class_split_train_queries.pickle"
        # self.queries = pickle.load(open(query_path, "rb"))

        self.image_size = image_size
        self.mask_size = mask_size
        self.render_size = render_size
        self.pc_transform = pc_transform
        self.pc_preprocess = pc_preprocess
        self.image_transform = image_transform
        self.render_transform = render_transform
        self.mask_transform = mask_transform
        self.render_view_num = render_view_num
        self.use_cloud = use_cloud
        self.use_render = use_render
        self.use_image = use_image
        self.use_mask = use_mask
        self.use_rgb_depth_label = use_rgb_depth_label
        self.ratio_strategy = ratio_strategy
        self.relative_strategy = relative_strategy
        self.img_neighbor_num = img_neighbor_num

        # mem_info = p.memory_info()
        # print(f"after make dataset, RSS: {mem_info.rss / 1024 / 1024} MB")
        # print(f"after make dataset, VMS: {mem_info.vms / 1024 / 1024} MB")

    def __len__(self):
        return self.traversal_cumsum[-1]

    def __getitem__(self, idx):
        result = {}
        result["ndx"] = idx
        curr_traversal = np.searchsorted(self.traversal_cumsum, idx, side="right")
        sequence_ID = self.seq_IDs[curr_traversal]
        if curr_traversal == 0:
            real_id = idx
        else:
            real_id = idx - self.traversal_cumsum[curr_traversal]
        real_id = self.minuse_lidar[sequence_ID][real_id]
        curr_seq = self.official_dataset.get_seq_from_ID(sequence_ID)
        curr_lidar_frame = curr_seq.lidar_frames[real_id]

        if self.use_cloud:
            if self.pc_dir_name == "Boreas_minuse":
                pc = load_lidar(curr_lidar_frame.path)[:, :3]
            elif self.pc_dir_name == "Boreas_lidar_4096":
                lidar_pre_path = curr_lidar_frame.path
                seq_ID, lidar_dir, pc_file_name = lidar_pre_path.split('/')[-3:]
                lidar_curr_path = os.path.join(self.pc_path, seq_ID, lidar_dir, pc_file_name)
                pc = np.load(lidar_curr_path.split('.')[0] + '.npy')
            elif self.pc_dir_name == "Boreas_minuse_163840_to_4096" or self.pc_dir_name == "Boreas_minuse_40960_to_4096_cliped_fov":
                lidar_pre_path = curr_lidar_frame.path
                seq_ID, lidar_dir, pc_file_name = lidar_pre_path.split('/')[-3:]
                lidar_curr_path_prefix = os.path.join(self.pc_path, seq_ID, lidar_dir, pc_file_name.split('.')[0])
                pc = np.load(lidar_curr_path_prefix + '_1.npy')
                pc_original = np.load(lidar_curr_path_prefix + '_2.npy')
                result['cloud_original'] = pc_original
                original_2_downsampled_indices = np.load(lidar_curr_path_prefix + '_3.npy')
                result['original_2_downsampled_indices'] = original_2_downsampled_indices
                result['cloud_pose_original'] = curr_lidar_frame.pose.astype(np.float32)
            pc_pose = curr_lidar_frame.pose.astype(np.float32)
            pc = self.preprocess_pc(pc)
            pc, P_random, P_remove_mask, P_shuffle_indices = self.pc_transform(pc)
            P_shuffle_indices = np.argsort(P_shuffle_indices)
            if P_random is not None:
                pc_pose = np.dot(pc_pose, np.linalg.inv(P_random))
            result['cloud'] = pc
            result['cloud_pose'] = pc_pose
            result['cloud_remove_mask'] = P_remove_mask
            result['cloud_shuffle_indices'] = P_shuffle_indices

            if self.use_semantic_label:
                pc_semantic_label_path = os.path.join(self.pc_semantic_label_path, seq_ID, lidar_dir, pc_file_name.split('.')[0] + "_semantic_label.npy")
                pc_cluster_label_path = os.path.join(self.pc_semantic_label_path, seq_ID, lidar_dir, pc_file_name.split('.')[0] + "_dbscan_cluster_label.npy")
                result['pc_semantic_label'] = np.load(pc_semantic_label_path) # (4096, )
                result['pc_dbscan_cluster_label'] = np.load(pc_cluster_label_path) # (4096, )

        if self.use_render:
            lidar_pre_path = curr_lidar_frame.path
            seq_ID, lidar_dir, pc_file_name = lidar_pre_path.split('/')[-3:]
            depth_file_name = pc_file_name.split('.')[0] + '.png'
            depth_file_path = os.path.join(self.rendered_path, seq_ID, lidar_dir, depth_file_name)
            render_img = cv2.imread(depth_file_path, cv2.IMREAD_ANYDEPTH)
            render_img = np.expand_dims(render_img.astype(np.float32) / 65535, axis=-1) # (H, W, 1)
            render_img = np.tile(render_img * 255, (1, 1, 3)) # (H, W, 3)
            render_img = self.render_transform(render_img) # (3, H, W)
            result['render_img'] = render_img

        if self.use_image:
            curr_image_idxs = self.lidar2image[sequence_ID][str(real_id)]
            curr_image_idx = random.choice(curr_image_idxs[:self.img_neighbor_num])
            curr_image_frame = curr_seq.camera_frames[curr_image_idx]
            image_pose = curr_image_frame.pose.astype(np.float32)
            image_pre_path = curr_image_frame.path
            seq_ID, image_dir, img_file_name = image_pre_path.split('/')[-3:]
            image_curr_path = os.path.join(self.image_path, seq_ID, image_dir, img_file_name)
            image = cv2.imread(image_curr_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.image_dir_name == "Boreas_224x224_image":
                P0_path = os.path.join(self.image_path, seq_ID, 'calib', 'intrinsics.npy')
                curr_P0 = np.load(P0_path).astype(np.float32)
            else:
                curr_P0 = curr_seq.calib.P0.astype(np.float32)
                curr_P0 = curr_P0[:3, :3]
            image, curr_P0_processed = self.image_transform(image, curr_P0)
            result['image'] = image
            result['P0'] = curr_P0_processed
            result['image_pose'] = image_pose
            result['image_path'] = image_curr_path

            if self.use_semantic_label:
                img_semantic_label_path = os.path.join(self.img_semantic_label_path, seq_ID, image_dir, img_file_name.split('.')[0] + "_semantic_label.npy")
                img_cluster_label_path = os.path.join(self.img_semantic_label_path, seq_ID, image_dir, img_file_name.split('.')[0] + "_ccl_cluster_label.npy")
                result['img_semantic_label'] = np.load(img_semantic_label_path) # (224, 224)
                result['img_ccl_cluster_label'] = np.load(img_cluster_label_path) # (224, 224)
            
        if self.use_mask:
            pass

        if self.use_rgb_depth_label:
            curr_image_idxs = self.lidar2image[sequence_ID][str(real_id)]
            curr_image_idx = random.choice(curr_image_idxs[:self.img_neighbor_num])
            curr_image_frame = curr_seq.camera_frames[curr_image_idx]
            image_pose = curr_image_frame.pose.astype(np.float32)
            image_pre_path = curr_image_frame.path
            seq_ID, image_dir, img_file_name = image_pre_path.split('/')[-3:]
            rgb_depth_label_curr_path = os.path.join(self.rgb_depth_label_path, seq_ID, image_dir, img_file_name)
            if self.rgb_depth_label_dir_name == "Boreas_minuse_image_depth_postprocess":
                rgb_depth_label = cv2.imread(rgb_depth_label_curr_path, cv2.IMREAD_ANYDEPTH)
                rgb_depth_label = rgb_depth_label.astype(np.float32) / 65535 # (H, W)
            elif self.rgb_depth_label_dir_name == "Boreas_minuse_image_depth_nonprocess":
                rgb_depth_label = np.load(rgb_depth_label_curr_path.split('.')[0] + '.npy')
            else:
                raise NotImplementedError
            result['rgb_depth_label'] = rgb_depth_label # (H, W)

        return result
    
    def preprocess_pc(self, pc):
        if self.pc_preprocess['mode'] == 0:
            return pc
        elif self.pc_preprocess['mode'] == 1:
            return voxel_downsample(pc, self.pc_preprocess['voxel_size'])
        elif self.pc_preprocess['mode'] == 2:
            return FPS_downsample(pc, self.pc_preprocess['num_points'])
    
    def get_positives(self, ndx):
        k = int(self.positive_distance * 10 + 400)
        if (not isinstance(ndx, list)) and (not isinstance(ndx, np.ndarray)) and (not isinstance(ndx, torch.Tensor)):
            D, I = self.UTM_index.search(self.UTM_coord_tensor[ndx:ndx+1, :], k)
            len_ndx = 1
        else:
            D, I = self.UTM_index.search(self.UTM_coord_tensor[ndx, :], k)
            len_ndx = len(ndx)
        all_inrange_indices = []
        for i in range(len_ndx):
            inrange_indices = np.nonzero(D[i] <= self.positive_distance**2)[0]
            all_inrange_indices.append(I[i, inrange_indices])
        if (not isinstance(ndx, list)) and (not isinstance(ndx, np.ndarray)):
            return list(all_inrange_indices[0])
        else:
            return all_inrange_indices
    
    def get_level_positives(self, ndx):
        k = int(self.positive_distance_list[-1] * 10 + 400)
        if (not isinstance(ndx, list)) and (not isinstance(ndx, np.ndarray)):
            D, I = self.UTM_index.search(self.UTM_coord_tensor[ndx:ndx+1, :], k)
            len_ndx = 1
        else:
            D, I = self.UTM_index.search(self.UTM_coord_tensor[ndx, :], k)
            len_ndx = len(ndx)
        all_indices = []
        for i in range(len_ndx):
            curr_ndx_indices = []
            previous_I_indices = np.array([], dtype=np.int64)
            for j in range(len(self.positive_distance_list)):
                I_indices = np.nonzero(D[i] <= self.positive_distance_list[j]**2)[0]
                curr_I_indices = np.setdiff1d(I_indices, previous_I_indices)
                curr_ndx_indices.append(I[i, curr_I_indices])
                previous_I_indices = I_indices
            all_indices.append(curr_ndx_indices)
        
        return all_indices

    def get_distance_positives(self, ndx, distance):
        k = int(distance * 10 + 400)
        if (not isinstance(ndx, list)) and (not isinstance(ndx, np.ndarray)):
            D, I = self.UTM_index.search(self.UTM_coord_tensor[ndx:ndx+1, :], k)
            len_ndx = 1
        else:
            D, I = self.UTM_index.search(self.UTM_coord_tensor[ndx, :], k)
            len_ndx = len(ndx)
        all_inrange_indices = []
        for i in range(len_ndx):
            inrange_indices = np.nonzero(D[i] <= distance**2)[0]
            all_inrange_indices.append(I[i, inrange_indices])
        
        return all_inrange_indices
    

    def get_non_negatives(self, ndx):
        k = int(self.non_negative_distance * 10 + 400)
        if (not isinstance(ndx, list)) and (not isinstance(ndx, np.ndarray)):
            D, I = self.UTM_index.search(self.UTM_coord_tensor[ndx:ndx+1, :], k)
            len_ndx = 1
        else:
            D, I = self.UTM_index.search(self.UTM_coord_tensor[ndx, :], k)
            len_ndx = len(ndx)
        all_inrange_indices = []
        for i in range(len_ndx):
            I_indices = np.nonzero(D[i] <= self.non_negative_distance**2)[0]
            all_inrange_indices.append(I[i, I_indices])
        if (not isinstance(ndx, list)) and (not isinstance(ndx, np.ndarray)):
            return list(all_inrange_indices[0])
        else:
            return all_inrange_indices
    
    def get_positives_v2(self, ndx):
        query_2_database_dist = torch.cdist(self.UTM_coord_tensor[ndx:ndx+1], self.UTM_coord_tensor).squeeze(0)
        all_inrange_indices = torch.nonzero(torch.le(query_2_database_dist, self.positive_distance), as_tuple=False).squeeze(-1)
        all_inrange_indices = set((all_inrange_indices.cpu().numpy()).tolist())
        return all_inrange_indices

    def get_level_positives_v2(self, ndx):
        all_indices = []
        query_2_database_dist = torch.cdist(self.UTM_coord_tensor[ndx:ndx+1], self.UTM_coord_tensor).squeeze(0)
        for i in range(len(self.positive_distance_list)):
            if i == 0:
                curr_indices = torch.nonzero(torch.le(query_2_database_dist, self.positive_distance_list[i]), as_tuple=False).squeeze(-1)
            else:
                mask1 = torch.gt(query_2_database_dist, self.positive_distance_list[i-1])
                mask2 = torch.le(query_2_database_dist, self.positive_distance_list[i])
                mask = torch.logical_and(mask1, mask2)
                curr_indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            curr_indices = set((curr_indices.cpu().numpy()).tolist())
            all_indices.append(curr_indices)
        return all_indices

    
    def get_level_positives_v3(self, ndx):
        all_indices = []
        query_2_database_dist = torch.cdist(self.UTM_coord_tensor[ndx:ndx+1], self.UTM_coord_tensor).squeeze(0)
        for i in range(len(self.positive_distance_list)):
            if i == 0:
                curr_indices = torch.nonzero(torch.le(query_2_database_dist, self.positive_distance_list[i]), as_tuple=False).squeeze(-1)
            else:
                mask1 = torch.gt(query_2_database_dist, self.positive_distance_list[i-1])
                mask2 = torch.le(query_2_database_dist, self.positive_distance_list[i])
                mask = torch.logical_and(mask1, mask2)
                curr_indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            curr_indices = np.array(curr_indices.cpu(), dtype=np.int32)
            all_indices.append(curr_indices)
        return all_indices
    

    def get_level_positives_v4(self, ndx):
        all_masks = []
        query_2_database_dist = torch.cdist(self.UTM_coord_tensor[ndx:ndx+1], self.UTM_coord_tensor).squeeze(0)
        for i in range(len(self.positive_distance_list)):
            if i == 0:
                curr_mask = torch.le(query_2_database_dist, self.positive_distance_list[i])
            else:
                mask1 = torch.gt(query_2_database_dist, self.positive_distance_list[i-1])
                mask2 = torch.le(query_2_database_dist, self.positive_distance_list[i])
                curr_mask = torch.logical_and(mask1, mask2)
            curr_mask = np.array(curr_mask.cpu(), dtype=np.bool_)
            all_masks.append(curr_mask)
        return all_masks
    

    def get_level_positives_v5(self, ndx):
        all_masks = []
        query_2_database_dist = torch.cdist(self.UTM_coord_tensor[ndx:ndx+1], self.UTM_coord_tensor).squeeze(0)
        for i in range(len(self.positive_distance_list)):
            if i == 0:
                curr_mask = torch.le(query_2_database_dist, self.positive_distance_list[i])
            else:
                mask1 = torch.gt(query_2_database_dist, self.positive_distance_list[i-1])
                mask2 = torch.le(query_2_database_dist, self.positive_distance_list[i])
                curr_mask = torch.logical_and(mask1, mask2)
            all_masks.append(curr_mask)
        return all_masks
    

    def get_level_positives_v6(self, ndx, interval=100):
        all_masks = []
        if self.dist_caculation_type == 'all_coords_L2':
            query_2_database_dist = torch.cdist(self.UTM_coord_tensor[ndx:ndx+1], self.UTM_coord_tensor).squeeze(0)
        elif self.dist_caculation_type == 'all_coords_L2_mean':
            query_2_database_dist = torch.cdist(self.UTM_coord_tensor[:, ndx:ndx+1, :], self.UTM_coord_tensor).mean(dim=0, keepdim=False).squeeze(0)
        query_2_database_dist_sorted, _ = torch.sort(query_2_database_dist)
        for i in range(10):
            if i == 0:
                curr_positive_distance = query_2_database_dist_sorted[(i+1)*interval]
                curr_mask = torch.le(query_2_database_dist, curr_positive_distance)
            else:
                curr_positive_distance = query_2_database_dist_sorted[(i+1)*interval]
                pre_positive_distance = query_2_database_dist_sorted[i*interval]
                mask1 = torch.gt(query_2_database_dist, pre_positive_distance)
                mask2 = torch.le(query_2_database_dist, curr_positive_distance)
                curr_mask = torch.logical_and(mask1, mask2)
            curr_mask = np.array(curr_mask.cpu(), dtype=np.bool_)
            all_masks.append(curr_mask)
        return all_masks

    def get_level_positives_v7(self, ndx, k=3):
        all_masks = []
        query_2_database_overlap_ratio = torch.tensor(self.area_overlap[ndx, :]).cuda()
        query_2_database_overlap_ratio_sorted, _ = torch.sort(query_2_database_overlap_ratio, descending=True)

        nonzero_overlap_num = torch.count_nonzero(query_2_database_overlap_ratio)

        range_inuse = k
        if nonzero_overlap_num < 50:
            interval_inuse = min(nonzero_overlap_num // k, 10)
        elif nonzero_overlap_num < 100:
            interval_inuse = min(nonzero_overlap_num // k, 20)
        elif nonzero_overlap_num < 150:
            interval_inuse = min(nonzero_overlap_num // k, 30)
        elif nonzero_overlap_num < 200:
            interval_inuse = min(nonzero_overlap_num // k, 40)
        else:
            interval_inuse = min(nonzero_overlap_num // k, 50)
        for i in range(range_inuse):
            if i == 0:
                curr_positive_overlap_ratio = query_2_database_overlap_ratio_sorted[(i+1)*interval_inuse]
                curr_mask = torch.gt(query_2_database_overlap_ratio, curr_positive_overlap_ratio)
            else:
                curr_positive_overlap_ratio = query_2_database_overlap_ratio_sorted[(i+1)*interval_inuse]
                pre_positive_overlap_ratio = query_2_database_overlap_ratio_sorted[i*interval_inuse]
                mask1 = torch.le(query_2_database_overlap_ratio, pre_positive_overlap_ratio)
                mask2 = torch.gt(query_2_database_overlap_ratio, curr_positive_overlap_ratio)
                curr_mask = torch.logical_and(mask1, mask2)
            curr_mask = np.array(curr_mask.cpu(), dtype=np.bool_)
            all_masks.append(curr_mask)
        return all_masks

    def get_level_positives_v8(self, ndx):
        query_2_database_dist = torch.cdist(self.pos_vec_vet_coords_tensor[ndx:ndx+1, :2], self.pos_vec_vet_coords_tensor[:, :2]) # (1, all_train_sample_size)
        query_2_database_pos_candidate_mask = torch.lt(query_2_database_dist.squeeze(0), self.pos_candidate_distance) # (all_train_sample_size, )
        query_vec = self.pos_vec_vet_coords_tensor[ndx:ndx+1, :2] - self.pos_vec_vet_coords_tensor[ndx:ndx+1, 2:] # (1, 2)
        query_vec_normalized = torch.nn.functional.normalize(query_vec, p=2.0, dim=-1) # (1, 2)
        pos_candidate_vec = self.pos_vec_vet_coords_tensor[query_2_database_pos_candidate_mask][:, :2] - self.pos_vec_vet_coords_tensor[query_2_database_pos_candidate_mask][:, 2:] # (all_train_sample_size, 2)
        pos_candidate_vec_normalized = torch.nn.functional.normalize(pos_candidate_vec, p=2.0, dim=-1) # (all_train_sample_size, 2)
        query_2_pos_candidate_cos_sim = torch.sum(query_vec_normalized * pos_candidate_vec_normalized, dim=-1) # (all_train_sample_size, )
        query_2_pos_candidate_rad = torch.acos(query_2_pos_candidate_cos_sim) # (all_train_sample_size, )
        query_2_pos_candidate_deg = torch.rad2deg(query_2_pos_candidate_rad) # (all_train_sample_size, )
        query_2_final_pos_mask = torch.lt(query_2_pos_candidate_deg, self.pos_final_degree)
        query_2_database_pos_candidate_mask_temp = query_2_database_pos_candidate_mask.detach().clone()
        query_2_database_pos_candidate_mask[query_2_database_pos_candidate_mask_temp] = query_2_final_pos_mask
        final_mask = np.array(query_2_database_pos_candidate_mask.cpu(), dtype=np.bool_)
        return final_mask

    def get_sp_positives_1(self, ndxs):
        query_2_query_dist = torch.cdist(self.pos_vec_vet_coords_tensor[ndxs, :2], self.pos_vec_vet_coords_tensor[ndxs, :2]) # (ndxs_num, ndxs_num)
        query_2_query_pos_candidate_mask_1 = torch.lt(query_2_query_dist, self.pos_candidate_distance) # (ndxs_num, ndxs_num)
        query_vec = self.pos_vec_vet_coords_tensor[ndxs, :2] - self.pos_vec_vet_coords_tensor[ndxs, 2:] # (ndxs_num, 2)
        query_vec_normalized = torch.nn.functional.normalize(query_vec, p=2.0, dim=-1) # (ndxs_num, 2)

        query_2_query_cos_sim = torch.sum(query_vec_normalized.unsqueeze(1) * query_vec_normalized.unsqueeze(0), dim=-1) # (ndxs_num, ndxs_num)
        query_2_query_rad = torch.acos(query_2_query_cos_sim) # (ndxs_num, ndxs_num)
        query_2_query_deg = torch.rad2deg(query_2_query_rad) # (ndxs_num, )
        query_2_query_pos_candidate_mask_2 = torch.lt(query_2_query_deg, self.pos_final_degree)
        final_mask = torch.logical_and(query_2_query_pos_candidate_mask_1, query_2_query_pos_candidate_mask_2)
        return final_mask

    def get_level_positives_v9(self, ndx, interval=100):
        all_masks = []
        query_2_database_dist = torch.cdist(self.UTM_coord_tensor[ndx:ndx+1, :2], self.UTM_coord_tensor[:, :2]).squeeze(0)
        query_2_database_sim = torch.exp(-self.exp_scale * query_2_database_dist)
        query_2_database_sim_sorted, _ = torch.sort(query_2_database_sim, descending=True)
        for i in range(10):
            if i == 0:
                curr_positive_sim = query_2_database_sim_sorted[(i+1)*interval]
                curr_mask = torch.ge(query_2_database_sim, curr_positive_sim)
            else:
                curr_positive_sim = query_2_database_sim_sorted[(i+1)*interval]
                pre_positive_sim = query_2_database_sim_sorted[i*interval]
                mask1 = torch.le(query_2_database_sim, pre_positive_sim)
                mask2 = torch.ge(query_2_database_sim, curr_positive_sim)
                curr_mask = torch.logical_and(mask1, mask2)
            curr_mask = np.array(curr_mask.cpu(), dtype=np.bool_)
            all_masks.append(curr_mask)
        return all_masks

    def get_level_positives_v10(self, ndx, interval=100):
        all_masks = []
        query_2_database_dist = torch.cdist(self.UTM_coord_tensor[ndx:ndx+1, 2:], self.UTM_coord_tensor[:, 2:]).squeeze(0)
        query_2_database_sim = torch.exp(-self.exp_scale * query_2_database_dist)
        query_2_database_sim_sorted, _ = torch.sort(query_2_database_sim, descending=True)
        for i in range(10):
            if i == 0:
                curr_positive_sim = query_2_database_sim_sorted[(i+1)*interval]
                curr_mask = torch.gt(query_2_database_sim, curr_positive_sim)
            else:
                curr_positive_sim = query_2_database_sim_sorted[(i+1)*interval]
                pre_positive_sim = query_2_database_sim_sorted[i*interval]
                mask1 = torch.le(query_2_database_sim, pre_positive_sim)
                mask2 = torch.gt(query_2_database_sim, curr_positive_sim)
                curr_mask = torch.logical_and(mask1, mask2)
            curr_mask = np.array(curr_mask.cpu(), dtype=np.bool_)
            all_masks.append(curr_mask)
        return all_masks

    def get_level_positives_v11(self, ndx, interval=100):
        all_masks = []
        if self.dist_caculation_type == 'all_coords_L2':
            query_2_database_dist = torch.cdist(self.UTM_coord_tensor[ndx:ndx+1], self.UTM_coord_tensor).squeeze(0)
        elif self.dist_caculation_type == 'all_coords_L2_mean':
            query_2_database_dist = torch.cdist(self.UTM_coord_tensor[:, ndx:ndx+1, :], self.UTM_coord_tensor).mean(dim=0, keepdim=False).squeeze(0)
        query_2_database_dist_sorted, _ = torch.sort(query_2_database_dist)

        nonzero_overlap_num = torch.count_nonzero(query_2_database_dist < 37.5)

        range_inuse = 3
        if nonzero_overlap_num < 50:
            interval_inuse = min(nonzero_overlap_num // range_inuse, 10)
        elif nonzero_overlap_num < 100:
            interval_inuse = min(nonzero_overlap_num // range_inuse, 20)
        elif nonzero_overlap_num < 150:
            interval_inuse = min(nonzero_overlap_num // range_inuse, 30)
        elif nonzero_overlap_num < 200:
            interval_inuse = min(nonzero_overlap_num // range_inuse, 40)
        else:
            interval_inuse = min(nonzero_overlap_num // range_inuse, 50)


        for i in range(range_inuse):
            if i == 0:
                curr_positive_distance = query_2_database_dist_sorted[(i+1)*interval_inuse]
                curr_mask = torch.le(query_2_database_dist, curr_positive_distance)
            else:
                curr_positive_distance = query_2_database_dist_sorted[(i+1)*interval_inuse]
                pre_positive_distance = query_2_database_dist_sorted[i*interval_inuse]
                mask1 = torch.gt(query_2_database_dist, pre_positive_distance)
                mask2 = torch.le(query_2_database_dist, curr_positive_distance)
                curr_mask = torch.logical_and(mask1, mask2)
            curr_mask = np.array(curr_mask.cpu(), dtype=np.bool_)
            all_masks.append(curr_mask)
        return all_masks

    def get_level_positives_v12(self, ndx, interval=100):
        all_masks = []
        query_2_database_overlap_ratio = torch.tensor(self.area_overlap[ndx, :]).cuda()
        query_2_database_overlap_ratio_sorted, _ = torch.sort(query_2_database_overlap_ratio, descending=True)

        for i in range(10):
            if i == 0:
                curr_positive_overlap_ratio = query_2_database_overlap_ratio_sorted[(i+1)*interval]
                curr_mask = torch.ge(query_2_database_overlap_ratio, curr_positive_overlap_ratio)
            else:
                curr_positive_overlap_ratio = query_2_database_overlap_ratio_sorted[(i+1)*interval]
                pre_positive_overlap_ratio = query_2_database_overlap_ratio_sorted[i*interval]
                mask1 = torch.le(query_2_database_overlap_ratio, pre_positive_overlap_ratio)
                mask2 = torch.ge(query_2_database_overlap_ratio, curr_positive_overlap_ratio)
                curr_mask = torch.logical_and(mask1, mask2)
            curr_mask = np.array(curr_mask.cpu(), dtype=np.bool_)
            all_masks.append(curr_mask)
        return all_masks

    def get_level_positives_v13(self, ndx, k=3):
        all_masks = []
        if self.dist_caculation_type == 'all_coords_L2':
            query_2_database_dist = torch.cdist(self.UTM_coord_tensor[ndx:ndx+1], self.UTM_coord_tensor).squeeze(0)
        elif self.dist_caculation_type == 'all_coords_L2_mean':
            query_2_database_dist = torch.cdist(self.UTM_coord_tensor[:, ndx:ndx+1, :], self.UTM_coord_tensor).mean(dim=0, keepdim=False).squeeze(0)

        for i in range(k):
            if i == 0:
                curr_positive_distance = self.pose_dist_threshold / k * (i+1)
                curr_mask = torch.le(query_2_database_dist, curr_positive_distance)
            else:
                curr_positive_distance = self.pose_dist_threshold / k * (i+1)
                pre_positive_distance = self.pose_dist_threshold / k * i
                mask1 = torch.gt(query_2_database_dist, pre_positive_distance)
                mask2 = torch.le(query_2_database_dist, curr_positive_distance)
                curr_mask = torch.logical_and(mask1, mask2)
            curr_mask = np.array(curr_mask.cpu(), dtype=np.bool_)
            all_masks.append(curr_mask)
        return all_masks
    
    def get_level_positives_v14(self, ndx, k=3):
        all_masks = []
        query_2_database_overlap_ratio = torch.tensor(self.area_overlap[ndx, :]).cuda()

        for i in range(k):
            if i == 0:
                curr_positive_overlap_ratio = 1.0 - 1.0 / k * (i+1)
                curr_mask = torch.gt(query_2_database_overlap_ratio, curr_positive_overlap_ratio)
            else:
                curr_positive_overlap_ratio = 1.0 - 1.0 / k * (i+1)
                pre_positive_overlap_ratio = 1.0 - 1.0 / k * i
                mask1 = torch.le(query_2_database_overlap_ratio, pre_positive_overlap_ratio)
                mask2 = torch.gt(query_2_database_overlap_ratio, curr_positive_overlap_ratio)
                curr_mask = torch.logical_and(mask1, mask2)
            curr_mask = np.array(curr_mask.cpu(), dtype=np.bool_)
            all_masks.append(curr_mask)
        return all_masks
    
    def get_level_positives_v15(self, ndx, k=3):
        all_masks = []
        query_2_database_overlap_ratio = torch.tensor(self.area_overlap[ndx, :]).cuda()

        for i in range(k):
            if i == 0:
                curr_positive_overlap_ratio = 1.0 - 1.0 / k * (i+1)
                curr_mask = torch.gt(query_2_database_overlap_ratio, curr_positive_overlap_ratio)
            else:
                curr_positive_overlap_ratio = 1.0 - 1.0 / k * (i+1)
                pre_positive_overlap_ratio = 1.0 - 1.0 / k * i
                mask1 = torch.le(query_2_database_overlap_ratio, pre_positive_overlap_ratio)
                mask2 = torch.gt(query_2_database_overlap_ratio, curr_positive_overlap_ratio)
                curr_mask = torch.logical_and(mask1, mask2)
            curr_mask = np.array(curr_mask.cpu(), dtype=np.bool_)
            all_masks.append(curr_mask)
        return all_masks

    def get_level_positives_v16(self, ndx, interval=100):
        all_masks = []
        if self.dist_caculation_type == 'all_coords_L2':
            query_2_database_dist = torch.cdist(self.UTM_coord_tensor[ndx:ndx+1], self.UTM_coord_tensor).squeeze(0)
        elif self.dist_caculation_type == 'all_coords_L2_mean':
            query_2_database_dist = torch.cdist(self.UTM_coord_tensor[:, ndx:ndx+1, :], self.UTM_coord_tensor).mean(dim=0, keepdim=False).squeeze(0)
        query_2_database_dist_sorted, _ = torch.sort(query_2_database_dist)
        curr_positive_distance = query_2_database_dist_sorted[interval]
        curr_mask = torch.le(query_2_database_dist, curr_positive_distance)
        curr_mask = np.array(curr_mask.cpu(), dtype=np.bool_)
        for i in range(4):
            all_masks.append(curr_mask)
        return all_masks
    
    def get_level_positives_v17(self, ndx, interval=100):
        all_masks = []
        query_2_database_overlap_ratio = torch.tensor(self.area_overlap[ndx, :]).cuda()
        query_2_database_overlap_ratio_sorted, _ = torch.sort(query_2_database_overlap_ratio, descending=True)

        curr_positive_overlap_ratio = query_2_database_overlap_ratio_sorted[interval]
        curr_mask = torch.ge(query_2_database_overlap_ratio, curr_positive_overlap_ratio)
        curr_mask = np.array(curr_mask.cpu(), dtype=np.bool_)
        for i in range(4):
            all_masks.append(curr_mask)
        return all_masks
    
    def get_level_positives_v18(self, ndxs, interval=100):
        if self.dist_caculation_type == 'all_coords_L2':
            query_2_database_dist = torch.cdist(self.UTM_coord_tensor[ndxs], self.UTM_coord_tensor).squeeze(0) # (ndxs_num, all_train_sample_size)
        elif self.dist_caculation_type == 'all_coords_L2_mean':
            query_2_database_dist = torch.cdist(self.UTM_coord_tensor[:, ndxs, :], self.UTM_coord_tensor).mean(dim=0, keepdim=False).squeeze(0) # (ndxs_num, all_train_sample_size)
        _, ndxs_top_interval_idxs = torch.topk(query_2_database_dist, k=interval, dim=-1, largest=False, sorted=False)
        ndxs_top_interval_idxs = ndxs_top_interval_idxs.reshape(-1).to('cpu').numpy()
        ndxs_top_interval_idxs = np.unique(ndxs_top_interval_idxs)
        return ndxs_top_interval_idxs

class boreas(Dataset):

    def __init__(self, 
                 data_root: str,
                 raw_dir_name: str, 
                 pc_dir_name: str,
                 rendered_dir_name: str,
                 image_dir_name: str,
                 mask_dir_name: str,
                 tool_name: str,
                 query_filename: str,
                 lidar2image_filename: str,
                 image_size: list,
                 mask_size: list,
                 render_size: list, 
                 pc_transform=None, 
                 pc_preprocess=None,
                 image_transform=None,
                 render_transform=None,
                 mask_transform=None,
                 render_view_num=None,
                 use_cloud: bool = False, 
                 use_render: bool = False,
                 use_image: bool = False,
                 use_mask: bool = False,
                 ratio_strategy: str = 'mean',
                 relative_strategy: bool = True,
                 img_neighbor_num: int = 5,):
        self.raw_path = os.path.join(data_root, raw_dir_name)
        self.official_dataset = BoreasDataset_U(self.raw_path)
        if pc_dir_name is not None:
            self.pc_path = os.path.join(data_root, pc_dir_name) # TODO: when needed, change every lidar root_path to this
            self.pc_dir_name = pc_dir_name
        else:
            self.pc_path = self.raw_path
            self.pc_dir_name = raw_dir_name
        if rendered_dir_name is not None:
            self.rendered_path = os.path.join(data_root, rendered_dir_name)
            self.rendered_dir_name = rendered_dir_name
        else:
            self.rendered_path = self.raw_path
            self.rendered_dir_name = raw_dir_name
        if image_dir_name is not None:
            self.image_path = os.path.join(data_root, image_dir_name)
            self.image_dir_name = image_dir_name
        else:
            self.image_path = self.raw_path
            self.image_dir_name = raw_dir_name
        if mask_dir_name is not None:
            self.mask_path = os.path.join(data_root, mask_dir_name)
        else:
            self.mask_path = self.raw_path
        
        self.query_path = os.path.join(self.raw_path, tool_name, query_filename)
        self.queries = pickle.load(open(self.query_path, 'rb'))
        self.traversal_num = len(self.official_dataset.sequences)
        self.traversal_idxs = {}
        for sq in self.official_dataset.sequences:
            self.traversal_idxs[sq.ID] = []
        for i, curr_query in enumerate(self.queries):
            self.traversal_idxs[curr_query['sequence_ID']].append(i)
        for sq_ID, sq_idxs in self.traversal_idxs.items():
            self.traversal_idxs[sq_ID] = sorted(sq_idxs)
            
        self.lidar2image_path = os.path.join(self.raw_path, tool_name, lidar2image_filename)
        self.lidar2image = pickle.load(open(self.lidar2image_path, 'rb'))

        self.image_size = image_size
        self.mask_size = mask_size
        self.render_size = render_size
        self.pc_transform = pc_transform
        self.pc_preprocess = pc_preprocess
        self.image_transform = image_transform
        self.render_transform = render_transform
        self.mask_transform = mask_transform
        self.render_view_num = render_view_num
        self.use_cloud = use_cloud
        self.use_render = use_render
        self.use_image = use_image
        self.use_mask = use_mask
        self.ratio_strategy = ratio_strategy
        self.relative_strategy = relative_strategy
        self.img_neighbor_num = img_neighbor_num

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        result = {}
        result["ndx"] = idx
        curr_query = self.queries[idx]
        curr_seq = self.official_dataset.get_seq_from_ID(curr_query['sequence_ID'])
        curr_lidar_frame = curr_seq.lidar_frames[int(curr_query['idx'])]

        if self.use_cloud:
            if self.pc_dir_name == "Boreas_minuse":
                pc = load_lidar(curr_lidar_frame.path)[:, :3]
            elif self.pc_dir_name == "Boreas_lidar_4096":
                lidar_pre_path = curr_lidar_frame.path
                seq_ID, lidar_dir, pc_file_name = lidar_pre_path.split('/')[-3:]
                lidar_curr_path = os.path.join(self.pc_path, seq_ID, lidar_dir, pc_file_name)
                pc = np.load(lidar_curr_path.split('.')[0] + '.npy')
            elif self.pc_dir_name == "Boreas_minuse_163840_to_4096" or self.pc_dir_name == "Boreas_minuse_40960_to_4096_cliped_fov":
                lidar_pre_path = curr_lidar_frame.path
                seq_ID, lidar_dir, pc_file_name = lidar_pre_path.split('/')[-3:]
                lidar_curr_path_prefix = os.path.join(self.pc_path, seq_ID, lidar_dir, pc_file_name.split('.')[0])
                pc = np.load(lidar_curr_path_prefix + '_1.npy')
                pc_original = np.load(lidar_curr_path_prefix + '_2.npy')
                result['cloud_original'] = pc_original
                original_2_downsampled_indices = np.load(lidar_curr_path_prefix + '_3.npy')
                result['original_2_downsampled_indices'] = original_2_downsampled_indices
                result['cloud_pose_original'] = curr_lidar_frame.pose.astype(np.float32)
            pc_pose = curr_lidar_frame.pose.astype(np.float32)
            pc = self.preprocess_pc(pc)
            pc, P_random, P_remove_mask, P_shuffle_indices = self.pc_transform(pc)
            P_shuffle_indices = np.argsort(P_shuffle_indices)
            if P_random is not None:
                pc_pose = np.dot(pc_pose, np.linalg.inv(P_random))
            result['cloud'] = pc
            result['cloud_pose'] = pc_pose
            result['cloud_remove_mask'] = P_remove_mask
            result['cloud_shuffle_indices'] = P_shuffle_indices
            if self.use_semantic_label:
                pc_semantic_label_path = os.path.join(self.pc_semantic_label_path, seq_ID, lidar_dir, pc_file_name.split('.')[0] + "_semantic_label.npy")
                pc_cluster_label_path = os.path.join(self.pc_semantic_label_path, seq_ID, lidar_dir, pc_file_name.split('.')[0] + "_dbscan_cluster_label.npy")
                result['pc_semantic_label'] = np.load(pc_semantic_label_path) # (4096, )
                result['pc_dbscan_cluster_label'] = np.load(pc_cluster_label_path) # (4096, )

        if self.use_render:
            lidar_pre_path = curr_lidar_frame.path
            seq_ID, lidar_dir, pc_file_name = lidar_pre_path.split('/')[-3:]
            depth_file_name = pc_file_name.split('.')[0] + '.png'
            depth_file_path = os.path.join(self.rendered_path, seq_ID, lidar_dir, depth_file_name)
            render_img = cv2.imread(depth_file_path, cv2.IMREAD_ANYDEPTH)
            render_img = np.expand_dims(render_img.astype(np.float32) / 65535, axis=-1) # (H, W, 1)
            render_img = np.tile(render_img * 255, (1, 1, 3)) # (H, W, 3)
            render_img = self.render_transform(render_img) # (3, H, W)
            result['render_img'] = render_img

        if self.use_image:
            curr_image_idxs = self.lidar2image[curr_query['sequence_ID']][str(curr_query['idx'])]
            curr_image_idx = random.choice(curr_image_idxs[:self.img_neighbor_num])
            curr_image_frame = curr_seq.camera_frames[curr_image_idx]
            image_pose = curr_image_frame.pose.astype(np.float32)
            image_pre_path = curr_image_frame.path
            seq_ID, image_dir, img_file_name = image_pre_path.split('/')[-3:]
            image_curr_path = os.path.join(self.image_path, seq_ID, image_dir, img_file_name)
            image = cv2.imread(image_curr_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.image_dir_name == "Boreas_224x224_image":
                P0_path = os.path.join(self.image_path, seq_ID, 'calib', 'intrinsics.npy')
                curr_P0 = np.load(P0_path).astype(np.float32)
            else:
                curr_P0 = curr_seq.calib.P0.astype(np.float32)
                curr_P0 = curr_P0[:3, :3]
            image, curr_P0_processed = self.image_transform(image, curr_P0)
            result['image'] = image
            result['P0'] = curr_P0_processed
            result['image_pose'] = image_pose
            result['image_path'] = image_curr_path

            if self.use_semantic_label:
                img_semantic_label_path = os.path.join(self.img_semantic_label_path, seq_ID, image_dir, img_file_name.split('.')[0] + "_semantic_label.npy")
                img_cluster_label_path = os.path.join(self.img_semantic_label_path, seq_ID, image_dir, img_file_name.split('.')[0] + "_ccl_cluster_label.npy")
                result['img_semantic_label'] = np.load(img_semantic_label_path) # (224, 224)
                result['img_ccl_cluster_label'] = np.load(img_cluster_label_path) # (224, 224)
            
        if self.use_mask:
            pass

        return result
    
    def preprocess_pc(self, pc):
        if self.pc_preprocess['mode'] == 0:
            return pc
        elif self.pc_preprocess['mode'] == 1:
            return voxel_downsample(pc, self.pc_preprocess['voxel_size'])
        elif self.pc_preprocess['mode'] == 2:
            return FPS_downsample(pc, self.pc_preprocess['num_points'])
    
    def get_positives(self, ndx):
        return self.queries[ndx]['positives']

    def get_non_negatives(self, ndx):
        return self.queries[ndx]['non_negatives']


class boreasEval(boreas):

    def __init__(self,
                 use_semantic_label_when_inference,
                 pc_semantic_label_dir_name,
                 img_semantic_label_dir_name, 
                 use_label_correspondence_table, 
                 **kargs):
        super(boreasEval, self).__init__(**kargs)
        self.traversal_cumsum = []
        self.true_neighbors_matrix = np.identity(len(self.queries), dtype=bool)
        for i in range(len(self.queries)):
            if i == 0:
                pre_traversal = self.queries[i]['sequence_ID']
            if self.queries[i]['sequence_ID'] != pre_traversal:
                pre_traversal = self.queries[i]['sequence_ID']
                self.traversal_cumsum.append(i)
            self.true_neighbors_matrix[i, self.queries[i]['true_neighbors']] = True
        self.traversal_cumsum.append(len(self.queries))
        self.img_neighbor_num = 1
        if use_semantic_label_when_inference:
            self.use_semantic_label = use_semantic_label_when_inference
            self.pc_semantic_label_path = os.path.join(kargs['data_root'], pc_semantic_label_dir_name)
            self.img_semantic_label_path = os.path.join(kargs['data_root'], img_semantic_label_dir_name)
            self.use_label_correspondence_table = use_label_correspondence_table
            if self.use_label_correspondence_table:
                with open(os.path.join(self.raw_path, kargs['tool_name'], 'label_correspondence_table.json'), 'r') as f:
                    self.label_correspondence_table = json.load(f)
            else:
                label_correspondence_table_ndarray = np.arange(0, 19, dtype=np.int32) # (19,) is the number of semantic labels of Cityscapes
                label_correspondence_table_ndarray = np.expand_dims(label_correspondence_table_ndarray, axis=1) # (19, 1)
                label_correspondence_table_ndarray = np.repeat(label_correspondence_table_ndarray, 2, axis=1) # (19, 2)
                self.label_correspondence_table = label_correspondence_table_ndarray.tolist()
        else:
            self.use_semantic_label = False

class boreasEvalv2(boreas):

    def __init__(self, 
                 true_neighbor_dist, 
                 coords_filename, 
                 tool_name,
                 use_semantic_label_when_inference,
                 pc_semantic_label_dir_name,
                 img_semantic_label_dir_name, 
                 use_label_correspondence_table,
                 **kargs):
        super(boreasEvalv2, self).__init__(tool_name=tool_name, **kargs)
        self.test_traversal_cumsum = []
        self.db_traversal_cumsum = []
        self.true_neighbor_dist = true_neighbor_dist
        self.UTM_coord_path = os.path.join(self.raw_path, tool_name, coords_filename)
        UTM_coord = np.load(self.UTM_coord_path)
        self.UTM_coord_tensor = torch.tensor(UTM_coord, device='cpu', dtype=torch.float32) 
        self.test_query_length = 0
        self.test_query_idx_list = []
        for i, curr_query in enumerate(self.queries):
            if curr_query['is_test']:
                self.test_query_length += 1
                self.test_query_idx_list.append(i)
            if i == 0:
                pre_traversal = curr_query['sequence_ID']
            if curr_query['sequence_ID'] != pre_traversal:
                pre_traversal = curr_query['sequence_ID']
                self.db_traversal_cumsum.append(i)
                self.test_traversal_cumsum.append(self.test_query_length)

        self.db_traversal_cumsum.append(len(self.queries))
        self.test_traversal_cumsum.append(self.test_query_length)
        self.img_neighbor_num = 1

        if use_semantic_label_when_inference:
            self.use_semantic_label = use_semantic_label_when_inference
            self.pc_semantic_label_path = os.path.join(kargs['data_root'], pc_semantic_label_dir_name)
            self.img_semantic_label_path = os.path.join(kargs['data_root'], img_semantic_label_dir_name)
            self.use_label_correspondence_table = use_label_correspondence_table
            if self.use_label_correspondence_table:
                with open(os.path.join(self.raw_path, kargs['tool_name'], 'label_correspondence_table.json'), 'r') as f:
                    self.label_correspondence_table = json.load(f)
            else:
                label_correspondence_table_ndarray = np.arange(0, 19, dtype=np.int32) # (19,) is the number of semantic labels of Cityscapes
                label_correspondence_table_ndarray = np.expand_dims(label_correspondence_table_ndarray, axis=1) # (19, 1)
                label_correspondence_table_ndarray = np.repeat(label_correspondence_table_ndarray, 2, axis=1) # (19, 2)
                self.label_correspondence_table = label_correspondence_table_ndarray.tolist()
        else:
            self.use_semantic_label = False