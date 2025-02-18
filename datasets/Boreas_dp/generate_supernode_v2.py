import cuml.cluster
import numpy as np
from mini_boreas import BoreasDataset_U
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import pickle
import torch
import argparse
import sys
sys.path.append("/home/pengjianyi/code_projects/CoarseFromFine/utils")
from config import Config
from logger import get_logger
sys.path.remove("/home/pengjianyi/code_projects/CoarseFromFine/utils")
import torch.nn.functional as F
import torch_scatter
from sklearn.cluster import DBSCAN, HDBSCAN
from mmseg.utils import register_all_modules
from mmseg.apis import inference_model, init_model
import cv2
import skimage.measure as ski
from skimage.segmentation import slic
import albumentations as A
import pypatchworkpp
import cuml
import time

@torch.no_grad()
def my_unique_v2(x, x_num, dim_info):
    """
    when the x is (16, 16, 40960, 3) and the k is 2, the unique_x takes 200 MB GPU memory mostly
    this function means we can select very high resolution corresponding pairs without choosing a small area first
    or compute a very memory_cost matrix
    it's worth noted that the CFI2P choose the high confident proxy first and choose the high confident points for every proxy,
    then choose the high confident patch for every proxy and got every pixel in this patch
    TODO: need to test the time and memory consumed by the torch.unique function
    Args:
        x: tensor of shape (dim1, dim2, ..., dimk, n, m)
        x_num: tensor of shape (dim1, dim2, ..., dimk, n)
        dim_info: list of shape (m)
    Returns:
        unique_x: tensor of shape (huge_num, k+m),
        unique_x_num: tensor of shape (huge_num)
    """
    device = x.device
    x_shape = x.shape
    indices = x_shape[:-2]
    coordinate_list = []
    for i in range(len(indices)):
        coordinate_list.append(torch.arange(0, indices[i], device=device))
    meshgrid_list = torch.meshgrid(*coordinate_list) # Produces (k,) list, the i th element is a (dim1, dim2, ..., dimk) tensor
    x_flattened = x.view(-1, x_shape[-1]) # produce (dim1 * dim2 * ... * dimk * n, m) tensor
    meshgrid_v1_list = []
    for i in range(len(meshgrid_list)):
        meshgrid_v1_list.append(meshgrid_list[i].reshape(-1).repeat_interleave(x_shape[-2]).unsqueeze(-1)) # produce (dim1 * dim2 * ... * dimk * n, 1) tensor

    x_v1 = torch.cat((*meshgrid_v1_list, x_flattened), dim=-1) # produce (dim1 * dim2 * ... * dimk * n, k+m) tensor
    x_num_flattened = x_num.view(-1)
    x_v1_sparse = torch.sparse_coo_tensor(x_v1.transpose(0, 1), x_num_flattened, indices + dim_info) # produce (dim1, dim2, ..., dimk, dim_info1, dim_info2, ..., dim_infom) tensor
    x_v1_sparse = x_v1_sparse.coalesce() # produce (dim1, dim2, ..., dimk, dim_info1, dim_info2, ..., dim_infom) tensor
    unique_x = x_v1_sparse.indices().transpose(0, 1) # produce (huge_num, k+m) tensor
    unique_x_num = x_v1_sparse.values() # produce (huge_num) tensor
    return unique_x, unique_x_num

def parse_args():
    parser = argparse.ArgumentParser(description='generate supernode')
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

def get_connected_components_by_label(points, labels, unique_labels, eps=0.15, min_samples=10):

    pre_label_id_num = 0
    final_label = np.full_like(labels, -1)
    for label in unique_labels:
        # 取出同一语义标签的点
        label_mask = (labels == label)

        points_with_same_label = points[label_mask]
        if points_with_same_label.shape[0] <= min_samples and points_with_same_label.shape[0] > 1:
            final_label[label_mask] = pre_label_id_num
            pre_label_id_num += 1
            continue
        elif points_with_same_label.shape[0] <= 1:
            continue

        if dbscan_type == 'hdbscan':




            # clustering = HDBSCAN(min_cluster_size=min_samples, metric='euclidean').fit(points_with_same_label)
            # cluster_labels = clustering.labels_
            cluster_labels = per_label_4096_hdbscaner.fit_predict(points_with_same_label)



        else:
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points_with_same_label)
            cluster_labels = clustering.labels_
        
        max_label = cluster_labels.max()
        cluster_labels_in_all_mask = cluster_labels != -1
        cluster_labels_in_all_mask = cluster_labels_in_all_mask.astype(cluster_labels.dtype)
        cluster_labels_in_all = cluster_labels + pre_label_id_num * cluster_labels_in_all_mask
        final_label[label_mask] = cluster_labels_in_all
        pre_label_id_num += max_label + 1

    return final_label


# def process_pc_sequence(pc_seq_ID):
#     source_pc_seq_path_1 = os.path.join(source_pc_path_1, pc_seq_ID)
#     source_pc_seq_path_2 = os.path.join(source_pc_path_2, pc_seq_ID)
#     target_pc_seq_path = os.path.join(target_pc_path, pc_seq_ID)
#     os.makedirs(target_pc_seq_path, exist_ok=True)
#     source_pc_seq_lidar_path_1 = os.path.join(source_pc_seq_path_1, "lidar")
#     source_pc_seq_lidar_path_2 = os.path.join(source_pc_seq_path_2, "lidar")
#     target_pc_seq_lidar_path = os.path.join(target_pc_seq_path, "lidar")
#     os.makedirs(target_pc_seq_lidar_path, exist_ok=True)
#     source_pc_seq_lidar_list = os.listdir(source_pc_seq_lidar_path_1)
#     source_pc_seq_lidar_list = [file_name.split('_')[0] for file_name in source_pc_seq_lidar_list]
#     source_pc_seq_lidar_list = sorted(list(set(source_pc_seq_lidar_list)))
#     for curr_lidar_file_name in source_pc_seq_lidar_list:
#         save_path_semantic_label = os.path.join(target_pc_seq_lidar_path, curr_lidar_file_name + "_semantic_label.npy")
#         save_path_dbscan_cluster_label = os.path.join(target_pc_seq_lidar_path, curr_lidar_file_name + "_dbscan_cluster_label.npy")

#         # save_path_all_semantic_label = os.path.join(target_pc_seq_lidar_path, curr_lidar_file_name + "_all_semantic_label.npy")
#         # if os.path.exists(save_path_all_semantic_label):
#         #     print(f"pass the {curr_lidar_file_name} in {pc_seq_ID}")
#         #     continue


#         if os.path.exists(save_path_dbscan_cluster_label):
#             print(f"pass the {curr_lidar_file_name} in {pc_seq_ID}")
#             continue



#         pc_4096_path = os.path.join(source_pc_seq_lidar_path_1, curr_lidar_file_name + "_1.npy")
#         pc_all_reflection_path = os.path.join(source_pc_seq_lidar_path_2, curr_lidar_file_name + "_all.npy")
#         pc_clip_fov_mask_path = os.path.join(source_pc_seq_lidar_path_2, curr_lidar_file_name + "cliped_fov_mask.npy")
#         pc_4096 = np.load(pc_4096_path)
#         pc_all_reflection = np.load(pc_all_reflection_path) # (N, 4) np array
#         pc_clip_fov_mask = np.load(pc_clip_fov_mask_path) # (N,) np array

#         pc_clip_fov_reflection = pc_all_reflection[pc_clip_fov_mask] # (M, 4) np array
#         pc_clip_fov = pc_clip_fov_reflection[:, :3] # (M, 3) np array
#         pc_clip_fov_tensor = torch.from_numpy(pc_clip_fov).float().cuda()
#         pc_4096_tensor = torch.from_numpy(pc_4096).float().cuda()
#         clip_fov_2_4096_dis = torch.cdist(pc_clip_fov_tensor, pc_4096_tensor) # (M, 4096) tensor on device
#         clip_fov_2_4096_idx = torch.argmin(clip_fov_2_4096_dis, dim=1) # (M,) tensor on device

#         num_points_current_frame = pc_all_reflection.shape[0]
#         pc_all_reflection_scaled_ = np.round(pc_all_reflection[:, :3] / pc_voxel_size).astype(np.int32)
#         pc_all_reflection_scaled_pos_ = pc_all_reflection_scaled_ - pc_all_reflection_scaled_.min(0, keepdims=1)
#         feat_all_reflection_ = pc_all_reflection
#         _, inds, inverse_map = sparse_quantize(
#             pc_all_reflection_scaled_pos_,
#             return_index=True,
#             return_inverse=True,
#         )
#         pc_all_reflection_scaled_pos = pc_all_reflection_scaled_pos_[inds]
#         feat_all_reflection = feat_all_reflection_[inds]
#         feat_all_reflection = torch.from_numpy(feat_all_reflection)
#         pc_all_reflection_scaled_pos = torch.from_numpy(pc_all_reflection_scaled_pos)
#         inverse_map = torch.from_numpy(inverse_map).long()
#         pc_all_reflection_scaled_pos_ = torch.from_numpy(pc_all_reflection_scaled_pos_)
#         lidar = SparseTensor(feat_all_reflection, pc_all_reflection_scaled_pos)
#         inverse_map = SparseTensor(inverse_map, pc_all_reflection_scaled_pos_)
#         lidar = lidar.cuda()
#         inverse_map = inverse_map.cuda()
#         ret = {
#             'lidar': lidar,
#             'inverse_map': inverse_map,
#             'num_points': torch.tensor([num_points_current_frame], device=device), # for multi frames
#         }
#         inputs = [ret]
#         batch_inputs = sparse_collate_fn(inputs)
#         with torch.no_grad():
#             batch_output = pc_model(batch_inputs)
#         pc_all_reflection_semantic_label = batch_output['point_predict'][0] # (N,) tensor on device
#         pc_all_reflection_semantic_label_clip_fov = pc_all_reflection_semantic_label[pc_clip_fov_mask] # (M,) tensor on device
#         pc_all_reflection_semantic_one_hot_label = F.one_hot(pc_all_reflection_semantic_label_clip_fov, num_classes=20) # (N, 20) tensor on device
#         pc_4096_semantic_label = torch_scatter.scatter_sum(pc_all_reflection_semantic_one_hot_label, 
#                                                            clip_fov_2_4096_idx.unsqueeze(1).expand(-1, 20),
#                                                            dim=0, 
#                                                            dim_size=pc_4096.shape[0]) # (4096, 20) tensor on device
#         pc_4096_semantic_label = pc_4096_semantic_label.argmax(dim=1) # (4096,) tensor on device
#         pc_4096_semantic_label_np = pc_4096_semantic_label.cpu().numpy()
#         pc_4096_semantic_label_np_filter_mask = np.isin(pc_4096_semantic_label_np, pc_label_to_ignore)
#         pc_4096_semantic_label_np[pc_4096_semantic_label_np_filter_mask] = -1
#         pc_4096_dbscan_cluster_label_np = get_connected_components_by_label(pc_4096, 
#                                                                             pc_4096_semantic_label_np, 
#                                                                             pc_inuse_labels,
#                                                                             eps=pc_dbscan_eps,
#                                                                             min_samples=pc_dbscan_min_samples)
#         pc_4096_semantic_label_np = pc_4096_semantic_label_np.astype(np.int32)
#         pc_4096_dbscan_cluster_label_np = pc_4096_dbscan_cluster_label_np.astype(np.int32)
#         np.save(save_path_semantic_label, pc_4096_semantic_label_np)
#         np.save(save_path_dbscan_cluster_label, pc_4096_dbscan_cluster_label_np)
#         print(f"saved {curr_lidar_file_name} in {pc_seq_ID}")

# def process_img_sequence(img_seq_ID):
#     source_real_img_seq_path = os.path.join(source_real_img_path, img_seq_ID)
#     source_guiding_img_seq_path = os.path.join(source_guiding_img_path, img_seq_ID)
#     target_img_seq_path = os.path.join(target_img_path, img_seq_ID)
#     os.makedirs(target_img_seq_path, exist_ok=True)
#     source_real_img_seq_lidar_path = os.path.join(source_real_img_seq_path, "camera")
#     source_guiding_img_seq_lidar_path = os.path.join(source_guiding_img_seq_path, "camera")
#     target_img_seq_lidar_path = os.path.join(target_img_seq_path, "camera")
#     os.makedirs(target_img_seq_lidar_path, exist_ok=True)
#     source_guiding_img_seq_lidar_list = sorted(os.listdir(source_guiding_img_seq_lidar_path))
#     source_guiding_img_seq_lidar_list = [file_name.split('.')[0] for file_name in source_guiding_img_seq_lidar_list]
#     for curr_img_file_name in source_guiding_img_seq_lidar_list:
#         save_path_semantic_label = os.path.join(target_img_seq_lidar_path, curr_img_file_name + "_semantic_label.npy")
#         save_path_ccl_cluster_label = os.path.join(target_img_seq_lidar_path, curr_img_file_name + "_ccl_cluster_label.npy")
#         if os.path.exists(save_path_ccl_cluster_label):
#             print(f"pass the {curr_img_file_name} in {img_seq_ID}")
#             continue
#         img_path = os.path.join(source_real_img_seq_lidar_path, curr_img_file_name + ".png")
#         img = cv2.imread(img_path)
#         result = inference_model(img_model, img)
#         img_semantic_label_original = np.asarray(result.pred_sem_seg.data.detach().clone().cpu()).squeeze(0) # (original_H, original_W) np array
#         img_semantic_label = img_transform_later(image=img_semantic_label_original)['image']
#         img_label_to_ignore_mask = np.isin(img_semantic_label, img_label_to_ignore)
#         img_semantic_label[img_label_to_ignore_mask] = -1
#         img_ccl_cluster_label = ski.label(img_semantic_label, background=-1, connectivity=2)  # (H, W) np array
#         img_ccl_cluster_label = img_ccl_cluster_label.astype(np.int32) - 1
#         img_semantic_label = img_semantic_label.astype(np.int32)
#         np.save(save_path_semantic_label, img_semantic_label)
#         np.save(save_path_ccl_cluster_label, img_ccl_cluster_label)
#         print(f"saved {curr_img_file_name} in {img_seq_ID}")

@torch.no_grad()
def process_all_modalities_sequence(sequence):
    seq_ID = sequence.ID

    source_img_seq_path = os.path.join(source_img_path, seq_ID)
    target_img_seq_path = os.path.join(target_img_path, seq_ID)
    os.makedirs(target_img_seq_path, exist_ok=True)
    source_img_seq_camera_path = os.path.join(source_img_seq_path, "camera")
    target_img_seq_camera_path = os.path.join(target_img_seq_path, "camera")
    os.makedirs(target_img_seq_camera_path, exist_ok=True)

    source_pc_seq_path_1 = os.path.join(source_pc_path_1, seq_ID)
    source_pc_seq_path_2 = os.path.join(source_pc_path_2, seq_ID)
    target_pc_seq_path = os.path.join(target_pc_path, seq_ID)
    os.makedirs(target_pc_seq_path, exist_ok=True)
    source_pc_seq_lidar_path_1 = os.path.join(source_pc_seq_path_1, "lidar")
    source_pc_seq_lidar_path_2 = os.path.join(source_pc_seq_path_2, "lidar")
    target_pc_seq_lidar_path = os.path.join(target_pc_seq_path, "lidar")
    os.makedirs(target_pc_seq_lidar_path, exist_ok=True)

    minuse_lidar_idxs_seq = minuse_lidar_idxs[seq_ID]
    lidar2image_idx_seq = lidar2image_idx[seq_ID]
    for lidar_id in minuse_lidar_idxs_seq:

        t0 = time.perf_counter()

        curr_lidar_frame = sequence.lidar_frames[lidar_id]
        curr_lidar_file_name = curr_lidar_frame.path.split('/')[-1].split('.')[0]

        image_id = lidar2image_idx_seq[str(lidar_id)][0]
        curr_image_frame = sequence.camera_frames[image_id]
        curr_image_file_name = curr_image_frame.path.split('/')[-1].split('.')[0]

        save_path_pc_semantic_label = os.path.join(target_pc_seq_lidar_path, curr_lidar_file_name + "_semantic_label.npy")
        save_path_dbscan_cluster_label = os.path.join(target_pc_seq_lidar_path, curr_lidar_file_name + "_dbscan_cluster_label.npy")
        save_path_img_semantic_label = os.path.join(target_img_seq_camera_path, curr_image_file_name + "_semantic_label.npy")
        save_path_ccl_cluster_label = os.path.join(target_img_seq_camera_path, curr_image_file_name + "_ccl_cluster_label.npy")
        if os.path.exists(save_path_ccl_cluster_label):
            print(f"pass the lidar_id: {lidar_id} in {seq_ID}")
            continue

        img_path = os.path.join(source_img_seq_camera_path, curr_image_file_name + ".png")
        img = cv2.imread(img_path)
        result = inference_model(img_model, img)
        img_semantic_label_original = np.asarray(result.pred_sem_seg.data.detach().clone().cpu()).squeeze(0) # (original_H, original_W) np array
        img_label_original_to_ignore_mask = np.isin(img_semantic_label_original, img_label_to_ignore)
        img_semantic_label_original[img_label_original_to_ignore_mask] = -1
        img_semantic_label = img_transform_later(image=img_semantic_label_original)['image']

        pc_4096_path = os.path.join(source_pc_seq_lidar_path_1, curr_lidar_file_name + "_1.npy")
        pc_all_reflection_path = os.path.join(source_pc_seq_lidar_path_2, curr_lidar_file_name + "_all.npy")
        # pc_clip_fov_mask_path = os.path.join(source_pc_seq_lidar_path_2, curr_lidar_file_name + "cliped_fov_mask.npy")
        pc_4096 = np.load(pc_4096_path)
        pc_all_reflection = np.load(pc_all_reflection_path) # (N, 4) np array
        pc_all = pc_all_reflection[:, :3] # (N, 3) np array
        # pc_clip_fov_mask = np.load(pc_clip_fov_mask_path) # (N,) np array

        PatchworkPLUSPLUS.estimateGround(pc_all_reflection)
        # pc_only_ground = PatchworkPLUSPLUS.getGround() # (M, 3) np array
        pc_only_ground_indices = PatchworkPLUSPLUS.getGroundIndices() # (M,) np array
        pc_without_ground = PatchworkPLUSPLUS.getNonground() # (N - M, 3) np array
        pc_without_ground_indices = PatchworkPLUSPLUS.getNongroundIndices() # (N - M,) np array



        # without_ground_clustering = HDBSCAN(min_cluster_size=pc_dbscan_min_samples, metric='euclidean').fit(pc_without_ground) 
        # without_ground_cluster_labels = without_ground_clustering.labels_ # (N - M,) np array


        without_ground_cluster_labels = nonground_hdbscaner.fit_predict(pc_without_ground)




        without_ground_cluster_labels_num = without_ground_cluster_labels.max() + 1

        pc_pose = curr_lidar_frame.pose.astype(np.float32)
        curr_P0 = sequence.calib.P0.astype(np.float32)
        curr_P0 = curr_P0[:3, :3]
        image_pose = curr_image_frame.pose.astype(np.float32)
        P_camera_lidar = np.linalg.inv(image_pose) @ pc_pose # (4, 4)
        pc_all_to_mult = np.concatenate([pc_all, np.ones((pc_all.shape[0], 1))], axis=1) # (N, 4)
        pc_all_in_camera = pc_all_to_mult @ P_camera_lidar.T # (N, 4)
        pc_all_in_camera = pc_all_in_camera[..., :3]
        mask_1 = pc_all_in_camera[:, 2] >= 0.0
        pc_all_in_image = pc_all_in_camera @ curr_P0.T # (N, 3)
        pc_all_in_plane = pc_all_in_image[:, :2] / pc_all_in_image[:, 2:] # (N, 2)
        mask_2 = (pc_all_in_plane[:, 0] >= 0.0) & (pc_all_in_plane[:, 0] < 2448.0) & (pc_all_in_plane[:, 1] >= 0.0) & (pc_all_in_plane[:, 1] < 2048.0)
        pc_all_in_plane_pixels = np.floor(pc_all_in_plane).astype(np.float32)
        pc_all_in_plane_pixels[..., 0] = np.clip(pc_all_in_plane_pixels[..., 0], 0.0, 2448.0 - 1.0)
        pc_all_in_plane_pixels[..., 1] = np.clip(pc_all_in_plane_pixels[..., 1], 0.0, 2048.0 - 1.0)
        mask_12 = mask_1 & mask_2 # (N,) np array
        pc_without_ground_mask12 = mask_12[pc_without_ground_indices] # (N - M,) np array
        


        # let the ground points have just the same label as the image semantic results
        pc_only_ground_in_plane_pixels = pc_all_in_plane_pixels[pc_only_ground_indices]
        pc_only_ground_semantic_label = img_semantic_label_original[pc_only_ground_in_plane_pixels[:, 1].astype(np.int32), pc_only_ground_in_plane_pixels[:, 0].astype(np.int32)] # (M,) np array
        pc_only_ground_mask12 = mask_12[pc_only_ground_indices] # (M,) np array
        pc_only_ground_semantic_label[~pc_only_ground_mask12] = -1

        # pc_only_ground_dbscan_min_samples = 10 #TODO: to adjust the min_samples
        # pc_only_ground_dbscan_eps = 0.15
        # pc_only_ground_cluster_label = get_connected_components_by_label(pc_only_ground, pc_only_ground_semantic_label, np.arange(-1, 20), eps=pc_only_ground_dbscan_eps, min_samples=pc_only_ground_dbscan_min_samples)


        pc_without_ground_in_plane_pixels = pc_all_in_plane_pixels[pc_without_ground_indices] # (N - M,) np array
        pc_without_ground_semantic_label = img_semantic_label_original[pc_without_ground_in_plane_pixels[:, 1].astype(np.int32), pc_without_ground_in_plane_pixels[:, 0].astype(np.int32)] # (N - M,) np array
        pc_without_ground_semantic_label_plus = pc_without_ground_semantic_label + 1
        without_ground_cluster_labels_plus = without_ground_cluster_labels + 1


        without_ground_cluster_labels_mask = without_ground_cluster_labels == -1 # (N - M,) np array

        pc_without_ground_semantic_label_plus_in_image = pc_without_ground_semantic_label_plus[pc_without_ground_mask12] # (num1,) tensor
        without_ground_cluster_labels_plus_in_image = without_ground_cluster_labels_plus[pc_without_ground_mask12] # (num1,) tensor
        pc_without_ground_semantic_label_plus_in_image_tensor = torch.from_numpy(pc_without_ground_semantic_label_plus_in_image).cuda().long() # (num1,) tensor
        without_ground_cluster_labels_plus_in_image_tensor = torch.from_numpy(without_ground_cluster_labels_plus_in_image).cuda().long()
        indices = torch.stack((pc_without_ground_semantic_label_plus_in_image_tensor, without_ground_cluster_labels_plus_in_image_tensor), dim=0) # (2, num1) tensor
        values = torch.ones_like(indices[0, :]) # (num1,) tensor
        without_ground_semantic_to_cluster_matrix_sparse = torch.sparse_coo_tensor(indices, values, (19 + 1, without_ground_cluster_labels_num + 1)).coalesce()
        without_ground_semantic_to_cluster_matrix = without_ground_semantic_to_cluster_matrix_sparse.to_dense()
        without_ground_semantic_label_per_cluster = torch.argmax(without_ground_semantic_to_cluster_matrix, dim=0) # (without_ground_cluster_labels_num + 1,) tensor
        without_ground_cluster_labels_plus_tensor = torch.from_numpy(without_ground_cluster_labels_plus).cuda().long()
        pc_without_ground_semantic_label = torch.gather(input=without_ground_semantic_label_per_cluster,
                                                        dim=0, 
                                                        index=without_ground_cluster_labels_plus_tensor) # (N - M,) tensor
        pc_without_ground_semantic_label -= 1
        pc_without_ground_semantic_label = pc_without_ground_semantic_label.cpu().numpy()
        pc_without_ground_semantic_label[without_ground_cluster_labels_mask] = -1




        pc_all_semantic_label = np.full_like(pc_all[:, 0], -1) # (N, ) np array
        pc_all_semantic_label[pc_only_ground_indices] = pc_only_ground_semantic_label
        pc_all_semantic_label[pc_without_ground_indices] = pc_without_ground_semantic_label
        pc_all_semantic_label[~mask_12] = -1
        pc_all_semantic_labels_plus = pc_all_semantic_label + 1

        pc_clip_fov_tensor = torch.from_numpy(pc_all[mask_12]).cuda() # (M2, 3) tensor
        pc_4096_tensor = torch.from_numpy(pc_4096).cuda() # (4096, 3) tensor
        clip_fov_2_4096_dis = torch.cdist(pc_clip_fov_tensor, pc_4096_tensor) # (M2, 4096) tensor on device
        clip_fov_2_4096_idx = torch.argmin(clip_fov_2_4096_dis, dim=1) # (M2,) tensor on device
        pc_clip_fov_semantic_label_plus = pc_all_semantic_labels_plus[mask_12] # (M2,) np array
        pc_clip_fov_semantic_label_plus_tensor = torch.from_numpy(pc_clip_fov_semantic_label_plus).cuda().long() # (M2,) tensor on device
        pc_clip_fov_semantic_plus_one_hot_label = F.one_hot(pc_clip_fov_semantic_label_plus_tensor, num_classes=19 + 1) # (M2, 20) tensor on device
        pc_4096_semantic_label_plus = torch_scatter.scatter_sum(pc_clip_fov_semantic_plus_one_hot_label, 
                                                           clip_fov_2_4096_idx.unsqueeze(1).expand(-1, 20),
                                                           dim=0, 
                                                           dim_size=pc_4096.shape[0]) # (4096, 20) tensor on device
        pc_4096_semantic_label_plus = pc_4096_semantic_label_plus.argmax(dim=1) # (4096,) tensor on device
        pc_4096_semantic_label = pc_4096_semantic_label_plus - 1
        pc_4096_semantic_label = pc_4096_semantic_label.cpu().numpy()
        pc_4096_cluster_label = get_connected_components_by_label(pc_4096, pc_4096_semantic_label, np.arange(0, 20), eps=pc_4096_dbscan_eps, min_samples=pc_4096_dbscan_min_samples) # (4096,) np array

        # mask_3 = (pc_all_in_plane[:, 0] >= 0.0 - width_left) & (pc_all_in_plane[:, 0] < 2448.0 + width_right) & (pc_all_in_plane[:, 1] >= 0.0 - height_left) & (pc_all_in_plane[:, 1] < 2048.0 + height_right)
        # mask_13 = mask_1 & mask_3 # (N,) np array

        # pc_only_ground_cluster_label_mask = pc_only_ground_cluster_label != -1 # (M,) np array
        # pc_only_ground_cluster_label_mask = pc_only_ground_cluster_label_mask.astype(np.int32) # (M,) np array
        # pc_only_ground_cluster_label_offset = pc_only_ground_cluster_label + without_ground_cluster_labels_num * pc_only_ground_cluster_label_mask  # (M,) np array
        # pc_all_cluster_label = np.full_like(pc_all[:, 0], -1) # (N,) np array
        # pc_all_cluster_label[pc_only_ground_indices] = pc_only_ground_cluster_label_offset
        # pc_all_cluster_label[pc_without_ground_indices] = without_ground_cluster_labels

        # pc_clip_plus_fov_tensor = torch.from_numpy(pc_all[mask_13]).cuda() # (M2, 3) tensor
        # pc_4096_tensor = torch.from_numpy(pc_4096).cuda() # (4096, 3) tensor
        # clip_plus_fov_2_4096_dis = torch.cdist(pc_clip_plus_fov_tensor, pc_4096_tensor) # (M2, 4096) tensor on device
        # clip_plus_fov_2_4096_idx = torch.argmin(clip_plus_fov_2_4096_dis, dim=1) # (M2,) tensor on device
        # pc_clip_plus_fov_cluster_label = pc_all_cluster_labels_plus[mask_13] # (M2,) np array
        # pc_clip_plus_fov_cluster_label_tensor = torch.from_numpy(pc_clip_plus_fov_cluster_label).cuda().long() # (M2,) tensor on device
        # pc_clip_plus_fov_cluster_one_hot_label = F.one_hot(pc_clip_plus_fov_cluster_label_tensor, num_classes=cluster_labels_num + 1) # (M2, cluster_labels_num + 1) tensor on device
        # pc_4096_cluster_label = torch_scatter.scatter_sum(pc_clip_plus_fov_cluster_one_hot_label, 
        #                                                    clip_plus_fov_2_4096_idx.unsqueeze(1).expand(-1, 20),
        #                                                    dim=0, 
        #                                                    dim_size=pc_4096.shape[0]) # (4096, 20) tensor on device
        # pc_4096_cluster_label = pc_4096_cluster_label.argmax(dim=1) # (4096,) tensor on device

        img_ccl_cluster_label = ski.label(img_semantic_label, background=-1, connectivity=2)  # (H, W) np array
        img_ccl_cluster_label = img_ccl_cluster_label.astype(np.int32) - 1
        img_semantic_label = img_semantic_label.astype(np.int32)

        np.save(save_path_pc_semantic_label, pc_4096_semantic_label)
        np.save(save_path_dbscan_cluster_label, pc_4096_cluster_label)
        np.save(save_path_img_semantic_label, img_semantic_label)
        np.save(save_path_ccl_cluster_label, img_ccl_cluster_label)
        t1 = time.perf_counter()
        print(f"saved {curr_image_file_name} in {seq_ID}   "
              f"cost time: {t1 - t0:.2f} s")


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(f"cuda:{args.local_rank}")
    torch.cuda.set_device(device)
    dataset_root = args.data_path
    source_pc_name_1 = "Boreas_minuse_40960_to_4096_cliped_fov"
    source_pc_path_1 = os.path.join(dataset_root, source_pc_name_1)
    source_pc_name_2 = "Boreas_minuse_cliped_fov_and_reflection"
    source_pc_path_2 = os.path.join(dataset_root, source_pc_name_2)
    target_pc_name = "Boreas_minuse_4096_cliped_fov_superpoint_mask"
    target_pc_path = os.path.join(dataset_root, target_pc_name)
    os.makedirs(target_pc_path, exist_ok=True)
    source_img_name = "Boreas_minuse"
    source_img_path = os.path.join(dataset_root, source_img_name)
    target_img_name = "Boreas_224x224_image_superpixel_mask"
    target_img_path = os.path.join(dataset_root, target_img_name)
    os.makedirs(target_img_path, exist_ok=True)

    tool_root_name = 'Boreas_minuse'
    tool_name = 'my_tool'
    minuse_lidar_name = 'minuse_lidar_idxs.pickle'
    minuse_lidar_path = os.path.join(dataset_root, tool_root_name, tool_name, minuse_lidar_name)
    minuse_lidar_idxs = pickle.load(open(minuse_lidar_path, 'rb'))
    lidar2image_name = 'lidar2image.pickle'
    lidar2image_path = os.path.join(dataset_root, tool_root_name, tool_name, lidar2image_name)
    lidar2image_idx = pickle.load(open(lidar2image_path, 'rb'))
    dataset = BoreasDataset_U(os.path.join(dataset_root, tool_root_name))
    
    params = pypatchworkpp.Parameters()
    params.sensor_height = 2.05
    params.verbose = False
    PatchworkPLUSPLUS = pypatchworkpp.patchworkpp(params)

    width_left = int(2448 * 0.1)
    width_right = int(2448 * 0.1)
    height_left = int(2048 * 0.1)
    height_right = int(2048 * 0.1)

    #1、handle the point cloud
    # pc_label_to_ignore = np.arange(1, 9)
    # pc_total_labels = np.arange(20)
    # pc_inuse_labels = np.setdiff1d(pc_total_labels, pc_label_to_ignore)
    dbscan_type = 'hdbscan'
    # pc_dbscan_min_samples = 10 # stable value 1
    # pc_dbscan_min_samples = 5 # stable value 2
    pc_dbscan_min_samples = 15 # stable value 3 
    pc_4096_dbscan_min_samples = 5 #TODO: to adjust the min_samples
    pc_4096_dbscan_eps = 0.15 #TODO: to adjust the min_samples




    nonground_hdbscaner = cuml.cluster.HDBSCAN(min_samples=pc_dbscan_min_samples)
    per_label_4096_hdbscaner = cuml.cluster.HDBSCAN(min_samples=pc_4096_dbscan_min_samples)



    # pc_model_cfgs = Config.fromfile('/home/pengjianyi/code_projects/CoarseFromFine/datasets/Boreas_dp/generate_supernode_dp/minkunet_cfgs.py')
    # pc_model_cfgs = pc_model_cfgs.model_cfgs
    # pc_model = MinkUNet(pc_model_cfgs, num_class=20)
    # pc_voxel_size = 0.05
    # pc_model.cuda()
    # pc_logger = get_logger('pclogger', None)
    # pc_model.load_params_from_file(
    #             filename='/home/pengjianyi/.cache/torch/hub/checkpoints/openpcseg_semkitti_minkunet_mk34_cr16_checkpoint_epoch_36.pth',
    #             to_cpu=True,
    #             logger=pc_logger
    #         )
    # pc_model.eval()
    # pc_sequence_list = os.listdir(source_pc_path_1)
    # pc_sequence_list = pc_sequence_list[args.part_num * 4 : (args.part_num + 1) * 4]
    # for pc_seq_ID in pc_sequence_list:
    #     process_pc_sequence(pc_seq_ID)

    #2、handle the image information
    img_label_to_ignore = np.arange(10, 19)
    register_all_modules() # register all modules for mmseg
    img_model_cfgs_path = '/home/pengjianyi/code_projects/CoarseFromFine/datasets/Boreas_dp/generate_supernode_dp/mask2former_cfgs.py'
    img_model = init_model(img_model_cfgs_path, '/home/pengjianyi/.cache/torch/hub/checkpoints/mask2former.pth', device=device)
    crop_location=dict(
        x_min=0,
        x_max=2448,
        y_min=683,
        y_max=1366,
    )
    image_size = [224, 224]
    t = [A.Crop(x_min=crop_location['x_min'],
                x_max=crop_location['x_max'],
                y_min=crop_location['y_min'],
                y_max=crop_location['y_max'],
                always_apply=True),
        A.Resize(image_size[0], 
                 image_size[1], 
                 cv2.INTER_NEAREST)
                ]
    img_transform_later = A.Compose(t)
    sequence_ID_list = sorted(os.listdir(source_pc_path_1))

    # pass_flag = False
    curr_sequence_ID_list = sequence_ID_list[args.part_num * 4 : (args.part_num + 1) * 4]
    for sequence_ID_inuse in curr_sequence_ID_list:
        curr_seq = dataset.get_seq_from_ID(sequence_ID_inuse)
        # if not pass_flag:
        #     pass_flag = True
        #     continue
        process_all_modalities_sequence(curr_seq)
    # img_sequence_list = os.listdir(source_guiding_img_path)
    # img_sequence_list = img_sequence_list[args.part_num * 4 : (args.part_num + 1) * 4]
    # for img_seq_ID in img_sequence_list:
    #     process_img_sequence(img_seq_ID)
    
    # TODO: if both the mask is not good, consider to use SAM model and the better call SAL to generate better masks