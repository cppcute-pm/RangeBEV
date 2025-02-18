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
from generate_supernode_dp import MinkUNet
import sys
sys.path.append("/home/pengjianyi/code_projects/CoarseFromFine/utils")
from config import Config
from logger import get_logger
sys.path.remove("/home/pengjianyi/code_projects/CoarseFromFine/utils")
from torchsparse.utils.quantize import sparse_quantize
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn
import torch.nn.functional as F
import torch_scatter
from sklearn.cluster import DBSCAN, HDBSCAN
from mmseg.utils import register_all_modules
from mmseg.apis import inference_model, init_model
import cv2
import skimage.measure as ski
from skimage.segmentation import slic
import albumentations as A


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
        if points_with_same_label.shape[0] <= 1:
            continue

        # 使用DBSCAN进行连通性分析
        # TODO: every label may need a unique dbscan parameter?
        if dbscan_type == 'hdbscan':
            clustering = HDBSCAN(min_cluster_size=min_samples, metric='euclidean').fit(points_with_same_label)
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


def process_pc_sequence(pc_seq_ID):
    source_pc_seq_path_1 = os.path.join(source_pc_path_1, pc_seq_ID)
    source_pc_seq_path_2 = os.path.join(source_pc_path_2, pc_seq_ID)
    target_pc_seq_path = os.path.join(target_pc_path, pc_seq_ID)
    os.makedirs(target_pc_seq_path, exist_ok=True)
    source_pc_seq_lidar_path_1 = os.path.join(source_pc_seq_path_1, "lidar")
    source_pc_seq_lidar_path_2 = os.path.join(source_pc_seq_path_2, "lidar")
    target_pc_seq_lidar_path = os.path.join(target_pc_seq_path, "lidar")
    os.makedirs(target_pc_seq_lidar_path, exist_ok=True)
    source_pc_seq_lidar_list = os.listdir(source_pc_seq_lidar_path_1)
    source_pc_seq_lidar_list = [file_name.split('_')[0] for file_name in source_pc_seq_lidar_list]
    source_pc_seq_lidar_list = sorted(list(set(source_pc_seq_lidar_list)))
    for curr_lidar_file_name in source_pc_seq_lidar_list:
        save_path_semantic_label = os.path.join(target_pc_seq_lidar_path, curr_lidar_file_name + "_semantic_label.npy")
        save_path_dbscan_cluster_label = os.path.join(target_pc_seq_lidar_path, curr_lidar_file_name + "_dbscan_cluster_label.npy")

        # save_path_all_semantic_label = os.path.join(target_pc_seq_lidar_path, curr_lidar_file_name + "_all_semantic_label.npy")
        # if os.path.exists(save_path_all_semantic_label):
        #     print(f"pass the {curr_lidar_file_name} in {pc_seq_ID}")
        #     continue


        if os.path.exists(save_path_dbscan_cluster_label):
            print(f"pass the {curr_lidar_file_name} in {pc_seq_ID}")
            continue



        pc_4096_path = os.path.join(source_pc_seq_lidar_path_1, curr_lidar_file_name + "_1.npy")
        pc_all_reflection_path = os.path.join(source_pc_seq_lidar_path_2, curr_lidar_file_name + "_all.npy")
        pc_clip_fov_mask_path = os.path.join(source_pc_seq_lidar_path_2, curr_lidar_file_name + "cliped_fov_mask.npy")
        pc_4096 = np.load(pc_4096_path)
        pc_all_reflection = np.load(pc_all_reflection_path) # (N, 4) np array
        pc_clip_fov_mask = np.load(pc_clip_fov_mask_path) # (N,) np array

        pc_clip_fov_reflection = pc_all_reflection[pc_clip_fov_mask] # (M, 4) np array
        pc_clip_fov = pc_clip_fov_reflection[:, :3] # (M, 3) np array
        pc_clip_fov_tensor = torch.from_numpy(pc_clip_fov).float().cuda()
        pc_4096_tensor = torch.from_numpy(pc_4096).float().cuda()
        clip_fov_2_4096_dis = torch.cdist(pc_clip_fov_tensor, pc_4096_tensor) # (M, 4096) tensor on device
        clip_fov_2_4096_idx = torch.argmin(clip_fov_2_4096_dis, dim=1) # (M,) tensor on device

        num_points_current_frame = pc_all_reflection.shape[0]
        pc_all_reflection_scaled_ = np.round(pc_all_reflection[:, :3] / pc_voxel_size).astype(np.int32)
        pc_all_reflection_scaled_pos_ = pc_all_reflection_scaled_ - pc_all_reflection_scaled_.min(0, keepdims=1)
        feat_all_reflection_ = pc_all_reflection
        _, inds, inverse_map = sparse_quantize(
            pc_all_reflection_scaled_pos_,
            return_index=True,
            return_inverse=True,
        )
        pc_all_reflection_scaled_pos = pc_all_reflection_scaled_pos_[inds]
        feat_all_reflection = feat_all_reflection_[inds]
        feat_all_reflection = torch.from_numpy(feat_all_reflection)
        pc_all_reflection_scaled_pos = torch.from_numpy(pc_all_reflection_scaled_pos)
        inverse_map = torch.from_numpy(inverse_map).long()
        pc_all_reflection_scaled_pos_ = torch.from_numpy(pc_all_reflection_scaled_pos_)
        lidar = SparseTensor(feat_all_reflection, pc_all_reflection_scaled_pos)
        inverse_map = SparseTensor(inverse_map, pc_all_reflection_scaled_pos_)
        lidar = lidar.cuda()
        inverse_map = inverse_map.cuda()
        ret = {
            'lidar': lidar,
            'inverse_map': inverse_map,
            'num_points': torch.tensor([num_points_current_frame], device=device), # for multi frames
        }
        inputs = [ret]
        batch_inputs = sparse_collate_fn(inputs)
        with torch.no_grad():
            batch_output = pc_model(batch_inputs)
        pc_all_reflection_semantic_label = batch_output['point_predict'][0] # (N,) tensor on device
        pc_all_reflection_semantic_label_clip_fov = pc_all_reflection_semantic_label[pc_clip_fov_mask] # (M,) tensor on device
        pc_all_reflection_semantic_one_hot_label = F.one_hot(pc_all_reflection_semantic_label_clip_fov, num_classes=20) # (N, 20) tensor on device
        pc_4096_semantic_label = torch_scatter.scatter_sum(pc_all_reflection_semantic_one_hot_label, 
                                                           clip_fov_2_4096_idx.unsqueeze(1).expand(-1, 20),
                                                           dim=0, 
                                                           dim_size=pc_4096.shape[0]) # (4096, 20) tensor on device
        pc_4096_semantic_label = pc_4096_semantic_label.argmax(dim=1) # (4096,) tensor on device
        pc_4096_semantic_label_np = pc_4096_semantic_label.cpu().numpy()
        pc_4096_semantic_label_np_filter_mask = np.isin(pc_4096_semantic_label_np, pc_label_to_ignore)
        pc_4096_semantic_label_np[pc_4096_semantic_label_np_filter_mask] = -1
        pc_4096_dbscan_cluster_label_np = get_connected_components_by_label(pc_4096, 
                                                                            pc_4096_semantic_label_np, 
                                                                            pc_inuse_labels,
                                                                            eps=pc_dbscan_eps,
                                                                            min_samples=pc_dbscan_min_samples)
        pc_4096_semantic_label_np = pc_4096_semantic_label_np.astype(np.int32)
        pc_4096_dbscan_cluster_label_np = pc_4096_dbscan_cluster_label_np.astype(np.int32)
        np.save(save_path_semantic_label, pc_4096_semantic_label_np)
        np.save(save_path_dbscan_cluster_label, pc_4096_dbscan_cluster_label_np)
        print(f"saved {curr_lidar_file_name} in {pc_seq_ID}")

def process_img_sequence(img_seq_ID):
    source_real_img_seq_path = os.path.join(source_real_img_path, img_seq_ID)
    source_guiding_img_seq_path = os.path.join(source_guiding_img_path, img_seq_ID)
    target_img_seq_path = os.path.join(target_img_path, img_seq_ID)
    os.makedirs(target_img_seq_path, exist_ok=True)
    source_real_img_seq_lidar_path = os.path.join(source_real_img_seq_path, "camera")
    source_guiding_img_seq_lidar_path = os.path.join(source_guiding_img_seq_path, "camera")
    target_img_seq_lidar_path = os.path.join(target_img_seq_path, "camera")
    os.makedirs(target_img_seq_lidar_path, exist_ok=True)
    source_guiding_img_seq_lidar_list = sorted(os.listdir(source_guiding_img_seq_lidar_path))
    source_guiding_img_seq_lidar_list = [file_name.split('.')[0] for file_name in source_guiding_img_seq_lidar_list]
    for curr_img_file_name in source_guiding_img_seq_lidar_list:
        save_path_semantic_label = os.path.join(target_img_seq_lidar_path, curr_img_file_name + "_semantic_label.npy")
        save_path_ccl_cluster_label = os.path.join(target_img_seq_lidar_path, curr_img_file_name + "_ccl_cluster_label.npy")
        if os.path.exists(save_path_ccl_cluster_label):
            print(f"pass the {curr_img_file_name} in {img_seq_ID}")
            continue
        img_path = os.path.join(source_real_img_seq_lidar_path, curr_img_file_name + ".png")
        img = cv2.imread(img_path)
        result = inference_model(img_model, img)
        img_semantic_label_original = np.asarray(result.pred_sem_seg.data.detach().clone().cpu()).squeeze(0) # (original_H, original_W) np array
        img_semantic_label = img_transform_later(image=img_semantic_label_original)['image']
        img_label_to_ignore_mask = np.isin(img_semantic_label, img_label_to_ignore)
        img_semantic_label[img_label_to_ignore_mask] = -1
        img_ccl_cluster_label = ski.label(img_semantic_label, background=-1, connectivity=2)  # (H, W) np array
        img_ccl_cluster_label = img_ccl_cluster_label.astype(np.int32) - 1
        img_semantic_label = img_semantic_label.astype(np.int32)
        np.save(save_path_semantic_label, img_semantic_label)
        np.save(save_path_ccl_cluster_label, img_ccl_cluster_label)
        print(f"saved {curr_img_file_name} in {img_seq_ID}")

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
    source_real_img_name = "Boreas_minuse"
    source_guiding_img_name = "Boreas_224x224_image"
    source_real_img_path = os.path.join(dataset_root, source_real_img_name)
    source_guiding_img_path = os.path.join(dataset_root, source_guiding_img_name)
    target_img_name = "Boreas_224x224_image_superpixel_mask"
    target_img_path = os.path.join(dataset_root, target_img_name)
    os.makedirs(target_img_path, exist_ok=True)
    
    #1、handle the point cloud
    # pc_label_to_ignore = np.arange(1, 9)
    # pc_total_labels = np.arange(20)
    # pc_inuse_labels = np.setdiff1d(pc_total_labels, pc_label_to_ignore)
    # dbscan_type = 'hdbscan'
    # pc_dbscan_eps = 0.15
    # pc_dbscan_min_samples = 2
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
    img_sequence_list = os.listdir(source_guiding_img_path)
    img_sequence_list = img_sequence_list[args.part_num * 4 : (args.part_num + 1) * 4]
    for img_seq_ID in img_sequence_list:
        process_img_sequence(img_seq_ID)
    
    # TODO: if both the mask is not good, consider to use SAM model and the better call SAL to generate better masks