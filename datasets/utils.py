# Author: Jacek Komorowski
# Warsaw University of Technology

import random
import copy
import torch
import numpy as np

from torch.utils.data import Sampler, Dataset
from torch.utils.data.distributed import DistributedSampler
from vision3d.ops import knn
import time
import einops
import torch
import time
import math


def in_sorted_array(e: int, array: np.ndarray) -> bool:
    pos = np.searchsorted(array, e)
    if pos == len(array) or pos == -1:
        return False
    else:
        return array[pos] == e


class ListDict(object):
    def __init__(self, items=None):
        if items is not None:
            self.items = copy.deepcopy(items)
            self.item_to_position = {item: ndx for ndx, item in enumerate(items)}
        else:
            self.items = []
            self.item_to_position = {}

    def add(self, item):
        if item in self.item_to_position:
            return
        self.items.append(item)
        self.item_to_position[item] = len(self.items)-1

    def remove(self, item):
        if item not in self.item_to_position.keys():
            return
        position = self.item_to_position.pop(item)
        last_item = self.items.pop()
        if position != len(self.items):
            self.items[position] = last_item
            self.item_to_position[last_item] = position

    def choose_random(self):
        return random.choice(self.items)

    def __contains__(self, item):
        return item in self.item_to_position

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)


class BatchSampler1(Sampler):

    # Sampler returning list of indices to form a mini-batch
    # Samples elements in groups consisting of k=2 similar elements (positives)
    # Batch has the following structure: item1_1, ..., item1_k, item2_1, ... item2_k, itemn_1, ..., itemn_k
    def __init__(self, 
                 dataset: Dataset,
                 train_val: str,
                 num_k: int,
                 sampler: DistributedSampler,
                 batch_size: int,
                 start_epoch: int = 0,
                 batch_size_limit: int = None,
                 batch_expansion_rate: float = None,
                 max_batches: int = None):

        if batch_expansion_rate is not None:
            assert batch_expansion_rate >= 1., 'batch_expansion_rate must be greater than 1'
            assert batch_size <= batch_size_limit, 'batch_size_limit must be greater or equal to batch_size'

        self.batch_size = batch_size
        self.batch_size_limit = batch_size_limit
        self.batch_expansion_rate = batch_expansion_rate
        self.max_batches = max_batches
        self.dataset = dataset
        self.train_val = train_val
        self.k = num_k 
        if self.batch_size < 2 * self.k:
            self.batch_size = 2 * self.k
            print('WARNING: Batch too small. Batch size increased to {}.'.format(self.batch_size))

        self.batch_idx = []     # Index of elements in each batch (re-generated every epoch)
        self.elems_ndx = [ i for i in range(len(self.dataset.queries))]    # List of point cloud indexes
        self.sampler = sampler
        self.epoch_ctr = start_epoch
        self.generate_batch_flag = False

    def __iter__(self):
        # Re-generate batches every epoch
        if self.sampler is not None:
            self.sampler.set_epoch(self.epoch_ctr)
        self.epoch_ctr += 1
        self.expand_batch()
        if self.train_val == 'train':
            self.generate_batches()
        elif self.train_val == 'val':
            if not self.generate_batch_flag:
                self.generate_batches()
                self.generate_batch_flag = True
        for batch in self.batch_idx:
            yield batch

    def __len__(self):
        return len(self.batch_idx)

    def expand_batch(self):
        if self.batch_expansion_rate is None:
            print('WARNING: batch_expansion_rate is None')
            return

        if self.batch_size >= self.batch_size_limit:
            return

        old_batch_size = self.batch_size
        self.batch_size = int(self.batch_size * self.batch_expansion_rate)
        self.batch_size = min(self.batch_size, self.batch_size_limit)
        print('=> Batch size increased from: {} to {}'.format(old_batch_size, self.batch_size))

    def generate_batches(self):
        # Generate training/evaluation batches.
        # batch_idx holds indexes of elements in each batch as a list of lists
        self.batch_idx = []

        if self.sampler is not None:
            unused_elements_ndx = ListDict([i for i in self.sampler])
        else:
            unused_elements_ndx = ListDict(self.elems_ndx)

        current_batch = []

        # assert self.k == 2, 'sampler can sample only k=2 elements from the same class'

        while True:
            if len(current_batch) >= self.batch_size or len(unused_elements_ndx) == 0:
                # Flush out batch, when it has a desired size, or a smaller batch, when there's no more
                # elements to process
                if len(current_batch) >= 2*self.k:
                    # Ensure there're at least two groups of similar elements, otherwise, it would not be possible
                    # to find negative examples in the batch
                    assert len(current_batch) % self.k == 0, 'Incorrect bach size: {}'.format(len(current_batch))
                    self.batch_idx.append(current_batch)
                    current_batch = []
                    if (self.max_batches is not None) and (len(self.batch_idx) >= self.max_batches):
                        break
                if len(unused_elements_ndx) == 0:
                    break

            # Add k=2 similar elements to the batch
            selected_element = unused_elements_ndx.choose_random()
            unused_elements_ndx.remove(selected_element)
            positives = self.dataset.get_positives(selected_element)
            if len(positives) == 0:
                # Broken dataset element without any positives
                continue

            unused_positives = [e for e in positives if e in unused_elements_ndx]
            # If there're unused elements similar to selected_element, sample from them
            # otherwise sample from all similar elements
            if len(unused_positives) > 0:
                second_positive = random.choice(unused_positives)
                unused_elements_ndx.remove(second_positive)
            else:
                second_positive = random.choice(list(positives))

            current_batch += [selected_element, second_positive]

        for batch in self.batch_idx:
            assert len(batch) % self.k == 0, 'Incorrect bach size: {}'.format(len(batch))


def make_collate_fn(dataset: Dataset):
    # set_transform: the transform to be applied to all batch elements
    def collate_fn(data_list):
        # Constructs a batch object
        labels = [e['ndx'] for e in data_list]

        # Compute positives and negatives mask
        positives_mask = [[in_sorted_array(e, dataset.queries[label]['positives']) for e in labels] for label in labels]
        negatives_mask = [[not in_sorted_array(e, dataset.queries[label]['non_negatives']) for e in labels] for label in labels]
        positives_mask = torch.tensor(positives_mask)
        negatives_mask = torch.tensor(negatives_mask)

        # Returns (batch_size, n_points, 3) tensor and positives_mask and
        # negatives_mask which are batch_size x batch_size boolean tensors
        result = {'positives_mask': positives_mask, 'negatives_mask': negatives_mask}

        if 'cloud' in data_list[0]:
            clouds = [torch.tensor(e['cloud']) for e in data_list]
            result['clouds'] = clouds

        if 'image' in data_list[0]:
            images = [e['image'] for e in data_list]
            result['images'] = torch.stack(images, dim=0) # Produces (N, C, H, W) tensor
        
        if 'render_images' in data_list[0]:
            render_imgs = [e['render_images'] for e in data_list]
            result['render_images'] = torch.stack(render_imgs, dim=0) # Produces (N, ViewNum, H, W) tensor
        
        if 'mask' in data_list[0]:
            masks = [e['mask'] for e in data_list]
            result['masks'] = torch.stack(masks, dim=0) # Produces (N, 1, H, W) tensor
        
            
        return result

    return collate_fn


def make_class_collate_fn(dataset, neighbor_class_weight=0.5, probability=False):
    # set_transform: the transform to be applied to all batch elements
    def collate_fn(data_list):
        # Constructs a batch object

        t6 = time.time()

        result = {}
        # Compute ground truth class matrix
        num_classes = dataset.class_num['total_class_num']
        gt_mat = torch.zeros((len(data_list), num_classes), dtype=torch.float)
        for i, data in enumerate(data_list):
            gt_mat[i, data['class']] = 1.0
            for neighbor_class in data['neighbor_class']:
                if gt_mat.shape[1] > neighbor_class:
                    gt_mat[i, neighbor_class] = neighbor_class_weight
        
        t7 = time.time()

        if probability:
            gt_sum_vet = torch.sum(gt_mat, dim=1, keepdim=True)
            gt_mat = gt_mat / gt_sum_vet
        result['gt_mat'] = gt_mat

        t8 = time.time()

        if 'cloud' in data_list[0]:
            clouds = [torch.tensor(e['cloud']) for e in data_list]

            if dataset.pc_set_transform is not None:
                # Apply the same transformation on all dataset elements
                clouds = torch.stack(clouds, dim=0)       # Produces (batch_size, n_points, 3) tensor
                clouds = dataset.set_transform(clouds)

            result['clouds'] = clouds
        
        t9 = time.time()

        if 'image' in data_list[0]:
            images = [e['image'] for e in data_list]
            result['images'] = torch.stack(images, dim=0) # Produces (N, C, H, W) tensor
        
        if 'render_images' in data_list[0]:
            render_imgs = [e['render_images'] for e in data_list]
            result['render_images'] = torch.stack(render_imgs, dim=0) # Produces (N, ViewNum, H, W) tensor
        
        t10 = time.time()

        # print(  f"t7 - t6: {t7 - t6:.4f}  "
        #         f"t8 - t7: {t8 - t7:.4f}  "
        #         f"t9 - t8: {t9 - t8:.4f}  "
        #         f"t10 - t9: {t10 - t9:.4f}  ")
            
        return result

    return collate_fn


def make_class_eval_collate_fn(dataset):
    # set_transform: the transform to be applied to all batch elements
    def collate_fn(data_list):
        # Constructs a batch object

        t6 = time.time()

        result = {}
        # Compute ground truth class matrix

        if 'cloud' in data_list[0]:
            clouds = [torch.tensor(e['cloud']) for e in data_list]

            if dataset.pc_set_transform is not None:
                # Apply the same transformation on all dataset elements
                clouds = torch.stack(clouds, dim=0)       # Produces (batch_size, n_points, 3) tensor
                clouds = dataset.set_transform(clouds)

            result['clouds'] = clouds
        
        t9 = time.time()

        if 'image' in data_list[0]:
            images = [e['image'] for e in data_list]
            result['images'] = torch.stack(images, dim=0) # Produces (N, C, H, W) tensor
        
        if 'render_images' in data_list[0]:
            render_imgs = [e['render_images'] for e in data_list]
            result['render_images'] = torch.stack(render_imgs, dim=0) # Produces (N, ViewNum, H, W) tensor
        
        t10 = time.time()

        # print(  f"t7 - t6: {t7 - t6:.4f}  "
        #         f"t8 - t7: {t8 - t7:.4f}  "
        #         f"t9 - t8: {t9 - t8:.4f}  "
        #         f"t10 - t9: {t10 - t9:.4f}  ")
            
        return result

    return collate_fn


def make_eval_collate_fn(dataset: Dataset):
    # set_transform: the transform to be applied to all batch elements
    def collate_fn(data_list):
        
        result = {}

        if 'cloud' in data_list[0]:
            clouds = [torch.tensor(e['cloud']) for e in data_list]

            result['clouds'] = clouds
            if 'pc_semantic_label' in data_list[0]:
                pc_semantic_labels = [torch.tensor(e['pc_semantic_label']) for e in data_list] 
                result['pc_semantic_labels'] = torch.stack(pc_semantic_labels, dim=0).unsqueeze(1) # Produces (B, 1, 4096) tensor
                pc_dbscan_cluster_labels = [torch.tensor(e['pc_dbscan_cluster_label']) for e in data_list]
                result['pc_dbscan_cluster_labels'] = torch.stack(pc_dbscan_cluster_labels, dim=0).unsqueeze(1) # Produces (B, 1, 4096) tensor
                remove_mask = [torch.tensor(e['cloud_remove_mask']) for e in data_list]
                result['cloud_remove_masks'] = torch.stack(remove_mask, dim=0) # Produces (B, N) tensor
                cloud_shuffle_indices = [torch.tensor(e['cloud_shuffle_indices']) for e in data_list]
                result['cloud_shuffle_indices'] = torch.stack(cloud_shuffle_indices, dim=0)

        if 'image' in data_list[0]:
            images = [e['image'] for e in data_list]
            result['images'] = torch.stack(images, dim=0) # Produces (N, C, H, W) tensor
            result['image_paths'] = [e['image_path'] for e in data_list]

            if 'img_semantic_label' in data_list[0]:
                img_semantic_labels = [torch.tensor(e['img_semantic_label']) for e in data_list] 
                result['img_semantic_labels'] = torch.stack(img_semantic_labels, dim=0).unsqueeze(1) # Produces (B, 1, 224, 224) tensor
                img_ccl_cluster_labels = [torch.tensor(e['img_ccl_cluster_label']) for e in data_list]
                result['img_ccl_cluster_labels'] = torch.stack(img_ccl_cluster_labels, dim=0).unsqueeze(1) # Produces (B, 1, 224, 224) tensor
        
        if 'render_img' in data_list[0]:
            render_imgs = [e['render_img'] for e in data_list]
            result['render_imgs'] = torch.stack(render_imgs, dim=0) # Produces (N, C, H, W) tensor
            
        
        if 'mask' in data_list[0]:
            masks = [e['mask'] for e in data_list]
            result['masks'] = torch.stack(masks, dim=0) # Produces (N, 1, H, W) tensor
        
            
        return result

    return collate_fn

def make_eval_collate_fn_kitti(dataset: Dataset):
    # set_transform: the transform to be applied to all batch elements
    def collate_fn(data_list):
        
        result = {}
        labels = [e['idx_list'] for e in data_list]
        result['labels'] = labels

        if 'cloud' in data_list[0]:
            clouds = [torch.tensor(e['cloud']) for e in data_list]
            result['clouds'] = clouds

        if 'image' in data_list[0]:
            images = [e['image'] for e in data_list]
            result['images'] = torch.stack(images, dim=0) # Produces (N, C, H, W) tensor
            result['image_paths'] = [e['image_path'] for e in data_list]
        
        if 'range_img' in data_list[0]:
            range_imgs = [e['range_img'] for e in data_list]
            result['range_imgs'] = torch.stack(range_imgs, dim=0) # Produces (B, C, range_img_H, range_img_W) tensor

        if 'pc_bev' in data_list[0]:
            pc_bevs = [e['pc_bev'] for e in data_list]
            result['pc_bevs'] = torch.stack(pc_bevs, dim=0)
        
        if 'image_bev' in data_list[0]:
            image_bevs = [e['image_bev'] for e in data_list]
            result['image_bevs'] = torch.stack(image_bevs, dim=0)

        return result

    return collate_fn


def make_collate_fn_boreas_1(dataset):
    
    def collate_fn(data_list):
        # just use the Boreas's function to complish the projection
        result = {}

        labels = [e['ndx'] for e in data_list]

        # Compute positives and negatives mask
        positives_mask = [[in_sorted_array(e, dataset.get_positives(label)) for e in labels] for label in labels] # the dataset.get_positives(label) need to be sorted first
        negatives_mask = [[not in_sorted_array(e, dataset.get_non_negatives(label)) for e in labels] for label in labels] # the dataset.get_non_negatives(label) need to be sorted first
        positives_mask = torch.tensor(positives_mask)
        negatives_mask = torch.tensor(negatives_mask)

        # Returns (batch_size, n_points, 3) tensor and positives_mask and
        # negatives_mask which are batch_size x batch_size boolean tensors
        result = {'positives_mask': positives_mask, 'negatives_mask': negatives_mask}

        if 'cloud' in data_list[0]:
            clouds = [torch.tensor(e['cloud']) for e in data_list]
            result['clouds'] = clouds
            cloud_poses = [torch.tensor(e['cloud_pose']) for e in data_list]
            result['cloud_poses'] = torch.stack(cloud_poses, dim=0) # Produces (B, 4, 4) tensor
            remove_mask = [torch.tensor(e['cloud_remove_mask']) for e in data_list]
            result['cloud_remove_masks'] = torch.stack(remove_mask, dim=0) # Produces (B, N) tensor
            if 'cloud_original' in data_list[0]:
                clouds_original = [torch.tensor(e['cloud_original']) for e in data_list]
                result['clouds_original'] = torch.stack(clouds_original, dim=0)
            if 'original_2_downsampled_indices' in data_list[0]:
                original_2_downsampled_indices = [torch.tensor(e['original_2_downsampled_indices']) for e in data_list]
                result['original_2_downsampled_indices'] = torch.stack(original_2_downsampled_indices, dim=0)
            if 'cloud_pose_original' in data_list[0]:
                cloud_poses_original = [torch.tensor(e['cloud_pose_original']) for e in data_list]
                result['cloud_poses_original'] = torch.stack(cloud_poses_original, dim=0)
            if 'cloud_shuffle_indices' in data_list[0]:
                cloud_shuffle_indices = [torch.tensor(e['cloud_shuffle_indices']) for e in data_list]
                result['cloud_shuffle_indices'] = torch.stack(cloud_shuffle_indices, dim=0)

        if 'image' in data_list[0]:
            images = [e['image'] for e in data_list]
            result['images'] = torch.stack(images, dim=0) # Produces (B, C, H, W) tensor
            image_poses = [torch.tensor(e['image_pose']) for e in data_list]
            result['image_poses'] = torch.stack(image_poses, dim=0) # Produces (B, 4, 4) tensor
            image_intrinscs = [torch.tensor(e['P0']) for e in data_list] 
            result['image_intrinscs'] = torch.stack(image_intrinscs, dim=0) # Produces (B, 3, 3) tensor
        
        if 'render_img' in data_list[0]:
            render_imgs = [e['render_img'] for e in data_list]
            result['render_imgs'] = torch.stack(render_imgs, dim=0) # Produces (B, C, H, W) tensor
            
        return result

    return collate_fn

def make_collate_fn_boreas_2(dataset):
    
    def collate_fn(data_list):
        # just use the Boreas's function to complish the projection
        result = {}

        labels = np.array([e['ndx'] for e in data_list])
        result['labels'] = labels

        # # Compute positives and negatives mask
        # all_positives = dataset.get_positives(labels)
        # all_non_negatives = dataset.get_non_negatives(labels)
        # positives_list = []
        # negatives_list = []
        # # print(np.sort(labels))
        # for i in range(len(labels)):
        #     positives_list.append(np.isin(labels, all_positives[i], assume_unique=True))
        #     negatives_list.append(~np.isin(labels, all_non_negatives[i], assume_unique=True))
        # positives_mask = torch.tensor(np.stack(positives_list, axis=0))
        # negatives_mask = torch.tensor(np.stack(negatives_list, axis=0))

        # positives_mask = [[in_sorted_array(e, dataset.get_positives(label)) for e in labels] for label in labels]
        # negatives_mask = [[not in_sorted_array(e, dataset.get_non_negatives(label)) for e in labels] for label in labels]
        # positives_mask = torch.tensor(positives_mask)
        # negatives_mask = torch.tensor(negatives_mask)
        # print(f"the new positive mask equal? {torch.equal(positives_mask, positives_mask_temp)}")
        # print(f"the new negative_mask equal? {torch.equal(negatives_mask, negatives_mask_temp)}")

        # result = {'positives_mask': positives_mask, 'negatives_mask': negatives_mask}

        if 'cloud' in data_list[0]:
            clouds = [torch.tensor(e['cloud']) for e in data_list]
            result['clouds'] = clouds
            cloud_poses = [torch.tensor(e['cloud_pose']) for e in data_list]
            result['cloud_poses'] = torch.stack(cloud_poses, dim=0) # Produces (B, 4, 4) tensor
            remove_mask = [torch.tensor(e['cloud_remove_mask']) for e in data_list]
            result['cloud_remove_masks'] = torch.stack(remove_mask, dim=0) # Produces (B, N) tensor
            if 'cloud_original' in data_list[0]:
                clouds_original = [torch.tensor(e['cloud_original']) for e in data_list]
                result['clouds_original'] = torch.stack(clouds_original, dim=0)
            if 'original_2_downsampled_indices' in data_list[0]:
                original_2_downsampled_indices = [torch.tensor(e['original_2_downsampled_indices']) for e in data_list]
                result['original_2_downsampled_indices'] = torch.stack(original_2_downsampled_indices, dim=0)
            if 'cloud_pose_original' in data_list[0]:
                cloud_poses_original = [torch.tensor(e['cloud_pose_original']) for e in data_list]
                result['cloud_poses_original'] = torch.stack(cloud_poses_original, dim=0) # Produces (B, 4, 4) tensor
            if 'cloud_shuffle_indices' in data_list[0]:
                cloud_shuffle_indices = [torch.tensor(e['cloud_shuffle_indices']) for e in data_list]
                result['cloud_shuffle_indices'] = torch.stack(cloud_shuffle_indices, dim=0)
            if 'pc_semantic_label' in data_list[0]:
                pc_semantic_labels = [torch.tensor(e['pc_semantic_label']) for e in data_list] 
                result['pc_semantic_labels'] = torch.stack(pc_semantic_labels, dim=0).unsqueeze(1) # Produces (B, 1, 4096) tensor
                pc_dbscan_cluster_labels = [torch.tensor(e['pc_dbscan_cluster_label']) for e in data_list]
                result['pc_dbscan_cluster_labels'] = torch.stack(pc_dbscan_cluster_labels, dim=0).unsqueeze(1) # Produces (B, 1, 4096) tensor

        if 'image' in data_list[0]:
            images = [e['image'] for e in data_list]
            result['images'] = torch.stack(images, dim=0) # Produces (B, C, H, W) tensor
            image_poses = [torch.tensor(e['image_pose']) for e in data_list]
            result['image_poses'] = torch.stack(image_poses, dim=0) # Produces (B, 4, 4) tensor
            image_intrinscs = [torch.tensor(e['P0']) for e in data_list] 
            result['image_intrinscs'] = torch.stack(image_intrinscs, dim=0) # Produces (B, 3, 3) tensor
            result['image_paths'] = [e['image_path'] for e in data_list]

            if 'img_semantic_label' in data_list[0]:
                img_semantic_labels = [torch.tensor(e['img_semantic_label']) for e in data_list] 
                result['img_semantic_labels'] = torch.stack(img_semantic_labels, dim=0).unsqueeze(1) # Produces (B, 1, 224, 224) tensor
                img_ccl_cluster_labels = [torch.tensor(e['img_ccl_cluster_label']) for e in data_list]
                result['img_ccl_cluster_labels'] = torch.stack(img_ccl_cluster_labels, dim=0).unsqueeze(1) # Produces (B, 1, 224, 224) tensor
        
        if 'render_img' in data_list[0]:
            render_imgs = [e['render_img'] for e in data_list]
            result['render_imgs'] = torch.stack(render_imgs, dim=0) # Produces (B, C, H, W) tensor
        
        if 'rgb_depth_label' in data_list[0]:
            rgb_depth_labels = [torch.tensor(e['rgb_depth_label']) for e in data_list] 
            result['rgb_depth_labels'] = torch.stack(rgb_depth_labels, dim=0) # Produces (B, H, W) tensor
            
        return result

    return collate_fn

def make_collate_fn_kitti(dataset):
    
    def collate_fn(data_list):
        # just use the Boreas's function to complish the projection
        result = {}

        labels = [e['idx_list'] for e in data_list]
        result['labels'] = labels

        # # Compute positives and negatives mask
        # all_positives = dataset.get_positives(labels)
        # all_non_negatives = dataset.get_non_negatives(labels)
        # positives_list = []
        # negatives_list = []
        # # print(np.sort(labels))
        # for i in range(len(labels)):
        #     positives_list.append(np.isin(labels, all_positives[i], assume_unique=True))
        #     negatives_list.append(~np.isin(labels, all_non_negatives[i], assume_unique=True))
        # positives_mask = torch.tensor(np.stack(positives_list, axis=0))
        # negatives_mask = torch.tensor(np.stack(negatives_list, axis=0))

        # positives_mask = [[in_sorted_array(e, dataset.get_positives(label)) for e in labels] for label in labels]
        # negatives_mask = [[not in_sorted_array(e, dataset.get_non_negatives(label)) for e in labels] for label in labels]
        # positives_mask = torch.tensor(positives_mask)
        # negatives_mask = torch.tensor(negatives_mask)
        # print(f"the new positive mask equal? {torch.equal(positives_mask, positives_mask_temp)}")
        # print(f"the new negative_mask equal? {torch.equal(negatives_mask, negatives_mask_temp)}")

        # result = {'positives_mask': positives_mask, 'negatives_mask': negatives_mask}

        if 'cloud' in data_list[0]:
            clouds = [torch.tensor(e['cloud']) for e in data_list]
            result['clouds'] = clouds
            cloud_poses = [torch.tensor(e['cloud_pose']) for e in data_list]
            result['cloud_poses'] = torch.stack(cloud_poses, dim=0) # Produces (B, 4, 4) tensor
            remove_mask = [torch.tensor(e['cloud_remove_mask']) for e in data_list]
            result['cloud_remove_masks'] = torch.stack(remove_mask, dim=0) # Produces (B, N) tensor
            if 'cloud_original' in data_list[0]:
                clouds_original = [torch.tensor(e['cloud_original']) for e in data_list]
                result['clouds_original'] = torch.stack(clouds_original, dim=0)
            if 'original_2_downsampled_indices' in data_list[0]:
                original_2_downsampled_indices = [torch.tensor(e['original_2_downsampled_indices']) for e in data_list]
                result['original_2_downsampled_indices'] = torch.stack(original_2_downsampled_indices, dim=0)
            if 'cloud_pose_original' in data_list[0]:
                cloud_poses_original = [torch.tensor(e['cloud_pose_original']) for e in data_list]
                result['cloud_poses_original'] = torch.stack(cloud_poses_original, dim=0) # Produces (B, 4, 4) tensor
            if 'cloud_shuffle_indices' in data_list[0]:
                cloud_shuffle_indices = [torch.tensor(e['cloud_shuffle_indices']) for e in data_list]
                result['cloud_shuffle_indices'] = torch.stack(cloud_shuffle_indices, dim=0)

        if 'image' in data_list[0]:
            images = [e['image'] for e in data_list]
            result['images'] = torch.stack(images, dim=0) # Produces (B, C, H, W) tensor
            image_poses = [torch.tensor(e['image_pose']) for e in data_list]
            result['image_poses'] = torch.stack(image_poses, dim=0) # Produces (B, 4, 4) tensor
            image_intrinscs = [torch.tensor(e['P0']) for e in data_list] 
            result['image_intrinscs'] = torch.stack(image_intrinscs, dim=0) # Produces (B, 3, 3) tensor
            result['image_paths'] = [e['image_path'] for e in data_list]
        
        if 'range_img' in data_list[0]:
            range_imgs = [e['range_img'] for e in data_list]
            result['range_imgs'] = torch.stack(range_imgs, dim=0) # Produces (B, C, range_img_H, range_img_W) tensor
            if 'range_to_pc_original_idxs' in data_list[0]:
                range_to_pc_original_idxs = [torch.tensor(e['range_to_pc_original_idxs']) for e in data_list]
                result['range_to_pc_original_idxs'] = torch.stack(range_to_pc_original_idxs, dim=0) # Produces (B, range_img_H, range_img_W) tensor
            else:
                result['range_to_pc_original_idxs'] = None
        
        if 'pc_bev' in data_list[0]:
            pc_bevs = [e['pc_bev'] for e in data_list]
            result['pc_bevs'] = torch.stack(pc_bevs, dim=0)
        
        if 'image_bev' in data_list[0]:
            image_bevs = [e['image_bev'] for e in data_list]
            result['image_bevs'] = torch.stack(image_bevs, dim=0)
        
        return result

    return collate_fn

def make_collate_fn_zenseact_1(dataset):

    def collate_fn(data_list):
        # just use the Boreas's function to complish the projection
        result = {}

        clouds = [torch.tensor(e['cloud']) for e in data_list]
        result['clouds'] = clouds
        cloud_extrinsics = [torch.tensor(e['lidar_extrinsics']) for e in data_list]
        result['cloud_extrinsics'] = torch.stack(cloud_extrinsics, dim=0) # Produces (B, 4, 4) tensor

        images = [e['image'] for e in data_list]
        result['images'] = torch.stack(images, dim=0) # Produces (B, C, H, W) tensor
        camera_intrinsics = [torch.tensor(e['camera_intrinsics']) for e in data_list]
        result['camera_intrinsics'] = torch.stack(camera_intrinsics, dim=0) # Produces (B, 3, 3) tensor
        camera_extrinsics = [torch.tensor(e['camera_extrinsics']) for e in data_list] 
        result['camera_extrinsics'] = torch.stack(camera_extrinsics, dim=0) # Produces (B, 4, 4) tensor
        camera_distortions = [torch.tensor(e['camera_distortion']) for e in data_list]
        result['camera_distortions'] = torch.stack(camera_distortions, dim=0) # Produces (B, 4) tensor
        
        positives_mask = torch.eye(len(data_list), dtype=torch.bool)
        result['positives_mask'] = positives_mask
        result['negatives_mask'] = ~positives_mask
            
        return result

    return collate_fn


class BatchSampler2(Sampler):

    # Sampler returning list of indices to form a mini-batch
    # Samples elements in groups consisting of k(really k) similar elements (positives)
    # Batch has the following structure: item1_1, ..., item1_k, item2_1, ... item2_k, itemn_1, ..., itemn_k
    def __init__(self, 
                 dataset: Dataset,
                 train_val: str,
                 num_k: int,
                 sampler: DistributedSampler,
                 batch_size: int,
                 start_epoch: int = 0,
                 batch_size_limit: int = None,
                 batch_expansion_rate: float = None,
                 max_batches: int = None):

        if batch_expansion_rate is not None:
            assert batch_expansion_rate >= 1., 'batch_expansion_rate must be greater than 1'
            assert batch_size <= batch_size_limit, 'batch_size_limit must be greater or equal to batch_size'

        self.batch_size = batch_size
        self.batch_size_limit = batch_size_limit
        self.batch_expansion_rate = batch_expansion_rate
        self.max_batches = max_batches
        self.dataset = dataset
        self.train_val = train_val
        self.k = num_k 
        if self.batch_size < 2 * self.k:
            self.batch_size = 2 * self.k
            print('WARNING: Batch too small. Batch size increased to {}.'.format(self.batch_size))

        self.batch_idx = []     # Index of elements in each batch (re-generated every epoch)
        self.elems_ndx = [ i for i in range(len(self.dataset))]    # List of point cloud indexes
        self.sampler = sampler
        self.epoch_ctr = start_epoch
        self.generate_batch_flag = False

    def __iter__(self):
        # Re-generate batches every epoch
        if self.sampler is not None:
            self.sampler.set_epoch(self.epoch_ctr)
        self.epoch_ctr += 1
        self.expand_batch()
        if self.train_val == 'train':
            self.generate_batches()
        elif self.train_val == 'val':
            if not self.generate_batch_flag:
                self.generate_batches()
                self.generate_batch_flag = True
        for batch in self.batch_idx:
            yield batch

    def __len__(self):
        return len(self.batch_idx)

    def expand_batch(self):
        if self.batch_expansion_rate is None:
            print('WARNING: batch_expansion_rate is None')
            return

        if self.batch_size >= self.batch_size_limit:
            return

        old_batch_size = self.batch_size
        self.batch_size = int(self.batch_size * self.batch_expansion_rate)
        self.batch_size = min(self.batch_size, self.batch_size_limit)
        print('=> Batch size increased from: {} to {}'.format(old_batch_size, self.batch_size))

    def generate_batches(self):
        # Generate training/evaluation batches.
        # batch_idx holds indexes of elements in each batch as a list of lists
        self.batch_idx = []

        if self.sampler is not None:
            unused_elements_ndx = ListDict([i for i in self.sampler])
        else:
            unused_elements_ndx = ListDict(self.elems_ndx)

        current_batch = []

        # assert self.k == 2, 'sampler can sample only k=2 elements from the same class'

        while True:
            if len(current_batch) >= self.batch_size or len(unused_elements_ndx) == 0:
                # Flush out batch, when it has a desired size, or a smaller batch, when there's no more
                # elements to process
                if len(current_batch) >= 2*self.k:
                    # Ensure there're at least two groups of similar elements, otherwise, it would not be possible
                    # to find negative examples in the batch
                    assert len(current_batch) % self.k == 0, 'Incorrect bach size: {}'.format(len(current_batch))
                    self.batch_idx.append(current_batch)
                    current_batch = []
                    if (self.max_batches is not None) and (len(self.batch_idx) >= self.max_batches):
                        break
                if len(unused_elements_ndx) == 0:
                    break

            # Add k=2 similar elements to the batch
            selected_element = unused_elements_ndx.choose_random()
            unused_elements_ndx.remove(selected_element)
            positives = self.dataset.get_positives(selected_element)
            if len(positives) == 1:
                # Broken dataset element without any positives except itself 
                # TODO: be careful that whether all the datasets used take itself as a positive
                continue
            elif len(positives) < self.k:
                positives_inuse = []
                while len(positives_inuse) < self.k:
                    positives_inuse += positives
            else:
                positives_inuse = positives

            unused_positives = [e for e in positives_inuse if e in unused_elements_ndx]
            if len(unused_positives) >= (self.k - 1):
                unused_positives_array = np.array(unused_positives)
                random_unused_positives = np.random.permutation(unused_positives_array)
                selected_unused_positives = list(random_unused_positives[:self.k-1])
                for e in selected_unused_positives:
                    unused_elements_ndx.remove(e)
                selected_positives = selected_unused_positives
            elif len(unused_positives) > 0:
                selected_unused_positives = unused_positives
                for e in selected_unused_positives:
                    unused_elements_ndx.remove(e)
                delta = self.k - 1 - len(unused_positives)
                used_positives_array = np.array(positives_inuse)
                random_used_positives = np.random.permutation(used_positives_array)
                selected_used_positives = list(random_used_positives[:delta])
                selected_positives = selected_unused_positives + selected_used_positives
            else:
                used_positives_array = np.array(positives_inuse)
                random_used_positives = np.random.permutation(used_positives_array)
                selected_used_positives = list(random_used_positives[:self.k - 1])
                selected_positives = selected_used_positives

            current_batch += ([selected_element] + selected_positives)

        for batch in self.batch_idx:
            assert len(batch) % self.k == 0, 'Incorrect bach size: {}'.format(len(batch))



class BatchSampler3(Sampler):

    # Sampler returning list of indices to form a mini-batch
    # Samples elements in groups consisting of k different similar level elements (positives)
    # Batch has the following structure: item1_1, item11_2, item12_3..., item1k_k+1, item2_k+2, item3_k+3... item(batchsize-k)_batchsize
    # this structure is convenient for ddp training
    def __init__(self, 
                 dataset: Dataset,
                 num_k: int,
                 batch_size: int,
                 safe_elem_region: int = 200,
                 start_epoch: int = 0,
                 iter_per_epoch: int = 1000):

        self.batch_size = batch_size
        self.dataset = dataset
        self.k = num_k  
        self.batch_idx = []     # Index of elements in each batch (re-generated every epoch)
        self.epoch_ctr = start_epoch
        self.safe_elem_region = safe_elem_region
        self.iter_per_epoch = iter_per_epoch
        self.sample_used = np.zeros(len(dataset.queries), dtype=np.int64)

    def __iter__(self):
        # Re-generate batches every epoch
        self.generate_batches(self.epoch_ctr)
        self.epoch_ctr += 1
        for batch in self.batch_idx:
            yield batch

    def __len__(self):
        return len(self.batch_idx)

    def generate_batches(self, epoch_ctr):
        # Generate training/evaluation batches.
        # batch_idx holds indexes of elements in each batch as a list of lists
        curr_traversal = epoch_ctr % self.dataset.traversal_num
        curr_seq_ID = self.dataset.official_dataset.sequences[curr_traversal].ID
        elements_ndx = np.array(self.dataset.traversal_idxs[curr_seq_ID], dtype=np.int64) # (ele_num,)
        elements_flag = np.zeros(len(self.dataset.traversal_idxs[curr_seq_ID]), dtype=np.int64) # (ele_num,)
        elements = np.stack([elements_ndx, elements_flag], axis=1) # (ele_num, 2)
        elements = np.expand_dims(elements, axis=0) # (1, ele_num, 2)
        elements = np.tile(elements, (elements.shape[1], 1, 1)) # (ele_num, ele_num, 2)
        # suppose the traversal_idxs is sorted by timestamp
        self.batch_idx = []
        current_batch = []
        random_start = np.random.randint(0, len(elements))
        for i in range(self.iter_per_epoch):
            real_i = (i + random_start) % len(elements)
            start_i = (real_i + len(elements) - self.safe_elem_region) % len(elements)
            end_i = (real_i + self.safe_elem_region) % len(elements)
            if start_i <= end_i:
                elements[real_i, start_i:end_i, 1] = 1e8
            else:
                elements[real_i, start_i:, 1] = 1e8
                elements[real_i, :end_i, 1] = 1e8
            elements_real_i_sorted = np.sort(elements[real_i, :, 1])
            max_num = elements_real_i_sorted[self.batch_size - self.k - 1]
            elements_chose_1 = np.nonzero(elements[real_i, :, 1] < max_num)[0]
            elements_to_rand = np.nonzero(elements[real_i, :, 1] == max_num)[0]
            elements_chose_2 = np.random.permutation(elements_to_rand)[:self.batch_size - self.k - 1 - len(elements_chose_1)]
            elements_chose = np.concatenate((elements_chose_1, elements_chose_2))
            elements[real_i, elements_chose, 1] += 1
            elements[elements_chose, real_i, 1] += 1
            elements_chose_idxs = elements[real_i, elements_chose, 0]
            current_batch = [elements[real_i, real_i, 0]]
            for elements_chose_idx in elements_chose_idxs:
                chose_level_positives = self.dataset.queries[elements_chose_idx]['level_positives']
                current_batch.append(random.choice(chose_level_positives['level_1']))
            level_positives = self.dataset.queries[elements[real_i, real_i, 0]]['level_positives']
            for curr_level, curr_level_positives in level_positives.items():
                if int(curr_level.split('_')[-1]) <= self.k:
                    current_batch.append(random.choice(curr_level_positives))
            self.sample_used[current_batch] += 1
            self.batch_idx.append(current_batch)
            current_batch = []
        print('generate_batches done!')

class BatchSampler3_2(Sampler):

    # Sampler returning list of indices to form a mini-batch
    # Samples elements in groups consisting of k different similar level elements (positives)
    # Batch has the following structure: item1_1, item11_2, item12_3..., item1k_k+1, item2_k+2, item3_k+3... item(batchsize-k)_batchsize
    # this structure is convenient for ddp training
    def __init__(self, 
                 dataset: Dataset,
                 num_k: int,
                 batch_size: int,
                 safe_elem_region: int = 200,
                 start_epoch: int = 0,
                 iter_per_epoch: int = 1000,
                 level1_distance: float = 10.0):

        self.batch_size = batch_size
        self.dataset = dataset
        self.k = num_k  
        self.batch_idx = []     # Index of elements in each batch (re-generated every epoch)
        self.epoch_ctr = start_epoch
        self.safe_elem_region = safe_elem_region
        self.iter_per_epoch = iter_per_epoch
        self.sample_used = np.zeros(len(dataset), dtype=np.int64)
        self.level1_distance = level1_distance

    def __iter__(self):
        # Re-generate batches every epoch
        self.generate_batches(self.epoch_ctr)
        self.epoch_ctr += 1
        for batch in self.batch_idx:
            yield batch

    def __len__(self):
        return len(self.batch_idx)

    def generate_batches(self, epoch_ctr):
        # Generate training/evaluation batches.
        # batch_idx holds indexes of elements in each batch as a list of lists
        curr_traversal = epoch_ctr % self.dataset.traversal_num
        curr_seq_ID = self.dataset.official_dataset.sequences[curr_traversal].ID
        elements_ndx = np.array(self.dataset.traversal_idxs[curr_seq_ID], dtype=np.int64) # (ele_num,)
        elements_flag = np.zeros(len(self.dataset.traversal_idxs[curr_seq_ID]), dtype=np.int64) # (ele_num,)
        elements = np.stack([elements_ndx, elements_flag], axis=1) # (ele_num, 2)
        elements = np.expand_dims(elements, axis=0) # (1, ele_num, 2)
        elements = np.tile(elements, (elements.shape[1], 1, 1)) # (ele_num, ele_num, 2)
        # suppose the traversal_idxs is sorted by timestamp
        self.batch_idx = []
        current_batch = []
        random_start = np.random.randint(0, len(elements))
        for i in range(self.iter_per_epoch):
            real_i = (i + random_start) % len(elements)
            start_i = (real_i + len(elements) - self.safe_elem_region) % len(elements)
            end_i = (real_i + self.safe_elem_region) % len(elements)
            if start_i <= end_i:
                elements[real_i, start_i:end_i, 1] = 1e8
            else:
                elements[real_i, start_i:, 1] = 1e8
                elements[real_i, :end_i, 1] = 1e8
            elements_real_i_sorted = np.sort(elements[real_i, :, 1])
            max_num = elements_real_i_sorted[self.batch_size - self.k - 1]
            elements_chose_1 = np.nonzero(elements[real_i, :, 1] < max_num)[0]
            elements_to_rand = np.nonzero(elements[real_i, :, 1] == max_num)[0]
            elements_chose_2 = np.random.permutation(elements_to_rand)[:self.batch_size - self.k - 1 - len(elements_chose_1)]
            elements_chose = np.concatenate((elements_chose_1, elements_chose_2))
            elements[real_i, elements_chose, 1] += 1
            elements[elements_chose, real_i, 1] += 1
            elements_chose_idxs = elements[real_i, elements_chose, 0]
            current_batch = [elements[real_i, real_i, 0]]



            # for elements_chose_idx in elements_chose_idxs:
            #     chose_level_positives = self.dataset.queries[elements_chose_idx]['level_positives']
            #     current_batch.append(random.choice(chose_level_positives['level_1']))
            # level_positives = self.dataset.queries[elements[real_i, real_i, 0]]['level_positives']
            # for curr_level, curr_level_positives in level_positives.items():
            #     if int(curr_level.split('_')[-1]) <= self.k:
            #         current_batch.append(random.choice(curr_level_positives))

            chose_level1_positives = self.dataset.get_distance_positives(elements_chose_idxs, self.level1_distance)
            for i in range(elements_chose_idxs.shape[0]):
                current_batch.append(random.choice(list(chose_level1_positives[i])))
            level_positives = self.dataset.get_level_positives(elements[real_i, real_i, 0])[0]
            for curr_level, curr_level_positives in enumerate(level_positives):
                if (curr_level + 1) <= self.k:
                    current_batch.append(random.choice(list(curr_level_positives)))

            self.sample_used[current_batch] += 1
            self.batch_idx.append(current_batch)
            current_batch = []
        print('generate_batches done!')

class BatchSampler4(Sampler):

    # Sampler returning list of indices to form a mini-batch
    # Samples elements in groups consisting of k different similar level elements (positives)
    # Batch has the following structure: item1_1, item11_2, item12_3..., item1k_k+1, item2_k+2, item3_k+3... item(batchsize-k)_batchsize
    # compared with BatchSampler3, this sampler use all the sample in one epoch and try best to avoid the repeated sample
    def __init__(self, 
                 dataset: Dataset,
                 num_k: int,
                 batch_size: int,
                 start_epoch: int = 0,
                 iter_per_epoch: int = 2000):

        self.batch_size = batch_size
        self.dataset = dataset
        self.k = num_k  
        self.batch_idx = []     # Index of elements in each batch (re-generated every epoch)
        self.epoch_ctr = start_epoch
        self.sample_used = np.zeros(len(dataset.queries), dtype=np.int64)
        self.elements_ndx = np.zeros((len(self.dataset.queries), len(self.dataset.queries)), dtype=np.int64) # (ele_num, ele_num)
        self.elements_positives_cumsum = np.zeros((len(self.dataset.queries), len(self.dataset.queries[0]["level_positives"])), dtype=np.int64) # (ele_num, level_num) to (ele_num, level_num + 1)
        self.init_elements_ndx()
        self.iter_per_epoch = iter_per_epoch
    
    def init_elements_ndx(self):
        for i in range(len(self.dataset.queries)):
            self.elements_ndx[i, i] = 1e9
            for curr_level_name, curr_level_positives in self.dataset.queries[i]['level_positives'].items():
                curr_level = int(curr_level_name.split('_')[-1])
                self.elements_positives_cumsum[i, curr_level - 1] = len(curr_level_positives)
                curr_level_positives = np.array(curr_level_positives, dtype=np.int64)
                self.elements_ndx[i, curr_level_positives] = curr_level * 1e6
            self.elements_ndx[i, np.argwhere(self.elements_ndx[i, :] == 0)] = 1e8
        self.elements_positives_cumsum = np.cumsum(self.elements_positives_cumsum, axis=1)
        self.elements_positives_cumsum = np.concatenate((np.zeros((len(self.dataset.queries), 1), dtype=np.int64), self.elements_positives_cumsum), axis=1) # (ele_num, level_num + 1)
    
    def load_elements_ndx(self, elements_ndx):
        """
        assume the input elements_ndx is a pytorch tensor with shape (ele_num, ele_num)
        """
        self.elements_ndx = np.array(elements_ndx, dtype=np.int64)
    
    def save_elements_ndx(self):
        return torch.tensor(self.elements_ndx, dtype=torch.int64)

    def __iter__(self):
        # Re-generate batches every epoch
        self.generate_batches(self.epoch_ctr)
        self.epoch_ctr += 1
        for batch in self.batch_idx:
            yield batch

    def __len__(self):
        return len(self.batch_idx)

    def generate_batches(self, epoch_ctr):
        # suppose the traversal_idxs is sorted by timestamp
        self.batch_idx = []
        current_batch = []
        for i in range(self.iter_per_epoch):
            elements_ndx_diag = np.diag(self.elements_ndx)
            elements_ndx_diag_sorted = np.sort(elements_ndx_diag)
            elements_ndx_min = elements_ndx_diag_sorted[0]
            real_i = np.random.permutation(np.nonzero(elements_ndx_diag == elements_ndx_min)[0])[0]
            elements_real_i_sorted = np.sort(self.elements_ndx[real_i, :]) # (ele_num,)
            elements_real_i_sorted_idx = np.argsort(self.elements_ndx[real_i, :]) # (ele_num,)
            elements_real_i_level_min = elements_real_i_sorted[self.elements_positives_cumsum[real_i, :self.k]] # (k,)
            elements_real_i_level_min_first = np.searchsorted(elements_real_i_sorted, elements_real_i_level_min, side="left")
            elements_real_i_level_min_last = np.searchsorted(elements_real_i_sorted, elements_real_i_level_min, side="right")
            num_to_choose_among_levels = elements_real_i_level_min_last - elements_real_i_level_min_first
            max_to_choose_num_among_levels = np.max(num_to_choose_among_levels)
            random_base = np.random.randint(low=0, high=max_to_choose_num_among_levels, size=self.k, dtype=np.int64)
            random_base = random_base % num_to_choose_among_levels
            random_index = random_base + self.elements_positives_cumsum[real_i, :self.k]
            elements_chose_1 = elements_real_i_sorted_idx[random_index]
            elements_chose_2 = np.random.permutation(elements_real_i_sorted_idx[self.elements_positives_cumsum[real_i, self.k]:])[:self.batch_size - self.k - 1]
            elements_chose = np.concatenate((elements_chose_1, elements_chose_2))
            self.elements_ndx[:, elements_chose] += 1
            self.elements_ndx[:, real_i] += 1
            current_batch = [real_i]
            current_batch += list(elements_chose)
            self.sample_used[current_batch] += 1
            self.batch_idx.append(current_batch)
            current_batch = []
        print('generate_batches done!')

class BatchSampler4_2(Sampler):

    # Sampler returning list of indices to form a mini-batch
    # Samples elements in groups consisting of k different similar level elements (positives)
    # Batch has the following structure: item1_1, item11_2, item12_3..., item1k_k+1, item2_k+2, item3_k+3... item(batchsize-k)_batchsize
    # compared with BatchSampler3, this sampler use all the sample in one epoch and try best to avoid the repeated sample
    def __init__(self, 
                 dataset: Dataset,
                 num_k: int,
                 batch_size: int,
                 start_epoch: int = 0,
                 iter_per_epoch: int = 2000):

        self.batch_size = batch_size
        self.dataset = dataset
        self.k = num_k  
        self.batch_idx = []     # Index of elements in each batch (re-generated every epoch)
        self.epoch_ctr = start_epoch
        self.sample_used = np.zeros(len(self.dataset), dtype=np.int64)
        self.elements_ndx = np.zeros((len(self.dataset), len(self.dataset)), dtype=np.int64) # (ele_num, ele_num)
        self.elements_positives_cumsum = np.zeros((len(self.dataset), len(self.dataset.positive_distance_list)), dtype=np.int64) # (ele_num, level_num) to (ele_num, level_num + 1)
        self.init_elements_ndx()
        self.iter_per_epoch = iter_per_epoch
    
    def init_elements_ndx(self):
        for i in range(len(self.dataset)):
            self.elements_ndx[i, i] = 1e9
            level_positives = self.dataset.get_level_positives(i)[0]
            for curr_level, curr_level_positives in enumerate(level_positives):
                curr_level += 1
                self.elements_positives_cumsum[i, curr_level - 1] = len(curr_level_positives)
                self.elements_ndx[i, curr_level_positives] = curr_level * 1e6
            self.elements_ndx[i, np.argwhere(self.elements_ndx[i, :] == 0)] = 1e8
        self.elements_positives_cumsum = np.cumsum(self.elements_positives_cumsum, axis=1)
        self.elements_positives_cumsum = np.concatenate((np.zeros((len(self.dataset), 1), dtype=np.int64), self.elements_positives_cumsum), axis=1) # (ele_num, level_num + 1)
    
    def load_elements_ndx(self, elements_ndx):
        """
        assume the input elements_ndx is a pytorch tensor with shape (ele_num, ele_num)
        """
        self.elements_ndx = np.array(elements_ndx, dtype=np.int64)
    
    def save_elements_ndx(self):
        return torch.tensor(self.elements_ndx, dtype=torch.int64)

    def __iter__(self):
        # Re-generate batches every epoch
        self.generate_batches(self.epoch_ctr)
        self.epoch_ctr += 1
        for batch in self.batch_idx:
            yield batch

    def __len__(self):
        return len(self.batch_idx)

    def generate_batches(self, epoch_ctr):
        # suppose the traversal_idxs is sorted by timestamp
        self.batch_idx = []
        current_batch = []
        for i in range(self.iter_per_epoch):
            elements_ndx_diag = np.diag(self.elements_ndx)
            elements_ndx_diag_sorted = np.sort(elements_ndx_diag)
            elements_ndx_min = elements_ndx_diag_sorted[0]
            real_i = np.random.permutation(np.nonzero(elements_ndx_diag == elements_ndx_min)[0])[0]
            elements_real_i_sorted = np.sort(self.elements_ndx[real_i, :]) # (ele_num,)
            elements_real_i_sorted_idx = np.argsort(self.elements_ndx[real_i, :]) # (ele_num,)
            elements_real_i_level_min = elements_real_i_sorted[self.elements_positives_cumsum[real_i, :self.k]] # (k,)
            elements_real_i_level_min_first = np.searchsorted(elements_real_i_sorted, elements_real_i_level_min, side="left")
            elements_real_i_level_min_last = np.searchsorted(elements_real_i_sorted, elements_real_i_level_min, side="right")
            num_to_choose_among_levels = elements_real_i_level_min_last - elements_real_i_level_min_first
            max_to_choose_num_among_levels = np.max(num_to_choose_among_levels)
            random_base = np.random.randint(low=0, high=max_to_choose_num_among_levels, size=self.k, dtype=np.int64)
            random_base = random_base % num_to_choose_among_levels
            random_index = random_base + self.elements_positives_cumsum[real_i, :self.k]
            elements_chose_1 = elements_real_i_sorted_idx[random_index]
            elements_chose_2 = np.random.permutation(elements_real_i_sorted_idx[self.elements_positives_cumsum[real_i, self.k]:])[:self.batch_size - self.k - 1]
            elements_chose = np.concatenate((elements_chose_1, elements_chose_2))
            self.elements_ndx[:, elements_chose] += 1
            self.elements_ndx[:, real_i] += 1
            current_batch = [real_i]
            current_batch += list(elements_chose)
            self.sample_used[current_batch] += 1
            self.batch_idx.append(current_batch)
            current_batch = []
        print('generate_batches done!')

class BatchSampler5(Sampler):
            
    # Sampler returning list of indices to form a mini-batch
    # Samples elements in groups consisting of k different similar level elements (positives)
    # Batch has the following structure: item1_1, item11_2, item12_3..., item1k_k+1, item2_k+2, item3_k+3... item(batchsize-k)_batchsize
    # compared with BatchSampler3, this sampler use all the sample in one epoch and try best to avoid the repeated sample
    def __init__(self, 
                 dataset: Dataset,
                 num_k: int,
                 batch_size: int,
                 start_epoch: int = 0,
                 iter_per_epoch: int = 2000):

        self.batch_size = batch_size
        self.dataset = dataset
        self.k = num_k  
        self.batch_idx = []     # Index of elements in each batch (re-generated every epoch)
        self.epoch_ctr = start_epoch
        self.sample_used = np.zeros(len(dataset.queries), dtype=np.int64)
        self.elements_ndx = np.zeros((len(self.dataset.queries), len(self.dataset.queries)), dtype=np.int64) # (ele_num, ele_num)
        self.elements_positives_cumsum = np.zeros((len(self.dataset.queries), len(self.dataset.queries[0]["level_positives"])), dtype=np.int64) # (ele_num, level_num) to (ele_num, level_num + 1)
        self.init_elements_ndx()
        self.iter_per_epoch = iter_per_epoch
        assert batch_size % (num_k + 1) == 0, 'batch_size must be divisible by num_k + 1'
    
    def init_elements_ndx(self):
        for i in range(len(self.dataset.queries)):
            self.elements_ndx[i, i] = 1e9
            for curr_level_name, curr_level_positives in self.dataset.queries[i]['level_positives'].items():
                curr_level = int(curr_level_name.split('_')[-1])
                self.elements_positives_cumsum[i, curr_level - 1] = len(curr_level_positives)
                curr_level_positives = np.array(curr_level_positives, dtype=np.int64)
                self.elements_ndx[i, curr_level_positives] = curr_level * 1e6
            self.elements_ndx[i, np.argwhere(self.elements_ndx[i, :] == 0)] = 1e8
        self.elements_positives_cumsum = np.cumsum(self.elements_positives_cumsum, axis=1)
        self.elements_positives_cumsum = np.concatenate((np.zeros((len(self.dataset.queries), 1), dtype=np.int64), self.elements_positives_cumsum), axis=1) # (ele_num, level_num + 1)
    
    def load_elements_ndx(self, elements_ndx):
        """
        assume the input elements_ndx is a pytorch tensor with shape (ele_num, ele_num)
        """
        self.elements_ndx = np.array(elements_ndx, dtype=np.int64)
    
    def save_elements_ndx(self):
        return torch.tensor(self.elements_ndx, dtype=torch.int64)

    def __iter__(self):
        # Re-generate batches every epoch
        self.generate_batches(self.epoch_ctr)
        self.epoch_ctr += 1
        for batch in self.batch_idx:
            yield batch

    def __len__(self):
        return len(self.batch_idx)

    def generate_batches(self, epoch_ctr):
        # suppose the traversal_idxs is sorted by timestamp
        self.batch_idx = []
        current_batch = []
        for i in range(self.iter_per_epoch * self.batch_size // (self.k + 1)):
            elements_ndx_diag = np.diag(self.elements_ndx)
            elements_ndx_diag_sorted = np.sort(elements_ndx_diag)
            elements_ndx_min = elements_ndx_diag_sorted[0]
            real_i = np.random.permutation(np.nonzero(elements_ndx_diag == elements_ndx_min)[0])[0]
            elements_real_i_sorted = np.sort(self.elements_ndx[real_i, :]) # (ele_num,)
            elements_real_i_sorted_idx = np.argsort(self.elements_ndx[real_i, :]) # (ele_num,)
            elements_real_i_level_min = elements_real_i_sorted[self.elements_positives_cumsum[real_i, :self.k]] # (k,)
            elements_real_i_level_min_first = np.searchsorted(elements_real_i_sorted, elements_real_i_level_min, side="left")
            elements_real_i_level_min_last = np.searchsorted(elements_real_i_sorted, elements_real_i_level_min, side="right")
            num_to_choose_among_levels = elements_real_i_level_min_last - elements_real_i_level_min_first
            max_to_choose_num_among_levels = np.max(num_to_choose_among_levels)
            random_base = np.random.randint(low=0, high=max_to_choose_num_among_levels, size=self.k, dtype=np.int64)
            random_base = random_base % num_to_choose_among_levels
            random_index = random_base + self.elements_positives_cumsum[real_i, :self.k]
            elements_chose = elements_real_i_sorted_idx[random_index]
            self.elements_ndx[:, elements_chose] += 1
            self.elements_ndx[:, real_i] += 1
            current_batch.append(real_i)
            current_batch += list(elements_chose)
            if len(current_batch) == self.batch_size:
                self.sample_used[current_batch] += 1
                self.batch_idx.append(current_batch)
                current_batch = []
        print('generate_batches done!')

class BatchSampler5_2(Sampler):
            
    # Sampler returning list of indices to form a mini-batch
    # Samples elements in groups consisting of k different similar level elements (positives)
    # Batch has the following structure: item1_1, item11_2, item12_3..., item1k_k+1, item2_k+2, item3_k+3... item(batchsize-k)_batchsize
    # compared with BatchSampler3, this sampler use all the sample in one epoch and try best to avoid the repeated sample
    def __init__(self, 
                 dataset: Dataset,
                 num_k: int,
                 batch_size: int,
                 start_epoch: int = 0,
                 iter_per_epoch: int = 2000):

        self.batch_size = batch_size
        self.dataset = dataset
        self.k = num_k  
        self.batch_idx = []     # Index of elements in each batch (re-generated every epoch)
        self.epoch_ctr = start_epoch
        self.sample_used = np.zeros(len(self.dataset), dtype=np.uint16)
        self.elements_ndx = np.zeros((len(self.dataset), len(self.dataset)), dtype=np.uint16) # (ele_num, ele_num)
        self.elements_positives_cumsum = np.zeros((len(self.dataset), len(self.dataset.positive_distance_list)), dtype=np.uint16) # (ele_num, level_num) to (ele_num, level_num + 1)
        self.init_elements_ndx()
        self.iter_per_epoch = iter_per_epoch
        assert batch_size % (num_k + 1) == 0, 'batch_size must be divisible by num_k + 1'
    
    def init_elements_ndx(self):
        for i in range(len(self.dataset)):
            self.elements_ndx[i, i] = 5e4 
            level_positives = self.dataset.get_level_positives(i)[0]
            for curr_level, curr_level_positives in enumerate(level_positives):
                curr_level += 1
                self.elements_positives_cumsum[i, curr_level - 1] = len(curr_level_positives)
                self.elements_ndx[i, curr_level_positives] = curr_level * 2e3
            self.elements_ndx[i, np.argwhere(self.elements_ndx[i, :] == 0)] = 4e4
        self.elements_positives_cumsum = np.cumsum(self.elements_positives_cumsum, axis=1)
        self.elements_positives_cumsum = np.concatenate((np.zeros((len(self.dataset), 1), dtype=np.uint16), self.elements_positives_cumsum), axis=1) # (ele_num, level_num + 1)
    
    def load_elements_ndx(self, elements_ndx):
        """
        assume the input elements_ndx is a pytorch tensor with shape (ele_num, ele_num)
        """
        self.elements_ndx = np.array(elements_ndx, dtype=np.uint16)
    
    def save_elements_ndx(self):
        return torch.tensor(self.elements_ndx, dtype=torch.uint16)

    def __iter__(self):
        # Re-generate batches every epoch
        self.generate_batches(self.epoch_ctr)
        self.epoch_ctr += 1
        for batch in self.batch_idx:
            yield batch

    def __len__(self):
        return len(self.batch_idx)

    def generate_batches(self, epoch_ctr):
        # suppose the traversal_idxs is sorted by timestamp
        self.batch_idx = []
        current_batch = []
        t0 = time.time()
        for i in range(self.iter_per_epoch * self.batch_size // (self.k + 1)):
            elements_ndx_diag = np.diag(self.elements_ndx)
            elements_ndx_diag_sorted = np.sort(elements_ndx_diag)
            elements_ndx_min = elements_ndx_diag_sorted[0]
            real_i = np.random.permutation(np.nonzero(elements_ndx_diag == elements_ndx_min)[0])[0]
            elements_real_i_sorted = np.sort(self.elements_ndx[real_i, :]) # (ele_num,)
            elements_real_i_sorted_idx = np.argsort(self.elements_ndx[real_i, :]) # (ele_num,)
            elements_real_i_level_min = elements_real_i_sorted[(self.elements_positives_cumsum[real_i, :self.k]).astype(np.int32)] # (k,)
            elements_real_i_level_min_first = np.searchsorted(elements_real_i_sorted, elements_real_i_level_min, side="left")
            elements_real_i_level_min_last = np.searchsorted(elements_real_i_sorted, elements_real_i_level_min, side="right")
            num_to_choose_among_levels = elements_real_i_level_min_last - elements_real_i_level_min_first
            max_to_choose_num_among_levels = np.max(num_to_choose_among_levels)
            random_base = np.random.randint(low=0, high=max_to_choose_num_among_levels, size=self.k, dtype=np.uint16)
            random_base = random_base % num_to_choose_among_levels
            random_index = random_base + self.elements_positives_cumsum[real_i, :self.k]
            elements_chose = elements_real_i_sorted_idx[random_index.astype(np.int32)]
            self.elements_ndx[:, elements_chose] += 1
            self.elements_ndx[:, real_i] += 1
            current_batch.append(real_i)
            current_batch += list(elements_chose)
            if len(current_batch) == self.batch_size:
                self.sample_used[current_batch] += 1
                self.batch_idx.append(current_batch)
                current_batch = []
        t1 = time.time()
        print('generate batches done!')
        print('batchsampler5_2 time cost: ', t1 - t0)

class BatchSampler5_3(Sampler):
            
    def __init__(self, 
                 dataset: Dataset,
                 num_k: int,
                 batch_size: int,
                 start_epoch: int = 0,
                 iter_per_epoch: int = 2000):

        self.batch_size = batch_size
        self.dataset = dataset
        self.k = num_k  
        self.batch_idx = []     # Index of elements in each batch (re-generated every epoch)
        self.epoch_ctr = start_epoch
        self.sample_used = np.zeros(len(self.dataset), dtype=np.uint16)
        self.elements_ndx = np.zeros((len(self.dataset), len(self.dataset)), dtype=np.uint16) # (ele_num, ele_num)
        self.elements_positives_cumsum = np.zeros((len(self.dataset), len(self.dataset.positive_distance_list)), dtype=np.uint16) # (ele_num, level_num) to (ele_num, level_num + 1)
        self.init_elements_ndx()
        self.iter_per_epoch = iter_per_epoch
        assert batch_size % (num_k + 1) == 0, 'batch_size must be divisible by num_k + 1'
    
    def init_elements_ndx(self):
        for i in range(len(self.dataset)):
            self.elements_ndx[i, i] = 5e4 
            level_positives = self.dataset.get_level_positives(i)[0]
            for curr_level, curr_level_positives in enumerate(level_positives):
                curr_level += 1
                self.elements_positives_cumsum[i, curr_level - 1] = len(curr_level_positives)
                self.elements_ndx[i, curr_level_positives] = curr_level * 2e3
            self.elements_ndx[i, np.argwhere(self.elements_ndx[i, :] == 0)] = 4e4
        self.elements_positives_cumsum = np.cumsum(self.elements_positives_cumsum, axis=1)
        self.elements_positives_cumsum = np.concatenate((np.zeros((len(self.dataset), 1), dtype=np.uint16), self.elements_positives_cumsum), axis=1) # (ele_num, level_num + 1)
    
    def load_elements_ndx(self, elements_ndx):
        """
        assume the input elements_ndx is a pytorch tensor with shape (ele_num, ele_num)
        """
        self.elements_ndx = np.array(elements_ndx, dtype=np.uint16)
    
    def save_elements_ndx(self):
        return torch.tensor(self.elements_ndx, dtype=torch.uint16)

    def __iter__(self):
        # Re-generate batches every epoch
        self.generate_batches(self.epoch_ctr)
        self.epoch_ctr += 1
        for batch in self.batch_idx:
            yield batch

    def __len__(self):
        return len(self.batch_idx)

    def generate_batches(self, epoch_ctr):
        # suppose the traversal_idxs is sorted by timestamp
        device = torch.cuda.current_device()
        self.batch_idx = []
        t0 = time.time()
        elements_ndx = torch.tensor((self.elements_ndx).astype(np.int32), dtype=torch.int32, device=device)
        current_batch = []
        for i in range(self.iter_per_epoch * self.batch_size // (self.k + 1)):
            elements_ndx_diag = torch.diag(elements_ndx, diagonal=0)
            elements_ndx_diag_sorted, _ = torch.sort(elements_ndx_diag)
            elements_ndx_min = elements_ndx_diag_sorted[0]
            elements_min_indices = torch.nonzero(elements_ndx_diag == elements_ndx_min, as_tuple=False)
            real_i = torch.randperm(elements_min_indices.shape[0], device=device)[0]
            real_i = elements_min_indices[real_i]
            elements_real_i_sorted, elements_real_i_sorted_idx = torch.sort(elements_ndx[real_i, :]) # (ele_num,)
            elements_positives_cumsum_inuse = torch.tensor((self.elements_positives_cumsum[real_i, :self.k]).astype(np.int32), dtype=torch.int64, device=device)
            elements_real_i_level_min = elements_real_i_sorted[elements_positives_cumsum_inuse] # (k,)
            elements_real_i_level_min_first = torch.searchsorted(elements_real_i_sorted, elements_real_i_level_min, side="left")
            elements_real_i_level_min_last = torch.searchsorted(elements_real_i_sorted, elements_real_i_level_min, side="right")
            num_to_choose_among_levels = elements_real_i_level_min_last - elements_real_i_level_min_first
            max_to_choose_num_among_levels = torch.max(num_to_choose_among_levels)
            random_base = torch.randint(low=0, high=max_to_choose_num_among_levels, size=self.k, dtype=torch.int64, device=device)
            random_base = random_base % num_to_choose_among_levels
            random_index = random_base + elements_positives_cumsum_inuse
            elements_chose = elements_real_i_sorted_idx[random_index]
            elements_ndx[:, elements_chose] += 1
            elements_ndx[:, real_i] += 1
            current_batch.append(real_i.cpu().numpy())
            current_batch += list(elements_chose.cpu().numpy())
            if len(current_batch) == self.batch_size:
                self.sample_used[current_batch] += 1
                self.batch_idx.append(current_batch)
                current_batch = []
        elements_ndx = np.array(elements_ndx.cpu(), dtype=np.int32)
        self.elements_ndx = elements_ndx.astype(np.uint16)
        t1 = time.time()
        print('generate batches done!')
        print('batchsampler5_3 time cost: ', t1 - t0)

class BatchSampler5_4(Sampler):
            
    def __init__(self, 
                 dataset: Dataset,
                 num_k: int,
                 batch_size: int,
                 start_epoch: int = 0):

        self.batch_size = batch_size
        self.dataset = dataset
        self.k = num_k  
        self.batch_idx = []     # Index of elements in each batch (re-generated every epoch)
        self.epoch_ctr = start_epoch
        self.dataset_length = len(self.dataset)
        self.sample_used = np.zeros(len(self.dataset), dtype=np.int32)
        assert batch_size % (num_k + 1) == 0, 'batch_size must be divisible by num_k + 1'
    
    def __iter__(self):
        # Re-generate batches every epoch
        self.generate_batches(self.epoch_ctr)
        self.epoch_ctr += 1
        for batch in self.batch_idx:
            yield batch

    def __len__(self):
        return len(self.batch_idx)

    def generate_batches(self, epoch_ctr):
        # suppose the traversal_idxs is sorted by timestamp
        self.batch_idx = []
        t0 = time.time()
        unused_elements_ndx = set(i for i in range(self.dataset_length))
        used_once_elements_ndx = set()
        used_twice_elements_ndx = set()
        current_batch = []
        while True:
            if len(current_batch) >= self.batch_size:
                self.sample_used[current_batch] += 1
                self.batch_idx.append(current_batch)
                current_batch = []
            if len(unused_elements_ndx) == 0:
                # last current_batch may be dropped
                break
            anchor_element = unused_elements_ndx.pop() # realy random ?
            current_batch.append(anchor_element)
            level_positive_elements = self.dataset.get_level_positives_v2(anchor_element)
            for i in range(self.k):
                curr_level_positive_elements = level_positive_elements[i]
                curr_level_unused_positive_elements = curr_level_positive_elements & unused_elements_ndx
                if curr_level_unused_positive_elements:
                    curr_level_choose_element = curr_level_unused_positive_elements.pop()
                    current_batch.append(curr_level_choose_element)
                    unused_elements_ndx.discard(curr_level_choose_element)
                    used_once_elements_ndx.add(curr_level_choose_element)
                    continue
                curr_level_used_once_positive_elements = curr_level_positive_elements & used_once_elements_ndx
                if curr_level_used_once_positive_elements:
                    curr_level_choose_element = curr_level_used_once_positive_elements.pop()
                    current_batch.append(curr_level_choose_element)
                    used_once_elements_ndx.discard(curr_level_choose_element)
                    used_twice_elements_ndx.add(curr_level_choose_element)
                    continue
                curr_level_used_twice_positive_elements = curr_level_positive_elements & used_twice_elements_ndx
                if curr_level_used_twice_positive_elements:
                    curr_level_choose_element = curr_level_used_twice_positive_elements.pop()
                    current_batch.append(curr_level_choose_element)
                    used_twice_elements_ndx.discard(curr_level_choose_element)
                    continue
                raise ValueError('redesign the sampling strategy')

        t1 = time.time()
        print('generate batches done!')
        print('batchsampler5_4 time cost: ', t1 - t0)

class BatchSampler5_5(Sampler):
    # use numpy's set operation
    def __init__(self, 
                 dataset: Dataset,
                 num_k: int,
                 batch_size: int,
                 start_epoch: int = 0):

        self.batch_size = batch_size
        self.dataset = dataset
        self.k = num_k  
        self.batch_idx = []     # Index of elements in each batch (re-generated every epoch)
        self.epoch_ctr = start_epoch
        self.dataset_length = len(self.dataset)
        self.sample_used = np.zeros(len(self.dataset), dtype=np.int32)
        assert batch_size % (num_k + 1) == 0, 'batch_size must be divisible by num_k + 1'
    
    def __iter__(self):
        # Re-generate batches every epoch
        self.generate_batches(self.epoch_ctr)
        self.epoch_ctr += 1
        for batch in self.batch_idx:
            yield batch

    def __len__(self):
        return len(self.batch_idx)

    def generate_batches(self, epoch_ctr):
        # suppose the traversal_idxs is sorted by timestamp
        self.batch_idx = []
        t0 = time.time()
        unused_elements_ndx = np.arange(0, self.dataset_length, dtype=np.int32)
        used_once_elements_ndx = np.array([], dtype=np.int32)
        used_twice_elements_ndx = np.array([], dtype=np.int32)
        current_batch = []
        while True:
            if len(current_batch) >= self.batch_size:
                self.sample_used[current_batch] += 1
                self.batch_idx.append(current_batch)
                current_batch = []
            if len(unused_elements_ndx) == 0:
                # last current_batch may be dropped
                break
            anchor_element_indice = np.random.randint(0, len(unused_elements_ndx), size=1, dtype=np.int32)
            anchor_element = unused_elements_ndx[anchor_element_indice]
            unused_elements_ndx = np.delete(unused_elements_ndx, anchor_element_indice)
            current_batch.append(int(anchor_element))
            level_positive_elements = self.dataset.get_level_positives_v3(int(anchor_element))
            for i in range(self.k):
                curr_level_positive_elements = level_positive_elements[i]
                curr_level_unused_positive_elements = np.intersect1d(curr_level_positive_elements, unused_elements_ndx)
                if len(curr_level_unused_positive_elements) > 0:
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_unused_positive_elements), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_unused_positive_elements[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    unused_elements_ndx = np.delete(unused_elements_ndx, np.argwhere(unused_elements_ndx == curr_level_choose_element))
                    used_once_elements_ndx = np.append(used_once_elements_ndx, curr_level_choose_element)
                    continue
                curr_level_used_once_positive_elements = np.intersect1d(curr_level_positive_elements, used_once_elements_ndx)
                if len(curr_level_used_once_positive_elements) > 0:
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_used_once_positive_elements), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_used_once_positive_elements[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    used_once_elements_ndx = np.delete(used_once_elements_ndx, np.argwhere(used_once_elements_ndx == curr_level_choose_element))
                    used_twice_elements_ndx = np.append(used_twice_elements_ndx, curr_level_choose_element)
                    continue
                curr_level_used_twice_positive_elements = np.intersect1d(curr_level_positive_elements, used_twice_elements_ndx)
                if len(curr_level_used_twice_positive_elements) > 0:
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_used_twice_positive_elements), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_used_twice_positive_elements[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    used_twice_elements_ndx = np.delete(used_twice_elements_ndx, np.argwhere(used_twice_elements_ndx == curr_level_choose_element))
                    continue
                raise ValueError('redesign the sampling strategy')

        t1 = time.time()
        print('generate batches done!')
        print('batchsampler5_5 time cost: ', t1 - t0)

class BatchSampler5_6(Sampler):
    # use numpy's mask
    def __init__(self, 
                 dataset: Dataset,
                 num_k: int,
                 batch_size: int,
                 start_epoch: int = 0):

        self.batch_size = batch_size
        self.dataset = dataset
        self.k = num_k  
        self.batch_idx = []     # Index of elements in each batch (re-generated every epoch)
        self.epoch_ctr = start_epoch
        self.dataset_length = len(self.dataset)
        self.sample_used = np.zeros(len(self.dataset), dtype=np.int32)
        assert batch_size % (num_k + 1) == 0, 'batch_size must be divisible by num_k + 1'
    
    def __iter__(self):
        # Re-generate batches every epoch
        self.generate_batches(self.epoch_ctr)
        self.epoch_ctr += 1
        for batch in self.batch_idx:
            yield batch

    def __len__(self):
        return len(self.batch_idx)

    def generate_batches(self, epoch_ctr):
        # suppose the traversal_idxs is sorted by timestamp
        self.batch_idx = []
        t0 = time.time()
        unused_elements_ndx = np.ones(self.dataset_length, dtype=np.bool_)
        used_once_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
        used_twice_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
        current_batch = []
        device = torch.cuda.current_device()
        self.dataset.UTM_coord_tensor = self.dataset.UTM_coord_tensor.to(device)
        while True:
            if len(current_batch) >= self.batch_size:
                self.sample_used[current_batch] += 1
                self.batch_idx.append(current_batch)
                current_batch = []
            if np.count_nonzero(unused_elements_ndx) == 0:
                # last current_batch may be dropped
                break
            unused_elements = np.nonzero(unused_elements_ndx)[0]
            unused_element_indice = np.random.randint(0, len(unused_elements), size=1, dtype=np.int32)
            anchor_element = unused_elements[unused_element_indice]
            unused_elements_ndx[anchor_element] = False
            used_once_elements_ndx[anchor_element] = True
            current_batch.append(int(anchor_element))
            level_positive_elements = self.dataset.get_level_positives_v6(int(anchor_element))
            for i in range(self.k):
                curr_level_positive_elements = level_positive_elements[i]
                curr_level_unused_positive_elements = curr_level_positive_elements & unused_elements_ndx
                if np.count_nonzero(curr_level_unused_positive_elements) > 0:
                    curr_level_unused_positive_elements_indices = np.nonzero(curr_level_unused_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_unused_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_unused_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    unused_elements_ndx[curr_level_choose_element] = False
                    used_once_elements_ndx[curr_level_choose_element] = True
                    continue
                curr_level_used_once_positive_elements = curr_level_positive_elements & used_once_elements_ndx
                if np.count_nonzero(curr_level_used_once_positive_elements) > 0:
                    curr_level_used_once_positive_elements_indices = np.nonzero(curr_level_used_once_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_used_once_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_used_once_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    used_once_elements_ndx[curr_level_choose_element] = False
                    used_twice_elements_ndx[curr_level_choose_element] = True
                    continue
                curr_level_used_twice_positive_elements = curr_level_positive_elements & used_twice_elements_ndx
                if np.count_nonzero(curr_level_used_twice_positive_elements) > 0:
                    curr_level_used_twice_positive_elements_indices = np.nonzero(curr_level_used_twice_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_used_twice_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_used_twice_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    used_twice_elements_ndx[curr_level_choose_element] = False
                    continue
                raise ValueError('redesign the positive distance list')

        self.dataset.UTM_coord_tensor = self.dataset.UTM_coord_tensor.cpu()
        t1 = time.time()
        print('generate batches done!')
        print('batchsampler5_6 time cost: ', t1 - t0)
        

class BatchSampler5_7(Sampler):
    # use torch's mask
    def __init__(self, 
                 dataset: Dataset,
                 num_k: int,
                 batch_size: int,
                 start_epoch: int = 0):

        self.batch_size = batch_size
        self.dataset = dataset
        self.k = num_k  
        self.batch_idx = []     # Index of elements in each batch (re-generated every epoch)
        self.epoch_ctr = start_epoch
        self.dataset_length = len(self.dataset)
        self.sample_used = np.zeros(len(self.dataset), dtype=np.int32)
        assert batch_size % (num_k + 1) == 0, 'batch_size must be divisible by num_k + 1'
    
    def __iter__(self):
        # Re-generate batches every epoch
        self.generate_batches(self.epoch_ctr)
        self.epoch_ctr += 1
        for batch in self.batch_idx:
            yield batch

    def __len__(self):
        return len(self.batch_idx)

    def generate_batches(self, epoch_ctr):
        # suppose the traversal_idxs is sorted by timestamp
        self.batch_idx = []
        t0 = time.time()
        device = torch.cuda.current_device()
        unused_elements_ndx = torch.ones(self.dataset_length, dtype=torch.bool, device=device)
        used_once_elements_ndx = torch.zeros(self.dataset_length, dtype=torch.bool, device=device)
        used_twice_elements_ndx = torch.zeros(self.dataset_length, dtype=torch.bool, device=device)
        current_batch = []
        while True:
            if len(current_batch) >= self.batch_size:
                self.sample_used[current_batch] += 1
                self.batch_idx.append(current_batch)
                current_batch = []
            if torch.count_nonzero(unused_elements_ndx) == 0:
                # last current_batch may be dropped
                break
            unused_elements = torch.nonzero(unused_elements_ndx).squeeze(-1)
            unused_element_indice = (torch.randint(0, len(unused_elements), size=(1,), dtype=torch.int32))[0]
            anchor_element = unused_elements[unused_element_indice.type(torch.int64)]
            unused_elements_ndx[anchor_element] = False
            current_batch.append(int(anchor_element.cpu()))
            level_positive_elements = self.dataset.get_level_positives_v5(int(anchor_element.cpu()))
            for i in range(self.k):
                curr_level_positive_elements = level_positive_elements[i]
                curr_level_unused_positive_elements = curr_level_positive_elements & unused_elements_ndx
                if torch.count_nonzero(curr_level_unused_positive_elements) > 0:
                    curr_level_unused_positive_elements_indices = torch.nonzero(curr_level_unused_positive_elements).squeeze(-1)
                    curr_level_choose_element_indice = (torch.randint(0, len(curr_level_unused_positive_elements_indices), size=(1,), dtype=torch.int32))[0]
                    curr_level_choose_element = curr_level_unused_positive_elements_indices[curr_level_choose_element_indice.type(torch.int64)]
                    current_batch.append(int(curr_level_choose_element.cpu()))
                    unused_elements_ndx[curr_level_choose_element] = False
                    used_once_elements_ndx[curr_level_choose_element] = True
                    continue
                curr_level_used_once_positive_elements = curr_level_positive_elements & used_once_elements_ndx
                if torch.count_nonzero(curr_level_used_once_positive_elements) > 0:
                    curr_level_used_once_positive_elements_indices = torch.nonzero(curr_level_used_once_positive_elements).squeeze(-1)
                    curr_level_choose_element_indice = (torch.randint(0, len(curr_level_used_once_positive_elements_indices), size=(1,), dtype=torch.int32))[0]
                    curr_level_choose_element = curr_level_used_once_positive_elements_indices[curr_level_choose_element_indice.type(torch.int64)]
                    current_batch.append((curr_level_choose_element.cpu()))
                    used_once_elements_ndx[curr_level_choose_element] = False
                    used_twice_elements_ndx[curr_level_choose_element] = True
                    continue
                curr_level_used_twice_positive_elements = curr_level_positive_elements & used_twice_elements_ndx
                if torch.count_nonzero(curr_level_used_twice_positive_elements) > 0:
                    curr_level_used_twice_positive_elements_indices = torch.nonzero(curr_level_used_twice_positive_elements).squeeze(-1)
                    curr_level_choose_element_indice = (torch.randint(0, len(curr_level_used_twice_positive_elements_indices), size=(1,), dtype=torch.int32))[0]
                    curr_level_choose_element = curr_level_used_twice_positive_elements_indices[curr_level_choose_element_indice.type(torch.int64)]
                    current_batch.append(int(curr_level_choose_element.cpu()))
                    used_twice_elements_ndx[curr_level_choose_element] = False
                    continue
                raise ValueError('redesign the sampling strategy')

        t1 = time.time()
        print('generate batches done!')
        print('batchsampler5_7 time cost: ', t1 - t0)

class BatchSampler5_8(Sampler):
    # use numpy's mask
    def __init__(self, 
                 dataset: Dataset,
                 num_k: int,
                 batch_size: int,
                 start_epoch: int = 0,
                 useout_times: int = 1,
                 interval_num: int = 100):

        self.batch_size = batch_size
        self.dataset = dataset
        self.k = num_k  
        self.batch_idx = []     # Index of elements in each batch (re-generated every epoch)
        self.epoch_ctr = start_epoch
        self.dataset_length = len(self.dataset)
        self.sample_used = np.zeros(len(self.dataset), dtype=np.int32)
        assert batch_size % (num_k + 1) == 0, 'batch_size must be divisible by num_k + 1'
        self.useout_times = useout_times # max num == 3
        self.interval_num = interval_num
    
    def __iter__(self):
        # Re-generate batches every epoch
        self.generate_batches(self.epoch_ctr)
        self.epoch_ctr += 1
        for batch in self.batch_idx:
            yield batch

    def __len__(self):
        return len(self.batch_idx)

    def generate_batches(self, epoch_ctr):
        # suppose the traversal_idxs is sorted by timestamp
        self.batch_idx = []
        t0 = time.time()
        unused_elements_ndx = np.ones(self.dataset_length, dtype=np.bool_)
        used_once_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
        used_twice_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
        current_batch = []
        device = torch.cuda.current_device()
        self.dataset.UTM_coord_tensor = self.dataset.UTM_coord_tensor.to(device)
        curr_time = 1
        while True:
            if len(current_batch) >= self.batch_size:
                self.sample_used[current_batch] += 1
                self.batch_idx.append(current_batch)
                current_batch = []
            if np.count_nonzero(unused_elements_ndx) == 0:
                if curr_time == self.useout_times:
                    break
                else:
                    curr_time += 1
                    unused_elements_ndx = ~used_once_elements_ndx
                    used_once_elements_ndx = used_twice_elements_ndx
                    used_twice_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
                    continue
            unused_elements = np.nonzero(unused_elements_ndx)[0]
            unused_element_indice = np.random.randint(0, len(unused_elements), size=1, dtype=np.int32)
            anchor_element = unused_elements[unused_element_indice]
            unused_elements_ndx[anchor_element] = False
            used_once_elements_ndx[anchor_element] = True
            current_batch.append(int(anchor_element))
            level_positive_elements = self.dataset.get_level_positives_v6(int(anchor_element), self.interval_num)
            for i in range(self.k):
                curr_level_positive_elements = level_positive_elements[i]
                curr_level_unused_positive_elements = curr_level_positive_elements & unused_elements_ndx
                if np.count_nonzero(curr_level_unused_positive_elements) > 0:
                    curr_level_unused_positive_elements_indices = np.nonzero(curr_level_unused_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_unused_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_unused_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    unused_elements_ndx[curr_level_choose_element] = False
                    used_once_elements_ndx[curr_level_choose_element] = True
                    continue
                curr_level_used_once_positive_elements = curr_level_positive_elements & used_once_elements_ndx
                if np.count_nonzero(curr_level_used_once_positive_elements) > 0:
                    curr_level_used_once_positive_elements_indices = np.nonzero(curr_level_used_once_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_used_once_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_used_once_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    used_once_elements_ndx[curr_level_choose_element] = False
                    used_twice_elements_ndx[curr_level_choose_element] = True
                    continue
                curr_level_used_twice_positive_elements = curr_level_positive_elements & used_twice_elements_ndx
                if np.count_nonzero(curr_level_used_twice_positive_elements) > 0:
                    curr_level_used_twice_positive_elements_indices = np.nonzero(curr_level_used_twice_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_used_twice_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_used_twice_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    used_twice_elements_ndx[curr_level_choose_element] = False
                    continue
                raise ValueError('redesign the positive distance list')

        self.dataset.UTM_coord_tensor = self.dataset.UTM_coord_tensor.cpu()
        t1 = time.time()
        print('generate batches done!')
        print('batchsampler5_6 time cost: ', t1 - t0)


class BatchSampler5_9(Sampler):
    # use numpy's mask
    def __init__(self, 
                 dataset: Dataset,
                 num_k: int,
                 batch_size: int,
                 start_epoch: int = 0,
                 useout_times: int = 1):

        self.batch_size = batch_size
        self.dataset = dataset
        self.k = num_k  
        self.batch_idx = []     # Index of elements in each batch (re-generated every epoch)
        self.epoch_ctr = start_epoch
        self.dataset_length = len(self.dataset)
        self.sample_used = np.zeros(len(self.dataset), dtype=np.int32)
        assert batch_size % (num_k + 1) == 0, 'batch_size must be divisible by num_k + 1'
        self.useout_times = useout_times # max num == 3
    
    def __iter__(self):
        # Re-generate batches every epoch
        self.generate_batches(self.epoch_ctr)
        self.epoch_ctr += 1
        for batch in self.batch_idx:
            yield batch

    def __len__(self):
        return len(self.batch_idx)

    def generate_batches(self, epoch_ctr):
        # suppose the traversal_idxs is sorted by timestamp
        self.batch_idx = []
        t0 = time.time()
        unused_elements_ndx = np.ones(self.dataset_length, dtype=np.bool_)
        used_once_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
        used_twice_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
        current_batch = []
        curr_time = 1
        while True:
            if len(current_batch) >= self.batch_size:
                self.sample_used[current_batch] += 1
                self.batch_idx.append(current_batch)
                current_batch = []
            if np.count_nonzero(unused_elements_ndx) == 0:
                if curr_time == self.useout_times:
                    break
                else:
                    curr_time += 1
                    unused_elements_ndx = ~used_once_elements_ndx
                    used_once_elements_ndx = used_twice_elements_ndx
                    used_twice_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
                    continue
            unused_elements = np.nonzero(unused_elements_ndx)[0]
            unused_element_indice = np.random.randint(0, len(unused_elements), size=1, dtype=np.int32)
            anchor_element = unused_elements[unused_element_indice]
            unused_elements_ndx[anchor_element] = False
            used_once_elements_ndx[anchor_element] = True
            current_batch.append(int(anchor_element))
            level_positive_elements = self.dataset.get_level_positives_v7(int(anchor_element), self.k)
            for i in range(self.k):
                curr_level_positive_elements = level_positive_elements[i]
                curr_level_unused_positive_elements = curr_level_positive_elements & unused_elements_ndx
                if np.count_nonzero(curr_level_unused_positive_elements) > 0:
                    curr_level_unused_positive_elements_indices = np.nonzero(curr_level_unused_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_unused_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_unused_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    unused_elements_ndx[curr_level_choose_element] = False
                    used_once_elements_ndx[curr_level_choose_element] = True
                    continue
                curr_level_used_once_positive_elements = curr_level_positive_elements & used_once_elements_ndx
                if np.count_nonzero(curr_level_used_once_positive_elements) > 0:
                    curr_level_used_once_positive_elements_indices = np.nonzero(curr_level_used_once_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_used_once_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_used_once_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    used_once_elements_ndx[curr_level_choose_element] = False
                    used_twice_elements_ndx[curr_level_choose_element] = True
                    continue
                curr_level_used_twice_positive_elements = curr_level_positive_elements & used_twice_elements_ndx
                if np.count_nonzero(curr_level_used_twice_positive_elements) > 0:
                    curr_level_used_twice_positive_elements_indices = np.nonzero(curr_level_used_twice_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_used_twice_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_used_twice_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    used_twice_elements_ndx[curr_level_choose_element] = False
                    continue
                raise ValueError('redesign the positive distance list')

        t1 = time.time()
        print('generate batches done!')
        print('batchsampler5_9 time cost: ', t1 - t0)


# for 
class BatchSampler5_10(Sampler):
    # use numpy's mask
    def __init__(self, 
                 dataset: Dataset,
                 num_k: int,
                 batch_size: int,
                 start_epoch: int = 0,
                 useout_times: int = 1):

        self.batch_size = batch_size
        self.dataset = dataset
        self.k = num_k  
        self.batch_idx = []     # Index of elements in each batch (re-generated every epoch)
        self.epoch_ctr = start_epoch
        self.dataset_length = len(self.dataset)
        self.sample_used = np.zeros(len(self.dataset), dtype=np.int32)
        assert batch_size % (num_k + 1) == 0, 'batch_size must be divisible by num_k + 1'
        self.useout_times = useout_times # max num == 3
    
    def __iter__(self):
        # Re-generate batches every epoch
        self.generate_batches(self.epoch_ctr)
        self.epoch_ctr += 1
        for batch in self.batch_idx:
            yield batch

    def __len__(self):
        return len(self.batch_idx)

    def generate_batches(self, epoch_ctr):
        # suppose the traversal_idxs is sorted by timestamp
        self.batch_idx = []
        t0 = time.time()
        unused_elements_ndx = np.ones(self.dataset_length, dtype=np.bool_)
        used_once_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
        used_twice_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
        current_batch = []
        device = torch.cuda.current_device()
        self.dataset.pos_vec_vet_coords_tensor = self.dataset.pos_vec_vet_coords_tensor.to(device)
        curr_time = 1
        while True:
            if len(current_batch) >= self.batch_size:
                self.sample_used[current_batch] += 1
                self.batch_idx.append(current_batch)
                current_batch = []
            if np.count_nonzero(unused_elements_ndx) == 0:
                if curr_time == self.useout_times:
                    break
                else:
                    curr_time += 1
                    unused_elements_ndx = ~used_once_elements_ndx
                    used_once_elements_ndx = used_twice_elements_ndx
                    used_twice_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
                    continue
            unused_elements = np.nonzero(unused_elements_ndx)[0]
            unused_element_indice = np.random.randint(0, len(unused_elements), size=1, dtype=np.int32)
            anchor_element = unused_elements[unused_element_indice]
            unused_elements_ndx[anchor_element] = False
            used_once_elements_ndx[anchor_element] = True
            current_batch.append(int(anchor_element))
            level_positive_elements = self.dataset.get_level_positives_v8(int(anchor_element))
            for i in range(self.k):
                curr_level_positive_elements = level_positive_elements
                curr_level_unused_positive_elements = curr_level_positive_elements & unused_elements_ndx
                if np.count_nonzero(curr_level_unused_positive_elements) > 0:
                    curr_level_unused_positive_elements_indices = np.nonzero(curr_level_unused_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_unused_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_unused_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    unused_elements_ndx[curr_level_choose_element] = False
                    used_once_elements_ndx[curr_level_choose_element] = True
                    continue
                curr_level_used_once_positive_elements = curr_level_positive_elements & used_once_elements_ndx
                if np.count_nonzero(curr_level_used_once_positive_elements) > 0:
                    curr_level_used_once_positive_elements_indices = np.nonzero(curr_level_used_once_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_used_once_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_used_once_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    used_once_elements_ndx[curr_level_choose_element] = False
                    used_twice_elements_ndx[curr_level_choose_element] = True
                    continue
                curr_level_used_twice_positive_elements = curr_level_positive_elements & used_twice_elements_ndx
                if np.count_nonzero(curr_level_used_twice_positive_elements) > 0:
                    curr_level_used_twice_positive_elements_indices = np.nonzero(curr_level_used_twice_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_used_twice_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_used_twice_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    used_twice_elements_ndx[curr_level_choose_element] = False
                    continue
                raise ValueError('redesign the positive distance list')

        t1 = time.time()
        self.dataset.pos_vec_vet_coords_tensor = self.dataset.pos_vec_vet_coords_tensor.to('cpu')
        print('generate batches done!')
        print('batchsampler5_10 time cost: ', t1 - t0)


class BatchSampler5_11(Sampler):
    # use numpy's mask
    def __init__(self, 
                 dataset: Dataset,
                 num_k: int,
                 batch_size: int,
                 start_epoch: int = 0,
                 useout_times: int = 1,
                 interval_num: int = 100):

        self.batch_size = batch_size
        self.dataset = dataset
        self.k = num_k  
        self.batch_idx = []     # Index of elements in each batch (re-generated every epoch)
        self.epoch_ctr = start_epoch
        self.dataset_length = len(self.dataset)
        self.sample_used = np.zeros(len(self.dataset), dtype=np.int32)
        assert batch_size % (num_k + 1) == 0, 'batch_size must be divisible by num_k + 1'
        self.useout_times = useout_times # max num == 3
        self.interval_num = interval_num
    
    def __iter__(self):
        # Re-generate batches every epoch
        self.generate_batches(self.epoch_ctr)
        self.epoch_ctr += 1
        for batch in self.batch_idx:
            yield batch

    def __len__(self):
        return len(self.batch_idx)

    def generate_batches(self, epoch_ctr):
        # suppose the traversal_idxs is sorted by timestamp
        self.batch_idx = []
        t0 = time.time()
        unused_elements_ndx = np.ones(self.dataset_length, dtype=np.bool_)
        used_once_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
        used_twice_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
        used_three_time_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
        used_four_time_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
        used_five_time_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
        used_six_time_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
        used_seven_time_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
        used_eight_time_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
        used_nine_time_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
        used_ten_time_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
        current_batch = []
        device = torch.cuda.current_device()
        self.dataset.UTM_coord_tensor = self.dataset.UTM_coord_tensor.to(device)
        curr_time = 1
        while True:
            if len(current_batch) >= self.batch_size:
                self.sample_used[current_batch] += 1
                self.batch_idx.append(current_batch)
                current_batch = []
            if np.count_nonzero(unused_elements_ndx) == 0:
                if curr_time == self.useout_times:
                    break
                else:
                    curr_time += 1
                    unused_elements_ndx = ~used_once_elements_ndx
                    used_once_elements_ndx = used_twice_elements_ndx
                    used_twice_elements_ndx = used_three_time_elements_ndx
                    used_three_time_elements_ndx = used_four_time_elements_ndx
                    used_four_time_elements_ndx = used_five_time_elements_ndx
                    used_five_time_elements_ndx = used_six_time_elements_ndx
                    used_six_time_elements_ndx = used_seven_time_elements_ndx
                    used_seven_time_elements_ndx = used_eight_time_elements_ndx
                    used_eight_time_elements_ndx = used_nine_time_elements_ndx
                    used_nine_time_elements_ndx = used_ten_time_elements_ndx
                    used_ten_time_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
                    continue
            unused_elements = np.nonzero(unused_elements_ndx)[0]
            unused_element_indice = np.random.randint(0, len(unused_elements), size=1, dtype=np.int32)
            anchor_element = unused_elements[unused_element_indice]
            unused_elements_ndx[anchor_element] = False
            used_once_elements_ndx[anchor_element] = True
            current_batch.append(int(anchor_element))
            level_positive_elements = self.dataset.get_level_positives_v9(int(anchor_element), self.interval_num)
            for i in range(self.k):
                curr_level_positive_elements = level_positive_elements[i]
                curr_level_unused_positive_elements = curr_level_positive_elements & unused_elements_ndx
                if np.count_nonzero(curr_level_unused_positive_elements) > 0:
                    curr_level_unused_positive_elements_indices = np.nonzero(curr_level_unused_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_unused_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_unused_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    unused_elements_ndx[curr_level_choose_element] = False
                    used_once_elements_ndx[curr_level_choose_element] = True
                    continue
                curr_level_used_once_positive_elements = curr_level_positive_elements & used_once_elements_ndx
                if np.count_nonzero(curr_level_used_once_positive_elements) > 0:
                    curr_level_used_once_positive_elements_indices = np.nonzero(curr_level_used_once_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_used_once_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_used_once_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    used_once_elements_ndx[curr_level_choose_element] = False
                    used_twice_elements_ndx[curr_level_choose_element] = True
                    continue
                curr_level_used_twice_positive_elements = curr_level_positive_elements & used_twice_elements_ndx
                if np.count_nonzero(curr_level_used_twice_positive_elements) > 0:
                    curr_level_used_twice_positive_elements_indices = np.nonzero(curr_level_used_twice_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_used_twice_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_used_twice_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    used_twice_elements_ndx[curr_level_choose_element] = False
                    used_three_time_elements_ndx[curr_level_choose_element] = True
                    continue
                curr_level_used_three_positive_elements = curr_level_positive_elements & used_three_time_elements_ndx
                if np.count_nonzero(curr_level_used_three_positive_elements) > 0:
                    curr_level_used_three_positive_elements_indices = np.nonzero(curr_level_used_three_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_used_three_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_used_three_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    used_three_time_elements_ndx[curr_level_choose_element] = False
                    used_four_time_elements_ndx[curr_level_choose_element] = True
                    continue
                curr_level_used_four_positive_elements = curr_level_positive_elements & used_four_time_elements_ndx
                if np.count_nonzero(curr_level_used_four_positive_elements) > 0:
                    curr_level_used_four_positive_elements_indices = np.nonzero(curr_level_used_four_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_used_four_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_used_four_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    used_four_time_elements_ndx[curr_level_choose_element] = False
                    used_five_time_elements_ndx[curr_level_choose_element] = True
                    continue
                curr_level_used_five_positive_elements = curr_level_positive_elements & used_five_time_elements_ndx
                if np.count_nonzero(curr_level_used_five_positive_elements) > 0:
                    curr_level_used_five_positive_elements_indices = np.nonzero(curr_level_used_five_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_used_five_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_used_five_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    used_five_time_elements_ndx[curr_level_choose_element] = False
                    used_six_time_elements_ndx[curr_level_choose_element] = True
                    continue
                curr_level_used_six_positive_elements = curr_level_positive_elements & used_six_time_elements_ndx
                if np.count_nonzero(curr_level_used_six_positive_elements) > 0:
                    curr_level_used_six_positive_elements_indices = np.nonzero(curr_level_used_six_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_used_six_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_used_six_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    used_six_time_elements_ndx[curr_level_choose_element] = False
                    used_seven_time_elements_ndx[curr_level_choose_element] = True
                    continue
                curr_level_used_seven_positive_elements = curr_level_positive_elements & used_seven_time_elements_ndx
                if np.count_nonzero(curr_level_used_seven_positive_elements) > 0:
                    curr_level_used_seven_positive_elements_indices = np.nonzero(curr_level_used_seven_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_used_seven_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_used_seven_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    used_seven_time_elements_ndx[curr_level_choose_element] = False
                    used_eight_time_elements_ndx[curr_level_choose_element] = True
                    continue
                curr_level_used_eight_positive_elements = curr_level_positive_elements & used_eight_time_elements_ndx
                if np.count_nonzero(curr_level_used_eight_positive_elements) > 0:
                    curr_level_used_eight_positive_elements_indices = np.nonzero(curr_level_used_eight_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_used_eight_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_used_eight_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    used_eight_time_elements_ndx[curr_level_choose_element] = False
                    used_nine_time_elements_ndx[curr_level_choose_element] = True
                    continue
                curr_level_used_nine_positive_elements = curr_level_positive_elements & used_nine_time_elements_ndx
                if np.count_nonzero(curr_level_used_nine_positive_elements) > 0:
                    curr_level_used_nine_positive_elements_indices = np.nonzero(curr_level_used_nine_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_used_nine_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_used_nine_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    used_nine_time_elements_ndx[curr_level_choose_element] = False
                    used_ten_time_elements_ndx[curr_level_choose_element] = True
                    continue
                curr_level_used_ten_positive_elements = curr_level_positive_elements & used_ten_time_elements_ndx
                if np.count_nonzero(curr_level_used_ten_positive_elements) > 0:
                    curr_level_used_ten_positive_elements_indices = np.nonzero(curr_level_used_ten_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_used_ten_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_used_ten_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    used_ten_time_elements_ndx[curr_level_choose_element] = False
                    continue
                raise ValueError('redesign the positive distance list')

        self.dataset.UTM_coord_tensor = self.dataset.UTM_coord_tensor.cpu()
        t1 = time.time()
        print('generate batches done!')
        print('batchsampler5_11 time cost: ', t1 - t0)


class BatchSampler5_12(Sampler):
    # use numpy's mask
    def __init__(self, 
                 dataset: Dataset,
                 num_k: int,
                 batch_size: int,
                 start_epoch: int = 0,
                 useout_times: int = 1,
                 interval_num: int = 100):

        self.batch_size = batch_size
        self.dataset = dataset
        self.k = num_k  
        self.batch_idx = []     # Index of elements in each batch (re-generated every epoch)
        self.epoch_ctr = start_epoch
        self.dataset_length = len(self.dataset)
        self.sample_used = np.zeros(len(self.dataset), dtype=np.int32)
        assert batch_size % (num_k + 1) == 0, 'batch_size must be divisible by num_k + 1'
        self.useout_times = useout_times # max num == 3
        self.interval_num = interval_num
    
    def __iter__(self):
        # Re-generate batches every epoch
        self.generate_batches(self.epoch_ctr)
        self.epoch_ctr += 1
        for batch in self.batch_idx:
            yield batch

    def __len__(self):
        return len(self.batch_idx)

    def generate_batches(self, epoch_ctr):
        # suppose the traversal_idxs is sorted by timestamp
        self.batch_idx = []
        t0 = time.time()
        unused_elements_ndx = np.ones(self.dataset_length, dtype=np.bool_)
        used_once_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
        used_twice_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
        current_batch = []
        device = torch.cuda.current_device()
        self.dataset.UTM_coord_tensor = self.dataset.UTM_coord_tensor.to(device)
        curr_time = 1
        while True:
            if len(current_batch) >= self.batch_size:
                self.sample_used[current_batch] += 1
                self.batch_idx.append(current_batch)
                current_batch = []
            if np.count_nonzero(unused_elements_ndx) == 0:
                if curr_time == self.useout_times:
                    break
                else:
                    curr_time += 1
                    unused_elements_ndx = ~used_once_elements_ndx
                    used_once_elements_ndx = used_twice_elements_ndx
                    used_twice_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
                    continue
            unused_elements = np.nonzero(unused_elements_ndx)[0]
            unused_element_indice = np.random.randint(0, len(unused_elements), size=1, dtype=np.int32)
            anchor_element = unused_elements[unused_element_indice]
            unused_elements_ndx[anchor_element] = False
            used_once_elements_ndx[anchor_element] = True
            current_batch.append(int(anchor_element))
            level_positive_elements = self.dataset.get_level_positives_v10(int(anchor_element), self.interval_num)
            for i in range(self.k):
                curr_level_positive_elements = level_positive_elements[i]
                curr_level_unused_positive_elements = curr_level_positive_elements & unused_elements_ndx
                if np.count_nonzero(curr_level_unused_positive_elements) > 0:
                    curr_level_unused_positive_elements_indices = np.nonzero(curr_level_unused_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_unused_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_unused_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    unused_elements_ndx[curr_level_choose_element] = False
                    used_once_elements_ndx[curr_level_choose_element] = True
                    continue
                curr_level_used_once_positive_elements = curr_level_positive_elements & used_once_elements_ndx
                if np.count_nonzero(curr_level_used_once_positive_elements) > 0:
                    curr_level_used_once_positive_elements_indices = np.nonzero(curr_level_used_once_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_used_once_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_used_once_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    used_once_elements_ndx[curr_level_choose_element] = False
                    used_twice_elements_ndx[curr_level_choose_element] = True
                    continue
                curr_level_used_twice_positive_elements = curr_level_positive_elements & used_twice_elements_ndx
                if np.count_nonzero(curr_level_used_twice_positive_elements) > 0:
                    curr_level_used_twice_positive_elements_indices = np.nonzero(curr_level_used_twice_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_used_twice_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_used_twice_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    used_twice_elements_ndx[curr_level_choose_element] = False
                    continue
                raise ValueError('redesign the positive distance list')

        self.dataset.UTM_coord_tensor = self.dataset.UTM_coord_tensor.cpu()
        t1 = time.time()
        print('generate batches done!')
        print('batchsampler5_11 time cost: ', t1 - t0)




class BatchSampler5_13(Sampler):
    # use numpy's mask
    def __init__(self, 
                 dataset: Dataset,
                 num_k: int,
                 batch_size: int,
                 start_epoch: int = 0,
                 useout_times: int = 1,
                 interval_num: int = 100):

        self.batch_size = batch_size
        self.dataset = dataset
        self.k = num_k  
        self.batch_idx = []     # Index of elements in each batch (re-generated every epoch)
        self.epoch_ctr = start_epoch
        self.dataset_length = len(self.dataset)
        self.sample_used = np.zeros(len(self.dataset), dtype=np.int32)
        assert batch_size % (num_k + 1) == 0, 'batch_size must be divisible by num_k + 1'
        self.useout_times = useout_times # max num == 3
        self.interval_num = interval_num
    
    def __iter__(self):
        # Re-generate batches every epoch
        self.generate_batches(self.epoch_ctr)
        self.epoch_ctr += 1
        for batch in self.batch_idx:
            yield batch

    def __len__(self):
        return len(self.batch_idx)

    def generate_batches(self, epoch_ctr):
        # suppose the traversal_idxs is sorted by timestamp
        self.batch_idx = []
        t0 = time.time()
        unused_elements_ndx = np.ones(self.dataset_length, dtype=np.bool_)
        used_once_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
        used_twice_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
        current_batch = []
        device = torch.cuda.current_device()
        self.dataset.UTM_coord_tensor = self.dataset.UTM_coord_tensor.to(device)
        curr_time = 1
        while True:
            if len(current_batch) >= self.batch_size:
                self.sample_used[current_batch] += 1
                self.batch_idx.append(current_batch)
                current_batch = []
            if np.count_nonzero(unused_elements_ndx) == 0:
                if curr_time == self.useout_times:
                    break
                else:
                    curr_time += 1
                    unused_elements_ndx = ~used_once_elements_ndx
                    used_once_elements_ndx = used_twice_elements_ndx
                    used_twice_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
                    continue
            unused_elements = np.nonzero(unused_elements_ndx)[0]
            unused_element_indice = np.random.randint(0, len(unused_elements), size=1, dtype=np.int32)
            anchor_element = unused_elements[unused_element_indice]
            unused_elements_ndx[anchor_element] = False
            used_once_elements_ndx[anchor_element] = True
            current_batch.append(int(anchor_element))
            level_positive_elements = self.dataset.get_level_positives_v11(int(anchor_element), self.interval_num)
            for i in range(self.k):
                curr_level_positive_elements = level_positive_elements[i]
                curr_level_unused_positive_elements = curr_level_positive_elements & unused_elements_ndx
                if np.count_nonzero(curr_level_unused_positive_elements) > 0:
                    curr_level_unused_positive_elements_indices = np.nonzero(curr_level_unused_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_unused_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_unused_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    unused_elements_ndx[curr_level_choose_element] = False
                    used_once_elements_ndx[curr_level_choose_element] = True
                    continue
                curr_level_used_once_positive_elements = curr_level_positive_elements & used_once_elements_ndx
                if np.count_nonzero(curr_level_used_once_positive_elements) > 0:
                    curr_level_used_once_positive_elements_indices = np.nonzero(curr_level_used_once_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_used_once_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_used_once_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    used_once_elements_ndx[curr_level_choose_element] = False
                    used_twice_elements_ndx[curr_level_choose_element] = True
                    continue
                curr_level_used_twice_positive_elements = curr_level_positive_elements & used_twice_elements_ndx
                if np.count_nonzero(curr_level_used_twice_positive_elements) > 0:
                    curr_level_used_twice_positive_elements_indices = np.nonzero(curr_level_used_twice_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_used_twice_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_used_twice_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    used_twice_elements_ndx[curr_level_choose_element] = False
                    continue
                raise ValueError('redesign the positive distance list')

        self.dataset.UTM_coord_tensor = self.dataset.UTM_coord_tensor.cpu()
        t1 = time.time()
        print('generate batches done!')
        print('batchsampler5_11 time cost: ', t1 - t0)


class BatchSampler5_14(Sampler):
    # use numpy's mask
    def __init__(self, 
                 dataset: Dataset,
                 num_k: int,
                 batch_size: int,
                 start_epoch: int = 0,
                 useout_times: int = 1,
                 interval_num: int = 100):

        self.batch_size = batch_size
        self.dataset = dataset
        self.k = num_k  
        self.batch_idx = []     # Index of elements in each batch (re-generated every epoch)
        self.epoch_ctr = start_epoch
        self.dataset_length = len(self.dataset)
        self.sample_used = np.zeros(len(self.dataset), dtype=np.int32)
        assert batch_size % (num_k + 1) == 0, 'batch_size must be divisible by num_k + 1'
        self.useout_times = useout_times # max num == 3
        self.interval_num = interval_num
    
    def __iter__(self):
        # Re-generate batches every epoch
        self.generate_batches(self.epoch_ctr)
        self.epoch_ctr += 1
        for batch in self.batch_idx:
            yield batch

    def __len__(self):
        return len(self.batch_idx)

    def generate_batches(self, epoch_ctr):
        # suppose the traversal_idxs is sorted by timestamp
        self.batch_idx = []
        t0 = time.time()
        unused_elements_ndx = np.ones(self.dataset_length, dtype=np.bool_)
        used_once_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
        used_twice_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
        used_three_times_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
        used_four_times_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
        used_five_times_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
        used_six_times_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
        used_seven_times_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
        used_eight_times_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
        used_nine_times_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
        used_ten_times_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
        current_batch = []
        curr_time = 1
        while True:
            if len(current_batch) >= self.batch_size:
                self.sample_used[current_batch] += 1
                self.batch_idx.append(current_batch)
                current_batch = []
            if np.count_nonzero(unused_elements_ndx) == 0:
                if curr_time == self.useout_times:
                    break
                else:
                    curr_time += 1
                    unused_elements_ndx = ~used_once_elements_ndx
                    used_once_elements_ndx = used_twice_elements_ndx
                    used_twice_elements_ndx = used_three_times_elements_ndx
                    used_three_times_elements_ndx = used_four_times_elements_ndx
                    used_four_times_elements_ndx = used_five_times_elements_ndx
                    used_five_times_elements_ndx = used_six_times_elements_ndx
                    used_six_times_elements_ndx = used_seven_times_elements_ndx
                    used_seven_times_elements_ndx = used_eight_times_elements_ndx
                    used_eight_times_elements_ndx = used_nine_times_elements_ndx
                    used_nine_times_elements_ndx = used_ten_times_elements_ndx
                    used_ten_times_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
                    continue
            unused_elements = np.nonzero(unused_elements_ndx)[0]
            unused_element_indice = np.random.randint(0, len(unused_elements), size=1, dtype=np.int32)
            anchor_element = unused_elements[unused_element_indice]
            unused_elements_ndx[anchor_element] = False
            used_once_elements_ndx[anchor_element] = True
            current_batch.append(int(anchor_element))
            level_positive_elements = self.dataset.get_level_positives_v12(int(anchor_element), self.interval_num)
            for i in range(self.k):
                curr_level_positive_elements = level_positive_elements[i]
                curr_level_unused_positive_elements = curr_level_positive_elements & unused_elements_ndx
                if np.count_nonzero(curr_level_unused_positive_elements) > 0:
                    curr_level_unused_positive_elements_indices = np.nonzero(curr_level_unused_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_unused_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_unused_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    unused_elements_ndx[curr_level_choose_element] = False
                    used_once_elements_ndx[curr_level_choose_element] = True
                    continue
                curr_level_used_once_positive_elements = curr_level_positive_elements & used_once_elements_ndx
                if np.count_nonzero(curr_level_used_once_positive_elements) > 0:
                    curr_level_used_once_positive_elements_indices = np.nonzero(curr_level_used_once_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_used_once_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_used_once_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    used_once_elements_ndx[curr_level_choose_element] = False
                    used_twice_elements_ndx[curr_level_choose_element] = True
                    continue
                curr_level_used_twice_positive_elements = curr_level_positive_elements & used_twice_elements_ndx
                if np.count_nonzero(curr_level_used_twice_positive_elements) > 0:
                    curr_level_used_twice_positive_elements_indices = np.nonzero(curr_level_used_twice_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_used_twice_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_used_twice_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    used_twice_elements_ndx[curr_level_choose_element] = False
                    used_three_times_elements_ndx[curr_level_choose_element] = True
                    continue
                curr_level_used_three_positive_elements = curr_level_positive_elements & used_three_times_elements_ndx
                if np.count_nonzero(curr_level_used_three_positive_elements) > 0:
                    curr_level_used_three_positive_elements_indices = np.nonzero(curr_level_used_three_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_used_three_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_used_three_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    used_three_times_elements_ndx[curr_level_choose_element] = False
                    used_four_times_elements_ndx[curr_level_choose_element] = True
                    continue
                curr_level_used_four_positive_elements = curr_level_positive_elements & used_four_times_elements_ndx
                if np.count_nonzero(curr_level_used_four_positive_elements) > 0:
                    curr_level_used_four_positive_elements_indices = np.nonzero(curr_level_used_four_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_used_four_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_used_four_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    used_four_times_elements_ndx[curr_level_choose_element] = False
                    used_five_times_elements_ndx[curr_level_choose_element] = True
                    continue
                curr_level_used_five_positive_elements = curr_level_positive_elements & used_five_times_elements_ndx
                if np.count_nonzero(curr_level_used_five_positive_elements) > 0:
                    curr_level_used_five_positive_elements_indices = np.nonzero(curr_level_used_five_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_used_five_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_used_five_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    used_five_times_elements_ndx[curr_level_choose_element] = False
                    used_six_times_elements_ndx[curr_level_choose_element] = True
                    continue
                curr_level_used_six_positive_elements = curr_level_positive_elements & used_six_times_elements_ndx
                if np.count_nonzero(curr_level_used_six_positive_elements) > 0:
                    curr_level_used_six_positive_elements_indices = np.nonzero(curr_level_used_six_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_used_six_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_used_six_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    used_six_times_elements_ndx[curr_level_choose_element] = False
                    used_seven_times_elements_ndx[curr_level_choose_element] = True
                    continue
                curr_level_used_seven_positive_elements = curr_level_positive_elements & used_seven_times_elements_ndx
                if np.count_nonzero(curr_level_used_seven_positive_elements) > 0:
                    curr_level_used_seven_positive_elements_indices = np.nonzero(curr_level_used_seven_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_used_seven_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_used_seven_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    used_seven_times_elements_ndx[curr_level_choose_element] = False
                    used_eight_times_elements_ndx[curr_level_choose_element] = True
                    continue
                curr_level_used_eight_positive_elements = curr_level_positive_elements & used_eight_times_elements_ndx
                if np.count_nonzero(curr_level_used_eight_positive_elements) > 0:
                    curr_level_used_eight_positive_elements_indices = np.nonzero(curr_level_used_eight_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_used_eight_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_used_eight_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    used_eight_times_elements_ndx[curr_level_choose_element] = False
                    used_nine_times_elements_ndx[curr_level_choose_element] = True
                    continue
                curr_level_used_nine_positive_elements = curr_level_positive_elements & used_nine_times_elements_ndx
                if np.count_nonzero(curr_level_used_nine_positive_elements) > 0:
                    curr_level_used_nine_positive_elements_indices = np.nonzero(curr_level_used_nine_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_used_nine_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_used_nine_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    used_nine_times_elements_ndx[curr_level_choose_element] = False
                    used_ten_times_elements_ndx[curr_level_choose_element] = True
                    continue
                curr_level_used_ten_positive_elements = curr_level_positive_elements & used_ten_times_elements_ndx
                if np.count_nonzero(curr_level_used_ten_positive_elements) > 0:
                    curr_level_used_ten_positive_elements_indices = np.nonzero(curr_level_used_ten_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_used_ten_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_used_ten_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    used_ten_times_elements_ndx[curr_level_choose_element] = False
                    continue

                raise ValueError('redesign the positive distance list')

        t1 = time.time()
        print('generate batches done!')
        print('batchsampler5_11 time cost: ', t1 - t0)




class BatchSampler5_15(Sampler):
    # use numpy's mask
    def __init__(self, 
                 dataset: Dataset,
                 num_k: int,
                 batch_size: int,
                 start_epoch: int = 0,
                 useout_times: int = 1):

        self.batch_size = batch_size
        self.dataset = dataset
        self.k = num_k  
        self.batch_idx = []     # Index of elements in each batch (re-generated every epoch)
        self.epoch_ctr = start_epoch
        self.dataset_length = len(self.dataset)
        self.sample_used = np.zeros(len(self.dataset), dtype=np.int32)
        assert batch_size % (num_k + 1) == 0, 'batch_size must be divisible by num_k + 1'
        self.useout_times = useout_times # max num == 3
    
    def __iter__(self):
        # Re-generate batches every epoch
        self.generate_batches(self.epoch_ctr)
        self.epoch_ctr += 1
        for batch in self.batch_idx:
            yield batch

    def __len__(self):
        return len(self.batch_idx)

    def generate_batches(self, epoch_ctr):
        # suppose the traversal_idxs is sorted by timestamp
        self.batch_idx = []
        t0 = time.time()
        unused_elements_ndx = np.ones(self.dataset_length, dtype=np.bool_)
        used_once_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
        used_twice_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
        current_batch = []
        device = torch.cuda.current_device()
        self.dataset.UTM_coord_tensor = self.dataset.UTM_coord_tensor.to(device)
        curr_time = 1
        while True:
            if len(current_batch) >= self.batch_size:
                self.sample_used[current_batch] += 1
                self.batch_idx.append(current_batch)
                current_batch = []
            if np.count_nonzero(unused_elements_ndx) == 0:
                if curr_time == self.useout_times:
                    break
                else:
                    curr_time += 1
                    unused_elements_ndx = ~used_once_elements_ndx
                    used_once_elements_ndx = used_twice_elements_ndx
                    used_twice_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
                    continue
            unused_elements = np.nonzero(unused_elements_ndx)[0]
            unused_element_indice = np.random.randint(0, len(unused_elements), size=1, dtype=np.int32)
            anchor_element = unused_elements[unused_element_indice]
            unused_elements_ndx[anchor_element] = False
            used_once_elements_ndx[anchor_element] = True
            current_batch.append(int(anchor_element))
            level_positive_elements = self.dataset.get_level_positives_v13(int(anchor_element), self.k)
            for i in range(self.k):
                curr_level_positive_elements = level_positive_elements[i]
                curr_level_unused_positive_elements = curr_level_positive_elements & unused_elements_ndx
                if np.count_nonzero(curr_level_unused_positive_elements) > 0:
                    curr_level_unused_positive_elements_indices = np.nonzero(curr_level_unused_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_unused_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_unused_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    unused_elements_ndx[curr_level_choose_element] = False
                    used_once_elements_ndx[curr_level_choose_element] = True
                    continue
                curr_level_used_once_positive_elements = curr_level_positive_elements & used_once_elements_ndx
                if np.count_nonzero(curr_level_used_once_positive_elements) > 0:
                    curr_level_used_once_positive_elements_indices = np.nonzero(curr_level_used_once_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_used_once_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_used_once_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    used_once_elements_ndx[curr_level_choose_element] = False
                    used_twice_elements_ndx[curr_level_choose_element] = True
                    continue
                curr_level_used_twice_positive_elements = curr_level_positive_elements & used_twice_elements_ndx
                if np.count_nonzero(curr_level_used_twice_positive_elements) > 0:
                    curr_level_used_twice_positive_elements_indices = np.nonzero(curr_level_used_twice_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_used_twice_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_used_twice_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    used_twice_elements_ndx[curr_level_choose_element] = False
                    continue
                raise ValueError('redesign the positive distance list')

        self.dataset.UTM_coord_tensor = self.dataset.UTM_coord_tensor.cpu()
        t1 = time.time()
        print('generate batches done!')
        print('batchsampler5_13 time cost: ', t1 - t0)


class BatchSampler5_16(Sampler):
    # use numpy's mask
    def __init__(self, 
                 dataset: Dataset,
                 num_k: int,
                 batch_size: int,
                 start_epoch: int = 0,
                 useout_times: int = 1):

        self.batch_size = batch_size
        self.dataset = dataset
        self.k = num_k  
        self.batch_idx = []     # Index of elements in each batch (re-generated every epoch)
        self.epoch_ctr = start_epoch
        self.dataset_length = len(self.dataset)
        self.sample_used = np.zeros(len(self.dataset), dtype=np.int32)
        assert batch_size % (num_k + 1) == 0, 'batch_size must be divisible by num_k + 1'
        self.useout_times = useout_times # max num == 3
    
    def __iter__(self):
        # Re-generate batches every epoch
        self.generate_batches(self.epoch_ctr)
        self.epoch_ctr += 1
        for batch in self.batch_idx:
            yield batch

    def __len__(self):
        return len(self.batch_idx)

    def generate_batches(self, epoch_ctr):
        # suppose the traversal_idxs is sorted by timestamp
        self.batch_idx = []
        t0 = time.time()
        unused_elements_ndx = np.ones(self.dataset_length, dtype=np.bool_)
        used_once_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
        used_twice_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
        used_three_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
        used_four_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
        used_five_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
        current_batch = []
        curr_time = 1
        while True:
            if len(current_batch) >= self.batch_size:
                self.sample_used[current_batch] += 1
                self.batch_idx.append(current_batch)
                current_batch = []
            if np.count_nonzero(unused_elements_ndx) == 0:
                if curr_time == self.useout_times:
                    break
                else:
                    curr_time += 1
                    unused_elements_ndx = ~used_once_elements_ndx
                    used_once_elements_ndx = used_twice_elements_ndx
                    used_twice_elements_ndx = used_three_elements_ndx
                    used_three_elements_ndx = used_four_elements_ndx
                    used_four_elements_ndx = used_five_elements_ndx
                    used_five_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
                    continue
            unused_elements = np.nonzero(unused_elements_ndx)[0]
            unused_element_indice = np.random.randint(0, len(unused_elements), size=1, dtype=np.int32)
            anchor_element = unused_elements[unused_element_indice]
            unused_elements_ndx[anchor_element] = False
            used_once_elements_ndx[anchor_element] = True
            current_batch.append(int(anchor_element))
            level_positive_elements = self.dataset.get_level_positives_v14(int(anchor_element), self.k)
            for i in range(self.k):
                curr_level_positive_elements = level_positive_elements[i]
                curr_level_unused_positive_elements = curr_level_positive_elements & unused_elements_ndx
                if np.count_nonzero(curr_level_unused_positive_elements) > 0:
                    curr_level_unused_positive_elements_indices = np.nonzero(curr_level_unused_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_unused_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_unused_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    unused_elements_ndx[curr_level_choose_element] = False
                    used_once_elements_ndx[curr_level_choose_element] = True
                    continue
                curr_level_used_once_positive_elements = curr_level_positive_elements & used_once_elements_ndx
                if np.count_nonzero(curr_level_used_once_positive_elements) > 0:
                    curr_level_used_once_positive_elements_indices = np.nonzero(curr_level_used_once_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_used_once_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_used_once_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    used_once_elements_ndx[curr_level_choose_element] = False
                    used_twice_elements_ndx[curr_level_choose_element] = True
                    continue
                curr_level_used_twice_positive_elements = curr_level_positive_elements & used_twice_elements_ndx
                if np.count_nonzero(curr_level_used_twice_positive_elements) > 0:
                    curr_level_used_twice_positive_elements_indices = np.nonzero(curr_level_used_twice_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_used_twice_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_used_twice_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    used_twice_elements_ndx[curr_level_choose_element] = False
                    used_three_elements_ndx[curr_level_choose_element] = True
                    continue
                curr_level_used_three_positive_elements = curr_level_positive_elements & used_three_elements_ndx
                if np.count_nonzero(curr_level_used_three_positive_elements) > 0:
                    curr_level_used_three_positive_elements_indices = np.nonzero(curr_level_used_three_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_used_three_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_used_three_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    used_three_elements_ndx[curr_level_choose_element] = False
                    used_four_elements_ndx[curr_level_choose_element] = True
                    continue
                curr_level_used_four_positive_elements = curr_level_positive_elements & used_four_elements_ndx
                if np.count_nonzero(curr_level_used_four_positive_elements) > 0:
                    curr_level_used_four_positive_elements_indices = np.nonzero(curr_level_used_four_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_used_four_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_used_four_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    used_four_elements_ndx[curr_level_choose_element] = False
                    used_five_elements_ndx[curr_level_choose_element] = True
                    continue
                curr_level_used_five_positive_elements = curr_level_positive_elements & used_five_elements_ndx
                if np.count_nonzero(curr_level_used_five_positive_elements) > 0:
                    curr_level_used_five_positive_elements_indices = np.nonzero(curr_level_used_five_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_used_five_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_used_five_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    used_five_elements_ndx[curr_level_choose_element] = False
                    continue
                raise ValueError('redesign the positive distance list')

        t1 = time.time()
        print('generate batches done!')
        print('batchsampler5_16 time cost: ', t1 - t0)
    



class BatchSampler5_17(Sampler):
    # use numpy's mask
    def __init__(self, 
                 dataset: Dataset,
                 num_k: int,
                 batch_size: int,
                 start_epoch: int = 0,
                 useout_times: int = 1,
                 interval_num: int = 100):

        self.batch_size = batch_size
        self.dataset = dataset
        self.k = num_k  
        self.batch_idx = []     # Index of elements in each batch (re-generated every epoch)
        self.epoch_ctr = start_epoch
        self.dataset_length = len(self.dataset)
        self.sample_used = np.zeros(len(self.dataset), dtype=np.int32)
        assert batch_size % (num_k + 1) == 0, 'batch_size must be divisible by num_k + 1'
        self.useout_times = useout_times # max num == 3
        self.interval_num = interval_num
    
    def __iter__(self):
        # Re-generate batches every epoch
        self.generate_batches(self.epoch_ctr)
        self.epoch_ctr += 1
        for batch in self.batch_idx:
            yield batch

    def __len__(self):
        return len(self.batch_idx)

    def generate_batches(self, epoch_ctr):
        # suppose the traversal_idxs is sorted by timestamp
        self.batch_idx = []
        t0 = time.time()
        unused_elements_ndx = np.ones(self.dataset_length, dtype=np.bool_)
        used_once_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
        used_twice_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
        current_batch = []
        device = torch.cuda.current_device()
        self.dataset.UTM_coord_tensor = self.dataset.UTM_coord_tensor.to(device)
        curr_time = 1
        while True:
            if len(current_batch) >= self.batch_size:
                self.sample_used[current_batch] += 1
                self.batch_idx.append(current_batch)
                current_batch = []
            if np.count_nonzero(unused_elements_ndx) == 0:
                if curr_time == self.useout_times:
                    break
                else:
                    curr_time += 1
                    unused_elements_ndx = ~used_once_elements_ndx
                    used_once_elements_ndx = used_twice_elements_ndx
                    used_twice_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
                    continue
            unused_elements = np.nonzero(unused_elements_ndx)[0]
            unused_element_indice = np.random.randint(0, len(unused_elements), size=1, dtype=np.int32)
            anchor_element = unused_elements[unused_element_indice]
            unused_elements_ndx[anchor_element] = False
            used_once_elements_ndx[anchor_element] = True
            current_batch.append(int(anchor_element))
            level_positive_elements = self.dataset.get_level_positives_v16(int(anchor_element), self.interval_num)
            for i in range(self.k):
                curr_level_positive_elements = level_positive_elements[i]
                curr_level_unused_positive_elements = curr_level_positive_elements & unused_elements_ndx
                if np.count_nonzero(curr_level_unused_positive_elements) > 0:
                    curr_level_unused_positive_elements_indices = np.nonzero(curr_level_unused_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_unused_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_unused_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    unused_elements_ndx[curr_level_choose_element] = False
                    used_once_elements_ndx[curr_level_choose_element] = True
                    continue
                curr_level_used_once_positive_elements = curr_level_positive_elements & used_once_elements_ndx
                if np.count_nonzero(curr_level_used_once_positive_elements) > 0:
                    curr_level_used_once_positive_elements_indices = np.nonzero(curr_level_used_once_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_used_once_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_used_once_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    used_once_elements_ndx[curr_level_choose_element] = False
                    used_twice_elements_ndx[curr_level_choose_element] = True
                    continue
                curr_level_used_twice_positive_elements = curr_level_positive_elements & used_twice_elements_ndx
                if np.count_nonzero(curr_level_used_twice_positive_elements) > 0:
                    curr_level_used_twice_positive_elements_indices = np.nonzero(curr_level_used_twice_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_used_twice_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_used_twice_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    used_twice_elements_ndx[curr_level_choose_element] = False
                    continue
                raise ValueError('redesign the positive distance list')

        self.dataset.UTM_coord_tensor = self.dataset.UTM_coord_tensor.cpu()
        t1 = time.time()
        print('generate batches done!')
        print('batchsampler5_17 time cost: ', t1 - t0)



class BatchSampler5_18(Sampler):
    # use numpy's mask
    def __init__(self, 
                 dataset: Dataset,
                 num_k: int,
                 batch_size: int,
                 start_epoch: int = 0,
                 useout_times: int = 1,
                 interval_num: int = 100):

        self.batch_size = batch_size
        self.dataset = dataset
        self.k = num_k  
        self.batch_idx = []     # Index of elements in each batch (re-generated every epoch)
        self.epoch_ctr = start_epoch
        self.dataset_length = len(self.dataset)
        self.sample_used = np.zeros(len(self.dataset), dtype=np.int32)
        assert batch_size % (num_k + 1) == 0, 'batch_size must be divisible by num_k + 1'
        self.useout_times = useout_times # max num == 3
        self.interval_num = interval_num
    
    def __iter__(self):
        # Re-generate batches every epoch
        self.generate_batches(self.epoch_ctr)
        self.epoch_ctr += 1
        for batch in self.batch_idx:
            yield batch

    def __len__(self):
        return len(self.batch_idx)

    def generate_batches(self, epoch_ctr):
        # suppose the traversal_idxs is sorted by timestamp
        self.batch_idx = []
        t0 = time.time()
        unused_elements_ndx = np.ones(self.dataset_length, dtype=np.bool_)
        used_once_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
        used_twice_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
        used_three_times_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
        used_four_times_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
        used_five_times_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
        used_six_times_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
        used_seven_times_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
        used_eight_times_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
        used_nine_times_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
        used_ten_times_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
        current_batch = []
        curr_time = 1
        while True:
            if len(current_batch) >= self.batch_size:
                self.sample_used[current_batch] += 1
                self.batch_idx.append(current_batch)
                current_batch = []
            if np.count_nonzero(unused_elements_ndx) == 0:
                if curr_time == self.useout_times:
                    break
                else:
                    curr_time += 1
                    unused_elements_ndx = ~used_once_elements_ndx
                    used_once_elements_ndx = used_twice_elements_ndx
                    used_twice_elements_ndx = used_three_times_elements_ndx
                    used_three_times_elements_ndx = used_four_times_elements_ndx
                    used_four_times_elements_ndx = used_five_times_elements_ndx
                    used_five_times_elements_ndx = used_six_times_elements_ndx
                    used_six_times_elements_ndx = used_seven_times_elements_ndx
                    used_seven_times_elements_ndx = used_eight_times_elements_ndx
                    used_eight_times_elements_ndx = used_nine_times_elements_ndx
                    used_nine_times_elements_ndx = used_ten_times_elements_ndx
                    used_ten_times_elements_ndx = np.zeros(self.dataset_length, dtype=np.bool_)
                    continue
            unused_elements = np.nonzero(unused_elements_ndx)[0]
            unused_element_indice = np.random.randint(0, len(unused_elements), size=1, dtype=np.int32)
            anchor_element = unused_elements[unused_element_indice]
            unused_elements_ndx[anchor_element] = False
            used_once_elements_ndx[anchor_element] = True
            current_batch.append(int(anchor_element))
            level_positive_elements = self.dataset.get_level_positives_v17(int(anchor_element), self.interval_num)
            for i in range(self.k):
                curr_level_positive_elements = level_positive_elements[i]
                curr_level_unused_positive_elements = curr_level_positive_elements & unused_elements_ndx
                if np.count_nonzero(curr_level_unused_positive_elements) > 0:
                    curr_level_unused_positive_elements_indices = np.nonzero(curr_level_unused_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_unused_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_unused_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    unused_elements_ndx[curr_level_choose_element] = False
                    used_once_elements_ndx[curr_level_choose_element] = True
                    continue
                curr_level_used_once_positive_elements = curr_level_positive_elements & used_once_elements_ndx
                if np.count_nonzero(curr_level_used_once_positive_elements) > 0:
                    curr_level_used_once_positive_elements_indices = np.nonzero(curr_level_used_once_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_used_once_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_used_once_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    used_once_elements_ndx[curr_level_choose_element] = False
                    used_twice_elements_ndx[curr_level_choose_element] = True
                    continue
                curr_level_used_twice_positive_elements = curr_level_positive_elements & used_twice_elements_ndx
                if np.count_nonzero(curr_level_used_twice_positive_elements) > 0:
                    curr_level_used_twice_positive_elements_indices = np.nonzero(curr_level_used_twice_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_used_twice_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_used_twice_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    used_twice_elements_ndx[curr_level_choose_element] = False
                    used_three_times_elements_ndx[curr_level_choose_element] = True
                    continue
                curr_level_used_three_positive_elements = curr_level_positive_elements & used_three_times_elements_ndx
                if np.count_nonzero(curr_level_used_three_positive_elements) > 0:
                    curr_level_used_three_positive_elements_indices = np.nonzero(curr_level_used_three_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_used_three_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_used_three_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    used_three_times_elements_ndx[curr_level_choose_element] = False
                    used_four_times_elements_ndx[curr_level_choose_element] = True
                    continue
                curr_level_used_four_positive_elements = curr_level_positive_elements & used_four_times_elements_ndx
                if np.count_nonzero(curr_level_used_four_positive_elements) > 0:
                    curr_level_used_four_positive_elements_indices = np.nonzero(curr_level_used_four_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_used_four_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_used_four_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    used_four_times_elements_ndx[curr_level_choose_element] = False
                    used_five_times_elements_ndx[curr_level_choose_element] = True
                    continue
                curr_level_used_five_positive_elements = curr_level_positive_elements & used_five_times_elements_ndx
                if np.count_nonzero(curr_level_used_five_positive_elements) > 0:
                    curr_level_used_five_positive_elements_indices = np.nonzero(curr_level_used_five_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_used_five_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_used_five_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    used_five_times_elements_ndx[curr_level_choose_element] = False
                    used_six_times_elements_ndx[curr_level_choose_element] = True
                    continue
                curr_level_used_six_positive_elements = curr_level_positive_elements & used_six_times_elements_ndx
                if np.count_nonzero(curr_level_used_six_positive_elements) > 0:
                    curr_level_used_six_positive_elements_indices = np.nonzero(curr_level_used_six_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_used_six_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_used_six_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    used_six_times_elements_ndx[curr_level_choose_element] = False
                    used_seven_times_elements_ndx[curr_level_choose_element] = True
                    continue
                curr_level_used_seven_positive_elements = curr_level_positive_elements & used_seven_times_elements_ndx
                if np.count_nonzero(curr_level_used_seven_positive_elements) > 0:
                    curr_level_used_seven_positive_elements_indices = np.nonzero(curr_level_used_seven_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_used_seven_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_used_seven_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    used_seven_times_elements_ndx[curr_level_choose_element] = False
                    used_eight_times_elements_ndx[curr_level_choose_element] = True
                    continue
                curr_level_used_eight_positive_elements = curr_level_positive_elements & used_eight_times_elements_ndx
                if np.count_nonzero(curr_level_used_eight_positive_elements) > 0:
                    curr_level_used_eight_positive_elements_indices = np.nonzero(curr_level_used_eight_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_used_eight_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_used_eight_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    used_eight_times_elements_ndx[curr_level_choose_element] = False
                    used_nine_times_elements_ndx[curr_level_choose_element] = True
                    continue
                curr_level_used_nine_positive_elements = curr_level_positive_elements & used_nine_times_elements_ndx
                if np.count_nonzero(curr_level_used_nine_positive_elements) > 0:
                    curr_level_used_nine_positive_elements_indices = np.nonzero(curr_level_used_nine_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_used_nine_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_used_nine_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    used_nine_times_elements_ndx[curr_level_choose_element] = False
                    used_ten_times_elements_ndx[curr_level_choose_element] = True
                    continue
                curr_level_used_ten_positive_elements = curr_level_positive_elements & used_ten_times_elements_ndx
                if np.count_nonzero(curr_level_used_ten_positive_elements) > 0:
                    curr_level_used_ten_positive_elements_indices = np.nonzero(curr_level_used_ten_positive_elements)[0]
                    curr_level_choose_element_indice = np.random.randint(0, len(curr_level_used_ten_positive_elements_indices), size=1, dtype=np.int32)
                    curr_level_choose_element = curr_level_used_ten_positive_elements_indices[curr_level_choose_element_indice]
                    current_batch.append(int(curr_level_choose_element))
                    used_ten_times_elements_ndx[curr_level_choose_element] = False
                    continue

                raise ValueError('redesign the positive distance list')

        t1 = time.time()
        print('generate batches done!')
        print('batchsampler5_11 time cost: ', t1 - t0)


class BatchSampler5_19(Sampler):
    # use numpy's mask
    def __init__(self, 
                 dataset: Dataset,
                 num_k: int,
                 batch_size: int,
                 start_epoch: int = 0,
                 useout_times: int = 1,
                 interval_num: int = 100,
                 batch_shuffle: bool = False):

        self.batch_size = batch_size
        self.dataset = dataset
        self.k = num_k  
        self.batch_idx = []     # Index of elements in each batch (re-generated every epoch)
        self.epoch_ctr = start_epoch
        self.dataset_length = len(self.dataset)
        assert batch_size % (num_k + 1) == 0, 'batch_size must be divisible by num_k + 1'
        self.useout_times = useout_times # max num == 3
        self.interval_num = interval_num
        self.batch_shuffle = batch_shuffle
    
    def __iter__(self):
        # Re-generate batches every epoch
        self.generate_batches(self.epoch_ctr)
        self.epoch_ctr += 1
        for batch in self.batch_idx:
            yield batch

    def __len__(self):
        return len(self.batch_idx)

    def generate_batches(self, epoch_ctr):
        # suppose the traversal_idxs is sorted by timestamp
        self.batch_idx = []
        t0 = time.time()

        unused_elements_ndx_dict = {}
        used_once_elements_ndx_dict = {}
        used_twice_elements_ndx_dict = {}
        used_three_elements_ndx_dict = {}
        used_four_elements_ndx_dict = {}
        used_five_elements_ndx_dict = {}
        used_six_elements_ndx_dict = {}
        used_seven_elements_ndx_dict = {}
        used_eight_elements_ndx_dict = {}
        used_nine_elements_ndx_dict = {}
        used_ten_elements_ndx_dict = {}
        self.unused_seq_ID_list = []
        device = torch.cuda.current_device()
        for seq_ID, seq_UTM_coords in self.dataset.UTM_coord_tensor.items():
            self.dataset.UTM_coord_tensor[seq_ID] = seq_UTM_coords.to(device)
            if self.dataset.reverse:
                seq_UTM_coords_reverse = self.dataset.UTM_coord_tensor_reverse[seq_ID]
                self.dataset.UTM_coord_tensor_reverse[seq_ID] = seq_UTM_coords_reverse.to(device)
            unused_elements_ndx_dict[seq_ID] = np.ones(self.dataset.seq_length_dict[seq_ID], dtype=np.bool_)
            used_once_elements_ndx_dict[seq_ID] = np.zeros(self.dataset.seq_length_dict[seq_ID], dtype=np.bool_)
            used_twice_elements_ndx_dict[seq_ID] = np.zeros(self.dataset.seq_length_dict[seq_ID], dtype=np.bool_)
            used_three_elements_ndx_dict[seq_ID] = np.zeros(self.dataset.seq_length_dict[seq_ID], dtype=np.bool_)
            used_four_elements_ndx_dict[seq_ID] = np.zeros(self.dataset.seq_length_dict[seq_ID], dtype=np.bool_)
            used_five_elements_ndx_dict[seq_ID] = np.zeros(self.dataset.seq_length_dict[seq_ID], dtype=np.bool_)
            used_six_elements_ndx_dict[seq_ID] = np.zeros(self.dataset.seq_length_dict[seq_ID], dtype=np.bool_)
            used_seven_elements_ndx_dict[seq_ID] = np.zeros(self.dataset.seq_length_dict[seq_ID], dtype=np.bool_)
            used_eight_elements_ndx_dict[seq_ID] = np.zeros(self.dataset.seq_length_dict[seq_ID], dtype=np.bool_)
            used_nine_elements_ndx_dict[seq_ID] = np.zeros(self.dataset.seq_length_dict[seq_ID], dtype=np.bool_)
            used_ten_elements_ndx_dict[seq_ID] = np.zeros(self.dataset.seq_length_dict[seq_ID], dtype=np.bool_)
            self.unused_seq_ID_list.append(seq_ID)

        current_batch = []
        curr_time = 1

        curr_seq_ID = self.unused_seq_ID_list.pop()
        curr_unused_elements_ndx = unused_elements_ndx_dict[curr_seq_ID]
        curr_used_once_elements_ndx = used_once_elements_ndx_dict[curr_seq_ID]
        curr_used_twice_elements_ndx = used_twice_elements_ndx_dict[curr_seq_ID]
        curr_used_three_elements_ndx = used_three_elements_ndx_dict[curr_seq_ID]
        curr_used_four_elements_ndx = used_four_elements_ndx_dict[curr_seq_ID]
        curr_used_five_elements_ndx = used_five_elements_ndx_dict[curr_seq_ID]
        curr_used_six_elements_ndx = used_six_elements_ndx_dict[curr_seq_ID]
        curr_used_seven_elements_ndx = used_seven_elements_ndx_dict[curr_seq_ID]
        curr_used_eight_elements_ndx = used_eight_elements_ndx_dict[curr_seq_ID]
        curr_used_nine_elements_ndx = used_nine_elements_ndx_dict[curr_seq_ID]
        curr_used_ten_elements_ndx = used_ten_elements_ndx_dict[curr_seq_ID]
        while True:
            if len(current_batch) >= self.batch_size:
                self.batch_idx.append(current_batch)
                current_batch = []
            if np.count_nonzero(curr_unused_elements_ndx) == 0:
                if len(self.unused_seq_ID_list) == 0:
                    if curr_time == self.useout_times:
                        if len(current_batch) > 0:
                            self.batch_idx.append(current_batch)
                            current_batch = []
                        break
                    else:
                        curr_time += 1
                        for seq_ID, _ in self.dataset.UTM_coord_tensor.items():
                            self.unused_seq_ID_list.append(seq_ID)
                        unused_elements_ndx_dict[curr_seq_ID] = ~curr_used_once_elements_ndx
                        used_once_elements_ndx_dict[curr_seq_ID] = curr_used_twice_elements_ndx
                        used_twice_elements_ndx_dict[curr_seq_ID] = curr_used_three_elements_ndx
                        used_three_elements_ndx_dict[curr_seq_ID] = curr_used_four_elements_ndx
                        used_four_elements_ndx_dict[curr_seq_ID] = curr_used_five_elements_ndx
                        used_five_elements_ndx_dict[curr_seq_ID] = curr_used_six_elements_ndx
                        used_six_elements_ndx_dict[curr_seq_ID] = curr_used_seven_elements_ndx
                        used_seven_elements_ndx_dict[curr_seq_ID] = curr_used_eight_elements_ndx
                        used_eight_elements_ndx_dict[curr_seq_ID] = curr_used_nine_elements_ndx
                        used_nine_elements_ndx_dict[curr_seq_ID] = curr_used_ten_elements_ndx
                        used_ten_elements_ndx_dict[curr_seq_ID] = np.zeros(self.dataset.seq_length_dict[curr_seq_ID], dtype=np.bool_)
                        curr_seq_ID = self.unused_seq_ID_list.pop()
                        curr_unused_elements_ndx = unused_elements_ndx_dict[curr_seq_ID]
                        curr_used_once_elements_ndx = used_once_elements_ndx_dict[curr_seq_ID]
                        curr_used_twice_elements_ndx = used_twice_elements_ndx_dict[curr_seq_ID]
                        curr_used_three_elements_ndx = used_three_elements_ndx_dict[curr_seq_ID]
                        curr_used_four_elements_ndx = used_four_elements_ndx_dict[curr_seq_ID]
                        curr_used_five_elements_ndx = used_five_elements_ndx_dict[curr_seq_ID]
                        curr_used_six_elements_ndx = used_six_elements_ndx_dict[curr_seq_ID]
                        curr_used_seven_elements_ndx = used_seven_elements_ndx_dict[curr_seq_ID]
                        curr_used_eight_elements_ndx = used_eight_elements_ndx_dict[curr_seq_ID]
                        curr_used_nine_elements_ndx = used_nine_elements_ndx_dict[curr_seq_ID]
                        curr_used_ten_elements_ndx = used_ten_elements_ndx_dict[curr_seq_ID]
                        if len(current_batch) > 0:
                            self.batch_idx.append(current_batch)
                            current_batch = []
                else:
                    unused_elements_ndx_dict[curr_seq_ID] = ~curr_used_once_elements_ndx
                    used_once_elements_ndx_dict[curr_seq_ID] = curr_used_twice_elements_ndx
                    used_twice_elements_ndx_dict[curr_seq_ID] = curr_used_three_elements_ndx
                    used_three_elements_ndx_dict[curr_seq_ID] = curr_used_four_elements_ndx
                    used_four_elements_ndx_dict[curr_seq_ID] = curr_used_five_elements_ndx
                    used_five_elements_ndx_dict[curr_seq_ID] = curr_used_six_elements_ndx
                    used_six_elements_ndx_dict[curr_seq_ID] = curr_used_seven_elements_ndx
                    used_seven_elements_ndx_dict[curr_seq_ID] = curr_used_eight_elements_ndx
                    used_eight_elements_ndx_dict[curr_seq_ID] = curr_used_nine_elements_ndx
                    used_nine_elements_ndx_dict[curr_seq_ID] = curr_used_ten_elements_ndx
                    used_ten_elements_ndx_dict[curr_seq_ID] = np.zeros(self.dataset.seq_length_dict[curr_seq_ID], dtype=np.bool_)
                    curr_seq_ID = self.unused_seq_ID_list.pop()
                    curr_unused_elements_ndx = unused_elements_ndx_dict[curr_seq_ID]
                    curr_used_once_elements_ndx = used_once_elements_ndx_dict[curr_seq_ID]
                    curr_used_twice_elements_ndx = used_twice_elements_ndx_dict[curr_seq_ID]
                    curr_used_three_elements_ndx = used_three_elements_ndx_dict[curr_seq_ID]
                    curr_used_four_elements_ndx = used_four_elements_ndx_dict[curr_seq_ID]
                    curr_used_five_elements_ndx = used_five_elements_ndx_dict[curr_seq_ID]
                    curr_used_six_elements_ndx = used_six_elements_ndx_dict[curr_seq_ID]
                    curr_used_seven_elements_ndx = used_seven_elements_ndx_dict[curr_seq_ID]
                    curr_used_eight_elements_ndx = used_eight_elements_ndx_dict[curr_seq_ID]
                    curr_used_nine_elements_ndx = used_nine_elements_ndx_dict[curr_seq_ID]
                    curr_used_ten_elements_ndx = used_ten_elements_ndx_dict[curr_seq_ID]
                    if len(current_batch) > 0:
                        self.batch_idx.append(current_batch)
                        current_batch = []
                    
            curr_unused_elements = np.nonzero(curr_unused_elements_ndx)[0]
            curr_unused_element_indice = np.random.randint(0, len(curr_unused_elements), size=1, dtype=np.int32)
            curr_anchor_element = curr_unused_elements[curr_unused_element_indice]
            curr_unused_elements_ndx[curr_anchor_element] = False
            curr_used_once_elements_ndx[curr_anchor_element] = True
            current_batch.append([curr_seq_ID, int(curr_anchor_element)])
            curr_level_positive_elements = self.dataset.get_level_positives([curr_seq_ID, int(curr_anchor_element)], self.interval_num)
            for i in range(self.k):
                curr_i_level_positive_elements = curr_level_positive_elements[i]
                curr_i_level_unused_positive_elements = curr_i_level_positive_elements & curr_unused_elements_ndx
                if np.count_nonzero(curr_i_level_unused_positive_elements) > 0:
                    curr_i_level_unused_positive_elements_indices = np.nonzero(curr_i_level_unused_positive_elements)[0]
                    curr_i_level_choose_element_indice = np.random.randint(0, len(curr_i_level_unused_positive_elements_indices), size=1, dtype=np.int32)
                    curr_i_level_choose_element = curr_i_level_unused_positive_elements_indices[curr_i_level_choose_element_indice]
                    current_batch.append([curr_seq_ID, int(curr_i_level_choose_element)])
                    curr_unused_elements_ndx[curr_i_level_choose_element] = False
                    curr_used_once_elements_ndx[curr_i_level_choose_element] = True
                    continue
                curr_i_level_used_once_positive_elements = curr_i_level_positive_elements & curr_used_once_elements_ndx
                if np.count_nonzero(curr_i_level_used_once_positive_elements) > 0:
                    curr_i_level_used_once_positive_elements_indices = np.nonzero(curr_i_level_used_once_positive_elements)[0]
                    curr_i_level_choose_element_indice = np.random.randint(0, len(curr_i_level_used_once_positive_elements_indices), size=1, dtype=np.int32)
                    curr_i_level_choose_element = curr_i_level_used_once_positive_elements_indices[curr_i_level_choose_element_indice]
                    current_batch.append([curr_seq_ID, int(curr_i_level_choose_element)])
                    curr_used_once_elements_ndx[curr_i_level_choose_element] = False
                    curr_used_twice_elements_ndx[curr_i_level_choose_element] = True
                    continue
                curr_i_level_used_twice_positive_elements = curr_i_level_positive_elements & curr_used_twice_elements_ndx
                if np.count_nonzero(curr_i_level_used_twice_positive_elements) > 0:
                    curr_i_level_used_twice_positive_elements_indices = np.nonzero(curr_i_level_used_twice_positive_elements)[0]
                    curr_i_level_choose_element_indice = np.random.randint(0, len(curr_i_level_used_twice_positive_elements_indices), size=1, dtype=np.int32)
                    curr_i_level_choose_element = curr_i_level_used_twice_positive_elements_indices[curr_i_level_choose_element_indice]
                    current_batch.append([curr_seq_ID, int(curr_i_level_choose_element)])
                    curr_used_twice_elements_ndx[curr_i_level_choose_element] = False
                    curr_used_three_elements_ndx[curr_i_level_choose_element] = True
                    continue
                curr_i_level_used_three_positive_elements = curr_i_level_positive_elements & curr_used_three_elements_ndx
                if np.count_nonzero(curr_i_level_used_three_positive_elements) > 0:
                    curr_i_level_used_three_positive_elements_indices = np.nonzero(curr_i_level_used_three_positive_elements)[0]
                    curr_i_level_choose_element_indice = np.random.randint(0, len(curr_i_level_used_three_positive_elements_indices), size=1, dtype=np.int32)
                    curr_i_level_choose_element = curr_i_level_used_three_positive_elements_indices[curr_i_level_choose_element_indice]
                    current_batch.append([curr_seq_ID, int(curr_i_level_choose_element)])
                    curr_used_three_elements_ndx[curr_i_level_choose_element] = False
                    curr_used_four_elements_ndx[curr_i_level_choose_element] = True
                    continue
                curr_i_level_used_four_positive_elements = curr_i_level_positive_elements & curr_used_four_elements_ndx
                if np.count_nonzero(curr_i_level_used_four_positive_elements) > 0:
                    curr_i_level_used_four_positive_elements_indices = np.nonzero(curr_i_level_used_four_positive_elements)[0]
                    curr_i_level_choose_element_indice = np.random.randint(0, len(curr_i_level_used_four_positive_elements_indices), size=1, dtype=np.int32)
                    curr_i_level_choose_element = curr_i_level_used_four_positive_elements_indices[curr_i_level_choose_element_indice]
                    current_batch.append([curr_seq_ID, int(curr_i_level_choose_element)])
                    curr_used_four_elements_ndx[curr_i_level_choose_element] = False
                    curr_used_five_elements_ndx[curr_i_level_choose_element] = True
                    continue
                curr_i_level_used_five_positive_elements = curr_i_level_positive_elements & curr_used_five_elements_ndx
                if np.count_nonzero(curr_i_level_used_five_positive_elements) > 0:
                    curr_i_level_used_five_positive_elements_indices = np.nonzero(curr_i_level_used_five_positive_elements)[0]
                    curr_i_level_choose_element_indice = np.random.randint(0, len(curr_i_level_used_five_positive_elements_indices), size=1, dtype=np.int32)
                    curr_i_level_choose_element = curr_i_level_used_five_positive_elements_indices[curr_i_level_choose_element_indice]
                    current_batch.append([curr_seq_ID, int(curr_i_level_choose_element)])
                    curr_used_five_elements_ndx[curr_i_level_choose_element] = False
                    curr_used_six_elements_ndx[curr_i_level_choose_element] = True
                    continue
                curr_i_level_used_six_positive_elements = curr_i_level_positive_elements & curr_used_six_elements_ndx
                if np.count_nonzero(curr_i_level_used_six_positive_elements) > 0:
                    curr_i_level_used_six_positive_elements_indices = np.nonzero(curr_i_level_used_six_positive_elements)[0]
                    curr_i_level_choose_element_indice = np.random.randint(0, len(curr_i_level_used_six_positive_elements_indices), size=1, dtype=np.int32)
                    curr_i_level_choose_element = curr_i_level_used_six_positive_elements_indices[curr_i_level_choose_element_indice]
                    current_batch.append([curr_seq_ID, int(curr_i_level_choose_element)])
                    curr_used_six_elements_ndx[curr_i_level_choose_element] = False
                    curr_used_seven_elements_ndx[curr_i_level_choose_element] = True
                    continue
                curr_i_level_used_seven_positive_elements = curr_i_level_positive_elements & curr_used_seven_elements_ndx
                if np.count_nonzero(curr_i_level_used_seven_positive_elements) > 0:
                    curr_i_level_used_seven_positive_elements_indices = np.nonzero(curr_i_level_used_seven_positive_elements)[0]
                    curr_i_level_choose_element_indice = np.random.randint(0, len(curr_i_level_used_seven_positive_elements_indices), size=1, dtype=np.int32)
                    curr_i_level_choose_element = curr_i_level_used_seven_positive_elements_indices[curr_i_level_choose_element_indice]
                    current_batch.append([curr_seq_ID, int(curr_i_level_choose_element)])
                    curr_used_seven_elements_ndx[curr_i_level_choose_element] = False
                    curr_used_eight_elements_ndx[curr_i_level_choose_element] = True
                    continue
                curr_i_level_used_eight_positive_elements = curr_i_level_positive_elements & curr_used_eight_elements_ndx
                if np.count_nonzero(curr_i_level_used_eight_positive_elements) > 0:
                    curr_i_level_used_eight_positive_elements_indices = np.nonzero(curr_i_level_used_eight_positive_elements)[0]
                    curr_i_level_choose_element_indice = np.random.randint(0, len(curr_i_level_used_eight_positive_elements_indices), size=1, dtype=np.int32)
                    curr_i_level_choose_element = curr_i_level_used_eight_positive_elements_indices[curr_i_level_choose_element_indice]
                    current_batch.append([curr_seq_ID, int(curr_i_level_choose_element)])
                    curr_used_eight_elements_ndx[curr_i_level_choose_element] = False
                    curr_used_nine_elements_ndx[curr_i_level_choose_element] = True
                    continue
                curr_i_level_used_nine_positive_elements = curr_i_level_positive_elements & curr_used_nine_elements_ndx
                if np.count_nonzero(curr_i_level_used_nine_positive_elements) > 0:
                    curr_i_level_used_nine_positive_elements_indices = np.nonzero(curr_i_level_used_nine_positive_elements)[0]
                    curr_i_level_choose_element_indice = np.random.randint(0, len(curr_i_level_used_nine_positive_elements_indices), size=1, dtype=np.int32)
                    curr_i_level_choose_element = curr_i_level_used_nine_positive_elements_indices[curr_i_level_choose_element_indice]
                    current_batch.append([curr_seq_ID, int(curr_i_level_choose_element)])
                    curr_used_nine_elements_ndx[curr_i_level_choose_element] = False
                    curr_used_ten_elements_ndx[curr_i_level_choose_element] = True
                    continue
                curr_i_level_used_ten_positive_elements = curr_i_level_positive_elements & curr_used_ten_elements_ndx
                if np.count_nonzero(curr_i_level_used_ten_positive_elements) > 0:
                    curr_i_level_used_ten_positive_elements_indices = np.nonzero(curr_i_level_used_ten_positive_elements)[0]
                    curr_i_level_choose_element_indice = np.random.randint(0, len(curr_i_level_used_ten_positive_elements_indices), size=1, dtype=np.int32)
                    curr_i_level_choose_element = curr_i_level_used_ten_positive_elements_indices[curr_i_level_choose_element_indice]
                    current_batch.append([curr_seq_ID, int(curr_i_level_choose_element)])
                    curr_used_ten_elements_ndx[curr_i_level_choose_element] = False
                    continue
                raise ValueError('redesign the positive distance list')

        for seq_ID, seq_UTM_coords in self.dataset.UTM_coord_tensor.items():
            self.dataset.UTM_coord_tensor[seq_ID] = seq_UTM_coords.to('cpu')
            if self.dataset.reverse:
                seq_UTM_coords_reverse = self.dataset.UTM_coord_tensor_reverse[seq_ID]
                self.dataset.UTM_coord_tensor_reverse[seq_ID] = seq_UTM_coords_reverse.to('cpu')
        t1 = time.time()
        if self.batch_shuffle:
            random.shuffle(self.batch_idx)
        print('generate batches done!')
        print('batchsampler5_19 time cost: ', t1 - t0)


class BatchSampler_eval_kitti(Sampler):
    # use numpy's mask
    def __init__(self, 
                 dataset: Dataset,
                 batch_size: int,
                 sampler: None):

        self.batch_size = batch_size
        self.dataset = dataset 
        self.batch_idx = []     # Index of elements in each batch (re-generated every epoch)
        self.sampler = sampler
    
    def __iter__(self):
        self.generate_batches()
        for batch in self.batch_idx:
            yield batch

    def __len__(self):
        return len(self.batch_idx)

    def generate_batches(self,):
        self.batch_idx = []
        current_batch = []
        pre_seq_ID = '-1'
        for sampler_idx in self.sampler:
            seq_num = np.searchsorted(self.dataset.samples_length_cumsum, sampler_idx, side='right')
            if pre_seq_ID != self.dataset.sequence_list[seq_num - 1]:
                if len(current_batch) > 0:
                    self.batch_idx.append(current_batch)
                    current_batch = []
                pre_seq_ID = self.dataset.sequence_list[seq_num - 1]
            current_batch.append([pre_seq_ID, sampler_idx - self.dataset.samples_length_cumsum[seq_num - 1]]) 
            if len(current_batch) == self.batch_size:
                self.batch_idx.append(current_batch)
                current_batch = []
        if len(current_batch) > 0:
            self.batch_idx.append(current_batch)
            current_batch = []



class BatchSampler5_20(Sampler):
    # use numpy's mask
    def __init__(self, 
                 dataset: Dataset,
                 num_k: int,
                 batch_size: int,
                 start_epoch: int = 0,
                 interval_num: int = 100,
                 expansion_neighbor: int = 10,
                 base_batch_num: int = 50,
                 new_epoch_data_ratio_type: str = 'equal', # 'equal' or 'all_new_related' or 'new_higher'
                 new_higher_layer: int = 1,
                 batch_expansion_strategy: str = 'v1'): # 'v1' means upper bound, 'v2' means no upper bound

        self.batch_size = batch_size
        self.dataset = dataset
        self.k = num_k  
        self.batch_idx = []     # Index of elements in each batch (re-generated every epoch)
        self.epoch_ctr = start_epoch
        self.dataset_length = len(self.dataset)
        self.sample_used = np.zeros(len(self.dataset), dtype=np.int32)
        assert batch_size % (num_k + 1) == 0, 'batch_size must be divisible by num_k + 1'
        self.interval_num = interval_num
        start_idx = np.random.randint(0, len(self.dataset)) # random start point
        self.curr_area_idx = [start_idx]
        self.expansion_neighbor = expansion_neighbor
        self.new_epoch_data_ratio_type = new_epoch_data_ratio_type
        self.base_batch_num = base_batch_num
        self.new_higher_layer = new_higher_layer
        self.batch_expansion_strategy = batch_expansion_strategy
     
    def __iter__(self):
        # Re-generate batches every epoch
        self.generate_batches(self.epoch_ctr)
        self.epoch_ctr += 1
        for batch in self.batch_idx:
            yield batch

    def __len__(self):
        return len(self.batch_idx)

    def state_dict(self,):
        print('batch sampler save state_dict done!')
        return copy.deepcopy(self.curr_area_idx)
    
    def load_state_dict(self, curr_area_idx):
        self.curr_area_idx = copy.deepcopy(curr_area_idx)
        print('batch sampler load_state_dict done!')

    def generate_batches(self, epoch_ctr):
        # suppose the traversal_idxs is sorted by timestamp
        device = torch.cuda.current_device()
        self.dataset.UTM_coord_tensor = self.dataset.UTM_coord_tensor.to(device)
        self.batch_idx = []
        t0 = time.time()
        curr_expansion_neighbor = 0
        real_new_area_idx_num = 0
        new_area_idx = 0
        real_new_area_idx = []
        while real_new_area_idx_num < self.expansion_neighbor:
            curr_expansion_neighbor += self.expansion_neighbor
            new_area_idx = self.dataset.get_level_positives_v18(self.curr_area_idx, curr_expansion_neighbor)
            if len(new_area_idx) == self.dataset_length:
                break
            curr_area_idx_array = np.array(self.curr_area_idx)
            real_new_area_idx = np.setdiff1d(new_area_idx, curr_area_idx_array)
            if curr_area_idx_array.shape[0] == 1:
                real_new_area_idx = np.append(real_new_area_idx, curr_area_idx_array)
            real_new_area_idx_num = real_new_area_idx.shape[0]
        self.curr_area_idx = new_area_idx.tolist()



        print(f'new_area_idx_num: {len(new_area_idx)}')




        # new_area_idx_length = len(new_area_idx)
        # dataset_all_length = len(self.dataset) 
        # unused_elements_ndx = np.ones((1, new_area_idx_length), dtype=np.bool_)
        # used_elements_num = torch.max(self.base_batch_num - epoch_ctr, 3)
        # used_elements_ndx_matrix = np.zeros((used_elements_num, new_area_idx_length), dtype=np.bool_) # (used_elements_num, new_area_idx_length)
        # elements_ndx_matrix = np.concatenate((unused_elements_ndx, used_elements_ndx_matrix), axis=0) # (used_elements_num + 1, new_area_idx_length)
        elements_num = 1000
        elements_ndx_matrix = np.zeros((elements_num, self.dataset_length), dtype=np.bool_)
        all_dataset_ndx = np.arange(self.dataset_length)
        not_used_idx = np.setdiff1d(all_dataset_ndx, new_area_idx)
        elements_ndx_matrix[:, not_used_idx] = True


        if self.batch_expansion_strategy == 'v1':
            curr_batch_num = min(self.base_batch_num * (epoch_ctr + 1), 1160)
        elif self.batch_expansion_strategy == 'v2':
            curr_batch_num = self.base_batch_num * (epoch_ctr + 1)
        else:
            raise ValueError('batch_expansion_strategy error')

        if self.new_epoch_data_ratio_type == 'equal':
            pass
        elif self.new_epoch_data_ratio_type == 'all_new_related':
            if len(real_new_area_idx) >= 0.5 * curr_expansion_neighbor:
                new_area_idx_v1 = self.dataset.get_level_positives_v18(real_new_area_idx, curr_expansion_neighbor)
                new_related_idx = np.intersect1d(new_area_idx_v1, new_area_idx)
                not_used_idx = np.setdiff1d(new_area_idx, new_related_idx)
                elements_ndx_matrix[:, not_used_idx] = True
        elif self.new_epoch_data_ratio_type == 'new_higher':
            if len(real_new_area_idx) > 0:
                new_area_idx_v1 = self.dataset.get_level_positives_v18(real_new_area_idx, curr_expansion_neighbor)
                new_related_idx = np.intersect1d(new_area_idx_v1, new_area_idx)
                not_used_idx = np.setdiff1d(new_area_idx, new_related_idx)
                elements_ndx_matrix[:self.new_higher_layer, not_used_idx] = True
        else:
            raise ValueError('new_epoch_data_ratio_type error')
        

        current_batch = []

        while True:
            if len(current_batch) >= self.batch_size:
                self.sample_used[current_batch] += 1
                self.batch_idx.append(current_batch)
                current_batch = []
            if len(self.batch_idx) == curr_batch_num:
                break
            unused_elements_num_per_row = np.count_nonzero(~elements_ndx_matrix, axis=-1, keepdims=False)
            unused_elements_first_row = np.nonzero(unused_elements_num_per_row)[0][0]
            unused_elements = np.nonzero(~elements_ndx_matrix[unused_elements_first_row])[0]
            unused_element_indice = np.random.randint(0, len(unused_elements), size=1, dtype=np.int32)
            anchor_element = unused_elements[unused_element_indice]
            elements_ndx_matrix[unused_elements_first_row][anchor_element] = True
            current_batch.append(int(anchor_element))
            curr_positive_elements = self.dataset.get_level_positives_v18([int(anchor_element)], self.interval_num)
            for i in range(self.k):
                unused_left_flag = False
                for j in range(unused_elements_first_row, elements_num):
                    curr_j_unused_elements_ndx = np.nonzero(~elements_ndx_matrix[j])[0]
                    curr_unused_positive_elements_indices = np.intersect1d(curr_positive_elements, curr_j_unused_elements_ndx)
                    if curr_unused_positive_elements_indices.shape[0] <= 0:
                        continue
                    curr_choose_element_indice = np.random.randint(0, len(curr_unused_positive_elements_indices), size=1, dtype=np.int32)
                    curr_choose_element = curr_unused_positive_elements_indices[curr_choose_element_indice]
                    current_batch.append(int(curr_choose_element))
                    elements_ndx_matrix[j][curr_choose_element] = True
                    unused_left_flag = True
                    break
                if not unused_left_flag:
                    raise ValueError('redesign the positive distance list')

        self.dataset.UTM_coord_tensor = self.dataset.UTM_coord_tensor.cpu()
        t1 = time.time()
        print('generate batches done!')
        print('batchsampler5_20 time cost: ', t1 - t0)



class BatchSampler5_21(Sampler):
    # use numpy's mask
    def __init__(self, 
                 dataset: Dataset,
                 num_k: int,
                 batch_size: int,
                 start_epoch: int = 0,
                 useout_times: int = 1,
                 interval_num: int = 100,
                 get_positive_type = 'v1',):

        self.batch_size = batch_size
        self.dataset = dataset
        self.k = num_k  
        self.batch_idx = []     # Index of elements in each batch (re-generated every epoch)
        self.epoch_ctr = start_epoch
        self.dataset_length = len(self.dataset)
        assert batch_size % (num_k + 1) == 0, 'batch_size must be divisible by num_k + 1'
        self.useout_times = useout_times # max num == 3
        self.interval_num = interval_num
        self.get_positive_type = get_positive_type # 'v1' : area_overlap, 'v2' : pos_vec_vet, 'v3' : exp_dist
    
    def __iter__(self):
        # Re-generate batches every epoch
        self.generate_batches(self.epoch_ctr)
        self.epoch_ctr += 1
        for batch in self.batch_idx:
            yield batch

    def __len__(self):
        return len(self.batch_idx)

    def generate_batches(self, epoch_ctr):
        # suppose the traversal_idxs is sorted by timestamp
        self.batch_idx = []
        t0 = time.time()

        unused_elements_ndx_dict = {}
        used_once_elements_ndx_dict = {}
        used_twice_elements_ndx_dict = {}
        used_three_elements_ndx_dict = {}
        used_four_elements_ndx_dict = {}
        used_five_elements_ndx_dict = {}
        used_six_elements_ndx_dict = {}
        used_seven_elements_ndx_dict = {}
        used_eight_elements_ndx_dict = {}
        used_nine_elements_ndx_dict = {}
        used_ten_elements_ndx_dict = {}
        self.unused_seq_ID_list = []
        device = torch.cuda.current_device()

        if self.get_positive_type == 'v1':
            for seq_ID, seq_UTM_coords in self.dataset.area_overlap.items():
                self.dataset.area_overlap[seq_ID] = seq_UTM_coords.to(device)
                unused_elements_ndx_dict[seq_ID] = np.ones(self.dataset.seq_length_dict[seq_ID], dtype=np.bool_)
                used_once_elements_ndx_dict[seq_ID] = np.zeros(self.dataset.seq_length_dict[seq_ID], dtype=np.bool_)
                used_twice_elements_ndx_dict[seq_ID] = np.zeros(self.dataset.seq_length_dict[seq_ID], dtype=np.bool_)
                used_three_elements_ndx_dict[seq_ID] = np.zeros(self.dataset.seq_length_dict[seq_ID], dtype=np.bool_)
                used_four_elements_ndx_dict[seq_ID] = np.zeros(self.dataset.seq_length_dict[seq_ID], dtype=np.bool_)
                used_five_elements_ndx_dict[seq_ID] = np.zeros(self.dataset.seq_length_dict[seq_ID], dtype=np.bool_)
                used_six_elements_ndx_dict[seq_ID] = np.zeros(self.dataset.seq_length_dict[seq_ID], dtype=np.bool_)
                used_seven_elements_ndx_dict[seq_ID] = np.zeros(self.dataset.seq_length_dict[seq_ID], dtype=np.bool_)
                used_eight_elements_ndx_dict[seq_ID] = np.zeros(self.dataset.seq_length_dict[seq_ID], dtype=np.bool_)
                used_nine_elements_ndx_dict[seq_ID] = np.zeros(self.dataset.seq_length_dict[seq_ID], dtype=np.bool_)
                used_ten_elements_ndx_dict[seq_ID] = np.zeros(self.dataset.seq_length_dict[seq_ID], dtype=np.bool_)
                self.unused_seq_ID_list.append(seq_ID)
        elif self.get_positive_type == 'v2':
            for seq_ID, seq_UTM_coords in self.dataset.pos_vec_vet_coords_tensor.items():
                self.dataset.pos_vec_vet_coords_tensor[seq_ID] = seq_UTM_coords.to(device)
                unused_elements_ndx_dict[seq_ID] = np.ones(self.dataset.seq_length_dict[seq_ID], dtype=np.bool_)
                used_once_elements_ndx_dict[seq_ID] = np.zeros(self.dataset.seq_length_dict[seq_ID], dtype=np.bool_)
                used_twice_elements_ndx_dict[seq_ID] = np.zeros(self.dataset.seq_length_dict[seq_ID], dtype=np.bool_)
                used_three_elements_ndx_dict[seq_ID] = np.zeros(self.dataset.seq_length_dict[seq_ID], dtype=np.bool_)
                used_four_elements_ndx_dict[seq_ID] = np.zeros(self.dataset.seq_length_dict[seq_ID], dtype=np.bool_)
                used_five_elements_ndx_dict[seq_ID] = np.zeros(self.dataset.seq_length_dict[seq_ID], dtype=np.bool_)
                used_six_elements_ndx_dict[seq_ID] = np.zeros(self.dataset.seq_length_dict[seq_ID], dtype=np.bool_)
                used_seven_elements_ndx_dict[seq_ID] = np.zeros(self.dataset.seq_length_dict[seq_ID], dtype=np.bool_)
                used_eight_elements_ndx_dict[seq_ID] = np.zeros(self.dataset.seq_length_dict[seq_ID], dtype=np.bool_)
                used_nine_elements_ndx_dict[seq_ID] = np.zeros(self.dataset.seq_length_dict[seq_ID], dtype=np.bool_)
                used_ten_elements_ndx_dict[seq_ID] = np.zeros(self.dataset.seq_length_dict[seq_ID], dtype=np.bool_)
                self.unused_seq_ID_list.append(seq_ID)
        elif self.get_positive_type == 'v3':
            for seq_ID, seq_UTM_coords in self.dataset.UTM_coord_tensor.items():
                self.dataset.UTM_coord_tensor[seq_ID] = seq_UTM_coords.to(device)
                unused_elements_ndx_dict[seq_ID] = np.ones(self.dataset.seq_length_dict[seq_ID], dtype=np.bool_)
                used_once_elements_ndx_dict[seq_ID] = np.zeros(self.dataset.seq_length_dict[seq_ID], dtype=np.bool_)
                used_twice_elements_ndx_dict[seq_ID] = np.zeros(self.dataset.seq_length_dict[seq_ID], dtype=np.bool_)
                used_three_elements_ndx_dict[seq_ID] = np.zeros(self.dataset.seq_length_dict[seq_ID], dtype=np.bool_)
                used_four_elements_ndx_dict[seq_ID] = np.zeros(self.dataset.seq_length_dict[seq_ID], dtype=np.bool_)
                used_five_elements_ndx_dict[seq_ID] = np.zeros(self.dataset.seq_length_dict[seq_ID], dtype=np.bool_)
                used_six_elements_ndx_dict[seq_ID] = np.zeros(self.dataset.seq_length_dict[seq_ID], dtype=np.bool_)
                used_seven_elements_ndx_dict[seq_ID] = np.zeros(self.dataset.seq_length_dict[seq_ID], dtype=np.bool_)
                used_eight_elements_ndx_dict[seq_ID] = np.zeros(self.dataset.seq_length_dict[seq_ID], dtype=np.bool_)
                used_nine_elements_ndx_dict[seq_ID] = np.zeros(self.dataset.seq_length_dict[seq_ID], dtype=np.bool_)
                used_ten_elements_ndx_dict[seq_ID] = np.zeros(self.dataset.seq_length_dict[seq_ID], dtype=np.bool_)
                self.unused_seq_ID_list.append(seq_ID)

        current_batch = []
        curr_time = 1

        curr_seq_ID = self.unused_seq_ID_list.pop()
        curr_unused_elements_ndx = unused_elements_ndx_dict[curr_seq_ID]
        curr_used_once_elements_ndx = used_once_elements_ndx_dict[curr_seq_ID]
        curr_used_twice_elements_ndx = used_twice_elements_ndx_dict[curr_seq_ID]
        curr_used_three_elements_ndx = used_three_elements_ndx_dict[curr_seq_ID]
        curr_used_four_elements_ndx = used_four_elements_ndx_dict[curr_seq_ID]
        curr_used_five_elements_ndx = used_five_elements_ndx_dict[curr_seq_ID]
        curr_used_six_elements_ndx = used_six_elements_ndx_dict[curr_seq_ID]
        curr_used_seven_elements_ndx = used_seven_elements_ndx_dict[curr_seq_ID]
        curr_used_eight_elements_ndx = used_eight_elements_ndx_dict[curr_seq_ID]
        curr_used_nine_elements_ndx = used_nine_elements_ndx_dict[curr_seq_ID]
        curr_used_ten_elements_ndx = used_ten_elements_ndx_dict[curr_seq_ID]
        while True:
            if len(current_batch) >= self.batch_size:
                self.batch_idx.append(current_batch)
                current_batch = []
            if np.count_nonzero(curr_unused_elements_ndx) == 0:
                if len(self.unused_seq_ID_list) == 0:
                    if curr_time == self.useout_times:
                        if len(current_batch) > 0:
                            self.batch_idx.append(current_batch)
                            current_batch = []
                        break
                    else:
                        curr_time += 1
                        for seq_ID, _ in self.dataset.seq_length_dict.items():
                            self.unused_seq_ID_list.append(seq_ID)
                        unused_elements_ndx_dict[curr_seq_ID] = ~curr_used_once_elements_ndx
                        used_once_elements_ndx_dict[curr_seq_ID] = curr_used_twice_elements_ndx
                        used_twice_elements_ndx_dict[curr_seq_ID] = curr_used_three_elements_ndx
                        used_three_elements_ndx_dict[curr_seq_ID] = curr_used_four_elements_ndx
                        used_four_elements_ndx_dict[curr_seq_ID] = curr_used_five_elements_ndx
                        used_five_elements_ndx_dict[curr_seq_ID] = curr_used_six_elements_ndx
                        used_six_elements_ndx_dict[curr_seq_ID] = curr_used_seven_elements_ndx
                        used_seven_elements_ndx_dict[curr_seq_ID] = curr_used_eight_elements_ndx
                        used_eight_elements_ndx_dict[curr_seq_ID] = curr_used_nine_elements_ndx
                        used_nine_elements_ndx_dict[curr_seq_ID] = curr_used_ten_elements_ndx
                        used_ten_elements_ndx_dict[curr_seq_ID] = np.zeros(self.dataset.seq_length_dict[curr_seq_ID], dtype=np.bool_)
                        curr_seq_ID = self.unused_seq_ID_list.pop()
                        curr_unused_elements_ndx = unused_elements_ndx_dict[curr_seq_ID]
                        curr_used_once_elements_ndx = used_once_elements_ndx_dict[curr_seq_ID]
                        curr_used_twice_elements_ndx = used_twice_elements_ndx_dict[curr_seq_ID]
                        curr_used_three_elements_ndx = used_three_elements_ndx_dict[curr_seq_ID]
                        curr_used_four_elements_ndx = used_four_elements_ndx_dict[curr_seq_ID]
                        curr_used_five_elements_ndx = used_five_elements_ndx_dict[curr_seq_ID]
                        curr_used_six_elements_ndx = used_six_elements_ndx_dict[curr_seq_ID]
                        curr_used_seven_elements_ndx = used_seven_elements_ndx_dict[curr_seq_ID]
                        curr_used_eight_elements_ndx = used_eight_elements_ndx_dict[curr_seq_ID]
                        curr_used_nine_elements_ndx = used_nine_elements_ndx_dict[curr_seq_ID]
                        curr_used_ten_elements_ndx = used_ten_elements_ndx_dict[curr_seq_ID]
                        if len(current_batch) > 0:
                            self.batch_idx.append(current_batch)
                            current_batch = []
                else:
                    unused_elements_ndx_dict[curr_seq_ID] = ~curr_used_once_elements_ndx
                    used_once_elements_ndx_dict[curr_seq_ID] = curr_used_twice_elements_ndx
                    used_twice_elements_ndx_dict[curr_seq_ID] = curr_used_three_elements_ndx
                    used_three_elements_ndx_dict[curr_seq_ID] = curr_used_four_elements_ndx
                    used_four_elements_ndx_dict[curr_seq_ID] = curr_used_five_elements_ndx
                    used_five_elements_ndx_dict[curr_seq_ID] = curr_used_six_elements_ndx
                    used_six_elements_ndx_dict[curr_seq_ID] = curr_used_seven_elements_ndx
                    used_seven_elements_ndx_dict[curr_seq_ID] = curr_used_eight_elements_ndx
                    used_eight_elements_ndx_dict[curr_seq_ID] = curr_used_nine_elements_ndx
                    used_nine_elements_ndx_dict[curr_seq_ID] = curr_used_ten_elements_ndx
                    used_ten_elements_ndx_dict[curr_seq_ID] = np.zeros(self.dataset.seq_length_dict[curr_seq_ID], dtype=np.bool_)
                    curr_seq_ID = self.unused_seq_ID_list.pop()
                    curr_unused_elements_ndx = unused_elements_ndx_dict[curr_seq_ID]
                    curr_used_once_elements_ndx = used_once_elements_ndx_dict[curr_seq_ID]
                    curr_used_twice_elements_ndx = used_twice_elements_ndx_dict[curr_seq_ID]
                    curr_used_three_elements_ndx = used_three_elements_ndx_dict[curr_seq_ID]
                    curr_used_four_elements_ndx = used_four_elements_ndx_dict[curr_seq_ID]
                    curr_used_five_elements_ndx = used_five_elements_ndx_dict[curr_seq_ID]
                    curr_used_six_elements_ndx = used_six_elements_ndx_dict[curr_seq_ID]
                    curr_used_seven_elements_ndx = used_seven_elements_ndx_dict[curr_seq_ID]
                    curr_used_eight_elements_ndx = used_eight_elements_ndx_dict[curr_seq_ID]
                    curr_used_nine_elements_ndx = used_nine_elements_ndx_dict[curr_seq_ID]
                    curr_used_ten_elements_ndx = used_ten_elements_ndx_dict[curr_seq_ID]
                    if len(current_batch) > 0:
                        self.batch_idx.append(current_batch)
                        current_batch = []
                    
            curr_unused_elements = np.nonzero(curr_unused_elements_ndx)[0]
            curr_unused_element_indice = np.random.randint(0, len(curr_unused_elements), size=1, dtype=np.int32)
            curr_anchor_element = curr_unused_elements[curr_unused_element_indice]
            curr_unused_elements_ndx[curr_anchor_element] = False
            curr_used_once_elements_ndx[curr_anchor_element] = True
            current_batch.append([curr_seq_ID, int(curr_anchor_element)])
            if self.get_positive_type == 'v1':
                curr_level_positive_elements = self.dataset.get_level_positives_v1([curr_seq_ID, int(curr_anchor_element)], self.interval_num)
            elif self.get_positive_type == 'v2':
                curr_level_positive_elements = self.dataset.get_level_positives_v2([curr_seq_ID, int(curr_anchor_element)])
            elif self.get_positive_type == 'v3':
                curr_level_positive_elements = self.dataset.get_level_positives_v3([curr_seq_ID, int(curr_anchor_element)], self.interval_num)
            for i in range(self.k):
                curr_i_level_positive_elements = curr_level_positive_elements[i]
                curr_i_level_unused_positive_elements = curr_i_level_positive_elements & curr_unused_elements_ndx
                if np.count_nonzero(curr_i_level_unused_positive_elements) > 0:
                    curr_i_level_unused_positive_elements_indices = np.nonzero(curr_i_level_unused_positive_elements)[0]
                    curr_i_level_choose_element_indice = np.random.randint(0, len(curr_i_level_unused_positive_elements_indices), size=1, dtype=np.int32)
                    curr_i_level_choose_element = curr_i_level_unused_positive_elements_indices[curr_i_level_choose_element_indice]
                    current_batch.append([curr_seq_ID, int(curr_i_level_choose_element)])
                    curr_unused_elements_ndx[curr_i_level_choose_element] = False
                    curr_used_once_elements_ndx[curr_i_level_choose_element] = True
                    continue
                curr_i_level_used_once_positive_elements = curr_i_level_positive_elements & curr_used_once_elements_ndx
                if np.count_nonzero(curr_i_level_used_once_positive_elements) > 0:
                    curr_i_level_used_once_positive_elements_indices = np.nonzero(curr_i_level_used_once_positive_elements)[0]
                    curr_i_level_choose_element_indice = np.random.randint(0, len(curr_i_level_used_once_positive_elements_indices), size=1, dtype=np.int32)
                    curr_i_level_choose_element = curr_i_level_used_once_positive_elements_indices[curr_i_level_choose_element_indice]
                    current_batch.append([curr_seq_ID, int(curr_i_level_choose_element)])
                    curr_used_once_elements_ndx[curr_i_level_choose_element] = False
                    curr_used_twice_elements_ndx[curr_i_level_choose_element] = True
                    continue
                curr_i_level_used_twice_positive_elements = curr_i_level_positive_elements & curr_used_twice_elements_ndx
                if np.count_nonzero(curr_i_level_used_twice_positive_elements) > 0:
                    curr_i_level_used_twice_positive_elements_indices = np.nonzero(curr_i_level_used_twice_positive_elements)[0]
                    curr_i_level_choose_element_indice = np.random.randint(0, len(curr_i_level_used_twice_positive_elements_indices), size=1, dtype=np.int32)
                    curr_i_level_choose_element = curr_i_level_used_twice_positive_elements_indices[curr_i_level_choose_element_indice]
                    current_batch.append([curr_seq_ID, int(curr_i_level_choose_element)])
                    curr_used_twice_elements_ndx[curr_i_level_choose_element] = False
                    curr_used_three_elements_ndx[curr_i_level_choose_element] = True
                    continue
                curr_i_level_used_three_positive_elements = curr_i_level_positive_elements & curr_used_three_elements_ndx
                if np.count_nonzero(curr_i_level_used_three_positive_elements) > 0:
                    curr_i_level_used_three_positive_elements_indices = np.nonzero(curr_i_level_used_three_positive_elements)[0]
                    curr_i_level_choose_element_indice = np.random.randint(0, len(curr_i_level_used_three_positive_elements_indices), size=1, dtype=np.int32)
                    curr_i_level_choose_element = curr_i_level_used_three_positive_elements_indices[curr_i_level_choose_element_indice]
                    current_batch.append([curr_seq_ID, int(curr_i_level_choose_element)])
                    curr_used_three_elements_ndx[curr_i_level_choose_element] = False
                    curr_used_four_elements_ndx[curr_i_level_choose_element] = True
                    continue
                curr_i_level_used_four_positive_elements = curr_i_level_positive_elements & curr_used_four_elements_ndx
                if np.count_nonzero(curr_i_level_used_four_positive_elements) > 0:
                    curr_i_level_used_four_positive_elements_indices = np.nonzero(curr_i_level_used_four_positive_elements)[0]
                    curr_i_level_choose_element_indice = np.random.randint(0, len(curr_i_level_used_four_positive_elements_indices), size=1, dtype=np.int32)
                    curr_i_level_choose_element = curr_i_level_used_four_positive_elements_indices[curr_i_level_choose_element_indice]
                    current_batch.append([curr_seq_ID, int(curr_i_level_choose_element)])
                    curr_used_four_elements_ndx[curr_i_level_choose_element] = False
                    curr_used_five_elements_ndx[curr_i_level_choose_element] = True
                    continue
                curr_i_level_used_five_positive_elements = curr_i_level_positive_elements & curr_used_five_elements_ndx
                if np.count_nonzero(curr_i_level_used_five_positive_elements) > 0:
                    curr_i_level_used_five_positive_elements_indices = np.nonzero(curr_i_level_used_five_positive_elements)[0]
                    curr_i_level_choose_element_indice = np.random.randint(0, len(curr_i_level_used_five_positive_elements_indices), size=1, dtype=np.int32)
                    curr_i_level_choose_element = curr_i_level_used_five_positive_elements_indices[curr_i_level_choose_element_indice]
                    current_batch.append([curr_seq_ID, int(curr_i_level_choose_element)])
                    curr_used_five_elements_ndx[curr_i_level_choose_element] = False
                    curr_used_six_elements_ndx[curr_i_level_choose_element] = True
                    continue
                curr_i_level_used_six_positive_elements = curr_i_level_positive_elements & curr_used_six_elements_ndx
                if np.count_nonzero(curr_i_level_used_six_positive_elements) > 0:
                    curr_i_level_used_six_positive_elements_indices = np.nonzero(curr_i_level_used_six_positive_elements)[0]
                    curr_i_level_choose_element_indice = np.random.randint(0, len(curr_i_level_used_six_positive_elements_indices), size=1, dtype=np.int32)
                    curr_i_level_choose_element = curr_i_level_used_six_positive_elements_indices[curr_i_level_choose_element_indice]
                    current_batch.append([curr_seq_ID, int(curr_i_level_choose_element)])
                    curr_used_six_elements_ndx[curr_i_level_choose_element] = False
                    curr_used_seven_elements_ndx[curr_i_level_choose_element] = True
                    continue
                curr_i_level_used_seven_positive_elements = curr_i_level_positive_elements & curr_used_seven_elements_ndx
                if np.count_nonzero(curr_i_level_used_seven_positive_elements) > 0:
                    curr_i_level_used_seven_positive_elements_indices = np.nonzero(curr_i_level_used_seven_positive_elements)[0]
                    curr_i_level_choose_element_indice = np.random.randint(0, len(curr_i_level_used_seven_positive_elements_indices), size=1, dtype=np.int32)
                    curr_i_level_choose_element = curr_i_level_used_seven_positive_elements_indices[curr_i_level_choose_element_indice]
                    current_batch.append([curr_seq_ID, int(curr_i_level_choose_element)])
                    curr_used_seven_elements_ndx[curr_i_level_choose_element] = False
                    curr_used_eight_elements_ndx[curr_i_level_choose_element] = True
                    continue
                curr_i_level_used_eight_positive_elements = curr_i_level_positive_elements & curr_used_eight_elements_ndx
                if np.count_nonzero(curr_i_level_used_eight_positive_elements) > 0:
                    curr_i_level_used_eight_positive_elements_indices = np.nonzero(curr_i_level_used_eight_positive_elements)[0]
                    curr_i_level_choose_element_indice = np.random.randint(0, len(curr_i_level_used_eight_positive_elements_indices), size=1, dtype=np.int32)
                    curr_i_level_choose_element = curr_i_level_used_eight_positive_elements_indices[curr_i_level_choose_element_indice]
                    current_batch.append([curr_seq_ID, int(curr_i_level_choose_element)])
                    curr_used_eight_elements_ndx[curr_i_level_choose_element] = False
                    curr_used_nine_elements_ndx[curr_i_level_choose_element] = True
                    continue
                curr_i_level_used_nine_positive_elements = curr_i_level_positive_elements & curr_used_nine_elements_ndx
                if np.count_nonzero(curr_i_level_used_nine_positive_elements) > 0:
                    curr_i_level_used_nine_positive_elements_indices = np.nonzero(curr_i_level_used_nine_positive_elements)[0]
                    curr_i_level_choose_element_indice = np.random.randint(0, len(curr_i_level_used_nine_positive_elements_indices), size=1, dtype=np.int32)
                    curr_i_level_choose_element = curr_i_level_used_nine_positive_elements_indices[curr_i_level_choose_element_indice]
                    current_batch.append([curr_seq_ID, int(curr_i_level_choose_element)])
                    curr_used_nine_elements_ndx[curr_i_level_choose_element] = False
                    curr_used_ten_elements_ndx[curr_i_level_choose_element] = True
                    continue
                curr_i_level_used_ten_positive_elements = curr_i_level_positive_elements & curr_used_ten_elements_ndx
                if np.count_nonzero(curr_i_level_used_ten_positive_elements) > 0:
                    curr_i_level_used_ten_positive_elements_indices = np.nonzero(curr_i_level_used_ten_positive_elements)[0]
                    curr_i_level_choose_element_indice = np.random.randint(0, len(curr_i_level_used_ten_positive_elements_indices), size=1, dtype=np.int32)
                    curr_i_level_choose_element = curr_i_level_used_ten_positive_elements_indices[curr_i_level_choose_element_indice]
                    current_batch.append([curr_seq_ID, int(curr_i_level_choose_element)])
                    curr_used_ten_elements_ndx[curr_i_level_choose_element] = False
                    continue
                raise ValueError('redesign the positive distance list')
        if self.get_positive_type == 'v1':
            for seq_ID, seq_UTM_coords in self.dataset.area_overlap.items():
                self.dataset.area_overlap[seq_ID] = seq_UTM_coords.to('cpu')
        elif self.get_positive_type == 'v2':
            for seq_ID, seq_UTM_coords in self.dataset.pos_vec_vet_coords_tensor.items():
                self.dataset.pos_vec_vet_coords_tensor[seq_ID] = seq_UTM_coords.to('cpu')
        elif self.get_positive_type == 'v3':
            for seq_ID, seq_UTM_coords in self.dataset.UTM_coord_tensor.items():
                self.dataset.UTM_coord_tensor[seq_ID] = seq_UTM_coords.to('cpu')
        t1 = time.time()
        print('generate batches done!')
        print('batchsampler5_21 time cost: ', t1 - t0)