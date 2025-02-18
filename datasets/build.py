from .utils import (make_collate_fn, 
                    make_eval_collate_fn, 
                    BatchSampler1, 
                    BatchSampler2, 
                    BatchSampler3,
                    BatchSampler4,
                    BatchSampler5,
                    BatchSampler3_2,
                    BatchSampler4_2,
                    BatchSampler5_2,
                    BatchSampler5_3,
                    BatchSampler5_4,
                    BatchSampler5_5,
                    BatchSampler5_6,
                    BatchSampler5_7,
                    BatchSampler5_8,
                    BatchSampler5_9,
                    BatchSampler5_10,
                    BatchSampler5_11,
                    BatchSampler5_12,
                    BatchSampler5_13,
                    BatchSampler5_14,
                    BatchSampler5_15,
                    BatchSampler5_16,
                    BatchSampler5_17,
                    BatchSampler5_18,
                    BatchSampler5_19,
                    BatchSampler5_20,
                    BatchSampler5_21,
                    BatchSampler_eval_kitti,
                    make_collate_fn_boreas_1,
                    make_collate_fn_boreas_2, 
                    make_collate_fn_zenseact_1,
                    make_collate_fn_kitti,
                    make_eval_collate_fn_kitti)
from .augmentation import PCTransform, RGBTransform, PCTransform_Pose, RGB_intrinscs_Transform
from .oxford import OxfordDataset, OxfordDatasetEval
from .Boreas import boreas, boreasEval, boreas_v2, boreasEvalv2
from .KITTI import kitti, kittiEval
from .Zenseact import zodframes
from torch.utils.data import BatchSampler, SequentialSampler, DataLoader, RandomSampler
import os
import torch

def build_dataset(cfgs, data_path, train_val_eval):
    if cfgs.data_type == 'oxford':
        dataset_path = os.path.join(data_path, cfgs.dataset_path)
        if train_val_eval == 'train':
            query_filename = cfgs.train_query_filename
            pc_transform = PCTransform(aug_mode=1, num_points=cfgs.num_points)
            image_transform=RGBTransform(aug_mode=1, image_size=cfgs.image_size)
            render_transform=RGBTransform(aug_mode=8, image_size=cfgs.render_size)
            dataset = OxfordDataset(dataset_path=dataset_path,
                    pc_dir=cfgs.pc_dir,
                    rendered_dir=cfgs.rendered_dir,
                    image_dir=cfgs.image_dir,
                    mask_dir=cfgs.mask_dir,
                    query_filename=query_filename,
                    lidar2image_ndx_path=cfgs.lidar2image_ndx_path,
                    image_size=cfgs.image_size,
                    render_size=cfgs.render_size,
                    mask_size=cfgs.mask_size,
                    pc_transform=pc_transform,
                    pc_set_transform=None,
                    image_transform=image_transform,
                    render_transform=render_transform,
                    mask_transform=None,
                    render_view_num=cfgs.render_view_num,
                    use_cloud=cfgs.use_cloud,
                    use_render=cfgs.use_render,
                    use_image=cfgs.use_image,
                    use_mask=cfgs.use_mask,)
        elif train_val_eval == 'val' or train_val_eval == 'eval':
            pc_transform = PCTransform(aug_mode=3, num_points=cfgs.num_points)
            image_transform=RGBTransform(aug_mode=3, image_size=cfgs.image_size)
            render_transform=RGBTransform(aug_mode=5, image_size=cfgs.render_size)
            if train_val_eval == 'val':
                query_filename = cfgs.val_query_filename
            else:
                query_filename = cfgs.eval_query_filename
            dataset = OxfordDatasetEval(dataset_path=dataset_path,
                    pc_dir=cfgs.pc_dir,
                    rendered_dir=cfgs.rendered_dir,
                    image_dir=cfgs.image_dir,
                    mask_dir=cfgs.mask_dir,
                    query_filename=query_filename,
                    lidar2image_ndx_path=cfgs.lidar2image_ndx_path,
                    image_size=cfgs.image_size,
                    render_size=cfgs.render_size,
                    mask_size=cfgs.mask_size,
                    pc_transform=pc_transform,
                    pc_set_transform=None,
                    image_transform=image_transform,
                    render_transform=render_transform,
                    mask_transform=None,
                    render_view_num=cfgs.render_view_num,
                    use_cloud=cfgs.use_cloud,
                    use_render=cfgs.use_render,
                    use_image=cfgs.use_image,
                    use_mask=cfgs.use_mask,)
        else:
            raise ValueError('Invalid train_val_eval: {}'.format(train_val_eval))
    
    elif cfgs.data_type == 'boreas':
        if train_val_eval == 'train':
            if 'train_render_aug_mode' in cfgs.keys(): 
                render_transform=RGB_intrinscs_Transform(aug_mode=cfgs.train_render_aug_mode, 
                                                         image_size=cfgs.render_size, 
                                                         crop_location=cfgs.crop_location)
            else:
                render_transform=None
            pc_transform = PCTransform_Pose(aug_mode=cfgs.train_pc_aug_mode, num_points=cfgs.num_points)
            image_transform=RGB_intrinscs_Transform(aug_mode=cfgs.train_image_aug_mode, image_size=cfgs.image_size, crop_location=cfgs.crop_location)
            if 'img_neighbor_num' in cfgs.keys():
                img_neighbor_num = cfgs.img_neighbor_num
            else:
                img_neighbor_num = 5
            dataset = boreas(data_root=data_path, 
                             raw_dir_name=cfgs.raw_dir_name,
                            pc_dir_name=cfgs.pc_dir_name,
                            rendered_dir_name=cfgs.rendered_dir_name,
                            image_dir_name=cfgs.image_dir_name,
                            mask_dir_name=cfgs.mask_dir_name,
                            tool_name=cfgs.tool_name,
                            query_filename=cfgs.train_query_filename,
                            lidar2image_filename=cfgs.lidar2image_filename,
                            image_size=cfgs.image_size,
                            mask_size=cfgs.mask_size,
                            render_size=cfgs.render_size, 
                            pc_transform=pc_transform, 
                            pc_preprocess=cfgs.pc_preprocess,
                            image_transform=image_transform,
                            render_transform=render_transform,
                            mask_transform=None,
                            render_view_num=None,
                            use_cloud=cfgs.use_cloud, 
                            use_render=cfgs.use_render,
                            use_image=cfgs.use_image,
                            use_mask=cfgs.use_mask,
                            ratio_strategy=cfgs.ratio_strategy,
                            relative_strategy=cfgs.relative_strategy,
                            img_neighbor_num=img_neighbor_num,)
        elif train_val_eval == 'val' or train_val_eval == 'eval':
            pc_transform = PCTransform_Pose(aug_mode=cfgs.eval_pc_aug_mode, num_points=cfgs.num_points)
            image_transform=RGB_intrinscs_Transform(aug_mode=cfgs.eval_image_aug_mode, image_size=cfgs.image_size, crop_location=cfgs.crop_location)
            if 'eval_render_aug_mode' in cfgs.keys():
                render_transform=RGB_intrinscs_Transform(aug_mode=cfgs.eval_render_aug_mode, image_size=cfgs.render_size, crop_location=cfgs.crop_location)
            else:
                render_transform=None
            if train_val_eval == 'val':
                query_filename = cfgs.val_query_filename
            else:
                query_filename = cfgs.eval_query_filename
            if 'img_neighbor_num' in cfgs.keys():
                img_neighbor_num = cfgs.img_neighbor_num
            else:
                img_neighbor_num = 5
            dataset = boreasEval(data_root=data_path, 
                             raw_dir_name=cfgs.raw_dir_name,
                            pc_dir_name=cfgs.pc_dir_name,
                            rendered_dir_name=cfgs.rendered_dir_name,
                            image_dir_name=cfgs.image_dir_name,
                            mask_dir_name=cfgs.mask_dir_name,
                            tool_name=cfgs.tool_name,
                            query_filename=query_filename,
                            lidar2image_filename=cfgs.lidar2image_filename,
                            image_size=cfgs.image_size,
                            mask_size=cfgs.mask_size,
                            render_size=cfgs.render_size, 
                            pc_transform=pc_transform, 
                            pc_preprocess=cfgs.pc_preprocess,
                            image_transform=image_transform,
                            render_transform=render_transform,
                            mask_transform=None,
                            render_view_num=None,
                            use_cloud=cfgs.use_cloud, 
                            use_render=cfgs.use_render,
                            use_image=cfgs.use_image,
                            use_mask=cfgs.use_mask,
                            ratio_strategy=cfgs.ratio_strategy,
                            relative_strategy=cfgs.relative_strategy,
                            img_neighbor_num=img_neighbor_num,)
    elif cfgs.data_type == 'zenseact':
        if train_val_eval == 'train':
            dataset_root = data_path
            pc_transform = PCTransform_Pose(aug_mode=cfgs.train_pc_aug_mode, num_points=cfgs.num_points)
            image_transform=RGB_intrinscs_Transform(aug_mode=cfgs.train_image_aug_mode, image_size=cfgs.image_size)
            dataset = zodframes(dataset_root=dataset_root,
                                raw_name=cfgs.raw_name,
                                pc_name=cfgs.pc_name,
                                image_size=cfgs.image_size,
                                pc_transform=pc_transform,
                                pc_preprocess=cfgs.pc_preprocess,
                                image_transform=image_transform,
                                split='train')
        elif train_val_eval == 'val' or train_val_eval == 'eval':
            dataset_root = data_path
            pc_transform = PCTransform_Pose(aug_mode=cfgs.eval_pc_aug_mode, num_points=cfgs.num_points)
            image_transform=RGB_intrinscs_Transform(aug_mode=cfgs.eval_image_aug_mode, image_size=cfgs.image_size)
            dataset = zodframes(dataset_root=dataset_root,
                                raw_name=cfgs.raw_name,
                                pc_name=cfgs.pc_name,
                                image_size=cfgs.image_size,
                                pc_transform=pc_transform,
                                pc_preprocess=cfgs.pc_preprocess,
                                image_transform=image_transform,
                                split='val')
    elif cfgs.data_type == 'boreas_v2':
        if train_val_eval == 'train':
            pc_transform = PCTransform_Pose(aug_mode=cfgs.train_pc_aug_mode, num_points=cfgs.num_points)
            image_transform=RGB_intrinscs_Transform(aug_mode=cfgs.train_image_aug_mode, image_size=cfgs.image_size, crop_location=cfgs.crop_location)
            if cfgs.train_image_aug_mode == 16:
                image_transform.transform = image_transform.transform[0]
            if 'train_render_aug_mode' in cfgs.keys(): 
                render_transform=RGB_intrinscs_Transform(aug_mode=cfgs.train_render_aug_mode, image_size=cfgs.render_size, crop_location=cfgs.crop_location)
            else:
                render_transform=None
            if 'img_neighbor_num' in cfgs.keys():
                img_neighbor_num = cfgs.img_neighbor_num
            else:
                img_neighbor_num = 5
            if 'dist_caculation_type' in cfgs.keys():
                dist_caculation_type = cfgs.dist_caculation_type
            else:
                dist_caculation_type = 'all_coords_L2'
            if 'use_rgb_depth_label' in cfgs.keys():
                use_rgb_depth_label = cfgs.use_rgb_depth_label
                rgb_depth_label_dir_name = cfgs.rgb_depth_label_dir_name
            else:
                use_rgb_depth_label = False
                rgb_depth_label_dir_name = None
            if 'use_semantic_label' in cfgs.keys() and cfgs.use_semantic_label:
                use_semantic_label = True
                pc_semantic_label_dir_name = cfgs.pc_semantic_label_dir_name
                img_semantic_label_dir_name = cfgs.img_semantic_label_dir_name
                use_label_correspondence_table = cfgs.use_label_correspondence_table
            else:
                use_semantic_label = False
                pc_semantic_label_dir_name = None
                img_semantic_label_dir_name = None
                use_label_correspondence_table = False
            
            if cfgs.overlap_ratio_type == 'area_overlap_ratio':
                overlap_ratio_type = 'area_overlap'
            elif cfgs.overlap_ratio_type == 'pos_vec_vet':
                overlap_ratio_type = 'pos_vec_vet'
            elif cfgs.overlap_ratio_type == 'exp_dist':
                overlap_ratio_type = 'exp_dist'
            elif cfgs.overlap_ratio_type == 'exp_dist_v2':
                overlap_ratio_type = 'exp_dist_v2'
            else:
                overlap_ratio_type = 'points_average_distance'
            dataset = boreas_v2(data_root=data_path, 
                             raw_dir_name=cfgs.raw_dir_name,
                            pc_dir_name=cfgs.pc_dir_name,
                            rendered_dir_name=cfgs.rendered_dir_name,
                            image_dir_name=cfgs.image_dir_name,
                            mask_dir_name=cfgs.mask_dir_name,
                            tool_name=cfgs.tool_name,
                            coords_filename=cfgs.train_coords_filename,
                            minuse_lidar_filename=cfgs.minuse_lidar_filename,
                            positive_distance=cfgs.positive_distance,
                            non_negative_distance=cfgs.non_negative_distance,
                            positive_distance_list=cfgs.positive_distance_list,
                            lidar2image_filename=cfgs.lidar2image_filename,
                            image_size=cfgs.image_size,
                            mask_size=cfgs.mask_size,
                            render_size=cfgs.render_size, 
                            pc_transform=pc_transform, 
                            pc_preprocess=cfgs.pc_preprocess,
                            image_transform=image_transform,
                            render_transform=render_transform,
                            mask_transform=None,
                            render_view_num=None,
                            use_cloud=cfgs.use_cloud, 
                            use_render=cfgs.use_render,
                            use_image=cfgs.use_image,
                            use_mask=cfgs.use_mask,
                            ratio_strategy=cfgs.ratio_strategy,
                            relative_strategy=cfgs.relative_strategy,
                            img_neighbor_num=img_neighbor_num,
                            dist_caculation_type=dist_caculation_type,
                            rgb_depth_label_dir_name=rgb_depth_label_dir_name,
                            use_rgb_depth_label=use_rgb_depth_label,
                            use_semantic_label=use_semantic_label,
                            pc_semantic_label_dir_name=pc_semantic_label_dir_name,
                            img_semantic_label_dir_name=img_semantic_label_dir_name,
                            use_label_correspondence_table=use_label_correspondence_table,
                            overlap_ratio_type=overlap_ratio_type)
        
        elif train_val_eval == 'val' or train_val_eval == 'eval':
            pc_transform = PCTransform_Pose(aug_mode=cfgs.eval_pc_aug_mode, num_points=cfgs.num_points)
            image_transform=RGB_intrinscs_Transform(aug_mode=cfgs.eval_image_aug_mode, image_size=cfgs.image_size, crop_location=cfgs.crop_location)
            if cfgs.eval_image_aug_mode == 17:
                image_transform.transform = image_transform.transform[0]
            if 'eval_render_aug_mode' in cfgs.keys():
                render_transform=RGB_intrinscs_Transform(aug_mode=cfgs.eval_render_aug_mode, image_size=cfgs.render_size, crop_location=cfgs.crop_location)
            else:
                render_transform=None
            if train_val_eval == 'val':
                query_filename = cfgs.val_query_filename
            else:
                query_filename = cfgs.eval_query_filename
            if 'img_neighbor_num' in cfgs.keys():
                img_neighbor_num = cfgs.img_neighbor_num
            else:
                img_neighbor_num = 5
            if 'use_semantic_label_when_inference' in cfgs.keys() and cfgs.use_semantic_label_when_inference:
                use_semantic_label_when_inference = True
                pc_semantic_label_dir_name = cfgs.pc_semantic_label_dir_name
                img_semantic_label_dir_name = cfgs.img_semantic_label_dir_name
                use_label_correspondence_table = cfgs.use_label_correspondence_table
            else:
                use_semantic_label_when_inference = False
                pc_semantic_label_dir_name = None
                img_semantic_label_dir_name = None
                use_label_correspondence_table = False
            if 'boreas_eval_type' in cfgs.keys() and cfgs.boreas_eval_type == 'v2':
                dataset = boreasEvalv2(data_root=data_path, 
                                raw_dir_name=cfgs.raw_dir_name,
                                pc_dir_name=cfgs.pc_dir_name,
                                rendered_dir_name=cfgs.rendered_dir_name,
                                image_dir_name=cfgs.image_dir_name,
                                mask_dir_name=cfgs.mask_dir_name,
                                tool_name=cfgs.tool_name,
                                query_filename=query_filename,
                                lidar2image_filename=cfgs.lidar2image_filename,
                                image_size=cfgs.image_size,
                                mask_size=cfgs.mask_size,
                                render_size=cfgs.render_size, 
                                pc_transform=pc_transform, 
                                pc_preprocess=cfgs.pc_preprocess,
                                image_transform=image_transform,
                                render_transform=render_transform,
                                mask_transform=None,
                                render_view_num=None,
                                use_cloud=cfgs.use_cloud, 
                                use_render=cfgs.use_render,
                                use_image=cfgs.use_image,
                                use_mask=cfgs.use_mask,
                                ratio_strategy=cfgs.ratio_strategy,
                                relative_strategy=cfgs.relative_strategy,
                                img_neighbor_num=img_neighbor_num,
                                true_neighbor_dist=cfgs.true_neighbor_dist,
                                coords_filename=cfgs.all_coords_filename,
                                use_semantic_label_when_inference=use_semantic_label_when_inference,
                                pc_semantic_label_dir_name=pc_semantic_label_dir_name,
                                img_semantic_label_dir_name=img_semantic_label_dir_name,
                                use_label_correspondence_table=use_label_correspondence_table)
            else:
                dataset = boreasEval(data_root=data_path, 
                                raw_dir_name=cfgs.raw_dir_name,
                                pc_dir_name=cfgs.pc_dir_name,
                                rendered_dir_name=cfgs.rendered_dir_name,
                                image_dir_name=cfgs.image_dir_name,
                                mask_dir_name=cfgs.mask_dir_name,
                                tool_name=cfgs.tool_name,
                                query_filename=query_filename,
                                lidar2image_filename=cfgs.lidar2image_filename,
                                image_size=cfgs.image_size,
                                mask_size=cfgs.mask_size,
                                render_size=cfgs.render_size, 
                                pc_transform=pc_transform, 
                                pc_preprocess=cfgs.pc_preprocess,
                                image_transform=image_transform,
                                render_transform=render_transform,
                                mask_transform=None,
                                render_view_num=None,
                                use_cloud=cfgs.use_cloud, 
                                use_render=cfgs.use_render,
                                use_image=cfgs.use_image,
                                use_mask=cfgs.use_mask,
                                ratio_strategy=cfgs.ratio_strategy,
                                relative_strategy=cfgs.relative_strategy,
                                img_neighbor_num=img_neighbor_num,
                                use_semantic_label_when_inference=use_semantic_label_when_inference,
                                pc_semantic_label_dir_name=pc_semantic_label_dir_name,
                                img_semantic_label_dir_name=img_semantic_label_dir_name,
                                use_label_correspondence_table=use_label_correspondence_table)
    elif cfgs.data_type == 'kitti':
        if train_val_eval == 'train':
            pc_transform = PCTransform_Pose(aug_mode=cfgs.train_pc_aug_mode, num_points=cfgs.num_points)
            image_transform=RGB_intrinscs_Transform(aug_mode=cfgs.train_image_aug_mode, image_size=cfgs.image_size, crop_location=cfgs.crop_location)
            if cfgs.train_image_aug_mode == 16:
                image_transform.transform = image_transform.transform[0]
            if 'dist_caculation_type' in cfgs.keys():
                dist_caculation_type = cfgs.dist_caculation_type
            else:
                dist_caculation_type = 'all_coords_L2'
            
            if cfgs.overlap_ratio_type == 'area_overlap':
                overlap_ratio_type = 'area_overlap'
            elif cfgs.overlap_ratio_type == 'pos_vec_vet':
                overlap_ratio_type = 'pos_vec_vet'
            elif cfgs.overlap_ratio_type == 'exp_dist':
                overlap_ratio_type = 'exp_dist'
            elif cfgs.overlap_ratio_type == 'exp_dist_v2':
                overlap_ratio_type = 'exp_dist_v2'
            else:
                overlap_ratio_type = 'points_average_distance'

            if 'use_range' in cfgs.keys() and cfgs.use_range:
                use_range = True
                range_dir_name = cfgs.range_dir_name
                range_transform = RGB_intrinscs_Transform(aug_mode=cfgs.train_range_aug_mode, image_size=cfgs.range_img_size)
                if cfgs.train_range_aug_mode == 16 or cfgs.train_range_aug_mode == 17:
                    range_transform.transform = range_transform.transform[0]
            else:
                use_range = False
                range_dir_name = None
                range_transform = None
            
            if 'use_memory_bank' in cfgs.keys():
                use_memory_bank = cfgs.use_memory_bank
            else:
                use_memory_bank = False
            
            if 'use_pc_bev' in cfgs.keys() and cfgs.use_pc_bev:
                use_pc_bev = cfgs.use_pc_bev
                pc_bev_dir_name = cfgs.pc_bev_dir_name
                pc_bev_transform = RGB_intrinscs_Transform(aug_mode=cfgs.train_pc_bev_aug_mode, image_size=cfgs.pc_bev_img_size)
                if cfgs.train_pc_bev_aug_mode == 16 or cfgs.train_pc_bev_aug_mode == 17:
                    pc_bev_transform.transform = pc_bev_transform.transform[0]
            else:
                use_pc_bev = False
                pc_bev_dir_name = None
                pc_bev_transform = None
            
            if 'use_image_bev' in cfgs.keys() and cfgs.use_image_bev:
                use_image_bev = cfgs.use_image_bev
                image_bev_dir_name = cfgs.image_bev_dir_name
                image_bev_transform = RGB_intrinscs_Transform(aug_mode=cfgs.train_image_bev_aug_mode, image_size=cfgs.image_bev_img_size)
                if cfgs.train_image_bev_aug_mode == 16 or cfgs.train_image_bev_aug_mode == 17:
                    image_bev_transform.transform = image_bev_transform.transform[0]
            else:
                use_image_bev = False
                image_bev_dir_name = None
                image_bev_transform = None
            
            if 'overlap_ratio_cfgs' in cfgs.keys():
                overlap_ratio_cfgs = cfgs.overlap_ratio_cfgs
            else:
                overlap_ratio_cfgs = None
            
            kitti_data_root = os.path.join(data_path, cfgs.kitti_data_root)
            semantickitti_data_root = os.path.join(data_path, cfgs.semantickitti_data_root)
            dataset = kitti(data_root=kitti_data_root, 
                            pose_root=semantickitti_data_root,
                            raw_dir_name=cfgs.raw_dir_name,
                            pc_dir_name=cfgs.pc_dir_name,
                            image_dir_name=cfgs.image_dir_name,
                            coords_filename=cfgs.train_coords_filename,
                            image_size=cfgs.image_size,
                            pc_transform=pc_transform, 
                            image_transform=image_transform,
                            use_cloud=cfgs.use_cloud, 
                            use_image=cfgs.use_image,
                            dist_caculation_type=dist_caculation_type,
                            overlap_ratio_type=overlap_ratio_type,
                            sequence_list=cfgs.train_sequence_list,
                            use_range=use_range,
                            range_dir_name=range_dir_name,
                            range_transform=range_transform,
                            use_memory_bank=use_memory_bank,
                            use_pc_bev=use_pc_bev,
                            pc_bev_dir_name=pc_bev_dir_name,
                            pc_bev_transform=pc_bev_transform,
                            use_image_bev=use_image_bev,
                            image_bev_dir_name=image_bev_dir_name,
                            image_bev_transform=image_bev_transform,
                            overlap_ratio_cfgs=overlap_ratio_cfgs,)
        elif train_val_eval == 'val' or train_val_eval == 'eval':
            pc_transform = PCTransform_Pose(aug_mode=cfgs.eval_pc_aug_mode, num_points=cfgs.num_points)
            image_transform=RGB_intrinscs_Transform(aug_mode=cfgs.eval_image_aug_mode, image_size=cfgs.image_size, crop_location=cfgs.crop_location)
            if cfgs.eval_image_aug_mode == 17:
                image_transform.transform = image_transform.transform[0]
            kitti_data_root = os.path.join(data_path, cfgs.kitti_data_root)
            semantickitti_data_root = os.path.join(data_path, cfgs.semantickitti_data_root)
            if 'use_range' in cfgs.keys() and cfgs.use_range:
                use_range = True
                range_dir_name = cfgs.range_dir_name
                range_transform = RGB_intrinscs_Transform(aug_mode=cfgs.eval_range_aug_mode, image_size=cfgs.range_img_size)
                if cfgs.eval_range_aug_mode == 16 or cfgs.eval_range_aug_mode == 17:
                    range_transform.transform = range_transform.transform[0]
            else:
                use_range = False
                range_dir_name = None
                range_transform = None
            
            if 'use_pc_bev' in cfgs.keys() and cfgs.use_pc_bev:
                use_pc_bev = cfgs.use_pc_bev
                pc_bev_dir_name = cfgs.pc_bev_dir_name
                pc_bev_transform = RGB_intrinscs_Transform(aug_mode=cfgs.eval_pc_bev_aug_mode, image_size=cfgs.pc_bev_img_size)
                if cfgs.eval_pc_bev_aug_mode == 16 or cfgs.eval_pc_bev_aug_mode == 17:
                    pc_bev_transform.transform = pc_bev_transform.transform[0]
            else:
                use_pc_bev = False
                pc_bev_dir_name = None
                pc_bev_transform = None
            
            if 'use_image_bev' in cfgs.keys() and cfgs.use_image_bev:
                use_image_bev = cfgs.use_image_bev
                image_bev_dir_name = cfgs.image_bev_dir_name
                image_bev_transform = RGB_intrinscs_Transform(aug_mode=cfgs.eval_image_bev_aug_mode, image_size=cfgs.image_bev_img_size)
                if cfgs.eval_image_bev_aug_mode == 16 or cfgs.eval_image_bev_aug_mode == 17:
                    image_bev_transform.transform = image_bev_transform.transform[0]
            else:
                use_image_bev = False
                image_bev_dir_name = None
                image_bev_transform = None
            
            dataset = kittiEval(data_root=kitti_data_root, 
                            raw_dir_name=cfgs.raw_dir_name,
                            pc_dir_name=cfgs.pc_dir_name,
                            image_dir_name=cfgs.image_dir_name,
                            coords_filename=cfgs.test_coords_filename,
                            image_size=cfgs.image_size,
                            pc_transform=pc_transform, 
                            image_transform=image_transform,
                            use_cloud=cfgs.use_cloud, 
                            use_image=cfgs.use_image,
                            sequence_list=cfgs.test_sequence_list,
                            true_neighbour_dist=cfgs.true_neighbour_dist,
                            use_range=use_range,
                            range_dir_name=range_dir_name,
                            range_transform=range_transform,
                            use_pc_bev=use_pc_bev,
                            pc_bev_dir_name=pc_bev_dir_name,
                            pc_bev_transform=pc_bev_transform,
                            use_image_bev=use_image_bev,
                            image_bev_dir_name=image_bev_dir_name,
                            image_bev_transform=image_bev_transform,)
        
    else: 
        raise ValueError('Invalid data_type: {}'.format(cfgs.data_type))
    
    return dataset

def build_sampler(cfgs, dataset, train_val_eval):
    if train_val_eval == 'train':
        if cfgs.sampler_type == 'none':
            sampler = None
        elif cfgs.sampler_type == 'random':
            sampler = RandomSampler(dataset)
        elif cfgs.sampler_type == 'distributed':
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True, seed=3407, drop_last=True)
    elif train_val_eval == 'val' or train_val_eval == 'eval':
        sampler = SequentialSampler(dataset)
    else:
        raise ValueError('Invalid train_val_eval: {}'.format(train_val_eval))
    
    return sampler

def build_batch_sampler(cfgs, dataset, sampler, start_epoch, train_val_eval, data_type):
    if train_val_eval == 'train':
        if cfgs.batch_sampler_type == 'ensure_pos':
            batch_sampler = BatchSampler1(dataset,
                                        train_val='train',
                                        num_k=cfgs.num_k,
                                        sampler=sampler,
                                        batch_size=cfgs.batch_size,
                                        start_epoch=start_epoch,
                                        batch_size_limit=cfgs.batch_size_limit,
                                        batch_expansion_rate=cfgs.batch_expansion_rate,
                                        max_batches=None)
        elif cfgs.batch_sampler_type == 'ensure_k_pos':
            batch_sampler = BatchSampler2(
                                        dataset,
                                        train_val='train',
                                        num_k=cfgs.num_k,
                                        sampler=sampler,
                                        batch_size=cfgs.batch_size,
                                        start_epoch=start_epoch,
                                        batch_size_limit=cfgs.batch_size_limit,
                                        batch_expansion_rate=cfgs.batch_expansion_rate,
                                        max_batches=None
                                        )
        elif cfgs.batch_sampler_type == 'ensure_k_levels_pos':
            batch_sampler = BatchSampler3(dataset,
                                        num_k=cfgs.num_k,
                                        batch_size=cfgs.batch_size,
                                        safe_elem_region=cfgs.safe_elem_region,
                                        start_epoch=start_epoch,
                                        iter_per_epoch=cfgs.iter_per_epoch,)
        elif cfgs.batch_sampler_type == 'ordinary':
            batch_sampler = BatchSampler(sampler=sampler,
                                    batch_size=cfgs.batch_size,
                                    drop_last=cfgs.drop_last)
        elif cfgs.batch_sampler_type == 'ensure_k_levels_pos_uniform':
            batch_sampler = BatchSampler4(dataset,
                                        num_k=cfgs.num_k,
                                        batch_size=cfgs.batch_size,
                                        start_epoch=start_epoch,
                                        iter_per_epoch=cfgs.iter_per_epoch,)
        elif cfgs.batch_sampler_type == 'ensure_k_levels_pos_uniform_v2':
            batch_sampler = BatchSampler4_2(dataset,
                                        num_k=cfgs.num_k,
                                        batch_size=cfgs.batch_size,
                                        start_epoch=start_epoch,
                                        iter_per_epoch=cfgs.iter_per_epoch)
        elif cfgs.batch_sampler_type == 'ensure_k_levels_pos_uniform_multi':
            batch_sampler = BatchSampler5(dataset,
                                        num_k=cfgs.num_k,
                                        batch_size=cfgs.batch_size,
                                        start_epoch=start_epoch,
                                        iter_per_epoch=cfgs.iter_per_epoch)
        elif cfgs.batch_sampler_type == 'ensure_k_levels_pos_v2':
            batch_sampler = BatchSampler3_2(dataset,
                                        num_k=cfgs.num_k,
                                        batch_size=cfgs.batch_size,
                                        safe_elem_region=cfgs.safe_elem_region,
                                        start_epoch=start_epoch,
                                        iter_per_epoch=cfgs.iter_per_epoch,
                                        level1_distance=10.0)
        elif cfgs.batch_sampler_type == 'ensure_k_levels_pos_uniform_multi_v2':
            batch_sampler = BatchSampler5_2(dataset,
                                        num_k=cfgs.num_k,
                                        batch_size=cfgs.batch_size,
                                        start_epoch=start_epoch,
                                        iter_per_epoch=cfgs.iter_per_epoch)
        elif cfgs.batch_sampler_type == 'ensure_k_levels_pos_uniform_multi_v3':
            batch_sampler = BatchSampler5_3(dataset,
                                        num_k=cfgs.num_k,
                                        batch_size=cfgs.batch_size,
                                        start_epoch=start_epoch,
                                        iter_per_epoch=cfgs.iter_per_epoch)
        elif cfgs.batch_sampler_type == 'ensure_k_levels_pos_uniform_multi_v4':
            batch_sampler = BatchSampler5_4(dataset,
                                        num_k=cfgs.num_k,
                                        batch_size=cfgs.batch_size,
                                        start_epoch=start_epoch)
        elif cfgs.batch_sampler_type == 'ensure_k_levels_pos_uniform_multi_v5':
            batch_sampler = BatchSampler5_5(dataset,
                                        num_k=cfgs.num_k,
                                        batch_size=cfgs.batch_size,
                                        start_epoch=start_epoch)
        elif cfgs.batch_sampler_type == 'ensure_k_levels_pos_uniform_multi_v6':
            batch_sampler = BatchSampler5_6(dataset,
                                        num_k=cfgs.num_k,
                                        batch_size=cfgs.batch_size,
                                        start_epoch=start_epoch)
        elif cfgs.batch_sampler_type == 'ensure_k_levels_pos_uniform_multi_v7':
            batch_sampler = BatchSampler5_7(dataset,
                                        num_k=cfgs.num_k,
                                        batch_size=cfgs.batch_size,
                                        start_epoch=start_epoch)
        elif cfgs.batch_sampler_type == 'ensure_k_levels_pos_uniform_multi_v8':
            batch_sampler = BatchSampler5_8(dataset,
                                        num_k=cfgs.num_k,
                                        batch_size=cfgs.batch_size,
                                        start_epoch=start_epoch,
                                        useout_times=cfgs.useout_times,
                                        interval_num=cfgs.interval_num)
        elif cfgs.batch_sampler_type == 'ensure_k_levels_pos_uniform_multi_v9':
            batch_sampler = BatchSampler5_9(dataset,
                                        num_k=cfgs.num_k,
                                        batch_size=cfgs.batch_size,
                                        start_epoch=start_epoch,
                                        useout_times=cfgs.useout_times)
        elif cfgs.batch_sampler_type == 'ensure_k_levels_pos_uniform_multi_v10':
            batch_sampler = BatchSampler5_10(dataset,
                                        num_k=cfgs.num_k,
                                        batch_size=cfgs.batch_size,
                                        start_epoch=start_epoch,
                                        useout_times=cfgs.useout_times)
        elif cfgs.batch_sampler_type == 'ensure_k_levels_pos_uniform_multi_v11':
            batch_sampler = BatchSampler5_11(dataset,
                                        num_k=cfgs.num_k,
                                        batch_size=cfgs.batch_size,
                                        start_epoch=start_epoch,
                                        useout_times=cfgs.useout_times,
                                        interval_num=cfgs.interval_num)
        elif cfgs.batch_sampler_type == 'ensure_k_levels_pos_uniform_multi_v12':
            batch_sampler = BatchSampler5_12(dataset,
                                        num_k=cfgs.num_k,
                                        batch_size=cfgs.batch_size,
                                        start_epoch=start_epoch,
                                        useout_times=cfgs.useout_times,
                                        interval_num=cfgs.interval_num)
        elif cfgs.batch_sampler_type == 'ensure_k_levels_pos_uniform_multi_v13':
            batch_sampler = BatchSampler5_13(dataset,
                                        num_k=cfgs.num_k,
                                        batch_size=cfgs.batch_size,
                                        start_epoch=start_epoch,
                                        useout_times=cfgs.useout_times,
                                        interval_num=cfgs.interval_num)
        elif cfgs.batch_sampler_type == 'ensure_k_levels_pos_uniform_multi_v14':
            batch_sampler = BatchSampler5_14(dataset,
                                        num_k=cfgs.num_k,
                                        batch_size=cfgs.batch_size,
                                        start_epoch=start_epoch,
                                        useout_times=cfgs.useout_times,
                                        interval_num=cfgs.interval_num)
        elif cfgs.batch_sampler_type == 'ensure_k_levels_pos_uniform_multi_v15':
            batch_sampler = BatchSampler5_15(dataset,
                                        num_k=cfgs.num_k,
                                        batch_size=cfgs.batch_size,
                                        start_epoch=start_epoch,
                                        useout_times=cfgs.useout_times)
        elif cfgs.batch_sampler_type == 'ensure_k_levels_pos_uniform_multi_v16':
            batch_sampler = BatchSampler5_16(dataset,
                                        num_k=cfgs.num_k,
                                        batch_size=cfgs.batch_size,
                                        start_epoch=start_epoch,
                                        useout_times=cfgs.useout_times)
        elif cfgs.batch_sampler_type == 'ensure_k_levels_pos_uniform_multi_v17':
            batch_sampler = BatchSampler5_17(dataset,
                                        num_k=cfgs.num_k,
                                        batch_size=cfgs.batch_size,
                                        start_epoch=start_epoch,
                                        useout_times=cfgs.useout_times,
                                        interval_num=cfgs.interval_num)
        elif cfgs.batch_sampler_type == 'ensure_k_levels_pos_uniform_multi_v18':
            batch_sampler = BatchSampler5_18(dataset,
                                        num_k=cfgs.num_k,
                                        batch_size=cfgs.batch_size,
                                        start_epoch=start_epoch,
                                        useout_times=cfgs.useout_times,
                                        interval_num=cfgs.interval_num)
        elif cfgs.batch_sampler_type == 'ensure_k_levels_pos_uniform_multi_v19':
            if 'batch_shuffle' in cfgs.keys():
                batch_shuffle = cfgs.batch_shuffle
            else:
                batch_shuffle = False
            batch_sampler = BatchSampler5_19(dataset,
                                        num_k=cfgs.num_k,
                                        batch_size=cfgs.batch_size,
                                        start_epoch=start_epoch,
                                        useout_times=cfgs.useout_times,
                                        interval_num=cfgs.interval_num,
                                        batch_shuffle=batch_shuffle)
        elif cfgs.batch_sampler_type == 'ensure_k_levels_pos_uniform_multi_v20':
            batch_sampler = BatchSampler5_20(dataset,
                                        num_k=cfgs.num_k,
                                        batch_size=cfgs.batch_size,
                                        start_epoch=start_epoch,
                                        interval_num=cfgs.interval_num,
                                        expansion_neighbor=cfgs.expansion_neighbor,
                                        base_batch_num=cfgs.base_batch_num,
                                        new_epoch_data_ratio_type=cfgs.new_epoch_data_ratio_type,
                                        new_higher_layer=cfgs.new_higher_layer,
                                        batch_expansion_strategy=cfgs.batch_expansion_strategy,)
        elif cfgs.batch_sampler_type == 'ensure_k_levels_pos_uniform_multi_v21':
            batch_sampler = BatchSampler5_21(dataset,
                                        num_k=cfgs.num_k,
                                        batch_size=cfgs.batch_size,
                                        start_epoch=start_epoch,
                                        useout_times=cfgs.useout_times,
                                        interval_num=cfgs.interval_num,
                                        get_positive_type=cfgs.get_positive_type,)
        else:
            raise ValueError('Invalid batch_sampler_type: {}'.format(cfgs.batch_sampler_type))
    elif train_val_eval == 'val' or train_val_eval == 'eval':
        if data_type == 'kitti':
            batch_sampler = BatchSampler_eval_kitti(dataset,
                                             batch_size=cfgs.eval_batch_size,
                                             sampler=sampler,)
        else:
            batch_sampler = BatchSampler(sampler=sampler,
                                        batch_size=cfgs.eval_batch_size,
                                        drop_last=False)
    else:
        raise ValueError('Invalid train_val_eval: {}'.format(train_val_eval))
            
    return batch_sampler

def build_collate_fn(cfgs, dataset, train_val_eval):
    if cfgs.collect_fn_type == '1':
        if train_val_eval == 'train':
            collate_fn = make_collate_fn(dataset)
        elif train_val_eval == 'val' or train_val_eval == 'eval':
            collate_fn = make_eval_collate_fn(dataset)
        else:
            raise ValueError('Invalid train_val_eval: {}'.format(train_val_eval))
    elif cfgs.collect_fn_type == '2':
        if train_val_eval == 'train':
            collate_fn = make_collate_fn_boreas_1(dataset)
        elif train_val_eval == 'val' or train_val_eval == 'eval':
            collate_fn = make_eval_collate_fn(dataset)
        else:
            raise ValueError('Invalid train_val_eval: {}'.format(train_val_eval))
    elif cfgs.collect_fn_type == '3':
        if train_val_eval == 'train':
            collate_fn = make_collate_fn_zenseact_1(dataset)
        elif train_val_eval == 'val' or train_val_eval == 'eval':
            collate_fn = make_eval_collate_fn(dataset)
        else:
            raise ValueError('Invalid train_val_eval: {}'.format(train_val_eval))
    elif cfgs.collect_fn_type == '4':
        if train_val_eval == 'train':
            collate_fn = make_collate_fn_boreas_2(dataset)
        elif train_val_eval == 'val' or train_val_eval == 'eval':
            collate_fn = make_eval_collate_fn(dataset)
        else:
            raise ValueError('Invalid train_val_eval: {}'.format(train_val_eval))
    elif cfgs.collect_fn_type == '5':
        if train_val_eval == 'train':
            collate_fn = make_collate_fn_kitti(dataset)
        elif train_val_eval == 'val' or train_val_eval == 'eval':
            collate_fn = make_eval_collate_fn_kitti(dataset)
        else:
            raise ValueError('Invalid train_val_eval: {}'.format(train_val_eval))
    else:
        raise ValueError('Invalid collect_fn_type: {}'.format(cfgs.collect_fn_type))
    return collate_fn

def make_dataloader(cfgs, data_path, start_epoch, train_val_eval):
    dataset = build_dataset(cfgs.dataset_cfgs, data_path, train_val_eval)
    sampler = build_sampler(cfgs.sampler_cfgs, dataset, train_val_eval)
    batch_sampler = build_batch_sampler(cfgs.batch_sampler_cfgs, dataset, sampler, start_epoch, train_val_eval, cfgs.dataset_cfgs.data_type)
    collate_fn = build_collate_fn(cfgs.collect_fn_cfgs, dataset, train_val_eval)
    dataloader = DataLoader(dataset=dataset,
                            batch_sampler=batch_sampler,
                            num_workers=cfgs.num_workers,
                            collate_fn=collate_fn,
                            pin_memory=False)
    if train_val_eval == 'train':
        return dataloader
    elif train_val_eval == 'val' or train_val_eval == 'eval':
        return dataloader, dataset
    else:
        raise ValueError('Invalid train_val_eval: {}'.format(train_val_eval))