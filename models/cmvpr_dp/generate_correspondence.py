import torch
import torch_scatter
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import numpy as np
from torch import tensor
import torch_sparse
import time
from .scatter_gather import keops_knn
import pointops
import copy
import itertools


def generate_multi_correspondence_phase4(data_dict, device, img_feats_list, pc_feats_list, pc_coords_list, img_H, img_W, cfgs):
    B = data_dict["clouds"].shape[0]
    fpoints_num = pc_coords_list[0].shape[1]
    f_points = pc_coords_list[0]
    clouds_2_fpoints_dis = torch.cdist(data_dict["clouds"], f_points) # Produces (B, 4096, f_points_num) tensor
    _, clouds_2_fpoints_idx = torch.topk(clouds_2_fpoints_dis, k=1, dim=2, largest=False, sorted=False) # Produces (B, 4096, 1) tensor
    clouds_2_fpoints_idx = clouds_2_fpoints_idx.squeeze(-1) # Produces (B, 4096) tensor
    original_pc_2_fpoints = torch.gather(input=clouds_2_fpoints_idx.unsqueeze(1).expand(-1, B, -1),
                                        dim=-1,
                                        index=data_dict["original_pc_2_many_1"][..., 0]) # (B, B, original_points_num)
    
    img_H_mesh = torch.arange(0, img_H, device=device).type(torch.float32)
    img_W_mesh = torch.arange(0, img_W, device=device).type(torch.float32)
    img_H_mesh = img_H_mesh.unsqueeze(1).unsqueeze(2).expand(-1, img_W, -1)
    img_W_mesh = img_W_mesh.unsqueeze(0).unsqueeze(2).expand(img_H, -1, -1)
    img_mesh = torch.cat((img_H_mesh, img_W_mesh), dim=-1) # Produces (img_H, img_W, 2)
    img_mesh = img_mesh.flatten(0, 1) # Produces (img_H * img_W, 2) tensor
    f_img_feats = img_feats_list[0]
    fimg_H, fimg_W = f_img_feats.shape[2:]
    fimg_H_mesh = torch.arange(0, fimg_H, device=device)
    fimg_W_mesh = torch.arange(0, fimg_W, device=device)
    img_2_fimg_scale_H = img_H * 1.0 / fimg_H
    img_2_fimg_scale_W = img_W * 1.0 / fimg_W
    delta_H = img_2_fimg_scale_H / 2 - 0.5
    delta_W = img_2_fimg_scale_W / 2 - 0.5
    fimg_H_mesh = fimg_H_mesh * img_2_fimg_scale_H + delta_H
    fimg_W_mesh = fimg_W_mesh * img_2_fimg_scale_W + delta_W
    fimg_H_mesh = fimg_H_mesh.unsqueeze(1).unsqueeze(2).expand(-1, fimg_W, -1) # Produces (fimg_H, fimg_W, 1) tensor
    fimg_W_mesh = fimg_W_mesh.unsqueeze(0).unsqueeze(2).expand(fimg_H, -1, -1) # Produces (fimg_H, fimg_W, 1) tensor
    fimg_mesh = torch.cat((fimg_H_mesh, fimg_W_mesh), dim=-1) # Produces (fimg_H, fimg_W, 2) tensor
    fimg_mesh = fimg_mesh.flatten(0, 1) # Produces (fimg_H * fimg_W, 2) tensor
    img_2_fimg_dis = torch.cdist(img_mesh, fimg_mesh) # Produces (img_H * img_W, fimg_H * fimg_W) tensor
    _, img_2_fimg_idx = torch.topk(img_2_fimg_dis, k=1, dim=1, largest=False, sorted=False) # Produces (img_H * img_W, 1) tensor
    img_2_fimg_idx = img_2_fimg_idx.squeeze(-1).unsqueeze(0).unsqueeze(0).expand(B, B, -1) # Produces (B, B, img_H * img_W) tensor
    original_pc_2_fimg = torch.gather(input=img_2_fimg_idx,
                                    dim=-1,
                                    index=data_dict["original_pc_2_many_1"][..., 1]) # Produces (B, B, original_points_num)

    fimg_2_fpoints_map = original_pc_2_fimg * fpoints_num + original_pc_2_fpoints
    overlap_mask = torch.logical_and(data_dict["original_pc_2_many_2"][..., 0], data_dict["original_pc_2_many_2"][..., 1])
    fimg_2_fpoints_map.masked_fill_(~overlap_mask, fpoints_num * fimg_H * fimg_W - 1)
    overlap_mask_num = torch.count_nonzero(~overlap_mask, dim=-1) # (B, B)
    fimg_2_fpoints_num_pt = torch_scatter.scatter_sum(torch.ones_like(fimg_2_fpoints_map, dtype=torch.int32), 
                                                    fimg_2_fpoints_map,
                                                    dim=-1,
                                                    dim_size=fimg_H * fimg_W * fpoints_num) # produces (B, B, fimg_H * fimg_W * fpoints_num)
    fimg_2_fpoints_num_pt[..., -1] -= overlap_mask_num
    fimg_2_fpoints_num_pt = fimg_2_fpoints_num_pt.reshape(B, B, fimg_H * fimg_W, fpoints_num)

    fimg_2_many_points_num_pt_list = [fimg_2_fpoints_num_pt]
    for i in range(len(pc_coords_list) - 1):
        fpoints_2_curr_points_dis = torch.cdist(f_points, pc_coords_list[i + 1]) # produces (B, fpoints_num, curr_points_num)
        _, fpoints_2_curr_points_idx = torch.topk(fpoints_2_curr_points_dis, k=1, dim=2, largest=False, sorted=False) # Produces (B, fpoints_num, 1) tensor
        fimg_2_curr_points_num_pt = torch_scatter.scatter_sum(fimg_2_fpoints_num_pt, 
                                                            fpoints_2_curr_points_idx.squeeze(-1).unsqueeze(1), 
                                                            dim=-1, 
                                                            dim_size=pc_coords_list[i + 1].shape[1]) # produces (B, B, fimg_H * fimg_W, curr_points_num)
        fimg_2_many_points_num_pt_list.append(fimg_2_curr_points_num_pt)
    
    fimg_2_many_img_idx_list = []
    for i in range(len(img_feats_list) - 1):
        curr_img_H, curr_img_W = img_feats_list[i+1].shape[2:]
        curr_img_H_mesh = torch.arange(0, curr_img_H, device=device)
        curr_img_W_mesh = torch.arange(0, curr_img_W, device=device)
        img_2_curr_img_scale_H = img_H * 1.0 / curr_img_H
        img_2_curr_img_scale_W = img_W * 1.0 / curr_img_W
        delta_H = img_2_curr_img_scale_H / 2 - 0.5
        delta_W = img_2_curr_img_scale_W / 2 - 0.5
        curr_img_H_mesh = curr_img_H_mesh * img_2_curr_img_scale_H + delta_H
        curr_img_W_mesh = curr_img_W_mesh * img_2_curr_img_scale_W + delta_W
        curr_img_H_mesh = curr_img_H_mesh.unsqueeze(1).unsqueeze(2).expand(-1, curr_img_W, -1) # Produces (curr_img_H, curr_img_W, 1) tensor
        curr_img_W_mesh = curr_img_W_mesh.unsqueeze(0).unsqueeze(2).expand(curr_img_H, -1, -1) # Produces (curr_img_H, curr_img_W, 1) tensor
        curr_img_mesh = torch.cat((curr_img_H_mesh, curr_img_W_mesh), dim=-1) # Produces (curr_img_H, curr_img_W, 2) tensor
        curr_img_mesh = curr_img_mesh.flatten(0, 1) # Produces (curr_img_H * curr_img_W, 2) tensor
        fimg_2_curr_img_dis = torch.cdist(fimg_mesh, curr_img_mesh) # Produces (fimg_H * fimg_W, curr_img_H * curr_img_W) tensor
        _, fimg_2_curr_img_idx = torch.topk(fimg_2_curr_img_dis, k=1, dim=1, largest=False, sorted=False) # Produces (fimg_H * fimg_W, 1) tensor
        fimg_2_curr_img_idx = fimg_2_curr_img_idx.squeeze(-1).unsqueeze(0).unsqueeze(0).expand(B, B, -1) # Produces (B, B, fimg_H * fimg_W) tensor
        fimg_2_many_img_idx_list.append(fimg_2_curr_img_idx)
    
    fpoints_2_many_points_list = []
    for i in range(len(pc_feats_list) - 1):
        fpoints_2_curr_points_dis = torch.cdist(f_points, pc_coords_list[i + 1]) # produces (B, fpoints_num, curr_points_num)
        _, fpoints_2_curr_points_idx = torch.topk(fpoints_2_curr_points_dis, k=1, dim=2, largest=False, sorted=False) # Produces (B, fpoints_num, 1) tensor
        fpoints_2_many_points_list.append(fpoints_2_curr_points_idx.squeeze(-1).squeeze(1).expand(-1, B, -1))
    
    many_img_2_many_points_num_pt_list = [fimg_2_many_points_num_pt_list]
    for i in range(len(img_feats_list) - 1):
        curr_img_2_many_points_num_pt_list = []
        for j in range(len(pc_coords_list)):
            curr_img_2_curr_points_num_pt = torch_scatter.scatter_sum(fimg_2_many_points_num_pt_list[j], 
                                                                    fimg_2_many_img_idx_list[i], 
                                                                    dim=-2, 
                                                                    dim_size=img_feats_list[i+1].shape[2] * img_feats_list[i+1].shape[3]) # produces (B, B, curr_img_H * curr_img_W, curr_points_num)
            curr_img_2_many_points_num_pt_list.append(curr_img_2_curr_points_num_pt)
        many_img_2_many_points_num_pt_list.append(curr_img_2_many_points_num_pt_list)
    
    original_pc_2_fpoints.masked_fill_(~data_dict["original_pc_2_many_2"][..., 1], fpoints_num - 1) # can't be reuse again
    remove_mask_num = torch.count_nonzero(~data_dict["original_pc_2_many_2"][..., 1], dim=-1) # (B, B)
    fpoints_num_pt = torch_scatter.scatter_sum(torch.ones_like(original_pc_2_fpoints, dtype=torch.int32),
                                            original_pc_2_fpoints,
                                            dim=-1,
                                            dim_size=fpoints_num) # produce (B, B, fpoints_num)
    fpoints_num_pt[..., -1] -= remove_mask_num # produce (B, B, fpoints_num)
    original_pc_2_fimg.masked_fill_(~data_dict["original_pc_2_many_2"][..., 0], fimg_H * fimg_W - 1) # can't be reuse again
    non_qualified_mask_num = torch.count_nonzero(~data_dict["original_pc_2_many_2"][..., 0], dim=-1) # produce (B, B)
    fimg_num_pt = torch_scatter.scatter_sum(torch.ones_like(original_pc_2_fimg, dtype=torch.int32),
                                            original_pc_2_fimg,
                                            dim=-1,
                                            dim_size=fimg_H * fimg_W) # produce (B, B, fimg_H * fimg_W)
    fimg_num_pt[..., -1] -= non_qualified_mask_num

    img_num_pt_list = [fimg_num_pt]
    for i in range(len(img_feats_list) - 1):
        curr_img_num_pt = torch_scatter.scatter_sum(fimg_num_pt, 
                                                    fimg_2_many_img_idx_list[i], 
                                                    dim=-1, 
                                                    dim_size=img_feats_list[i+1].shape[2] * img_feats_list[i+1].shape[3])
        img_num_pt_list.append(curr_img_num_pt)
    points_num_pt_list = [fpoints_num_pt]
    for i in range(len(pc_coords_list) - 1):
        curr_points_num_pt = torch_scatter.scatter_sum(fpoints_num_pt, 
                                                    fpoints_2_many_points_list[i], 
                                                    dim=-1, 
                                                    dim_size=pc_coords_list[i + 1].shape[1])
        points_num_pt_list.append(curr_points_num_pt)
    
    # cfgs.phase4_correspondence_pair example: [[0, 0], [0, 1], [1, 0], [1, 1]]
    # the first index is the index of img_feats_list, the second index is the index of pc_feats_list
    phase4_correspondence_pair = torch.tensor(cfgs.phase4_correspondence_pair, dtype=torch.int32, device=device)
    img_level_indices = torch.unique(phase4_correspondence_pair[..., 0], sorted=True)
    points_level_indices = torch.unique(phase4_correspondence_pair[..., 1], sorted=True)
    img_pair_embeddings_list = []
    img_pair_indices_list = []
    img_pair_list_reverse = {}
    for img_level_indice in img_level_indices:
        curr_img_pair_indices = torch.nonzero(torch.gt(img_num_pt_list[img_level_indices], 
                                                       cfgs.phase4_min_img_num_pt[img_level_indices]), 
                                                as_tuple=False) # produce (curr_num_img, 3)
        curr_img_pair_indices = torch.unique(curr_img_pair_indices[:, 1:], dim=0, sorted=True) # produce (curr_num_img_chose, 2)
        curr_img_pair_embeddings = img_feats_list[img_level_indice].flatten(start_dim=2)[curr_img_pair_indices[:, 0], :, curr_img_pair_indices[:, 1]] # produce (curr_num_img_chose, out_dim)
        img_pair_embeddings_list.append(curr_img_pair_embeddings)
        img_pair_indices_list.append(curr_img_pair_indices)
        img_pair_list_reverse[img_level_indice] = len(img_pair_embeddings_list) - 1
    
    points_pair_embeddings_list = []
    points_pair_indices_list = []
    points_pair_list_reverse = {}
    for points_level_indice in points_level_indices:
        curr_points_overlap_num_pt = torch.sum(many_img_2_many_points_num_pt_list[0][points_level_indice], 
                                               dim=2, 
                                               keepdim=False) # produce (B, B, curr_points_num)
        curr_points_pair_indices = torch.nonzero(torch.gt(curr_points_overlap_num_pt, 
                                                          cfgs.phase4_min_pc_overlap_num_pt[points_level_indice]), 
                                                          as_tuple=False) # produce (curr_num_pc_chose, 3)
        curr_points_pair_indices = torch.unique(curr_points_pair_indices[:, ::2], 
                                                dim=0, 
                                                sorted=True) # produce (curr_num_pc_chose, 2)
        curr_points_pair_embeddings = pc_feats_list[points_level_indice][curr_points_pair_indices[:, 0], :, curr_points_pair_indices[:, 1]] # produce (curr_num_pc_chose, out_dim)
        points_pair_embeddings_list.append(curr_points_pair_embeddings)
        points_pair_indices_list.append(curr_points_pair_indices)
        points_pair_list_reverse[points_level_indice] = len(points_pair_embeddings_list) - 1

    if cfgs.phase4_multi_level_overlap_ratio_type == "corresponds_between_all":
        # 1、all the overlap_matrix are merged into one matrix
        many_img_2_many_points_overlap_ratio_matrix_list = []
        for img_idx in range(len(img_level_indices)):
            curr_img_2_many_points_overlap_ratio_matrix_list = []
            for points_idx in range(len(points_level_indices)):
                img_level_indice = img_level_indices[img_idx]
                points_level_indice = points_level_indices[points_idx]
                curr_pc_overlap_ratio_matrix = many_img_2_many_points_num_pt_list[img_level_indice][points_level_indice] * 1.0 / torch.clamp(points_num_pt_list[points_level_indice].unsqueeze(2), min=1)
                curr_img_overlap_ratio_matrix = many_img_2_many_points_num_pt_list[img_level_indice][points_level_indice] * 1.0 / torch.clamp(img_num_pt_list[img_level_indice].unsqueeze(3), min=1)
                curr_overlap_ratio_matrix = curr_pc_overlap_ratio_matrix * 0.5 + curr_img_overlap_ratio_matrix * 0.5
                curr_overlap_ratio_matrix_inuse = curr_overlap_ratio_matrix[points_pair_indices_list[points_idx][:, 0], :, :, points_pair_indices_list[points_idx][:, 1]][:, img_pair_indices_list[img_idx][:, 0], img_pair_indices_list[img_idx][:, 1]] # (curr_num_pc_chose, curr_num_img_chose)
                curr_img_2_many_points_overlap_ratio_matrix_list.append(curr_overlap_ratio_matrix_inuse)
            curr_img_2_many_points_overlap_ratio_matrix = torch.cat(curr_img_2_many_points_overlap_ratio_matrix_list, dim=0) # (num_pc_chose, curr_num_img_chose)
            many_img_2_many_points_overlap_ratio_matrix_list.append(curr_img_2_many_points_overlap_ratio_matrix)
        many_img_2_many_points_overlap_ratio_matrix = torch.cat(many_img_2_many_points_overlap_ratio_matrix_list, dim=1) # (num_pc_chose, num_img_chose)
        img_pair_embeddings = torch.cat(img_pair_embeddings_list, dim=0) # (num_img_chose, out_dim)
        points_pair_embeddings = torch.cat(points_pair_embeddings_list, dim=0) # (num_pc_chose, out_dim)
        return many_img_2_many_points_overlap_ratio_matrix, img_pair_embeddings, points_pair_embeddings
    elif cfgs.phase4_multi_level_overlap_ratio_type == "corresponds_between_pair":
        # 2、use the num of cfgs.phase4_correspondence_pair of circle losses, so as many matrixs and embeddings
        overlap_ratio_matrix_list = []
        image_pair_embeddings_output_list = []
        points_pair_embeddings_output_list = []
        for curr_pair in cfgs.phase4_correspondence_pair:
            curr_img_level_indice = curr_pair[0]
            curr_points_level_indice = curr_pair[1]
            curr_img_pair_embeddings = img_pair_embeddings_list[img_pair_list_reverse[curr_img_level_indice]]
            curr_points_pair_embeddings = points_pair_embeddings_list[points_pair_list_reverse[curr_points_level_indice]]
            curr_img_pair_indices = img_pair_indices_list[img_pair_list_reverse[curr_img_level_indice]]
            curr_points_pair_indices = points_pair_indices_list[points_pair_list_reverse[curr_points_level_indice]]
            curr_img_2_curr_points_num_pt = many_img_2_many_points_num_pt_list[curr_img_level_indice][curr_points_level_indice]
            img_overlap_ratio_matrix = curr_img_2_curr_points_num_pt * 1.0 / torch.clamp(img_num_pt_list[curr_img_level_indice].unsqueeze(3), min=1)
            points_overlap_ratio_matrix = curr_img_2_curr_points_num_pt * 1.0 / torch.clamp(points_num_pt_list[curr_points_level_indice].unsqueeze(2), min=1)
            curr_overlap_ratio_matrix = points_overlap_ratio_matrix * 0.5 + img_overlap_ratio_matrix * 0.5
            curr_overlap_ratio_matrix_inuse = curr_overlap_ratio_matrix[curr_points_pair_indices[:, 0], :, :, curr_points_pair_indices[:, 1]][:, curr_img_pair_indices[:, 0], curr_img_pair_indices[:, 1]] # (num_pc_chose, num_img_chose)
            overlap_ratio_matrix_list.append(curr_overlap_ratio_matrix_inuse)
            image_pair_embeddings_output_list.append(curr_img_pair_embeddings)
            points_pair_embeddings_output_list.append(curr_points_pair_embeddings)
        return overlap_ratio_matrix_list, image_pair_embeddings_output_list, points_pair_embeddings_output_list




def generate_single_correspondence_phase4(c_points, data_dict, img_H, img_W, device, c_img_feats, c_pc_feats, cfgs):
    B, cpoints_num = c_points.shape[:2]
    clouds_2_cpoints_dis = torch.cdist(data_dict["clouds"], c_points) # Produces (B, 4096, c_points_num) tensor
    _, clouds_2_cpoints_idx = torch.topk(clouds_2_cpoints_dis, k=1, dim=2, largest=False, sorted=False) # Produces (B, 4096, 1) tensor
    clouds_2_cpoints_idx = clouds_2_cpoints_idx.squeeze(-1) # Produces (B, 4096) tensor
    original_pc_2_cpoints = torch.gather(input=clouds_2_cpoints_idx.unsqueeze(1).expand(-1, B, -1),
                                        dim=-1,
                                        index=data_dict["original_pc_2_many_1"][..., 0]) # (B, B, original_points_num)
    
    img_H_mesh = torch.arange(0, img_H, device=device).type(torch.float32)
    img_W_mesh = torch.arange(0, img_W, device=device).type(torch.float32)
    img_H_mesh = img_H_mesh.unsqueeze(1).unsqueeze(2).expand(-1, img_W, -1)
    img_W_mesh = img_W_mesh.unsqueeze(0).unsqueeze(2).expand(img_H, -1, -1)
    img_mesh = torch.cat((img_H_mesh, img_W_mesh), dim=-1) # Produces (img_H, img_W, 2)
    img_mesh = img_mesh.flatten(0, 1) # Produces (img_H * img_W, 2) tensor
    cimg_H, cimg_W = c_img_feats.shape[2:]
    cimg_H_mesh = torch.arange(0, cimg_H, device=device)
    cimg_W_mesh = torch.arange(0, cimg_W, device=device)
    img_2_cimg_scale_H = img_H * 1.0 / cimg_H
    img_2_cimg_scale_W = img_W * 1.0 / cimg_W
    delta_H = img_2_cimg_scale_H / 2 - 0.5
    delta_W = img_2_cimg_scale_W / 2 - 0.5
    cimg_H_mesh = cimg_H_mesh * img_2_cimg_scale_H + delta_H
    cimg_W_mesh = cimg_W_mesh * img_2_cimg_scale_W + delta_W
    cimg_H_mesh = cimg_H_mesh.unsqueeze(1).unsqueeze(2).expand(-1, cimg_W, -1) # Produces (cimg_H, cimg_W, 1) tensor
    cimg_W_mesh = cimg_W_mesh.unsqueeze(0).unsqueeze(2).expand(cimg_H, -1, -1) # Produces (cimg_H, cimg_W, 1) tensor
    cimg_mesh = torch.cat((cimg_H_mesh, cimg_W_mesh), dim=-1) # Produces (cimg_H, cimg_W, 2) tensor
    cimg_mesh = cimg_mesh.flatten(0, 1) # Produces (cimg_H * cimg_W, 2) tensor
    img_2_cimg_dis = torch.cdist(img_mesh, cimg_mesh) # Produces (img_H * img_W, cimg_H * cimg_W) tensor
    _, img_2_cimg_idx = torch.topk(img_2_cimg_dis, k=1, dim=1, largest=False, sorted=False) # Produces (img_H * img_W, 1) tensor
    img_2_cimg_idx = img_2_cimg_idx.squeeze(-1).unsqueeze(0).unsqueeze(0).expand(B, B, -1) # Produces (B, B, img_H * img_W) tensor
    original_pc_2_cimg = torch.gather(input=img_2_cimg_idx,
                                    dim=-1,
                                    index=data_dict["original_pc_2_many_1"][..., 1]) # Produces (B, B, original_points_num)

    cimg_2_cpoints_map = original_pc_2_cimg * cpoints_num + original_pc_2_cpoints
    overlap_mask = torch.logical_and(data_dict["original_pc_2_many_2"][..., 0], data_dict["original_pc_2_many_2"][..., 1])
    cimg_2_cpoints_map.masked_fill_(~overlap_mask, cimg_H * cimg_W * cpoints_num - 1)
    overlap_mask_num = torch.count_nonzero(~overlap_mask, dim=-1) # (B, B)
    cimg_2_cpoints_num_pt = torch_scatter.scatter_sum(torch.ones_like(cimg_2_cpoints_map, dtype=torch.int32), 
                                                    cimg_2_cpoints_map,
                                                    dim=-1,
                                                    dim_size=cimg_H * cimg_W * cpoints_num) # produces (B, B, cimg_H * cimg_W * cpoints_num)
    cimg_2_cpoints_num_pt[..., -1] -= overlap_mask_num
    cimg_2_cpoints_num_pt = cimg_2_cpoints_num_pt.reshape(B, B, cimg_H * cimg_W, cpoints_num)

    if cfgs.phase4_overlap_matrix_modal == "pc": # only consider the overlap ratio in pointcloud
        original_pc_2_cpoints.masked_fill_(~data_dict["original_pc_2_many_2"][..., 1], cpoints_num - 1)  # can't be reuse again
        remove_mask_num = torch.count_nonzero(~data_dict["original_pc_2_many_2"][..., 1], dim=-1) # (B, B)
        cpoints_num_pt = torch_scatter.scatter_sum(torch.ones_like(original_pc_2_cpoints, dtype=torch.int32),
                                                original_pc_2_cpoints,
                                                dim=-1,
                                                dim_size=cpoints_num) # produce (B, B, cpoints_num)
        cpoints_num_pt[..., -1] -= remove_mask_num # produce (B, B, cpoints_num)
        cpoints_num_pt = torch.clamp(cpoints_num_pt, min=1)
        overlap_ratio_matrix = cimg_2_cpoints_num_pt * 1.0 / cpoints_num_pt.unsqueeze(2)  # produce (B, B, cimg_H * cimg_W, cpoints_num)
    elif cfgs.phase4_overlap_matrix_modal == "img": # only consider the overlap ratio in image
        original_pc_2_cimg.masked_fill_(~data_dict["original_pc_2_many_2"][..., 0], cimg_H * cimg_W - 1) # can't be reuse again
        non_qualified_mask_num = torch.count_nonzero(~data_dict["original_pc_2_many_2"][..., 0], dim=-1) # produce (B, B)
        cimg_num_pt = torch_scatter.scatter_sum(torch.ones_like(original_pc_2_cimg, dtype=torch.int32),
                                                original_pc_2_cimg,
                                                dim=-1,
                                                dim_size=cimg_H * cimg_W) # produce (B, B, cimg_H * cimg_W)
        cimg_num_pt[..., -1] -= non_qualified_mask_num
        cimg_num_pt = torch.clamp(cimg_num_pt, min=1)
        overlap_ratio_matrix = cimg_2_cpoints_num_pt * 1.0 / cimg_num_pt.unsqueeze(3) # produce (B, B, cimg_H * cimg_W, cpoints_num)
    elif cfgs.phase4_overlap_matrix_modal == "pc_and_img":
        original_pc_2_cpoints.masked_fill_(~data_dict["original_pc_2_many_2"][..., 1], cpoints_num - 1) # can't be reuse again
        remove_mask_num = torch.count_nonzero(~data_dict["original_pc_2_many_2"][..., 1], dim=-1) # (B, B)
        cpoints_num_pt = torch_scatter.scatter_sum(torch.ones_like(original_pc_2_cpoints, dtype=torch.int32),
                                                original_pc_2_cpoints,
                                                dim=-1,
                                                dim_size=cpoints_num) # produce (B, B, cpoints_num)
        cpoints_num_pt[..., -1] -= remove_mask_num # produce (B, B, cpoints_num)
        cpoints_num_pt = torch.clamp(cpoints_num_pt, min=1)
        pc_overlap_ratio_matrix = cimg_2_cpoints_num_pt * 1.0 / cpoints_num_pt.unsqueeze(2) # produce (B, B, cimg_H * cimg_W, cpoints_num)
        original_pc_2_cimg.masked_fill_(~data_dict["original_pc_2_many_2"][..., 0], cimg_H * cimg_W - 1) # can't be reuse again
        non_qualified_mask_num = torch.count_nonzero(~data_dict["original_pc_2_many_2"][..., 0], dim=-1) # produce (B, B)
        cimg_num_pt = torch_scatter.scatter_sum(torch.ones_like(original_pc_2_cimg, dtype=torch.int32),
                                                original_pc_2_cimg,
                                                dim=-1,
                                                dim_size=cimg_H * cimg_W) # produce (B, B, cimg_H * cimg_W)
        cimg_num_pt[..., -1] -= non_qualified_mask_num
        cimg_num_pt = torch.clamp(cimg_num_pt, min=1)
        img_overlap_ratio_matrix = cimg_2_cpoints_num_pt * 1.0 / cimg_num_pt.unsqueeze(3) # produce (B, B, cimg_H * cimg_W, cpoints_num)
        if cfgs.phase4_overlap_matrix_fuse_type == "mean":
            overlap_ratio_matrix = 0.5 * pc_overlap_ratio_matrix + 0.5 * img_overlap_ratio_matrix
        elif cfgs.phase4_overlap_matrix_fuse_type == "max":
            overlap_ratio_matrix = torch.maximum(pc_overlap_ratio_matrix, img_overlap_ratio_matrix)
        elif cfgs.phase4_overlap_matrix_fuse_type == "min":
            overlap_ratio_matrix = torch.minimum(pc_overlap_ratio_matrix, img_overlap_ratio_matrix)
    
    # filter out the correspondding pair according to the min num_pt in cimg_num_pt、total_overlap_num_pt(for every cpoint)
    img_pair_indices = torch.nonzero(torch.gt(cimg_num_pt, cfgs.phase4_min_img_num_pt), as_tuple=False) # produce (num_img, 3)
    img_pair_indices = torch.unique(img_pair_indices[:, 1:], dim=0, sorted=True) # produce (num_img_1, 2)
    img_pair_embeddings = c_img_feats.flatten(start_dim=2)[img_pair_indices[:, 0], :, img_pair_indices[:, 1]] # produce (num_img_1, out_dim)
    cpoints_overlap_num_pt = torch.sum(cimg_2_cpoints_num_pt, dim=2, keepdim=False) # produce (B, B, cpoints_num)
    pc_pair_indices = torch.nonzero(torch.gt(cpoints_overlap_num_pt, cfgs.phase4_min_pc_overlap_num_pt), as_tuple=False) # produce (num_pc, 3)
    pc_pair_indices = torch.unique(pc_pair_indices[:, ::2], dim=0, sorted=True) # produce (num_pc_1, 2)
    pc_pair_embeddings = c_pc_feats[pc_pair_indices[:, 0], :, pc_pair_indices[:, 1]] # produce (num_pc_1, out_dim)
    overlap_ratio_matrix_inuse = overlap_ratio_matrix[pc_pair_indices[:, 0], :, :, pc_pair_indices[:, 1]][:, img_pair_indices[:, 0], img_pair_indices[:, 1]] # produce (num_pc_1, num_img_1)

    if 'bn' in data_dict.keys():
        if data_dict['bn'] in data_dict['visualization_batches']:
            original_pc_2_cpoints_new = torch.gather(input=clouds_2_cpoints_idx.unsqueeze(1).expand(-1, B, -1),
                                        dim=-1,
                                        index=data_dict["original_pc_2_many_1"][..., 0]) # (B, B, original_points_num)
            original_pc_2_cimg_new = torch.gather(input=img_2_cimg_idx,
                                        dim=-1,
                                        index=data_dict["original_pc_2_many_1"][..., 1]) # Produces (B, B, original_points_num)
            preffix = '/home/pengjianyi/code_projects/visualization_open3d'
            curr_bn = data_dict['bn']
            for i in range(len(data_dict['labels'])):
                label_i = data_dict['labels'][i]
                curr_pc_point = c_points[i, :, :].cpu().numpy() # (cpoints_num, 3)
                file_name_1 = os.path.join(preffix, f"train_batch{curr_bn}_{label_i}_pc.npy")
                np.save(file_name_1, curr_pc_point)
                curr_pc_original_pc_2_curr_points = original_pc_2_cpoints_new[i, 0, :].cpu().numpy() # (original_points_num)
                file_name_2 = os.path.join(preffix, f'train_batch{curr_bn}_{label_i}_pc_original_pc_2_curr_points.npy')
                np.save(file_name_2, curr_pc_original_pc_2_curr_points)
                curr_pc_embeddings = c_pc_feats[i, :, :] # (out_dim, cpoints_num)
                curr_pc_embeddings_normed = F.normalize(curr_pc_embeddings, p=2, dim=0)
                for j in range(len(data_dict['labels'])):
                    label_j = data_dict['labels'][j]
                    curr_pc_original_pc_2_image = original_pc_2_cimg_new[i, j, :].cpu().numpy() # (original_points_num)
                    file_name_3 = os.path.join(preffix, f"train_batch{curr_bn}_{label_i}_2_{label_j}_pc_original_pc_2_image.npy")
                    np.save(file_name_3, curr_pc_original_pc_2_image)
                    curr_pc_overlap_ratio_matrix = pc_overlap_ratio_matrix[i, j, :, :].cpu().numpy() # (cimg_H * cimg_W, cpoints_num)
                    file_name_4 = os.path.join(preffix, f"train_batch{curr_bn}_{label_i}_2_{label_j}_pc_overlap_ratio_matrix.npy")
                    np.save(file_name_4, curr_pc_overlap_ratio_matrix)
                    curr_img_overlap_ratio_matrix = img_overlap_ratio_matrix[i, j, :, :].cpu().numpy() # (cimg_H * cimg_W, cpoints_num)
                    file_name_5 = os.path.join(preffix, f"train_batch{curr_bn}_{label_i}_2_{label_j}_img_overlap_ratio_matrix.npy")
                    np.save(file_name_5, curr_img_overlap_ratio_matrix)

                    curr_image_embeddings = c_img_feats[j, :, :, :] # (out_dim, cimg_H, cimg_W)
                    curr_image_embeddings_normed = F.normalize(curr_image_embeddings, p=2, dim=0)
                    curr_image_embeddings_normed = curr_image_embeddings_normed.flatten(start_dim=1) # (out_dim, cimg_H * cimg_W)
                    curr_pc_similarity = torch.matmul(curr_pc_embeddings_normed.T, curr_image_embeddings_normed) # (cpoints_num, cimg_H * cimg_W)
                    curr_pc_similarity = curr_pc_similarity.detach().cpu().numpy()
                    file_name_6 = os.path.join(preffix, f"train_batch{curr_bn}_{label_i}_2_{label_j}_pc_similarity.npy")
                    np.save(file_name_6, curr_pc_similarity)


    # import numpy as np
    # size = 224
    # block_size = 4
    # checkerboard = np.zeros((size, size), dtype=int)
    # num_blocks = size // block_size

    # for i in range(num_blocks):
    #     for j in range(num_blocks):
    #         if (i + j) % 2 == 0:
    #             checkerboard[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = 1
    # checkerboard = checkerboard.astype(np.uint8)
    # checkerboard[checkerboard == 1] = 170
    # checkerboard[checkerboard == 0] = 85
    # checkerboard = np.expand_dims(checkerboard, axis=2)

    # img_color = img_2_cimg_idx.reshape(B, B, img_H, img_W) # (B, B, img_H, img_W)
    # pc_pos = data_dict["original_pc_2_many_3"] # (B, B, original_points_num, 2)
    # pc_color = original_pc_2_cpoints # Produces (B, B, original_points_num)
    # pc_mask = torch.logical_and(data_dict["original_pc_2_many_2"][..., 0], data_dict["original_pc_2_many_2"][..., 1]) # Produces (B, B, original_points_num)
    # alpha = 0.4
    # for i in range(B):
    #     for j in range(B):
    #         curr_pc_color = pc_color[i, j, :]
    #         curr_pc_mask = pc_mask[i, j, :]
    #         curr_img_color = img_color[i, j, :, :]
    #         curr_pc_pos = pc_pos[i, j, :, :]
    #         curr_pc_color = np.array(curr_pc_color[curr_pc_mask].to('cpu'))
    #         curr_pc_pos = np.array(curr_pc_pos[curr_pc_mask, :].to('cpu'))
            
    #         curr_img_color = checkerboard
    #         # curr_img_color = cv2.applyColorMap(checkerboard, cv2.COLORMAP_JET)

    #         fig = plt.figure(figsize=(3.00, 3.00), dpi=100)
    #         ax = fig.add_subplot()
    #         ax.imshow(curr_img_color, alpha=alpha)
    #         ax.set_xlim(0, 224)
    #         ax.set_ylim(224, 0)
    #         ax.scatter(curr_pc_pos[:, 0], curr_pc_pos[:, 1], c=curr_pc_color, marker=',', s=3, edgecolors='none', alpha=0.7, cmap='jet')
    #         text = "555"
    #         position = (50, 50)  # (x, y) 位置
    #         ax.text(position[0], position[1], text, fontsize=12, color='white',)
    #         bbox=dict(facecolor='none', edgecolor='none', boxstyle='round,pad=0.2')
    #         ax.set_axis_off()
    #         plt.savefig(f'/home/test5/code_project/visualization/boreas_overlap_vis_{i}_{j}.jpg', bbox_inches='tight', pad_inches=0, dpi=200)

    #         # color = clouds_in_camera[i, j, :, 2]
    #         # curr_mask = mask_0[i, j, :] & mask_2[i, j, :]
    #         # color = color[curr_mask]
    #         # # color[~mask_0[i, j, :]] = -500.0
    #         # uv = clouds_in_plane[i, j, :, :]
    #         # uv = uv[curr_mask, :]
    #         # fig = plt.figure(figsize=(30.00, 30.00), dpi=100)
    #         # ax = fig.add_subplot()
    #         # ax.imshow(result['images'][j, :].permute(1, 2, 0))
    #         # # ax.set_xlim(-500, 1000)
    #         # # ax.set_ylim(1000, -500)
    #         # ax.set_xlim(-250, 2500)
    #         # ax.set_ylim(2500, -250)
    #         # ax.scatter(uv[:, 0], uv[:, 1], c=color, marker=',', s=3, edgecolors='none', alpha=0.7, cmap='jet')
    #         # ax.set_axis_off()
    #         # plt.savefig(f'/home/test5/code_project/visualization/heihei_boreas_crop_{i}_{j}.jpg', bbox_inches='tight', pad_inches=0, dpi=200)

    return overlap_ratio_matrix_inuse, img_pair_embeddings, pc_pair_embeddings




def generate_single_correspondence_phase4_v1(c_points, data_dict, img_H, img_W, device, c_img_feats, c_pc_feats, cfgs):
    B, cpoints_num = c_points.shape[:2]
    clouds_2_cpoints_dis = torch.cdist(data_dict["clouds"], c_points) # Produces (B, 4096, c_points_num) tensor
    _, clouds_2_cpoints_idx = torch.topk(clouds_2_cpoints_dis, k=1, dim=2, largest=False, sorted=False) # Produces (B, 4096, 1) tensor
    clouds_2_cpoints_idx = clouds_2_cpoints_idx.squeeze(-1) # Produces (B, 4096) tensor
    original_pc_2_cpoints = torch.gather(input=clouds_2_cpoints_idx.unsqueeze(1).expand(-1, B, -1),
                                        dim=-1,
                                        index=data_dict["original_pc_2_many_1"][..., 0]) # (B, B, original_points_num)
    
    img_H_mesh = torch.arange(0, img_H, device=device).type(torch.float32)
    img_W_mesh = torch.arange(0, img_W, device=device).type(torch.float32)
    img_H_mesh = img_H_mesh.unsqueeze(1).unsqueeze(2).expand(-1, img_W, -1)
    img_W_mesh = img_W_mesh.unsqueeze(0).unsqueeze(2).expand(img_H, -1, -1)
    img_mesh = torch.cat((img_H_mesh, img_W_mesh), dim=-1) # Produces (img_H, img_W, 2)
    img_mesh = img_mesh.flatten(0, 1) # Produces (img_H * img_W, 2) tensor
    cimg_H, cimg_W = c_img_feats.shape[2:]
    cimg_H_mesh = torch.arange(0, cimg_H, device=device)
    cimg_W_mesh = torch.arange(0, cimg_W, device=device)
    img_2_cimg_scale_H = img_H * 1.0 / cimg_H
    img_2_cimg_scale_W = img_W * 1.0 / cimg_W
    delta_H = img_2_cimg_scale_H / 2 - 0.5
    delta_W = img_2_cimg_scale_W / 2 - 0.5
    cimg_H_mesh = cimg_H_mesh * img_2_cimg_scale_H + delta_H
    cimg_W_mesh = cimg_W_mesh * img_2_cimg_scale_W + delta_W
    cimg_H_mesh = cimg_H_mesh.unsqueeze(1).unsqueeze(2).expand(-1, cimg_W, -1) # Produces (cimg_H, cimg_W, 1) tensor
    cimg_W_mesh = cimg_W_mesh.unsqueeze(0).unsqueeze(2).expand(cimg_H, -1, -1) # Produces (cimg_H, cimg_W, 1) tensor
    cimg_mesh = torch.cat((cimg_H_mesh, cimg_W_mesh), dim=-1) # Produces (cimg_H, cimg_W, 2) tensor
    cimg_mesh = cimg_mesh.flatten(0, 1) # Produces (cimg_H * cimg_W, 2) tensor
    img_2_cimg_dis = torch.cdist(img_mesh, cimg_mesh) # Produces (img_H * img_W, cimg_H * cimg_W) tensor
    _, img_2_cimg_idx = torch.topk(img_2_cimg_dis, k=1, dim=1, largest=False, sorted=False) # Produces (img_H * img_W, 1) tensor
    img_2_cimg_idx = img_2_cimg_idx.squeeze(-1).unsqueeze(0).unsqueeze(0).expand(B, B, -1) # Produces (B, B, img_H * img_W) tensor
    original_pc_2_cimg = torch.gather(input=img_2_cimg_idx,
                                    dim=-1,
                                    index=data_dict["original_pc_2_many_1"][..., 1]) # Produces (B, B, original_points_num)

    cimg_2_cpoints_map = original_pc_2_cimg * cpoints_num + original_pc_2_cpoints
    overlap_mask = torch.logical_and(data_dict["original_pc_2_many_2"][..., 0], data_dict["original_pc_2_many_2"][..., 1])
    cimg_2_cpoints_map.masked_fill_(~overlap_mask, cimg_H * cimg_W * cpoints_num - 1)
    overlap_mask_num = torch.count_nonzero(~overlap_mask, dim=-1) # (B, B)
    cimg_2_cpoints_num_pt = torch_scatter.scatter_sum(torch.ones_like(cimg_2_cpoints_map, dtype=torch.int32), 
                                                    cimg_2_cpoints_map,
                                                    dim=-1,
                                                    dim_size=cimg_H * cimg_W * cpoints_num) # produces (B, B, cimg_H * cimg_W * cpoints_num)
    cimg_2_cpoints_num_pt[..., -1] -= overlap_mask_num
    cimg_2_cpoints_num_pt = cimg_2_cpoints_num_pt.reshape(B, B, cimg_H * cimg_W, cpoints_num)

    if cfgs.phase4_overlap_matrix_modal == "pc": # only consider the overlap ratio in pointcloud
        original_pc_2_cpoints.masked_fill_(~data_dict["original_pc_2_many_2"][..., 1], cpoints_num - 1)  # can't be reuse again
        remove_mask_num = torch.count_nonzero(~data_dict["original_pc_2_many_2"][..., 1], dim=-1) # (B, B)
        cpoints_num_pt = torch_scatter.scatter_sum(torch.ones_like(original_pc_2_cpoints, dtype=torch.int32),
                                                original_pc_2_cpoints,
                                                dim=-1,
                                                dim_size=cpoints_num) # produce (B, B, cpoints_num)
        cpoints_num_pt[..., -1] -= remove_mask_num # produce (B, B, cpoints_num)
        cpoints_num_pt = torch.clamp(cpoints_num_pt, min=1)
        overlap_ratio_mask = torch.gt(cpoints_num_pt, cfgs.phase4_min_pc_num_pt) # produce (B, B, cpoints_num) some cpoints have not enough points due to remove augmentation
        overlap_ratio_matrix = cimg_2_cpoints_num_pt * 1.0 / cpoints_num_pt.unsqueeze(2)  # produce (B, B, cimg_H * cimg_W, cpoints_num)
        overlap_ratio_matrix = overlap_ratio_matrix.masked_fill(~(overlap_ratio_mask.unsqueeze(2)), 0.0)
    elif cfgs.phase4_overlap_matrix_modal == "img": # only consider the overlap ratio in image
        original_pc_2_cimg.masked_fill_(~data_dict["original_pc_2_many_2"][..., 0], cimg_H * cimg_W - 1) # can't be reuse again
        non_qualified_mask_num = torch.count_nonzero(~data_dict["original_pc_2_many_2"][..., 0], dim=-1) # produce (B, B)
        cimg_num_pt = torch_scatter.scatter_sum(torch.ones_like(original_pc_2_cimg, dtype=torch.int32),
                                                original_pc_2_cimg,
                                                dim=-1,
                                                dim_size=cimg_H * cimg_W) # produce (B, B, cimg_H * cimg_W)
        cimg_num_pt[..., -1] -= non_qualified_mask_num
        cimg_num_pt = torch.clamp(cimg_num_pt, min=1)
        overlap_ratio_mask = torch.gt(cimg_num_pt, cfgs.phase4_min_img_num_pt) # produce (B, B, cimg_H * cimg_W)
        overlap_ratio_matrix = cimg_2_cpoints_num_pt * 1.0 / cimg_num_pt.unsqueeze(3) # produce (B, B, cimg_H * cimg_W, cpoints_num)
        overlap_ratio_matrix = overlap_ratio_matrix.masked_fill(~(overlap_ratio_mask.unsqueeze(3)), 0.0)
    elif cfgs.phase4_overlap_matrix_modal == "pc_and_img":
        original_pc_2_cpoints.masked_fill_(~data_dict["original_pc_2_many_2"][..., 1], cpoints_num - 1) # can't be reuse again
        remove_mask_num = torch.count_nonzero(~data_dict["original_pc_2_many_2"][..., 1], dim=-1) # (B, B)
        cpoints_num_pt = torch_scatter.scatter_sum(torch.ones_like(original_pc_2_cpoints, dtype=torch.int32),
                                                original_pc_2_cpoints,
                                                dim=-1,
                                                dim_size=cpoints_num) # produce (B, B, cpoints_num)
        cpoints_num_pt[..., -1] -= remove_mask_num # produce (B, B, cpoints_num)
        cpoints_num_pt = torch.clamp(cpoints_num_pt, min=1)
        pc_overlap_ratio_mask = torch.gt(cpoints_num_pt, cfgs.phase4_min_pc_num_pt) # produce (B, B, cpoints_num) some cpoints have not enough points due to remove augmentation
        pc_overlap_ratio_matrix = cimg_2_cpoints_num_pt * 1.0 / cpoints_num_pt.unsqueeze(2) # produce (B, B, cimg_H * cimg_W, cpoints_num)
        pc_overlap_ratio_matrix = pc_overlap_ratio_matrix.masked_fill(~(pc_overlap_ratio_mask.unsqueeze(2)), 0.0)
        original_pc_2_cimg.masked_fill_(~data_dict["original_pc_2_many_2"][..., 0], cimg_H * cimg_W - 1) # can't be reuse again
        non_qualified_mask_num = torch.count_nonzero(~data_dict["original_pc_2_many_2"][..., 0], dim=-1) # produce (B, B)
        cimg_num_pt = torch_scatter.scatter_sum(torch.ones_like(original_pc_2_cimg, dtype=torch.int32),
                                                original_pc_2_cimg,
                                                dim=-1,
                                                dim_size=cimg_H * cimg_W) # produce (B, B, cimg_H * cimg_W)
        cimg_num_pt[..., -1] -= non_qualified_mask_num
        cimg_num_pt = torch.clamp(cimg_num_pt, min=1)
        img_overlap_ratio_mask = torch.gt(cimg_num_pt, cfgs.phase4_min_img_num_pt) # produce (B, B, cimg_H * cimg_W)
        img_overlap_ratio_matrix = cimg_2_cpoints_num_pt * 1.0 / cimg_num_pt.unsqueeze(3) # produce (B, B, cimg_H * cimg_W, cpoints_num)
        img_overlap_ratio_matrix = img_overlap_ratio_matrix.masked_fill(~(img_overlap_ratio_mask.unsqueeze(3)), 0.0)
        if cfgs.phase4_overlap_matrix_fuse_type == "mean":
            overlap_ratio_matrix = 0.5 * pc_overlap_ratio_matrix + 0.5 * img_overlap_ratio_matrix
        elif cfgs.phase4_overlap_matrix_fuse_type == "max":
            overlap_ratio_matrix = torch.maximum(pc_overlap_ratio_matrix, img_overlap_ratio_matrix)
        elif cfgs.phase4_overlap_matrix_fuse_type == "min":
            overlap_ratio_matrix = torch.minimum(pc_overlap_ratio_matrix, img_overlap_ratio_matrix)
    
    # filter out the correspondding pair according to the min num_pt in cimg_num_pt、total_overlap_num_pt(for every cpoint)
    overlap_ratio_matrix_flatten = overlap_ratio_matrix.flatten(start_dim=0) # produce (B * B * cimg_H * cimg_W * cpoints_num)
    _, topk_indices = torch.topk(overlap_ratio_matrix_flatten, k=cfgs.phase4_topk, dim=-1, largest=True, sorted=False) # produce (cfgs.phase4_topk,)
    batch_indices1 = torch.div(topk_indices, B * cimg_H * cimg_W * cpoints_num, rounding_mode='floor') # produce (cfgs.phase4_topk,)
    batch_indices2 = torch.div(topk_indices % (B * cimg_H * cimg_W * cpoints_num), cimg_H * cimg_W * cpoints_num, rounding_mode='floor') # produce (cfgs.phase4_topk,)
    cimg_indices = torch.div(topk_indices % (cimg_H * cimg_W * cpoints_num), cpoints_num, rounding_mode='floor') # produce (cfgs.phase4_topk,)
    cpoint_indices = topk_indices % (cpoints_num) # produce (cfgs.phase4_topk,)
    img_pair_indices = torch.stack((batch_indices2, cimg_indices), dim=1) # produce (cfgs.phase4_topk, 2)
    img_pair_indices = torch.unique(img_pair_indices, dim=0, sorted=True) # produce (num_img_1, 2)
    pc_pair_indices = torch.stack((batch_indices1, cpoint_indices), dim=1) # produce (cfgs.phase4_topk, 2)
    pc_pair_indices = torch.unique(pc_pair_indices, dim=0, sorted=True) # produce (num_pc_1, 2)
    img_pair_embeddings = c_img_feats.flatten(start_dim=2)[img_pair_indices[:, 0], :, img_pair_indices[:, 1]] # produce (num_img_1, out_dim)
    pc_pair_embeddings = c_pc_feats[pc_pair_indices[:, 0], :, pc_pair_indices[:, 1]] # produce (num_pc_1, out_dim)
    overlap_ratio_matrix_inuse = overlap_ratio_matrix[pc_pair_indices[:, 0], :, :, pc_pair_indices[:, 1]][:, img_pair_indices[:, 0], img_pair_indices[:, 1]] # produce (num_pc_1, num_img_1)

    if 'bn' in data_dict.keys():
        if data_dict['bn'] in data_dict['visualization_batches']:
            original_pc_2_cpoints_new = torch.gather(input=clouds_2_cpoints_idx.unsqueeze(1).expand(-1, B, -1),
                                        dim=-1,
                                        index=data_dict["original_pc_2_many_1"][..., 0]) # (B, B, original_points_num)
            original_pc_2_cimg_new = torch.gather(input=img_2_cimg_idx,
                                        dim=-1,
                                        index=data_dict["original_pc_2_many_1"][..., 1]) # Produces (B, B, original_points_num)
            preffix = '/home/pengjianyi/code_projects/visualization_open3d'
            curr_bn = data_dict['bn']
            for i in range(len(data_dict['labels'])):
                label_i = data_dict['labels'][i]
                curr_pc_point = c_points[i, :, :].cpu().numpy() # (cpoints_num, 3)
                file_name_1 = os.path.join(preffix, f"train_batch{curr_bn}_{label_i}_pc.npy")
                np.save(file_name_1, curr_pc_point)
                curr_pc_original_pc_2_curr_points = original_pc_2_cpoints_new[i, 0, :].cpu().numpy() # (original_points_num)
                file_name_2 = os.path.join(preffix, f'train_batch{curr_bn}_{label_i}_pc_original_pc_2_curr_points.npy')
                np.save(file_name_2, curr_pc_original_pc_2_curr_points)
                curr_pc_embeddings = c_pc_feats[i, :, :] # (out_dim, cpoints_num)
                curr_pc_embeddings_normed = F.normalize(curr_pc_embeddings, p=2, dim=0)
                for j in range(len(data_dict['labels'])):
                    label_j = data_dict['labels'][j]
                    curr_pc_original_pc_2_image = original_pc_2_cimg_new[i, j, :].cpu().numpy() # (original_points_num)
                    file_name_3 = os.path.join(preffix, f"train_batch{curr_bn}_{label_i}_2_{label_j}_pc_original_pc_2_image.npy")
                    np.save(file_name_3, curr_pc_original_pc_2_image)
                    curr_pc_overlap_ratio_matrix = pc_overlap_ratio_matrix[i, j, :, :].cpu().numpy() # (cimg_H * cimg_W, cpoints_num)
                    file_name_4 = os.path.join(preffix, f"train_batch{curr_bn}_{label_i}_2_{label_j}_pc_overlap_ratio_matrix.npy")
                    np.save(file_name_4, curr_pc_overlap_ratio_matrix)
                    curr_img_overlap_ratio_matrix = img_overlap_ratio_matrix[i, j, :, :].cpu().numpy() # (cimg_H * cimg_W, cpoints_num)
                    file_name_5 = os.path.join(preffix, f"train_batch{curr_bn}_{label_i}_2_{label_j}_img_overlap_ratio_matrix.npy")
                    np.save(file_name_5, curr_img_overlap_ratio_matrix)

                    curr_image_embeddings = c_img_feats[j, :, :, :] # (out_dim, cimg_H, cimg_W)
                    curr_image_embeddings_normed = F.normalize(curr_image_embeddings, p=2, dim=0)
                    curr_image_embeddings_normed = curr_image_embeddings_normed.flatten(start_dim=1) # (out_dim, cimg_H * cimg_W)
                    curr_pc_similarity = torch.matmul(curr_pc_embeddings_normed.T, curr_image_embeddings_normed) # (cpoints_num, cimg_H * cimg_W)
                    curr_pc_similarity = curr_pc_similarity.detach().cpu().numpy()
                    file_name_6 = os.path.join(preffix, f"train_batch{curr_bn}_{label_i}_2_{label_j}_pc_similarity.npy")
                    np.save(file_name_6, curr_pc_similarity)

    return overlap_ratio_matrix_inuse, img_pair_embeddings, pc_pair_embeddings


@torch.no_grad()
def my_unique(x: tensor, x_num: tensor):
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
    unique_x, x_v2_indices = torch.unique(x_v1, dim=0, 
                                        sorted=True, 
                                        return_inverse=True, 
                                        return_counts=False) # TODO: need test the time spend、the real memory cost
                                                            # produce(huge_num, k+m), (dim1 * dim2 * ... * dimk * n)tensor
    x_num_flattened = x_num.view(-1)
    unique_x_num = torch_scatter.scatter_sum(x_num_flattened, x_v2_indices, dim=-1) # produce (huge_num) tensor
    return unique_x, unique_x_num

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

@torch.no_grad()
def my_unique_v3(x, x_num, dim_info):
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
    torch_sparse.coalesce(x_v1, x_num_flattened, indices + dim_info) # produce (dim1, dim2, ..., dimk, dim_info1, dim_info2, ..., dim_infom) tensor
    x_v1_sparse = torch.sparse_coo_tensor(x_v1.transpose(0, 1), x_num_flattened, indices + dim_info) # produce (dim1, dim2, ..., dimk, dim_info1, dim_info2, ..., dim_infom) tensor
    x_v1_sparse = x_v1_sparse.coalesce() # produce (dim1, dim2, ..., dimk, dim_info1, dim_info2, ..., dim_infom) tensor
    unique_x = x_v1_sparse.indices().transpose(0, 1) # produce (huge_num, k+m) tensor
    unique_x_num = x_v1_sparse.values() # produce (huge_num) tensor
    return unique_x, unique_x_num



def generate_single_correspondence_phase4_v2(c_points, data_dict, img_H, img_W, device, c_img_feats, c_pc_feats, cfgs):
    B, cpoints_num = c_points.shape[:2]
    _, clouds_2_cpoints_idx = keops_knn(device, data_dict["clouds"], c_points, 1)


    clouds_2_cpoints_idx = clouds_2_cpoints_idx.squeeze(-1) # Produces (B, 4096) tensor
    original_pc_2_cpoints = torch.gather(input=clouds_2_cpoints_idx.unsqueeze(1).expand(-1, B, -1),
                                        dim=-1,
                                        index=data_dict["original_pc_2_many_1"][..., 0]) # (B, B, original_points_num)
    
    img_H_mesh = torch.arange(0, img_H, device=device).type(torch.float32)
    img_W_mesh = torch.arange(0, img_W, device=device).type(torch.float32)
    img_H_mesh = img_H_mesh.unsqueeze(1).unsqueeze(2).expand(-1, img_W, -1)
    img_W_mesh = img_W_mesh.unsqueeze(0).unsqueeze(2).expand(img_H, -1, -1)
    img_mesh = torch.cat((img_H_mesh, img_W_mesh), dim=-1) # Produces (img_H, img_W, 2)
    img_mesh = img_mesh.flatten(0, 1) # Produces (img_H * img_W, 2) tensor
    cimg_H, cimg_W = c_img_feats.shape[2:]
    cimg_H_mesh = torch.arange(0, cimg_H, device=device)
    cimg_W_mesh = torch.arange(0, cimg_W, device=device)
    img_2_cimg_scale_H = img_H * 1.0 / cimg_H
    img_2_cimg_scale_W = img_W * 1.0 / cimg_W
    delta_H = img_2_cimg_scale_H / 2 - 0.5
    delta_W = img_2_cimg_scale_W / 2 - 0.5
    cimg_H_mesh = cimg_H_mesh * img_2_cimg_scale_H + delta_H
    cimg_W_mesh = cimg_W_mesh * img_2_cimg_scale_W + delta_W
    cimg_H_mesh = cimg_H_mesh.unsqueeze(1).unsqueeze(2).expand(-1, cimg_W, -1) # Produces (cimg_H, cimg_W, 1) tensor
    cimg_W_mesh = cimg_W_mesh.unsqueeze(0).unsqueeze(2).expand(cimg_H, -1, -1) # Produces (cimg_H, cimg_W, 1) tensor
    cimg_mesh = torch.cat((cimg_H_mesh, cimg_W_mesh), dim=-1) # Produces (cimg_H, cimg_W, 2) tensor
    cimg_mesh = cimg_mesh.flatten(0, 1) # Produces (cimg_H * cimg_W, 2) tensor


    # img_2_cimg_dis = torch.cdist(img_mesh, cimg_mesh) # Produces (img_H * img_W, cimg_H * cimg_W) tensor
    # _, img_2_cimg_idx = torch.topk(img_2_cimg_dis, k=1, dim=1, largest=False, sorted=False) # Produces (img_H * img_W, 1) tensor

    _, img_2_cimg_idx = keops_knn(device, img_mesh, cimg_mesh, 1)

    img_2_cimg_idx = img_2_cimg_idx.squeeze(-1).unsqueeze(0).unsqueeze(0).expand(B, B, -1) # Produces (B, B, img_H * img_W) tensor
    original_pc_2_cimg = torch.gather(input=img_2_cimg_idx,
                                    dim=-1,
                                    index=data_dict["original_pc_2_many_1"][..., 1]) # Produces (B, B, original_points_num)

    overlap_mask = torch.logical_and(data_dict["original_pc_2_many_2"][..., 0], data_dict["original_pc_2_many_2"][..., 1]) # Produces (B, B, original_points_num)
    overlap_mask = overlap_mask.type(original_pc_2_cimg.dtype)


    original_pc_2_cimg_cpoints = torch.stack((original_pc_2_cimg, original_pc_2_cpoints, overlap_mask), dim=-1) # Produces (B, B, original_points_num, 3)
    # method 1: use my unique function
    # t0 = time.time()
    # original_pc_2_cimg_cpoints_unique, original_pc_2_cimg_cpoints_num = my_unique(original_pc_2_cimg_cpoints, 
    #                                                                               torch.ones_like(original_pc_2_cimg_cpoints[..., 0])) # produce (huge_num, 5), (huge_num)
    # t1 = time.time()
    # method 2: use sparse coo tensor's coalesce function
    original_pc_2_cimg_cpoints_unique, original_pc_2_cimg_cpoints_num = my_unique_v2(original_pc_2_cimg_cpoints, 
                                                                                  torch.ones_like(original_pc_2_cimg_cpoints[..., 0]),
                                                                                  (cimg_H * cimg_W, cpoints_num, 2)) # produce (huge_num, 5), (huge_num)
    # t2 = time.time()

    original_pc_2_cimg_cpoints_mask = torch.eq(original_pc_2_cimg_cpoints_unique[..., -1], 1) # produce (huge_num)
    original_pc_2_cimg_cpoints_unique = original_pc_2_cimg_cpoints_unique[original_pc_2_cimg_cpoints_mask, :-1] # produce (huge_num_1, 4)
    original_pc_2_cimg_cpoints_num = original_pc_2_cimg_cpoints_num[original_pc_2_cimg_cpoints_mask] # produce (huge_num_1)
    original_pc_2_cpoints.masked_fill_(~data_dict["original_pc_2_many_2"][..., 1], cpoints_num - 1) # can't be reuse again
    remove_mask_num = torch.count_nonzero(~data_dict["original_pc_2_many_2"][..., 1], dim=-1) # (B, B)
    cpoints_num_pt = torch_scatter.scatter_sum(torch.ones_like(original_pc_2_cpoints, dtype=torch.int32),
                                            original_pc_2_cpoints,
                                            dim=-1,
                                            dim_size=cpoints_num) # produce (B, B, cpoints_num)
    cpoints_num_pt[..., -1] -= remove_mask_num # produce (B, B, cpoints_num)
    cpoints_num_pt = torch.clamp(cpoints_num_pt, min=1)
    original_pc_2_cimg.masked_fill_(~data_dict["original_pc_2_many_2"][..., 0], cimg_H * cimg_W - 1) # can't be reuse again
    non_qualified_mask_num = torch.count_nonzero(~data_dict["original_pc_2_many_2"][..., 0], dim=-1) # produce (B, B)
    cimg_num_pt = torch_scatter.scatter_sum(torch.ones_like(original_pc_2_cimg, dtype=torch.int32),
                                            original_pc_2_cimg,
                                            dim=-1,
                                            dim_size=cimg_H * cimg_W) # produce (B, B, cimg_H * cimg_W)
    cimg_num_pt[..., -1] -= non_qualified_mask_num
    cimg_num_pt = torch.clamp(cimg_num_pt, min=1)
    cpoints_num_pt_unique = cpoints_num_pt[original_pc_2_cimg_cpoints_unique[:, 0], original_pc_2_cimg_cpoints_unique[:, 1], original_pc_2_cimg_cpoints_unique[:, 3]] # produce (huge_num_1)
    cimg_num_pt_unique = cimg_num_pt[original_pc_2_cimg_cpoints_unique[:, 0], original_pc_2_cimg_cpoints_unique[:, 1], original_pc_2_cimg_cpoints_unique[:, 2]] # produce (huge_num_1)
    pc_overlap_ratio_mask = torch.gt(cpoints_num_pt_unique, cfgs.phase4_min_pc_num_pt)
    pc_overlap_ratio = original_pc_2_cimg_cpoints_num * 1.0 / cpoints_num_pt_unique # produce (huge_num_1)
    pc_overlap_ratio = pc_overlap_ratio.masked_fill(~pc_overlap_ratio_mask, 0.0)
    img_overlap_ratio_mask = torch.gt(cimg_num_pt_unique, cfgs.phase4_min_img_num_pt)
    img_overlap_ratio = original_pc_2_cimg_cpoints_num * 1.0 / cimg_num_pt_unique # produce (huge_num_1)
    img_overlap_ratio = img_overlap_ratio.masked_fill(~img_overlap_ratio_mask, 0.0)
    if cfgs.phase4_overlap_matrix_fuse_type == "mean":
        overlap_ratio = 0.5 * pc_overlap_ratio + 0.5 * img_overlap_ratio
    elif cfgs.phase4_overlap_matrix_fuse_type == "max":
        overlap_ratio = torch.maximum(pc_overlap_ratio, img_overlap_ratio)
    elif cfgs.phase4_overlap_matrix_fuse_type == "min":
        overlap_ratio = torch.minimum(pc_overlap_ratio, img_overlap_ratio)
    _, topk_indices = torch.topk(overlap_ratio, k=cfgs.phase4_topk, dim=-1, largest=True, sorted=False) # produce (cfgs.phase4_topk,)
    choose_indices_indices = torch.randperm(cfgs.phase4_topk, dtype=torch.int64, device=device)[:cfgs.phase4_choose_num]
    choose_indices = topk_indices[choose_indices_indices] # produce (cfgs.phase4_choose_num,)
    original_pc_2_cimg_cpoints_unique_choose = original_pc_2_cimg_cpoints_unique[choose_indices] # produce (cfgs.phase4_choose_num, 4)

    # num_pc_1 = pc_pair_indices.shape[0]
    # num_img_1 = img_pair_indices.shape[0]
    # original_pc_2_cimg_cpoints_overlap_ratio_sparse = torch.sparse_coo_tensor(original_pc_2_cimg_cpoints_unique.transpose(0, 1), overlap_ratio, (B, B, cimg_H * cimg_W, cpoints_num)) # produce (B, B, cimg_H * cimg_W, cpoints_num)
    # pc_pair_indices_1 = torch.repeat_interleave(pc_pair_indices, repeats=num_img_1, dim=0) # produce (num_pc_1 * num_img_1)
    # img_pair_indices_1 = img_pair_indices.repeat(num_pc_1, 1) # produce (num_pc_1 * num_img_1)
    # pair_indices_1 = torch.cat((pc_pair_indices_1, img_pair_indices_1), dim=-1) # produce (num_pc_1 * num_img_1, 4)
    # overlap_ratio_vector_inuse = original_pc_2_cimg_cpoints_overlap_ratio_sparse[pair_indices_1[:, 0], pair_indices_1[:, 2], pair_indices_1[:, 3], pair_indices_1[:, 1]] # produce (num_pc_1 * num_img_1)
    # overlap_ratio_matrix_inuse = overlap_ratio_vector_inuse.reshape(num_pc_1, num_img_1) # produce (num_pc_1, num_img_1)

    # t0 = time.time()
    # original_pc_2_cimg_cpoints_unique_v2_1 = original_pc_2_cimg_cpoints_unique[:, 0] * cpoints_num + original_pc_2_cimg_cpoints_unique[:, 3]
    # original_pc_2_cimg_cpoints_unique_v2_2 = original_pc_2_cimg_cpoints_unique[:, 1] * cimg_H * cimg_W + original_pc_2_cimg_cpoints_unique[:, 2]
    # original_pc_2_cimg_cpoints_unique_v2 = torch.stack((original_pc_2_cimg_cpoints_unique_v2_1, original_pc_2_cimg_cpoints_unique_v2_2), dim=-1)
    # original_pc_2_cimg_cpoints_overlap_ratio_sparse_v2 = torch.sparse_coo_tensor(original_pc_2_cimg_cpoints_unique_v2.transpose(0, 1), overlap_ratio, (B * cpoints_num, B * cimg_H * cimg_W))
    # pc_pair_indices_v1 = pc_pair_indices[:, 0] * cpoints_num + pc_pair_indices[:, 1]
    # img_pair_indices_v1 = img_pair_indices[:, 0] * cimg_H * cimg_W + img_pair_indices[:, 1]
    # overlap_ratio_matrix_inuse_temp1 = torch.index_select(original_pc_2_cimg_cpoints_overlap_ratio_sparse_v2, index=pc_pair_indices_v1, dim=0)
    # overlap_ratio_matrix_inuse_temp2 = torch.index_select(overlap_ratio_matrix_inuse_temp1, index=img_pair_indices_v1, dim=1)
    # overlap_ratio_matrix_inuse_temp3 = overlap_ratio_matrix_inuse_temp2.to_dense()
    # overlap_ratio_matrix_inuse_3 = overlap_ratio_matrix_inuse_temp3
    # t1 = time.time()

    # t0 = time.time()
    original_pc_2_cimg_cpoints_unique_v2_1 = original_pc_2_cimg_cpoints_unique[:, 0] * cpoints_num + original_pc_2_cimg_cpoints_unique[:, 3]
    original_pc_2_cimg_cpoints_unique_v2_2 = original_pc_2_cimg_cpoints_unique[:, 1] * cimg_H * cimg_W + original_pc_2_cimg_cpoints_unique[:, 2]
    original_pc_2_cimg_cpoints_unique_v2 = torch.stack((original_pc_2_cimg_cpoints_unique_v2_1, original_pc_2_cimg_cpoints_unique_v2_2), dim=-1)
    original_pc_2_cimg_cpoints_overlap_ratio_sparse_v2 = torch_sparse.SparseTensor(row=original_pc_2_cimg_cpoints_unique_v2[:, 0], 
                                                                                   col=original_pc_2_cimg_cpoints_unique_v2[:, 1], 
                                                                                   value=overlap_ratio, 
                                                                                   sparse_sizes=(B * cpoints_num, B * cimg_H * cimg_W))
    # t1 = time.time()

    if 'attention_in_local' in cfgs.keys() and cfgs.attention_in_local == 1:
        img_pair_indices_non_unique = torch.stack((original_pc_2_cimg_cpoints_unique_choose[:, 1], original_pc_2_cimg_cpoints_unique_choose[:, 2]), dim=-1) # produce (cfgs.phase4_choose_num, 2)
        _, img_pair_nonunique_to_unique_idx = torch.unique(img_pair_indices_non_unique, dim=0, sorted=True, return_inverse=True) # produce (cfgs.phase4_choose_num, )
        new_cimg_mesh = cimg_mesh.unsqueeze(0).expand(B, -1, -1) # produce (B, cimg_H * cimg_W, 2)
        img_pair_mesh_non_unique = new_cimg_mesh[img_pair_indices_non_unique[:, 0], img_pair_indices_non_unique[:, 1]] # produce (cfgs.phase4_choose_num, 2)
        _, img_pair_nonunique_knn_idx = keops_knn(device, img_pair_mesh_non_unique, cimg_mesh, cfgs.img_attention_in_local_k) # produce (cfgs.phase4_choose_num, cfgs.img_attention_in_local_k)
        img_pair_nonunique_knn_sorted, img_pair_nonunique_knn_sorted_idx = torch.sort(img_pair_nonunique_knn_idx, dim=-1) # produce (cfgs.phase4_choose_num, cfgs.img_attention_in_local_k)
        img_pair_nonunique_knn_reverse = torch.nonzero(img_pair_nonunique_knn_sorted_idx == 0, as_tuple=True) # produce ((cfgs.phase4_choose_num,), (cfgs.phase4_choose_num,))
        assert img_pair_nonunique_knn_reverse[0].shape[0] == cfgs.phase4_choose_num
        assert torch.equal(img_pair_nonunique_knn_reverse[0], torch.arange(0, cfgs.phase4_choose_num, device=device))
        img_pair_indices_non_unique_batch = img_pair_indices_non_unique[:, 0] # produce (cfgs.phase4_choose_num)
        img_pair_indices_non_unique_knn_batch = img_pair_indices_non_unique_batch.unsqueeze(-1).expand(-1, cfgs.img_attention_in_local_k) # produce (cfgs.phase4_choose_num, cfgs.img_attention_in_local_k)
        img_pair_nonunique_knn_embeddings = c_img_feats.flatten(start_dim=2)[img_pair_indices_non_unique_knn_batch.reshape(-1), :, img_pair_nonunique_knn_sorted.reshape(-1).type(torch.int64)] # produce (cfgs.phase4_choose_num * cfgs.img_attention_in_local_k, out_dim)
        img_pair_nonunique_knn_embeddings = img_pair_nonunique_knn_embeddings.view(cfgs.phase4_choose_num, cfgs.img_attention_in_local_k, -1)

        pc_pair_indices_non_unique = torch.stack((original_pc_2_cimg_cpoints_unique_choose[:, 0], original_pc_2_cimg_cpoints_unique_choose[:, 3]), dim=-1) # produce (cfgs.phase4_choose_num, 2)
        _, pc_pair_nonunique_to_unique_idx = torch.unique(pc_pair_indices_non_unique, dim=0, sorted=True, return_inverse=True) # produce (cfgs.phase4_choose_num, )
        pc_pair_coord_non_unique = c_points[pc_pair_indices_non_unique[:, 0], pc_pair_indices_non_unique[:, 1]] # produce (cfgs.phase4_choose_num, 3)
        pc_pair_non_unique_sorted_idx = torch.argsort(pc_pair_indices_non_unique[:, 0])
        pc_pair_coord_non_unique_sorted = torch.gather(pc_pair_coord_non_unique,
                                                       dim=0,
                                                       index=pc_pair_non_unique_sorted_idx.unsqueeze(-1).expand(-1, 3))
        pc_pair_coord_non_unique_sorted_idx_reverse = torch_scatter.scatter_sum(src=torch.arange(0, cfgs.phase4_choose_num, device=device),
                                                                                index=pc_pair_non_unique_sorted_idx,
                                                                                dim=-1,
                                                                                dim_size=cfgs.phase4_choose_num) # [cfgs.phase4_choose_num]
        cpoints_batch = torch.full((B,), cpoints_num, dtype=torch.int64, device=device)
        cpoints_offset = torch.cumsum(cpoints_batch, dim=0).int()
        pc_pair_nonunique_batch = torch.bincount(pc_pair_indices_non_unique[:, 0], minlength=B) # produce (B)
        pc_pair_offset = torch.cumsum(pc_pair_nonunique_batch, dim=0).int()
        pc_pair_nonunique_sorted_knn_idx, _ = pointops.knn_query(cfgs.pc_attention_in_local_k, 
                                                          c_points.reshape(-1, 3), 
                                                          cpoints_offset, 
                                                          pc_pair_coord_non_unique_sorted.reshape(-1, 3), 
                                                          pc_pair_offset) 
        pc_pair_nonunique_sorted_knn_idx = pc_pair_nonunique_sorted_knn_idx % cpoints_num # produce (cfgs.phase4_choose_num, cfgs.pc_attention_in_local_k)
        pc_pair_nonunique_knn_idx = torch.gather(pc_pair_nonunique_sorted_knn_idx,
                                                 dim=0,
                                                 index=pc_pair_coord_non_unique_sorted_idx_reverse.unsqueeze(-1).expand(-1, cfgs.pc_attention_in_local_k)) # produce (cfgs.phase4_choose_num, cfgs.pc_attention_in_local_k)
        pc_pair_nonunique_knn_sorted, pc_pair_nonunique_knn_sorted_idx = torch.sort(pc_pair_nonunique_knn_idx, dim=-1) # produce (cfgs.phase4_choose_num, cfgs.pc_attention_in_local_k)
        pc_pair_nonunique_knn_reverse = torch.nonzero(pc_pair_nonunique_knn_sorted_idx == 0, as_tuple=True) # produce ((cfgs.phase4_choose_num,), (cfgs.phase4_choose_num,))
        assert pc_pair_nonunique_knn_reverse[0].shape[0] == cfgs.phase4_choose_num
        assert torch.equal(pc_pair_nonunique_knn_reverse[0], torch.arange(0, cfgs.phase4_choose_num, device=device))
        pc_pair_indices_non_unique_batch = pc_pair_indices_non_unique[:, 0] # produce (cfgs.phase4_choose_num)
        pc_pair_indices_non_unique_knn_batch = pc_pair_indices_non_unique_batch.unsqueeze(-1).expand(-1, cfgs.pc_attention_in_local_k) # produce (cfgs.phase4_choose_num, cfgs.pc_attention_in_local_k)
        pc_pair_nonunique_knn_embeddings = c_pc_feats[pc_pair_indices_non_unique_knn_batch.reshape(-1), :, pc_pair_nonunique_knn_sorted.reshape(-1).type(torch.int64)] # produce (cfgs.phase4_choose_num * cfgs.pc_attention_in_local_k, out_dim)
        pc_pair_nonunique_knn_embeddings = pc_pair_nonunique_knn_embeddings.view(cfgs.phase4_choose_num, cfgs.pc_attention_in_local_k, -1)

        return overlap_ratio_matrix_inuse, img_pair_nonunique_knn_reverse, img_pair_nonunique_knn_embeddings, img_pair_nonunique_to_unique_idx, pc_pair_nonunique_knn_reverse, pc_pair_nonunique_knn_embeddings, pc_pair_nonunique_to_unique_idx
    elif 'attention_in_local' in cfgs.keys() and cfgs.attention_in_local == 2:
        img_pair_indices_non_unique = torch.stack((original_pc_2_cimg_cpoints_unique_choose[:, 1], original_pc_2_cimg_cpoints_unique_choose[:, 2]), dim=-1) # produce (cfgs.phase4_choose_num, 2)
        new_cimg_mesh = cimg_mesh.unsqueeze(0).expand(B, -1, -1) # produce (B, cimg_H * cimg_W, 2)
        img_pair_mesh_non_unique = new_cimg_mesh[img_pair_indices_non_unique[:, 0], img_pair_indices_non_unique[:, 1]] # produce (cfgs.phase4_choose_num, 2)
        _, img_pair_nonunique_knn_idx = keops_knn(device, img_pair_mesh_non_unique, cimg_mesh, cfgs.img_attention_in_local_k) # produce (cfgs.phase4_choose_num, cfgs.img_attention_in_local_k)
        img_pair_nonunique_knn_sorted, _ = torch.sort(img_pair_nonunique_knn_idx, dim=-1) # produce (cfgs.phase4_choose_num, cfgs.img_attention_in_local_k)
        img_pair_indices_non_unique_batch = img_pair_indices_non_unique[:, 0] # produce (cfgs.phase4_choose_num)
        img_pair_indices_non_unique_knn_batch = img_pair_indices_non_unique_batch.unsqueeze(-1).expand(-1, cfgs.img_attention_in_local_k) # produce (cfgs.phase4_choose_num, cfgs.img_attention_in_local_k)
        img_pair_nonunique_knn_embeddings = c_img_feats.flatten(start_dim=2)[img_pair_indices_non_unique_knn_batch.reshape(-1), :, img_pair_nonunique_knn_sorted.reshape(-1).type(torch.int64)] # produce (cfgs.phase4_choose_num * cfgs.img_attention_in_local_k, out_dim)
        img_pair_nonunique_knn_embeddings = img_pair_nonunique_knn_embeddings.view(cfgs.phase4_choose_num, cfgs.img_attention_in_local_k, -1)

        pc_pair_indices_non_unique = torch.stack((original_pc_2_cimg_cpoints_unique_choose[:, 0], original_pc_2_cimg_cpoints_unique_choose[:, 3]), dim=-1) # produce (cfgs.phase4_choose_num, 2)
        pc_pair_coord_non_unique = c_points[pc_pair_indices_non_unique[:, 0], pc_pair_indices_non_unique[:, 1]] # produce (cfgs.phase4_choose_num, 3)
        pc_pair_non_unique_sorted_idx = torch.argsort(pc_pair_indices_non_unique[:, 0])
        pc_pair_coord_non_unique_sorted = torch.gather(pc_pair_coord_non_unique,
                                                       dim=0,
                                                       index=pc_pair_non_unique_sorted_idx.unsqueeze(-1).expand(-1, 3))
        pc_pair_coord_non_unique_sorted_idx_reverse = torch_scatter.scatter_sum(src=torch.arange(0, cfgs.phase4_choose_num, device=device),
                                                                                index=pc_pair_non_unique_sorted_idx,
                                                                                dim=-1,
                                                                                dim_size=cfgs.phase4_choose_num) # [cfgs.phase4_choose_num]
        cpoints_batch = torch.full((B,), cpoints_num, dtype=torch.int64, device=device)
        cpoints_offset = torch.cumsum(cpoints_batch, dim=0).int()
        pc_pair_nonunique_batch = torch.bincount(pc_pair_indices_non_unique[:, 0], minlength=B) # produce (B)
        pc_pair_offset = torch.cumsum(pc_pair_nonunique_batch, dim=0).int()
        pc_pair_nonunique_sorted_knn_idx, _ = pointops.knn_query(cfgs.pc_attention_in_local_k, 
                                                          c_points.reshape(-1, 3), 
                                                          cpoints_offset, 
                                                          pc_pair_coord_non_unique_sorted.reshape(-1, 3), 
                                                          pc_pair_offset) 
        pc_pair_nonunique_sorted_knn_idx = pc_pair_nonunique_sorted_knn_idx % cpoints_num # produce (cfgs.phase4_choose_num, cfgs.pc_attention_in_local_k)
        pc_pair_nonunique_knn_idx = torch.gather(pc_pair_nonunique_sorted_knn_idx,
                                                 dim=0,
                                                 index=pc_pair_coord_non_unique_sorted_idx_reverse.unsqueeze(-1).expand(-1, cfgs.pc_attention_in_local_k)) # produce (cfgs.phase4_choose_num, cfgs.pc_attention_in_local_k)
        pc_pair_nonunique_knn_sorted, _ = torch.sort(pc_pair_nonunique_knn_idx, dim=-1) # produce (cfgs.phase4_choose_num, cfgs.pc_attention_in_local_k)
        pc_pair_indices_non_unique_batch = pc_pair_indices_non_unique[:, 0] # produce (cfgs.phase4_choose_num)
        pc_pair_indices_non_unique_knn_batch = pc_pair_indices_non_unique_batch.unsqueeze(-1).expand(-1, cfgs.pc_attention_in_local_k) # produce (cfgs.phase4_choose_num, cfgs.pc_attention_in_local_k)
        pc_pair_nonunique_knn_embeddings = c_pc_feats[pc_pair_indices_non_unique_knn_batch.reshape(-1), :, pc_pair_nonunique_knn_sorted.reshape(-1).type(torch.int64)] # produce (cfgs.phase4_choose_num * cfgs.pc_attention_in_local_k, out_dim)
        pc_pair_nonunique_knn_embeddings = pc_pair_nonunique_knn_embeddings.view(cfgs.phase4_choose_num, cfgs.pc_attention_in_local_k, -1)

        pc_pair_indices_non_unique_knn_v1 = pc_pair_indices_non_unique_knn_batch * cpoints_num + pc_pair_nonunique_knn_sorted
        img_pair_indices_non_unique_knn_v1 = img_pair_indices_non_unique_knn_batch * cimg_H * cimg_W + img_pair_nonunique_knn_sorted
        pc_pair_indices_non_unique_knn_v1 = pc_pair_indices_non_unique_knn_v1.reshape(-1) # produce (cfgs.phase4_choose_num * cfgs.pc_attention_in_local_k)
        img_pair_indices_non_unique_knn_v1 = img_pair_indices_non_unique_knn_v1.reshape(-1) # produce (cfgs.phase4_choose_num * cfgs.img_attention_in_local_k)
        overlap_ratio_matrix_inuse_temp1 = torch_sparse.index_select(original_pc_2_cimg_cpoints_overlap_ratio_sparse_v2, idx=pc_pair_indices_non_unique_knn_v1, dim=0)
        overlap_ratio_matrix_inuse_temp2 = torch_sparse.index_select(overlap_ratio_matrix_inuse_temp1, idx=img_pair_indices_non_unique_knn_v1, dim=1)
        if not overlap_ratio_matrix_inuse_temp2.is_coalesced():
            overlap_ratio_matrix_inuse_temp2 = overlap_ratio_matrix_inuse_temp2.coalesced()
        overlap_ratio_matrix_inuse_temp2_sparse_coo = overlap_ratio_matrix_inuse_temp2.to_torch_sparse_coo_tensor()
        if not overlap_ratio_matrix_inuse_temp2_sparse_coo.is_coalesced():
            overlap_ratio_matrix_inuse_temp2_sparse_coo = overlap_ratio_matrix_inuse_temp2_sparse_coo.coalesce()
        overlap_ratio_matrix_inuse_temp2_sparse_coo_indices = overlap_ratio_matrix_inuse_temp2_sparse_coo.indices() # produce (2, num_selected)
        overlap_ratio_matrix_inuse_temp2_sparse_coo_value = overlap_ratio_matrix_inuse_temp2_sparse_coo.values() # produce (num_selected)
        mask_to_select = torch.eq(overlap_ratio_matrix_inuse_temp2_sparse_coo_indices[0] // cfgs.pc_attention_in_local_k, overlap_ratio_matrix_inuse_temp2_sparse_coo_indices[1] // cfgs.img_attention_in_local_k)
        overlap_ratio_matrix_inuse_temp2_sparse_coo_indices_selected = overlap_ratio_matrix_inuse_temp2_sparse_coo_indices[:, mask_to_select]
        overlap_ratio_matrix_inuse_temp2_sparse_coo_value_selected = overlap_ratio_matrix_inuse_temp2_sparse_coo_value[mask_to_select]
        new_sparse_coo_indices_selected_dim0 = overlap_ratio_matrix_inuse_temp2_sparse_coo_indices_selected[0] // cfgs.pc_attention_in_local_k
        new_sparse_coo_indices_selected_dim1 = overlap_ratio_matrix_inuse_temp2_sparse_coo_indices_selected[0] % cfgs.pc_attention_in_local_k
        new_sparse_coo_indices_selected_dim2 = overlap_ratio_matrix_inuse_temp2_sparse_coo_indices_selected[1] % cfgs.img_attention_in_local_k
        new_sparse_coo_indices_selected = torch.stack((new_sparse_coo_indices_selected_dim0, new_sparse_coo_indices_selected_dim1, new_sparse_coo_indices_selected_dim2), dim=0) # produce (3, num_selected)
        overlap_ratio_matrix_inuse_temp3 = torch.sparse_coo_tensor(new_sparse_coo_indices_selected, overlap_ratio_matrix_inuse_temp2_sparse_coo_value_selected, (cfgs.phase4_choose_num, cfgs.pc_attention_in_local_k, cfgs.img_attention_in_local_k))
        overlap_ratio_matrix_inuse = overlap_ratio_matrix_inuse_temp3.to_dense() # produce (cfgs.phase4_choose_num, cfgs.pc_attention_in_local_k, cfgs.img_attention_in_local_k)

        return overlap_ratio_matrix_inuse, img_pair_nonunique_knn_embeddings, pc_pair_nonunique_knn_embeddings 
    else:
        img_pair_indices = torch.stack((original_pc_2_cimg_cpoints_unique_choose[:, 1], original_pc_2_cimg_cpoints_unique_choose[:, 2]), dim=-1) # produce (cfgs.phase4_choose_num, 2)
        img_pair_indices = torch.unique(img_pair_indices, dim=0, sorted=True) # produce (num_img_1, 2)
        pc_pair_indices = torch.stack((original_pc_2_cimg_cpoints_unique_choose[:, 0], original_pc_2_cimg_cpoints_unique_choose[:, 3]), dim=-1) # produce (cfgs.phase4__choose_num, 2)
        pc_pair_indices = torch.unique(pc_pair_indices, dim=0, sorted=True) # produce (num_pc_1, 2)
        img_pair_embeddings = c_img_feats.flatten(start_dim=2)[img_pair_indices[:, 0], :, img_pair_indices[:, 1]] # produce (num_img_1, out_dim)
        pc_pair_embeddings = c_pc_feats[pc_pair_indices[:, 0], :, pc_pair_indices[:, 1]] # produce (num_pc_1, out_dim)
        pc_pair_indices_v1 = pc_pair_indices[:, 0] * cpoints_num + pc_pair_indices[:, 1]
        img_pair_indices_v1 = img_pair_indices[:, 0] * cimg_H * cimg_W + img_pair_indices[:, 1]
        overlap_ratio_matrix_inuse_temp1 = torch_sparse.index_select(original_pc_2_cimg_cpoints_overlap_ratio_sparse_v2, idx=pc_pair_indices_v1, dim=0)
        overlap_ratio_matrix_inuse_temp2 = torch_sparse.index_select(overlap_ratio_matrix_inuse_temp1, idx=img_pair_indices_v1, dim=1)
        overlap_ratio_matrix_inuse_temp3 = overlap_ratio_matrix_inuse_temp2.to_dense()
        overlap_ratio_matrix_inuse = overlap_ratio_matrix_inuse_temp3
        return overlap_ratio_matrix_inuse, img_pair_embeddings, pc_pair_embeddings

def generate_single_correspondence_in_pair(c_points, data_dict, img_H, img_W, device, c_img_feats, c_pc_feats, cfgs):
    B, cpoints_num = c_points.shape[:2]
    out_dim = c_img_feats.shape[1]
    _, clouds_2_cpoints_idx = keops_knn(device, data_dict["clouds"], c_points, 1) # Produces (B, 4096, 1) tensor


    clouds_2_cpoints_idx = clouds_2_cpoints_idx.squeeze(-1) # Produces (B, 4096) tensor
    original_pc_2_cpoints = torch.gather(input=clouds_2_cpoints_idx,
                                        dim=-1,
                                        index=data_dict["original_pc_2_many_1"][..., 0]) # (B, original_points_num)
    
    img_H_mesh = torch.arange(0, img_H, device=device).type(torch.float32)
    img_W_mesh = torch.arange(0, img_W, device=device).type(torch.float32)
    img_H_mesh = img_H_mesh.unsqueeze(1).unsqueeze(2).expand(-1, img_W, -1)
    img_W_mesh = img_W_mesh.unsqueeze(0).unsqueeze(2).expand(img_H, -1, -1)
    img_mesh = torch.cat((img_H_mesh, img_W_mesh), dim=-1) # Produces (img_H, img_W, 2)
    img_mesh = img_mesh.flatten(0, 1) # Produces (img_H * img_W, 2) tensor
    cimg_H, cimg_W = c_img_feats.shape[2:]
    cimg_H_mesh = torch.arange(0, cimg_H, device=device)
    cimg_W_mesh = torch.arange(0, cimg_W, device=device)
    img_2_cimg_scale_H = img_H * 1.0 / cimg_H
    img_2_cimg_scale_W = img_W * 1.0 / cimg_W
    delta_H = img_2_cimg_scale_H / 2 - 0.5
    delta_W = img_2_cimg_scale_W / 2 - 0.5
    cimg_H_mesh = cimg_H_mesh * img_2_cimg_scale_H + delta_H
    cimg_W_mesh = cimg_W_mesh * img_2_cimg_scale_W + delta_W
    cimg_H_mesh = cimg_H_mesh.unsqueeze(1).unsqueeze(2).expand(-1, cimg_W, -1) # Produces (cimg_H, cimg_W, 1) tensor
    cimg_W_mesh = cimg_W_mesh.unsqueeze(0).unsqueeze(2).expand(cimg_H, -1, -1) # Produces (cimg_H, cimg_W, 1) tensor
    cimg_mesh = torch.cat((cimg_H_mesh, cimg_W_mesh), dim=-1) # Produces (cimg_H, cimg_W, 2) tensor
    cimg_mesh = cimg_mesh.flatten(0, 1) # Produces (cimg_H * cimg_W, 2) tensor

    _, img_2_cimg_idx = keops_knn(device, img_mesh, cimg_mesh, 1) # Produces (img_H * img_W, 1) tensor

    img_2_cimg_idx = img_2_cimg_idx.squeeze(-1).unsqueeze(0).expand(B, -1) # Produces (B, img_H * img_W) tensor
    original_pc_2_cimg = torch.gather(input=img_2_cimg_idx,
                                    dim=-1,
                                    index=data_dict["original_pc_2_many_1"][..., 1]) # Produces (B, original_points_num)

    overlap_mask = torch.logical_and(data_dict["original_pc_2_many_2"][..., 0], data_dict["original_pc_2_many_2"][..., 1]) # Produces (B, original_points_num)
    overlap_mask = overlap_mask.type(original_pc_2_cimg.dtype)


    original_pc_2_cimg_cpoints = torch.stack((original_pc_2_cimg, original_pc_2_cpoints, overlap_mask), dim=-1) # Produces (B, original_points_num, 3)
    original_pc_2_cimg_cpoints_unique, original_pc_2_cimg_cpoints_num = my_unique_v2(original_pc_2_cimg_cpoints, 
                                                                                  torch.ones_like(original_pc_2_cimg_cpoints[..., 0]),
                                                                                  (cimg_H * cimg_W, cpoints_num, 2)) # produce (huge_num, 4), (huge_num)
    # t2 = time.time()

    original_pc_2_cimg_cpoints_mask = torch.eq(original_pc_2_cimg_cpoints_unique[..., -1], 1) # produce (huge_num)
    original_pc_2_cimg_cpoints_unique = original_pc_2_cimg_cpoints_unique[original_pc_2_cimg_cpoints_mask, :-1] # produce (huge_num_1, 3)
    original_pc_2_cimg_cpoints_num = original_pc_2_cimg_cpoints_num[original_pc_2_cimg_cpoints_mask] # produce (huge_num_1)

    original_pc_2_cpoints.masked_fill_(~data_dict["original_pc_2_many_2"][..., 1], cpoints_num - 1) # can't be reuse again
    remove_mask_num = torch.count_nonzero(~data_dict["original_pc_2_many_2"][..., 1], dim=-1) # (B,)
    cpoints_num_pt = torch_scatter.scatter_sum(torch.ones_like(original_pc_2_cpoints, dtype=torch.int32),
                                            original_pc_2_cpoints,
                                            dim=-1,
                                            dim_size=cpoints_num) # produce (B, cpoints_num)
    cpoints_num_pt[..., -1] -= remove_mask_num # produce (B, cpoints_num)
    cpoints_num_pt = torch.clamp(cpoints_num_pt, min=1)
    original_pc_2_cimg.masked_fill_(~data_dict["original_pc_2_many_2"][..., 0], cimg_H * cimg_W - 1) # can't be reuse again
    non_qualified_mask_num = torch.count_nonzero(~data_dict["original_pc_2_many_2"][..., 0], dim=-1) # produce (B,)
    cimg_num_pt = torch_scatter.scatter_sum(torch.ones_like(original_pc_2_cimg, dtype=torch.int32),
                                            original_pc_2_cimg,
                                            dim=-1,
                                            dim_size=cimg_H * cimg_W) # produce (B, cimg_H * cimg_W)
    cimg_num_pt[..., -1] -= non_qualified_mask_num
    cimg_num_pt = torch.clamp(cimg_num_pt, min=1)
    cpoints_num_pt_unique = cpoints_num_pt[original_pc_2_cimg_cpoints_unique[:, 0], original_pc_2_cimg_cpoints_unique[:, 2]] # produce (huge_num_1)
    cimg_num_pt_unique = cimg_num_pt[original_pc_2_cimg_cpoints_unique[:, 0], original_pc_2_cimg_cpoints_unique[:, 1]] # produce (huge_num_1)
    pc_overlap_ratio_mask = torch.gt(cpoints_num_pt_unique, cfgs.min_pc_num_pt)
    pc_overlap_ratio = original_pc_2_cimg_cpoints_num * 1.0 / cpoints_num_pt_unique # produce (huge_num_1)
    pc_overlap_ratio = pc_overlap_ratio.masked_fill(~pc_overlap_ratio_mask, 0.0)
    img_overlap_ratio_mask = torch.gt(cimg_num_pt_unique, cfgs.min_img_num_pt)
    img_overlap_ratio = original_pc_2_cimg_cpoints_num * 1.0 / cimg_num_pt_unique # produce (huge_num_1)
    img_overlap_ratio = img_overlap_ratio.masked_fill(~img_overlap_ratio_mask, 0.0)
    if cfgs.overlap_matrix_fuse_type == "mean":
        overlap_ratio = 0.5 * pc_overlap_ratio + 0.5 * img_overlap_ratio # produce (huge_num_1)
    elif cfgs.overlap_matrix_fuse_type == "max":
        overlap_ratio = torch.maximum(pc_overlap_ratio, img_overlap_ratio) # produce (huge_num_1)
    elif cfgs.overlap_matrix_fuse_type == "min":
        overlap_ratio = torch.minimum(pc_overlap_ratio, img_overlap_ratio) # produce (huge_num_1)


    if cfgs.pair_type == "top_overlap_among_batch":
        _, topk_indices = torch.topk(overlap_ratio, k=cfgs.topk, dim=-1, largest=True, sorted=False) # produce (cfgs.topk,)
        choose_indices_indices = torch.randperm(cfgs.topk, dtype=torch.int64, device=device)[:cfgs.choose_num]
        choose_indices = topk_indices[choose_indices_indices] # produce (cfgs.choose_num,)
        original_pc_2_cimg_cpoints_unique_choose = original_pc_2_cimg_cpoints_unique[choose_indices] # produce (cfgs.choose_num, 4)

        original_pc_2_cimg_cpoints_unique_v2_1 = original_pc_2_cimg_cpoints_unique[:, 0] * cpoints_num + original_pc_2_cimg_cpoints_unique[:, 2]
        original_pc_2_cimg_cpoints_unique_v2_2 = original_pc_2_cimg_cpoints_unique[:, 1]
        original_pc_2_cimg_cpoints_unique_v2 = torch.stack((original_pc_2_cimg_cpoints_unique_v2_1, original_pc_2_cimg_cpoints_unique_v2_2), dim=-1)
        original_pc_2_cimg_cpoints_overlap_ratio_sparse_v2 = torch_sparse.SparseTensor(row=original_pc_2_cimg_cpoints_unique_v2[:, 0], 
                                                                                        col=original_pc_2_cimg_cpoints_unique_v2[:, 1], 
                                                                                        value=overlap_ratio, 
                                                                                        sparse_sizes=(B * cpoints_num, cimg_H * cimg_W))

        img_pair_indices = torch.stack((original_pc_2_cimg_cpoints_unique_choose[:, 0], original_pc_2_cimg_cpoints_unique_choose[:, 1]), dim=-1) # produce (cfgs.choose_num, 2)
        img_pair_indices = torch.unique(img_pair_indices, dim=0, sorted=True) # produce (num_img_1, 2)
        pc_pair_indices = torch.stack((original_pc_2_cimg_cpoints_unique_choose[:, 0], original_pc_2_cimg_cpoints_unique_choose[:, 2]), dim=-1) # produce (cfgs.choose_num, 2)
        pc_pair_indices = torch.unique(pc_pair_indices, dim=0, sorted=True) # produce (num_pc_1, 2)
        img_pair_embeddings = c_img_feats.flatten(start_dim=2)[img_pair_indices[:, 0], :, img_pair_indices[:, 1]] # produce (num_img_1, out_dim)
        pc_pair_embeddings = c_pc_feats[pc_pair_indices[:, 0], :, pc_pair_indices[:, 1]] # produce (num_pc_1, out_dim)
        pc_pair_indices_v1 = pc_pair_indices[:, 0] * cpoints_num + pc_pair_indices[:, 1]
        img_pair_indices_v1 = img_pair_indices[:, 1]
        overlap_ratio_matrix_inuse_temp1 = torch_sparse.index_select(original_pc_2_cimg_cpoints_overlap_ratio_sparse_v2, idx=pc_pair_indices_v1, dim=0)
        overlap_ratio_matrix_inuse_temp2 = torch_sparse.index_select(overlap_ratio_matrix_inuse_temp1, idx=img_pair_indices_v1, dim=1)
        overlap_ratio_matrix_inuse_temp3 = overlap_ratio_matrix_inuse_temp2.to_dense()
        overlap_ratio_matrix_inuse = overlap_ratio_matrix_inuse_temp3 # produce (num_pc_1, num_img_1)
        in_batch_flag_matrix = pc_pair_indices[:, 0].unsqueeze(-1)  == img_pair_indices[:, 0].unsqueeze(0) # produce (num_pc_1, num_img_1)

        overlap_ratio_matrix_inuse = overlap_ratio_matrix_inuse.unsqueeze(0) # produce (1, num_pc_1, num_img_1)
        img_pair_embeddings = img_pair_embeddings.unsqueeze(0) # produce (1, num_img_1, out_dim)
        pc_pair_embeddings = pc_pair_embeddings.unsqueeze(0) # produce (1, num_pc_1, out_dim)
        in_batch_flag_matrix = in_batch_flag_matrix.unsqueeze(0) # produce (1, num_pc_1, num_img_1)
        return overlap_ratio_matrix_inuse, img_pair_embeddings, pc_pair_embeddings, in_batch_flag_matrix
    
    elif cfgs.pair_type == "fixed_num":
        original_pc_2_cimg_cpoints_unique_pixels = torch.unique(original_pc_2_cimg_cpoints_unique[:, :2], 
                                                                dim=0, 
                                                                sorted=True) # produce (huge_num_2, 2)
        


        # some pair don't have any corresponding c_pixel-to-c_point pairs 
        original_pc_2_cimg_cpoints_unique_pixels_num = torch.bincount(original_pc_2_cimg_cpoints_unique_pixels[:, 0].long(), minlength=B) # produce (B,)





        original_pc_2_cimg_cpoints_unique_pixels_cumsum = torch.cumsum(original_pc_2_cimg_cpoints_unique_pixels_num, dim=0) # produce (B,)
        original_pc_2_cimg_cpoints_unique_pixels_cumsum = torch.cat((torch.zeros(1, device=device), original_pc_2_cimg_cpoints_unique_pixels_cumsum[:-1]), dim=0) # produce (B,)
        pixels_selected_indices_base_1 = original_pc_2_cimg_cpoints_unique_pixels_cumsum.unsqueeze(1).expand(-1, cfgs.pixel_selection_num_each_pair) # produce (B, cfgs.pixel_selection_num_each_pair)
        pixels_selected_indices_base_2 = torch.arange(0, cfgs.pixel_selection_num_each_pair, device=device).unsqueeze(0).expand(B, -1) # produce (B, cfgs.pixel_selection_num_each_pair)
        pixels_selected_indices_qt = torch.div(torch.tensor([cfgs.pixel_selection_num_each_pair], device=device, dtype=torch.int64).expand(B), original_pc_2_cimg_cpoints_unique_pixels_num, rounding_mode='floor') # produce (B,)



        # there exists a zero division problem with fmod
        pixels_selected_indices_base_2_rmd = torch.fmod(pixels_selected_indices_base_2, 
                                                        original_pc_2_cimg_cpoints_unique_pixels_num.unsqueeze(-1).expand(-1, cfgs.pixel_selection_num_each_pair)) # produce (B, cfgs.pixel_selection_num_each_pair)
        pixels_selected_indices_mult = pixels_selected_indices_qt * original_pc_2_cimg_cpoints_unique_pixels_num # produce (B,)
        pixels_selected_indices_base_rand = torch.randint(0, 1000, (B, cfgs.pixel_selection_num_each_pair), device=device) # produce (B, cfgs.pixel_selection_num_each_pair)
        pixels_selected_indices_base_rand_rmd = torch.fmod(pixels_selected_indices_base_rand, 
                                                           original_pc_2_cimg_cpoints_unique_pixels_num.unsqueeze(-1).expand(-1, cfgs.pixel_selection_num_each_pair)) # produce (B, cfgs.pixel_selection_num_each_pair)
        pixels_selected_indices_base_2 = torch.where(pixels_selected_indices_base_2 < pixels_selected_indices_mult.unsqueeze(-1).expand(-1, cfgs.pixel_selection_num_each_pair),
                                                     pixels_selected_indices_base_2_rmd,
                                                     pixels_selected_indices_base_rand_rmd) # produce (B, cfgs.pixel_selection_num_each_pair)
        pixels_selected_indices = pixels_selected_indices_base_1 + pixels_selected_indices_base_2 # produce (B, cfgs.pixel_selection_num_each_pair)


        


        pixels_selected_indices = original_pc_2_cimg_cpoints_unique_pixels[pixels_selected_indices.type(torch.int64), :]  # (B, cfgs.pixel_selection_num_each_pair, 2)
        pixels_selected_indices_v1 = pixels_selected_indices.flatten(0, 1)  # (B * cfgs.pixel_selection_num_each_pair, 2)
        # pixels_selected_indices = pixels_selected_indices.type(torch.int64).flatten(0, 1) # (B * cfgs.pixel_selection_num_each_pair, 2)
        # pixels_selected_indices_v1 = original_pc_2_cimg_cpoints_unique_pixels[pixels_selected_indices, :]  # (B * cfgs.pixel_selection_num_each_pair, 2)





        # pixels_selected_indices_v2 = pixels_selected_indices_v1[:, 0] * cimg_W + pixels_selected_indices_v1[:, 1] # (B * cfgs.pixel_selection_num_each_pair)
        # pixels_selected_indices_v3 = pixels_selected_indices_v2.reshape(B, cfgs.pixel_selection_num_each_pair) # (B, cfgs.pixel_selection_num_each_pair)
        # img_pair_embeddings = torch.gather(c_img_feats.flatten(start_dim=2).permute(0, 2, 1), 
        #                                    dim=1, 
        #                                    index=pixels_selected_indices_v3.unsqueeze(-1).expand(-1, -1, out_dim)) # (B, cfgs.pixel_selection_num_each_pair, out_dim)

        img_pair_embeddings = (c_img_feats.flatten(start_dim=2))[pixels_selected_indices_v1[:, 0], :, pixels_selected_indices_v1[:, 1]] # (B * cfgs.pixel_selection_num_each_pair, out_dim)
        img_pair_embeddings = img_pair_embeddings.reshape(B, -1, out_dim) # (B, cfgs.pixel_selection_num_each_pair, out_dim)


        original_pc_2_cimg_cpoints_unique_v2_1 = original_pc_2_cimg_cpoints_unique[:, 0] * cimg_H * cimg_W + original_pc_2_cimg_cpoints_unique[:, 1]
        original_pc_2_cimg_cpoints_unique_v2_2 = original_pc_2_cimg_cpoints_unique[:, 2]

        # print(f'batch size max: {original_pc_2_cimg_cpoints_unique[:, 0].max()}  '
        #       f'cimg_H * cimg_W max: {original_pc_2_cimg_cpoints_unique[:, 1].max()}  '
        #       f'cpoints_num max: {original_pc_2_cimg_cpoints_unique[:, 2].max()}')

        original_pc_2_cimg_cpoints_unique_v2 = torch.stack((original_pc_2_cimg_cpoints_unique_v2_1, original_pc_2_cimg_cpoints_unique_v2_2), dim=-1)
        original_pc_2_cimg_cpoints_overlap_ratio_sparse_v2 = torch_sparse.SparseTensor(row=original_pc_2_cimg_cpoints_unique_v2[:, 0], 
                                                                                        col=original_pc_2_cimg_cpoints_unique_v2[:, 1], 
                                                                                        value=overlap_ratio, 
                                                                                        sparse_sizes=(B * cimg_H * cimg_W, cpoints_num))


        pixels_selected_indices_v4 = pixels_selected_indices_v1[:, 0] * cimg_H * cimg_W + pixels_selected_indices_v1[:, 1] # (B * cfgs.pixel_selection_num_each_pair)
        overlap_ratio_matrix_temp1 = torch_sparse.index_select(original_pc_2_cimg_cpoints_overlap_ratio_sparse_v2, idx=pixels_selected_indices_v4, dim=0)
        overlap_ratio_matrix_temp1 = overlap_ratio_matrix_temp1.to_dense() # (B * cfgs.pixel_selection_num_each_pair, cpoints)
        _, overlap_ratio_matrix_topk_indices = torch.topk(overlap_ratio_matrix_temp1, 
                                                          k=cfgs.points_selection_num, 
                                                          dim=-1, 
                                                          largest=True, 
                                                          sorted=False) # (B * cfgs.pixel_selection_num_each_pair, cfgs.points_selection_num)
        points_selected_indices = overlap_ratio_matrix_topk_indices.reshape(-1, cfgs.pixel_selection_num_each_pair, cfgs.points_selection_num).flatten(1, 2) # (B, cfgs.pixel_selection_num_each_pair * cfgs.points_selection_num)

        pc_pair_embeddings = torch.gather(c_pc_feats.permute(0, 2, 1),
                                          dim=1,
                                          index=points_selected_indices.unsqueeze(-1).expand(-1, -1, out_dim)) # (B, cfgs.pixel_selection_num_each_pair * cfgs.points_selection_num, out_dim)

        points_selected_indices_v1 = points_selected_indices.unsqueeze(1).expand(-1, cfgs.pixel_selection_num_each_pair, -1).flatten(0, 1) # (B * cfgs.pixel_selection_num_each_pair, cfgs.pixel_selection_num_each_pair * cfgs.points_selection_num)
        pixels_to_points_dist = torch.gather(overlap_ratio_matrix_temp1,
                                             dim=-1,
                                             index=points_selected_indices_v1) # (B * cfgs.pixel_selection_num_each_pair, cfgs.pixel_selection_num_each_pair * cfgs.points_selection_num)
        overlap_ratio_matrix_inuse = pixels_to_points_dist.reshape(B, cfgs.pixel_selection_num_each_pair, cfgs.pixel_selection_num_each_pair * cfgs.points_selection_num)


        overlap_ratio_matrix_inuse = overlap_ratio_matrix_inuse.permute(0, 2, 1) # (B, cfgs.pixel_selection_num_each_pair * cfgs.points_selection_num, cfgs.pixel_selection_num_each_pair)

        in_batch_flag_matrix = torch.ones_like(overlap_ratio_matrix_inuse, dtype=torch.bool) # (B, cfgs.pixel_selection_num_each_pair * cfgs.points_selection_num, cfgs.pixel_selection_num_each_pair)

        return overlap_ratio_matrix_inuse, img_pair_embeddings, pc_pair_embeddings, in_batch_flag_matrix

# just select the nearest point as the correspondence
# also make 2 situations: 1、select topk points among all and it's best match among all(except current frame)
# 2、select topk points for every frame and it's best match among all(except current frame)
# use generate_original_pc_correspondence_v3 to generate original_pc_2_many
def generate_single_correspondence_for_pc(c_points, data_dict, device, c_pc_feats, cfgs):
    B, cpoints_num = c_points.shape[:2]
    out_dim = c_pc_feats.shape[1]
    cloud_poses = data_dict["cloud_poses"] # (B, 4, 4)
    # _, clouds_2_cpoints_idx = keops_knn(device, data_dict["clouds"], c_points, 1)

    # clouds_2_cpoints_idx = clouds_2_cpoints_idx.squeeze(-1) # Produces (B, 4096) tensor

    clouds_original = data_dict["clouds_original"] # (B, original_points_num, 3)
    cloud_poses_original = data_dict['cloud_poses_original'] # (B, 4, 4)

    original_pc_mask = data_dict['original_cloud_remove_masks'] # Produces (B, original_points_num)
    original_pc_mask = original_pc_mask.type(torch.int64) # Produces (B, original_points_num)

    knn_num = clouds_original.shape[1] // cpoints_num # TODO: may need to be set as a hyperparameter

    local_correspondence_in_k = data_dict['local_correspondence_in_k']

    clouds_original[~original_pc_mask, :] = 1e6 # in this way, the removed points won't be selected as correspondence

    cpoints_to_original_pc_dist, cpoints_to_original_pc_idx = keops_knn(device, c_points, clouds_original, knn_num) # (B, cpoints_num, knn_num)
    cpoints_to_original_pc_dist_max = torch.max(cpoints_to_original_pc_dist, dim=-1)[0] # (B, cpoints_num)
    pair_indices = list(itertools.combinations(range(local_correspondence_in_k + 1), 2))
    pair_indices = torch.tensor(pair_indices, dtype=torch.int64, device=device) # (local_correspondence_in_k * (local_correspondence_in_k + 1) / 2, 2)
    pair_indices = pair_indices.unsqueeze(0).expand(B // (1+local_correspondence_in_k), -1, -1).flatten(0, 1) # (B * local_correspondence_in_k / 2, 2)
    batch_indices = torch.arange(0, B // (1 + local_correspondence_in_k), device=device).unsqueeze(-1).expand(-1, local_correspondence_in_k * (local_correspondence_in_k + 1) // 2).flatten() # (B * local_correspondence_in_k / 2)
    batch_indices = batch_indices * (local_correspondence_in_k + 1) # (B * local_correspondence_in_k / 2)
    pair_indices = batch_indices.unsqueeze(-1) + pair_indices # (B * local_correspondence_in_k / 2, 2)
    pair_1_cpoints = c_points[pair_indices[:, 0]] # (B * local_correspondence_in_k / 2, cpoints_num, 3)
    pair_2_cpoints = c_points[pair_indices[:, 1]] # (B * local_correspondence_in_k / 2, cpoints_num, 3)

    T_pair_1_pair_2_cpoints = torch.matmul(torch.linalg.inv(cloud_poses[pair_indices[:, 0], ...]), cloud_poses[pair_indices[:, 1], ...].permute(0, 2, 1)) # (B * local_correspondence_in_k / 2, 4, 4)
    pair_2_cpoints_to_mult = torch.cat((pair_2_cpoints, torch.ones_like(pair_2_cpoints[..., :1])), dim=-1) # (B * local_correspondence_in_k / 2, cpoints_num, 4)
    pair_2_cpoints_mult = torch.matmul(pair_2_cpoints_to_mult, T_pair_1_pair_2_cpoints.permute(0, 2, 1)) # (B * local_correspondence_in_k / 2, cpoints_num, 4)
    pair_2_cpoints = pair_2_cpoints_mult[..., :3] # (B * local_correspondence_in_k / 2, cpoints_num, 3)

    cpoints_to_cpoints_dist = torch.cdist(pair_1_cpoints, pair_2_cpoints, p=2.0) # (B * local_correspondence_in_k / 2, cpoints_num, cpoints_num)
    cpoints_to_original_pc_dist_max_pair_1 = cpoints_to_original_pc_dist_max[pair_indices[:, 0]] # (B * local_correspondence_in_k / 2, cpoints_num)
    cpoints_to_original_pc_dist_max_pair_2 = cpoints_to_original_pc_dist_max[pair_indices[:, 1]] # (B * local_correspondence_in_k / 2, cpoints_num)
    intersect_mat = torch.gt(cpoints_to_original_pc_dist_max_pair_1.unsqueeze(-1) + cpoints_to_original_pc_dist_max_pair_2.unsqueeze(1) + cfgs.onlypc_pos_radius - cpoints_to_cpoints_dist, 0) # (B * local_correspondence_in_k / 2, cpoints_num, cpoints_num)
    potential_pair_indices, potential_cpoints_indices_1, potential_cpoints_indices_2 = torch.nonzero(intersect_mat) # (huge_num1,) (huge_num1,) (huge_num1,)
    cpoints_to_original_knn_points = torch.gather(clouds_original.unsqueeze(1).expand(-1, cpoints_num, -1, -1),
                                                  dim=2,
                                                  index=cpoints_to_original_pc_idx.unsqueeze(-1).expand(-1, -1, -1, 3)) # (B, cpoints_num, knn_num, 3)
    
    cpoints_to_original_knn_points_pair_1 = cpoints_to_original_knn_points[pair_indices[:, 0], ...] # (B * local_correspondence_in_k / 2, cpoints_num, knn_num, 3)
    cpoints_to_original_knn_points_pair_2 = cpoints_to_original_knn_points[pair_indices[:, 1], ...] # (B * local_correspondence_in_k / 2, cpoints_num, knn_num, 3)
    cpoints_to_original_knn_points_pair_2_to_mult = torch.cat((cpoints_to_original_knn_points_pair_2, torch.ones_like(cpoints_to_original_knn_points_pair_2[..., :1])), dim=-1) # (B * local_correspondence_in_k / 2, cpoints_num, knn_num, 4)
    T_pair_1_pair_2_original = torch.matmul(torch.linalg.inv(cloud_poses_original[pair_indices[:, 0], ...]), cloud_poses_original[pair_indices[:, 1], ...].permute(0, 2, 1)) # (B * local_correspondence_in_k / 2, 4, 4)
    cpoints_to_original_knn_points_pair_2_mult = torch.matmul(cpoints_to_original_knn_points_pair_2_to_mult, T_pair_1_pair_2_original.unsqueeze(1).permute(0, 1, 3, 2)) # (B * local_correspondence_in_k / 2, cpoints_num, knn_num, 4)
    cpoints_to_original_knn_points_pair_2 = cpoints_to_original_knn_points_pair_2_mult[..., :3] # (B * local_correspondence_in_k / 2, cpoints_num, knn_num, 3)

    cpoints_to_original_knn_points_potential_1 = cpoints_to_original_knn_points_pair_1[potential_pair_indices, potential_cpoints_indices_1, ...] # (huge_num1, knn_num, 3)
    cpoints_to_original_knn_points_potential_2 = cpoints_to_original_knn_points_pair_2[potential_pair_indices, potential_cpoints_indices_2, ...] # (huge_num1, knn_num, 3)
    cpoints_to_original_knn_points_potential_dist = torch.cdist(cpoints_to_original_knn_points_potential_1, cpoints_to_original_knn_points_potential_2, p=2.0) # (huge_num1, knn_num, knn_num)
    cpoints_to_original_knn_points_potential_overlap_mat = torch.lt(cpoints_to_original_knn_points_potential_dist, cfgs.onlypc_pos_radius)
    potential_1_overlap_counts = torch.count_nonzero(cpoints_to_original_knn_points_potential_overlap_mat.sum(-1), dim=-1).float()  # (huge_num1,)
    potential_2_overlap_counts = torch.count_nonzero(cpoints_to_original_knn_points_potential_overlap_mat.sum(-2), dim=-1).float()  # (huge_num1,)
    potential_1_overlap_ratio = potential_1_overlap_counts / float(knn_num)
    potential_2_overlap_ratio = potential_2_overlap_counts / float(knn_num)
    overlap_ratio = (potential_1_overlap_ratio + potential_2_overlap_ratio) / 2.0 # (huge_num1,)

    overlap_ratio_mask = torch.gt(overlap_ratio, 0.0)
    potential_pair_indices = potential_pair_indices[overlap_ratio_mask] # (huge_num2,)
    potential_cpoints_indices_1 = potential_cpoints_indices_1[overlap_ratio_mask] # (huge_num2,)
    potential_cpoints_indices_2 = potential_cpoints_indices_2[overlap_ratio_mask] # (huge_num2,)
    overlap_ratio = overlap_ratio[overlap_ratio_mask] # (huge_num2,)

    overlap_ratio_mask2 = torch.gt(overlap_ratio, cfgs.onlypc_min_overlap_ratio)
    potential_pair_indices = potential_pair_indices[overlap_ratio_mask2] # (huge_num3,)
    potential_cpoints_indices_1 = potential_cpoints_indices_1[overlap_ratio_mask2] # (huge_num3,)
    potential_cpoints_indices_2 = potential_cpoints_indices_2[overlap_ratio_mask2] # (huge_num3,)
    overlap_ratio = overlap_ratio[overlap_ratio_mask2] # (huge_num3,)

    sparse_tensor_indices = torch.stack((potential_pair_indices, potential_cpoints_indices_1, potential_cpoints_indices_2), dim=0) # (3, huge_num3)
    sparse_tensor_values = overlap_ratio # (huge_num3,)
    whole_overlap_ratio_matrix = torch.sparse_coo_tensor(sparse_tensor_indices, sparse_tensor_values, (B * local_correspondence_in_k / 2, cpoints_num, cpoints_num), device=device) # (B * local_correspondence_in_k / 2, cpoints_num, cpoints_num)
    whole_overlap_ratio_matrix = whole_overlap_ratio_matrix.to_dense() # (B * local_correspondence_in_k / 2, cpoints_num, cpoints_num)
    c_pc_feats_pair_1 = c_pc_feats[pair_indices[..., 0], ...].permute(0, 2, 1) # (B * local_correspondence_in_k / 2, cpoints_num, out_dim)
    c_pc_feats_pair_2 = c_pc_feats[pair_indices[..., 1], ...].permute(0, 2, 1) # (B * local_correspondence_in_k / 2, cpoints_num, out_dim)

    if cfgs.onlypc_pair_type == "topk_overlap_in_pair":
        whole_overlap_ratio_matrix_flatten = whole_overlap_ratio_matrix.flatten(1, 2) # (B * local_correspondence_in_k / 2, cpoints_num * cpoints_num)
        _, topk_overlap_ratio_in_pair_indices = torch.topk(whole_overlap_ratio_matrix_flatten, k=cfgs.onlypc_topk, dim=-1, largest=True, sorted=False) # (B * local_correspondence_in_k / 2, cfgs.onlypc_topk)
        topk_overlap_ratio_in_pair_indices_1 = topk_overlap_ratio_in_pair_indices // cpoints_num # (B * local_correspondence_in_k / 2, cfgs.onlypc_topk)
        topk_overlap_ratio_in_pair_indices_2 = topk_overlap_ratio_in_pair_indices % cpoints_num # (B * local_correspondence_in_k / 2, cfgs.onlypc_topk)
        topk_overlap_embeddings_1 = torch.gather(c_pc_feats_pair_1, dim=1, index=topk_overlap_ratio_in_pair_indices_1.unsqueeze(-1).expand(-1, -1, out_dim)) # (B * local_correspondence_in_k / 2, cfgs.onlypc_topk, out_dim)
        topk_overlap_embeddings_2 = torch.gather(c_pc_feats_pair_2, dim=1, index=topk_overlap_ratio_in_pair_indices_2.unsqueeze(-1).expand(-1, -1, out_dim)) # (B * local_correspondence_in_k / 2, cfgs.onlypc_topk, out_dim)
        whole_overlap_ratio_matrix_temp1 = torch.gather(whole_overlap_ratio_matrix, dim=1, index=topk_overlap_ratio_in_pair_indices_1.unsqueeze(-1).expand(-1, -1, cpoints_num)) # (B * local_correspondence_in_k / 2, cfgs.onlypc_topk, cpoints_num)
        final_overlap_ratio_matrix = torch.gather(whole_overlap_ratio_matrix_temp1, dim=2, index=topk_overlap_ratio_in_pair_indices_2.unsqueeze(1).expand(-1, cfgs.onlypc_topk, -1)) # (B * local_correspondence_in_k / 2, cfgs.onlypc_topk, cfgs.onlypc_topk)
        in_batch_flag_matrix = torch.ones_like(final_overlap_ratio_matrix, dtype=torch.bool) # (B * local_correspondence_in_k / 2, cfgs.onlypc_topk, cfgs.onlypc_topk)
        return final_overlap_ratio_matrix, topk_overlap_embeddings_1, topk_overlap_embeddings_2, in_batch_flag_matrix
    elif cfgs.onlypc_pair_type == "topk_overlap_among_all":
        _, topk_indices = torch.topk(overlap_ratio, dim=-1, k=cfgs.onlypc_topk, largest=True, sorted=False) # (cfgs.onlypc_topk,)
        topk_potential_pair_indices = potential_pair_indices[topk_indices] # (cfgs.onlypc_topk,)
        topk_potential_cpoints_indices_1 = potential_cpoints_indices_1[topk_indices] # (cfgs.onlypc_topk,)
        topk_potential_cpoints_indices_2 = potential_cpoints_indices_2[topk_indices] # (cfgs.onlypc_topk,)

        topk_overlap_embeddings_1 = c_pc_feats_pair_1[topk_potential_pair_indices, topk_potential_cpoints_indices_1, :] # (cfgs.onlypc_topk, out_dim)
        topk_overlap_embeddings_2 = c_pc_feats_pair_2[topk_potential_pair_indices, topk_potential_cpoints_indices_2, :] # (cfgs.onlypc_topk, out_dim)
        whole_overlap_ratio_matrix_temp1 = whole_overlap_ratio_matrix[topk_potential_pair_indices, topk_potential_cpoints_indices_1] # (cfgs.onlypc_topk, cpoints_num)
        final_overlap_ratio_matrix = whole_overlap_ratio_matrix_temp1[:, topk_potential_cpoints_indices_2] # (cfgs.onlypc_topk, cfgs.onlypc_topk)
        in_batch_flag_matrix = topk_potential_pair_indices.unsqueeze(-1) == topk_potential_pair_indices.unsqueeze(0) # (cfgs.onlypc_topk, cfgs.onlypc_topk)

        return (final_overlap_ratio_matrix.unsqueeze(0), 
                topk_overlap_embeddings_1.unsqueeze(0), 
                topk_overlap_embeddings_2.unsqueeze(0), 
                in_batch_flag_matrix.unsqueeze(0))
    else:
        raise NotImplementedError
    

def generate_cluster_correspondence(c_points, data_dict, img_H, img_W, device, c_img_feats, c_pc_feats, cfgs):
    B, cpoints_num = c_points.shape[:2]
    out_dim = c_img_feats.shape[1]
    _, clouds_2_cpoints_idx = keops_knn(device, data_dict["clouds"], c_points, 1)


    clouds_2_cpoints_idx = clouds_2_cpoints_idx.squeeze(-1) # Produces (B, 4096) tensor
    original_pc_2_cpoints = torch.gather(input=clouds_2_cpoints_idx.unsqueeze(1).expand(-1, B, -1),
                                        dim=-1,
                                        index=data_dict["original_pc_2_many_1"][..., 0]) # (B, B, original_points_num)
    
    img_H_mesh = torch.arange(0, img_H, device=device).type(torch.float32)
    img_W_mesh = torch.arange(0, img_W, device=device).type(torch.float32)
    img_H_mesh = img_H_mesh.unsqueeze(1).unsqueeze(2).expand(-1, img_W, -1)
    img_W_mesh = img_W_mesh.unsqueeze(0).unsqueeze(2).expand(img_H, -1, -1)
    img_mesh = torch.cat((img_H_mesh, img_W_mesh), dim=-1) # Produces (img_H, img_W, 2)
    img_mesh = img_mesh.flatten(0, 1) # Produces (img_H * img_W, 2) tensor
    cimg_H, cimg_W = c_img_feats.shape[2:]
    cimg_H_mesh = torch.arange(0, cimg_H, device=device)
    cimg_W_mesh = torch.arange(0, cimg_W, device=device)
    img_2_cimg_scale_H = img_H * 1.0 / cimg_H
    img_2_cimg_scale_W = img_W * 1.0 / cimg_W
    delta_H = img_2_cimg_scale_H / 2 - 0.5
    delta_W = img_2_cimg_scale_W / 2 - 0.5
    cimg_H_mesh = cimg_H_mesh * img_2_cimg_scale_H + delta_H
    cimg_W_mesh = cimg_W_mesh * img_2_cimg_scale_W + delta_W
    cimg_H_mesh = cimg_H_mesh.unsqueeze(1).unsqueeze(2).expand(-1, cimg_W, -1) # Produces (cimg_H, cimg_W, 1) tensor
    cimg_W_mesh = cimg_W_mesh.unsqueeze(0).unsqueeze(2).expand(cimg_H, -1, -1) # Produces (cimg_H, cimg_W, 1) tensor
    cimg_mesh = torch.cat((cimg_H_mesh, cimg_W_mesh), dim=-1) # Produces (cimg_H, cimg_W, 2) tensor
    cimg_mesh = cimg_mesh.flatten(0, 1) # Produces (cimg_H * cimg_W, 2) tensor

    _, img_2_cimg_idx = keops_knn(device, img_mesh, cimg_mesh, 1)

    img_2_cimg_idx = img_2_cimg_idx.squeeze(-1).unsqueeze(0).unsqueeze(0).expand(B, B, -1) # Produces (B, B, img_H * img_W) tensor
    original_pc_2_cimg = torch.gather(input=img_2_cimg_idx,
                                    dim=-1,
                                    index=data_dict["original_pc_2_many_1"][..., 1]) # Produces (B, B, original_points_num)
    
    img_semantic_label = data_dict['img_semantic_label'] # (B, 1, semantic_img_H, semantic_img_W)
    img_ccl_cluster_label = data_dict['img_ccl_cluster_label'] # (B, 1, semantic_img_H, semantic_img_W)

    # TODO: this may lead some problems if there exist some new semantic-cluster corresponding relationships
    if img_semantic_label.shape[2:] != (cimg_H, cimg_W):
        img_semantic_label_float = img_semantic_label.type(torch.float32)
        img_ccl_cluster_label_float = img_ccl_cluster_label.type(torch.float32)
        img_semantic_label_float = F.interpolate(img_semantic_label_float, (cimg_H, cimg_W), mode='nearest')
        img_ccl_cluster_label_float = F.interpolate(img_ccl_cluster_label_float, (cimg_H, cimg_W), mode='nearest')
        img_semantic_label = img_semantic_label_float.type(torch.int64)
        img_ccl_cluster_label = img_ccl_cluster_label_float.type(torch.int64)

    pc_semantic_label = data_dict['pc_semantic_label'] # (B, 1, cpoints_num)
    pc_dbscan_cluster_label = data_dict['pc_dbscan_cluster_label'] # (B, 1, cpoints_num)
    img_semantic_label += 1 # let the -1 to be 0, it's convenient for the later process
    img_ccl_cluster_label += 1
    pc_semantic_label += 1
    pc_dbscan_cluster_label += 1
    max_cimg_semantic_inuse = 10 + 1
    max_cimg_ccl_cluster = torch.max(img_ccl_cluster_label) + 1

    # use the pc semantic seg model or not
    cityscapes_label_in_semanticKitti_label_list = copy.deepcopy(data_dict['cityscapes_label_in_semanticKitti_label_list'])
    label_dim = len(cityscapes_label_in_semanticKitti_label_list[0])
    cityscapes_label_in_semanticKitti_label_list.insert(0, [-1 for _ in range(label_dim)])
    cityscapes_label_in_semanticKitti_label = torch.tensor(cityscapes_label_in_semanticKitti_label_list, device=device) # (max_cimg_semantic, label_dim)

    if cityscapes_label_in_semanticKitti_label.max() != 18:
        raise ValueError("The max value of cityscapes_label_in_semanticKitti_label should be 9")
        max_cpoints_semantic_inuse = 12 + 1
    else:
        max_cpoints_semantic_inuse = 10 + 1
    
    max_cpoints_dbscan_cluster = torch.max(pc_dbscan_cluster_label) + 1
    assert max_cimg_semantic_inuse >= (torch.max(img_semantic_label) + 1)
    assert max_cpoints_semantic_inuse >= (torch.max(pc_semantic_label) + 1)
    
    original_pc_2_cimg_semantic = torch.gather(input=img_semantic_label.squeeze(1).unsqueeze(0).expand(B, -1, -1, -1).flatten(2),
                                            dim=-1,
                                            index=original_pc_2_cimg) # Produces (B, B, original_points_num)
    original_pc_2_cimg_ccl_cluster = torch.gather(input=img_ccl_cluster_label.squeeze(1).unsqueeze(0).expand(B, -1, -1, -1).flatten(2),
                                            dim=-1,
                                            index=original_pc_2_cimg) # Produces (B, B, original_points_num)
    
    original_pc_2_pc_semantic = torch.gather(input=pc_semantic_label.expand(-1, B, -1),
                                            dim=-1,
                                            index=original_pc_2_cpoints) # Produces (B, B, original_points_num)
    original_pc_2_pc_dbscan_cluster = torch.gather(input=pc_dbscan_cluster_label.expand(-1, B, -1),
                                            dim=-1,
                                            index=original_pc_2_cpoints) # Produces (B, B, original_points_num)

    overlap_mask = torch.logical_and(data_dict["original_pc_2_many_2"][..., 0], data_dict["original_pc_2_many_2"][..., 1]) # Produces (B, B, original_points_num)
    overlap_mask = overlap_mask.type(original_pc_2_cimg.dtype)


    original_pc_2_cimg_cpoints = torch.stack((original_pc_2_cimg_semantic,
                                                  original_pc_2_cimg_ccl_cluster, 
                                                  original_pc_2_pc_semantic,
                                                  original_pc_2_pc_dbscan_cluster, 
                                                  overlap_mask), dim=-1) # Produces (B, B, original_points_num, 5)
    original_pc_2_cimg_cpoints_unique, original_pc_2_cimg_cpoints_num = my_unique_v2(original_pc_2_cimg_cpoints, 
                                                                                  torch.ones_like(original_pc_2_cimg_cpoints[..., 0]),
                                                                                  (max_cimg_semantic_inuse, 
                                                                                   max_cimg_ccl_cluster, 
                                                                                   max_cpoints_semantic_inuse, 
                                                                                   max_cpoints_dbscan_cluster, 
                                                                                   2)) # produce (huge_num, 7), (huge_num)
    # t2 = time.time()

    original_pc_2_cimg_cpoints_mask = torch.eq(original_pc_2_cimg_cpoints_unique[..., -1], 1) # produce (huge_num)
    original_pc_2_cimg_cpoints_unique = original_pc_2_cimg_cpoints_unique[original_pc_2_cimg_cpoints_mask, :-1] # produce (huge_num_1, 6)
    original_pc_2_cimg_cpoints_num = original_pc_2_cimg_cpoints_num[original_pc_2_cimg_cpoints_mask] # produce (huge_num_1)

    original_pc_2_cimg_cpoints_mask1 = torch.eq(original_pc_2_cimg_cpoints_unique[:, 2], 0)
    original_pc_2_cimg_cpoints_mask2 = torch.eq(original_pc_2_cimg_cpoints_unique[:, 3], 0)
    original_pc_2_cimg_cpoints_mask3 = torch.eq(original_pc_2_cimg_cpoints_unique[:, 4], 0)
    original_pc_2_cimg_cpoints_mask4 = torch.eq(original_pc_2_cimg_cpoints_unique[:, 5], 0)
    original_pc_2_cimg_cpoints_mask_v2 = original_pc_2_cimg_cpoints_mask1 | original_pc_2_cimg_cpoints_mask2 | original_pc_2_cimg_cpoints_mask3 | original_pc_2_cimg_cpoints_mask4
    original_pc_2_cimg_cpoints_unique_2 = original_pc_2_cimg_cpoints_unique[~original_pc_2_cimg_cpoints_mask_v2] # produce (huge_num_2, 6)
    original_pc_2_cimg_cpoints_num_2 = original_pc_2_cimg_cpoints_num[~original_pc_2_cimg_cpoints_mask_v2] # produce (huge_num_2)

    original_pc_2_cimg_cpoints_unique_img_semantic = original_pc_2_cimg_cpoints_unique_2[:, 2] # produce (huge_num_2)

    original_pc_2_cimg_cpoints_unique_choose1_img_to_pc_semantic = torch.gather(cityscapes_label_in_semanticKitti_label,
                                                                                index=original_pc_2_cimg_cpoints_unique_img_semantic.unsqueeze(1).expand(-1, label_dim),
                                                                                dim=0) # produce (huge_num_2, label_dim)
    original_pc_2_cimg_cpoints_unique_choose1_img_to_pc_semantic += 1
    original_pc_2_cimg_cpoints_unique_mask1 = original_pc_2_cimg_cpoints_unique_choose1_img_to_pc_semantic == original_pc_2_cimg_cpoints_unique_2[:, 4].unsqueeze(-1).expand(-1, label_dim) # produce (huge_num_2, label_dim)
    original_pc_2_cimg_cpoints_unique_mask2 = original_pc_2_cimg_cpoints_unique_mask1.any(dim=-1) # produce (huge_num_2)
    original_pc_2_cimg_cpoints_unique_3 = original_pc_2_cimg_cpoints_unique_2[original_pc_2_cimg_cpoints_unique_mask2] # produce (huge_num_3, 6)
    original_pc_2_cimg_cpoints_num_3 = original_pc_2_cimg_cpoints_num_2[original_pc_2_cimg_cpoints_unique_mask2] # produce (huge_num_3)

    original_pc_2_cpoints.masked_fill_(~data_dict["original_pc_2_many_2"][..., 1], cpoints_num - 1) # can't be reuse again
    remove_mask_num = torch.count_nonzero(~data_dict["original_pc_2_many_2"][..., 1], dim=-1) # (B, B)
    cpoints_num_pt = torch_scatter.scatter_sum(torch.ones_like(original_pc_2_cpoints, dtype=torch.int32),
                                            original_pc_2_cpoints,
                                            dim=-1,
                                            dim_size=cpoints_num) # produce (B, B, cpoints_num)
    cpoints_num_pt[..., -1] -= remove_mask_num # produce (B, B, cpoints_num)
    dbscan_cluster_num_pt = torch_scatter.scatter_sum(cpoints_num_pt,
                                                      pc_dbscan_cluster_label.expand(-1, B, -1),
                                                      dim=-1,
                                                      dim_size=max_cpoints_dbscan_cluster) # produce (B, B, max_cpoints_dbscan_cluster)
    dbscan_cluster_num_pt = torch.clamp(dbscan_cluster_num_pt, min=1.0)
    dbscan_cluster_num_pt_unique = dbscan_cluster_num_pt[original_pc_2_cimg_cpoints_unique_3[:, 0], original_pc_2_cimg_cpoints_unique_3[:, 1], original_pc_2_cimg_cpoints_unique_3[:, 5]] # produce (huge_num_3)
    dbscan_cluster_overlap_ratio_mask = torch.gt(dbscan_cluster_num_pt_unique, cfgs.min_pc_dbscan_cluster_num_pt)
    dbscan_cluster_overlap_ratio = original_pc_2_cimg_cpoints_num_3 * 1.0 / dbscan_cluster_num_pt_unique # produce (huge_num_3)
    dbscan_cluster_overlap_ratio = dbscan_cluster_overlap_ratio.masked_fill(~dbscan_cluster_overlap_ratio_mask, 0.0)



    original_pc_2_cimg.masked_fill_(~data_dict["original_pc_2_many_2"][..., 0], cimg_H * cimg_W - 1) # can't be reuse again
    non_qualified_mask_num = torch.count_nonzero(~data_dict["original_pc_2_many_2"][..., 0], dim=-1) # produce (B, B)
    cimg_num_pt = torch_scatter.scatter_sum(torch.ones_like(original_pc_2_cimg, dtype=torch.int32),
                                            original_pc_2_cimg,
                                            dim=-1,
                                            dim_size=cimg_H * cimg_W) # produce (B, B, cimg_H * cimg_W)
    cimg_num_pt[..., -1] -= non_qualified_mask_num
    ccl_cluster_num_pt = torch_scatter.scatter_sum(cimg_num_pt,
                                                  img_ccl_cluster_label.squeeze(1).unsqueeze(0).expand(B, -1, -1, -1).flatten(2),
                                                  dim=-1,
                                                  dim_size=max_cimg_ccl_cluster) # produce (B, B, max_cimg_ccl_cluster)
    ccl_cluster_num_pt = torch.clamp(ccl_cluster_num_pt, min=1.0)
    ccl_cluster_num_pt_unique = ccl_cluster_num_pt[original_pc_2_cimg_cpoints_unique_3[:, 0], original_pc_2_cimg_cpoints_unique_3[:, 1], original_pc_2_cimg_cpoints_unique_3[:, 3]] # produce (huge_num_3)
    ccl_cluster_overlap_ratio_mask = torch.gt(ccl_cluster_num_pt_unique, cfgs.min_img_ccl_cluster_num_pt)
    ccl_cluster_overlap_ratio = original_pc_2_cimg_cpoints_num_3 * 1.0 / ccl_cluster_num_pt_unique # produce (huge_num_3)
    ccl_cluster_overlap_ratio = ccl_cluster_overlap_ratio.masked_fill(~ccl_cluster_overlap_ratio_mask, 0.0)

    if cfgs.overlap_matrix_fuse_type == "mean":
        cluster_overlap_ratio = 0.5 * dbscan_cluster_overlap_ratio + 0.5 * ccl_cluster_overlap_ratio
    elif cfgs.overlap_matrix_fuse_type == "max":
        cluster_overlap_ratio = torch.maximum(dbscan_cluster_overlap_ratio, ccl_cluster_overlap_ratio)
    elif cfgs.overlap_matrix_fuse_type == "min":
        cluster_overlap_ratio = torch.minimum(dbscan_cluster_overlap_ratio, ccl_cluster_overlap_ratio)
    
    if cluster_overlap_ratio.shape[0] <= cfgs.cluster_topk:
        cluster_topk = cluster_overlap_ratio.shape[0]
    else:
        cluster_topk = cfgs.cluster_topk
    _, cluster_topk_indices = torch.topk(cluster_overlap_ratio, k=cluster_topk, dim=-1, largest=True, sorted=False) # produce (cfgs.cluster_topk,)

    cluster_overlap_ratio_choose = cluster_overlap_ratio[cluster_topk_indices] # produce (cfgs.cluster_topk,)
    original_pc_2_cimg_cpoints_unique_choose = original_pc_2_cimg_cpoints_unique_3[cluster_topk_indices] # produce (cfgs.cluster_topk, 6)

    # TODO: maybe the scatter_mean can be alternated by some other nn operation
    # TODO: correspondence_point_project_threshold is ignored
    pc_cluster_embeddings = torch_scatter.scatter_mean(c_pc_feats,
                                                    pc_dbscan_cluster_label.expand(-1, c_pc_feats.shape[1], -1),
                                                    dim=-1,
                                                    dim_size=max_cpoints_dbscan_cluster) # produce (B, out_dim, max_cpoints_dbscan_cluster)
    pc_pair_embeddings = pc_cluster_embeddings[original_pc_2_cimg_cpoints_unique_choose[:, 0], :, original_pc_2_cimg_cpoints_unique_choose[:, 5]] # produce (cfgs.cluster_topk, out_dim)
    img_cluster_embeddings = torch_scatter.scatter_mean(c_img_feats.flatten(2),
                                                    img_ccl_cluster_label.expand(-1, c_img_feats.shape[1], -1, -1).flatten(2),
                                                    dim=-1,
                                                    dim_size=max_cimg_ccl_cluster) # produce (B, out_dim, max_cimg_ccl_cluster)
    img_pair_embeddings = img_cluster_embeddings[original_pc_2_cimg_cpoints_unique_choose[:, 1], :, original_pc_2_cimg_cpoints_unique_choose[:, 3]] # produce (cfgs.cluster_topk, out_dim)

    if cfgs.semantic_fuse_type == 'cluster_topk_corresponded':
        original_pc_2_cimg_cpoints_unique_ccl_cluster = torch.unique(original_pc_2_cimg_cpoints_unique_choose[:, 1:4], dim=0) # produce (ccl_cluster_num, 3)
        original_pc_2_cimg_cpoints_unique_dbscan_cluster = torch.unique(torch.cat([original_pc_2_cimg_cpoints_unique_choose[:, 0:1], 
                                                                                original_pc_2_cimg_cpoints_unique_choose[:, 4:6]], dim=-1), 
                                                                                dim=0) # produce (dbscan_cluster_num, 3)
    elif cfgs.semantic_fuse_type == 'semantic_corresponded':
        original_pc_2_cimg_cpoints_unique_ccl_cluster = torch.unique(original_pc_2_cimg_cpoints_unique_3[:, 1:4], dim=0) # produce (ccl_cluster_num, 3)
        original_pc_2_cimg_cpoints_unique_dbscan_cluster = torch.unique(torch.cat([original_pc_2_cimg_cpoints_unique_3[:, 0:1], 
                                                                                original_pc_2_cimg_cpoints_unique_3[:, 4:6]], dim=-1), 
                                                                                dim=0) # produce (dbscan_cluster_num, 3)
    elif cfgs.semantic_fuse_type == 'semantic_inuse':
        original_pc_2_cimg_cpoints_unique_ccl_cluster = torch.unique(original_pc_2_cimg_cpoints_unique_2[:, 1:4], dim=0) # produce (ccl_cluster_num, 3)
        original_pc_2_cimg_cpoints_unique_dbscan_cluster = torch.unique(torch.cat([original_pc_2_cimg_cpoints_unique_2[:, 0:1], 
                                                                                original_pc_2_cimg_cpoints_unique_2[:, 4:6]], dim=-1), 
                                                                                dim=0) # produce (dbscan_cluster_num, 3)
    else:
        raise ValueError("The semantic_fuse_type is not supported")
    
    original_pc_2_cimg_cpoints_unique_img_semantic, img_semantic_inverse_indices = torch.unique(original_pc_2_cimg_cpoints_unique_ccl_cluster[:, :2], return_inverse=True, dim=0) # produce (img_semantic_num, 2)、(ccl_cluster_num,)
    original_pc_2_cimg_cpoints_unique_pc_semantic, pc_semantic_inverse_indices = torch.unique(original_pc_2_cimg_cpoints_unique_dbscan_cluster[:, :2], return_inverse=True, dim=0) # produce (pc_semantic_num, 2)、(dbscan_cluster_num,)
    original_pc_2_cimg_cpoints_unique_dbscan_cluster_embeddings = pc_cluster_embeddings[original_pc_2_cimg_cpoints_unique_dbscan_cluster[:, 0], :, original_pc_2_cimg_cpoints_unique_dbscan_cluster[:, 2]] # produce (dbscan_cluster_num, out_dim)
    original_pc_2_cimg_cpoints_unique_ccl_cluster_embeddings = img_cluster_embeddings[original_pc_2_cimg_cpoints_unique_ccl_cluster[:, 0], :, original_pc_2_cimg_cpoints_unique_ccl_cluster[:, 2]] # produce (ccl_cluster_num, out_dim)
    img_semantic_num = original_pc_2_cimg_cpoints_unique_img_semantic.shape[0]
    pc_semantic_num = original_pc_2_cimg_cpoints_unique_pc_semantic.shape[0]
    original_pc_2_cimg_cpoints_unique_img_semantic_embeddings = torch_scatter.scatter_mean(original_pc_2_cimg_cpoints_unique_ccl_cluster_embeddings, 
                                                                             dim=0, 
                                                                             index=img_semantic_inverse_indices.unsqueeze(-1).expand(-1, c_img_feats.shape[1]),
                                                                             dim_size=img_semantic_num) # produce (img_semantic_num, out_dim)
    original_pc_2_cimg_cpoints_unique_pc_semantic_embeddings = torch_scatter.scatter_mean(original_pc_2_cimg_cpoints_unique_dbscan_cluster_embeddings, 
                                                                             dim=0, 
                                                                             index=pc_semantic_inverse_indices.unsqueeze(-1).expand(-1, c_img_feats.shape[1]),
                                                                             dim_size=pc_semantic_num) # produce (pc_semantic_num, out_dim)
    original_pc_2_cimg_cpoints_unique_img_in_pc_semantic = torch.gather(input=cityscapes_label_in_semanticKitti_label[:, 0],
                                                                        dim=0,
                                                                        index=original_pc_2_cimg_cpoints_unique_img_semantic[:, 1]) # produce (img_semantic_num,)
    
    original_pc_2_cimg_cpoints_unique_img_in_pc_semantic += 1

    assert torch.equal(original_pc_2_cimg_cpoints_unique_img_in_pc_semantic, original_pc_2_cimg_cpoints_unique_img_semantic[:, 1])

    img_semantic_flag_matrix = torch.zeros((B, max_cpoints_semantic_inuse), dtype=torch.bool, device=device) # produce (B, max_cpoints_semantic_inuse)
    pc_semantic_flag_matrix = torch.zeros((B, max_cpoints_semantic_inuse), dtype=torch.bool, device=device) # produce (B, max_cpoints_semantic_inuse)
    img_semantic_embeddings_matrix = torch.zeros((B, max_cpoints_semantic_inuse, out_dim), dtype=original_pc_2_cimg_cpoints_unique_img_semantic_embeddings.dtype, device=device) # produce (B, max_cpoints_semantic_inuse, out_dim)
    pc_semantic_embeddings_matrix = torch.zeros((B, max_cpoints_semantic_inuse, out_dim), dtype=original_pc_2_cimg_cpoints_unique_pc_semantic_embeddings.dtype, device=device) # produce (B, max_cpoints_semantic_inuse, out_dim)
    img_semantic_flag_matrix[original_pc_2_cimg_cpoints_unique_img_semantic[:, 0], original_pc_2_cimg_cpoints_unique_img_in_pc_semantic] = True
    pc_semantic_flag_matrix[original_pc_2_cimg_cpoints_unique_pc_semantic[:, 0], original_pc_2_cimg_cpoints_unique_pc_semantic[:, 1]] = True
    semantic_flag_matrix = img_semantic_flag_matrix & pc_semantic_flag_matrix
    semantic_to_choose = torch.nonzero(semantic_flag_matrix, as_tuple=False) # produce (num_semantic, 2)
    img_semantic_embeddings_matrix[original_pc_2_cimg_cpoints_unique_img_semantic[:, 0], original_pc_2_cimg_cpoints_unique_img_in_pc_semantic, :] = original_pc_2_cimg_cpoints_unique_img_semantic_embeddings 
    pc_semantic_embeddings_matrix[original_pc_2_cimg_cpoints_unique_pc_semantic[:, 0], original_pc_2_cimg_cpoints_unique_pc_semantic[:, 1], :] = original_pc_2_cimg_cpoints_unique_pc_semantic_embeddings
    img_in_batch_semantic_embeddings = img_semantic_embeddings_matrix[semantic_to_choose[:, 0], semantic_to_choose[:, 1], :] # produce (num_semantic, out_dim)
    pc_in_batch_semantic_embeddings = pc_semantic_embeddings_matrix[semantic_to_choose[:, 0], semantic_to_choose[:, 1], :] # produce (num_semantic, out_dim)


    original_pc_2_cimg_cpoints_unique_all_img_semantic, all_img_semantic_inverse_indices = torch.unique(original_pc_2_cimg_cpoints_unique_img_semantic[:, 1], return_inverse=True, dim=0) # produce (num_all_img_semantic,)
    original_pc_2_cimg_cpoints_unique_all_pc_semantic, all_pc_semantic_inverse_indices = torch.unique(original_pc_2_cimg_cpoints_unique_pc_semantic[:, 1], return_inverse=True, dim=0) # produce (num_all_pc_semantic,)
    original_pc_2_cimg_cpoints_unique_all_img_semantic_embeddings = torch_scatter.scatter_mean(original_pc_2_cimg_cpoints_unique_img_semantic_embeddings,
                                                                                               dim=0,
                                                                                               index=all_img_semantic_inverse_indices.unsqueeze(-1).expand(-1, out_dim)) # produce (num_all_img_semantic, out_dim)
    original_pc_2_cimg_cpoints_unique_all_pc_semantic_embeddings = torch_scatter.scatter_mean(original_pc_2_cimg_cpoints_unique_pc_semantic_embeddings,
                                                                                               dim=0,
                                                                                               index=all_pc_semantic_inverse_indices.unsqueeze(-1).expand(-1, out_dim)) # produce (num_all_pc_semantic, out_dim)
    
    original_pc_2_cimg_cpoints_unique_all_img_in_pc_semantic = torch.gather(input=cityscapes_label_in_semanticKitti_label[:, 0],
                                                                        dim=0,
                                                                        index=original_pc_2_cimg_cpoints_unique_all_img_semantic) # produce (all_img_semantic_num,)
    
    original_pc_2_cimg_cpoints_unique_all_img_in_pc_semantic += 1

    all_img_semantic_flag_matrix = torch.zeros((max_cpoints_semantic_inuse), dtype=torch.bool, device=device) # produce (max_cpoints_semantic_inuse,)
    all_pc_semantic_flag_matrix = torch.zeros((max_cpoints_semantic_inuse), dtype=torch.bool, device=device) # produce (max_cpoints_semantic_inuse,)
    all_img_semantic_embeddings_matrix = torch.zeros((max_cpoints_semantic_inuse, out_dim), dtype=original_pc_2_cimg_cpoints_unique_all_img_semantic_embeddings.dtype, device=device) # produce (max_cpoints_semantic_inuse, out_dim)
    all_pc_semantic_embeddings_matrix = torch.zeros((max_cpoints_semantic_inuse, out_dim), dtype=original_pc_2_cimg_cpoints_unique_all_pc_semantic_embeddings.dtype, device=device) # produce (max_cpoints_semantic_inuse, out_dim)
    all_img_semantic_flag_matrix[original_pc_2_cimg_cpoints_unique_all_img_in_pc_semantic] = True
    all_pc_semantic_flag_matrix[original_pc_2_cimg_cpoints_unique_all_pc_semantic] = True
    all_semantic_flag_matrix = all_img_semantic_flag_matrix & all_pc_semantic_flag_matrix
    all_semantic_to_choose = torch.nonzero(all_semantic_flag_matrix, as_tuple=False) # produce (all_num_semantic,)
    all_semantic_to_choose = all_semantic_to_choose.squeeze(-1)
    all_img_semantic_embeddings_matrix[original_pc_2_cimg_cpoints_unique_all_img_in_pc_semantic, :] = original_pc_2_cimg_cpoints_unique_all_img_semantic_embeddings 
    all_pc_semantic_embeddings_matrix[original_pc_2_cimg_cpoints_unique_all_pc_semantic, :] = original_pc_2_cimg_cpoints_unique_all_pc_semantic_embeddings
    img_semantic_embeddings = all_img_semantic_embeddings_matrix[all_semantic_to_choose, :] # produce (all_num_semantic, out_dim)
    pc_semantic_embeddings = all_pc_semantic_embeddings_matrix[all_semantic_to_choose, :] # produce (all_num_semantic, out_dim)

    return (cluster_overlap_ratio_choose, 
            img_pair_embeddings, 
            pc_pair_embeddings, 
            img_in_batch_semantic_embeddings, 
            pc_in_batch_semantic_embeddings, 
            img_semantic_embeddings, 
            pc_semantic_embeddings)

def generate_single_correspondence_phase4_v3(c_points, 
                                             data_dict, 
                                             img_H, 
                                             img_W, 
                                             device, 
                                             c_img_feats, 
                                             c_pc_feats, 
                                             cfgs,
                                             curr_min_pc_num_pt,
                                             curr_min_img_num_pt,
                                             curr_topk):
    B, cpoints_num = c_points.shape[:2]
    _, clouds_2_cpoints_idx = keops_knn(device, data_dict["clouds"], c_points, 1)
    clouds_2_cpoints_idx = clouds_2_cpoints_idx.squeeze(-1) # Produces (B, 4096) tensor
    original_pc_2_cpoints = torch.gather(input=clouds_2_cpoints_idx.unsqueeze(1).expand(-1, B, -1),
                                        dim=-1,
                                        index=data_dict["original_pc_2_many_1"][..., 0]) # (B, B, original_points_num)
    
    img_H_mesh = torch.arange(0, img_H, device=device).type(torch.float32)
    img_W_mesh = torch.arange(0, img_W, device=device).type(torch.float32)
    img_H_mesh = img_H_mesh.unsqueeze(1).unsqueeze(2).expand(-1, img_W, -1)
    img_W_mesh = img_W_mesh.unsqueeze(0).unsqueeze(2).expand(img_H, -1, -1)
    img_mesh = torch.cat((img_H_mesh, img_W_mesh), dim=-1) # Produces (img_H, img_W, 2)
    img_mesh = img_mesh.flatten(0, 1) # Produces (img_H * img_W, 2) tensor
    cimg_H, cimg_W = c_img_feats.shape[2:]
    cimg_H_mesh = torch.arange(0, cimg_H, device=device)
    cimg_W_mesh = torch.arange(0, cimg_W, device=device)
    img_2_cimg_scale_H = img_H * 1.0 / cimg_H
    img_2_cimg_scale_W = img_W * 1.0 / cimg_W
    delta_H = img_2_cimg_scale_H / 2 - 0.5
    delta_W = img_2_cimg_scale_W / 2 - 0.5
    cimg_H_mesh = cimg_H_mesh * img_2_cimg_scale_H + delta_H
    cimg_W_mesh = cimg_W_mesh * img_2_cimg_scale_W + delta_W
    cimg_H_mesh = cimg_H_mesh.unsqueeze(1).unsqueeze(2).expand(-1, cimg_W, -1) # Produces (cimg_H, cimg_W, 1) tensor
    cimg_W_mesh = cimg_W_mesh.unsqueeze(0).unsqueeze(2).expand(cimg_H, -1, -1) # Produces (cimg_H, cimg_W, 1) tensor
    cimg_mesh = torch.cat((cimg_H_mesh, cimg_W_mesh), dim=-1) # Produces (cimg_H, cimg_W, 2) tensor
    cimg_mesh = cimg_mesh.flatten(0, 1) # Produces (cimg_H * cimg_W, 2) tensor


    # img_2_cimg_dis = torch.cdist(img_mesh, cimg_mesh) # Produces (img_H * img_W, cimg_H * cimg_W) tensor
    # _, img_2_cimg_idx = torch.topk(img_2_cimg_dis, k=1, dim=1, largest=False, sorted=False) # Produces (img_H * img_W, 1) tensor

    _, img_2_cimg_idx = keops_knn(device, img_mesh, cimg_mesh, 1)

    img_2_cimg_idx = img_2_cimg_idx.squeeze(-1).unsqueeze(0).unsqueeze(0).expand(B, B, -1) # Produces (B, B, img_H * img_W) tensor
    original_pc_2_cimg = torch.gather(input=img_2_cimg_idx,
                                    dim=-1,
                                    index=data_dict["original_pc_2_many_1"][..., 1]) # Produces (B, B, original_points_num)

    overlap_mask = torch.logical_and(data_dict["original_pc_2_many_2"][..., 0], data_dict["original_pc_2_many_2"][..., 1]) # Produces (B, B, original_points_num)
    overlap_mask = overlap_mask.type(original_pc_2_cimg.dtype)


    original_pc_2_cimg_cpoints = torch.stack((original_pc_2_cimg, original_pc_2_cpoints, overlap_mask), dim=-1) # Produces (B, B, original_points_num, 3)
    # method 1: use my unique function
    # t0 = time.time()
    # original_pc_2_cimg_cpoints_unique, original_pc_2_cimg_cpoints_num = my_unique(original_pc_2_cimg_cpoints, 
    #                                                                               torch.ones_like(original_pc_2_cimg_cpoints[..., 0])) # produce (huge_num, 5), (huge_num)
    # t1 = time.time()
    # method 2: use sparse coo tensor's coalesce function
    original_pc_2_cimg_cpoints_unique, original_pc_2_cimg_cpoints_num = my_unique_v2(original_pc_2_cimg_cpoints, 
                                                                                  torch.ones_like(original_pc_2_cimg_cpoints[..., 0]),
                                                                                  (cimg_H * cimg_W, cpoints_num, 2)) # produce (huge_num, 5), (huge_num)
    # t2 = time.time()

    original_pc_2_cimg_cpoints_mask = torch.eq(original_pc_2_cimg_cpoints_unique[..., -1], 1) # produce (huge_num)
    original_pc_2_cimg_cpoints_unique = original_pc_2_cimg_cpoints_unique[original_pc_2_cimg_cpoints_mask, :-1] # produce (huge_num_1, 4)
    original_pc_2_cimg_cpoints_num = original_pc_2_cimg_cpoints_num[original_pc_2_cimg_cpoints_mask] # produce (huge_num_1)
    original_pc_2_cpoints.masked_fill_(~data_dict["original_pc_2_many_2"][..., 1], cpoints_num - 1) # can't be reuse again
    remove_mask_num = torch.count_nonzero(~data_dict["original_pc_2_many_2"][..., 1], dim=-1) # (B, B)
    cpoints_num_pt = torch_scatter.scatter_sum(torch.ones_like(original_pc_2_cpoints, dtype=torch.int32),
                                            original_pc_2_cpoints,
                                            dim=-1,
                                            dim_size=cpoints_num) # produce (B, B, cpoints_num)
    cpoints_num_pt[..., -1] -= remove_mask_num # produce (B, B, cpoints_num)
    cpoints_num_pt = torch.clamp(cpoints_num_pt, min=1)
    original_pc_2_cimg.masked_fill_(~data_dict["original_pc_2_many_2"][..., 0], cimg_H * cimg_W - 1) # can't be reuse again
    non_qualified_mask_num = torch.count_nonzero(~data_dict["original_pc_2_many_2"][..., 0], dim=-1) # produce (B, B)
    cimg_num_pt = torch_scatter.scatter_sum(torch.ones_like(original_pc_2_cimg, dtype=torch.int32),
                                            original_pc_2_cimg,
                                            dim=-1,
                                            dim_size=cimg_H * cimg_W) # produce (B, B, cimg_H * cimg_W)
    cimg_num_pt[..., -1] -= non_qualified_mask_num
    cimg_num_pt = torch.clamp(cimg_num_pt, min=1)
    cpoints_num_pt_unique = cpoints_num_pt[original_pc_2_cimg_cpoints_unique[:, 0], original_pc_2_cimg_cpoints_unique[:, 1], original_pc_2_cimg_cpoints_unique[:, 3]] # produce (huge_num_1)
    cimg_num_pt_unique = cimg_num_pt[original_pc_2_cimg_cpoints_unique[:, 0], original_pc_2_cimg_cpoints_unique[:, 1], original_pc_2_cimg_cpoints_unique[:, 2]] # produce (huge_num_1)
    pc_overlap_ratio_mask = torch.gt(cpoints_num_pt_unique, curr_min_pc_num_pt)
    pc_overlap_ratio = original_pc_2_cimg_cpoints_num * 1.0 / cpoints_num_pt_unique # produce (huge_num_1)
    pc_overlap_ratio = pc_overlap_ratio.masked_fill(~pc_overlap_ratio_mask, 0.0)
    img_overlap_ratio_mask = torch.gt(cimg_num_pt_unique, curr_min_img_num_pt)
    img_overlap_ratio = original_pc_2_cimg_cpoints_num * 1.0 / cimg_num_pt_unique # produce (huge_num_1)
    img_overlap_ratio = img_overlap_ratio.masked_fill(~img_overlap_ratio_mask, 0.0)
    if cfgs.phase4_overlap_matrix_fuse_type == "mean":
        overlap_ratio = 0.5 * pc_overlap_ratio + 0.5 * img_overlap_ratio
    elif cfgs.phase4_overlap_matrix_fuse_type == "max":
        overlap_ratio = torch.maximum(pc_overlap_ratio, img_overlap_ratio)
    elif cfgs.phase4_overlap_matrix_fuse_type == "min":
        overlap_ratio = torch.minimum(pc_overlap_ratio, img_overlap_ratio)
    _, topk_indices = torch.topk(overlap_ratio, k=curr_topk, dim=-1, largest=True, sorted=False) # produce (cfgs.phase4_topk,)
    original_pc_2_cimg_cpoints_unique_choose = original_pc_2_cimg_cpoints_unique[topk_indices] # produce (cfgs.phase4_topk, 4)
    img_pair_indices = torch.stack((original_pc_2_cimg_cpoints_unique_choose[:, 1], original_pc_2_cimg_cpoints_unique_choose[:, 2]), dim=-1) # produce (cfgs.phase4_topk, 2)
    img_pair_indices = torch.unique(img_pair_indices, dim=0, sorted=True) # produce (num_img_1, 2)
    pc_pair_indices = torch.stack((original_pc_2_cimg_cpoints_unique_choose[:, 0], original_pc_2_cimg_cpoints_unique_choose[:, 3]), dim=-1) # produce (cfgs.phase4_topk, 2)
    pc_pair_indices = torch.unique(pc_pair_indices, dim=0, sorted=True) # produce (num_pc_1, 2)
    img_pair_embeddings = c_img_feats.flatten(start_dim=2)[img_pair_indices[:, 0], :, img_pair_indices[:, 1]] # produce (num_img_1, out_dim)
    pc_pair_embeddings = c_pc_feats[pc_pair_indices[:, 0], :, pc_pair_indices[:, 1]] # produce (num_pc_1, out_dim)

    original_pc_2_cimg_cpoints_unique_v2_1 = original_pc_2_cimg_cpoints_unique[:, 0] * cpoints_num + original_pc_2_cimg_cpoints_unique[:, 3]
    original_pc_2_cimg_cpoints_unique_v2_2 = original_pc_2_cimg_cpoints_unique[:, 1] * cimg_H * cimg_W + original_pc_2_cimg_cpoints_unique[:, 2]
    original_pc_2_cimg_cpoints_unique_v2 = torch.stack((original_pc_2_cimg_cpoints_unique_v2_1, original_pc_2_cimg_cpoints_unique_v2_2), dim=-1)
    original_pc_2_cimg_cpoints_overlap_ratio_sparse_v2 = torch_sparse.SparseTensor(row=original_pc_2_cimg_cpoints_unique_v2[:, 0], col=original_pc_2_cimg_cpoints_unique_v2[:, 1], value=overlap_ratio, sparse_sizes=(B * cpoints_num, B * cimg_H * cimg_W))
    pc_pair_indices_v1 = pc_pair_indices[:, 0] * cpoints_num + pc_pair_indices[:, 1]
    img_pair_indices_v1 = img_pair_indices[:, 0] * cimg_H * cimg_W + img_pair_indices[:, 1]
    overlap_ratio_matrix_inuse_temp1 = torch_sparse.index_select(original_pc_2_cimg_cpoints_overlap_ratio_sparse_v2, idx=pc_pair_indices_v1, dim=0)
    overlap_ratio_matrix_inuse_temp2 = torch_sparse.index_select(overlap_ratio_matrix_inuse_temp1, idx=img_pair_indices_v1, dim=1)
    overlap_ratio_matrix_inuse_temp3 = overlap_ratio_matrix_inuse_temp2.to_dense()
    overlap_ratio_matrix_inuse = overlap_ratio_matrix_inuse_temp3

    return overlap_ratio_matrix_inuse, img_pair_embeddings, pc_pair_embeddings


# def generate_multi_correspondence_phase4_v2(data_dict, device, img_feats_list, pc_feats_list, pc_coords_list, img_H, img_W, cfgs):
#     B = data_dict["clouds"].shape[0]
#     fpoints_num = pc_coords_list[0].shape[1]
#     f_points = pc_coords_list[0]
#     clouds_2_fpoints_dis = torch.cdist(data_dict["clouds"], f_points) # Produces (B, 4096, f_points_num) tensor
#     _, clouds_2_fpoints_idx = torch.topk(clouds_2_fpoints_dis, k=1, dim=2, largest=False, sorted=False) # Produces (B, 4096, 1) tensor
#     clouds_2_fpoints_idx = clouds_2_fpoints_idx.squeeze(-1) # Produces (B, 4096) tensor
#     original_pc_2_fpoints = torch.gather(input=clouds_2_fpoints_idx.unsqueeze(1).expand(-1, B, -1),
#                                         dim=-1,
#                                         index=data_dict["original_pc_2_many_1"][..., 0]) # (B, B, original_points_num)
    
#     img_H_mesh = torch.arange(0, img_H, device=device).type(torch.float32)
#     img_W_mesh = torch.arange(0, img_W, device=device).type(torch.float32)
#     img_H_mesh = img_H_mesh.unsqueeze(1).unsqueeze(2).expand(-1, img_W, -1)
#     img_W_mesh = img_W_mesh.unsqueeze(0).unsqueeze(2).expand(img_H, -1, -1)
#     img_mesh = torch.cat((img_H_mesh, img_W_mesh), dim=-1) # Produces (img_H, img_W, 2)
#     img_mesh = img_mesh.flatten(0, 1) # Produces (img_H * img_W, 2) tensor
#     f_img_feats = img_feats_list[0]
#     fimg_H, fimg_W = f_img_feats.shape[2:]
#     fimg_H_mesh = torch.arange(0, fimg_H, device=device)
#     fimg_W_mesh = torch.arange(0, fimg_W, device=device)
#     img_2_fimg_scale_H = img_H * 1.0 / fimg_H
#     img_2_fimg_scale_W = img_W * 1.0 / fimg_W
#     delta_H = img_2_fimg_scale_H / 2 - 0.5
#     delta_W = img_2_fimg_scale_W / 2 - 0.5
#     fimg_H_mesh = fimg_H_mesh * img_2_fimg_scale_H + delta_H
#     fimg_W_mesh = fimg_W_mesh * img_2_fimg_scale_W + delta_W
#     fimg_H_mesh = fimg_H_mesh.unsqueeze(1).unsqueeze(2).expand(-1, fimg_W, -1) # Produces (fimg_H, fimg_W, 1) tensor
#     fimg_W_mesh = fimg_W_mesh.unsqueeze(0).unsqueeze(2).expand(fimg_H, -1, -1) # Produces (fimg_H, fimg_W, 1) tensor
#     fimg_mesh = torch.cat((fimg_H_mesh, fimg_W_mesh), dim=-1) # Produces (fimg_H, fimg_W, 2) tensor
#     fimg_mesh = fimg_mesh.flatten(0, 1) # Produces (fimg_H * fimg_W, 2) tensor
#     img_2_fimg_dis = torch.cdist(img_mesh, fimg_mesh) # Produces (img_H * img_W, fimg_H * fimg_W) tensor
#     _, img_2_fimg_idx = torch.topk(img_2_fimg_dis, k=1, dim=1, largest=False, sorted=False) # Produces (img_H * img_W, 1) tensor
#     img_2_fimg_idx = img_2_fimg_idx.squeeze(-1).unsqueeze(0).unsqueeze(0).expand(B, B, -1) # Produces (B, B, img_H * img_W) tensor
#     original_pc_2_fimg = torch.gather(input=img_2_fimg_idx,
#                                     dim=-1,
#                                     index=data_dict["original_pc_2_many_1"][..., 1]) # Produces (B, B, original_points_num)

#     overlap_mask = torch.logical_and(data_dict["original_pc_2_many_2"][..., 0], data_dict["original_pc_2_many_2"][..., 1]) # Produces (B, B, original_points_num)
#     overlap_mask = overlap_mask.type(original_pc_2_fimg.dtype)
#     original_pc_2_fimg_fpoints = torch.stack((original_pc_2_fimg, original_pc_2_fpoints, overlap_mask), dim=-1) # Produces (B, B, original_points_num, 3)
#     original_pc_2_fimg_fpoints_unique, original_pc_2_fimg_fpoints_num = my_unique_v2(original_pc_2_fimg_fpoints, 
#                                                                                   torch.ones_like(original_pc_2_fimg_fpoints[..., 0]),
#                                                                                   (fimg_H * fimg_W, fpoints_num, 2)) # produce (huge_num, 5), (huge_num)
#     original_pc_2_fimg_fpoints_mask = torch.eq(original_pc_2_fimg_fpoints_unique[..., -1], 1) # produce (huge_num)
#     original_pc_2_fimg_fpoints_unique = original_pc_2_fimg_fpoints_unique[original_pc_2_fimg_fpoints_mask, :-1] # produce (huge_num_1, 4)
#     original_pc_2_fimg_fpoints_num = original_pc_2_fimg_fpoints_num[original_pc_2_fimg_fpoints_mask] # produce (huge_num_1)
#     fimg_2_many_img_idx_list = []
#     for i in range(len(img_feats_list) - 1):
#         curr_img_H, curr_img_W = img_feats_list[i+1].shape[2:]
#         curr_img_H_mesh = torch.arange(0, curr_img_H, device=device)
#         curr_img_W_mesh = torch.arange(0, curr_img_W, device=device)
#         img_2_curr_img_scale_H = img_H * 1.0 / curr_img_H
#         img_2_curr_img_scale_W = img_W * 1.0 / curr_img_W
#         delta_H = img_2_curr_img_scale_H / 2 - 0.5
#         delta_W = img_2_curr_img_scale_W / 2 - 0.5
#         curr_img_H_mesh = curr_img_H_mesh * img_2_curr_img_scale_H + delta_H
#         curr_img_W_mesh = curr_img_W_mesh * img_2_curr_img_scale_W + delta_W
#         curr_img_H_mesh = curr_img_H_mesh.unsqueeze(1).unsqueeze(2).expand(-1, curr_img_W, -1) # Produces (curr_img_H, curr_img_W, 1) tensor
#         curr_img_W_mesh = curr_img_W_mesh.unsqueeze(0).unsqueeze(2).expand(curr_img_H, -1, -1) # Produces (curr_img_H, curr_img_W, 1) tensor
#         curr_img_mesh = torch.cat((curr_img_H_mesh, curr_img_W_mesh), dim=-1) # Produces (curr_img_H, curr_img_W, 2) tensor
#         curr_img_mesh = curr_img_mesh.flatten(0, 1) # Produces (curr_img_H * curr_img_W, 2) tensor
#         fimg_2_curr_img_dis = torch.cdist(fimg_mesh, curr_img_mesh) # Produces (fimg_H * fimg_W, curr_img_H * curr_img_W) tensor
#         _, fimg_2_curr_img_idx = torch.topk(fimg_2_curr_img_dis, k=1, dim=1, largest=False, sorted=False) # Produces (fimg_H * fimg_W, 1) tensor
#         fimg_2_curr_img_idx = fimg_2_curr_img_idx.squeeze(-1).unsqueeze(0).unsqueeze(0).expand(B, B, -1) # Produces (B, B, fimg_H * fimg_W) tensor
#         fimg_2_many_img_idx_list.append(fimg_2_curr_img_idx)
    
#     fpoints_2_many_points_list = []
#     for i in range(len(pc_feats_list) - 1):
#         fpoints_2_curr_points_dis = torch.cdist(f_points, pc_coords_list[i + 1]) # produces (B, fpoints_num, curr_points_num)
#         _, fpoints_2_curr_points_idx = torch.topk(fpoints_2_curr_points_dis, k=1, dim=2, largest=False, sorted=False) # Produces (B, fpoints_num, 1) tensor
#         fpoints_2_many_points_list.append(fpoints_2_curr_points_idx.squeeze(-1).squeeze(1).expand(-1, B, -1))


#     fimg_2_fpoints_map = original_pc_2_fimg * fpoints_num + original_pc_2_fpoints
#     overlap_mask = torch.logical_and(data_dict["original_pc_2_many_2"][..., 0], data_dict["original_pc_2_many_2"][..., 1])
#     fimg_2_fpoints_map.masked_fill_(~overlap_mask, fpoints_num * fimg_H * fimg_W - 1)
#     overlap_mask_num = torch.count_nonzero(~overlap_mask, dim=-1) # (B, B)
#     fimg_2_fpoints_num_pt = torch_scatter.scatter_sum(torch.ones_like(fimg_2_fpoints_map, dtype=torch.int32), 
#                                                     fimg_2_fpoints_map,
#                                                     dim=-1,
#                                                     dim_size=fimg_H * fimg_W * fpoints_num) # produces (B, B, fimg_H * fimg_W * fpoints_num)
#     fimg_2_fpoints_num_pt[..., -1] -= overlap_mask_num
#     fimg_2_fpoints_num_pt = fimg_2_fpoints_num_pt.reshape(B, B, fimg_H * fimg_W, fpoints_num)

#     fimg_2_many_points_num_pt_list = [fimg_2_fpoints_num_pt]
#     for i in range(len(pc_coords_list) - 1):
#         fpoints_2_curr_points_dis = torch.cdist(f_points, pc_coords_list[i + 1]) # produces (B, fpoints_num, curr_points_num)
#         _, fpoints_2_curr_points_idx = torch.topk(fpoints_2_curr_points_dis, k=1, dim=2, largest=False, sorted=False) # Produces (B, fpoints_num, 1) tensor
#         fimg_2_curr_points_num_pt = torch_scatter.scatter_sum(fimg_2_fpoints_num_pt, 
#                                                             fpoints_2_curr_points_idx.squeeze(-1).unsqueeze(1), 
#                                                             dim=-1, 
#                                                             dim_size=pc_coords_list[i + 1].shape[1]) # produces (B, B, fimg_H * fimg_W, curr_points_num)
#         fimg_2_many_points_num_pt_list.append(fimg_2_curr_points_num_pt)
    
#     fimg_2_many_img_idx_list = []
#     for i in range(len(img_feats_list) - 1):
#         curr_img_H, curr_img_W = img_feats_list[i+1].shape[2:]
#         curr_img_H_mesh = torch.arange(0, curr_img_H, device=device)
#         curr_img_W_mesh = torch.arange(0, curr_img_W, device=device)
#         img_2_curr_img_scale_H = img_H * 1.0 / curr_img_H
#         img_2_curr_img_scale_W = img_W * 1.0 / curr_img_W
#         delta_H = img_2_curr_img_scale_H / 2 - 0.5
#         delta_W = img_2_curr_img_scale_W / 2 - 0.5
#         curr_img_H_mesh = curr_img_H_mesh * img_2_curr_img_scale_H + delta_H
#         curr_img_W_mesh = curr_img_W_mesh * img_2_curr_img_scale_W + delta_W
#         curr_img_H_mesh = curr_img_H_mesh.unsqueeze(1).unsqueeze(2).expand(-1, curr_img_W, -1) # Produces (curr_img_H, curr_img_W, 1) tensor
#         curr_img_W_mesh = curr_img_W_mesh.unsqueeze(0).unsqueeze(2).expand(curr_img_H, -1, -1) # Produces (curr_img_H, curr_img_W, 1) tensor
#         curr_img_mesh = torch.cat((curr_img_H_mesh, curr_img_W_mesh), dim=-1) # Produces (curr_img_H, curr_img_W, 2) tensor
#         curr_img_mesh = curr_img_mesh.flatten(0, 1) # Produces (curr_img_H * curr_img_W, 2) tensor
#         fimg_2_curr_img_dis = torch.cdist(fimg_mesh, curr_img_mesh) # Produces (fimg_H * fimg_W, curr_img_H * curr_img_W) tensor
#         _, fimg_2_curr_img_idx = torch.topk(fimg_2_curr_img_dis, k=1, dim=1, largest=False, sorted=False) # Produces (fimg_H * fimg_W, 1) tensor
#         fimg_2_curr_img_idx = fimg_2_curr_img_idx.squeeze(-1).unsqueeze(0).unsqueeze(0).expand(B, B, -1) # Produces (B, B, fimg_H * fimg_W) tensor
#         fimg_2_many_img_idx_list.append(fimg_2_curr_img_idx)
    
#     fpoints_2_many_points_list = []
#     for i in range(len(pc_feats_list) - 1):
#         fpoints_2_curr_points_dis = torch.cdist(f_points, pc_coords_list[i + 1]) # produces (B, fpoints_num, curr_points_num)
#         _, fpoints_2_curr_points_idx = torch.topk(fpoints_2_curr_points_dis, k=1, dim=2, largest=False, sorted=False) # Produces (B, fpoints_num, 1) tensor
#         fpoints_2_many_points_list.append(fpoints_2_curr_points_idx.squeeze(-1).squeeze(1).expand(-1, B, -1))
    
#     many_img_2_many_points_num_pt_list = [fimg_2_many_points_num_pt_list]
#     for i in range(len(img_feats_list) - 1):
#         curr_img_2_many_points_num_pt_list = []
#         for j in range(len(pc_coords_list)):
#             curr_img_2_curr_points_num_pt = torch_scatter.scatter_sum(fimg_2_many_points_num_pt_list[j], 
#                                                                     fimg_2_many_img_idx_list[i], 
#                                                                     dim=-2, 
#                                                                     dim_size=img_feats_list[i+1].shape[2] * img_feats_list[i+1].shape[3]) # produces (B, B, curr_img_H * curr_img_W, curr_points_num)
#             curr_img_2_many_points_num_pt_list.append(curr_img_2_curr_points_num_pt)
#         many_img_2_many_points_num_pt_list.append(curr_img_2_many_points_num_pt_list)
    
#     original_pc_2_fpoints.masked_fill_(~data_dict["original_pc_2_many_2"][..., 1], fpoints_num - 1) # can't be reuse again
#     remove_mask_num = torch.count_nonzero(~data_dict["original_pc_2_many_2"][..., 1], dim=-1) # (B, B)
#     fpoints_num_pt = torch_scatter.scatter_sum(torch.ones_like(original_pc_2_fpoints, dtype=torch.int32),
#                                             original_pc_2_fpoints,
#                                             dim=-1,
#                                             dim_size=fpoints_num) # produce (B, B, fpoints_num)
#     fpoints_num_pt[..., -1] -= remove_mask_num # produce (B, B, fpoints_num)
#     original_pc_2_fimg.masked_fill_(~data_dict["original_pc_2_many_2"][..., 0], fimg_H * fimg_W - 1) # can't be reuse again
#     non_qualified_mask_num = torch.count_nonzero(~data_dict["original_pc_2_many_2"][..., 0], dim=-1) # produce (B, B)
#     fimg_num_pt = torch_scatter.scatter_sum(torch.ones_like(original_pc_2_fimg, dtype=torch.int32),
#                                             original_pc_2_fimg,
#                                             dim=-1,
#                                             dim_size=fimg_H * fimg_W) # produce (B, B, fimg_H * fimg_W)
#     fimg_num_pt[..., -1] -= non_qualified_mask_num

#     img_num_pt_list = [fimg_num_pt]
#     for i in range(len(img_feats_list) - 1):
#         curr_img_num_pt = torch_scatter.scatter_sum(fimg_num_pt, 
#                                                     fimg_2_many_img_idx_list[i], 
#                                                     dim=-1, 
#                                                     dim_size=img_feats_list[i+1].shape[2] * img_feats_list[i+1].shape[3])
#         img_num_pt_list.append(curr_img_num_pt)
#     points_num_pt_list = [fpoints_num_pt]
#     for i in range(len(pc_coords_list) - 1):
#         curr_points_num_pt = torch_scatter.scatter_sum(fpoints_num_pt, 
#                                                     fpoints_2_many_points_list[i], 
#                                                     dim=-1, 
#                                                     dim_size=pc_coords_list[i + 1].shape[1])
#         points_num_pt_list.append(curr_points_num_pt)
    
#     # cfgs.phase4_correspondence_pair example: [[0, 0], [0, 1], [1, 0], [1, 1]]
#     # the first index is the index of img_feats_list, the second index is the index of pc_feats_list
#     phase4_correspondence_pair = torch.tensor(cfgs.phase4_correspondence_pair, dtype=torch.int32, device=device)
#     img_level_indices = torch.unique(phase4_correspondence_pair[..., 0], sorted=True)
#     points_level_indices = torch.unique(phase4_correspondence_pair[..., 1], sorted=True)
#     img_pair_embeddings_list = []
#     img_pair_indices_list = []
#     img_pair_list_reverse = {}
#     for img_level_indice in img_level_indices:
#         curr_img_pair_indices = torch.nonzero(torch.gt(img_num_pt_list[img_level_indices], 
#                                                        cfgs.phase4_min_img_num_pt[img_level_indices]), 
#                                                 as_tuple=False) # produce (curr_num_img, 3)
#         curr_img_pair_indices = torch.unique(curr_img_pair_indices[:, 1:], dim=0, sorted=True) # produce (curr_num_img_chose, 2)
#         curr_img_pair_embeddings = img_feats_list[img_level_indice].flatten(start_dim=2)[curr_img_pair_indices[:, 0], :, curr_img_pair_indices[:, 1]] # produce (curr_num_img_chose, out_dim)
#         img_pair_embeddings_list.append(curr_img_pair_embeddings)
#         img_pair_indices_list.append(curr_img_pair_indices)
#         img_pair_list_reverse[img_level_indice] = len(img_pair_embeddings_list) - 1
    
#     points_pair_embeddings_list = []
#     points_pair_indices_list = []
#     points_pair_list_reverse = {}
#     for points_level_indice in points_level_indices:
#         curr_points_overlap_num_pt = torch.sum(many_img_2_many_points_num_pt_list[0][points_level_indice], 
#                                                dim=2, 
#                                                keepdim=False) # produce (B, B, curr_points_num)
#         curr_points_pair_indices = torch.nonzero(torch.gt(curr_points_overlap_num_pt, 
#                                                           cfgs.phase4_min_pc_overlap_num_pt[points_level_indice]), 
#                                                           as_tuple=False) # produce (curr_num_pc_chose, 3)
#         curr_points_pair_indices = torch.unique(curr_points_pair_indices[:, ::2], 
#                                                 dim=0, 
#                                                 sorted=True) # produce (curr_num_pc_chose, 2)
#         curr_points_pair_embeddings = pc_feats_list[points_level_indice][curr_points_pair_indices[:, 0], :, curr_points_pair_indices[:, 1]] # produce (curr_num_pc_chose, out_dim)
#         points_pair_embeddings_list.append(curr_points_pair_embeddings)
#         points_pair_indices_list.append(curr_points_pair_indices)
#         points_pair_list_reverse[points_level_indice] = len(points_pair_embeddings_list) - 1

#     if cfgs.phase4_multi_level_overlap_ratio_type == "corresponds_between_all":
#         # 1、all the overlap_matrix are merged into one matrix
#         many_img_2_many_points_overlap_ratio_matrix_list = []
#         for img_idx in range(len(img_level_indices)):
#             curr_img_2_many_points_overlap_ratio_matrix_list = []
#             for points_idx in range(len(points_level_indices)):
#                 img_level_indice = img_level_indices[img_idx]
#                 points_level_indice = points_level_indices[points_idx]
#                 curr_pc_overlap_ratio_matrix = many_img_2_many_points_num_pt_list[img_level_indice][points_level_indice] * 1.0 / torch.clamp(points_num_pt_list[points_level_indice].unsqueeze(2), min=1)
#                 curr_img_overlap_ratio_matrix = many_img_2_many_points_num_pt_list[img_level_indice][points_level_indice] * 1.0 / torch.clamp(img_num_pt_list[img_level_indice].unsqueeze(3), min=1)
#                 curr_overlap_ratio_matrix = curr_pc_overlap_ratio_matrix * 0.5 + curr_img_overlap_ratio_matrix * 0.5
#                 curr_overlap_ratio_matrix_inuse = curr_overlap_ratio_matrix[points_pair_indices_list[points_idx][:, 0], :, :, points_pair_indices_list[points_idx][:, 1]][:, img_pair_indices_list[img_idx][:, 0], img_pair_indices_list[img_idx][:, 1]] # (curr_num_pc_chose, curr_num_img_chose)
#                 curr_img_2_many_points_overlap_ratio_matrix_list.append(curr_overlap_ratio_matrix_inuse)
#             curr_img_2_many_points_overlap_ratio_matrix = torch.cat(curr_img_2_many_points_overlap_ratio_matrix_list, dim=0) # (num_pc_chose, curr_num_img_chose)
#             many_img_2_many_points_overlap_ratio_matrix_list.append(curr_img_2_many_points_overlap_ratio_matrix)
#         many_img_2_many_points_overlap_ratio_matrix = torch.cat(many_img_2_many_points_overlap_ratio_matrix_list, dim=1) # (num_pc_chose, num_img_chose)
#         img_pair_embeddings = torch.cat(img_pair_embeddings_list, dim=0) # (num_img_chose, out_dim)
#         points_pair_embeddings = torch.cat(points_pair_embeddings_list, dim=0) # (num_pc_chose, out_dim)
#         return many_img_2_many_points_overlap_ratio_matrix, img_pair_embeddings, points_pair_embeddings
#     elif cfgs.phase4_multi_level_overlap_ratio_type == "corresponds_between_pair":
#         # 2、use the num of cfgs.phase4_correspondence_pair of circle losses, so as many matrixs and embeddings
#         overlap_ratio_matrix_list = []
#         image_pair_embeddings_output_list = []
#         points_pair_embeddings_output_list = []
#         for curr_pair in cfgs.phase4_correspondence_pair:
#             curr_img_level_indice = curr_pair[0]
#             curr_points_level_indice = curr_pair[1]
#             curr_img_pair_embeddings = img_pair_embeddings_list[img_pair_list_reverse[curr_img_level_indice]]
#             curr_points_pair_embeddings = points_pair_embeddings_list[points_pair_list_reverse[curr_points_level_indice]]
#             curr_img_pair_indices = img_pair_indices_list[img_pair_list_reverse[curr_img_level_indice]]
#             curr_points_pair_indices = points_pair_indices_list[points_pair_list_reverse[curr_points_level_indice]]
#             curr_img_2_curr_points_num_pt = many_img_2_many_points_num_pt_list[curr_img_level_indice][curr_points_level_indice]
#             img_overlap_ratio_matrix = curr_img_2_curr_points_num_pt * 1.0 / torch.clamp(img_num_pt_list[curr_img_level_indice].unsqueeze(3), min=1)
#             points_overlap_ratio_matrix = curr_img_2_curr_points_num_pt * 1.0 / torch.clamp(points_num_pt_list[curr_points_level_indice].unsqueeze(2), min=1)
#             curr_overlap_ratio_matrix = points_overlap_ratio_matrix * 0.5 + img_overlap_ratio_matrix * 0.5
#             curr_overlap_ratio_matrix_inuse = curr_overlap_ratio_matrix[curr_points_pair_indices[:, 0], :, :, curr_points_pair_indices[:, 1]][:, curr_img_pair_indices[:, 0], curr_img_pair_indices[:, 1]] # (num_pc_chose, num_img_chose)
#             overlap_ratio_matrix_list.append(curr_overlap_ratio_matrix_inuse)
#             image_pair_embeddings_output_list.append(curr_img_pair_embeddings)
#             points_pair_embeddings_output_list.append(curr_points_pair_embeddings)
#         return overlap_ratio_matrix_list, image_pair_embeddings_output_list, points_pair_embeddings_output_list