import pickle
import os
import numpy as np
import tqdm
import argparse
import torch
import torch.nn.functional as F
import MinkowskiEngine as ME
import time
from datasets import make_dataloader
DEBUG = False
from utils import process_labels


def evaluate_5(model, device, cfgs, dataset_path, val_eval):
    start_epoch = 0
    dataloader, dataset = make_dataloader(cfgs.dataloader_cfgs, dataset_path, start_epoch, val_eval)
    temp = evaluate_dataset_kitti_odometry(model, device, cfgs, dataloader, dataset, False)
    temp_out = {}
    temp_out['kitti_odometry'] = temp
    return temp_out

def evaluate_dataset_kitti_odometry(model, device, cfgs, dataloader, dataset, silent=True):
    # Run evaluation on a single dataset
    recall = np.zeros(25)
    count = 0
    dist = []
    one_percent_recall = []

    database_embeddings = {}
    query_embeddings = {}

    model.eval()
    model.to(device)

    DEBUG_CTR = 0
    t1 = time.time()
    for data in tqdm.tqdm(dataloader):
        #-----------------------------------data-----------------------------------
        curr_seq_ID = data['labels'][0][0]
        if curr_seq_ID not in database_embeddings.keys():
            query_embeddings[curr_seq_ID] = []
            database_embeddings[curr_seq_ID] = []
        t2 = time.time()
        data_input = {}
        if "imgnet_cfgs" in cfgs.model_cfgs.keys() or ("cmvpr_cfgs" in cfgs.model_cfgs.keys() and "image_encoder_type" in cfgs.model_cfgs.cmvpr_cfgs) or ("cmvpr2_cfgs" in cfgs.model_cfgs.keys() and "image_encoder_type" in cfgs.model_cfgs.cmvpr2_cfgs):
            if 'images' in data.keys():
                data_input['images'] = data['images'].to(device)
        if "rendernet_cfgs" in cfgs.model_cfgs.keys() or ("cmvpr2_cfgs" in cfgs.model_cfgs.keys() and "render_encoder_type" in cfgs.model_cfgs.cmvpr2_cfgs):
            if 'render_imgs' in data.keys():
                data_input['render_imgs'] = data['render_imgs'].to(device)
            elif 'range_imgs' in data.keys():
                data_input['render_imgs'] = data['range_imgs'].to(device)
            else:
                pass
        if "imagebevnet_cfgs" in cfgs.model_cfgs.keys() or ("cmvpr2_cfgs" in cfgs.model_cfgs.keys() and "image_bev_encoder_type" in cfgs.model_cfgs.cmvpr2_cfgs):
            data_input['image_bevs'] = data['image_bevs'].to(device)
        if "pcbevnet_cfgs" in cfgs.model_cfgs.keys() or ("cmvpr2_cfgs" in cfgs.model_cfgs.keys() and "pc_bev_encoder_type" in cfgs.model_cfgs.cmvpr2_cfgs):
            data_input['pc_bevs'] = data['pc_bevs'].to(device)
        
        if "pcnet_cfgs" in cfgs.model_cfgs.keys() or ("cmvpr_cfgs" in cfgs.model_cfgs.keys() and "pc_encoder_type" in cfgs.model_cfgs.cmvpr_cfgs):
            another_input_list = [pc.type(torch.float32) for pc in data['clouds']]
            if 'pcnet_cfgs' in cfgs.model_cfgs.keys():
                pc_backbone_type = cfgs.model_cfgs.pcnet_cfgs.backbone_type
            else:
                pc_backbone_type = cfgs.model_cfgs.cmvpr_cfgs.pc_encoder_type
            if pc_backbone_type=='FPT':
                if 'pcnet_cfgs' in cfgs.model_cfgs.keys():
                    radius_max_raw = cfgs.model_cfgs.pcnet_cfgs.backbone_config.radius_max_raw
                    voxel_size = cfgs.model_cfgs.pcnet_cfgs.backbone_config.voxel_size
                else:
                    radius_max_raw = cfgs.model_cfgs.cmvpr_cfgs.pc_encoder_cfgs.radius_max_raw
                    voxel_size = cfgs.model_cfgs.cmvpr_cfgs.pc_encoder_cfgs.voxel_size
                coordinates_input = [coords * radius_max_raw / voxel_size for coords in another_input_list]
                features_input = [torch.ones(coords.shape, device=coords.device, dtype=torch.float32) for coords in another_input_list] # let the z coordinate be the feature
                # features_input = [coords for coords in another_input_list]
                coords, feats = ME.utils.sparse_collate(
                        coordinates_input,
                        features_input,
                        dtype=torch.float32
                    )
                data_input['clouds'] = ME.TensorField(features=feats,
                                    coordinates=coords,
                                    quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                                    device=device)
            elif pc_backbone_type=='MinkUNet':
                if 'pcnet_cfgs' in cfgs.model_cfgs.keys():
                    voxel_size = cfgs.model_cfgs.pcnet_cfgs.backbone_config.voxel_size
                else:
                    voxel_size = cfgs.model_cfgs.cmvpr_cfgs.pc_encoder_cfgs.voxel_size
                coordinates_input = torch.stack(another_input_list, dim=0)
                x, y, z = coordinates_input[..., 0], coordinates_input[..., 1], coordinates_input[..., 2]
                rho = torch.sqrt(x ** 2 + y ** 2) / voxel_size
                phi = torch.atan2(y, x) * 180 / np.pi  # corresponds to a split each 1°
                z = z / voxel_size
                coordinates_input = torch.stack((rho, phi, z), dim=-1) # (B, N, 3)
                coordinates_input = [coordinates_input[i] for i in range(coordinates_input.shape[0])]
                features_input = [coords[:, 2:] for coords in another_input_list] # let the z coordinate be the feature
                coords, feats = ME.utils.sparse_collate(
                        coordinates_input,
                        features_input,
                        dtype=torch.float32
                    )
                data_input['clouds'] = ME.TensorField(features=feats,
                                    coordinates=coords,
                                    quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                                    device=device).sparse(quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)
            else:
                data_input['clouds'] = torch.stack(another_input_list, dim=0).to(device)
        x = model(data_input)
        if 'embeddings1' in x.keys() and 'embeddings2' in x.keys():
            query_embedding = x['embeddings1'] # "image"
            database_embedding = x['embeddings2'] # "cloud"
        elif 'embeddings' in x.keys():
            query_embedding = x['embeddings']
            database_embedding = x['embeddings']
        else:
            raise ValueError('no embeddings or embeddings1 or embeddings2 in the output')

        if 'evaluate_normalize' in cfgs.keys() and cfgs.evaluate_normalize:
            print('embeddings normalized when evaluating')
            query_embedding = F.normalize(query_embedding, p=2, dim=1)
            database_embedding = F.normalize(database_embedding, p=2, dim=1)
        
        query_embeddings[curr_seq_ID].append(query_embedding)
        database_embeddings[curr_seq_ID].append(database_embedding)

        DEBUG_CTR += 1

        if DEBUG and DEBUG_CTR == 50:
            break
        

    # torch.cuda.empty_cache()
    distmat = {}
    for key in query_embeddings.keys():
        query_embeddings[key] = torch.cat(query_embeddings[key], dim=0) # [num_query, dim]
        database_embeddings[key] = torch.cat(database_embeddings[key], dim=0) # [num_database, dim]
        distmat[key] = torch.cdist(query_embeddings[key].unsqueeze(0), database_embeddings[key].unsqueeze(0), p=2.0).squeeze(0) # [num_query, num_database]

    stats = {}
    for curr_seq_ID, curr_seq_distmat in tqdm.tqdm(distmat.items(), disable=silent):
        curr_seq_pos_mat_eval = torch.tensor(dataset.true_neighbors_matrix[curr_seq_ID], device=device)
        assert curr_seq_pos_mat_eval.shape[0] == curr_seq_distmat.shape[1]
        curr_seq_pos_mat_eval[torch.eye(curr_seq_pos_mat_eval.shape[0], dtype=torch.bool, device=device)] = False # as the CMVM did
        num_evaluated = torch.count_nonzero(torch.sum(curr_seq_pos_mat_eval.type(torch.int32), dim=-1, keepdim=False))
        if num_evaluated == 0:
            print(f"no positive pairs in {curr_seq_ID}")
            continue

        pair_recall, pair_dist, pair_opr = get_recall_CMC(                
            curr_seq_distmat,
            curr_seq_pos_mat_eval)
        curr_seq_ave_recall = np.array(pair_recall)
        curr_seq_ave_one_percent_recall = pair_opr
        for x in pair_dist:
            dist.append(x)
        curr_seq_average_dist = np.mean(pair_dist)

        stats[curr_seq_ID] = {'ave_one_percent_recall': float(curr_seq_ave_one_percent_recall), 
                              'ave_recall': curr_seq_ave_recall.tolist(),
                              'average_sim': float(curr_seq_average_dist)}
    return stats


def get_recall_CMC(distmat, pos_mat_eval):

    mask = ~torch.eye(distmat.shape[0], dtype=torch.bool, device=distmat.device)
    distmat = distmat[mask].reshape(distmat.shape[0], distmat.shape[0] - 1)
    pos_mat_eval = pos_mat_eval[mask].reshape(pos_mat_eval.shape[0], pos_mat_eval.shape[0] - 1)
    num_neighbors = 60
    num_evaluated = torch.count_nonzero(torch.sum(pos_mat_eval.type(torch.int32), dim=-1, keepdim=False))
    DB = distmat.shape[1]
    sorted_mat, sorted_indices = torch.sort(distmat, dim=-1, descending=False)
    sorted_mat = sorted_mat.contiguous()
    distmat = distmat.contiguous()
    rank_mat = torch.searchsorted(sorted_mat, distmat, right=False)
    rank_mat[~pos_mat_eval] = num_neighbors
    t4 = time.time()
    recall = [0] * num_neighbors

    top1_dist_score = [0.0]
    one_percent_threshold = max(int(round(DB/100.0)), 1)
    min_rank_row, _ = torch.min(rank_mat, dim=1, keepdim=False)
    for r in range(num_neighbors):
        recall[r] = np.array(torch.count_nonzero(min_rank_row == r).to('cpu')).item()
    t5 = time.time()
    recall = (np.cumsum(recall)/float(num_evaluated))*100  #after caculation, recall[i] means recall@(i+1)
    one_percent_recall = recall[one_percent_threshold - 1]
    return recall, top1_dist_score, one_percent_recall