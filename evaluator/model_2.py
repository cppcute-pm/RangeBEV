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


def evaluate_2(model, device, cfgs, dataset_path, val_eval):
    start_epoch = 0
    dataloader, dataset = make_dataloader(cfgs.dataloader_cfgs, dataset_path, start_epoch, val_eval)
    temp = {}
    temp['RGB2LiDAR'] = evaluate_dataset_zenseact(model, device, cfgs, dataloader, dataset, False)
    temp_out = {}
    temp_out['oxford'] = temp
    return temp_out

def evaluate_dataset_zenseact(model, device, cfgs, dataloader, dataset, silent=True):
    # Run evaluation on a single dataset
    recall = np.zeros(500)
    count = 0
    dist = []
    one_percent_recall = []

    database_embeddings = []
    query_embeddings = []

    model.eval()
    model.to(device)

    DEBUG_CTR = 0
    t1 = time.time()
    for data in tqdm.tqdm(dataloader):
        #-----------------------------------data-----------------------------------
        t2 = time.time()
        temp = t2 - t1
        data_input = {}
        if "imgnet_cfgs" in cfgs.model_cfgs.keys() or ("cmvpr_cfgs" in cfgs.model_cfgs.keys() and "image_encoder_type" in cfgs.model_cfgs.cmvpr_cfgs) or ("cmvpr2_cfgs" in cfgs.model_cfgs.keys() and "image_encoder_type" in cfgs.model_cfgs.cmvpr2_cfgs):
            data_input['images'] = data['images'].to(device)
        if "rendernet_cfgs" in cfgs.model_cfgs.keys() or ("cmvpr2_cfgs" in cfgs.model_cfgs.keys() and "render_encoder_type" in cfgs.model_cfgs.cmvpr2_cfgs):
            data_input['render_imgs'] = data['render_imgs'].to(device)
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


        query_embeddings.append(query_embedding)
        database_embeddings.append(database_embedding)

        DEBUG_CTR += 1

        if DEBUG and DEBUG_CTR == 50:
            break
        

    torch.cuda.empty_cache()
    query_embeddings = torch.cat(query_embeddings, dim=0)
    database_embeddings = torch.cat(database_embeddings, dim=0)

    if 'evaluate_normalize' in cfgs.keys() and cfgs.evaluate_normalize:
        print('embeddings normalized when evaluating')
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        database_embeddings = F.normalize(database_embeddings, p=2, dim=1)
    
    QB = query_embeddings.shape[0]
    DB = database_embeddings.shape[0]
    distmat = torch.pow(query_embeddings, 2).sum(dim=1, keepdim=True).expand(QB, DB) + \
            torch.pow(database_embeddings, 2).sum(dim=1, keepdim=True).expand(DB, QB).t()
    distmat.addmm_(mat1=query_embeddings, 
                   mat2=database_embeddings.t(), 
                   beta=1, 
                   alpha=-2) # [qbs, dbs]

    pair_recall, pair_dist, pair_opr = get_recall_CMC(distmat)
    recall = np.array(pair_recall)
    one_percent_recall = pair_opr
    for x in pair_dist:
        dist.append(x)

    ave_recall = recall
    average_dist = np.mean(dist)
    ave_one_percent_recall = one_percent_recall
    stats = {'ave_one_percent_recall': float(ave_one_percent_recall), 'ave_recall': ave_recall.tolist(),
             'average_sim': float(average_dist)}
    return stats

def get_recall_CMC(distmat_all):

    num_neighbors = 500
    QB, DB = distmat_all.shape
    pos_mat_eval = torch.eye(QB, dtype=torch.bool, device=distmat_all.device)
    distmat = distmat_all
    sorted_mat, sorted_indices = torch.sort(distmat, dim=-1, descending=False)
    sorted_mat = sorted_mat.contiguous()
    distmat = distmat.contiguous()
    rank_mat =  torch.searchsorted(sorted_mat, distmat, right=False)
    rank_mat[~pos_mat_eval] = num_neighbors
    num_evaluated = torch.count_nonzero(torch.sum(pos_mat_eval.type(torch.int32), dim=-1, keepdim=False))
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