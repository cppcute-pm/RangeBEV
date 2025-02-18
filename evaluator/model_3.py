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

# in this evaluator, all the embeddings should be normalized, so be careful with the loss function
def evaluate_3(model, device, cfgs, dataset_path, val_eval):
    start_epoch = 0
    dataloader, dataset = make_dataloader(cfgs.dataloader_cfgs, dataset_path, start_epoch, val_eval)
    temp = {}
    temp['RGB2LiDAR'] = evaluate_dataset_SES(model, device, cfgs, dataloader, dataset, False)
    temp_out = {}
    temp_out['oxford'] = temp
    return temp_out

def evaluate_dataset_SES(model, device, cfgs, dataloader, dataset, silent=True):
    # Run evaluation on a single dataset
    recall = np.zeros(25)
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

        query_embedding = F.normalize(query_embedding, p=2, dim=1)
        database_embedding = F.normalize(database_embedding, p=2, dim=1)
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
    if torch.isnan(distmat).any() or torch.isinf(distmat).any():
        print("the problem is the dis_mat")
    else:
        print("the sim_mat is good")

    for i in tqdm.tqdm(range(len(dataset.traversal_cumsum)), disable=silent):
        for j in range(len(dataset.traversal_cumsum)): # i reprensents the index of run
            if i == j: # TODO: is this necessary for cross modal retrieval?
                continue

            if i != 0:
                q_start_idxs = dataset.traversal_cumsum[i - 1]
            else:
                q_start_idxs = 0
            q_end_idxs = dataset.traversal_cumsum[i]
            if j != 0:
                d_start_idxs = dataset.traversal_cumsum[j - 1]
            else:
                d_start_idxs = 0
            d_end_idxs = dataset.traversal_cumsum[j]
            pos_mat_eval = torch.tensor(dataset.true_neighbors_matrix[q_start_idxs:q_end_idxs, d_start_idxs:d_end_idxs])
            num_evaluated = torch.count_nonzero(torch.sum(pos_mat_eval.type(torch.int32), dim=-1, keepdim=False))
            if num_evaluated == 0:
                continue

            pair_recall, pair_dist, pair_opr = get_recall_CMC(                
                distmat,
                q_start_idxs,
                q_end_idxs,
                d_start_idxs,
                d_end_idxs,
                dataset)
            recall += np.array(pair_recall)
            count += 1
            one_percent_recall.append(pair_opr)
            for x in pair_dist:
                dist.append(x)

    ave_recall = recall / count
    average_dist = np.mean(dist)
    ave_one_percent_recall = np.mean(one_percent_recall)
    stats = {'ave_one_percent_recall': float(ave_one_percent_recall), 'ave_recall': ave_recall.tolist(),
             'average_sim': float(average_dist)}
    return stats

def get_recall_CMC(distmat_all, q_start_idxs, q_end_idxs, d_start_idxs, d_end_idxs, dataset):

    num_neighbors = 25
    QB = q_end_idxs - q_start_idxs
    DB = d_end_idxs - d_start_idxs
    pos_mat_eval = torch.tensor(dataset.true_neighbors_matrix[q_start_idxs:q_end_idxs, d_start_idxs:d_end_idxs])
    distmat = distmat_all[q_start_idxs:q_end_idxs, d_start_idxs:d_end_idxs] # [qbs, dbs]
    sorted_mat, sorted_indices = torch.sort(distmat, dim=-1, descending=False)
    sorted_mat = sorted_mat.contiguous()
    distmat = distmat.contiguous()
    rank_mat =  torch.searchsorted(sorted_mat, distmat, right=False)
    rank_mat[~pos_mat_eval] = num_neighbors
    num_evaluated = torch.count_nonzero(torch.sum(pos_mat_eval.type(torch.int32), dim=-1, keepdim=False))
    if num_evaluated == 0:
        print(
            f"the problem is the num_evaluated  "
            f"q_start_idxs: {q_start_idxs}  "
            f"q_end_idxs: {q_end_idxs}  "
            f"d_start_idxs: {d_start_idxs}  "
            f"d_end_idxs: {d_end_idxs}  "
            )
        return 
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