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


def evaluate_9(model, device, cfgs, dataset_path, val_eval):
    start_epoch = 0
    dataloader, dataset = make_dataloader(cfgs.dataloader_cfgs, dataset_path, start_epoch, val_eval)
    temp = evaluate_dataset_kitti_odometry(model, device, cfgs, dataloader, dataset, False)
    temp_out = {}
    temp_out['kitti_odometry'] = temp
    return temp_out

def evaluate_dataset_kitti_odometry(model, device, cfgs, dataloader, dataset, silent=True):
    # Run evaluation on a single dataset
    dist = []

    database_embeddings1 = {}
    query_embeddings1 = {}
    database_embeddings2 = {}
    query_embeddings2 = {}
    database_embeddings3 = {}
    query_embeddings3 = {}

    model.eval()
    model.to(device)

    DEBUG_CTR = 0
    t1 = time.time()
    for data in tqdm.tqdm(dataloader):
        #-----------------------------------data-----------------------------------
        curr_seq_ID = data['labels'][0][0]
        if curr_seq_ID not in database_embeddings1.keys():
            query_embeddings1[curr_seq_ID] = []
            database_embeddings1[curr_seq_ID] = []
            query_embeddings2[curr_seq_ID] = []
            database_embeddings2[curr_seq_ID] = []
            query_embeddings3[curr_seq_ID] = []
            database_embeddings3[curr_seq_ID] = []
        t2 = time.time()
        data_input = {}
        if "imgnet_cfgs" in cfgs.model_cfgs.keys() or ("cmvpr2_cfgs" in cfgs.model_cfgs.keys() and "image_encoder_type" in cfgs.model_cfgs.cmvpr2_cfgs):
            data_input['images'] = data['images'].to(device)
        if "rendernet_cfgs" in cfgs.model_cfgs.keys() or ("cmvpr2_cfgs" in cfgs.model_cfgs.keys() and "render_encoder_type" in cfgs.model_cfgs.cmvpr2_cfgs):
            if 'render_imgs' in data.keys():
                data_input['render_imgs'] = data['render_imgs'].to(device)
            elif 'range_imgs' in data.keys():
                data_input['render_imgs'] = data['range_imgs'].to(device)
            else:
                raise ValueError('no render_imgs or range_imgs in the data')
        if "imagebevnet_cfgs" in cfgs.model_cfgs.keys() or ("cmvpr2_cfgs" in cfgs.model_cfgs.keys() and "image_bev_encoder_type" in cfgs.model_cfgs.cmvpr2_cfgs):
            data_input['image_bevs'] = data['image_bevs'].to(device)
        if "pcbevnet_cfgs" in cfgs.model_cfgs.keys() or ("cmvpr2_cfgs" in cfgs.model_cfgs.keys() and "pc_bev_encoder_type" in cfgs.model_cfgs.cmvpr2_cfgs):
            data_input['pc_bevs'] = data['pc_bevs'].to(device)
        
        x = model(data_input)
        query_embedding1 = x['embeddings1'] # "image"
        database_embedding1 = x['embeddings2'] # "cloud"
        query_embedding2 = x['embeddings3'] # "image"
        database_embedding2 = x['embeddings4'] # "cloud"
        query_embedding3 = x['embeddings5'] # "image"
        database_embedding3 = x['embeddings6'] # "cloud"

        if 'evaluate_normalize' in cfgs.keys() and cfgs.evaluate_normalize:
            print('embeddings normalized when evaluating')
            query_embedding1 = F.normalize(query_embedding1, p=2, dim=1)
            database_embedding1 = F.normalize(database_embedding1, p=2, dim=1)
            query_embedding2 = F.normalize(query_embedding2, p=2, dim=1)
            database_embedding2 = F.normalize(database_embedding2, p=2, dim=1)
            query_embedding3 = F.normalize(query_embedding3, p=2, dim=1)
            database_embedding3 = F.normalize(database_embedding3, p=2, dim=1)
    
        
        query_embeddings1[curr_seq_ID].append(query_embedding1)
        database_embeddings1[curr_seq_ID].append(database_embedding1)
        query_embeddings2[curr_seq_ID].append(query_embedding2)
        database_embeddings2[curr_seq_ID].append(database_embedding2)
        query_embeddings3[curr_seq_ID].append(query_embedding3)
        database_embeddings3[curr_seq_ID].append(database_embedding3)

        DEBUG_CTR += 1

        if DEBUG and DEBUG_CTR == 50:
            break
        

    # torch.cuda.empty_cache()
    distmat1 = {}
    distmat2 = {}
    distmat3 = {}
    for key in query_embeddings1.keys():
        query_embeddings1[key] = torch.cat(query_embeddings1[key], dim=0) # [num_query, dim]
        database_embeddings1[key] = torch.cat(database_embeddings1[key], dim=0) # [num_database, dim]
        distmat1[key] = torch.cdist(query_embeddings1[key].unsqueeze(0), database_embeddings1[key].unsqueeze(0), p=2.0).squeeze(0) # [num_query, num_database]
        query_embeddings2[key] = torch.cat(query_embeddings2[key], dim=0) # [num_query, dim]
        database_embeddings2[key] = torch.cat(database_embeddings2[key], dim=0) # [num_database, dim]
        distmat2[key] = torch.cdist(query_embeddings2[key].unsqueeze(0), database_embeddings2[key].unsqueeze(0), p=2.0).squeeze(0) # [num_query, num_database]
        query_embeddings3[key] = torch.cat(query_embeddings3[key], dim=0)
        database_embeddings3[key] = torch.cat(database_embeddings3[key], dim=0)
        distmat3[key] = torch.cdist(query_embeddings3[key].unsqueeze(0), database_embeddings3[key].unsqueeze(0), p=2.0).squeeze(0) # [num_query, num_database]

    stats = {}
    for curr_seq_ID, curr_seq_distmat1 in tqdm.tqdm(distmat1.items(), disable=silent):
        curr_seq_pos_mat_eval = torch.tensor(dataset.true_neighbors_matrix[curr_seq_ID], device=device)
        assert curr_seq_pos_mat_eval.shape[0] == curr_seq_distmat1.shape[1]
        curr_seq_pos_mat_eval[torch.eye(curr_seq_pos_mat_eval.shape[0], dtype=torch.bool, device=device)] = False # as the CMVM did
        num_evaluated = torch.count_nonzero(torch.sum(curr_seq_pos_mat_eval.type(torch.int32), dim=-1, keepdim=False))
        if num_evaluated == 0:
            print(f"no positive pairs in {curr_seq_ID}")
            continue
        curr_seq_distmat2 = distmat2[curr_seq_ID]
        curr_seq_distmat3 = distmat3[curr_seq_ID]
        pair_recall, pair_dist, pair_opr = get_recall_CMC(                
            curr_seq_distmat1,
            curr_seq_distmat2,
            curr_seq_distmat3,
            curr_seq_pos_mat_eval,
            cfgs.evaluate_9_cfgs)
        curr_seq_ave_recall = np.array(pair_recall)
        curr_seq_ave_one_percent_recall = pair_opr
        for x in pair_dist:
            dist.append(x)
        curr_seq_average_dist = np.mean(pair_dist)

        stats[curr_seq_ID] = {'ave_one_percent_recall': float(curr_seq_ave_one_percent_recall), 
                              'ave_recall': curr_seq_ave_recall.tolist(),
                              'average_sim': float(curr_seq_average_dist)}
    return stats


def get_recall_CMC(distmat1, distmat2, distmat3, pos_mat_eval, evaluate_9_cfgs):

    mask = ~torch.eye(distmat1.shape[0], dtype=torch.bool, device=distmat1.device)
    distmat1 = distmat1[mask].reshape(distmat1.shape[0], distmat1.shape[0] - 1)
    distmat2 = distmat2[mask].reshape(distmat2.shape[0], distmat2.shape[0] - 1)
    distmat3 = distmat3[mask].reshape(distmat3.shape[0], distmat3.shape[0] - 1)
    pos_mat_eval = pos_mat_eval[mask].reshape(pos_mat_eval.shape[0], pos_mat_eval.shape[0] - 1)
    num_neighbors = 60
    num_evaluated = torch.count_nonzero(torch.sum(pos_mat_eval.type(torch.int32), dim=-1, keepdim=False))
    DB = distmat1.shape[1]

    sorted_mat1, sorted_indices1 = torch.sort(distmat1, dim=-1, descending=False)
    sorted_mat1 = sorted_mat1.contiguous()

    sorted_mat1_topk = sorted_mat1[:, :num_neighbors]
    sorted_indices1_topk = sorted_indices1[:, :num_neighbors] # [num_query, num_neighbors]
    distmat2_topk = torch.gather(distmat2, 1, sorted_indices1_topk)
    distmat3_topk = torch.gather(distmat3, 1, sorted_indices1_topk)
    pos_mat_eval_topk = torch.gather(pos_mat_eval, 1, sorted_indices1_topk)
    sorted_mat2_topk, sorted_indices2_topk = torch.sort(distmat2_topk, dim=-1, descending=False)
    sorted_mat3_topk, sorted_indices3_topk = torch.sort(distmat3_topk, dim=-1, descending=False)

    sorted_mat1_topk = sorted_mat1_topk.contiguous()
    sorted_mat2_topk = sorted_mat2_topk.contiguous()
    sorted_mat3_topk = sorted_mat3_topk.contiguous()
    distmat2_topk = distmat2_topk.contiguous()
    distmat3_topk = distmat3_topk.contiguous()

    rank_mat1 = torch.searchsorted(sorted_mat1_topk, sorted_mat1_topk, right=False)
    rank_mat2 = torch.searchsorted(sorted_mat2_topk, distmat2_topk, right=False)
    rank_mat3 = torch.searchsorted(sorted_mat3_topk, distmat3_topk, right=False)
    rank_mat_all = evaluate_9_cfgs.rank_weight1 * rank_mat1 + evaluate_9_cfgs.rank_weight2 * rank_mat2 + evaluate_9_cfgs.rank_weight3 * rank_mat3
    sorted_rank_mat_all, sorted_indices_all = torch.sort(rank_mat_all, dim=-1, descending=False)
    sorted_rank_mat_all = sorted_rank_mat_all.contiguous()
    rank_mat_all = rank_mat_all.contiguous()
    rank_mat = torch.searchsorted(sorted_rank_mat_all, rank_mat_all, right=False)
    rank_mat[~pos_mat_eval_topk] = num_neighbors
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