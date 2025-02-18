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

VIS_RERANK = False
WANDB_ID = 'None'
print('VIS_RERANK:', VIS_RERANK)
print('WANDB_ID:', WANDB_ID)

def evaluate_8(model, device, cfgs, dataset_path, val_eval):
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

        if 'evaluate_normalize' in cfgs.keys() and cfgs.evaluate_normalize:
            print('embeddings normalized when evaluating')
            query_embedding1 = F.normalize(query_embedding1, p=2, dim=1)
            database_embedding1 = F.normalize(database_embedding1, p=2, dim=1)
            query_embedding2 = F.normalize(query_embedding2, p=2, dim=1)
            database_embedding2 = F.normalize(database_embedding2, p=2, dim=1)
    
        
        query_embeddings1[curr_seq_ID].append(query_embedding1)
        database_embeddings1[curr_seq_ID].append(database_embedding1)
        query_embeddings2[curr_seq_ID].append(query_embedding2)
        database_embeddings2[curr_seq_ID].append(database_embedding2)

        DEBUG_CTR += 1

        if DEBUG and DEBUG_CTR == 50:
            break
        

    # torch.cuda.empty_cache()
    distmat1 = {}
    distmat2 = {}
    for key in query_embeddings1.keys():
        query_embeddings1[key] = torch.cat(query_embeddings1[key], dim=0) # [num_query, dim]
        database_embeddings1[key] = torch.cat(database_embeddings1[key], dim=0) # [num_database, dim]
        distmat1[key] = torch.cdist(query_embeddings1[key].unsqueeze(0), database_embeddings1[key].unsqueeze(0), p=2.0).squeeze(0) # [num_query, num_database]
        query_embeddings2[key] = torch.cat(query_embeddings2[key], dim=0) # [num_query, dim]
        database_embeddings2[key] = torch.cat(database_embeddings2[key], dim=0) # [num_database, dim]
        distmat2[key] = torch.cdist(query_embeddings2[key].unsqueeze(0), database_embeddings2[key].unsqueeze(0), p=2.0).squeeze(0) # [num_query, num_database]

    stats = {}
    if VIS_RERANK:
        to_vis_all_indices_all_dict = {}
    for curr_seq_ID, curr_seq_distmat1 in tqdm.tqdm(distmat1.items(), disable=silent):
        curr_seq_pos_mat_eval = torch.tensor(dataset.true_neighbors_matrix[curr_seq_ID], device=device)
        assert curr_seq_pos_mat_eval.shape[0] == curr_seq_distmat1.shape[1]
        curr_seq_pos_mat_eval[torch.eye(curr_seq_pos_mat_eval.shape[0], dtype=torch.bool, device=device)] = False # as the CMVM did
        num_evaluated = torch.count_nonzero(torch.sum(curr_seq_pos_mat_eval.type(torch.int32), dim=-1, keepdim=False))
        if num_evaluated == 0:
            print(f"no positive pairs in {curr_seq_ID}")
            continue
        curr_seq_distmat2 = distmat2[curr_seq_ID]
        pair_recall, pair_dist, pair_opr = get_recall_CMC(                
            curr_seq_distmat1,
            curr_seq_distmat2,
            curr_seq_pos_mat_eval,
            cfgs.evaluate_8_cfgs)
        
        if VIS_RERANK:
            to_vis_all_indices = get_recall_CMC_vis(
                curr_seq_distmat1,
                curr_seq_distmat2,
                curr_seq_pos_mat_eval,
                cfgs.evaluate_8_cfgs)
            to_vis_all_indices_all_dict[curr_seq_ID] = to_vis_all_indices
        curr_seq_ave_recall = np.array(pair_recall)
        curr_seq_ave_one_percent_recall = pair_opr
        for x in pair_dist:
            dist.append(x)
        curr_seq_average_dist = np.mean(pair_dist)

        stats[curr_seq_ID] = {'ave_one_percent_recall': float(curr_seq_ave_one_percent_recall), 
                              'ave_recall': curr_seq_ave_recall.tolist(),
                              'average_sim': float(curr_seq_average_dist)}
    if VIS_RERANK:
        data_save_root = '/DATA5/pengjianyi/vis_rerank'
        data_save_dir = os.path.join(data_save_root, WANDB_ID)
        data_save_path = os.path.join(data_save_dir, 'vis_rerank.pkl')
        os.makedirs(data_save_dir, exist_ok=True)
        with open(data_save_path, 'wb') as f:
            pickle.dump(to_vis_all_indices_all_dict, f)
    return stats


def get_recall_CMC(distmat1, distmat2, pos_mat_eval, evaluate_8_cfgs):

    mask = ~torch.eye(distmat1.shape[0], dtype=torch.bool, device=distmat1.device)
    distmat1 = distmat1[mask].reshape(distmat1.shape[0], distmat1.shape[0] - 1)
    distmat2 = distmat2[mask].reshape(distmat2.shape[0], distmat2.shape[0] - 1)
    pos_mat_eval = pos_mat_eval[mask].reshape(pos_mat_eval.shape[0], pos_mat_eval.shape[0] - 1)
    num_neighbors = 60
    num_evaluated = torch.count_nonzero(torch.sum(pos_mat_eval.type(torch.int32), dim=-1, keepdim=False))
    DB = distmat1.shape[1]

    if evaluate_8_cfgs.type == 1:
        pass
    elif evaluate_8_cfgs.type == 2:
        distmat_temp = distmat1
        distmat1 = distmat2
        distmat2 = distmat_temp
    sorted_mat1, sorted_indices1 = torch.sort(distmat1, dim=-1, descending=False)
    sorted_mat1 = sorted_mat1.contiguous()

    sorted_mat1_topk = sorted_mat1[:, :num_neighbors]
    sorted_indices1_topk = sorted_indices1[:, :num_neighbors] # [num_query, num_neighbors]
    distmat2_topk = torch.gather(distmat2, 1, sorted_indices1_topk)
    pos_mat_eval_topk = torch.gather(pos_mat_eval, 1, sorted_indices1_topk)
    sorted_mat2_topk, sorted_indices2_topk = torch.sort(distmat2_topk, dim=-1, descending=False)

    sorted_mat1_topk = sorted_mat1_topk.contiguous()
    sorted_mat2_topk = sorted_mat2_topk.contiguous()
    distmat2_topk = distmat2_topk.contiguous()

    rank_mat1 = torch.searchsorted(sorted_mat1_topk, sorted_mat1_topk, right=False)
    rank_mat2 = torch.searchsorted(sorted_mat2_topk, distmat2_topk, right=False)
    rank_mat_all = evaluate_8_cfgs.rank_weight1 * rank_mat1 + evaluate_8_cfgs.rank_weight2 * rank_mat2
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

def get_recall_CMC_vis(distmat1, distmat2, pos_mat_eval, evaluate_8_cfgs):

    vis_FP_neighbors = 4
    mask = ~torch.eye(distmat1.shape[0], dtype=torch.bool, device=distmat1.device)
    distmat1 = distmat1[mask].reshape(distmat1.shape[0], distmat1.shape[0] - 1)
    distmat2 = distmat2[mask].reshape(distmat2.shape[0], distmat2.shape[0] - 1)
    pos_mat_eval = pos_mat_eval[mask].reshape(pos_mat_eval.shape[0], pos_mat_eval.shape[0] - 1)
    num_neighbors = 60
    DB = distmat1.shape[1]
    QB = distmat1.shape[0]

    if evaluate_8_cfgs.type == 1:
        pass
    elif evaluate_8_cfgs.type == 2:
        distmat_temp = distmat1
        distmat1 = distmat2
        distmat2 = distmat_temp
    sorted_mat1, sorted_indices1 = torch.sort(distmat1, dim=-1, descending=False)
    sorted_mat1 = sorted_mat1.contiguous()

    sorted_mat1_topk = sorted_mat1[:, :num_neighbors]
    sorted_indices1_topk = sorted_indices1[:, :num_neighbors] # [num_query, num_neighbors]
    distmat2_topk = torch.gather(distmat2, 1, sorted_indices1_topk)
    pos_mat_eval_topk = torch.gather(pos_mat_eval, 1, sorted_indices1_topk)
    sorted_mat2_topk, sorted_indices2_topk = torch.sort(distmat2_topk, dim=-1, descending=False)

    sorted_mat1_topk = sorted_mat1_topk.contiguous()
    sorted_mat2_topk = sorted_mat2_topk.contiguous()
    distmat2_topk = distmat2_topk.contiguous()

    rank_mat1 = torch.searchsorted(sorted_mat1_topk, sorted_mat1_topk, right=False)

    rank_mat2 = torch.searchsorted(sorted_mat2_topk, distmat2_topk, right=False)
    rank_mat_all = evaluate_8_cfgs.rank_weight1 * rank_mat1 + evaluate_8_cfgs.rank_weight2 * rank_mat2
    sorted_rank_mat_all, sorted_indices_all = torch.sort(rank_mat_all, dim=-1, descending=False)
    sorted_rank_mat_all = sorted_rank_mat_all.contiguous()
    rank_mat_all = rank_mat_all.contiguous()
    rank_mat = torch.searchsorted(sorted_rank_mat_all, rank_mat_all, right=False)
    rank_mat_copy = rank_mat.detach().clone()
    rank_mat[~pos_mat_eval_topk] = num_neighbors
    t4 = time.time()

    min_rank_row, min_rank_row_indices = torch.min(rank_mat, dim=1, keepdim=False)

    
    # need review and debug
    distmat1 = distmat1.contiguous()
    sorted_mat1 = sorted_mat1.contiguous()
    rank_mat1_original = torch.searchsorted(sorted_mat1, distmat1, right=False) # [num_query, num_database]
    rank_mat1_original_copy = rank_mat1_original.detach().clone()
    rank_mat1_original[~pos_mat_eval] = num_neighbors
    min_rank_row1_original, min_rank_row1_original_indices = torch.min(rank_mat1_original, dim=1, keepdim=False) # [num_query]
    to_vis_row_mask = ((min_rank_row1_original >= vis_FP_neighbors) & (min_rank_row == 0)) # [num_query]
    to_vis_QB = torch.count_nonzero(to_vis_row_mask)
    to_vis_TP_indices_temp_q_indices = torch.nonzero(to_vis_row_mask, as_tuple=False).squeeze(-1) # [to_vis_QB]
    to_vis_TP_indices_temp = min_rank_row_indices[to_vis_row_mask]
    to_vis_TP_indices = sorted_indices1_topk[to_vis_TP_indices_temp_q_indices, to_vis_TP_indices_temp] # [to_vis_QB]
    to_vis_FP_indices_temp = torch.nonzero(rank_mat1_original_copy[to_vis_row_mask, :] < vis_FP_neighbors, as_tuple=False)
    assert to_vis_FP_indices_temp.shape[0] == to_vis_QB * vis_FP_neighbors
    to_vis_FP_indices = to_vis_FP_indices_temp[:, 1].reshape(to_vis_QB, vis_FP_neighbors)


    to_vis_TP_original_rank = rank_mat1_original_copy[to_vis_TP_indices_temp_q_indices, to_vis_TP_indices]
    # to_vis_rank_mat_copy = rank_mat_copy[to_vis_TP_indices_temp_q_indices, :] # [to_vis_QB, num_database]
    # to_vis_FP_current_rank = torch.gather(to_vis_rank_mat_copy, 1, to_vis_FP_indices) # [to_vis_QB, vis_FP_neighbors]

    to_vis_TP_indices = torch.where(to_vis_TP_indices_temp_q_indices <= to_vis_TP_indices, to_vis_TP_indices + 1, to_vis_TP_indices)
    to_vis_FP_indices = torch.where(to_vis_TP_indices_temp_q_indices.unsqueeze(-1) <= to_vis_FP_indices, to_vis_FP_indices + 1, to_vis_FP_indices)

    to_vis_all_indices = torch.cat((to_vis_TP_indices_temp_q_indices.unsqueeze(-1), 
                                    to_vis_TP_indices.unsqueeze(-1), 
                                    to_vis_FP_indices), dim=-1) # [to_vis_QB, vis_FP_neighbors + 2]

    return to_vis_all_indices.cpu().numpy()