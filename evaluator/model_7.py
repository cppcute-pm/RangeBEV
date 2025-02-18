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
import copy
JACCARD_TYPE = 'v3' # 'v1'、'v2'、'v3'
JACCARD_SPECIFIC_TYPE = 'v2' # 'v1'、'v2'、'v3'
GET_TYPE_V2 = 'v1' # 'v1'、'v2'、'v3'
print(f"JACCARD_TYPE: {JACCARD_TYPE}")
print(f"JACCARD_SPECIFIC_TYPE: {JACCARD_SPECIFIC_TYPE}")
print(f"GET_TYPE_V2: {GET_TYPE_V2}")

def k_reciprocal_neigh(initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i, :k1 + 1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
    fi = np.where(backward_k_neigh_index == i)[0]
    return forward_k_neigh_index[fi]

def evaluate_7(model_img, model_pc, device, cfgs, dataset_path, val_eval, out_put_pair_idxs, wandb_id):
    start_epoch = 0
    dataloader, dataset = make_dataloader(cfgs.dataloader_cfgs, dataset_path, start_epoch, val_eval)
    temp = {}
    temp['RGB2LiDAR'] = evaluate_dataset_SES(model_img, model_pc, device, cfgs, dataloader, dataset, out_put_pair_idxs, wandb_id, False)
    temp_out = {}
    temp_out['oxford'] = temp
    return temp_out

def evaluate_dataset_SES(model_img, model_pc, device, cfgs, dataloader, dataset, out_put_pair_idxs, wandb_id, silent=True):
    # Run evaluation on a single dataset
    recall = np.zeros(25)
    count = 0
    dist = []
    one_percent_recall = []

    img_database_embeddings = []
    img_query_embeddings = []
    pc_database_embeddings = []
    pc_query_embeddings = []

    model_img.eval()
    model_pc.eval()
    model_img.to(device)
    model_pc.to(device)

    DEBUG_CTR = 0
    t1 = time.time()
    for data in tqdm.tqdm(dataloader):
        #-----------------------------------data-----------------------------------
        t2 = time.time()
        temp = t2 - t1
        data_input = {}
        data_input['images'] = data['images'].to(device)
        another_input_list = [pc.type(torch.float32) for pc in data['clouds']]
        data_input['clouds'] = torch.stack(another_input_list, dim=0).to(device)
        x_img = model_img(data_input)
        img_query_embedding = x_img['embeddings']
        img_database_embedding = x_img['embeddings']
        x_pc = model_pc(data_input)
        pc_query_embedding = x_pc['embeddings']
        pc_database_embedding = x_pc['embeddings']

        img_query_embeddings.append(img_query_embedding)
        img_database_embeddings.append(img_database_embedding)
        pc_query_embeddings.append(pc_query_embedding)
        pc_database_embeddings.append(pc_database_embedding)

        DEBUG_CTR += 1

        if DEBUG and DEBUG_CTR == 50:
            break
        

    # torch.cuda.empty_cache()
    img_query_embeddings = torch.cat(img_query_embeddings, dim=0)
    if JACCARD_TYPE is None:
        img_query_embeddings = img_query_embeddings[dataset.test_query_idx_list, ...]
    img_database_embeddings = torch.cat(img_database_embeddings, dim=0)
    pc_query_embeddings = torch.cat(pc_query_embeddings, dim=0)
    if JACCARD_TYPE is None:
        pc_query_embeddings = pc_query_embeddings[dataset.test_query_idx_list, ...]
    pc_database_embeddings = torch.cat(pc_database_embeddings, dim=0)
    print('embeddings normalized when evaluating')
    F.normalize(pc_query_embeddings, p=2, dim=1, out=pc_query_embeddings)
    F.normalize(pc_database_embeddings, p=2, dim=1, out=pc_database_embeddings)
    F.normalize(img_query_embeddings, p=2, dim=1, out=img_query_embeddings)
    F.normalize(img_database_embeddings, p=2, dim=1, out=img_database_embeddings)
    assert img_query_embeddings.shape[0] == pc_query_embeddings.shape[0]
    assert img_database_embeddings.shape[0] == pc_database_embeddings.shape[0]
    distmat = torch.cdist(img_query_embeddings, img_database_embeddings, p=2.0) # [QB, DB]
    QB = img_query_embeddings.shape[0]
    if JACCARD_TYPE is None:
        test_coords_tensor = dataset.UTM_coord_tensor[dataset.test_query_idx_list, ...].to(device)
    else:
        test_coords_tensor = dataset.UTM_coord_tensor.to(device)
    db_coords_tensor = dataset.UTM_coord_tensor.to(device)
    test_to_db_dist = torch.cdist(test_coords_tensor.unsqueeze(0), db_coords_tensor.unsqueeze(0), p=2.0).squeeze(0)
    true_neighbors_matrix = torch.lt(test_to_db_dist, dataset.true_neighbor_dist)
    if torch.isnan(distmat).any() or torch.isinf(distmat).any():
        print("the problem is the dis_mat")
    else:
        print("the sim_mat is good")
    if out_put_pair_idxs:
        all_out_put_pair_idxs_list = []
        out_put_pair_idxs_list = []

    if JACCARD_TYPE is not None:
        pc_distmat = torch.cdist(pc_query_embeddings, pc_database_embeddings, p=2.0) # [QB, DB]
        pc_distmat[dataset.test_query_idx_list, :] = 1e6
        coords_distmat = test_to_db_dist # [QB, DB]
        coords_distmat[dataset.test_query_idx_list, :] = 1e6
        distmat = compute_jaccard_distance_v2(distmat, pc_distmat, coords_distmat, dataset.test_query_idx_list)
        distmat = distmat[dataset.test_query_idx_list, ...] # [qbs, dbs]
        true_neighbors_matrix = true_neighbors_matrix[dataset.test_query_idx_list, ...] # [qbs, dbs]
    
    for i in tqdm.tqdm(range(len(dataset.test_traversal_cumsum)), disable=silent):
        if out_put_pair_idxs:
            out_put_pair_idxs_list = []
        for j in range(len(dataset.db_traversal_cumsum)): # i reprensents the index of run
            if i == j: # TODO: is this necessary for cross modal retrieval?
                continue

            if i != 0:
                q_start_idxs = dataset.test_traversal_cumsum[i - 1]
            else:
                q_start_idxs = 0
            q_end_idxs = dataset.test_traversal_cumsum[i]
            if j != 0:
                d_start_idxs = dataset.db_traversal_cumsum[j - 1]
            else:
                d_start_idxs = 0
            d_end_idxs = dataset.db_traversal_cumsum[j]
            pos_mat_eval = true_neighbors_matrix[q_start_idxs:q_end_idxs, d_start_idxs:d_end_idxs]
            num_evaluated = torch.count_nonzero(torch.sum(pos_mat_eval.type(torch.int32), dim=-1, keepdim=False))
            if num_evaluated == 0:
                continue
            if GET_TYPE_V2 == 'v1':
                if out_put_pair_idxs:
                    pair_recall, pair_dist, pair_opr = get_recall_CMC(                
                        distmat,
                        q_start_idxs,
                        q_end_idxs,
                        d_start_idxs,
                        d_end_idxs,
                        true_neighbors_matrix,
                        out_put_pair_idxs,
                        out_put_pair_idxs_list)
                else:
                    pair_recall, pair_dist, pair_opr = get_recall_CMC(                
                        distmat,
                        q_start_idxs,
                        q_end_idxs,
                        d_start_idxs,
                        d_end_idxs,
                        true_neighbors_matrix,
                        out_put_pair_idxs)
                recall += np.array(pair_recall)
                count += 1
                one_percent_recall.append(pair_opr)
                for x in pair_dist:
                    dist.append(x)
            elif GET_TYPE_V2 == 'v2':
                if out_put_pair_idxs:
                    pair_recall, pair_dist, pair_opr = get_recall_CMC_v2(                
                        distmat,
                        q_start_idxs,
                        q_end_idxs,
                        d_start_idxs,
                        d_end_idxs,
                        true_neighbors_matrix,
                        img_query_embeddings,
                        img_database_embeddings,
                        pc_query_embeddings,
                        pc_database_embeddings,
                        test_coords_tensor,
                        db_coords_tensor,
                        out_put_pair_idxs,
                        out_put_pair_idxs_list)
                else:
                    pair_recall, pair_dist, pair_opr = get_recall_CMC_v2(                
                        distmat,
                        q_start_idxs,
                        q_end_idxs,
                        d_start_idxs,
                        d_end_idxs,
                        true_neighbors_matrix,
                        img_query_embeddings,
                        img_database_embeddings,
                        pc_query_embeddings,
                        pc_database_embeddings,
                        test_coords_tensor,
                        db_coords_tensor,
                        out_put_pair_idxs)
            recall += np.array(pair_recall)
            count += 1
            one_percent_recall.append(pair_opr)
            for x in pair_dist:
                dist.append(x)
        if out_put_pair_idxs:
            all_out_put_pair_idxs_list.append(torch.cat(out_put_pair_idxs_list, dim=-1))
    
    if out_put_pair_idxs:
        data_save_root = '/DATA5/vis_vpr'
        data_save_dir = os.path.join(data_save_root, wandb_id)
        os.makedirs(data_save_dir, exist_ok=True)
        all_out_put_pair_idxs = torch.cat(all_out_put_pair_idxs_list, dim=0) # [qbs, traversal_num * num_neighbors]
        assert all_out_put_pair_idxs.shape[0] == QB
        all_out_put_pair_idxs_path = os.path.join(data_save_dir, 'all_out_put_pair_idxs.npy')
        all_out_put_pair_idxs_npy = all_out_put_pair_idxs.cpu().numpy()
        np.save(all_out_put_pair_idxs_path, all_out_put_pair_idxs_npy)


    ave_recall = recall / count
    average_dist = np.mean(dist)
    ave_one_percent_recall = np.mean(one_percent_recall)
    stats = {'ave_one_percent_recall': float(ave_one_percent_recall), 'ave_recall': ave_recall.tolist(),
             'average_sim': float(average_dist)}
    return stats


def get_recall_CMC_v2(img_distmat_all, 
                   q_start_idxs, 
                   q_end_idxs, 
                   d_start_idxs, 
                   d_end_idxs,                  
                   true_neighbors_matrix,
                   img_query_embeddings_all, 
                   img_database_embeddings_all,
                   pc_query_embeddings_all, 
                   pc_database_embeddings_all,
                   test_coords_tensor_all,
                   db_coords_tensor_all,
                   out_put_pair_idxs,
                   out_put_pair_idxs_list=None):

    # should normalize the embeddings first
    QB = q_end_idxs - q_start_idxs
    DB = d_end_idxs - d_start_idxs


    # pc对性能引起大幅度下降，不管采用哪种additional enhancement，UTM坐标的性能下降要好一丢丢
    top_rank = 100
    topk_in_top_rank = 5
    beta=0.15
    use_score_type = 'q_2_to_db_2'
    additional_enhancement = None # 'use_imgsim_for_pc'、'use_pcsim_for_pc'、None
    use_sim = 'UTM_coords' # 'pc_embeddings'
    threshold = 20.0 # threshold越大，性能下降越多


    
    out_dim=pc_query_embeddings_all.shape[1]
    top_rank_inuse = min(top_rank, DB)
    img_distmat = img_distmat_all[q_start_idxs:q_end_idxs, d_start_idxs:d_end_idxs] # [QB, DB]
    rank_candidate_idxs = torch.topk(img_distmat, dim=-1, k=top_rank_inuse, largest=False, sorted=False)[1] # [QB, top_rank_inuse]
    img_rank_candidate_embeddings = img_database_embeddings_all[d_start_idxs:d_end_idxs][rank_candidate_idxs] # [QB, top_rank_inuse, out_dim]
    pc_rank_candidate_embeddings = pc_database_embeddings_all[d_start_idxs:d_end_idxs][rank_candidate_idxs] # [QB, top_rank_inuse, out_dim]

    if use_sim == 'UTM_coords':
        db_coords_tensor = db_coords_tensor_all[d_start_idxs:d_end_idxs] # [DB, 2]
        rank_candidate_coords_tensor = db_coords_tensor[rank_candidate_idxs] # [QB, top_rank_inuse, 2]
        rank_candidate_distmat = torch.cdist(rank_candidate_coords_tensor, rank_candidate_coords_tensor, p=2.0).squeeze(0) # [QB, top_rank_inuse, top_rank_inuse]
        zero_mask = torch.ge(rank_candidate_distmat, threshold)
        pc_rank_candidate_simmat = (threshold - rank_candidate_distmat) * 1.0 / threshold # [QB, top_rank_inuse, top_rank_inuse]
        pc_rank_candidate_simmat[zero_mask] = 0.0
    else:
        pc_rank_candidate_simmat = torch.matmul(pc_rank_candidate_embeddings, pc_rank_candidate_embeddings.permute(0, 2, 1)) # [QB, top_rank_inuse, top_rank_inuse]
    pc_rank_candidate_knn_sim, pc_rank_candidate_knn_idxs = torch.topk(pc_rank_candidate_simmat, 
                                                                 dim=-1, 
                                                                 k=topk_in_top_rank, 
                                                                 largest=True, 
                                                                 sorted=True) # [QB, top_rank_inuse, topk_in_top_rank]
    pc_rank_candidate_knn_sim[..., 1:] *= beta # [QB, top_rank_inuse, topk_in_top_rank]
    pc_rank_candidate_knn_sim[..., 0] *= 1.0 # [QB, top_rank_inuse, topk_in_top_rank]
    img_rank_candidate_knn_embeddings = img_rank_candidate_embeddings.gather(dim=1, 
                                                                     index=pc_rank_candidate_knn_idxs.flatten(1, 2).unsqueeze(-1).expand(-1, -1, out_dim)) # [QB, top_rank_inuse * topk_in_top_rank, dim]
    img_rank_candidate_knn_embeddings = img_rank_candidate_knn_embeddings.reshape(QB, top_rank_inuse, topk_in_top_rank, out_dim) # [QB, top_rank_inuse, topk_in_top_rank, dim]
    img_rank_candidate_embeddings_enhanced = torch.sum(img_rank_candidate_knn_embeddings * pc_rank_candidate_knn_sim.unsqueeze(-1), dim=2) / torch.sum(pc_rank_candidate_knn_sim.unsqueeze(-1), dim=2) # [QB, top_rank_inuse, dim]
    img_query_embeddings_inuse = img_query_embeddings_all[q_start_idxs:q_end_idxs] # [QB, dim]
    img_query_to_rank_candidate_enhanced_simmat = torch.einsum('bnd,bd->bn', img_rank_candidate_embeddings_enhanced, img_query_embeddings_inuse) # [QB, top_rank_inuse]


    if additional_enhancement == 'use_imgsim_for_pc':
        img_rank_candidate_simmat = torch.matmul(img_rank_candidate_embeddings, img_rank_candidate_embeddings.permute(0, 2, 1)) # [QB, top_rank_inuse, top_rank_inuse]
        img_rank_candidate_knn_sim, img_rank_candidate_knn_idxs = torch.topk(img_rank_candidate_simmat, 
                                                                 dim=-1, 
                                                                 k=topk_in_top_rank, 
                                                                 largest=True, 
                                                                 sorted=True) # [QB, top_rank_inuse, topk_in_top_rank]
        img_rank_candidate_knn_sim[..., 1:] *= beta # [QB, top_rank_inuse, topk_in_top_rank]
        img_rank_candidate_knn_sim[..., 0] *= 1.0 # [QB, top_rank_inuse, topk_in_top_rank]
        pc_rank_candidate_knn_embeddings = pc_rank_candidate_embeddings.gather(dim=1, 
                                                                        index=img_rank_candidate_knn_idxs.flatten(1, 2).unsqueeze(-1).expand(-1, -1, out_dim)) # [QB, top_rank_inuse * topk_in_top_rank, dim]
        pc_rank_candidate_knn_embeddings = pc_rank_candidate_knn_embeddings.reshape(QB, top_rank_inuse, topk_in_top_rank, out_dim) # [QB, top_rank_inuse, topk_in_top_rank, dim]
        pc_rank_candidate_embeddings_enhanced = torch.sum(pc_rank_candidate_knn_embeddings * img_rank_candidate_knn_sim.unsqueeze(-1), dim=2) / torch.sum(img_rank_candidate_knn_sim.unsqueeze(-1), dim=2) # [QB, top_rank_inuse, dim]

    img_query_to_db_new_knn_idxs = torch.topk(img_query_to_rank_candidate_enhanced_simmat, 
                                        dim=1, 
                                        k=topk_in_top_rank, 
                                        largest=True, 
                                        sorted=False)[1] # [QB, topk_in_top_rank]
    img_query_to_db_new_knn_embeddings = torch.gather(img_rank_candidate_embeddings_enhanced,
                                                dim=1,
                                                index=img_query_to_db_new_knn_idxs.unsqueeze(-1).expand(-1, -1, out_dim)) # [QB, topk_in_top_rank, dim]
    img_query_embeddings_enhanced = torch.max(img_query_to_db_new_knn_embeddings, dim=1, keepdim=False)[0] # [QB, dim]

    if additional_enhancement == 'use_imgsim_for_pc':
        pc_query_to_db_new_knn_embeddings = torch.gather(pc_rank_candidate_embeddings_enhanced,
                                                dim=1,
                                                index=img_query_to_db_new_knn_idxs.unsqueeze(-1).expand(-1, -1, out_dim)) # [QB, topk_in_top_rank, dim]
        pc_query_embeddings_enhanced = torch.max(pc_query_to_db_new_knn_embeddings, dim=1, keepdim=False)[0] # [QB, dim]
    
    if use_score_type == 'q_2_to_db_1':
        img_query_enhanced_to_rank_candidate_enhanced_simmat = torch.einsum('bnd,bd->bn', img_rank_candidate_embeddings, img_query_embeddings_enhanced) # [QB, top_rank_inuse]
        if additional_enhancement == 'use_imgsim_for_pc':
            pc_query_enhanced_to_rank_candidate_enhanced_simmat = torch.einsum('bnd,bd->bn', pc_rank_candidate_embeddings, pc_query_embeddings_enhanced) # [QB, top_rank_inuse]
    elif use_score_type == 'q_2_to_db_2':
        img_query_enhanced_to_rank_candidate_enhanced_simmat = torch.einsum('bnd,bd->bn', img_rank_candidate_embeddings_enhanced, img_query_embeddings_enhanced) # [QB, top_rank_inuse]
        if additional_enhancement == 'use_imgsim_for_pc':
            pc_query_enhanced_to_rank_candidate_enhanced_simmat = torch.einsum('bnd,bd->bn', pc_rank_candidate_embeddings_enhanced, pc_query_embeddings_enhanced) # [QB, top_rank_inuse]

    query_to_rank_candidate_distmat = 1 - (img_query_to_rank_candidate_enhanced_simmat + img_query_enhanced_to_rank_candidate_enhanced_simmat) / 2.0 # [QB, top_rank_inuse]
    if additional_enhancement == 'use_imgsim_for_pc':
        query_to_rank_candidate_distmat = 1 - (img_query_to_rank_candidate_enhanced_simmat + pc_query_enhanced_to_rank_candidate_enhanced_simmat + img_query_enhanced_to_rank_candidate_enhanced_simmat) / 3.0

    num_neighbors = 25
    pos_mat_eval = true_neighbors_matrix[q_start_idxs:q_end_idxs, d_start_idxs:d_end_idxs] # [QB, DB]
    pos_mat_eval_inuse = torch.gather(pos_mat_eval, dim=1, index=rank_candidate_idxs) # [QB, top_rank_inuse]

    sorted_mat, sorted_indices = torch.sort(query_to_rank_candidate_distmat, dim=-1, descending=False)
    sorted_mat = sorted_mat.contiguous()
    query_to_rank_candidate_distmat = query_to_rank_candidate_distmat.contiguous()
    rank_mat = torch.searchsorted(sorted_mat, query_to_rank_candidate_distmat, right=True) - 1
    if out_put_pair_idxs:
        out_put_pair_idxs_list.append(sorted_indices[:, :num_neighbors] + d_start_idxs)
    rank_mat[~pos_mat_eval_inuse] = num_neighbors
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

def get_recall_CMC(distmat_all, 
                   q_start_idxs, 
                   q_end_idxs, 
                   d_start_idxs, 
                   d_end_idxs, 
                   true_neighbors_matrix,
                   out_put_pair_idxs,
                   out_put_pair_idxs_list=None):

    num_neighbors = 25
    QB = q_end_idxs - q_start_idxs
    DB = d_end_idxs - d_start_idxs
    pos_mat_eval = true_neighbors_matrix[q_start_idxs:q_end_idxs, d_start_idxs:d_end_idxs]
    distmat = distmat_all[q_start_idxs:q_end_idxs, d_start_idxs:d_end_idxs] # [qbs, dbs]
    sorted_mat, sorted_indices = torch.sort(distmat, dim=-1, descending=False)
    sorted_mat = sorted_mat.contiguous()
    distmat = distmat.contiguous()
    rank_mat = torch.searchsorted(sorted_mat, distmat, right=False)
    if out_put_pair_idxs:
        out_put_pair_idxs_list.append(sorted_indices[:, :num_neighbors] + d_start_idxs)
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

@torch.no_grad()
def compute_jaccard_distance_v2(distmat_img, distmat_pc, distmat_coords, test_query_idx_list):

    # ensure the distmat_pc with test query id is 1e6
    original_dist_img = distmat_img
    assert distmat_img.shape[0] == distmat_img.shape[1] == distmat_pc.shape[0] == distmat_pc.shape[1] == distmat_coords.shape[0] == distmat_coords.shape[1]
    N = distmat_img.shape[0]
    mat_type = np.float32
    original_dist_img = original_dist_img.cpu().numpy()
    # the k_img need to be bigger or equal to k_pc and k_coords
    k_img = 100
    k_pc = 100
    k_coords = 100
    global_rank_img = np.argpartition(original_dist_img, range(k_img + 2))
    nn_k_img = [k_reciprocal_neigh(global_rank_img, i, k_img) for i in range(N)]
    nn_k_img_half = [k_reciprocal_neigh(global_rank_img, i, int(np.around(k_img / 2))) for i in range(N)]

    original_dist_pc = distmat_pc.cpu().numpy()
    global_rank_pc = np.argpartition(original_dist_pc, range(k_pc + 2))
    nn_k_pc = []
    nn_k_pc_half = []
    for i in range(N):
        if i in test_query_idx_list:
            nn_k_pc.append(copy.deepcopy(nn_k_img[i]))
            nn_k_pc_half.append(copy.deepcopy(nn_k_img_half[i]))
        else:
            nn_k_pc.append(k_reciprocal_neigh(global_rank_pc, i, k_pc))
            nn_k_pc_half.append(k_reciprocal_neigh(global_rank_pc, i, int(np.around(k_pc / 2))))
    
    for i in range(N):
        if i in test_query_idx_list:
            for j in nn_k_pc[i]:
                nn_k_pc[j] = np.append(nn_k_pc[j], i)
            for j in nn_k_pc_half[i]:
                nn_k_pc_half[j] = np.append(nn_k_pc_half[j], i)


    original_dist_coords = distmat_coords.cpu().numpy()
    global_rank_coords = np.argpartition(original_dist_coords, range(k_coords + 2))
    nn_k_coords = []
    nn_k_coords_half = []
    for i in range(N):
        if i in test_query_idx_list:
            nn_k_coords.append(copy.deepcopy(nn_k_img[i]))
            nn_k_coords_half.append(copy.deepcopy(nn_k_img_half[i]))
        else:
            nn_k_coords.append(k_reciprocal_neigh(global_rank_coords, i, k_coords))
            nn_k_coords_half.append(k_reciprocal_neigh(global_rank_coords, i, int(np.around(k_coords / 2))))
    
    for i in range(N):
        if i in test_query_idx_list:
            for j in nn_k_coords[i]:
                nn_k_coords[j] = np.append(nn_k_coords[j], i)
            for j in nn_k_coords_half[i]:
                nn_k_coords_half[j] = np.append(nn_k_coords_half[j], i)

    if JACCARD_TYPE == 'v1': 
        V_img = np.zeros((N, N), dtype=mat_type)
        for i in range(N):
            k_reciprocal_index = nn_k_img[i]
            k_reciprocal_expansion_index = k_reciprocal_index

            # Jaccard recall
            for candidate in k_reciprocal_index:
                candidate_k_reciprocal_index = nn_k_img_half[candidate]
                if (len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                        candidate_k_reciprocal_index)):
                    k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

            ## element-wise unique
            k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
            img_dist = torch.from_numpy(original_dist_img[i][k_reciprocal_expansion_index]).unsqueeze(0)
            V_img[i, k_reciprocal_expansion_index] = F.softmax(-img_dist, dim=1).view(-1).cpu().numpy()
        
        V_pc = np.zeros((N, N), dtype=mat_type)
        for i in range(N):
            k_reciprocal_index = nn_k_pc[i]
            k_reciprocal_expansion_index = k_reciprocal_index

            # Jaccard recall
            for candidate in k_reciprocal_index:
                candidate_k_reciprocal_index = nn_k_pc_half[candidate]
                if (len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                        candidate_k_reciprocal_index)):
                    k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

            ## element-wise unique
            k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
            if i in test_query_idx_list:
                pc_dist = torch.from_numpy(original_dist_img[i][k_reciprocal_expansion_index]).unsqueeze(0)
            else:
                pc_dist = torch.from_numpy(original_dist_pc[i][k_reciprocal_expansion_index]).unsqueeze(0)
            V_pc[i, k_reciprocal_expansion_index] = F.softmax(-pc_dist, dim=1).view(-1).cpu().numpy()
        
        V_coords = np.zeros((N, N), dtype=mat_type)
        for i in range(N):
            k_reciprocal_index = nn_k_coords[i]
            k_reciprocal_expansion_index = k_reciprocal_index

            # Jaccard recall
            for candidate in k_reciprocal_index:
                candidate_k_reciprocal_index = nn_k_coords_half[candidate]
                if (len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                        candidate_k_reciprocal_index)):
                    k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

            ## element-wise unique
            k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
            if i in test_query_idx_list:
                coords_dist = torch.from_numpy(original_dist_img[i][k_reciprocal_expansion_index]).unsqueeze(0)
            else:
                coords_dist = torch.from_numpy(original_dist_coords[i][k_reciprocal_expansion_index]).unsqueeze(0)
            V_coords[i, k_reciprocal_expansion_index] = F.softmax(-coords_dist, dim=1).view(-1).cpu().numpy()

        if JACCARD_SPECIFIC_TYPE == 'v1':   
            jaccard_dist_img = v2jaccard(V_img, N, mat_type)

            jaccard_dist_pc = v2jaccard(V_pc, N, mat_type)

            jaccard_dist_coords = v2jaccard(V_coords, N, mat_type)

            jaccard_dist = (jaccard_dist_img + jaccard_dist_pc + jaccard_dist_coords) / 3.0
            jaccard_dist = torch.from_numpy(jaccard_dist).cuda()
        
        elif JACCARD_SPECIFIC_TYPE == 'v2':   
            jaccard_dist_img = v2jaccard(V_img, N, mat_type)

            jaccard_dist_pc = v2jaccard(V_pc, N, mat_type)

            jaccard_dist_coords = v2jaccard(V_coords, N, mat_type)

            jaccard_dist_all = np.stack((jaccard_dist_img, jaccard_dist_pc, jaccard_dist_coords), axis=-1) # [N, N, 3]
            jaccard_dist = np.min(jaccard_dist_all, axis=-1)
            jaccard_dist = torch.from_numpy(jaccard_dist).cuda()

        elif JACCARD_SPECIFIC_TYPE == 'v3':   
            jaccard_dist_img = v2jaccard(V_img, N, mat_type)

            jaccard_dist_pc = v2jaccard(V_pc, N, mat_type)

            jaccard_dist_coords = v2jaccard(V_coords, N, mat_type)

            jaccard_dist_all = np.stack((jaccard_dist_img, jaccard_dist_pc, jaccard_dist_coords), axis=-1) # [N, N, 3]
            jaccard_dist = np.max(jaccard_dist_all, axis=-1)
            jaccard_dist = torch.from_numpy(jaccard_dist).cuda()
        elif JACCARD_SPECIFIC_TYPE == 'v4':
            V_all = np.stack((V_img, V_pc, V_coords), axis=-1) # [N, N, 3]
            V = np.min(V_all, axis=-1) # [N, N]
            jaccard_dist = v2jaccard(V, N, mat_type)
            jaccard_dist = torch.from_numpy(jaccard_dist).cuda()
        elif JACCARD_SPECIFIC_TYPE == 'v5':
            V_all = np.stack((V_img, V_pc, V_coords), axis=-1) # [N, N, 3]
            V = np.max(V_all, axis=-1) # [N, N]
            jaccard_dist = v2jaccard(V, N, mat_type)
            jaccard_dist = torch.from_numpy(jaccard_dist).cuda()
        elif JACCARD_SPECIFIC_TYPE == 'v6':
            V_all = np.stack((V_img, V_pc, V_coords), axis=-1) # [N, N, 3]
            V = np.mean(V_all, axis=-1) # [N, N]
            jaccard_dist = v2jaccard(V, N, mat_type)
            jaccard_dist = torch.from_numpy(jaccard_dist).cuda()
        else:
            raise NotImplementedError(f"JACCARD_SPECIFIC_TYPE: {JACCARD_SPECIFIC_TYPE} is not implemented")
    elif JACCARD_TYPE == 'v2':
        V = np.zeros((N, N), dtype=mat_type)
        for i in range(N):
            k_reciprocal_index_img = nn_k_img[i]
            k_reciprocal_expansion_index_img = k_reciprocal_index_img
            # Jaccard recall
            for candidate in k_reciprocal_index_img:
                candidate_k_reciprocal_index = nn_k_img_half[candidate]
                if (len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index_img)) > 2 / 3 * len(
                        candidate_k_reciprocal_index)):
                    k_reciprocal_expansion_index_img = np.append(k_reciprocal_expansion_index_img, candidate_k_reciprocal_index)
            ## element-wise unique
            k_reciprocal_expansion_index_img = np.unique(k_reciprocal_expansion_index_img)

            k_reciprocal_index_pc = nn_k_pc[i]
            k_reciprocal_expansion_index_pc = k_reciprocal_index_pc
            # Jaccard recall
            for candidate in k_reciprocal_index_pc:
                candidate_k_reciprocal_index = nn_k_pc_half[candidate]
                if (len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index_pc)) > 2 / 3 * len(
                        candidate_k_reciprocal_index)):
                    k_reciprocal_expansion_index_pc = np.append(k_reciprocal_expansion_index_pc, candidate_k_reciprocal_index)
            ## element-wise unique
            k_reciprocal_expansion_index_pc = np.unique(k_reciprocal_expansion_index_pc)

            k_reciprocal_index_coords = nn_k_coords[i]
            k_reciprocal_expansion_index_coords = k_reciprocal_index_coords
            # Jaccard recall
            for candidate in k_reciprocal_index_coords:
                candidate_k_reciprocal_index = nn_k_coords_half[candidate]
                if (len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index_coords)) > 2 / 3 * len(
                        candidate_k_reciprocal_index)):
                    k_reciprocal_expansion_index_coords = np.append(k_reciprocal_expansion_index_coords, candidate_k_reciprocal_index)
            ## element-wise unique
            k_reciprocal_expansion_index_coords = np.unique(k_reciprocal_expansion_index_coords)

            if JACCARD_SPECIFIC_TYPE == 'v1':
                k_reciprocal_expansion_index = np.unique(np.concatenate((k_reciprocal_expansion_index_img, k_reciprocal_expansion_index_pc, k_reciprocal_expansion_index_coords))) # union result
            elif JACCARD_SPECIFIC_TYPE == 'v2':
                k_reciprocal_expansion_index = np.intersect1d(np.intersect1d(k_reciprocal_expansion_index_img, k_reciprocal_expansion_index_pc), k_reciprocal_expansion_index_coords) # intersection result
            else:
                raise NotImplementedError(f"JACCARD_SPECIFIC_TYPE: {JACCARD_SPECIFIC_TYPE} is not implemented")
            
            img_dist = torch.from_numpy(original_dist_img[i][k_reciprocal_expansion_index]).unsqueeze(0)
            V[i, k_reciprocal_expansion_index] = F.softmax(-img_dist, dim=1).view(-1).cpu().numpy()

        jaccard_dist = v2jaccard(V, N, mat_type)
        jaccard_dist = torch.from_numpy(jaccard_dist).cuda()
    elif JACCARD_TYPE == 'v3':
        nn_k = []
        nn_k_half = []
        for i in range(N):
            if JACCARD_SPECIFIC_TYPE == 'v1':
                curr_nn_k = np.intersect1d(np.intersect1d(nn_k_img[i], nn_k_pc[i]), nn_k_coords[i])
                curr_nn_k_half = np.intersect1d(np.intersect1d(nn_k_img_half[i], nn_k_pc_half[i]), nn_k_coords_half[i])
            elif JACCARD_SPECIFIC_TYPE == 'v2':
                curr_nn_k = np.union1d(np.union1d(nn_k_img[i], nn_k_pc[i]), nn_k_coords[i])
                curr_nn_k_half = np.union1d(np.union1d(nn_k_img_half[i], nn_k_pc_half[i]), nn_k_coords_half[i])
            else:
                raise NotImplementedError(f"JACCARD_SPECIFIC_TYPE: {JACCARD_SPECIFIC_TYPE} is not implemented")
            nn_k.append(curr_nn_k)
            nn_k_half.append(curr_nn_k_half)
                
        V = np.zeros((N, N), dtype=mat_type)
        for i in range(N):
            k_reciprocal_index = nn_k[i]
            k_reciprocal_expansion_index = k_reciprocal_index
            # Jaccard recall
            for candidate in k_reciprocal_index:
                candidate_k_reciprocal_index = nn_k_half[candidate]
                if (len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                        candidate_k_reciprocal_index)):
                    k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)
            ## element-wise unique
            k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
            
            img_dist = torch.from_numpy(original_dist_img[i][k_reciprocal_expansion_index]).unsqueeze(0)
            V[i, k_reciprocal_expansion_index] = F.softmax(-img_dist, dim=1).view(-1).cpu().numpy()

        jaccard_dist = v2jaccard(V, N, mat_type)
        jaccard_dist = torch.from_numpy(jaccard_dist).cuda()
    else:
        raise NotImplementedError(f"JACCARD_TYPE: {JACCARD_TYPE} is not implemented")
    return jaccard_dist

def v2jaccard(V, N, mat_type):
    invIndex = []
    for i in range(N):
        invIndex.append(np.where(V[:, i] != 0)[0])  # len(invIndex)=all_num

    jaccard_dist = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        temp_min = np.zeros((1, N), dtype=mat_type)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    del invIndex, V

    pos_bool = (jaccard_dist < 0)
    jaccard_dist[pos_bool] = 0.0
    return jaccard_dist