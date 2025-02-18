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
from sklearn.metrics import average_precision_score
from vis_tsne import t_SNE_v2
GET_TYPE = 'recall' # 'recall', 'recall_v1' 'recall_v2' 'mAP'
print('GET_TYPE:', GET_TYPE)
USE_JACCARD_DISTANCE = True
print('USE_JACCARD_DISTANCE:', USE_JACCARD_DISTANCE)
USE_TSNE_VIS = False
print('USE_TSNE_VIS:', USE_TSNE_VIS)

def k_reciprocal_neigh(initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i, :k1 + 1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
    fi = np.where(backward_k_neigh_index == i)[0]
    return forward_k_neigh_index[fi]


def evaluate_6(model, device, cfgs, dataset_path, val_eval, out_put_pair_idxs, wandb_id):
    start_epoch = 0
    dataloader, dataset = make_dataloader(cfgs.dataloader_cfgs, dataset_path, start_epoch, val_eval)
    temp = {}
    temp['RGB2LiDAR'] = evaluate_dataset_SES(model, device, cfgs, dataloader, dataset, out_put_pair_idxs, wandb_id, False)
    temp_out = {}
    temp_out['oxford'] = temp
    return temp_out

def evaluate_dataset_SES(model, device, cfgs, dataloader, dataset, out_put_pair_idxs, wandb_id, silent=True):
    # Run evaluation on a single dataset
    recall = np.zeros(25)
    count = 0
    dist = []
    one_percent_recall = []

    if GET_TYPE == 'mAP':
        final_map = 0.0

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
        

    # torch.cuda.empty_cache()
    query_embeddings = torch.cat(query_embeddings, dim=0)
    if not USE_JACCARD_DISTANCE:
        query_embeddings = query_embeddings[dataset.test_query_idx_list, ...]
    database_embeddings = torch.cat(database_embeddings, dim=0)

    if USE_TSNE_VIS:
        t_SNE_v2(database_embeddings.to('cpu').numpy(), dataset.UTM_coord_tensor.to('cpu').numpy())
        exit(0)

    print('embeddings normalized when evaluating')
    F.normalize(query_embeddings, p=2, dim=1, out=query_embeddings)
    F.normalize(database_embeddings, p=2, dim=1, out=database_embeddings)
    QB = query_embeddings.shape[0]
    DB = database_embeddings.shape[0]
    distmat = torch.pow(query_embeddings, 2).sum(dim=1, keepdim=True).expand(QB, DB) + \
            torch.pow(database_embeddings, 2).sum(dim=1, keepdim=True).expand(DB, QB).t()
    distmat.addmm_(mat1=query_embeddings, 
                   mat2=database_embeddings.t(), 
                   beta=1, 
                   alpha=-2) # [qbs, dbs]
    test_coords_tensor = dataset.UTM_coord_tensor[dataset.test_query_idx_list, ...].to(device)
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
    
    if USE_JACCARD_DISTANCE:
        distmat = compute_jaccard_distance(distmat)
        distmat = distmat[dataset.test_query_idx_list, ...] # [qbs, dbs]
    
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

            if GET_TYPE == 'recall_v2':
                if out_put_pair_idxs:
                    pair_recall, pair_dist, pair_opr = get_recall_CMC_v2(                
                        distmat,
                        q_start_idxs,
                        q_end_idxs,
                        d_start_idxs,
                        d_end_idxs,
                        true_neighbors_matrix,
                        query_embeddings,
                        database_embeddings,
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
                        query_embeddings,
                        database_embeddings,
                        out_put_pair_idxs)
                recall += np.array(pair_recall)
                count += 1
                one_percent_recall.append(pair_opr)
                for x in pair_dist:
                    dist.append(x)
            elif GET_TYPE == 'recall_v1':
                if out_put_pair_idxs:
                    pair_recall, pair_dist, pair_opr = get_recall_CMC_v1(                
                        distmat,
                        q_start_idxs,
                        q_end_idxs,
                        d_start_idxs,
                        d_end_idxs,
                        true_neighbors_matrix,
                        query_embeddings,
                        database_embeddings,
                        out_put_pair_idxs,
                        out_put_pair_idxs_list)
                else:
                    pair_recall, pair_dist, pair_opr = get_recall_CMC_v1(                
                        distmat,
                        q_start_idxs,
                        q_end_idxs,
                        d_start_idxs,
                        d_end_idxs,
                        true_neighbors_matrix,
                        query_embeddings,
                        database_embeddings,
                        out_put_pair_idxs)
                recall += np.array(pair_recall)
                count += 1
                one_percent_recall.append(pair_opr)
                for x in pair_dist:
                    dist.append(x)
            elif GET_TYPE == 'recall':
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
            elif GET_TYPE == 'mAP':
                count += 1
                one_percent_recall.append(0.0)
                sequence_map = get_mAP(distmat, 
                                       q_start_idxs, 
                                       q_end_idxs, 
                                       d_start_idxs, 
                                       d_end_idxs,                  
                                       true_neighbors_matrix)
                final_map += sequence_map
            else:
                raise ValueError('GET_TYPE not supported')
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
    

    if GET_TYPE == 'mAP':
        final_map /= count
        print(f"final_map: {final_map}")
    
    return stats


def get_recall_CMC_v1(distmat_all, 
                   q_start_idxs, 
                   q_end_idxs, 
                   d_start_idxs, 
                   d_end_idxs,                  
                   true_neighbors_matrix,
                   query_embeddings_all, 
                   database_embeddings_all,
                   out_put_pair_idxs,
                   out_put_pair_idxs_list=None):

    # should normalize the embeddings first
    QB = q_end_idxs - q_start_idxs
    DB = d_end_idxs - d_start_idxs

    # 两个模态都会降， 但是LiDAR降得更少
    top_rank = 100 # 100 目前没效果
    topk_in_top_rank = 2 # 5 越大降点越多, 越小降点越少
    beta=0.95 # 0.15 越大降点越多, 越小降点越少
    use_score_type = 'q_2_to_db_1' # 'q_2_to_db_1'   'q_2_to_db_2'跟'q_2_to_db_1'效果差不多，甚至要差一些


    
    out_dim=query_embeddings_all.shape[1]
    top_rank_inuse = min(top_rank, DB)
    distmat = distmat_all[q_start_idxs:q_end_idxs, d_start_idxs:d_end_idxs] # [QB, DB]
    rank_candidate_idxs = torch.topk(distmat, dim=-1, k=top_rank_inuse, largest=False, sorted=False)[1] # [QB, top_rank_inuse]
    rank_candidate_embeddings = database_embeddings_all[d_start_idxs:d_end_idxs][rank_candidate_idxs] # [QB, top_rank_inuse, out_dim]
    rank_candidate_simmat = torch.matmul(rank_candidate_embeddings, rank_candidate_embeddings.permute(0, 2, 1)) # [QB, top_rank_inuse, top_rank_inuse]
    rank_candidate_knn_sim, rank_candidate_knn_idxs = torch.topk(rank_candidate_simmat, 
                                                                 dim=-1, 
                                                                 k=topk_in_top_rank, 
                                                                 largest=True, 
                                                                 sorted=True) # [QB, top_rank_inuse, topk_in_top_rank]
    rank_candidate_knn_sim[..., 1:] *= beta # [QB, top_rank_inuse, topk_in_top_rank]
    rank_candidate_knn_sim[..., 0] *= 1.0 # [QB, top_rank_inuse, topk_in_top_rank]
    rank_candidate_knn_embeddings = rank_candidate_embeddings.gather(dim=1, 
                                                                     index=rank_candidate_knn_idxs.flatten(1, 2).unsqueeze(-1).expand(-1, -1, out_dim)) # [QB, top_rank_inuse * topk_in_top_rank, dim]
    rank_candidate_knn_embeddings = rank_candidate_knn_embeddings.reshape(QB, top_rank_inuse, topk_in_top_rank, out_dim) # [QB, top_rank_inuse, topk_in_top_rank, dim]
    rank_candidate_embeddings_enhanced = torch.sum(rank_candidate_knn_embeddings * rank_candidate_knn_sim.unsqueeze(-1), dim=2) / torch.sum(rank_candidate_knn_sim.unsqueeze(-1), dim=2) # [QB, top_rank_inuse, dim]
    query_embeddings_inuse = query_embeddings_all[q_start_idxs:q_end_idxs] # [QB, dim]
    query_to_rank_candidate_enhanced_simmat = torch.einsum('bnd,bd->bn', rank_candidate_embeddings_enhanced, query_embeddings_inuse) # [QB, top_rank_inuse]

    query_to_db_new_knn_idxs = torch.topk(query_to_rank_candidate_enhanced_simmat, 
                                        dim=1, 
                                        k=topk_in_top_rank, 
                                        largest=True, 
                                        sorted=False)[1] # [QB, topk_in_top_rank]
    query_to_db_new_knn_embeddings = torch.gather(rank_candidate_embeddings_enhanced,
                                                dim=1,
                                                index=query_to_db_new_knn_idxs.unsqueeze(-1).expand(-1, -1, out_dim)) # [QB, topk_in_top_rank, dim]
    query_embeddings_enhanced = torch.max(query_to_db_new_knn_embeddings, dim=1, keepdim=False)[0] # [QB, dim]
    
    if use_score_type == 'q_2_to_db_1':
        query_enhanced_to_rank_candidate_enhanced_simmat = torch.einsum('bnd,bd->bn', rank_candidate_embeddings, query_embeddings_enhanced) # [QB, top_rank_inuse]
    elif use_score_type == 'q_2_to_db_2':
        query_enhanced_to_rank_candidate_enhanced_simmat = torch.einsum('bnd,bd->bn', rank_candidate_embeddings_enhanced, query_embeddings_enhanced) # [QB, top_rank_inuse]

    query_to_rank_candidate_distmat = 1 - (query_to_rank_candidate_enhanced_simmat + query_enhanced_to_rank_candidate_enhanced_simmat) / 2.0 # [QB, top_rank_inuse]

    num_neighbors = 25
    pos_mat_eval = true_neighbors_matrix[q_start_idxs:q_end_idxs, d_start_idxs:d_end_idxs] # [QB, DB]
    pos_mat_eval_inuse = torch.gather(pos_mat_eval, dim=1, index=rank_candidate_idxs) # [QB, top_rank_inuse]

    sorted_mat, sorted_indices = torch.sort(query_to_rank_candidate_distmat, dim=-1, descending=False)
    sorted_mat = sorted_mat.contiguous()
    query_to_rank_candidate_distmat = query_to_rank_candidate_distmat.contiguous()
    rank_mat = torch.searchsorted(sorted_mat, query_to_rank_candidate_distmat, right=False)
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


def get_mAP(distmat_all, 
            q_start_idxs, 
            q_end_idxs, 
            d_start_idxs, 
            d_end_idxs,                  
            true_neighbors_matrix_all):

    # should normalize the embeddings first
    QB = q_end_idxs - q_start_idxs
    DB = d_end_idxs - d_start_idxs
    sort_distmat = True
    
    distmat_inuse = distmat_all[q_start_idxs:q_end_idxs, d_start_idxs:d_end_idxs] # [QB, DB]
    true_neighbors_matrix_inuse = true_neighbors_matrix_all[q_start_idxs:q_end_idxs, d_start_idxs:d_end_idxs] # [QB, DB]

    aps = []

    if not sort_distmat:
        pass
    else:
        indices = torch.argsort(distmat_inuse, dim=-1, descending=False) # [QB, DB]
        true_neighbors_matrix_inuse = true_neighbors_matrix_inuse.gather(dim=1, index=indices) # [QB, DB]
        distmat_inuse = distmat_inuse.gather(dim=1, index=indices) # [QB, DB]
    
    for i in range(QB):
        y_score = -distmat_inuse[i]
        y_true = true_neighbors_matrix_inuse[i]
        y_score_np = y_score.cpu().numpy()
        y_true_np = y_true.cpu().numpy()
        if not np.any(y_true_np):
            continue
        aps.append(average_precision_score(y_true_np, y_score_np))

    return np.mean(aps)


def compute_jaccard_distance(distmat_inuse):

    original_dist = distmat_inuse
    QB, DB = distmat_inuse.shape
    assert QB == DB
    N = QB
    mat_type = np.float32
    original_dist = original_dist.cpu().numpy()
    k1 = 100
    global_rank = np.argpartition(original_dist, range(k1 + 2))
    nn_k1 = [k_reciprocal_neigh(global_rank, i, k1) for i in range(N)]
    nn_k1_half = [k_reciprocal_neigh(global_rank, i, int(np.around(k1 / 2))) for i in range(N)]

    V = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        k_reciprocal_index = nn_k1[i]
        k_reciprocal_expansion_index = k_reciprocal_index

        # Jaccard recall
        for candidate in k_reciprocal_index:
            candidate_k_reciprocal_index = nn_k1_half[candidate]
            if (len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index)):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        ## element-wise unique
        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        dist = torch.from_numpy(original_dist[i][k_reciprocal_expansion_index]).unsqueeze(0)
        V[i, k_reciprocal_expansion_index] = F.softmax(-dist, dim=1).view(-1).cpu().numpy()

    jaccard_dist = v2jaccard(V, N, mat_type)
    jaccard_dist = torch.from_numpy(jaccard_dist).cuda()
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


# simplest version of QE
def get_recall_CMC_v2(distmat_all, 
                   q_start_idxs, 
                   q_end_idxs, 
                   d_start_idxs, 
                   d_end_idxs,                  
                   true_neighbors_matrix,
                   query_embeddings_all, 
                   database_embeddings_all,
                   out_put_pair_idxs,
                   out_put_pair_idxs_list=None):

    # should normalize the embeddings first
    QB = q_end_idxs - q_start_idxs
    DB = d_end_idxs - d_start_idxs

    # top_rank越大降点越多
    top_rank = 3 # 100 目前没效果
    out_dim=query_embeddings_all.shape[1]
    top_rank_inuse = min(top_rank, DB)
    distmat = distmat_all[q_start_idxs:q_end_idxs, d_start_idxs:d_end_idxs] # [QB, DB]
    rank_candidate_idxs = torch.topk(distmat, dim=-1, k=top_rank_inuse, largest=False, sorted=True)[1] # [QB, top_rank_inuse]
    rank_candidate_embeddings = database_embeddings_all[d_start_idxs:d_end_idxs][rank_candidate_idxs] # [QB, top_rank_inuse, out_dim]
    query_embeddings_inuse = query_embeddings_all[q_start_idxs:q_end_idxs] # [QB, dim]
    rank_candidate_weights_base1 = torch.arange(0, top_rank_inuse, dtype=torch.float32).cuda() # [top_rank_inuse,]
    rank_candidate_weights = (top_rank_inuse - rank_candidate_weights_base1) * 1.0 / top_rank_inuse # [top_rank_inuse,]
    rank_candidate_weights = rank_candidate_weights.unsqueeze(0).unsqueeze(-1).expand(QB, -1, out_dim) # [QB, top_rank_inuse, dim]
    query_embeddings_inuse_enhanced = torch.sum(query_embeddings_inuse.unsqueeze(1) + rank_candidate_embeddings * rank_candidate_weights, dim=1) / (torch.sum(rank_candidate_weights, dim=1) + 1.0) # [QB, dim]
    database_embeddings_inuse = database_embeddings_all[d_start_idxs:d_end_idxs]
    distmat_inuse = torch.cdist(query_embeddings_inuse_enhanced, database_embeddings_inuse, p=2.0) # [QB, DB]

    num_neighbors = 25
    pos_mat_eval = true_neighbors_matrix[q_start_idxs:q_end_idxs, d_start_idxs:d_end_idxs] # [QB, DB]

    sorted_mat, sorted_indices = torch.sort(distmat_inuse, dim=-1, descending=False)
    sorted_mat = sorted_mat.contiguous()
    distmat_inuse = distmat_inuse.contiguous()
    rank_mat = torch.searchsorted(sorted_mat, distmat_inuse, right=False)
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