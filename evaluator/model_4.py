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
PAIR_IDX_AND_EMBEDDING_OUTPUT = False
print('PAIR_IDX_AND_EMBEDDING_OUTPUT:', PAIR_IDX_AND_EMBEDDING_OUTPUT)


def evaluate_4(model, device, cfgs, dataset_path, val_eval, out_put_pair_idxs, wandb_id):
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
            if 'render_imgs' in data.keys():
                data_input['render_imgs'] = data['render_imgs'].to(device)
            elif 'range_imgs' in data.keys():
                data_input['render_imgs'] = data['range_imgs'].to(device)
            else:
                raise ValueError('no render_imgs or range_imgs in the data')
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
    query_embeddings = query_embeddings[dataset.test_query_idx_list, ...]
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
    test_coords_tensor = dataset.UTM_coord_tensor[dataset.test_query_idx_list, ...].to(device)
    db_coords_tensor = dataset.UTM_coord_tensor.to(device)
    test_to_db_dist = torch.cdist(test_coords_tensor.unsqueeze(0), db_coords_tensor.unsqueeze(0), p=2.0).squeeze(0)
    true_neighbors_matrix = torch.lt(test_to_db_dist, dataset.true_neighbor_dist)
    if torch.isnan(distmat).any() or torch.isinf(distmat).any():
        print("the problem is the dis_mat")
    else:
        print("the sim_mat is good")
    if out_put_pair_idxs or PAIR_IDX_AND_EMBEDDING_OUTPUT:
        all_out_put_pair_idxs_list = []
        out_put_pair_idxs_list = []
    for i in tqdm.tqdm(range(len(dataset.test_traversal_cumsum)), disable=silent):
        if out_put_pair_idxs or PAIR_IDX_AND_EMBEDDING_OUTPUT:
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
            if out_put_pair_idxs or PAIR_IDX_AND_EMBEDDING_OUTPUT:
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
        if out_put_pair_idxs or PAIR_IDX_AND_EMBEDDING_OUTPUT:
            all_out_put_pair_idxs_list.append(torch.cat(out_put_pair_idxs_list, dim=-1))
    
    if out_put_pair_idxs or PAIR_IDX_AND_EMBEDDING_OUTPUT:
        data_save_root = '/DATA5/pengjianyi/vis_vpr'
        data_save_dir = os.path.join(data_save_root, wandb_id)
        os.makedirs(data_save_dir, exist_ok=True)
        all_out_put_pair_idxs = torch.cat(all_out_put_pair_idxs_list, dim=0) # [qbs, (traversal_num - 1) * num_neighbors]
        assert all_out_put_pair_idxs.shape[0] == QB
        all_out_put_pair_idxs_path = os.path.join(data_save_dir, 'all_out_put_pair_idxs.npy')
        all_out_put_pair_idxs_npy = all_out_put_pair_idxs.cpu().numpy()
        np.save(all_out_put_pair_idxs_path, all_out_put_pair_idxs_npy)

        if PAIR_IDX_AND_EMBEDDING_OUTPUT:
            query_embeddings_path = os.path.join(data_save_dir, 'query_embeddings.npy')
            query_embeddings_npy = query_embeddings.cpu().numpy()
            np.save(query_embeddings_path, query_embeddings_npy)
            database_embeddings_path = os.path.join(data_save_dir, 'database_embeddings.npy')
            database_embeddings_npy = database_embeddings.cpu().numpy()
            np.save(database_embeddings_path, database_embeddings_npy)
            test_idx_path = os.path.join(data_save_dir, 'test_idx.pkl')
            if os.path.exists(test_idx_path):
                print('test_idx.pkl exists')
            else:
                with open(test_idx_path, 'wb') as f:
                    pickle.dump(dataset.test_query_idx_list, f)


    ave_recall = recall / count
    average_dist = np.mean(dist)
    ave_one_percent_recall = np.mean(one_percent_recall)
    stats = {'ave_one_percent_recall': float(ave_one_percent_recall), 'ave_recall': ave_recall.tolist(),
             'average_sim': float(average_dist)}
    return stats


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
    if out_put_pair_idxs or PAIR_IDX_AND_EMBEDDING_OUTPUT:
        out_put_pair_idxs_list.append(sorted_indices[:, :num_neighbors] + d_start_idxs) # [qbs, num_neighbors]
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