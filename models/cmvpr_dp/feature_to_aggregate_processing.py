import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
import copy
from .residual_attention import ResidualSelfAttentionBlock
from copy import deepcopy
from .scatter_gather import keops_knn
from .generate_correspondence import my_unique_v2

class ClusterTransformer(nn.Module):

    def __init__(self, cfgs, out_norm):
        super(ClusterTransformer, self).__init__()
        self.attn_type = cfgs.attn_type
        if self.attn_type == 'attn_all':
            ResidualAttentionBlock = ResidualSelfAttentionBlock(d_model=cfgs.d_model, n_head=cfgs.n_head)
            self.layers = nn.ModuleList([
                deepcopy(ResidualAttentionBlock) for _ in range(cfgs.num_layers)
            ])
        elif self.attn_type == 'attn_all_and_semantic':
            pass
        elif self.attn_type == 'attn_local':
            pass
        elif self.attn_type == 'attn_local_and_semantic':
            pass
        else:
            raise ValueError("The attn_type should be one of ['attn_all', 'attn_all_and_semantic', 'attn_local', 'attn_local_and_semantic']")
        self.num_layers = cfgs.num_layers
        self.norm = out_norm(normalized_shape=cfgs.d_model) if out_norm is not None else None
    
    def forward(self, x, mask):
        '''
            x: b, d, L
            mask: b, L (True for padding)
        '''
        if self.attn_type == 'attn_all':
            x = x.permute(2, 0, 1) # L, b, d
            L, b, d = x.shape
            attn_mask = torch.logical_or(mask.unsqueeze(2), mask.unsqueeze(1))
            output = x
            for layer in self.layers:
                output_temp = layer(output, attn_mask=attn_mask)
                output = torch.zeros_like(output_temp)
                output[~mask.T.unsqueeze(2).expand(-1, -1, d)] = output_temp[~mask.T.unsqueeze(2).expand(-1, -1, d)]
            if self.norm is not None:
                # L, b, d -> b, L, d
                output = self.norm(output).permute(1, 2, 0)
                return output
            else:
                return output.permute(1, 2, 0)
        elif self.attn_type == 'attn_all_and_semantic':
            pass
        elif self.attn_type == 'attn_local':
            pass
        elif self.attn_type == 'attn_local_and_semantic':
            pass
        else:
            raise ValueError("The attn_type should be one of ['attn_all', 'attn_all_and_semantic', 'attn_local', 'attn_local_and_semantic']")

class SemanticTransformer(nn.Module):

    def __init__(self, cfgs, out_norm):
        super(SemanticTransformer, self).__init__()
        ResidualAttentionBlock = ResidualSelfAttentionBlock(d_model=cfgs.d_model, n_head=cfgs.n_head)
        self.layers = nn.ModuleList([
            deepcopy(ResidualAttentionBlock) for _ in range(cfgs.num_layers)
        ])
        self.num_layers = cfgs.num_layers
        self.norm = out_norm(normalized_shape=cfgs.d_model) if out_norm is not None else None
    
    def forward(self, x, mask):
        '''
            x: b, d, L
            mask: b, L (True for padding)
        '''
        x = x.permute(2, 0, 1) # L, b, d
        L, b, d = x.shape
        attn_mask = torch.logical_or(mask.unsqueeze(2), mask.unsqueeze(1))
        output = x
        for layer in self.layers:
            output_temp = layer(output, attn_mask=attn_mask)
            output = torch.zeros_like(output_temp)
            output[~mask.T.unsqueeze(2).expand(-1, -1, d)] = output_temp[~mask.T.unsqueeze(2).expand(-1, -1, d)]
        if self.norm is not None:
            # L, b, d -> b, d, L
            output = self.norm(output).permute(1, 2, 0)
            return output
        else:
            return output.permute(1, 2, 0)

def aggregate_clusterly_and_semantically(data_dict, 
                                         device, 
                                         img_feats, 
                                         pc_feats, 
                                         cfgs,
                                         cluster_transformer,
                                         semantic_transformer):

    B, out_dim, curr_img_H, curr_img_W = img_feats.shape
    curr_pc_N = pc_feats.shape[-1]

    img_semantic_label = data_dict['img_semantic_label'] # (B, 1, 224, 224)
    img_ccl_cluster_label = data_dict['img_ccl_cluster_label'] # (B, 1, 224, 224)

    # TODO: this may lead some problems if there exist some new semantic-cluster corresponding relationships
    if img_semantic_label.shape[2:] != (curr_img_H, curr_img_W):
        img_semantic_label_float = img_semantic_label.type(torch.float32)
        img_ccl_cluster_label_float = img_ccl_cluster_label.type(torch.float32)
        img_semantic_label_float = F.interpolate(img_semantic_label_float, (curr_img_H, curr_img_W), mode='nearest')
        img_ccl_cluster_label_float = F.interpolate(img_ccl_cluster_label_float, (curr_img_H, curr_img_W), mode='nearest')
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

    pc_cluster_masks = torch_scatter.scatter_sum(torch.ones_like(pc_dbscan_cluster_label),
                                                 pc_dbscan_cluster_label,
                                                 dim=-1,
                                                 dim_size=max_cpoints_dbscan_cluster) # produce (B, 1, max_cpoints_dbscan_cluster)
    pc_cluster_masks = pc_cluster_masks > 0 # produce (B, 1, max_cpoints_dbscan_cluster)
    pc_cluster_masks = pc_cluster_masks[:, :, 1:] # produce (B, 1, max_cpoints_dbscan_cluster - 1)
    pc_cluster_masks = pc_cluster_masks.squeeze(1) # produce (B, max_cpoints_dbscan_cluster - 1)
    img_cluster_masks = torch_scatter.scatter_sum(torch.ones_like(img_ccl_cluster_label.flatten(2)),
                                                    img_ccl_cluster_label.flatten(2),
                                                    dim=-1,
                                                    dim_size=max_cimg_ccl_cluster) # produce (B, 1, max_cimg_ccl_cluster)
    img_cluster_masks = img_cluster_masks > 0 # produce (B, 1, max_cimg_ccl_cluster)
    img_cluster_masks = img_cluster_masks[:, :, 1:] # produce (B, 1, max_cimg_ccl_cluster - 1)
    img_cluster_masks = img_cluster_masks.squeeze(1) # produce (B, max_cimg_ccl_cluster - 1)

    pc_cluster_embeddings = torch_scatter.scatter_mean(pc_feats,
                                                    pc_dbscan_cluster_label.expand(-1, out_dim, -1),
                                                    dim=-1,
                                                    dim_size=max_cpoints_dbscan_cluster) # produce (B, out_dim, max_cpoints_dbscan_cluster)
    pc_cluster_embeddings = pc_cluster_embeddings[:, :, 1:] # produce (B, out_dim, max_cpoints_dbscan_cluster - 1)
    img_cluster_embeddings = torch_scatter.scatter_mean(img_feats.flatten(2),
                                                    img_ccl_cluster_label.expand(-1, out_dim, -1, -1).flatten(2),
                                                    dim=-1,
                                                    dim_size=max_cimg_ccl_cluster) # produce (B, out_dim, max_cimg_ccl_cluster)
    img_cluster_embeddings = img_cluster_embeddings[:, :, 1:] # produce (B, out_dim, max_cimg_ccl_cluster - 1)

    img_semantic_embeddings = torch.zeros((B, out_dim, max_cimg_semantic_inuse), device=device)
    pc_semantic_embeddings = torch.zeros((B, out_dim, max_cpoints_semantic_inuse), device=device)
    img_semantic_mask = torch.zeros((B, max_cimg_semantic_inuse), device=device)
    pc_semantic_mask = torch.zeros((B, max_cpoints_semantic_inuse), device=device)
    img_cluster_masks = torch.ones_like(img_cluster_masks, device=device)
    pc_cluster_masks = torch.ones_like(pc_cluster_masks, device=device)

    # try to make the cluster embeddings belong to the same semantic category to input into the attention machanism
    # incase there is only one cluster in the semantic category, use a transformer with different mask setting


    # img_cluster_masks = ~img_cluster_masks
    # pc_cluster_masks = ~pc_cluster_masks
    # if cfgs.aggregate_cluster_attn_type == 'attn_all':
    #     # TODO: may need to add the positional encoding, refer the EGllnet
    #     img_cluster_embeddings = cluster_transformer(img_cluster_embeddings, img_cluster_masks)
    #     pc_cluster_embeddings = cluster_transformer(pc_cluster_embeddings, pc_cluster_masks)
    # elif cfgs.aggregate_cluster_attn_type == 'attn_all_and_semantic':
    #     pass
    # elif cfgs.aggregate_cluster_attn_type == 'attn_local':
    #     pass
    # elif cfgs.aggregate_cluster_attn_type == 'attn_local_and_semantic':
    #     pass
    # else:
    #     raise ValueError("The aggregate_cluster_attn_type should be one of ['attn_all', 'attn_all_and_semantic', 'attn_local', 'attn_local_and_semantic']")
    # img_cluster_masks = ~img_cluster_masks
    # pc_cluster_masks = ~pc_cluster_masks

    # img_semantic_label_one_hot = F.one_hot(img_semantic_label.squeeze(1), num_classes=max_cimg_semantic_inuse) # (B, curr_img_H, curr_img_W, max_cimg_semantic_inuse)
    # img_cluster_to_semantic_one_hot = torch_scatter.scatter_sum(img_semantic_label_one_hot.flatten(1, 2),
    #                                                 img_ccl_cluster_label.squeeze(1).flatten(1).unsqueeze(-1).expand(-1, -1, max_cimg_semantic_inuse),
    #                                                 dim=1,
    #                                                 dim_size=max_cimg_ccl_cluster) # produce (B, max_cimg_ccl_cluster, max_cimg_semantic_inuse)
    # img_cluster_to_semantic = torch.argmax(img_cluster_to_semantic_one_hot, dim=-1) # produce (B, max_cimg_ccl_cluster)
    # img_cluster_to_semantic = img_cluster_to_semantic[:, 1:] # produce (B, max_cimg_ccl_cluster - 1)
    # img_semantic_embeddings_sum = torch_scatter.scatter_sum(img_cluster_embeddings,
    #                                                 img_cluster_to_semantic.unsqueeze(1).expand(-1, out_dim, -1),
    #                                                 dim=-1,
    #                                                 dim_size=max_cimg_semantic_inuse) # produce (B, out_dim, max_semantic_inuse)
    # img_semantic_embeddings_num = torch_scatter.scatter_sum(img_cluster_masks.type(torch.int64),
    #                                                         img_cluster_to_semantic,
    #                                                         dim=-1,
    #                                                         dim_size=max_cimg_semantic_inuse) # produce (B, max_semantic_inuse)
    # img_semantic_embeddings = img_semantic_embeddings_sum / (img_semantic_embeddings_num.unsqueeze(1) + 1e-6) # produce (B, out_dim, max_semantic_inuse)

    # pc_semantic_label_one_hot = F.one_hot(pc_semantic_label.squeeze(1), num_classes=max_cpoints_semantic_inuse) # (B, cpoints_num, max_cpoints_semantic_inuse)
    # pc_cluster_to_semantic_one_hot = torch_scatter.scatter_sum(pc_semantic_label_one_hot,
    #                                                 pc_dbscan_cluster_label.squeeze(1).unsqueeze(-1).expand(-1, -1, max_cpoints_semantic_inuse),
    #                                                 dim=1,
    #                                                 dim_size=max_cpoints_dbscan_cluster) # produce (B, max_cpoints_dbscan_cluster, max_cpoints_semantic_inuse)
    # pc_cluster_to_semantic = torch.argmax(pc_cluster_to_semantic_one_hot, dim=-1) # produce (B, max_cpoints_dbscan_cluster)
    # pc_cluster_to_semantic = pc_cluster_to_semantic[:, 1:] # produce (B, max_cpoints_dbscan_cluster - 1)
    # pc_semantic_embeddings_sum = torch_scatter.scatter_sum(pc_cluster_embeddings,
    #                                                 pc_cluster_to_semantic.unsqueeze(1).expand(-1, out_dim, -1),
    #                                                 dim=-1,
    #                                                 dim_size=max_cpoints_semantic_inuse) # produce (B, out_dim, max_semantic_inuse)
    # pc_semantic_embeddings_num = torch_scatter.scatter_sum(pc_cluster_masks.type(torch.int64),
    #                                                         pc_cluster_to_semantic,
    #                                                         dim=-1,
    #                                                         dim_size=max_cpoints_semantic_inuse) # produce (B, max_semantic_inuse)
    # pc_semantic_embeddings = pc_semantic_embeddings_sum / (pc_semantic_embeddings_num.unsqueeze(1) + 1e-6) # produce (B, out_dim, max_semantic_inuse)

    # pc_semantic_mask = pc_semantic_embeddings_num > 0 # produce (B, max_semantic_inuse)
    # pc_semantic_mask = pc_semantic_mask[:, 1:] # produce (B, max_semantic_inuse - 1)
    # img_semantic_mask = img_semantic_embeddings_num > 0 # produce (B, max_semantic_inuse)
    # img_semantic_mask = img_semantic_mask[:, 1:] # produce (B, max_semantic_inuse - 1)
    # pc_semantic_embeddings = pc_semantic_embeddings[:, :, 1:] # produce (B, out_dim, max_semantic_inuse - 1)
    # img_semantic_embeddings = img_semantic_embeddings[:, :, 1:] # produce (B, out_dim, max_semantic_inuse - 1)

    # pc_semantic_mask = ~pc_semantic_mask
    # img_semantic_mask = ~img_semantic_mask
    # img_semantic_embeddings = semantic_transformer(img_semantic_embeddings, img_semantic_mask)
    # pc_semantic_embeddings = semantic_transformer(pc_semantic_embeddings, pc_semantic_mask)
    # pc_semantic_mask = ~pc_semantic_mask
    # img_semantic_mask = ~img_semantic_mask

    # # TODO: maybe the semantic embeddings and the cluster embeddings could be attentioned another time
    # # TODO: insert the local correspondence before or after the transformer layer
    
    return (img_semantic_embeddings, 
            img_semantic_mask, 
            pc_semantic_embeddings, 
            pc_semantic_mask, 
            img_cluster_embeddings, 
            img_cluster_masks, 
            pc_cluster_embeddings, 
            pc_cluster_masks)



def aggregate_and_match(data_dict, 
                        device, 
                        img_feats, 
                        pc_feats, 
                        cfgs,
                        cluster_transformer,
                        semantic_transformer,
                        points,
                        train_or_eval):

    B, out_dim, curr_img_H, curr_img_W = img_feats.shape
    curr_pc_N = pc_feats.shape[-1]

    img_semantic_label = data_dict['img_semantic_label'] # (B, 1, 224, 224)
    img_ccl_cluster_label = data_dict['img_ccl_cluster_label'] # (B, 1, 224, 224)

    pre_img_H, pre_img_W = img_semantic_label.shape[2:]

    # TODO: this may lead some problems if there exist some new semantic-cluster corresponding relationships
    if img_semantic_label.shape[2:] != (curr_img_H, curr_img_W):
        img_semantic_label_float = img_semantic_label.type(torch.float32)
        img_ccl_cluster_label_float = img_ccl_cluster_label.type(torch.float32)
        img_semantic_label_float = F.interpolate(img_semantic_label_float, (curr_img_H, curr_img_W), mode='nearest')
        img_ccl_cluster_label_float = F.interpolate(img_ccl_cluster_label_float, (curr_img_H, curr_img_W), mode='nearest')
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
        cityscapes_label_in_semanticKitti_label_inuse = torch.arange(0, 11, dtype=torch.int64, device=device)
    
    max_cpoints_dbscan_cluster = torch.max(pc_dbscan_cluster_label) + 1
    assert max_cimg_semantic_inuse >= (torch.max(img_semantic_label) + 1)
    assert max_cpoints_semantic_inuse >= (torch.max(pc_semantic_label) + 1)


    if train_or_eval == "train":
        original_pc_2_points = data_dict["original_pc_2_many_1"][..., 0]
        pre_img_H_mesh = torch.arange(0, pre_img_H, device=device).type(torch.float32)
        pre_img_W_mesh = torch.arange(0, pre_img_W, device=device).type(torch.float32)
        pre_img_H_mesh = pre_img_H_mesh.unsqueeze(1).unsqueeze(2).expand(-1, pre_img_W, -1)
        pre_img_W_mesh = pre_img_W_mesh.unsqueeze(0).unsqueeze(2).expand(pre_img_H, -1, -1)
        pre_img_mesh = torch.cat((pre_img_H_mesh, pre_img_W_mesh), dim=-1) # Produces (pre_img_H, pre_img_W, 2)
        pre_img_mesh = pre_img_mesh.flatten(0, 1) # Produces (pre_img_H * pre_img_W, 2) tensor
        curr_img_H_mesh = torch.arange(0, curr_img_H, device=device)
        curr_img_W_mesh = torch.arange(0, curr_img_W, device=device)
        pre_img_2_curr_img_scale_H = pre_img_H * 1.0 / curr_img_H
        pre_img_2_curr_img_scale_W = pre_img_W * 1.0 / curr_img_W
        delta_H = pre_img_2_curr_img_scale_H / 2 - 0.5
        delta_W = pre_img_2_curr_img_scale_W / 2 - 0.5
        curr_img_H_mesh = curr_img_H_mesh * pre_img_2_curr_img_scale_H + delta_H
        curr_img_W_mesh = curr_img_W_mesh * pre_img_2_curr_img_scale_W + delta_W
        curr_img_H_mesh = curr_img_H_mesh.unsqueeze(1).unsqueeze(2).expand(-1, curr_img_W, -1) # Produces (curr_img_H, curr_img_W, 1) tensor
        curr_img_W_mesh = curr_img_W_mesh.unsqueeze(0).unsqueeze(2).expand(curr_img_H, -1, -1) # Produces (curr_img_H, curr_img_W, 1) tensor
        curr_img_mesh = torch.cat((curr_img_H_mesh, curr_img_W_mesh), dim=-1) # Produces (curr_img_H, curr_img_W, 2) tensor
        curr_img_mesh = curr_img_mesh.flatten(0, 1) # Produces (curr_img_H * curr_img_W, 2) tensor
        _, pre_img_2_curr_img_idx = keops_knn(device, pre_img_mesh, curr_img_mesh, 1)
        pre_img_2_curr_img_idx = pre_img_2_curr_img_idx.squeeze(-1).unsqueeze(0).unsqueeze(0).expand(B, B, -1) # Produces (B, B, pre_img_H * pre_img_W) tensor
        original_pc_2_curr_img = torch.gather(input=pre_img_2_curr_img_idx,
                                        dim=-1,
                                        index=data_dict["original_pc_2_many_1"][..., 1]) # Produces (B, B, original_points_num)
        original_pc_2_img_semantic = torch.gather(input=img_semantic_label.squeeze(1).unsqueeze(0).expand(B, -1, -1, -1).flatten(2),
                                                dim=-1,
                                                index=original_pc_2_curr_img) # Produces (B, B, original_points_num)
        original_pc_2_img_ccl_cluster = torch.gather(input=img_ccl_cluster_label.squeeze(1).unsqueeze(0).expand(B, -1, -1, -1).flatten(2),
                                                dim=-1,
                                                index=original_pc_2_curr_img) # Produces (B, B, original_points_num)
        
        original_pc_2_pc_semantic = torch.gather(input=pc_semantic_label.expand(-1, B, -1),
                                                dim=-1,
                                                index=original_pc_2_points) # Produces (B, B, original_points_num)
        original_pc_2_pc_dbscan_cluster = torch.gather(input=pc_dbscan_cluster_label.expand(-1, B, -1),
                                                dim=-1,
                                                index=original_pc_2_points) # Produces (B, B, original_points_num)

        overlap_mask = torch.logical_and(data_dict["original_pc_2_many_2"][..., 0], data_dict["original_pc_2_many_2"][..., 1]) # Produces (B, B, original_points_num)
        overlap_mask = overlap_mask.type(original_pc_2_curr_img.dtype)


        original_pc_2_img_points = torch.stack((original_pc_2_img_semantic,
                                                    original_pc_2_img_ccl_cluster, 
                                                    original_pc_2_pc_semantic,
                                                    original_pc_2_pc_dbscan_cluster, 
                                                    overlap_mask), dim=-1) # Produces (B, B, original_points_num, 5)
        original_pc_2_img_points_unique, original_pc_2_img_points_num = my_unique_v2(original_pc_2_img_points, 
                                                                                    torch.ones_like(original_pc_2_img_points[..., 0]),
                                                                                    (max_cimg_semantic_inuse, 
                                                                                    max_cimg_ccl_cluster, 
                                                                                    max_cpoints_semantic_inuse, 
                                                                                    max_cpoints_dbscan_cluster, 
                                                                                    2)) # produce (huge_num, 7), (huge_num)
        # t2 = time.time()

        original_pc_2_img_points_mask = torch.eq(original_pc_2_img_points_unique[..., -1], 1) # produce (huge_num)
        original_pc_2_img_points_unique = original_pc_2_img_points_unique[original_pc_2_img_points_mask, :-1] # produce (huge_num_1, 6)
        original_pc_2_img_points_num = original_pc_2_img_points_num[original_pc_2_img_points_mask] # produce (huge_num_1)

        original_pc_2_img_points_mask1 = torch.eq(original_pc_2_img_points_unique[:, 2], 0)
        original_pc_2_img_points_mask2 = torch.eq(original_pc_2_img_points_unique[:, 3], 0)
        original_pc_2_img_points_mask3 = torch.eq(original_pc_2_img_points_unique[:, 4], 0)
        original_pc_2_img_points_mask4 = torch.eq(original_pc_2_img_points_unique[:, 5], 0)
        original_pc_2_img_points_mask_v2 = original_pc_2_img_points_mask1 | original_pc_2_img_points_mask2 | original_pc_2_img_points_mask3 | original_pc_2_img_points_mask4
        original_pc_2_img_points_unique_2 = original_pc_2_img_points_unique[~original_pc_2_img_points_mask_v2] # produce (huge_num_2, 6)
        original_pc_2_img_points_num_2 = original_pc_2_img_points_num[~original_pc_2_img_points_mask_v2] # produce (huge_num_2)

        original_pc_2_img_points_unique_img_semantic = original_pc_2_img_points_unique_2[:, 2] # produce (huge_num_2)

        original_pc_2_img_points_unique_choose1_img_to_pc_semantic = torch.gather(cityscapes_label_in_semanticKitti_label,
                                                                                    index=original_pc_2_img_points_unique_img_semantic.unsqueeze(1).expand(-1, label_dim),
                                                                                    dim=0) # produce (huge_num_2, label_dim)
        original_pc_2_img_points_unique_choose1_img_to_pc_semantic += 1
        original_pc_2_img_points_unique_mask1 = original_pc_2_img_points_unique_choose1_img_to_pc_semantic == original_pc_2_img_points_unique_2[:, 4].unsqueeze(-1).expand(-1, label_dim) # produce (huge_num_2, label_dim)
        original_pc_2_img_points_unique_mask2 = original_pc_2_img_points_unique_mask1.any(dim=-1) # produce (huge_num_2)
        original_pc_2_img_points_unique_3 = original_pc_2_img_points_unique_2[original_pc_2_img_points_unique_mask2] # produce (huge_num_3, 6)
        original_pc_2_img_points_num_3 = original_pc_2_img_points_num_2[original_pc_2_img_points_unique_mask2] # produce (huge_num_3)

        original_pc_2_points.masked_fill_(~data_dict["original_pc_2_many_2"][..., 1], curr_pc_N - 1) # can't be reuse again
        remove_mask_num = torch.count_nonzero(~data_dict["original_pc_2_many_2"][..., 1], dim=-1) # (B, B)
        points_num_pt = torch_scatter.scatter_sum(torch.ones_like(original_pc_2_points, dtype=torch.int32),
                                                original_pc_2_points,
                                                dim=-1,
                                                dim_size=curr_pc_N) # produce (B, B, curr_pc_N)
        points_num_pt[..., -1] -= remove_mask_num # produce (B, B, cpoints_num)
        dbscan_cluster_num_pt = torch_scatter.scatter_sum(points_num_pt,
                                                        pc_dbscan_cluster_label.expand(-1, B, -1),
                                                        dim=-1,
                                                        dim_size=max_cpoints_dbscan_cluster) # produce (B, B, max_cpoints_dbscan_cluster)
        dbscan_cluster_num_pt = torch.clamp(dbscan_cluster_num_pt, min=1.0)
        dbscan_cluster_num_pt_unique = dbscan_cluster_num_pt[original_pc_2_img_points_unique_3[:, 0], original_pc_2_img_points_unique_3[:, 1], original_pc_2_img_points_unique_3[:, 5]] # produce (huge_num_3)
        dbscan_cluster_overlap_ratio_mask = torch.gt(dbscan_cluster_num_pt_unique, cfgs.min_pc_dbscan_cluster_num_pt)
        dbscan_cluster_overlap_ratio = original_pc_2_img_points_num_3 * 1.0 / dbscan_cluster_num_pt_unique # produce (huge_num_3)
        dbscan_cluster_overlap_ratio = dbscan_cluster_overlap_ratio.masked_fill(~dbscan_cluster_overlap_ratio_mask, 0.0)



        original_pc_2_curr_img.masked_fill_(~data_dict["original_pc_2_many_2"][..., 0], curr_img_H * curr_img_W - 1) # can't be reuse again
        non_qualified_mask_num = torch.count_nonzero(~data_dict["original_pc_2_many_2"][..., 0], dim=-1) # produce (B, B)
        img_num_pt = torch_scatter.scatter_sum(torch.ones_like(original_pc_2_curr_img, dtype=torch.int32),
                                                original_pc_2_curr_img,
                                                dim=-1,
                                                dim_size=curr_img_H * curr_img_W) # produce (B, B, curr_img_H * curr_img_W)
        img_num_pt[..., -1] -= non_qualified_mask_num
        ccl_cluster_num_pt = torch_scatter.scatter_sum(img_num_pt,
                                                    img_ccl_cluster_label.squeeze(1).unsqueeze(0).expand(B, -1, -1, -1).flatten(2),
                                                    dim=-1,
                                                    dim_size=max_cimg_ccl_cluster) # produce (B, B, max_cimg_ccl_cluster)
        ccl_cluster_num_pt = torch.clamp(ccl_cluster_num_pt, min=1.0)
        ccl_cluster_num_pt_unique = ccl_cluster_num_pt[original_pc_2_img_points_unique_3[:, 0], original_pc_2_img_points_unique_3[:, 1], original_pc_2_img_points_unique_3[:, 3]] # produce (huge_num_3)
        ccl_cluster_overlap_ratio_mask = torch.gt(ccl_cluster_num_pt_unique, cfgs.min_img_ccl_cluster_num_pt)
        ccl_cluster_overlap_ratio = original_pc_2_img_points_num_3 * 1.0 / ccl_cluster_num_pt_unique # produce (huge_num_3)
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
        original_pc_2_img_points_unique_choose = original_pc_2_img_points_unique_3[cluster_topk_indices] # produce (cfgs.cluster_topk, 6)



    pc_cluster_masks = torch_scatter.scatter_sum(torch.ones_like(pc_dbscan_cluster_label),
                                                 pc_dbscan_cluster_label,
                                                 dim=-1,
                                                 dim_size=max_cpoints_dbscan_cluster) # produce (B, 1, max_cpoints_dbscan_cluster)
    pc_cluster_masks = pc_cluster_masks > 0 # produce (B, 1, max_cpoints_dbscan_cluster)
    pc_cluster_masks = pc_cluster_masks[:, :, 1:] # produce (B, 1, max_cpoints_dbscan_cluster - 1)
    pc_cluster_masks = pc_cluster_masks.squeeze(1) # produce (B, max_cpoints_dbscan_cluster - 1)
    img_cluster_masks = torch_scatter.scatter_sum(torch.ones_like(img_ccl_cluster_label.flatten(2)),
                                                    img_ccl_cluster_label.flatten(2),
                                                    dim=-1,
                                                    dim_size=max_cimg_ccl_cluster) # produce (B, 1, max_cimg_ccl_cluster)
    img_cluster_masks = img_cluster_masks > 0 # produce (B, 1, max_cimg_ccl_cluster)
    img_cluster_masks = img_cluster_masks[:, :, 1:] # produce (B, 1, max_cimg_ccl_cluster - 1)
    img_cluster_masks = img_cluster_masks.squeeze(1) # produce (B, max_cimg_ccl_cluster - 1)

    pc_cluster_embeddings = torch_scatter.scatter_mean(pc_feats,
                                                    pc_dbscan_cluster_label.expand(-1, out_dim, -1),
                                                    dim=-1,
                                                    dim_size=max_cpoints_dbscan_cluster) # produce (B, out_dim, max_cpoints_dbscan_cluster)
    pc_cluster_embeddings = pc_cluster_embeddings[:, :, 1:] # produce (B, out_dim, max_cpoints_dbscan_cluster - 1)
    img_cluster_embeddings = torch_scatter.scatter_mean(img_feats.flatten(2),
                                                    img_ccl_cluster_label.expand(-1, out_dim, -1, -1).flatten(2),
                                                    dim=-1,
                                                    dim_size=max_cimg_ccl_cluster) # produce (B, out_dim, max_cimg_ccl_cluster)
    img_cluster_embeddings = img_cluster_embeddings[:, :, 1:] # produce (B, out_dim, max_cimg_ccl_cluster - 1)


    if train_or_eval == 'train' and cfgs.cluster_correspondence_type == 'before':
        pc_cluster_embeddings_temp = torch.cat((torch.zeros_like(pc_cluster_embeddings[:, :, 0:1]), pc_cluster_embeddings), dim=-1) # produce (B, out_dim, max_cpoints_dbscan_cluster)
        pc_cluster_out_embeddings = pc_cluster_embeddings_temp[original_pc_2_img_points_unique_choose[:, 0], :, original_pc_2_img_points_unique_choose[:, 5]] # produce (cfgs.cluster_topk, out_dim)
        img_cluster_embeddings_temp = torch.cat((torch.zeros_like(img_cluster_embeddings[:, :, 0:1]), img_cluster_embeddings), dim=-1) # produce (B, out_dim, max_cimg_ccl_cluster)
        img_cluster_out_embeddings = img_cluster_embeddings_temp[original_pc_2_img_points_unique_choose[:, 1], :, original_pc_2_img_points_unique_choose[:, 3]] # produce (cfgs.cluster_topk, out_dim)


    # try to make the cluster embeddings belong to the same semantic category to input into the attention machanism
    # incase there is only one cluster in the semantic category, use a transformer with different mask setting


    img_cluster_masks = ~img_cluster_masks
    pc_cluster_masks = ~pc_cluster_masks
    if cfgs.aggregate_cluster_attn_type == 'attn_all':
        # TODO: may need to add the positional encoding, refer the EGllnet
        img_cluster_embeddings = cluster_transformer(img_cluster_embeddings, img_cluster_masks)
        pc_cluster_embeddings = cluster_transformer(pc_cluster_embeddings, pc_cluster_masks)
    elif cfgs.aggregate_cluster_attn_type == 'attn_all_and_semantic':
        pass
    elif cfgs.aggregate_cluster_attn_type == 'attn_local':
        pass
    elif cfgs.aggregate_cluster_attn_type == 'attn_local_and_semantic':
        pass
    else:
        raise ValueError("The aggregate_cluster_attn_type should be one of ['attn_all', 'attn_all_and_semantic', 'attn_local', 'attn_local_and_semantic']")
    img_cluster_embeddings[img_cluster_masks.unsqueeze(1).expand(-1, out_dim, -1)] = 0.0
    pc_cluster_embeddings[pc_cluster_masks.unsqueeze(1).expand(-1, out_dim, -1)] = 0.0
    img_cluster_masks = ~img_cluster_masks
    pc_cluster_masks = ~pc_cluster_masks



    if train_or_eval == 'train' and cfgs.cluster_correspondence_type == 'after':
        pc_cluster_embeddings_temp = torch.cat((torch.zeros_like(pc_cluster_embeddings[:, :, 0:1]), pc_cluster_embeddings), dim=-1) # produce (B, out_dim, max_cpoints_dbscan_cluster)
        pc_cluster_out_embeddings = pc_cluster_embeddings_temp[original_pc_2_img_points_unique_choose[:, 0], :, original_pc_2_img_points_unique_choose[:, 5]] # produce (cfgs.cluster_topk, out_dim)
        img_cluster_embeddings_temp = torch.cat((torch.zeros_like(img_cluster_embeddings[:, :, 0:1]), img_cluster_embeddings), dim=-1) # produce (B, out_dim, max_cimg_ccl_cluster)
        img_cluster_out_embeddings = img_cluster_embeddings_temp[original_pc_2_img_points_unique_choose[:, 1], :, original_pc_2_img_points_unique_choose[:, 3]] # produce (cfgs.cluster_topk, out_dim)


    img_semantic_label_one_hot = F.one_hot(img_semantic_label.squeeze(1), num_classes=max_cimg_semantic_inuse) # (B, curr_img_H, curr_img_W, max_cimg_semantic_inuse)
    img_cluster_to_semantic_one_hot = torch_scatter.scatter_sum(img_semantic_label_one_hot.flatten(1, 2),
                                                    img_ccl_cluster_label.squeeze(1).flatten(1).unsqueeze(-1).expand(-1, -1, max_cimg_semantic_inuse),
                                                    dim=1,
                                                    dim_size=max_cimg_ccl_cluster) # produce (B, max_cimg_ccl_cluster, max_cimg_semantic_inuse)
    img_cluster_to_semantic = torch.argmax(img_cluster_to_semantic_one_hot, dim=-1) # produce (B, max_cimg_ccl_cluster)
    img_cluster_to_semantic = img_cluster_to_semantic[:, 1:] # produce (B, max_cimg_ccl_cluster - 1)
    img_semantic_embeddings_sum = torch_scatter.scatter_sum(img_cluster_embeddings,
                                                    img_cluster_to_semantic.unsqueeze(1).expand(-1, out_dim, -1),
                                                    dim=-1,
                                                    dim_size=max_cimg_semantic_inuse) # produce (B, out_dim, max_cimg_semantic_inuse)
    img_semantic_embeddings_num = torch_scatter.scatter_sum(img_cluster_masks.type(torch.int64),
                                                            img_cluster_to_semantic,
                                                            dim=-1,
                                                            dim_size=max_cimg_semantic_inuse) # produce (B, max_cimg_semantic_inuse)
    img_semantic_embeddings = img_semantic_embeddings_sum / (img_semantic_embeddings_num.unsqueeze(1) + 1e-6) # produce (B, out_dim, max_cimg_semantic_inuse)

    pc_semantic_label_one_hot = F.one_hot(pc_semantic_label.squeeze(1), num_classes=max_cpoints_semantic_inuse) # (B, cpoints_num, max_cpoints_semantic_inuse)
    pc_cluster_to_semantic_one_hot = torch_scatter.scatter_sum(pc_semantic_label_one_hot,
                                                    pc_dbscan_cluster_label.squeeze(1).unsqueeze(-1).expand(-1, -1, max_cpoints_semantic_inuse),
                                                    dim=1,
                                                    dim_size=max_cpoints_dbscan_cluster) # produce (B, max_cpoints_dbscan_cluster, max_cpoints_semantic_inuse)
    pc_cluster_to_semantic = torch.argmax(pc_cluster_to_semantic_one_hot, dim=-1) # produce (B, max_cpoints_dbscan_cluster)
    pc_cluster_to_semantic = pc_cluster_to_semantic[:, 1:] # produce (B, max_cpoints_dbscan_cluster - 1)
    pc_semantic_embeddings_sum = torch_scatter.scatter_sum(pc_cluster_embeddings,
                                                    pc_cluster_to_semantic.unsqueeze(1).expand(-1, out_dim, -1),
                                                    dim=-1,
                                                    dim_size=max_cpoints_semantic_inuse) # produce (B, out_dim, max_cpoints_semantic_inuse)
    pc_semantic_embeddings_num = torch_scatter.scatter_sum(pc_cluster_masks.type(torch.int64),
                                                            pc_cluster_to_semantic,
                                                            dim=-1,
                                                            dim_size=max_cpoints_semantic_inuse) # produce (B, max_cpoints_semantic_inuse)
    pc_semantic_embeddings = pc_semantic_embeddings_sum / (pc_semantic_embeddings_num.unsqueeze(1) + 1e-6) # produce (B, out_dim, max_cpoints_semantic_inuse)

    pc_semantic_mask = pc_semantic_embeddings_num > 0 # produce (B, max_cpoints_semantic_inuse)
    pc_semantic_mask = pc_semantic_mask[:, 1:] # produce (B, max_cpoints_semantic_inuse - 1)
    img_semantic_mask = img_semantic_embeddings_num > 0 # produce (B, max_cimg_semantic_inuse)
    img_semantic_mask = img_semantic_mask[:, 1:] # produce (B, max_cimg_semantic_inuse - 1)
    pc_semantic_embeddings = pc_semantic_embeddings[:, :, 1:] # produce (B, out_dim, max_cpoints_semantic_inuse - 1)
    img_semantic_embeddings = img_semantic_embeddings[:, :, 1:] # produce (B, out_dim, max_cimg_semantic_inuse - 1)


    if train_or_eval == 'train':
        if cfgs.semantic_fuse_type == 'cluster_topk_corresponded':
            original_pc_2_img_points_unique_ccl_cluster = torch.unique(original_pc_2_img_points_unique_choose[:, 1:4], dim=0) # produce (ccl_cluster_num, 3)
            original_pc_2_img_points_unique_dbscan_cluster = torch.unique(torch.cat([original_pc_2_img_points_unique_choose[:, 0:1], 
                                                                                    original_pc_2_img_points_unique_choose[:, 4:6]], dim=-1), 
                                                                                    dim=0) # produce (dbscan_cluster_num, 3)
        elif cfgs.semantic_fuse_type == 'semantic_corresponded':
            original_pc_2_img_points_unique_ccl_cluster = torch.unique(original_pc_2_img_points_unique_3[:, 1:4], dim=0) # produce (ccl_cluster_num, 3)
            original_pc_2_img_points_unique_dbscan_cluster = torch.unique(torch.cat([original_pc_2_img_points_unique_3[:, 0:1], 
                                                                                    original_pc_2_img_points_unique_3[:, 4:6]], dim=-1), 
                                                                                    dim=0) # produce (dbscan_cluster_num, 3)
        elif cfgs.semantic_fuse_type == 'semantic_inuse':
            original_pc_2_img_points_unique_ccl_cluster = torch.unique(original_pc_2_img_points_unique_2[:, 1:4], dim=0) # produce (ccl_cluster_num, 3)
            original_pc_2_img_points_unique_dbscan_cluster = torch.unique(torch.cat([original_pc_2_img_points_unique_2[:, 0:1], 
                                                                                    original_pc_2_img_points_unique_2[:, 4:6]], dim=-1), 
                                                                                    dim=0) # produce (dbscan_cluster_num, 3)
        else:
            raise ValueError("The semantic_fuse_type is not supported")
        original_pc_2_img_points_unique_img_semantic = torch.unique(original_pc_2_img_points_unique_ccl_cluster[:, :2], return_inverse=False, dim=0) # produce (img_semantic_num, 2)
        original_pc_2_img_points_unique_pc_semantic = torch.unique(original_pc_2_img_points_unique_dbscan_cluster[:, :2], return_inverse=False, dim=0) # produce (pc_semantic_num, 2)
        original_pc_2_img_points_unique_img_in_pc_semantic = torch.gather(input=cityscapes_label_in_semanticKitti_label[:, 0],
                                                                            dim=0,
                                                                            index=original_pc_2_img_points_unique_img_semantic[:, 1]) # produce (img_semantic_num,)
        original_pc_2_img_points_unique_img_in_pc_semantic += 1
        img_semantic_flag_matrix = torch.zeros((B, max_cpoints_semantic_inuse), dtype=torch.bool, device=device) # produce (B, max_cpoints_semantic_inuse)
        pc_semantic_flag_matrix = torch.zeros((B, max_cpoints_semantic_inuse), dtype=torch.bool, device=device) # produce (B, max_cpoints_semantic_inuse)
        img_semantic_flag_matrix[original_pc_2_img_points_unique_img_semantic[:, 0], original_pc_2_img_points_unique_img_in_pc_semantic] = True
        pc_semantic_flag_matrix[original_pc_2_img_points_unique_pc_semantic[:, 0], original_pc_2_img_points_unique_pc_semantic[:, 1]] = True
        semantic_flag_matrix = img_semantic_flag_matrix & pc_semantic_flag_matrix
        semantic_to_choose = torch.nonzero(semantic_flag_matrix, as_tuple=False) # produce (num_semantic, 2)

        if cfgs.semantic_correspondence_type == 'before':
            pc_semantic_embeddings_temp = torch.cat((torch.zeros_like(pc_semantic_embeddings[:, :, 0:1]), pc_semantic_embeddings), dim=-1) # produce (B, out_dim, max_cpoints_semantic_inuse)
            img_semantic_embeddings_temp = torch.cat((torch.zeros_like(img_semantic_embeddings[:, :, 0:1]), img_semantic_embeddings), dim=-1) # produce (B, out_dim, max_cimg_semantic_inuse)
            img_semantic_embeddings_in_pc_order = torch.gather(img_semantic_embeddings_temp,
                                                            dim=-1,
                                                            index=cityscapes_label_in_semanticKitti_label_inuse.unsqueeze(0).unsqueeze(0).expand(B, out_dim, -1)) # produce (B, out_dim, max_cpoints_semantic)
            img_semantic_out_embeddings = img_semantic_embeddings_in_pc_order[semantic_to_choose[:, 0], :, semantic_to_choose[:, 1]] # produce (num_semantic, out_dim)
            pc_semantic_out_embeddings = pc_semantic_embeddings_temp[semantic_to_choose[:, 0], :, semantic_to_choose[:, 1]] # produce (num_semantic, out_dim)


    pc_semantic_mask = ~pc_semantic_mask
    img_semantic_mask = ~img_semantic_mask
    img_semantic_embeddings = semantic_transformer(img_semantic_embeddings, img_semantic_mask)
    pc_semantic_embeddings = semantic_transformer(pc_semantic_embeddings, pc_semantic_mask)
    img_semantic_embeddings[img_semantic_mask.unsqueeze(1).expand(-1, out_dim, -1)] = 0.0
    pc_semantic_embeddings[pc_semantic_mask.unsqueeze(1).expand(-1, out_dim, -1)] = 0.0
    pc_semantic_mask = ~pc_semantic_mask
    img_semantic_mask = ~img_semantic_mask

    if train_or_eval == 'train' and cfgs.semantic_correspondence_type == 'after':
        pc_semantic_embeddings_temp = torch.cat((torch.zeros_like(pc_semantic_embeddings[:, :, 0:1]), pc_semantic_embeddings), dim=-1) # produce (B, out_dim, max_cpoints_semantic_inuse)
        img_semantic_embeddings_temp = torch.cat((torch.zeros_like(img_semantic_embeddings[:, :, 0:1]), img_semantic_embeddings), dim=-1) # produce (B, out_dim, max_cimg_semantic_inuse)
        img_semantic_embeddings_in_pc_order = torch.gather(img_semantic_embeddings_temp,
                                                           dim=-1,
                                                           index=cityscapes_label_in_semanticKitti_label_inuse.unsqueeze(0).unsqueeze(0).expand(B, out_dim, -1)) # produce (B, out_dim, max_cpoints_semantic_inuse)
        img_semantic_out_embeddings = img_semantic_embeddings_in_pc_order[semantic_to_choose[:, 0], :, semantic_to_choose[:, 1]] # produce (num_semantic, out_dim)
        pc_semantic_out_embeddings = pc_semantic_embeddings_temp[semantic_to_choose[:, 0], :, semantic_to_choose[:, 1]] # produce (num_semantic, out_dim)
    
    if train_or_eval == 'train':
        img_all_semantic_embeddings_sum = torch.sum(img_semantic_embeddings, dim=0, keepdim=False) # produce (out_dim, max_cimg_semantic_inuse)
        pc_all_semantic_embeddings_sum = torch.sum(pc_semantic_embeddings, dim=0, keepdim=False) # produce (out_dim, max_cpoints_semantic_inuse)
        img_all_semantic_num = torch.sum(img_semantic_mask.type(torch.int64), dim=0, keepdim=False) # produce (max_cimg_semantic_inuse)
        pc_all_semantic_num = torch.sum(pc_semantic_mask.type(torch.int64), dim=0, keepdim=False) # produce (max_cpoints_semantic_inuse)
        img_all_semantic_embeddings = img_all_semantic_embeddings_sum / (img_all_semantic_num.unsqueeze(0) + 1e-6) # produce (out_dim, max_cimg_semantic_inuse)
        pc_all_semantic_embeddings = pc_all_semantic_embeddings_sum / (pc_all_semantic_num.unsqueeze(0) + 1e-6) # produce (out_dim, max_cpoints_semantic_inuse)

        original_pc_2_img_points_unique_all_img_semantic = torch.unique(original_pc_2_img_points_unique_img_semantic[:, 1], return_inverse=False, dim=0) # produce (num_all_img_semantic,)
        original_pc_2_img_points_unique_all_pc_semantic = torch.unique(original_pc_2_img_points_unique_pc_semantic[:, 1], return_inverse=False, dim=0) # produce (num_all_pc_semantic,)
        original_pc_2_img_points_unique_all_img_in_pc_semantic = torch.gather(input=cityscapes_label_in_semanticKitti_label[:, 0],
                                                                            dim=0,
                                                                            index=original_pc_2_img_points_unique_all_img_semantic) # produce (num_all_img_semantic,)
        original_pc_2_img_points_unique_all_img_in_pc_semantic += 1
        all_img_semantic_flag_matrix = torch.zeros((max_cpoints_semantic_inuse), dtype=torch.bool, device=device) # produce (max_cpoints_semantic_inuse,)
        all_pc_semantic_flag_matrix = torch.zeros((max_cpoints_semantic_inuse), dtype=torch.bool, device=device) # produce (max_cpoints_semantic_inuse,)
        all_img_semantic_flag_matrix[original_pc_2_img_points_unique_all_img_in_pc_semantic] = True
        all_pc_semantic_flag_matrix[original_pc_2_img_points_unique_all_pc_semantic] = True
        all_semantic_flag_matrix = all_img_semantic_flag_matrix & all_pc_semantic_flag_matrix
        all_semantic_to_choose = torch.nonzero(all_semantic_flag_matrix, as_tuple=False) # produce (all_num_semantic,)
        all_semantic_to_choose = all_semantic_to_choose.squeeze(-1)

        pc_all_semantic_embeddings_temp = torch.cat((torch.zeros_like(pc_all_semantic_embeddings[:, 0:1]), pc_all_semantic_embeddings), dim=-1) # produce (out_dim, max_cpoints_semantic_inuse)
        img_all_semantic_embeddings_temp = torch.cat((torch.zeros_like(img_all_semantic_embeddings[:, 0:1]), img_all_semantic_embeddings), dim=-1) # produce (out_dim, max_cimg_semantic_inuse)
        img_all_semantic_embeddings_in_pc_order = torch.gather(img_all_semantic_embeddings_temp,
                                                            dim=-1,
                                                            index=cityscapes_label_in_semanticKitti_label_inuse.unsqueeze(0).expand(out_dim, -1)) # produce (out_dim, max_cpoints_semantic_inuse)
        img_all_semantic_out_embeddings = img_all_semantic_embeddings_in_pc_order[:, all_semantic_to_choose] # produce (out_dim, all_num_semantic)
        pc_all_semantic_out_embeddings = pc_all_semantic_embeddings_temp[:, all_semantic_to_choose] # produce (out_dim, all_num_semantic)
        img_all_semantic_out_embeddings = img_all_semantic_out_embeddings.permute(1, 0) # produce (all_num_semantic, out_dim)
        pc_all_semantic_out_embeddings = pc_all_semantic_out_embeddings.permute(1, 0) # produce (all_num_semantic, out_dim)
    
    if not (train_or_eval == 'train'):
        pc_cluster_out_embeddings = None
        img_cluster_out_embeddings = None
        pc_semantic_out_embeddings = None
        img_semantic_out_embeddings = None
        pc_all_semantic_out_embeddings = None
        img_all_semantic_out_embeddings = None

    # if not (train_or_eval == 'train'):
    #     pc_cluster_out_embeddings = None
    #     img_cluster_out_embeddings = None
    #     pc_semantic_out_embeddings = None
    #     img_semantic_out_embeddings = None
    #     pc_all_semantic_out_embeddings = None
    #     img_all_semantic_out_embeddings = None
    # else:
    #     pc_semantic_out_embeddings = torch.zeros((11, out_dim), device=device, dtype=torch.float32)
    #     img_semantic_out_embeddings = torch.zeros((11, out_dim), device=device, dtype=torch.float32)
    #     pc_all_semantic_out_embeddings = torch.zeros((11, out_dim), device=device, dtype=torch.float32)
    #     img_all_semantic_out_embeddings = torch.zeros((11, out_dim), device=device, dtype=torch.float32)
    # img_semantic_embeddings=None
    # img_semantic_mask=None
    # pc_semantic_embeddings=None 
    # pc_semantic_mask=None
    
    return (img_semantic_embeddings, 
            img_semantic_mask, 
            pc_semantic_embeddings, 
            pc_semantic_mask, 
            img_cluster_embeddings, 
            img_cluster_masks, 
            pc_cluster_embeddings, 
            pc_cluster_masks,
            pc_cluster_out_embeddings,
            img_cluster_out_embeddings,
            pc_semantic_out_embeddings,
            img_semantic_out_embeddings,
            pc_all_semantic_out_embeddings,
            img_all_semantic_out_embeddings)