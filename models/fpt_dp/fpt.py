from .fpt_dependency import FastPointTransformer
import torch
import torch.nn as nn
import MinkowskiEngine as ME
import time


class fast_point_transformer(FastPointTransformer):

    def __init__(self, config, out_dim):
        super(fast_point_transformer, self).__init__(config.in_channels,
                                                              config.out_channels)
        self.final = nn.Sequential(
            nn.Linear(self.PLANES[7] + self.ENC_DIM, self.PLANES[7], bias=False),
            nn.BatchNorm1d(self.PLANES[7]),
            nn.ReLU(inplace=True)
        )
        self.out_dim = out_dim
        self.f_layer_num = config.f_layer_num
        self.f_fc = nn.Sequential(
            nn.Linear(self.PLANES[self.f_layer_num + 2], self.out_dim, bias=False),
            nn.BatchNorm1d(self.out_dim)
        )
        self.c_fc = nn.Sequential(
            nn.Linear(self.PLANES[3], self.out_dim, bias=False),
            nn.BatchNorm1d(self.out_dim)
        )
        self.radius_max_raw = config.radius_max_raw
        self.voxel_size = config.voxel_size

    
    def forward(self, x: ME.TensorField):

        out, norm_points_p1, points_p1, count_p1, pos_embs = self.voxelize_with_centroids(x)
        out = self.relu(self.bn0(self.attn0p1(out, norm_points_p1)))
        out_p1 = self.relu(self.bn1(self.attn1p1(out, norm_points_p1)))

        out, points_p2, count_p2 = self.pool(out_p1, points_p1, count_p1)
        norm_points_p2 = self.normalize_centroids(points_p2, out.C, out.tensor_stride[0])
        for module in self.block1:
            out = module(out, norm_points_p2)
        out_p2 = self.relu(self.bn2(self.attn2p2(out, norm_points_p2)))

        out, points_p4, count_p4 = self.pool(out_p2, points_p2, count_p2)
        norm_points_p4 = self.normalize_centroids(points_p4, out.C, out.tensor_stride[0])
        for module in self.block2:
            out = module(out, norm_points_p4)
        out_p4 = self.relu(self.bn3(self.attn3p4(out, norm_points_p4)))

        out, points_p8, count_p8 = self.pool(out_p4, points_p4, count_p4)
        norm_points_p8 = self.normalize_centroids(points_p8, out.C, out.tensor_stride[0])
        for module in self.block3:
            out = module(out, norm_points_p8)
        out_p8 = self.relu(self.bn4(self.attn4p8(out, norm_points_p8)))

        out, points_p16 = self.pool(out_p8, points_p8, count_p8)[:2]
        norm_points_p16 = self.normalize_centroids(points_p16, out.C, out.tensor_stride[0])
        for module in self.block4:
            out = module(out, norm_points_p16)
        
        out_F_list = []
        out_C_list = []
        point_list = []
        out_F_list.append(out.F)
        out_C_list.append(out.C)
        point_list.append(points_p16)
        if self.f_layer_num == 1:
            out_F_list.append(out.F)
            out_C_list.append(out.C)
            point_list.append(points_p16)
            return out_F_list, out_C_list, point_list

        out = self.pooltr(out) # it's essential
        out = ME.cat(out, out_p8)
        out = self.relu(self.bn5(self.attn5p8(out, norm_points_p8)))
        for module in self.block5:
            out = module(out, norm_points_p8)
        if self.f_layer_num == 2:
            out_F_list.append(out.F)
            out_C_list.append(out.C)
            point_list.append(points_p8)
            return out_F_list, out_C_list, point_list

        out = self.pooltr(out)
        out = ME.cat(out, out_p4)
        out = self.relu(self.bn6(self.attn6p4(out, norm_points_p4)))
        for module in self.block6:
            out = module(out, norm_points_p4)
        if self.f_layer_num == 3:
            out_F_list.append(out.F)
            out_C_list.append(out.C)
            point_list.append(points_p4)
            return out_F_list, out_C_list, point_list


        out = self.pooltr(out)
        out = ME.cat(out, out_p2)
        out = self.relu(self.bn7(self.attn7p2(out, norm_points_p2)))
        for module in self.block7:
            out = module(out, norm_points_p2)
        if self.f_layer_num == 4:
            out_F_list.append(out.F)
            out_C_list.append(out.C)
            point_list.append(points_p2)
            return out_F_list, out_C_list, point_list

        out = self.pooltr(out)
        out = ME.cat(out, out_p1)
        out = self.relu(self.bn8(self.attn8p1(out, norm_points_p1)))
        for module in self.block8:
            out = module(out, norm_points_p1)
        if self.f_layer_num == 5:
            out_F_list.append(out.F)
            out_C_list.append(out.C)
            point_list.append(points_p1)
            return out_F_list, out_C_list, point_list
        
        out = self.devoxelize_with_centroids(out, x, pos_embs)
        if self.f_layer_num == 6:
            out_F_list.append(out.F)
            out_C_list.append(out.C)
            point_list.append(x.C)
            return out_F_list, out_C_list, point_list


    def forward_plus(self, x: ME.TensorField):
        t1 = time.time()
        B = int(x.C[-1, 0]) + 1
        out_F_list, out_C_list, point_list = self.forward(x)
        t2 = time.time()
        
        # TODO: check if the following code is correct
        f_ebds = out_F_list[-1]
        c_ebds = out_F_list[0]
        f_points = point_list[-1] / self.radius_max_raw * self.voxel_size
        c_points = point_list[0] / self.radius_max_raw * self.voxel_size
        f_ebds = self.f_fc(f_ebds)
        c_ebds = self.c_fc(c_ebds)
        f_batch_info = out_C_list[-1][:, 0]
        c_batch_info = out_C_list[0][:, 0]
        f_batch_real_squence = torch.bincount(f_batch_info, minlength=B)
        c_batch_real_squence = torch.bincount(c_batch_info, minlength=B)
        f_max_squence_length = torch.max(f_batch_real_squence)
        c_max_squence_length = torch.max(c_batch_real_squence)
        f_batch_real_squence_cum = torch.cat((torch.zeros(1, dtype=f_batch_real_squence.dtype, device=f_batch_real_squence.device), f_batch_real_squence[:-1]), dim=0)
        c_batch_real_squence_cum = torch.cat((torch.zeros(1, dtype=c_batch_real_squence.dtype, device=c_batch_real_squence.device), c_batch_real_squence[:-1]), dim=0)
        f_batch_real_squence_cumsum = torch.repeat_interleave(torch.cumsum(f_batch_real_squence_cum, dim=0), f_max_squence_length)
        c_batch_real_squence_cumsum = torch.repeat_interleave(torch.cumsum(c_batch_real_squence_cum, dim=0), c_max_squence_length)
        f_select_index = torch.arange(0, f_max_squence_length, device=f_ebds.device).repeat(B)
        c_select_index = torch.arange(0, c_max_squence_length, device=c_ebds.device).repeat(B)
        f_select_index = torch.add(f_select_index, f_batch_real_squence_cumsum)
        c_select_index = torch.add(c_select_index, c_batch_real_squence_cumsum)
        f_max_index = f_ebds.shape[0]
        c_max_index = c_ebds.shape[0]
        f_select_index.masked_fill_(torch.ge(f_select_index, f_max_index), f_max_index - 1)
        c_select_index.masked_fill_(torch.ge(c_select_index, c_max_index), c_max_index - 1)
        f_points_output = f_points.index_select(dim=0, index=f_select_index)
        c_points_output = c_points.index_select(dim=0, index=c_select_index)
        f_ebds_output = f_ebds.index_select(dim=0, index=f_select_index)
        c_ebds_output = c_ebds.index_select(dim=0, index=c_select_index)
        f_points_output = f_points_output.reshape(B, f_max_squence_length, -1)
        c_points_output = c_points_output.reshape(B, c_max_squence_length, -1)
        f_ebds_output = f_ebds_output.reshape(B, f_max_squence_length, -1)
        c_ebds_output = c_ebds_output.reshape(B, c_max_squence_length, -1)
        f_batch_real_squence_mat = f_batch_real_squence.unsqueeze(1).repeat(1, f_max_squence_length)
        c_batch_real_squence_mat = c_batch_real_squence.unsqueeze(1).repeat(1, c_max_squence_length)
        f_mask_vets = torch.lt(torch.arange(0, f_max_squence_length, device=f_ebds_output.device).unsqueeze(0), f_batch_real_squence_mat)
        c_mask_vets = torch.lt(torch.arange(0, c_max_squence_length, device=c_ebds_output.device).unsqueeze(0), c_batch_real_squence_mat)

        f_ebds_output = f_ebds_output.permute(0, 2, 1)
        c_ebds_output = c_ebds_output.permute(0, 2, 1)
        return f_ebds_output, c_ebds_output, f_points_output, c_points_output, f_mask_vets, c_mask_vets # (B, out_dim, f_L_max), (B, out_dim, c_L_max), (B, f_L_max, 3), (B, c_L_max, 3), (B, f_L_max), (B, c_L_max)
    
        # f_mask_vets = torch.ones((B, f_max_squence_length), dtype=torch.bool, device=f_out.device)
        # c_mask_vets = torch.ones((B, c_max_squence_length), dtype=torch.bool, device=c_out.device)
        # c_ebds_list = []
        # f_ebds_list = []
        # c_points_list = []
        # f_points_list = []
        # f_ebds = f_out.F
        # c_ebds = c_out.F
        # t3 = time.time()
        # f_ebds = self.f_fc(f_ebds)
        # c_ebds = self.c_fc(c_ebds)
        # # return sp_ebds
        # for i in range(B):
        #     curr_f_ebds = f_ebds.index_select(dim=0, index=torch.where(f_out.C[:, 0]==i)[0])
        #     curr_c_ebds = c_ebds.index_select(dim=0, index=torch.where(c_out.C[:, 0]==i)[0])
        #     curr_f_points = f_points.index_select(dim=0, index=torch.where(f_out.C[:, 0]==i)[0])
        #     curr_c_points = c_points.index_select(dim=0, index=torch.where(c_out.C[:, 0]==i)[0])
        #     curr_f_points = torch.cat((curr_f_points, 
        #                               torch.zeros((f_max_squence_length - f_batch_real_squence[i], curr_f_points.shape[-1]),
        #                                           device=curr_f_points.device, dtype=curr_f_points.dtype)), 
        #                                           dim=0)
        #     curr_c_points = torch.cat((curr_c_points,
        #                                 torch.zeros((c_max_squence_length - c_batch_real_squence[i], curr_c_points.shape[-1]),
        #                                             device=curr_c_points.device, dtype=curr_c_points.dtype)), 
        #                                             dim=0)
        #     curr_f_ebds = torch.cat((curr_f_ebds, 
        #                               torch.zeros((f_max_squence_length - f_batch_real_squence[i], curr_f_ebds.shape[-1]),
        #                                           device=curr_f_ebds.device, dtype=curr_f_ebds.dtype)), 
        #                                           dim=0)
        #     curr_c_ebds = torch.cat((curr_c_ebds, 
        #                               torch.zeros((c_max_squence_length - c_batch_real_squence[i], curr_c_ebds.shape[-1]),
        #                                           device=curr_c_ebds.device, dtype=curr_c_ebds.dtype)), 
        #                                           dim=0)
        #     f_mask_vets[i, f_batch_real_squence[i]:] = False
        #     c_mask_vets[i, c_batch_real_squence[i]:] = False
        #     f_ebds_list.append(curr_f_ebds)
        #     c_ebds_list.append(curr_c_ebds)
        #     f_points_list.append(curr_f_points)
        #     c_points_list.append(curr_c_points)
        # t4 = time.time()
        # # print(  f"t2 - t1: {t2 - t1:.4f}  "  
        # #         f"t3 - t2: {t3 - t2:.4f}  "
        # #         f"t4 - t3: {t4 - t3:.4f}  ")
        # f_ebds_output = torch.stack(f_ebds_list, dim=0) 
        # c_ebds_output = torch.stack(c_ebds_list, dim=0)
        # f_points_output = torch.stack(f_points_list, dim=0)
        # c_points_output = torch.stack(c_points_list, dim=0)
        # return f_ebds_output, c_ebds_output, f_points_output, c_points_output, f_mask_vets, c_mask_vets # (B, f_L_max, out_dim), (B, c_L_max, out_dim), (B, f_L_max, 3), (B, c_L_max, 3), (B, f_L_max), (B, c_L_max)