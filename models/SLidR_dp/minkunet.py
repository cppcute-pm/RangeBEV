from .res16unet import Res16UNet34C
from MinkowskiEngine import MinkowskiReLU
from MinkowskiEngine import SparseTensor
import MinkowskiEngine.MinkowskiOps as me
import torch
import torch.nn as nn


class MinkUNet(Res16UNet34C):

    def __init__(self, cfgs, out_dim):
        config = {}
        config["normalize_features"] = True
        config["bn_momentum"] = 0.05
        config["kernel_size"] = 3
        super(MinkUNet, self).__init__(in_channels=1, 
                                       out_channels=cfgs.model_n_out, # 64
                                       config=config,)
        self.c_layer = cfgs.c_layer
        self.f_layer = cfgs.f_layer
        self.voxel_size = cfgs.voxel_size
        self.out_dim = out_dim
        self.ALL_PLANES = [self.INIT_DIM] + list(self.PLANES) + [cfgs.model_n_out]
        self.f_fc = nn.Sequential(
            nn.Linear(self.ALL_PLANES[self.f_layer - 1], self.out_dim, bias=False),
            nn.BatchNorm1d(self.out_dim)
        )
        self.c_fc = nn.Sequential(
            nn.Linear(self.ALL_PLANES[self.c_layer - 1], self.out_dim, bias=False),
            nn.BatchNorm1d(self.out_dim)
        )
    
    def forward(self, x):
        c_feats = None
        f_feats = None
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)
        if self.c_layer == 1:
            c_feats = out_p1.F
            c_points = out_p1.C
        if self.f_layer == 1:
            f_feats = out_p1.F
            f_points = out_p1.C
        if c_feats is not None and f_feats is not None:
            return c_feats, f_feats, c_points, f_points

        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)

        if self.c_layer == 2:
            c_feats = out_b1p2.F
            c_points = out_b1p2.C
        if self.f_layer == 2:
            f_feats = out_b1p2.F
            f_points = out_b1p2.C
        if c_feats is not None and f_feats is not None:
            return c_feats, f_feats, c_points, f_points

        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)

        if self.c_layer == 3:
            c_feats = out_b2p4.F
            c_points = out_b2p4.C
        if self.f_layer == 3:
            f_feats = out_b2p4.F
            f_points = out_b2p4.C
        if c_feats is not None and f_feats is not None:
            return c_feats, f_feats, c_points, f_points

        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)

        if self.c_layer == 4:
            c_feats = out_b3p8.F
            c_points = out_b3p8.C
        if self.f_layer == 4:
            f_feats = out_b3p8.F
            f_points = out_b3p8.C
        if c_feats is not None and f_feats is not None:
            return c_feats, f_feats, c_points, f_points

        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        encoder_out = self.block4(out)

        if self.c_layer == 5:
            c_feats = encoder_out.F
            c_points = encoder_out.C
        if self.f_layer == 5:
            f_feats = encoder_out.F
            f_points = encoder_out.C
        if c_feats is not None and f_feats is not None:
            return c_feats, f_feats, c_points, f_points

        out = self.convtr4p16s2(encoder_out)
        out = self.bntr4(out)
        out = self.relu(out)

        out = me.cat(out, out_b3p8)
        out = self.block5(out)

        if self.c_layer == 6:
            c_feats = out.F
            c_points = out.C
        if self.f_layer == 6:
            f_feats = out.F
            f_points = out.C
        if c_feats is not None and f_feats is not None:
            return c_feats, f_feats, c_points, f_points

        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)

        out = me.cat(out, out_b2p4)
        out = self.block6(out)

        if self.c_layer == 7:
            c_feats = out.F
            c_points = out.C
        if self.f_layer == 7:
            f_feats = out.F
            f_points = out.C
        if c_feats is not None and f_feats is not None:
            return c_feats, f_feats, c_points, f_points

        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.relu(out)

        out = me.cat(out, out_b1p2)
        out = self.block7(out)

        if self.c_layer == 8:
            c_feats = out.F
            c_points = out.C
        if self.f_layer == 8:
            f_feats = out.F
            f_points = out.C
        if c_feats is not None and f_feats is not None:
            return c_feats, f_feats, c_points, f_points

        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.relu(out)

        out = me.cat(out, out_p1)
        out = self.block8(out)

        if self.c_layer == 9:
            c_feats = out.F
            c_points = out.C
        if self.f_layer == 9:
            f_feats = out.F
            f_points = out.C
        if c_feats is not None and f_feats is not None:
            return c_feats, f_feats, c_points, f_points

        out = self.final(out)

        if self.c_layer == 10:
            c_feats = out.F
            c_points = out.C
        if self.f_layer == 10:
            f_feats = out.F
            f_points = out.C
        if c_feats is not None and f_feats is not None:
            return c_feats, f_feats, c_points, f_points
        
    def forward_plus(self, x):
        B = int(x.C[-1, 0]) + 1
        c_feats, f_feats, c_points, f_points = self.forward(x)
        c_feats = self.c_fc(c_feats)
        f_feats = self.f_fc(f_feats)
        c_batch_info = c_points[:, 0]
        f_batch_info = f_points[:, 0]
        f_batch_real_squence = torch.bincount(f_batch_info, minlength=B)
        c_batch_real_squence = torch.bincount(c_batch_info, minlength=B)
        f_max_squence_length = torch.max(f_batch_real_squence)
        c_max_squence_length = torch.max(c_batch_real_squence)
        f_batch_real_squence_cum = torch.cat((torch.zeros(1, dtype=f_batch_real_squence.dtype, device=f_batch_real_squence.device), f_batch_real_squence[:-1]), dim=0)
        c_batch_real_squence_cum = torch.cat((torch.zeros(1, dtype=c_batch_real_squence.dtype, device=c_batch_real_squence.device), c_batch_real_squence[:-1]), dim=0)
        f_batch_real_squence_cumsum = torch.repeat_interleave(torch.cumsum(f_batch_real_squence_cum, dim=0), f_max_squence_length)
        c_batch_real_squence_cumsum = torch.repeat_interleave(torch.cumsum(c_batch_real_squence_cum, dim=0), c_max_squence_length)
        f_select_index = torch.arange(0, f_max_squence_length, device=f_feats.device).repeat(B) # (B * f_max_squence_length)
        c_select_index = torch.arange(0, c_max_squence_length, device=c_feats.device).repeat(B) # (B * c_max_squence_length)
        f_select_index = f_select_index.reshape(B, f_max_squence_length)
        c_select_index = c_select_index.reshape(B, c_max_squence_length)
        f_select_index = torch.fmod(f_select_index, f_batch_real_squence.unsqueeze(1))
        c_select_index = torch.fmod(c_select_index, c_batch_real_squence.unsqueeze(1))
        f_select_index = f_select_index.reshape(-1)
        c_select_index = c_select_index.reshape(-1)
        f_select_index = torch.add(f_select_index, f_batch_real_squence_cumsum)
        c_select_index = torch.add(c_select_index, c_batch_real_squence_cumsum)
        f_max_index = f_feats.shape[0]
        c_max_index = c_feats.shape[0]
        f_select_index.masked_fill_(torch.ge(f_select_index, f_max_index), f_max_index - 1)
        c_select_index.masked_fill_(torch.ge(c_select_index, c_max_index), c_max_index - 1)
        f_points_output = f_points.index_select(dim=0, index=f_select_index)
        c_points_output = c_points.index_select(dim=0, index=c_select_index)
        f_ebds_output = f_feats.index_select(dim=0, index=f_select_index)
        c_ebds_output = c_feats.index_select(dim=0, index=c_select_index)
        f_points_output = f_points_output.reshape(B, f_max_squence_length, -1)
        c_points_output = c_points_output.reshape(B, c_max_squence_length, -1)
        f_ebds_output = f_ebds_output.reshape(B, f_max_squence_length, -1)
        c_ebds_output = c_ebds_output.reshape(B, c_max_squence_length, -1)
        # f_mask_vets = torch.zeros_like(f_ebds_output[..., 0], dtype=torch.bool)
        # c_mask_vets = torch.zeros_like(c_ebds_output[..., 0], dtype=torch.bool)
        f_points_z = f_points_output[..., 2] * self.voxel_size
        f_points_y = f_points_output[..., 0] * self.voxel_size * torch.sin(f_points_output[..., 1])
        f_points_x = f_points_output[..., 0] * self.voxel_size * torch.cos(f_points_output[..., 1])
        f_points_output = torch.stack((f_points_x, f_points_y, f_points_z), dim=-1)
        c_points_z = c_points_output[..., 2] * self.voxel_size
        c_points_y = c_points_output[..., 0] * self.voxel_size * torch.sin(c_points_output[..., 1])
        c_points_x = c_points_output[..., 0] * self.voxel_size * torch.cos(c_points_output[..., 1])
        c_points_output = torch.stack((c_points_x, c_points_y, c_points_z), dim=-1)
        f_batch_real_squence_mat = f_batch_real_squence.unsqueeze(1).repeat(1, f_max_squence_length)
        c_batch_real_squence_mat = c_batch_real_squence.unsqueeze(1).repeat(1, c_max_squence_length)
        f_mask_vets = torch.lt(torch.arange(0, f_max_squence_length, device=f_ebds_output.device).unsqueeze(0), f_batch_real_squence_mat)
        c_mask_vets = torch.lt(torch.arange(0, c_max_squence_length, device=c_ebds_output.device).unsqueeze(0), c_batch_real_squence_mat)
        f_ebds_output = f_ebds_output.permute(0, 2, 1)
        c_ebds_output = c_ebds_output.permute(0, 2, 1)
        return f_ebds_output, c_ebds_output, f_points_output, c_points_output, f_mask_vets, c_mask_vets
        # assert torch.max(f_batch_real_squence) == torch.min(f_batch_real_squence)
        # assert torch.max(c_batch_real_squence) == torch.min(c_batch_real_squence)