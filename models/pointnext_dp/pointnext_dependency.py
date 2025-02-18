#   choose the PointNext-b
#   the PointNext config is referenced to the https://github.com/guochengqian/PointNeXt/blob/master/cfgs/s3dis/pointnext-b.yaml
import torch.nn as nn
from .point_next import (
    PointNextEncoder_v1, 
    PointNextDecoder_v1, 
    PointNextEncoder_v2, 
    PointNextDecoder_v2, 
    InvResMLP,
    PointNextEncoder_v3)
import torch


class PointNext(nn.Module):

    def __init__(self, config, out_dim):
        super(PointNext, self).__init__()
        self.encoder = PointNextEncoder_v1(
                cfgs=config,
                in_channels=config.in_channels, # 3
                width=config.width, # 32
                blocks=[1, 2, 3, 2, 2],
                strides=[1, 4, 4, 4, 4],
                radius=config.radius, # 0.1
                aggr_args={'feature_type': 'dp_fj', "reduction": 'max'},
                group_args={'NAME': 'ballquery',
                            'normalize_dp': True},
                expansion=4,
                sa_layers=1,
                sa_use_res=False,
                nsample=config.nsample, # 32
                conv_args={'order': 'conv-norm-act'},
                act_args={'act': 'relu'},
                norm_args={'norm': 'bn'}
        )
        encoder_channel_list = self.encoder.channel_list if hasattr(self.encoder, 'channel_list') else None
        self.decoder = PointNextDecoder_v1(
                cfgs=config,
                encoder_channel_list=encoder_channel_list,
                in_channels=config.in_channels,
                width=config.width,
                blocks=[1, 2, 3, 2, 2],
                strides=[1, 4, 4, 4, 4],
                radius=config.radius,
                aggr_args={'feature_type': 'dp_fj', "reduction": 'max'},
                group_args={'NAME': 'ballquery',
                            'normalize_dp': True},
                expansion=4,
                sa_layers=1,
                sa_use_res=False,
                nsample=config.nsample,
                conv_args={'order': 'conv-norm-act'},
                act_args={'act': 'relu'},
                norm_args={'norm': 'bn'}
        )
        self.out_dim = out_dim
        self.f_layer = config.f_layer
        self.c_layer = config.c_layer
        self.layer_dims = [config.width, 
                           config.width * 2, 
                           config.width * 4, 
                           config.width * 8, 
                           config.width * 16, 
                           config.width * 8, 
                           config.width * 4,
                           config.width * 2,
                           config.width]
        self.f_fc = nn.Sequential(
            nn.Conv1d(self.layer_dims[self.f_layer - 1], self.out_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(self.out_dim),
            )
        self.c_fc = nn.Sequential(
            nn.Conv1d(self.layer_dims[self.c_layer - 1], self.out_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(self.out_dim),
            )
    
    def forward(self, x):
        device = x.device
        f_input = torch.cat((x, x[:, :, 2:]), dim=-1) # (B, N, 4)
        f_input = f_input.permute(0, 2, 1) # (B, 4, N)
        p, f, c_feats, f_feats, c_points, f_points = self.encoder(x, f_input)
        if c_feats is not None and f_feats is not None:
            pass
        elif c_feats is None and f_feats is not None:
            c_feats, _, c_points, _ = self.decoder(p, f)
        elif f_feats is None and c_feats is not None:
            _, f_feats, _, f_points = self.decoder(p, f)
        else:
            c_feats, f_feats, c_points, f_points = self.decoder(p, f)
        c_ebds_output = self.c_fc(c_feats)
        f_ebds_output = self.f_fc(f_feats)
        c_points = c_points[:, :, :3]
        f_points = f_points[:, :, :3]
        f_mask_vets = torch.ones(f_ebds_output.shape[::2], dtype=torch.bool, device=device)
        c_mask_vets = torch.ones(c_ebds_output.shape[::2], dtype=torch.bool, device=device)
        return f_ebds_output, c_ebds_output, f_points, c_points, f_mask_vets, c_mask_vets

class PointNextv2(nn.Module):

    def __init__(self, config, out_dim):
        super(PointNextv2, self).__init__()
        self.encoder = PointNextEncoder_v2(
                cfgs=config,
                in_channels=config.in_channels, # 3
                width=config.width, # 32
                blocks=[1, 2, 3, 2, 2],
                strides=[1, 4, 4, 4, 4],
                radius=config.radius, # 0.1
                aggr_args={'feature_type': 'dp_fj', "reduction": 'max'},
                group_args={'NAME': 'ballquery',
                            'normalize_dp': True},
                expansion=4,
                sa_layers=1,
                sa_use_res=False,
                nsample=config.nsample, # 32
                conv_args={'order': 'conv-norm-act'},
                act_args={'act': 'relu'},
                norm_args={'norm': 'bn'}
        )
        encoder_channel_list = self.encoder.channel_list if hasattr(self.encoder, 'channel_list') else None
        self.decoder = PointNextDecoder_v2(
                cfgs=config,
                encoder_channel_list=encoder_channel_list,
                in_channels=config.in_channels,
                width=config.width,
                blocks=[1, 2, 3, 2, 2],
                strides=[1, 4, 4, 4, 4],
                radius=config.radius,
                aggr_args={'feature_type': 'dp_fj', "reduction": 'max'},
                group_args={'NAME': 'ballquery',
                            'normalize_dp': True},
                expansion=4,
                sa_layers=1,
                sa_use_res=False,
                nsample=config.nsample,
                conv_args={'order': 'conv-norm-act'},
                act_args={'act': 'relu'},
                norm_args={'norm': 'bn'}
        )
        self.out_dim = out_dim
        self.layer_dims = [config.width, 
                           config.width * 2, 
                           config.width * 4, 
                           config.width * 8, 
                           config.width * 16, 
                           config.width * 8, 
                           config.width * 4,
                           config.width * 2,
                           config.width]
        self.layer1_fc = nn.Sequential(
            nn.Conv1d(self.layer_dims[5 - 1], self.out_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(self.out_dim),
            )
        self.layer2_fc = nn.Sequential(
            nn.Conv1d(self.layer_dims[7 - 1], self.out_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(self.out_dim),
            )
        self.layer3_fc = nn.Sequential(
            nn.Conv1d(self.layer_dims[8 - 1], self.out_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(self.out_dim),
            )
        self.layer4_fc = nn.Sequential(
            nn.Conv1d(self.layer_dims[9 - 1], self.out_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(self.out_dim),
            )
    
    def forward(self, x):
        f_input = torch.cat((x, x[:, :, 2:]), dim=-1) # (B, N, 4)
        f_input = f_input.permute(0, 2, 1) # (B, 4, N)
        p, f = self.encoder(x, f_input)
        p, f = self.decoder(p, f)
        feats1 = f[-1]
        points1 = p[-1]
        feats2 = f[-3]
        points2 = p[-3]
        feats3 = f[-4]
        points3 = p[-4]
        feats4 = f[-5]
        points4 = p[-5]
        feats1_out = self.layer1_fc(feats1)
        feats2_out = self.layer2_fc(feats2)
        feats3_out = self.layer3_fc(feats3)
        feats4_out = self.layer4_fc(feats4)
        points1 = points1[:, :, :3]
        points2 = points2[:, :, :3]
        points3 = points3[:, :, :3]
        points4 = points4[:, :, :3]
        return feats1_out, feats2_out, feats3_out, feats4_out, points1, points2, points3, points4

class PointNextv3(nn.Module):

    def __init__(self, config, out_dim):
        super(PointNextv3, self).__init__()
        self.encoder = PointNextEncoder_v1(
                cfgs=config,
                in_channels=config.in_channels, # 3
                width=config.width, # 32
                blocks=[1, 2, 3, 2, 2],
                strides=[1, 4, 4, 4, 4],
                radius=config.radius, # 0.1
                aggr_args={'feature_type': 'dp_fj', "reduction": 'max'},
                group_args={'NAME': 'ballquery',
                            'normalize_dp': True},
                expansion=4,
                sa_layers=1,
                sa_use_res=False,
                nsample=config.nsample, # 32
                conv_args={'order': 'conv-norm-act'},
                act_args={'act': 'relu'},
                norm_args={'norm': 'bn'}
        )
        encoder_channel_list = self.encoder.channel_list if hasattr(self.encoder, 'channel_list') else None
        self.decoder = PointNextDecoder_v1(
                cfgs=config,
                encoder_channel_list=encoder_channel_list,
                in_channels=config.in_channels,
                width=config.width,
                blocks=[1, 2, 3, 2, 2],
                strides=[1, 4, 4, 4, 4],
                radius=config.radius,
                aggr_args={'feature_type': 'dp_fj', "reduction": 'max'},
                group_args={'NAME': 'ballquery',
                            'normalize_dp': True},
                expansion=4,
                sa_layers=1,
                sa_use_res=False,
                nsample=config.nsample,
                conv_args={'order': 'conv-norm-act'},
                act_args={'act': 'relu'},
                norm_args={'norm': 'bn'}
        )
        self.out_dim = out_dim
        self.f_layer = config.f_layer # 7(256 14.0 2*2) 8(1024 7.0 2*1) 9(4096 3.5 2*0)
        self.c_layer = config.c_layer
        self.layer_dims = [config.width, 
                           config.width * 2, 
                           config.width * 4, 
                           config.width * 8, 
                           config.width * 16, 
                           config.width * 8, 
                           config.width * 4,
                           config.width * 2,
                           config.width]
        self.middle_process = False
        if 'fc_type' in config.keys():
            self.middle_process = True
            if config.fc_type == 1:
                self.middle_layer1 = nn.Sequential(
                    InvResMLP(
                        in_channels=self.layer_dims[self.f_layer - 1],
                        norm_args={'norm': 'bn'},
                        act_args={'act': 'relu'},
                        aggr_args={'feature_type': 'dp_fj', "reduction": 'max'},
                        group_args={'NAME': 'ballquery',
                                    'normalize_dp': True,
                                    'radius':config.radius * 2**(len(self.layer_dims) - config.f_layer), # need to debug to see the value
                                    'nsample':32,},
                        conv_args={'order': 'conv-norm-act'},
                        expansion=4,
                        use_res=True,
                        num_posconvs=2,
                        less_act=False,
                    ),
                    InvResMLP(
                        in_channels=self.layer_dims[self.f_layer - 1],
                        norm_args={'norm': 'bn'},
                        act_args={'act': 'relu'},
                        aggr_args={'feature_type': 'dp_fj', "reduction": 'max'},
                        group_args={'NAME': 'ballquery',
                                    'normalize_dp': True,
                                    'radius':config.radius * 2**(len(self.layer_dims) - config.f_layer), # need to debug to see the value
                                    'nsample':32,},
                        conv_args={'order': 'conv-norm-act'},
                        expansion=4,
                        use_res=True,
                        num_posconvs=2,
                        less_act=False,
                    ),
                )
                self.middle_layer2 = nn.Sequential(
                    InvResMLP(
                        in_channels=self.layer_dims[self.f_layer - 1],
                        norm_args={'norm': 'bn'},
                        act_args={'act': 'relu'},
                        aggr_args={'feature_type': 'dp_fj', "reduction": 'max'},
                        group_args={'NAME': 'ballquery',
                                    'normalize_dp': True,
                                    'radius':config.radius * 2**(len(self.layer_dims) - config.f_layer), # need to debug to see the value
                                    'nsample':32,},
                        conv_args={'order': 'conv-norm-act'},
                        expansion=4,
                        use_res=True,
                        num_posconvs=2,
                        less_act=False,
                    ),
                    InvResMLP(
                        in_channels=self.layer_dims[self.f_layer - 1],
                        norm_args={'norm': 'bn'},
                        act_args={'act': 'relu'},
                        aggr_args={'feature_type': 'dp_fj', "reduction": 'max'},
                        group_args={'NAME': 'ballquery',
                                    'normalize_dp': True,
                                    'radius':config.radius * 2**(len(self.layer_dims) - config.f_layer), # need to debug to see the value
                                    'nsample':32,},
                        conv_args={'order': 'conv-norm-act'},
                        expansion=4,
                        use_res=True,
                        num_posconvs=2,
                        less_act=False,
                    ),
                )
            elif config.fc_type == 2:
                self.middle_layer1 = nn.Sequential(
                    InvResMLP(
                        in_channels=self.layer_dims[self.f_layer - 1],
                        norm_args={'norm': 'bn'},
                        act_args={'act': 'relu'},
                        aggr_args={'feature_type': 'dp_fj', "reduction": 'max'},
                        group_args={'NAME': 'ballquery',
                                    'normalize_dp': True,
                                    'radius':config.radius * 2**(len(self.layer_dims) - config.f_layer), # need to debug to see the value
                                    'nsample':32,},
                        conv_args={'order': 'conv-norm-act'},
                        expansion=4,
                        use_res=True,
                        num_posconvs=2,
                        less_act=False,
                    ),
                )
                self.middle_layer2 = nn.Sequential(
                    InvResMLP(
                        in_channels=self.layer_dims[self.f_layer - 1],
                        norm_args={'norm': 'bn'},
                        act_args={'act': 'relu'},
                        aggr_args={'feature_type': 'dp_fj', "reduction": 'max'},
                        group_args={'NAME': 'ballquery',
                                    'normalize_dp': True,
                                    'radius':config.radius * 2**(len(self.layer_dims) - config.f_layer), # need to debug to see the value
                                    'nsample':32,},
                        conv_args={'order': 'conv-norm-act'},
                        expansion=4,
                        use_res=True,
                        num_posconvs=2,
                        less_act=False,
                    ),
                )
            elif config.fc_type == 3:
                self.middle_layer1 = nn.Sequential(
                    InvResMLP(
                        in_channels=self.layer_dims[self.f_layer - 1],
                        norm_args={'norm': 'bn'},
                        act_args={'act': 'relu'},
                        aggr_args={'feature_type': 'dp_fj', "reduction": 'max'},
                        group_args={'NAME': 'ballquery',
                                    'normalize_dp': True,
                                    'radius':config.radius * 2**(len(self.layer_dims) - config.f_layer), # need to debug to see the value
                                    'nsample':32,},
                        conv_args={'order': 'conv-norm-act'},
                        expansion=4,
                        use_res=True,
                        num_posconvs=2,
                        less_act=False,
                    ),
                    InvResMLP(
                        in_channels=self.layer_dims[self.f_layer - 1],
                        norm_args={'norm': 'bn'},
                        act_args={'act': 'relu'},
                        aggr_args={'feature_type': 'dp_fj', "reduction": 'max'},
                        group_args={'NAME': 'ballquery',
                                    'normalize_dp': True,
                                    'radius':config.radius * 2**(len(self.layer_dims) - config.f_layer), # need to debug to see the value
                                    'nsample':32,},
                        conv_args={'order': 'conv-norm-act'},
                        expansion=4,
                        use_res=True,
                        num_posconvs=2,
                        less_act=False,
                    ),
                    InvResMLP(
                        in_channels=self.layer_dims[self.f_layer - 1],
                        norm_args={'norm': 'bn'},
                        act_args={'act': 'relu'},
                        aggr_args={'feature_type': 'dp_fj', "reduction": 'max'},
                        group_args={'NAME': 'ballquery',
                                    'normalize_dp': True,
                                    'radius':config.radius * 2**(len(self.layer_dims) - config.f_layer), # need to debug to see the value
                                    'nsample':32,},
                        conv_args={'order': 'conv-norm-act'},
                        expansion=4,
                        use_res=True,
                        num_posconvs=2,
                        less_act=False,
                    ),
                )
                self.middle_layer2 = nn.Sequential(
                    InvResMLP(
                        in_channels=self.layer_dims[self.f_layer - 1],
                        norm_args={'norm': 'bn'},
                        act_args={'act': 'relu'},
                        aggr_args={'feature_type': 'dp_fj', "reduction": 'max'},
                        group_args={'NAME': 'ballquery',
                                    'normalize_dp': True,
                                    'radius':config.radius * 2**(len(self.layer_dims) - config.f_layer), # need to debug to see the value
                                    'nsample':32,},
                        conv_args={'order': 'conv-norm-act'},
                        expansion=4,
                        use_res=True,
                        num_posconvs=2,
                        less_act=False,
                    ),
                    InvResMLP(
                        in_channels=self.layer_dims[self.f_layer - 1],
                        norm_args={'norm': 'bn'},
                        act_args={'act': 'relu'},
                        aggr_args={'feature_type': 'dp_fj', "reduction": 'max'},
                        group_args={'NAME': 'ballquery',
                                    'normalize_dp': True,
                                    'radius':config.radius * 2**(len(self.layer_dims) - config.f_layer), # need to debug to see the value
                                    'nsample':32,},
                        conv_args={'order': 'conv-norm-act'},
                        expansion=4,
                        use_res=True,
                        num_posconvs=2,
                        less_act=False,
                    ),
                    InvResMLP(
                        in_channels=self.layer_dims[self.f_layer - 1],
                        norm_args={'norm': 'bn'},
                        act_args={'act': 'relu'},
                        aggr_args={'feature_type': 'dp_fj', "reduction": 'max'},
                        group_args={'NAME': 'ballquery',
                                    'normalize_dp': True,
                                    'radius':config.radius * 2**(len(self.layer_dims) - config.f_layer), # need to debug to see the value
                                    'nsample':32,},
                        conv_args={'order': 'conv-norm-act'},
                        expansion=4,
                        use_res=True,
                        num_posconvs=2,
                        less_act=False,
                    ),
                )
        self.f_fc1 = nn.Sequential(
            nn.Conv1d(self.layer_dims[self.f_layer - 1], self.out_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(self.out_dim),
            )
        self.f_fc2 = nn.Sequential(
            nn.Conv1d(self.layer_dims[self.f_layer - 1], self.out_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(self.out_dim),
            )
        self.c_fc = nn.Sequential(
            nn.Conv1d(self.layer_dims[self.c_layer - 1], self.out_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(self.out_dim),
            )
        if 'pointnextv3_op' in config.keys():
            self.pointnextv3_op = config.pointnextv3_op
        else:
            self.pointnextv3_op = 1
    
    def forward(self, x):
        device = x.device
        f_input = torch.cat((x, x[:, :, 2:]), dim=-1) # (B, N, 4)
        f_input = f_input.permute(0, 2, 1) # (B, 4, N)
        p, f, c_feats, f_feats, c_points, f_points = self.encoder(x, f_input)
        if self.pointnextv3_op == 2:
            points_1 = p[-1]
            points_2 = p[-3]
            points_3 = p[-4]
            points_4 = p[-5]
        if c_feats is not None and f_feats is not None:
            pass
        elif c_feats is None and f_feats is not None:
            c_feats, _, c_points, _ = self.decoder(p, f)
        elif f_feats is None and c_feats is not None:
            _, f_feats, _, f_points = self.decoder(p, f)
        else:
            c_feats, f_feats, c_points, f_points = self.decoder(p, f)
        c_ebds_output = self.c_fc(c_feats)
        if self.middle_process:
            _, f_feats_1 = self.middle_layer1([f_points, f_feats])
            _, f_feats_2 = self.middle_layer2([f_points, f_feats])
            f1_ebds_output = self.f_fc1(f_feats_1)
            f2_ebds_output = self.f_fc2(f_feats_2)
        else:
            f1_ebds_output = self.f_fc1(f_feats)
            f2_ebds_output = self.f_fc2(f_feats)
        c_points = c_points[:, :, :3]
        f_points = f_points[:, :, :3]
        f_mask_vets = torch.ones(f1_ebds_output.shape[::2], dtype=torch.bool, device=device)
        c_mask_vets = torch.ones(c_ebds_output.shape[::2], dtype=torch.bool, device=device)
        if self.pointnextv3_op == 2:
            return f1_ebds_output, f2_ebds_output, c_ebds_output, points_1, points_2, points_3, points_4
        else:
            return f1_ebds_output, f2_ebds_output, c_ebds_output, f_points, c_points, f_mask_vets, c_mask_vets

class PointNextv4(nn.Module):

    def __init__(self, config, out_dim):
        super(PointNextv4, self).__init__()
        self.encoder = PointNextEncoder_v2(
                cfgs=config,
                in_channels=4,
                width=32,
                blocks=[1, 2, 3, 2, 2],
                strides=[1, 4, 4, 4, 4],
                radius=3.5, 
                aggr_args={'feature_type': 'dp_fj', "reduction": 'max'},
                group_args={'NAME': 'ballquery',
                            'normalize_dp': True},
                expansion=4,
                sa_layers=1,
                sa_use_res=False,
                nsample=32,
                conv_args={'order': 'conv-norm-act'},
                act_args={'act': 'relu'},
                norm_args={'norm': 'bn'}
        )
        encoder_channel_list = self.encoder.channel_list if hasattr(self.encoder, 'channel_list') else None
        self.decoder = PointNextDecoder_v2(
                cfgs=config,
                encoder_channel_list=encoder_channel_list,
                in_channels=4,
                width=32,
                blocks=[1, 2, 3, 2, 2],
                strides=[1, 4, 4, 4, 4],
                radius=3.5,
                aggr_args={'feature_type': 'dp_fj', "reduction": 'max'},
                group_args={'NAME': 'ballquery',
                            'normalize_dp': True},
                expansion=4,
                sa_layers=1,
                sa_use_res=False,
                nsample=32,
                conv_args={'order': 'conv-norm-act'},
                act_args={'act': 'relu'},
                norm_args={'norm': 'bn'}
        )
        self.out_dim = out_dim
        self.layer_dims = [32, #（4096）
                           32 * 2, #（1024）
                           32 * 4, #（256）
                           32 * 8, #（64）
                           32 * 16, #（16）
                           32 * 8, #（64）
                           32 * 4, #（256）
                           32 * 2, #（1024）
                           32] #（4096)
        self.encoder_v2 = PointNextEncoder_v3(
                 channels_list=[32, 64, 128, 256, 512],
                 blocks=[4, 7, 4, 4],
                 strides=[4, 4, 4, 4],
                 block='InvResMLP',
                 nsample=32,
                 radius=3.5,
                 aggr_args={'feature_type': 'dp_fj', "reduction": 'max'},
                 group_args={'NAME': 'ballquery'},
                 sa_layers=1,
                 sa_use_res=False,
                 conv_args={'order': 'conv-norm-act'},
                 act_args={'act': 'relu'},
                 norm_args={'norm': 'bn'}
                )
        self.layer1_fc = nn.Sequential(
            nn.Conv1d(self.layer_dims[5 - 1], self.out_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(self.out_dim),
            )
        self.layer2_fc = nn.Sequential(
            nn.Conv1d(self.layer_dims[7 - 1], self.out_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(self.out_dim),
            )
        self.layer3_fc = nn.Sequential(
            nn.Conv1d(self.layer_dims[8 - 1], self.out_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(self.out_dim),
            )
        self.layer4_fc = nn.Sequential(
            nn.Conv1d(self.layer_dims[9 - 1], self.out_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(self.out_dim),
            )
        self.layer5_fc = nn.Sequential(
            nn.Conv1d(self.layer_dims[5 - 1], self.out_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(self.out_dim),
            )
        
        self.multi_layer_aggregator = False
        if 'multi_layer_aggregator' in config.keys() and config.multi_layer_aggregator:
            self.multi_layer_aggregator = True
            self.layer2_2_fc = nn.Sequential(
                nn.Conv1d(self.layer_dims[7 - 1], self.out_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm1d(self.out_dim),
            )
            self.layer3_2_fc = nn.Sequential(
                nn.Conv1d(self.layer_dims[8 - 1], self.out_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm1d(self.out_dim),
                )
            self.layer4_2_fc = nn.Sequential(
                nn.Conv1d(self.layer_dims[9 - 1], self.out_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm1d(self.out_dim),
                )
    
    def forward(self, x):
        f_input = torch.cat((x, x[:, :, 2:]), dim=-1) # (B, N, 4)
        f_input = f_input.permute(0, 2, 1) # (B, 4, N)
        p, f = self.encoder(x, f_input)
        p, f = self.decoder(p, f)
        feats1 = f[-1]
        points1 = p[-1]
        feats2 = f[-3]
        points2 = p[-3]
        feats3 = f[-4]
        points3 = p[-4]
        feats4 = f[-5]
        points4 = p[-5]
        p.pop(0)
        f.pop(0)
        points5, feats5 = self.encoder_v2(p, f)
        feats1_out = self.layer1_fc(feats1)
        feats2_out = self.layer2_fc(feats2)
        feats3_out = self.layer3_fc(feats3)
        feats4_out = self.layer4_fc(feats4)
        feats5_out = self.layer5_fc(feats5)
        if self.multi_layer_aggregator:
            feats2_2_out = self.layer2_2_fc(feats2)
            feats3_2_out = self.layer3_2_fc(feats3)
            feats4_2_out = self.layer4_2_fc(feats4)
        points1 = points1[:, :, :3]
        points2 = points2[:, :, :3]
        points3 = points3[:, :, :3]
        points4 = points4[:, :, :3]
        points5 = points5[:, :, :3]
        if self.multi_layer_aggregator:
            return feats1_out, feats2_out, feats2_2_out, feats3_out, feats3_2_out, feats4_out, feats4_2_out, feats5_out, points1, points2, points3, points4, points5
        else:
            return feats1_out, feats2_out, feats3_out, feats4_out, feats5_out, points1, points2, points3, points4, points5