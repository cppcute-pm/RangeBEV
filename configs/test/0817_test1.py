# img feature resolution: [256, 4096]

PcAttnRe_cfgs=dict(
    high_score_ratio_list=[0.33, 0.33],
    low_score_ratio_list=[0.66, 0.66],
    sa_cfgs_list=[[dict(neighbours=16,
                       drop_path_rate=0.,
                       groups=32,
                       attn_drop_rate=0.,
                       qkv_bias=True,
                       pe_multiplier=False,
                       pe_bias=True,),
                    dict(neighbours=4,
                       drop_path_rate=0.,
                       groups=32,
                       attn_drop_rate=0.,
                       qkv_bias=True,
                       pe_multiplier=False,
                       pe_bias=True,)],],
    ca_cfgs_list=[dict(drop_path_rate=0.,
                       groups=32,
                       attn_drop_rate=0.,
                       qkv_bias=True,
                       pe_multiplier=False,
                       pe_bias=True,)],
    same_res_num_neighbor_list=[16, 16],
    diff_res_num_neighbor_list=[256, 16],
    depth_in_one_layer=1,
    sequence_length=64,
    use_pos_emb=False,
    first_SA_block_cfgs=dict(num_blocks=3,
                             n_head=8),
    attn_type='mean',
    attn_use_type='mean',
    feat_attn_type='only_re_feat',
    )

out_dim=256