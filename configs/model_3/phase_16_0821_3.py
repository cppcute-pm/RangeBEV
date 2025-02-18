_base_ = ['../base/datasets/boreas_v2.py',
            '../base/optimizers/adam.py',
            '../base/schedulers/CosineAnnealingWarmRestarts.py',
            '../base/runtime.py',]

# img feature resolution: [16, 256, 1024, 4096]
# pc feature resolution: [[7, 7], [28, 28], [56, 56], [112, 112]]

model_cfgs=dict(
    modal_type=3,
    out_dim=256,
    cmvpr_cfgs=dict(
        image_encoder_type='ResUNetmmsegv2',
        image_out_layer=None,
        pc_out_layer=None,
        image_encoder_out_layer=7,
        image_encoder_cfgs=dict(
            decoder_head_1=dict(
                block_type='Bottleneck_exp_rate_2',
                stage_blocks=[3, 6, 4, 3],
                channels_list=[2048, 1024, 512, 256, 64],
                upsample_cfg=dict(type='InterpConv'),
                last_layer_num_blocks=2,
                ), 
        ),
        pc_encoder_type='PointNextv2',
        pc_encoder_out_layer=7,
        pc_encoder_cfgs=dict(
            in_channels=4,
            width=32,
            radius=3.5,
            nsample=32,
            dropout=0.5,
            pointnextv3_op=1,
            ),
        image_aggregator_type='GeM',
        image_aggregator_cfgs=dict(
            p=3,
            eps=1e-6,
            ),
        pc_aggregator_type='GeM',
        pc_aggregator_cfgs=dict(
            p=3,
            eps=1e-6,
            ),
        phase16_aggregator_type='GeM',
        phase16_aggregator_cfgs=dict(
            p=3,
            eps=1e-6,
            ),
        phase16_aggregator1_type='GeM',
        phase16_aggregator1_cfgs=dict(
            p=3,
            eps=1e-6,
            ),
        phase16_aggregator2_type='GeM',
        phase16_aggregator2_cfgs=dict(
            p=3,
            eps=1e-6,
            ),
        phase16_aggregator3_type='GeM',
        phase16_aggregator3_cfgs=dict(
            p=3,
            eps=1e-6,
            ),
        phase16_aggregator4_type='GeM',
        phase16_aggregator4_cfgs=dict(
            p=3,
            eps=1e-6,
            ),
        phase=16,
        multi_layer=4,

        img_attn_re_cfgs=dict(
            high_score_ratio_list=[0.4, 0.4, 0.4, 0.4],
            low_score_ratio_list=[0.73, 0.73, 0.73, 0.73],
            sa_cfgs_list=[[dict(with_cp=False,
                        window_size_int=6,
                        mlp_ratio=4,
                        norm_cfg=dict(type='LN', requires_grad=True),
                        drop_rate=0.,
                        drop_path_rate=0.,
                        act_cfg=dict(type='GELU'),
                        window_size=[6, 6],
                        num_heads=8,
                        qk_scale=None,
                        qkv_bias=True,
                        attn_drop_rate=0.,
                        ), 
                   dict(with_cp=False,
                        window_size_int=3,
                        mlp_ratio=4,
                        norm_cfg=dict(type='LN', requires_grad=True),
                        drop_rate=0.,
                        drop_path_rate=0.,
                        act_cfg=dict(type='GELU'),
                        window_size=[3, 3],
                        num_heads=8,
                        qk_scale=None,
                        qkv_bias=True,
                        attn_drop_rate=0.,),], 
                  [dict(with_cp=False,
                        window_size_int=6,
                        mlp_ratio=4,
                        norm_cfg=dict(type='LN', requires_grad=True),
                        drop_rate=0.,
                        drop_path_rate=0.,
                        act_cfg=dict(type='GELU'),
                        window_size=[6, 6],
                        num_heads=8,
                        qk_scale=None,
                        qkv_bias=True,
                        attn_drop_rate=0.,), 
                   dict(with_cp=False,
                        window_size_int=3,
                        mlp_ratio=4,
                        norm_cfg=dict(type='LN', requires_grad=True),
                        drop_rate=0.,
                        drop_path_rate=0.,
                        act_cfg=dict(type='GELU'),
                        window_size=[3, 3],
                        num_heads=8,
                        qk_scale=None,
                        qkv_bias=True,
                        attn_drop_rate=0.,),], 
                  [dict(with_cp=False,
                        window_size_int=6,
                        mlp_ratio=4,
                        norm_cfg=dict(type='LN', requires_grad=True),
                        drop_rate=0.,
                        drop_path_rate=0.,
                        act_cfg=dict(type='GELU'),
                        window_size=[6, 6],
                        num_heads=8,
                        qk_scale=None,
                        qkv_bias=True,
                        attn_drop_rate=0.,), 
                   dict(with_cp=False,
                        window_size_int=3,
                        mlp_ratio=4,
                        norm_cfg=dict(type='LN', requires_grad=True),
                        drop_rate=0.,
                        drop_path_rate=0.,
                        act_cfg=dict(type='GELU'),
                        window_size=[3, 3],
                        num_heads=8,
                        qk_scale=None,
                        qkv_bias=True,
                        attn_drop_rate=0.,),]],
            ca_cfgs_list=[dict(mlp_ratio=4,
                       norm_cfg=dict(type='LN', requires_grad=True),
                       drop_rate=0.,
                       drop_path_rate=0.,
                       act_cfg=dict(type='GELU'),
                        window_size=[12, 12],
                        pre_window_size=[3, 3],
                        num_heads=8,
                        qk_scale=None,
                        qkv_bias=True,
                        attn_drop_rate=0.,), 
                  dict(mlp_ratio=4,
                       norm_cfg=dict(type='LN', requires_grad=True),
                       drop_rate=0.,
                       drop_path_rate=0.,
                       act_cfg=dict(type='GELU'),
                        window_size=[6, 6],
                        pre_window_size=[3, 3],
                        num_heads=8,
                        qk_scale=None,
                        qkv_bias=True,
                        attn_drop_rate=0.,), 
                  dict(mlp_ratio=4,
                       norm_cfg=dict(type='LN', requires_grad=True),
                       drop_rate=0.,
                       drop_path_rate=0.,
                       act_cfg=dict(type='GELU'),
                        window_size=[6, 6],
                        pre_window_size=[3, 3],
                        num_heads=8,
                        qk_scale=None,
                        qkv_bias=True,
                        attn_drop_rate=0.,),],
            num_neighbor_list=[9, 9, 9, 9],
            depth_in_one_layer=1,
            sequence_length=49,
            first_SA_block_cfgs=dict(num_blocks=3,
                                    n_head=8),
            attn_type='max',
            attn_use_type='mean',
            feat_attn_type='only_re_feat',
            only_re_feat_type=1,
            to_aggr_feat_select_type=2,
            feat_fuse_type='mean',
        ),

        pc_attn_re_cfgs=dict(
            high_score_ratio_list=[0.4, 0.4, 0.4, 0.4],
            low_score_ratio_list=[0.73, 0.73, 0.73, 0.73],
            sa_cfgs_list=[[dict(neighbours=8,
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
                       pe_bias=True,)],
                  [dict(neighbours=8,
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
                       pe_bias=True,)],
                  [dict(neighbours=8,
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
                       pe_bias=True,), 
                  dict(drop_path_rate=0.,
                       groups=32,
                       attn_drop_rate=0.,
                       qkv_bias=True,
                       pe_multiplier=False,
                       pe_bias=True,),
                  dict(drop_path_rate=0.,
                       groups=32,
                       attn_drop_rate=0.,
                       qkv_bias=True,
                       pe_multiplier=False,
                       pe_bias=True,),],
        same_res_num_neighbor_list=[8, 8, 8, 8],
        diff_res_num_neighbor_list=[64, 16, 4, 4],
        depth_in_one_layer=1,
        sequence_length=16,
        use_pos_emb=True,
        first_SA_block_cfgs=dict(num_blocks=3,
                                n_head=8),
        attn_type='max',
        attn_use_type='mean',
        feat_attn_type='only_re_feat',
        to_aggr_feat_select_type=2,
        feat_fuse_type='mean',
        ),
    ),
)

dataset_cfgs=dict(
    train_pc_aug_mode=8,
    eval_pc_aug_mode=3,
    train_image_aug_mode=16,
    eval_image_aug_mode=17,
    raw_dir_name='Boreas_minuse',
    pc_dir_name="Boreas_minuse_40960_to_4096_cliped_fov",
    rendered_dir_name=None,
    image_dir_name='Boreas_224x224_image',
    mask_dir_name=None,
    rgb_depth_label_dir_name=None,
    tool_name='my_tool',
    train_coords_filename='train_UTM_coords_v3_50m.npy',
    val_query_filename='class_split_test_queries.pickle',
    eval_query_filename='class_split_test_queries.pickle',
    minuse_lidar_filename='train_minuse_lidar_idx.pickle',
    positive_distance=10.0,
    non_negative_distance=50.0,
    positive_distance_list=[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
    lidar2image_filename='lidar2image.pickle',
    class_num_path='class_split_num.pickle',
    image_size=[224, 224],
    render_size=[224, 224],
    mask_size=[224, 224],
    pc_preprocess=dict(
        mode=0,
    ),
    use_cloud=True,
    use_render=False,
    use_image=True,
    use_mask=False,
    use_rgb_depth_label=False,
    num_points=4096,
    crop_location=dict(
        x_min=0,
        x_max=2448,
        y_min=683,
        y_max=1366,
    ),
    img_neighbor_num=1,
    dist_caculation_type='all_coords_L2_mean',
    use_overlap_ratio=True,
    overlap_ratio_type='pose_dist_sim',
    pose_dist_threshold=37.5,
    use_original_pc_correspondence=False,
    correspondence_point_project_threshold=110.0,
)

sampler_cfgs=dict(
    sampler_type='none',
)

batch_sampler_cfgs=dict(
    batch_sampler_type='ensure_k_levels_pos_uniform_multi_v8',
    num_k=3,
    batch_size=16,
    eval_batch_size=16,
    interval_num=50,
    useout_times=1,
)

collect_fn_cfgs=dict(
    collect_fn_type='4',
)

dataloader_cfgs=dict(
    num_workers=4,
    dataset_cfgs=dataset_cfgs,
    sampler_cfgs=sampler_cfgs,
    batch_sampler_cfgs=batch_sampler_cfgs,
    collect_fn_cfgs=collect_fn_cfgs,
)

# loss_cfgs=dict(
#     loss_type='GL_v2_triplet_circle',
#     global_loss_cfgs=dict(
#         base_margin=0.6,
#         normalize_embeddings=False,
#         positive_overlap_ratio=0.2,
#         negative_overlap_ratio=0.05,
#         delta_overlap_ratio=0.01,
#         tuple_formtype='relative_delta',
#         choose_nonzero=True,
#     ),
#     local_loss_cfgs=dict(
#         m=0.2,
#         gamma=40,
#     ),
#     global_loss_weight=0.5,
#     local_loss_weight=0.5,
#     local_positive_mask_margin=0.5,
#     local_negative_mask_margin=0.2,
# )

loss_cfgs=dict(
    loss_type='v2_triplet_cm',
    base_margin=0.6,
    normalize_embeddings=False,
    positive_overlap_ratio=0.2,
    negative_overlap_ratio=0.05,
    delta_overlap_ratio=0.01,
    tuple_formtype='relative_delta',
    choose_nonzero=True,
)

optimizer_cfgs=dict(
    optimizer_type='adam',
    lr=1e-5,
    weight_decay=1e-4,
)

scheduler_cfgs=dict(
    scheduler_type='CosineAnnealingWarmRestarts',
    warmup_epoch=5,
    min_lr=1e-7,
    T_mult=2,
)

dgcnn_path="cx7jkf53/epoch_73.pth"
resnet_path="thk0qhd5/epoch_73.pth"
# dgcnn_path=None
cmvpr_path=None
pretrained_cfgs=dict(
    cmvprnet=dict(
        image_path=resnet_path,
        image_cfgs=dict(
            backbone_part=0,
        ),
        image_aggregator_cfgs=dict(
            aggregate_part=0,
        ),
        pc_path=dgcnn_path,
        pc_cfgs=dict(
            backbone_part=2,
        ),
        pc_aggregator_cfgs=dict(
            aggregate_part=0,
        ),
        cmvpr_path=cmvpr_path,
    )
)

freeze_cfgs=dict(
    freeze_cmvpr=dict(
        freeze_img_encoder=dict(
            backbone=0,
        ),
        freeze_img_aggregator=dict(
            aggregate=0,
        ),
        freeze_pc_encoder=dict(
            backbone=2,
        ),
        freeze_pc_aggregator=dict(
            aggregate=0,
        ),
    )
)
use_mp=True
need_eval=True














